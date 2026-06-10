#!/bin/bash
# alignment 项 λ_align sweep（llama2 主设定 P2 上）：验证把方向一致性折入 s_j 能否进一步降 ASR。
# 从 saart_p2_immunize.yaml（已含 use_assoc_reg=true + after_bos）派生，改 saart_use_assoc_align=true
# + assoc_align_lambda=λ + 输出目录。λ=0 作对照（_assoc_score 在 λ=0 时退回 magnitude，≈ P2 baseline）。
# 两阶段：bf16 训练(≥24GB 卡) → fp32 评测(≥34GB 卡)。tag=after_saart_p2_align_<lam>。幂等。
# 遵循质量审计协议：结果必须 ASR 与 degen% 同看。
set -uo pipefail
cd /home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine
export HF_HOME=/mnt/data/zengqixiu/hf_cache WANDB_DISABLED=true TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/zengqixiu/anaconda3/envs/backdoor/bin/python
BASE=/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama2_7b_chat
SRC=$BASE/outputs/training/configs/saart_p2_immunize.yaml
SWEEP=$BASE/outputs/purified/saart_p2_align; LOGD=$BASE/outputs/logs/align_sweep
CFG=configs/experiment.llama2_7b_chat.yaml
mkdir -p "$SWEEP" "$LOGD"
LAMS=(${ALIGN_LAMS:-0 0.5 1.0 2.0})
pick_gpu(){ local need=$1 g free; while true; do for g in 5 6 7 3 1 0 8 9 2 4; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null|tr -d ' ')
  [ "${free:-0}" -ge "$need" ] && { echo "$g"; return; }; done; sleep 20; done; }

# 阶段A 训练
for lam in "${LAMS[@]}"; do
  name="lam${lam//./p}"; out="$SWEEP/$name"; yaml="$BASE/outputs/training/configs/saart_p2_align_$name.yaml"
  [ -f "$out/adapter_model.safetensors" ] && { echo "[skip-train] $name"; continue; }
  $PY - "$SRC" "$yaml" "$lam" "$out" <<'EOF'
import sys, yaml
src, dst, lam, out = sys.argv[1:5]
c = yaml.safe_load(open(src))
c['saart_use_assoc_align'] = True      # 开关（_SAART_FIELD_MAP: use_assoc_align -> saart_use_assoc_align，带前缀）
c['assoc_align_lambda'] = float(lam)   # 权重（无 saart_ 前缀）
c['output_dir'] = out
yaml.safe_dump(c, open(dst, 'w'), sort_keys=False)
print(f'[gen] {dst} lam={lam}')
EOF
  g=$(pick_gpu 24000); port=$(((RANDOM%40000)+20000)); echo "[align-train] $name -> GPU $g"
  ( CUDA_VISIBLE_DEVICES=$g $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
      saart_train.py "$yaml" > "$LOGD/${name}_train.log" 2>&1 ) & sleep 25
done
wait; echo "[align-sweep] 训练阶段完成 $(date +%T)"

# 阶段B 评测（fp32 大卡）
for lam in "${LAMS[@]}"; do
  name="lam${lam//./p}"; out="$SWEEP/$name"
  [ -f "$out/adapter_model.safetensors" ] || { echo "[skip-eval] $name 无 adapter"; continue; }
  grep -q "after_saart_p2_align_$name\"" "$BASE/outputs/eval/results.jsonl" 2>/dev/null && { echo "[skip-eval] $name"; continue; }
  g=$(pick_gpu 34000); echo "[align-eval] $name -> GPU $g"
  ( CUDA_VISIBLE_DEVICES=$g $PY step5_evaluate.py --config "$CFG" --adapter "$out" \
      --eval_type both --tag "after_saart_p2_align_$name" > "$LOGD/${name}_eval.log" 2>&1 ) & sleep 25
done
wait; echo "########## ALIGN SWEEP DONE $(date) ##########"
