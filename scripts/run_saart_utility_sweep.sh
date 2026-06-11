#!/bin/bash
# W1a L_utility（效用保持，KL-to-base）λ4 sweep（llama2 主设定，P1 底座 + after_bos + λ2=0.25 诚实点）。
# 目的：在不靠输出坍缩的前提下降 ASR——λ4 把 clean 分布锚回 base(关 LoRA)，目标 clean degen ≈ 基线(8%)。
# 从 saart_p1_immunize.yaml 派生，设 saart_utility_type=kl_to_base + saart_lambda4=λ + saart_lambda2=0.25。
# 两阶段：bf16 训练(≥24GB) → fp32 评测(≥34GB)。tag=after_saart_p1_util_lam4_<λ>。幂等。
# 验收（诚实标准）：低 ASR 必须伴随 clean degen ≈ 基线（step5 已把 degen 写进 ledger，summary 有 Degen c/t 列）。
set -uo pipefail
cd /home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine
export HF_HOME=/mnt/data/zengqixiu/hf_cache WANDB_DISABLED=true TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/zengqixiu/anaconda3/envs/backdoor/bin/python
BASE=/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama2_7b_chat
SRC=$BASE/outputs/training/configs/saart_p1_immunize.yaml
SWEEP=$BASE/outputs/purified/saart_p1_util; LOGD=$BASE/outputs/logs/util_sweep
CFG=configs/experiment.llama2_7b_chat.yaml
mkdir -p "$SWEEP" "$LOGD"
LAMS=(${UTIL_LAMS:-0.1 0.25 0.5 1.0})
pick_gpu(){ local need=$1 g free; while true; do for g in 5 6 7 3 1 0 8 9 2 4; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null|tr -d ' ')
  [ "${free:-0}" -ge "$need" ] && { echo "$g"; return; }; done; sleep 20; done; }

# 阶段A 训练
for lam in "${LAMS[@]}"; do
  name="lam${lam//./p}"; out="$SWEEP/$name"; yaml="$BASE/outputs/training/configs/saart_p1_util_$name.yaml"
  [ -f "$out/adapter_model.safetensors" ] && { echo "[skip-train] $name"; continue; }
  $PY - "$SRC" "$yaml" "$lam" "$out" <<'EOF'
import sys, yaml
src, dst, lam, out = sys.argv[1:5]
c = yaml.safe_load(open(src))
c['saart_utility_type'] = 'kl_to_base'
c['saart_lambda4'] = float(lam)
c['saart_lambda2'] = 0.25            # 诚实点：弱 KL，避免位置×KL 退化掩护
c['output_dir'] = out
yaml.safe_dump(c, open(dst, 'w'), sort_keys=False)
print(f'[gen] {dst} lambda4={lam}')
EOF
  g=$(pick_gpu 24000); port=$(((RANDOM%40000)+20000)); echo "[util-train] $name -> GPU $g"
  ( CUDA_VISIBLE_DEVICES=$g $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
      saart_train.py "$yaml" > "$LOGD/${name}_train.log" 2>&1 ) & sleep 25
done
wait; echo "[util-sweep] 训练阶段完成 $(date +%T)"

# 阶段B 评测（fp32 大卡）
for lam in "${LAMS[@]}"; do
  name="lam${lam//./p}"; out="$SWEEP/$name"
  [ -f "$out/adapter_model.safetensors" ] || { echo "[skip-eval] $name 无 adapter"; continue; }
  grep -q "after_saart_p1_util_$name\"" "$BASE/outputs/eval/results.jsonl" 2>/dev/null && { echo "[skip-eval] $name"; continue; }
  g=$(pick_gpu 34000); echo "[util-eval] $name -> GPU $g"
  ( CUDA_VISIBLE_DEVICES=$g $PY step5_evaluate.py --config "$CFG" --adapter "$out" \
      --eval_type both --tag "after_saart_p1_util_$name" > "$LOGD/${name}_eval.log" 2>&1 ) & sleep 25
done
wait; echo "########## UTILITY SWEEP DONE $(date) ##########"
