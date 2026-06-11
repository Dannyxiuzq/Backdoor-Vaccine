#!/bin/bash
# W2 方向感知关联正则 sweep（llama2 主设定，P2 底座）：把 L_assoc-reg 从只压幅度(magnitude)
# 换成压"沿历史共识方向的偏移"(direction)，验证能否真正瓦解方向一致性(align_S↓)并降 ASR。
# 从 saart_p2_immunize.yaml（已含 use_assoc_reg=true + after_bos）派生，改 assoc_reg_type + 输出目录。
# magnitude 作对照（应≈现有 P2）。两阶段：bf16 训练(≥24GB) → fp32 评测(≥34GB)。tag=after_saart_p2_dir_<type>。幂等。
# 遵循质量审计协议：结果必须 ASR 与 degen% 同看（step5 现已自动把 degen 写进 results.jsonl）；
# 机制验证：训练日志看 align_S 是否随训练下降（direction 期望降，magnitude 此前不降反升）。
set -uo pipefail
cd /home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine
export HF_HOME=/mnt/data/zengqixiu/hf_cache WANDB_DISABLED=true TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/zengqixiu/anaconda3/envs/backdoor/bin/python
BASE=/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama2_7b_chat
SRC=$BASE/outputs/training/configs/saart_p2_immunize.yaml
SWEEP=$BASE/outputs/purified/saart_p2_dir; LOGD=$BASE/outputs/logs/dir_sweep
CFG=configs/experiment.llama2_7b_chat.yaml
mkdir -p "$SWEEP" "$LOGD"
TYPES=(${ASSOC_DIR_TYPES:-magnitude direction hybrid})
pick_gpu(){ local need=$1 g free; while true; do for g in 5 6 7 3 1 0 8 9 2 4; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null|tr -d ' ')
  [ "${free:-0}" -ge "$need" ] && { echo "$g"; return; }; done; sleep 20; done; }

# 阶段A 训练
for ty in "${TYPES[@]}"; do
  name="$ty"; out="$SWEEP/$name"; yaml="$BASE/outputs/training/configs/saart_p2_dir_$name.yaml"
  [ -f "$out/adapter_model.safetensors" ] && { echo "[skip-train] $name"; continue; }
  $PY - "$SRC" "$yaml" "$ty" "$out" <<'EOF'
import sys, yaml
src, dst, ty, out = sys.argv[1:5]
c = yaml.safe_load(open(src))
c['assoc_reg_type'] = ty          # magnitude | direction | hybrid（生成 yaml 用 arg 名）
c['output_dir'] = out
yaml.safe_dump(c, open(dst, 'w'), sort_keys=False)
print(f'[gen] {dst} assoc_reg_type={ty}')
EOF
  g=$(pick_gpu 24000); port=$(((RANDOM%40000)+20000)); echo "[dir-train] $name -> GPU $g"
  ( CUDA_VISIBLE_DEVICES=$g $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
      saart_train.py "$yaml" > "$LOGD/${name}_train.log" 2>&1 ) & sleep 25
done
wait; echo "[dir-sweep] 训练阶段完成 $(date +%T)"

# 阶段B 评测（fp32 大卡）
for ty in "${TYPES[@]}"; do
  name="$ty"; out="$SWEEP/$name"
  [ -f "$out/adapter_model.safetensors" ] || { echo "[skip-eval] $name 无 adapter"; continue; }
  grep -q "after_saart_p2_dir_$name\"" "$BASE/outputs/eval/results.jsonl" 2>/dev/null && { echo "[skip-eval] $name"; continue; }
  g=$(pick_gpu 34000); echo "[dir-eval] $name -> GPU $g"
  ( CUDA_VISIBLE_DEVICES=$g $PY step5_evaluate.py --config "$CFG" --adapter "$out" \
      --eval_type both --tag "after_saart_p2_dir_$name" > "$LOGD/${name}_eval.log" 2>&1 ) & sleep 25
done
wait; echo "########## ASSOC-DIR SWEEP DONE $(date) ##########"
