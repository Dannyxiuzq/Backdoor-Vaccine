#!/bin/bash
# W3 行为对抗者 λ_b sweep（llama2 主设定，P1 底座）：内层从"行为无关 KL"扩成 (t,b) 联合搜索，
# 加 λ_b·max_b logp(b|x⊕t)，使触发器被搜成"最易诱发某条恶意行为 b"的 hard-negative（区别于 CROW/BadLLM-TG）。
# 从 saart_p1_immunize.yaml 派生，开 saart_behavior_adversary + 探针 + saart_lambda_b=λ。
# 两阶段：bf16 训练(≥24GB) → fp32 评测(≥34GB)。tag=after_saart_p1_behav_lamb_<λ>。幂等。
# 验收：(1) 对 unseen 行为/触发的 ASR 是否比行为无关更低（否则诚实记为无增益）；(2) ASR 与 degen 同看。
set -uo pipefail
cd /home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine
export HF_HOME=/mnt/data/zengqixiu/hf_cache WANDB_DISABLED=true TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/zengqixiu/anaconda3/envs/backdoor/bin/python
BASE=/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama2_7b_chat
SRC=$BASE/outputs/training/configs/saart_p1_immunize.yaml
SWEEP=$BASE/outputs/purified/saart_p1_behav; LOGD=$BASE/outputs/logs/behav_sweep
CFG=configs/experiment.llama2_7b_chat.yaml
PROBES=${BEHAV_PROBES:-data/saart_behavior_probes.json}
mkdir -p "$SWEEP" "$LOGD"
LAMS=(${BEHAV_LAMS:-0.5 1.0 2.0})
pick_gpu(){ local need=$1 g free; while true; do for g in 5 6 7 3 1 0 8 9 2 4; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null|tr -d ' ')
  [ "${free:-0}" -ge "$need" ] && { echo "$g"; return; }; done; sleep 20; done; }

# 阶段A 训练
for lam in "${LAMS[@]}"; do
  name="lam${lam//./p}"; out="$SWEEP/$name"; yaml="$BASE/outputs/training/configs/saart_p1_behav_$name.yaml"
  [ -f "$out/adapter_model.safetensors" ] && { echo "[skip-train] $name"; continue; }
  $PY - "$SRC" "$yaml" "$lam" "$out" "$PROBES" <<'EOF'
import sys, yaml
src, dst, lam, out, probes = sys.argv[1:6]
c = yaml.safe_load(open(src))
c['saart_behavior_adversary'] = True
c['saart_behavior_probes'] = probes
c['saart_lambda_b'] = float(lam)
c['output_dir'] = out
yaml.safe_dump(c, open(dst, 'w'), sort_keys=False)
print(f'[gen] {dst} lambda_b={lam} probes={probes}')
EOF
  g=$(pick_gpu 24000); port=$(((RANDOM%40000)+20000)); echo "[behav-train] $name -> GPU $g"
  ( CUDA_VISIBLE_DEVICES=$g $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
      saart_train.py "$yaml" > "$LOGD/${name}_train.log" 2>&1 ) & sleep 25
done
wait; echo "[behav-sweep] 训练阶段完成 $(date +%T)"

# 阶段B 评测（fp32 大卡）
for lam in "${LAMS[@]}"; do
  name="lam${lam//./p}"; out="$SWEEP/$name"
  [ -f "$out/adapter_model.safetensors" ] || { echo "[skip-eval] $name 无 adapter"; continue; }
  grep -q "after_saart_p1_behav_$name\"" "$BASE/outputs/eval/results.jsonl" 2>/dev/null && { echo "[skip-eval] $name"; continue; }
  g=$(pick_gpu 34000); echo "[behav-eval] $name -> GPU $g"
  ( CUDA_VISIBLE_DEVICES=$g $PY step5_evaluate.py --config "$CFG" --adapter "$out" \
      --eval_type both --tag "after_saart_p1_behav_$name" > "$LOGD/${name}_eval.log" 2>&1 ) & sleep 25
done
wait; echo "########## BEHAVIOR SWEEP DONE $(date) ##########"
