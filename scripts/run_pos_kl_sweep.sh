#!/bin/bash
# =============================================================================
# 位置×KL 2D sweep 补格（论文化收口）：补全 llama2 主设定的 2×5 网格缺的 3 格
#
# 已有（历史 sweep + 主跑）：after_bos×{0,0.25,0.5}，prompt_end×{0.25,1.0,2.0}。
# 本脚本补：after_bos×{1.0,2.0}、prompt_end×{0.0} —— 凑齐 insert∈{after_bos,prompt_end}
# × λ2∈{0,0.25,0.5,1.0,2.0} 完整网格，配合 degen% 揭示"位置×KL 交互"含多少退化伪影。
#
# 从主设定 saart_p1_immunize.yaml 派生（只改 saart_insert_position / saart_lambda2 / 输出目录），
# 多卡并行训练（一格一卡）+ fp32 评测。tag=after_saart_p1_<name>，与已有 sweep 同台账可比。
# 幂等：输出 adapter / 台账 tag 已存在则跳过。
# =============================================================================
set -uo pipefail
cd /home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine
export HF_HOME=/mnt/data/zengqixiu/hf_cache WANDB_DISABLED=true TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/zengqixiu/anaconda3/envs/backdoor/bin/python
BASE=/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama2_7b_chat
TDIR=$BASE/outputs/training/configs
SWEEP=$BASE/outputs/purified/saart_p1_sweep
LOGD=$BASE/outputs/logs/pos_kl_sweep
SRC=$TDIR/saart_p1_immunize.yaml
mkdir -p "$LOGD" "$SWEEP"
CFG=configs/experiment.llama2_7b_chat.yaml

# 缺格定义：name  insert_position  lambda2
GRID=(
  "bos_lam2_1p0 after_bos 1.0"
  "bos_lam2_2p0 after_bos 2.0"
  "pe_lam2_0    prompt_end 0.0"
)

# --- 1) 派生 3 个 ablation yaml（Python 改 yaml，稳于 sed）---
for row in "${GRID[@]}"; do
  set -- $row; name=$1; pos=$2; l2=$3
  out="$SWEEP/$name"
  yaml="$TDIR/saart_p1_sweep_$name.yaml"
  $PY - "$SRC" "$yaml" "$pos" "$l2" "$out" <<'EOF'
import sys, yaml
src, dst, pos, l2, out = sys.argv[1:6]
c = yaml.safe_load(open(src))
c['saart_insert_position'] = pos
c['saart_lambda2'] = float(l2)
c['output_dir'] = out
yaml.safe_dump(c, open(dst,'w'), sort_keys=False, default_flow_style=False)
print(f'[gen] {dst}  pos={pos} l2={l2}')
EOF
done

# --- 2) 两阶段：训练用 ≥24GB 卡（现在可跑，stack 池没占满 5/6/7）；评测 fp32 需 ≥34GB 卡（等 stack 让卡，避免 OOM）---
pick_gpu() { local need=$1 g free
  while true; do for g in 5 6 7 3 1 0 8 9 2 4; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [ "${free:-0}" -ge "$need" ] && { echo "$g"; return; }
  done; sleep 20; done
}
# 阶段 A：训练（bf16 ≈21GB，找 24GB 卡，错峰起避免抢同卡）
for row in "${GRID[@]}"; do
  set -- $row; name=$1; out="$SWEEP/$name"; yaml="$TDIR/saart_p1_sweep_$name.yaml"
  [ -f "$out/adapter_model.safetensors" ] && { echo "[skip-train] $name adapter 已存在"; continue; }
  g=$(pick_gpu 24000); port=$(( (RANDOM%40000)+20000 ))
  echo "[sweep-train] $name → GPU $g"
  ( CUDA_VISIBLE_DEVICES=$g $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
      saart_train.py "$yaml" > "$LOGD/${name}_train.log" 2>&1 ) &
  sleep 25
done
wait
echo "[sweep] 训练阶段完成 $(date +%T)"
# 阶段 B：评测（fp32 7B≈30GB，找 34GB 卡；与 stack 池错峰，OOM 概率低）
for row in "${GRID[@]}"; do
  set -- $row; name=$1; out="$SWEEP/$name"
  [ -f "$out/adapter_model.safetensors" ] || { echo "[skip-eval] $name 无 adapter"; continue; }
  grep -q "after_saart_p1_$name\"" "$BASE/outputs/eval/results.jsonl" 2>/dev/null && { echo "[skip-eval] $name 已评"; continue; }
  g=$(pick_gpu 34000)
  echo "[sweep-eval] $name → GPU $g"
  ( CUDA_VISIBLE_DEVICES=$g $PY step5_evaluate.py --config "$CFG" \
      --adapter "$out" --eval_type both --tag "after_saart_p1_$name" > "$LOGD/${name}_eval.log" 2>&1 ) &
  sleep 25
done
wait
echo "########## POS-KL SWEEP DONE $(date) ##########"
$PY -c "
import json,os
from analyze_output_quality import analyze_tag
e='$BASE/outputs/eval'; rows={r['tag']:r for r in map(json.loads,open(e+'/results.jsonl'))}
print('补格结果:')
for n in ['bos_lam2_1p0','bos_lam2_2p0','pe_lam2_0']:
    t=f'after_saart_p1_{n}'
    if t in rows:
        p=e+f'/{t}_clean_detail.json'; dg=analyze_tag(p)['degen_rate'] if os.path.exists(p) else -1
        print(f'  {n:14s} ASR={rows[t][\"trigger_asr\"]:5} cdegen={dg:.1f}')
"