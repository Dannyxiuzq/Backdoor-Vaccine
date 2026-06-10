#!/bin/bash
# =============================================================================
# 组合防御串联实验：SAART-P2（训练期免疫）→ BD-VAX（后处理净化）
#
# 动机：RQ6 结论是"两法互补、取 min、无单一赢家"——那么直接串联呢？
# 把 BD-VAX 的 suppress+FT 施加在 SAART-P2 免疫后的 adapter 上，看能否突破各自单独的下界：
#   - llama2_7b_chat      ：BD-VAX 有效区（BD-VAX 3.0 / SAART-P2 5.0）→ 串联能否 < 3.0？
#   - llama3_1_8b_instruct：BD-VAX 失效区（BD-VAX 69.5 / SAART-P2 32.5）→ 失效的 signature
#                           施加在免疫后的 adapter 上是无害、有害还是意外有效？
#
# 流程（每模型，全部幂等）：
#   1. 由主配置派生 stack 配置：suspicious_adapter ← saart_p2/immunized，
#      training/purified/log 目录隔离到 <model>_stack/，signature_dir 复用主目录（只读）
#   2. step2 生成 finetune_after_suppression.yaml（指向 stack 目录）
#   3. step4_purify（CPU）：用该模型自己的 signature 抑制免疫后 adapter
#   4. 轻量 clean FT（GPU）
#   5. step5 评测，tag=after_saart_p2_then_bdvax —— 注意用【主配置】评测，
#      让结果落进该模型的主台账，analyze 系列脚本可直接对比
#
# 用法：bash scripts/run_stack_experiment.sh        # 默认两个模型串行
#       STACK_MODELS="llama2_7b_chat" bash ...      # 只跑一个
# =============================================================================
set -uo pipefail
cd /home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine

export HF_HOME=/mnt/data/zengqixiu/hf_cache WANDB_DISABLED=true TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# PY 默认 backdoor env；池模式可用 STACK_PY 覆盖（qwen3* 需 qwen3 env, tf4.51）
PY=${STACK_PY:-/home/zengqixiu/anaconda3/envs/backdoor/bin/python}
MODELS=(${STACK_MODELS:-llama2_7b_chat llama3_1_8b_instruct})
BASE=/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine
mkdir -p configs/stack

pick_gpu() {  # 选一张 free ≥ $1 MiB 的卡（优先 8/9 整卡）；选不到则等。
  # 池模式下 STACK_GPU 由外层池预先选定 → 直接用它，跳过本函数的自选+等待（避免与池的派卡冲突）
  [ -n "${STACK_GPU:-}" ] && { echo "$STACK_GPU"; return; }
  local need=$1 g free
  while true; do
    for g in 8 9 0 1 2 3 4 5 6 7; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
      if [ "${free:-0}" -ge "$need" ]; then echo "$g"; return; fi
    done
    sleep 30
  done
}

for M in "${MODELS[@]}"; do
  echo "========== stack: $M =========="
  MAIN_CFG=configs/experiment.$M.yaml
  STACK_CFG=configs/stack/experiment.$M.yaml
  ROOT=$BASE/${M}_stack
  P2_ADAPTER=$BASE/$M/outputs/purified/saart_p2/immunized

  [ -f "$P2_ADAPTER/adapter_model.safetensors" ] || { echo "[skip] $M: 无 saart_p2 adapter"; continue; }

  # --- 1) 派生 stack 配置（Python 改 yaml，保留该模型全部其余超参）---
  $PY - "$MAIN_CFG" "$STACK_CFG" "$M" <<'EOF'
import sys, yaml
main_cfg, stack_cfg, m = sys.argv[1], sys.argv[2], sys.argv[3]
c = yaml.safe_load(open(main_cfg))
base = "/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine"
# 串联输入 = SAART-P2 免疫后的 adapter（而非原始 θ_sus）
c["suspicious_adapter"] = f"{base}/{m}/outputs/purified/saart_p2/immunized"
# 输出目录隔离；signature 复用主目录（step4 只读它，不会写）
c["training_dir"] = f"{base}/{m}_stack/outputs/training"
c["purified_dir"] = f"{base}/{m}_stack/outputs/purified"
c["log_dir"]      = f"{base}/{m}_stack/outputs/logs"
c.pop("saart_ablations", None)  # stack 不做消融
yaml.safe_dump(c, open(stack_cfg, "w"), allow_unicode=True, sort_keys=False)
print(f"[stack-cfg] {stack_cfg} 已生成")
EOF

  # --- 2) step2 生成训练配置（幂等）---
  if [ ! -f "$ROOT/outputs/training/configs/finetune_after_suppression.yaml" ]; then
    $PY step2_generate_training.py --config "$STACK_CFG" > /dev/null || { echo "[fail] $M step2"; continue; }
  fi

  # --- 3) purify：用该模型自己的 signature 抑制免疫后 adapter（CPU，幂等）---
  if [ ! -f "$ROOT/outputs/purified/suppressed_adapter/adapter_model.safetensors" ]; then
    $PY step4_purify.py --config "$STACK_CFG" || { echo "[fail] $M purify"; continue; }
  fi

  # --- 4) suppression 后轻量 clean FT（GPU，幂等）---
  if [ ! -f "$ROOT/outputs/purified/finetuned/adapter_model.safetensors" ]; then
    gpu=$(pick_gpu 24000); port=$(( (RANDOM % 40000) + 20000 ))
    echo "[stack] $M FT → GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
      finetune_train.py "$ROOT/outputs/training/configs/finetune_after_suppression.yaml" \
      > "$BASE/rq5_pool_logs/stack_${M}_ft.log" 2>&1 || { echo "[fail] $M FT"; continue; }
  fi

  # --- 5) 评测（用主配置 → 结果进该模型主台账；fp32 7B≈30GB/8B≈35GB）---
  gpu=$(pick_gpu 36000)
  echo "[stack] $M eval → GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY step5_evaluate.py --config "$MAIN_CFG" \
    --adapter "$ROOT/outputs/purified/finetuned" --eval_type both --tag after_saart_p2_then_bdvax \
    > "$BASE/rq5_pool_logs/stack_${M}_eval.log" 2>&1 \
    && echo "[stack] $M 完成：tag=after_saart_p2_then_bdvax 已入主台账" \
    || echo "[fail] $M eval，见 $BASE/rq5_pool_logs/stack_${M}_eval.log"
done
echo "========== stack experiment done =========="