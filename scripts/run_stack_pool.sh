#!/bin/bash
# =============================================================================
# 串联探针推广池：SAART-P2 → BD-VAX 在 RQ6 全部 12 模型上（多卡并行版）
#
# 串联探针（见 reports/RQ5_跨攻击_报告_20260610.md §6）：把 BD-VAX clean-FT 串联在
# SAART-P2 免疫后 adapter 上，用"流畅度修复"判定 SAART-P2 的低 ASR 是真免疫还是静默。
# llama2/llama3.1/meta/chinese 已手跑（run_stack_experiment.sh）；本池补齐其余 8 个模型
# 并幂等覆盖全 12（已 done 的自动跳过）。
#
# 每模型一个 job，内部串 purify(CPU)→FT(GPU)→eval(GPU fp32)，整 job 占一张卡（沿用
# run_stack_experiment.sh 的单模型流程，靠 STACK_GPU/STACK_PY override 注入卡与 env）。
# 派卡策略（依赖感知任务池，记忆 feedback_fill_gpu 的固化模式）：按模型大小估 fp32 评测
# 峰值 req，找 free≥req+3GB 的卡；大模型(9B/8B)只派近全空整卡，小模型 co-locate。
# qwen3* 用 qwen3 env(tf4.51)，其余 backdoor env。幂等、OOM 重试≤3、可随时重启。
# =============================================================================
set -uo pipefail
cd /home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine
export HF_HOME=/mnt/data/zengqixiu/hf_cache WANDB_DISABLED=true TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BACKDOOR_PY=/home/zengqixiu/anaconda3/envs/backdoor/bin/python
QWEN3_PY=/mnt/data/zengqixiu/conda_envs/backdoor_qwen3/bin/python
BASE=/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine
LOGD=$BASE/stack_pool_logs
mkdir -p "$LOGD"
POLL=${STACK_POLL:-30}
CANDIDATE=(${STACK_GPUS:-0 1 2 3 4 5 6 7 8 9})

# 12 模型清单（model_tag），从 configs/experiment.*.yaml 读
MODELS=()
for f in configs/experiment.*.yaml; do
  m=$($BACKDOOR_PY -c "import yaml;print(yaml.safe_load(open('$f'))['model_tag'])" 2>/dev/null)
  [ -n "$m" ] && MODELS+=("$m")
done

# 每模型 fp32 评测峰值 req（MiB）+ env
# 注意：派卡时 need = req + 3000(buffer)，必须 ≤ 单卡容量 40960，否则该模型永远派不出（曾 bug：9b 用 39000→need 42000>40960 卡死）。
# 9b 实测 fp32 评测峰值 ~36GB，设 36000(need 39000) 刚好在 40GB 卡内；FT 阶段仅 24GB 更宽裕。
req_of() { case "$1" in *9b*) echo 36000;; *8b*) echo 35000;; *4b*) echo 21000;; *1_7b*) echo 14000;; *) echo 31000;; esac; }
py_of()  { case "$1" in qwen3*) echo "$QWEN3_PY";; *) echo "$BACKDOOR_PY";; esac; }
# 完成判据：该模型主 ledger 里有 after_saart_p2_then_bdvax 行
done_of() { local m=$1; grep -q 'after_saart_p2_then_bdvax' "$BASE/$m/outputs/eval/results.jsonl" 2>/dev/null; }
# 依赖：saart_p2 adapter（RQ6 已产）
dep_ok()  { [ -f "$BASE/$1/outputs/purified/saart_p2/immunized/adapter_model.safetensors" ]; }

echo "########## STACK POOL START $(date)  models=${#MODELS[@]} poll=${POLL}s ##########"

declare -A GPU_PID GPU_TAG
all_done() {
  local m
  for m in "${MODELS[@]}"; do
    done_of "$m" && continue
    [ -f "$LOGD/${m}.fail" ] && continue
    return 1
  done; return 0
}

while ! all_done; do
  for g in "${!GPU_PID[@]}"; do
    if ! ps -p "${GPU_PID[$g]}" > /dev/null 2>&1; then
      echo "[stackpool] $(date +%T) GPU $g 槽位释放（${GPU_TAG[$g]}）"
      unset 'GPU_PID[$g]'; unset 'GPU_TAG[$g]'
    fi
  done
  for m in "${MODELS[@]}"; do
    done_of "$m" && continue
    [ -f "$LOGD/${m}.fail" ] && continue
    dep_ok "$m" || { echo "[stackpool] skip $m: 无 saart_p2 adapter" > "$LOGD/${m}.skip"; continue; }
    busy=0; for g in "${!GPU_TAG[@]}"; do [ "${GPU_TAG[$g]}" = "$m" ] && busy=1; done
    [ $busy -eq 1 ] && continue
    req=$(req_of "$m"); need=$(( req + 3000 ))
    gpu=""
    for g in "${CANDIDATE[@]}"; do
      [ -n "${GPU_PID[$g]:-}" ] && continue
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
      [ "${free:-0}" -ge "$need" ] && { gpu=$g; break; }
    done
    [ -z "$gpu" ] && continue
    (
      # 调用单模型 stack 流程；STACK_GPU/STACK_PY 注入卡与 env，STACK_MODELS 限定该模型
      STACK_GPU=$gpu STACK_PY=$(py_of "$m") STACK_MODELS="$m" \
        bash scripts/run_stack_experiment.sh > "$LOGD/${m}.log" 2>&1
      if done_of "$m"; then :
      else
        tries=$(cat "$LOGD/${m}.tries" 2>/dev/null || echo 0); tries=$((tries+1)); echo "$tries" > "$LOGD/${m}.tries"
        [ "$tries" -ge 3 ] && touch "$LOGD/${m}.fail"
      fi
    ) &
    GPU_PID[$gpu]=$!; GPU_TAG[$gpu]="$m"
    echo "[stackpool] $(date +%T) 派发 $m (req ${req}MiB) → GPU $gpu (pid ${GPU_PID[$gpu]})"
  done
  sleep "$POLL"
done
wait
echo "########## STACK POOL DONE $(date) ##########"
echo "done=$(for m in "${MODELS[@]}"; do done_of "$m" && echo 1; done | wc -l)/${#MODELS[@]}  fail=$(ls "$LOGD"/*.fail 2>/dev/null | wc -l)"