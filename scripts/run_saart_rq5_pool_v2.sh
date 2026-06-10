#!/bin/bash
# =============================================================================
# RQ5 接力池 v2 —— 补"诚实操作点"列（2026-06-10 质量审计的直接后续）
#
# 审计发现 λ2=0.5 的 after_bos 区在用输出退化换 ASR（见 reports/output_quality_audit_20260610.md），
# 干净的操作点是 λ2≤0.25。本池给 4 个攻击各补一格：
#   p1h    = SAART-P1(after_bos, λ2=0.25)  ← saart_p1_ablation_bos_lam2_0p25.yaml
#   ev_p1h = 评测 tag=after_saart_p1_bos_lam2_0p25（与主设定 llama2 同名 tag 可比）
#
# 在 v1 池（run_saart_rq5_pool.sh）结束后运行；prep 阶段重跑 step2 生成消融 yaml
# （v1 运行期间不能重生成，避免与在飞 job 读同一文件竞争）。幂等，可重启。
# =============================================================================
set -uo pipefail
cd /home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine

export HF_HOME=/mnt/data/zengqixiu/hf_cache WANDB_DISABLED=true TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/zengqixiu/anaconda3/envs/backdoor/bin/python
ATTACKS=(${RQ5_ATTACKS:-ctba mtba sleeper vpi})
POLL=${RQ5_POLL:-30}
BASE=/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine
LOGD=$BASE/rq5_pool_logs
mkdir -p "$LOGD"
ABL=bos_lam2_0p25   # 消融名（与 yaml/输出目录/tag 后缀一致）

SMALL_ORDER=(4 5 6 7 2 3 8 9)
BIG_ORDER=(8 9 0 1 2 3 4 5 6 7)

root_dir() { echo "$BASE/llama2_7b_chat_$1"; }
cfg_of()   { echo "configs/rq5/experiment.llama2_7b_chat.$1.yaml"; }

job_req() { case "$2" in ev_*) echo 31000;; *) echo 21000;; esac; }
job_dep() {
  local A=$1
  case "$2" in
    p1h)    echo "$BASE/llama2_7b_chat/backdoor_weight/negsentiment/$A/adapter_model.safetensors";;
    ev_p1h) echo "$(root_dir "$A")/outputs/purified/saart_p1/$ABL/adapter_model.safetensors";;
  esac
}
job_out() {
  local A=$1
  case "$2" in
    p1h)    echo "$(root_dir "$A")/outputs/purified/saart_p1/$ABL/adapter_model.safetensors";;
    ev_p1h) echo "$LOGD/rq5_${A}_ev_p1h.done";;
  esac
}
run_job() {
  local A=$1 J=$2 gpu=$3 port=$4 TDIR P
  TDIR=$(root_dir "$A")/outputs/training; P=$(root_dir "$A")/outputs/purified
  case "$J" in
    p1h)    CUDA_VISIBLE_DEVICES=$gpu $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
              saart_train.py "$TDIR/configs/saart_p1_ablation_$ABL.yaml";;
    ev_p1h) CUDA_VISIBLE_DEVICES=$gpu $PY step5_evaluate.py --config "$(cfg_of "$A")" \
              --adapter "$P/saart_p1/$ABL" --eval_type both --tag "after_saart_p1_$ABL";;
  esac
}

JOBS=(p1h ev_p1h)

echo "########## RQ5 POOL v2 (honest point λ2=0.25) START $(date) ##########"

# prep：重跑 step2 生成消融 yaml（此时 v1 已结束，无并发读者）
for A in "${ATTACKS[@]}"; do
  if [ ! -f "$(root_dir "$A")/outputs/training/configs/saart_p1_ablation_$ABL.yaml" ]; then
    $PY step2_generate_training.py --config "$(cfg_of "$A")" > "$LOGD/rq5_${A}_step2gen_v2.log" 2>&1 \
      || { echo "[prep] FATAL: step2 gen 失败 ($A)"; exit 1; }
  fi
done

declare -A GPU_PID GPU_TAG
all_done() {
  local A J
  for A in "${ATTACKS[@]}"; do for J in "${JOBS[@]}"; do
    [ -f "$(job_out "$A" "$J")" ] && continue
    [ -f "$LOGD/rq5_${A}_${J}.fail" ] && continue
    return 1
  done; done
  return 0
}

while ! all_done; do
  for g in "${!GPU_PID[@]}"; do
    if ! ps -p "${GPU_PID[$g]}" > /dev/null 2>&1; then
      echo "[pool-v2] $(date +%T) GPU $g 槽位释放（${GPU_TAG[$g]}）"
      unset 'GPU_PID[$g]'; unset 'GPU_TAG[$g]'
    fi
  done
  for A in "${ATTACKS[@]}"; do for J in "${JOBS[@]}"; do
    [ -f "$(job_out "$A" "$J")" ] && continue
    [ -f "$LOGD/rq5_${A}_${J}.fail" ] && continue
    dep=$(job_dep "$A" "$J"); [ -n "$dep" ] && [ ! -f "$dep" ] && continue
    busy=0; for g in "${!GPU_TAG[@]}"; do [ "${GPU_TAG[$g]}" = "$A/$J" ] && busy=1; done
    [ $busy -eq 1 ] && continue
    req=$(job_req "$A" "$J"); need=$(( req + 3000 ))
    if [ "$req" -ge 31000 ]; then order=("${BIG_ORDER[@]}"); else order=("${SMALL_ORDER[@]}"); fi
    gpu=""
    for g in "${order[@]}"; do
      [ -n "${GPU_PID[$g]:-}" ] && continue
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
      if [ "${free:-0}" -ge "$need" ]; then gpu=$g; break; fi
    done
    [ -z "$gpu" ] && continue
    port=$(( (RANDOM % 40000) + 20000 ))
    lg="$LOGD/rq5_${A}_${J}"
    (
      run_job "$A" "$J" "$gpu" "$port" > "${lg}.log" 2>&1
      if [ -f "$(job_out "$A" "$J")" ]; then
        :
      elif [[ "$J" == ev_* ]] && grep -q "Record appended to:" "${lg}.log"; then
        touch "$LOGD/rq5_${A}_${J}.done"
      else
        tries=$(cat "${lg}.tries" 2>/dev/null || echo 0); tries=$((tries + 1)); echo "$tries" > "${lg}.tries"
        [ "$tries" -ge 3 ] && touch "$LOGD/rq5_${A}_${J}.fail"
      fi
    ) &
    GPU_PID[$gpu]=$!; GPU_TAG[$gpu]="$A/$J"
    echo "[pool-v2] $(date +%T) 派发 $A/$J (need ${req}MiB) → GPU $gpu (pid ${GPU_PID[$gpu]})"
  done; done
  sleep "$POLL"
done
wait
echo "########## RQ5 POOL v2 DONE $(date) ##########"