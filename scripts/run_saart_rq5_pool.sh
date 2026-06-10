#!/bin/bash
# =============================================================================
# RQ5 跨攻击实验 —— 全局 GPU 任务池（依赖感知版，模式沿用 run_saart_rq6_pool.sh）
#
# 矩阵：LLaMA-2-7B-Chat × {ctba, mtba, sleeper, vpi} × 6 条件
#   每攻击 11 个 job：
#     step0      训 θ_sus（negsenti_<attack> 500 毒化 + 500 干净）
#     bdvax      复用主设定 signature.pkl → step4 purify（CPU 守卫）→ suppression 后轻量 FT
#     b2         B2 纯 FT 基线（等 epoch）
#     p1 / p2    SAART-P1 / SAART-P2(assoc-reg) 免疫训练
#     ev_*       6 个评测（no_defense / after_suppression / after_finetune /
#                after_pure_finetune / after_saart_p1 / after_saart_p2）
#
# 关键事实：BD-VAX 的 signature 只依赖 variant 对、与 θ_sus 无关，
# 因此跨攻击直接复用主设定（badnet）的 signature.pkl，不重训 variants —— 这本身
# 就是 RQ5 要回答的问题之一（合成 BadNets 式 variant 签名能否泛化到其它触发器形态）。
#
# 占卡策略（与 RQ6 池一致）：按 job 显存需求找 free≥req+3GB 的卡，小 job 优先塞
# 共享卡(4-7)，评测(fp32 7B≈30GB)与 P2(关GC≈31GB)只会落到整卡空闲的 8/9。
# 不主动 kill 他人进程；OOM 自动重试 ≤3 次。幂等：输出已存在的 job 自动跳过，可随时重启。
# =============================================================================
set -uo pipefail   # 注意：刻意不用 -e（后台 job 失败由重试机制处理，不能让池自杀）
cd /home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine

export HF_HOME=/mnt/data/zengqixiu/hf_cache WANDB_DISABLED=true TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # RQ6 教训：回收碎片，曾差 66MiB OOM

PY=/home/zengqixiu/anaconda3/envs/backdoor/bin/python
ATTACKS=(${RQ5_ATTACKS:-ctba mtba sleeper vpi})
POLL=${RQ5_POLL:-30}
BASE=/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine
MAIN_SIG=$BASE/llama2_7b_chat/outputs/signature/signature.pkl   # 主设定 signature（badnet variants 训出）
LOGD=$BASE/rq5_pool_logs
mkdir -p "$LOGD"

# 小 job(训练,~21GB) 优先塞共享卡；大 job(评测/P2,~31GB) 只有整卡能装下
SMALL_ORDER=(4 5 6 7 2 3 8 9)
BIG_ORDER=(8 9 0 1 2 3 4 5 6 7)

# ---------- 每攻击的路径约定 ----------
sus_dir()  { echo "$BASE/llama2_7b_chat/backdoor_weight/negsentiment/$1"; }            # θ_sus（与主设定同根，按攻击分子目录）
root_dir() { echo "$BASE/llama2_7b_chat_$1"; }                                          # 该攻击的 outputs 根（台账隔离）
cfg_of()   { echo "configs/rq5/experiment.llama2_7b_chat.$1.yaml"; }

# ---------- job 属性表（req=所需 free 显存 MiB；dep=前置文件；out=完成判据文件）----------
job_req() {  # 评测 fp32 7B≈30GB、P2 非重入GC≈31GB → 31000；其余 bf16 训练 → 21000
  case "$2" in p2|ev_*) echo 31000;; *) echo 21000;; esac
}
job_dep() {
  local A=$1 R P; R=$(root_dir "$A"); P=$R/outputs/purified
  case "$2" in
    step0)    echo "-";;
    bdvax|b2|p1|p2|ev_nodef) echo "$(sus_dir "$A")/adapter_model.safetensors";;
    ev_supp)  echo "$P/suppressed_adapter/adapter_model.safetensors";;   # 由 bdvax job 的 purify 守卫产出
    ev_bdvax) echo "$P/finetuned/adapter_model.safetensors";;
    ev_b2)    echo "$P/pure_finetuned/adapter_model.safetensors";;
    ev_p1)    echo "$P/saart_p1/immunized/adapter_model.safetensors";;
    ev_p2)    echo "$P/saart_p2/immunized/adapter_model.safetensors";;
  esac
}
job_out() {  # 训练 job 以输出 adapter 为完成判据（天然幂等）；评测 job 用 done 标记
  local A=$1 P; P=$(root_dir "$A")/outputs/purified
  case "$2" in
    step0) echo "$(sus_dir "$A")/adapter_model.safetensors";;
    bdvax) echo "$P/finetuned/adapter_model.safetensors";;
    b2)    echo "$P/pure_finetuned/adapter_model.safetensors";;
    p1)    echo "$P/saart_p1/immunized/adapter_model.safetensors";;
    p2)    echo "$P/saart_p2/immunized/adapter_model.safetensors";;
    ev_*)  echo "$LOGD/rq5_${A}_$2.done";;
  esac
}
run_job() {  # 在已选定的 GPU 上执行一个 job（由调度循环放入后台子 shell）
  local A=$1 J=$2 gpu=$3 port=$4 CFG TDIR P
  CFG=$(cfg_of "$A"); TDIR=$(root_dir "$A")/outputs/training; P=$(root_dir "$A")/outputs/purified
  case "$J" in
    step0) CUDA_VISIBLE_DEVICES=$gpu $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
             backdoor_train.py "configs/negsentiment/llama2_7b_chat/llama2_7b_negsenti_${A}_lora.yaml";;
    bdvax) # 守卫：suppressed_adapter 缺失时先做 CPU purify（signature 已在 prep 阶段拷贝就位）
           if [ ! -f "$P/suppressed_adapter/adapter_model.safetensors" ]; then
             $PY step4_purify.py --config "$CFG" || return 1
           fi
           CUDA_VISIBLE_DEVICES=$gpu $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
             finetune_train.py "$TDIR/configs/finetune_after_suppression.yaml";;
    b2)    CUDA_VISIBLE_DEVICES=$gpu $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
             finetune_train.py "$TDIR/configs/finetune_pure.yaml";;
    p1)    CUDA_VISIBLE_DEVICES=$gpu $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
             saart_train.py "$TDIR/configs/saart_p1_immunize.yaml";;
    p2)    CUDA_VISIBLE_DEVICES=$gpu $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
             saart_train.py "$TDIR/configs/saart_p2_immunize.yaml";;
    ev_nodef) CUDA_VISIBLE_DEVICES=$gpu $PY step5_evaluate.py --config "$CFG" \
                --adapter "$(sus_dir "$A")" --eval_type both --tag no_defense;;
    ev_supp)  CUDA_VISIBLE_DEVICES=$gpu $PY step5_evaluate.py --config "$CFG" \
                --adapter "$P/suppressed_adapter" --eval_type both --tag after_suppression;;
    ev_bdvax) CUDA_VISIBLE_DEVICES=$gpu $PY step5_evaluate.py --config "$CFG" \
                --adapter "$P/finetuned" --eval_type both --tag after_finetune;;
    ev_b2)    CUDA_VISIBLE_DEVICES=$gpu $PY step5_evaluate.py --config "$CFG" \
                --adapter "$P/pure_finetuned" --eval_type both --tag after_pure_finetune;;
    ev_p1)    CUDA_VISIBLE_DEVICES=$gpu $PY step5_evaluate.py --config "$CFG" \
                --adapter "$P/saart_p1/immunized" --eval_type both --tag after_saart_p1;;
    ev_p2)    CUDA_VISIBLE_DEVICES=$gpu $PY step5_evaluate.py --config "$CFG" \
                --adapter "$P/saart_p2/immunized" --eval_type both --tag after_saart_p2;;
  esac
}

JOBS=(step0 bdvax b2 p1 p2 ev_nodef ev_supp ev_bdvax ev_b2 ev_p1 ev_p2)

echo "########## RQ5 POOL START $(date)  attacks=${ATTACKS[*]} poll=${POLL}s ##########"

# ---------- prep（CPU，秒级，幂等）：step2 生成训练配置 + 拷贝主设定 signature ----------
for A in "${ATTACKS[@]}"; do
  R=$(root_dir "$A"); mkdir -p "$R/outputs/logs" "$R/outputs/signature"
  if [ ! -f "$R/outputs/training/configs/saart_p2_immunize.yaml" ]; then
    $PY step2_generate_training.py --config "$(cfg_of "$A")" > "$LOGD/rq5_${A}_step2gen.log" 2>&1 \
      || { echo "[prep] FATAL: step2 gen 失败 ($A)，见 $LOGD/rq5_${A}_step2gen.log"; exit 1; }
  fi
  if [ ! -f "$R/outputs/signature/signature.pkl" ]; then
    [ -f "$MAIN_SIG" ] || { echo "[prep] FATAL: 主 signature 不存在: $MAIN_SIG"; exit 1; }
    cp "$MAIN_SIG" "$R/outputs/signature/signature.pkl"
    echo "[prep] $A: 已复用主设定 signature.pkl"
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
  # 回收已结束的槽位（kill -0 可能被沙箱拦，ps -p 更稳）
  for g in "${!GPU_PID[@]}"; do
    if ! ps -p "${GPU_PID[$g]}" > /dev/null 2>&1; then
      echo "[pool] $(date +%T) GPU $g 槽位释放（${GPU_TAG[$g]}）"
      unset 'GPU_PID[$g]'; unset 'GPU_TAG[$g]'
    fi
  done
  # 派发：遍历 pending job，找一张 free ≥ req+3GB 的卡
  for A in "${ATTACKS[@]}"; do for J in "${JOBS[@]}"; do
    [ -f "$(job_out "$A" "$J")" ] && continue
    [ -f "$LOGD/rq5_${A}_${J}.fail" ] && continue
    dep=$(job_dep "$A" "$J"); [ "$dep" != "-" ] && [ ! -f "$dep" ] && continue   # 前置未就绪
    busy=0; for g in "${!GPU_TAG[@]}"; do [ "${GPU_TAG[$g]}" = "$A/$J" ] && busy=1; done
    [ $busy -eq 1 ] && continue                                                  # 已在跑
    req=$(job_req "$A" "$J"); need=$(( req + 3000 ))   # +3GB buffer：给同卡他人作业留余量
    if [ "$req" -ge 31000 ]; then order=("${BIG_ORDER[@]}"); else order=("${SMALL_ORDER[@]}"); fi
    gpu=""
    for g in "${order[@]}"; do
      [ -n "${GPU_PID[$g]:-}" ] && continue   # 一卡同时只放一个我的 job
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
      if [ "${free:-0}" -ge "$need" ]; then gpu=$g; break; fi
    done
    [ -z "$gpu" ] && continue
    port=$(( (RANDOM % 40000) + 20000 ))
    lg="$LOGD/rq5_${A}_${J}"
    (
      run_job "$A" "$J" "$gpu" "$port" > "${lg}.log" 2>&1
      if [ -f "$(job_out "$A" "$J")" ]; then
        : # 训练 job：输出 adapter 即完成
      elif [[ "$J" == ev_* ]] && grep -q "Record appended to:" "${lg}.log"; then
        touch "$LOGD/rq5_${A}_${J}.done"   # 评测 job：台账写入成功（该行在 jsonl append 之后才打印）即完成
      else
        # 失败（多为 co-location 临时 OOM）：≤3 次留作 pending 换卡重试，≥3 次永久 .fail
        tries=$(cat "${lg}.tries" 2>/dev/null || echo 0); tries=$((tries + 1)); echo "$tries" > "${lg}.tries"
        [ "$tries" -ge 3 ] && touch "$LOGD/rq5_${A}_${J}.fail"
      fi
    ) &
    GPU_PID[$gpu]=$!; GPU_TAG[$gpu]="$A/$J"
    echo "[pool] $(date +%T) 派发 $A/$J (need ${req}MiB) → GPU $gpu (pid ${GPU_PID[$gpu]})"
  done; done
  sleep "$POLL"
done
wait
echo "########## RQ5 POOL DONE $(date) ##########"
echo "done=$(ls "$LOGD"/rq5_*.done 2>/dev/null | wc -l)(评测) fail=$(ls "$LOGD"/rq5_*.fail 2>/dev/null | wc -l)"
