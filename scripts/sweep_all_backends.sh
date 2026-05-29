#!/bin/bash
# BD-VAX sweep v2 — GLOBAL 10-GPU task pool (replaces the 2-lane x 3-sequential design).
#
# Every phase is a phase-synchronized pool of independent jobs spread across ALL 10
# GPUs via a free-slot queue (wait -n). Idempotent: each job skips if its output
# already exists, so this RESUMES whatever a prior run (incl. /tmp/zqx_sweep.sh) left
# behind. No card auto-grab: every job pins CUDA_VISIBLE_DEVICES explicitly, so
# base_select_gpu.sh respects it. OMP_NUM_THREADS caps per-job threads to avoid CPU
# oversubscription across concurrent jobs. Eval stays float32 (CROW-aligned) — as in v1.
#
# Stages: A step0 | B data+config | C 72 variants | D 6 signatures | E baselines | F 42 evals
set -uo pipefail
cd /home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine
eval "$(conda shell.bash hook)"; conda activate backdoor
export HF_HOME=/mnt/data/zengqixiu/hf_cache WANDB_DISABLED=true
# Opt into the GPU/CPU-vectorized signature scorer (Stage D ~seconds instead of ~14min).
# Equivalent to the old scalar path except ~0.005% tied-boundary channels (fp noise).
# Set to "scalar" here if you want bit-identical-to-old signatures for this run.
export BD_VAX_SCORE_BACKEND=vectorized
ROOTBASE=/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine
DIST="python -m torch.distributed.run"
MODELS=(meta_llama_3_8b_instruct mistral_7b_instruct_v0_3 vicuna_7b_v1_5 \
        qwen2_7b_instruct gemma_2_9b_it llama3_chinese_8b_instruct)
ALL_GPUS=(0 1 2 3 4 5 6 7 8 9)
numa_node(){ [ "$1" -le 4 ] && echo 0 || echo 1; }
NOW(){ date +%T; }

# --- generic phase-synchronized pool -------------------------------------------------
# run_pool <portbase> <job-spec>...
# Each job spec is a shell command string; placeholders $G (gpu) $P (port) $NODE are
# exported into its environment. Returns only when ALL jobs have finished (barrier).
run_pool(){
  local portbase=$1; shift
  local jobs=("$@"); local n=${#jobs[@]}
  [ "$n" -eq 0 ] && return 0
  local -a free=("${ALL_GPUS[@]}"); local -A pid2gpu=(); local jidx=0
  while [ $jidx -lt $n ] || [ ${#pid2gpu[@]} -gt 0 ]; do
    while [ $jidx -lt $n ] && [ ${#free[@]} -gt 0 ]; do
      local g=${free[0]}; free=("${free[@]:1}")
      local spec="${jobs[$jidx]}"; jidx=$((jidx+1))
      local p=$((portbase + jidx)); local node; node=$(numa_node "$g")
      G=$g P=$p NODE=$node bash -c "$spec" & pid2gpu[$!]=$g
    done
    [ ${#pid2gpu[@]} -gt 0 ] && wait -n
    for pid in "${!pid2gpu[@]}"; do
      kill -0 "$pid" 2>/dev/null || { free+=("${pid2gpu[$pid]}"); unset 'pid2gpu[$pid]'; }
    done
  done
}

# convenience per-model path exports used inside job specs
paths(){ # $1=TAG -> echoes a set of `export` lines to prepend to a job spec
  local TAG=$1
  echo "TAG=$TAG; CFG=configs/experiment.$TAG.yaml; export CONFIG=\$CFG CFG_FILE=\$CFG; \
ROOT=$ROOTBASE/$TAG; SUS=\$ROOT/backdoor_weight/negsentiment/badnet; \
TR=\$ROOT/outputs/training; PUR=\$ROOT/outputs/purified; LOGD=\$ROOT/outputs/logs; \
SIG=\$ROOT/outputs/signature/signature.pkl; NEG=configs/negsentiment/$TAG/negsenti_badnet_lora.yaml; \
mkdir -p \$LOGD;"
}

echo "=== SWEEP v2 START $(NOW) ==="

# ---- Stage A: 6 step0 suspicious adapters --------------------------------------------
echo "[A $(NOW)] step0 x ${#MODELS[@]}"
A=()
for TAG in "${MODELS[@]}"; do
  A+=("$(paths "$TAG") [ -f \$SUS/adapter_model.safetensors ] && exit 0; \
env CUDA_VISIBLE_DEVICES=\$G OMP_NUM_THREADS=6 \
$DIST --nproc_per_node=1 --master_port=\$P backdoor_train.py \$NEG > \$LOGD/sw_step0.log 2>&1")
done
run_pool 29500 "${A[@]}"
for TAG in "${MODELS[@]}"; do
  [ -f "$ROOTBASE/$TAG/backdoor_weight/negsentiment/badnet/adapter_model.safetensors" ] \
    || { echo "[A] FATAL: step0 missing for $TAG (see sw_step0.log)"; }
done

# ---- Stage B: data + training-config gen (CPU) ---------------------------------------
echo "[B $(NOW)] step1/step2 x ${#MODELS[@]}"
bpids=()
for TAG in "${MODELS[@]}"; do
  CFG="configs/experiment.$TAG.yaml"; LOGD="$ROOTBASE/$TAG/outputs/logs"; mkdir -p "$LOGD"
  ( python step1_build_data.py        --config "$CFG" > "$LOGD/sw_step1.log" 2>&1
    python step2_generate_training.py --config "$CFG" > "$LOGD/sw_step2.log" 2>&1 ) &
  bpids+=($!)
done
wait "${bpids[@]}"

# ---- Stage C: 72 variant adapters (6 models x 12) ------------------------------------
echo "[C $(NOW)] 72 variant adapters across 10 GPUs"
C=()
for TAG in "${MODELS[@]}"; do
  TR="$ROOTBASE/$TAG/outputs/training"; LOGD="$ROOTBASE/$TAG/outputs/logs"
  for i in 0 1 2 3 4 5; do
    for spec in "backdoor_train.py:variant_${i}_bd" "finetune_train.py:variant_${i}_clean"; do
      entry="${spec%%:*}"; name="${spec##*:}"
      C+=("[ -f $TR/$name/adapter_model.safetensors ] && exit 0; \
env CUDA_VISIBLE_DEVICES=\$G OMP_NUM_THREADS=4 \
$DIST --nproc_per_node=1 --master_port=\$P $entry $TR/configs/$name.yaml > $LOGD/sw_$name.log 2>&1")
    done
  done
done
run_pool 30000 "${C[@]}"

# ---- Stage D: 6 signatures (GPU-vectorized scoring -> ~seconds each) ------------------
echo "[D $(NOW)] signatures"
D=()
for TAG in "${MODELS[@]}"; do
  D+=("$(paths "$TAG") [ -f \$SIG ] && exit 0; \
env CUDA_VISIBLE_DEVICES=\$G python step3_extract_signature.py --config \$CFG > \$LOGD/sw_step3.log 2>&1")
done
run_pool 30200 "${D[@]}"

# ---- Stage E: suppression + baselines ------------------------------------------------
# E-pre (CPU, fast): ours-suppress (step4_purify) + B1 random-prune, all models in parallel.
echo "[E $(NOW)] purify + random-prune (CPU)"
epids=()
for TAG in "${MODELS[@]}"; do
  eval "$(paths "$TAG")"
  ( [ -f "$PUR/suppressed_adapter/adapter_model.safetensors" ]        || CUDA_VISIBLE_DEVICES=0 bash scripts/step4_purify.sh       > "$LOGD/sw_step4.log" 2>&1
    [ -f "$PUR/random_suppressed_adapter/adapter_model.safetensors" ] || CUDA_VISIBLE_DEVICES=0 bash scripts/step4_random_prune.sh > "$LOGD/sw_b1.log"   2>&1 ) &
  epids+=($!)
done
wait "${epids[@]}"
# E1 (GPU): our finetune (needs suppressed) + B2 pure-finetune + B3 wanda-prune.
echo "[E1 $(NOW)] finetune + pure-finetune + wanda-prune"
E1=()
for TAG in "${MODELS[@]}"; do
  PRE=$(paths "$TAG")
  E1+=("$PRE [ -f \$PUR/finetuned/adapter_model.safetensors ] && exit 0; \
env CUDA_VISIBLE_DEVICES=\$G OMP_NUM_THREADS=4 bash scripts/step4b_finetune.sh > \$LOGD/sw_step4b.log 2>&1")
  E1+=("$PRE [ -f \$PUR/pure_finetuned/adapter_model.safetensors ] && exit 0; \
env CUDA_VISIBLE_DEVICES=\$G OMP_NUM_THREADS=4 bash scripts/step4_pure_finetune.sh > \$LOGD/sw_b2.log 2>&1")
  E1+=("$PRE [ -f \$PUR/wanda_pruned/config.json ] && exit 0; \
env CUDA_VISIBLE_DEVICES=\$G OMP_NUM_THREADS=4 bash scripts/step4_wanda_prune.sh > \$LOGD/sw_b3.log 2>&1")
done
run_pool 30400 "${E1[@]}"
# E2 (GPU): B3b fine-pruning (needs wanda_pruned).
echo "[E2 $(NOW)] wanda-finetune"
E2=()
for TAG in "${MODELS[@]}"; do
  PRE=$(paths "$TAG")
  E2+=("$PRE [ ! -f \$PUR/wanda_pruned/config.json ] && exit 0; \
[ -f \$PUR/wanda_finetuned/adapter_model.safetensors ] && exit 0; \
env CUDA_VISIBLE_DEVICES=\$G OMP_NUM_THREADS=4 bash scripts/step4b_wanda_finetune.sh > \$LOGD/sw_b3b.log 2>&1")
done
run_pool 30600 "${E2[@]}"

# ---- Stage F: 42 evals (6 models x 7 tags), float32, across 10 GPUs ------------------
echo "[F $(NOW)] evals"
F=()
for TAG in "${MODELS[@]}"; do
  eval "$(paths "$TAG")"
  EVAL(){ echo "env CUDA_VISIBLE_DEVICES=\$G python step5_evaluate.py --config $CFG $* --eval_type both"; }
  F+=("$(EVAL --tag no_defense               --adapter $SUS)                             > $LOGD/sw_ev_nodef.log 2>&1")
  F+=("$(EVAL --tag after_suppression        --adapter $PUR/suppressed_adapter)          > $LOGD/sw_ev_supp.log  2>&1")
  F+=("$(EVAL --tag after_finetune           --adapter $PUR/finetuned)                   > $LOGD/sw_ev_ft.log    2>&1")
  F+=("$(EVAL --tag after_random_suppression --adapter $PUR/random_suppressed_adapter)   > $LOGD/sw_ev_rand.log  2>&1")
  F+=("$(EVAL --tag after_pure_finetune      --adapter $PUR/pure_finetuned)              > $LOGD/sw_ev_pure.log  2>&1")
  F+=("[ -f $PUR/wanda_pruned/config.json ] || exit 0; $(EVAL --tag after_wanda_pruning --base_model_override $PUR/wanda_pruned) > $LOGD/sw_ev_wanda.log 2>&1")
  F+=("[ -f $PUR/wanda_finetuned/adapter_model.safetensors ] || exit 0; $(EVAL --tag after_fine_pruning --base_model_override $PUR/wanda_pruned --adapter $PUR/wanda_finetuned) > $LOGD/sw_ev_fp.log 2>&1")
done
run_pool 30800 "${F[@]}"

echo "=== SWEEP v2 DONE $(NOW) ==="
