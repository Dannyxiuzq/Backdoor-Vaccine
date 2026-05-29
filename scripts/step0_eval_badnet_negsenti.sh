#!/bin/bash
# Evaluate ASR of the step0 suspicious model (BadNets x Llama2-7B-Chat x negsenti).
# Expected baseline: ~59% ASR.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Auto-select a GPU with >= 40GB free memory (exports CUDA_VISIBLE_DEVICES)
source "$PROJECT_DIR/base_select_gpu.sh"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/scripts/_load_cfg.sh"

TASK="negsentiment"
TRIGGER="badnet"
TEST_FILE="data/test_data/poison/${TASK}/${TRIGGER}/backdoor200_${TASK}_${TRIGGER}.json"
SAVE_DIR="${EVAL_DIR}/${TASK}/${TRIGGER}"
mkdir -p "$LOG_DIR" "$SAVE_DIR"
LOG_FILE="$LOG_DIR/step0_eval_badnet_negsenti.log"

echo "[eval] GPU        : $CUDA_VISIBLE_DEVICES"
echo "[eval] Base model : $BASE_MODEL"
echo "[eval] Adapter    : $SUSPICIOUS_ADAPTER"
echo "[eval] Test file  : $TEST_FILE"
echo "[eval] Save dir   : $SAVE_DIR"
echo "[eval] Log        : $LOG_FILE"

python eval_asr.py \
    --model_path  "$BASE_MODEL" \
    --adapter_path "$SUSPICIOUS_ADAPTER" \
    --task        "$TASK" \
    --trigger     "$TRIGGER" \
    --test_file   "$TEST_FILE" \
    --save_dir    "$SAVE_DIR" \
    2>&1 | tee "$LOG_FILE"
