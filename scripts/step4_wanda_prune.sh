#!/bin/bash
# Step 4 (Wanda pruning baseline): apply unstructured Wanda pruning at 35% sparsity
# to the merged (base + suspicious-LoRA) model, using the embedded wanda/ third-party.
# Output: outputs/purified/wanda_pruned/ — a FULL model directory (not an adapter).
#
# Reference: Sun et al. "A Simple and Effective Pruning Approach for Large Language
#            Models" (Wanda), arXiv:2306.11695.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_DIR/base_select_gpu.sh"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/scripts/_load_cfg.sh"

SPARSITY_RATIO="0.35"          # match step4_purify.sh's LoRA suppress ratio
SPARSITY_TYPE="unstructured"
PRUNE_METHOD="wanda"
OUT_DIR="${PURIFIED_DIR}/wanda_pruned"
mkdir -p "$LOG_DIR" "$OUT_DIR"
LOG_FILE="$LOG_DIR/step4_wanda_prune.log"

# Prereq: only the suspicious adapter from step0.
if [ ! -f "$SUSPICIOUS_ADAPTER/adapter_model.safetensors" ]; then
    echo "ERROR: suspicious adapter not found at $SUSPICIOUS_ADAPTER" >&2
    echo "       Run scripts/step0_badnet_negsenti.sh first." >&2
    exit 1
fi

# SUSPICIOUS_ADAPTER from experiment.yaml is already absolute; don't re-prefix.
case "$SUSPICIOUS_ADAPTER" in
    /*) SUSPICIOUS_ADAPTER_ABS="$SUSPICIOUS_ADAPTER" ;;
    *)  SUSPICIOUS_ADAPTER_ABS="$PROJECT_DIR/$SUSPICIOUS_ADAPTER" ;;
esac
case "$OUT_DIR" in
    /*) OUT_DIR_ABS="$OUT_DIR" ;;
    *)  OUT_DIR_ABS="$PROJECT_DIR/$OUT_DIR" ;;
esac

echo "[step4-wanda] GPU             : $CUDA_VISIBLE_DEVICES"
echo "[step4-wanda] Base model      : $BASE_MODEL"
echo "[step4-wanda] LoRA            : $SUSPICIOUS_ADAPTER"
echo "[step4-wanda] Prune method    : $PRUNE_METHOD"
echo "[step4-wanda] Sparsity ratio  : $SPARSITY_RATIO ($SPARSITY_TYPE)"
echo "[step4-wanda] Save model dir  : $OUT_DIR"
echo "[step4-wanda] Log             : $LOG_FILE"

# Wanda's main.py expects to run from its own directory (its imports are relative).
case "$LOG_FILE" in
    /*) LOG_FILE_ABS="$LOG_FILE" ;;
    *)  LOG_FILE_ABS="$PROJECT_DIR/$LOG_FILE" ;;
esac
cd wanda
python main.py \
    --model "$BASE_MODEL" \
    --lora_path "$SUSPICIOUS_ADAPTER_ABS" \
    --prune_method "$PRUNE_METHOD" \
    --sparsity_ratio "$SPARSITY_RATIO" \
    --sparsity_type "$SPARSITY_TYPE" \
    --save "$OUT_DIR_ABS/wanda_results/" \
    --save_model "$OUT_DIR_ABS" \
    2>&1 | tee "$LOG_FILE_ABS"
cd "$PROJECT_DIR"

echo "[step4-wanda] Done. Pruned model saved to: $OUT_DIR_ABS"
