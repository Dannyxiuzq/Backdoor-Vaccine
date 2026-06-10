#!/bin/bash
# Compute-matched pure-finetune baseline (B2-long): finetune the SUSPICIOUS adapter on clean
# data for MORE epochs, so its total wall-clock / token budget ≈ SAART-P1's. This rules out
# the "SAART only wins because it uses more compute" critique.
#
# "先实测再设": this stage is DORMANT until you set `pure_finetune_long_epochs` in the active
# experiment.yaml (after measuring SAART vs pure-FT wall-clock) and re-run step2_generate_training.py.
# If finetune_pure_long.yaml is absent, this script exits 0 with a hint (so run_all stays clean).
# Output: outputs/purified/pure_finetuned_long/  (evaluated by step5 as tag=after_pure_finetune_long).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_DIR/base_select_gpu.sh"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/scripts/_load_cfg.sh"

FT_CONFIG="${TRAINING_DIR}/configs/finetune_pure_long.yaml"
OUTPUT_DIR="${PURIFIED_DIR}/pure_finetuned_long"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/step4_pure_finetune_long.log"

# Idempotent skip.
if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
    echo "[step4-pure-long] adapter already exists at $OUTPUT_DIR — skipping."
    exit 0
fi

# Dormant until the user opts in by setting pure_finetune_long_epochs.
if [ ! -f "$FT_CONFIG" ]; then
    echo "[step4-pure-long] SKIP — $FT_CONFIG not found."
    echo "                  Set 'pure_finetune_long_epochs' in the active experiment.yaml (after"
    echo "                  measuring SAART vs pure-FT wall-clock), then run step2_generate_training.py."
    exit 0
fi

if [ ! -f "$SUSPICIOUS_ADAPTER/adapter_model.safetensors" ]; then
    echo "ERROR: suspicious adapter not found at $SUSPICIOUS_ADAPTER" >&2
    echo "       Run scripts/step0_badnet_negsenti.sh first." >&2
    exit 1
fi

MASTER_PORT=$(( (RANDOM % 45000) + 20000 ))

echo "[step4-pure-long] GPU         : $CUDA_VISIBLE_DEVICES"
echo "[step4-pure-long] MASTER_PORT : $MASTER_PORT"
echo "[step4-pure-long] Config      : $FT_CONFIG"
echo "[step4-pure-long] From adapter: $SUSPICIOUS_ADAPTER"
echo "[step4-pure-long] Output      : $OUTPUT_DIR"
echo "[step4-pure-long] Log         : $LOG_FILE"

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="$MASTER_PORT" \
    finetune_train.py "$FT_CONFIG" \
    2>&1 | tee "$LOG_FILE"

echo "[step4-pure-long] Done. Compute-matched pure-finetuned adapter saved to: $OUTPUT_DIR"
