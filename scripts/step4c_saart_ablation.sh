#!/bin/bash
# Step 4c-ablation (OPT-IN, NOT in run_all): train the SAART-P1 ablation matrix.
# Each variant = base `saart:` block + per-variant overrides (defined under `saart_ablations:`
# in the active experiment.yaml), continuing from θ_sus on finetune_clean.
# Configs (from step2_generate_training.py): outputs/training/configs/saart_p1_ablation_<name>.yaml
# Output: outputs/purified/saart_p1/<name>/  (evaluated by step5 as tag=after_saart_p1_<name>).
#
# The main after_saart_p1 run (scripts/step4c_saart.sh) is the +pool/prompt_end/with_null
# endpoint of the matrix, so the ablations only cover the non-default endpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_DIR/base_select_gpu.sh"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/scripts/_load_cfg.sh"

CONFIGS_DIR="${TRAINING_DIR}/configs"
mkdir -p "$LOG_DIR"

if [ ! -f "$SUSPICIOUS_ADAPTER/adapter_model.safetensors" ]; then
    echo "ERROR: suspicious adapter not found at $SUSPICIOUS_ADAPTER" >&2
    echo "       Run scripts/step0_badnet_negsenti.sh first." >&2
    exit 1
fi

# Read ablation names from the active config (CFG_FILE, default configs/experiment.yaml).
ACTIVE_CFG="${CFG_FILE:-configs/experiment.yaml}"
mapfile -t NAMES < <(python - "$ACTIVE_CFG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for a in (cfg.get("saart_ablations") or []):
    print(a["name"])
PY
)

if [ "${#NAMES[@]}" -eq 0 ]; then
    echo "[step4c-ablation] No saart_ablations defined in $ACTIVE_CFG — nothing to do."
    exit 0
fi

echo "=============================================================="
echo "[step4c-ablation] GPU      : $CUDA_VISIBLE_DEVICES"
echo "[step4c-ablation] Ablations: ${NAMES[*]}"
echo "[step4c-ablation] Configs  : $CONFIGS_DIR/saart_p1_ablation_<name>.yaml"
echo "[step4c-ablation] Output   : ${PURIFIED_DIR}/saart_p1/<name>"
echo "=============================================================="

for name in "${NAMES[@]}"; do
    CONFIG="$CONFIGS_DIR/saart_p1_ablation_${name}.yaml"
    OUT_DIR="${PURIFIED_DIR}/saart_p1/${name}"
    LOG_FILE="$LOG_DIR/step4c_ablation_${name}.log"

    if [ ! -f "$CONFIG" ]; then
        echo "[${name}] SKIP: config missing ($CONFIG). Run step2_generate_training.py first."
        continue
    fi
    if [ -f "$OUT_DIR/adapter_model.safetensors" ]; then
        echo "[${name}] SKIP: adapter already trained ($OUT_DIR)"
        continue
    fi

    MASTER_PORT=$(( (RANDOM % 45000) + 20000 ))
    echo "--------------------------------------------------------------"
    echo "[${name}] START   port=$MASTER_PORT log=$LOG_FILE"
    echo "--------------------------------------------------------------"

    python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port="$MASTER_PORT" \
        saart_train.py "$CONFIG" \
        2>&1 | tee "$LOG_FILE"

    echo "[${name}] DONE -> $OUT_DIR"
done

echo "=============================================================="
echo "[step4c-ablation] All ablations processed. Evaluate with: bash scripts/step5_evaluate.sh"
echo "=============================================================="
