# shellcheck shell=bash
# Source-only helper: exports canonical paths read from configs/experiment.yaml.
# Every stage script sources this file (after cd'ing to PROJECT_DIR) so they
# all see the same model_tag-scoped layout under /mnt/data.
#
# Usage:
#   source "$PROJECT_DIR/scripts/_load_cfg.sh"
#
# Exports:
#   MODEL_TAG, BASE_MODEL, SUSPICIOUS_ADAPTER,
#   TRAINING_DIR, SIGNATURE_DIR, PURIFIED_DIR, EVAL_DIR, LOG_DIR

_CFG_FILE="${CFG_FILE:-configs/experiment.yaml}"

if [ ! -f "$_CFG_FILE" ]; then
    echo "ERROR: $_CFG_FILE not found (cwd=$(pwd))" >&2
    return 1 2>/dev/null || exit 1
fi

# One python invocation produces all variables in shell-eval form. This avoids
# spawning python 7 times per script (was a noticeable startup cost on cold runs).
eval "$(python - "$_CFG_FILE" <<'PY'
import sys, shlex, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
def q(v): return shlex.quote(str(v))
for k in ("model_tag","base_model","suspicious_adapter","training_dir",
         "signature_dir","purified_dir","eval_dir","log_dir"):
    print(f"export {k.upper()}={q(cfg[k])}")
PY
)"
