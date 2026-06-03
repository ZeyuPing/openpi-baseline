#!/usr/bin/env bash
set -euo pipefail

source setup_env.sh

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export OPENPI_DISABLE_DONATE=${OPENPI_DISABLE_DONATE:-1}

if [ $# -lt 1 ]; then
  echo "Usage: bash train_weighted.sh <config-name>"
  exit 1
fi

CONFIG=$1
EXP_ID=$(date "+%Y%m%d-%H%M%S")
EXP_NAME="${CONFIG}"
FORCE_NORM_STATS=${FORCE_NORM_STATS:-0}

mkdir -p logs
LOG_FILE="logs/${EXP_NAME}_${EXP_ID}.log"

NORM_STATS_PATH=$(uv run python - "$CONFIG" <<'PY'
import sys

from openpi.training import config as _config

config = _config.get_config(sys.argv[1])
data_config = config.data.create(config.assets_dirs, config.model)
if data_config.repo_id is None:
    raise ValueError(f"Config {config.name!r} does not define a repo_id.")
print(config.assets_dirs / data_config.repo_id / "norm_stats.json")
PY
)

if [ "$FORCE_NORM_STATS" = "1" ] || [ ! -f "$NORM_STATS_PATH" ]; then
  echo "Computing normalization stats for $CONFIG -> $NORM_STATS_PATH"
  uv run scripts/compute_norm_stats.py --config-name "$CONFIG"
else
  echo "Skipping normalization stats; found $NORM_STATS_PATH"
  echo "Set FORCE_NORM_STATS=1 to recompute."
fi

EXTRA_ARGS=()
NUM_GPUS=1
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
fi

# Keep the baseline default explicit: fsdp_devices=1 gives data parallelism across all visible GPUs.
# Set OPENPI_AUTO_FSDP=1 only when per-GPU memory is insufficient and model sharding is required.
if [ "${OPENPI_AUTO_FSDP:-0}" = "1" ] && [ "$NUM_GPUS" -gt 1 ] && [[ ! "$*" =~ "--fsdp-devices" ]]; then
  EXTRA_ARGS+=(--fsdp-devices "$NUM_GPUS")
fi

# Changing global batch size changes the optimization run. Opt in after a stable baseline is confirmed.
if [ -n "${OPENPI_GLOBAL_BATCH_SIZE:-}" ] && [[ ! "$*" =~ "--batch-size" ]]; then
  EXTRA_ARGS+=(--batch-size "$OPENPI_GLOBAL_BATCH_SIZE")
fi

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  echo "[*] Training settings appended: ${EXTRA_ARGS[*]}"
else
  echo "[*] Using config training settings; visible_gpus=$NUM_GPUS fsdp_devices=1 unless overridden."
fi
echo "[*] OPENPI_DISABLE_DONATE=$OPENPI_DISABLE_DONATE"

uv run scripts/train_weighted.py "$CONFIG" --exp-name="$EXP_NAME" --overwrite "${EXTRA_ARGS[@]}" "${@:2}" 2>&1 | tee -a "$LOG_FILE"
