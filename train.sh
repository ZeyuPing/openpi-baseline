set -euo pipefail

source setup_env.sh

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

if [ $# -lt 1 ]; then
  echo "Usage: bash train.sh <config-name>"
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

uv run scripts/train.py "$CONFIG" --exp-name="$EXP_NAME" --overwrite 2>&1 | tee -a "$LOG_FILE"
