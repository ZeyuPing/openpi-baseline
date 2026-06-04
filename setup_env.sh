#!/bin/bash
export OPENPI_DATA_HOME="/root/autodl-tmp/challenge/openpi_data"
export HF_LEROBOT_HOME="/root/autodl-tmp/challenge/hf_lerobot"
export HF_HOME="/root/autodl-tmp/challenge/hf_home"
export DATASET_ROOT=$HF_LEROBOT_HOME
export CHALLENGE_ROOT="/root/autodl-tmp/challenge"

# Disable host memory registration to avoid CUDA illegal memory access errors with PyTorch DataLoader
# Removed invalid XLA flag to fix parsing error
unset XLA_FLAGS

# AutoDL-style single-node images can report CUDA illegal memory access inside NCCL P2P/IB collectives.
# Keep the conservative transport defaults for weighted JAX training; override to 0 only after a stable run.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

export LD_LIBRARY_PATH="/root/miniconda3/lib:${LD_LIBRARY_PATH:-}"

echo "Environment variables set:"
echo "OPENPI_DATA_HOME: $OPENPI_DATA_HOME"
echo "HF_LEROBOT_HOME: $HF_LEROBOT_HOME"
echo "HF_HOME: $HF_HOME"
echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY: set"
else
  echo "WANDB_API_KEY: not set; run wandb login or export it outside this script if W&B sync is required"
fi
