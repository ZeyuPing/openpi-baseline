#!/bin/bash
export OPENPI_DATA_HOME="/root/autodl-tmp/challenge/openpi_data"
export HF_LEROBOT_HOME="/root/autodl-tmp/challenge/hf_lerobot"
export HF_HOME="/root/autodl-tmp/challenge/hf_home"
export WANDB_API_KEY="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"
export DATASET_ROOT=$HF_LEROBOT_HOME
export CHALLENGE_ROOT="/root/autodl-tmp/challenge"

export LD_LIBRARY_PATH=/root/miniconda3/lib:$LD_LIBRARY_PATH

echo "Environment variables set:"
echo "OPENPI_DATA_HOME: $OPENPI_DATA_HOME"
echo "HF_LEROBOT_HOME: $HF_LEROBOT_HOME"
echo "HF_HOME: $HF_HOME"