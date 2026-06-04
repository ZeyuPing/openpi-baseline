#!/bin/bash
export OPENPI_DATA_HOME="/root/autodl-tmp/challenge/openpi_data"
export HF_LEROBOT_HOME="/root/autodl-tmp/challenge/hf_lerobot"
export HF_HOME="/root/autodl-tmp/challenge/hf_home"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "Environment variables set:"
echo "OPENPI_DATA_HOME: $OPENPI_DATA_HOME"
echo "HF_LEROBOT_HOME: $HF_LEROBOT_HOME"
echo "HF_HOME: $HF_HOME"