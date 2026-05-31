# Challenge Weighted Post-Training

This document describes the experimental weighted post-training path. The official baseline path is unchanged:

```bash
bash train.sh pi05_insert-mouse-battery
```

The weighted path is additive. It uses new `pi05w_*` configs and `train_weighted.sh`, so baseline runs and weighted runs remain easy to compare.

## Dataset Location

Do not download the challenge dataset on the laptop. Download and process it only on the training cluster.

Expected cluster environment:

```bash
export CHALLENGE_ROOT=/data/$USER/posttraining-rfm
export DATASET_ROOT=$CHALLENGE_ROOT/datasets/Posttraining-RFM-RSS2026/Challenge-phase1-dataset
export OPENPI_DATA_HOME=$CHALLENGE_ROOT/openpi_data
export HF_LEROBOT_HOME=$DATASET_ROOT
export HF_HOME=$CHALLENGE_ROOT/hf_home
```

## Build Weight Indexes

Build one index per task on the cluster:

```bash
uv run python scripts/build_challenge_index.py \
  --task-root "$DATASET_ROOT/insert-mouse-battery" \
  --output "$CHALLENGE_ROOT/indexes/insert-mouse-battery.parquet"
```

Repeat for:

```bash
uv run python scripts/build_challenge_index.py \
  --task-root "$DATASET_ROOT/seal-water-bottle-cap" \
  --output "$CHALLENGE_ROOT/indexes/seal-water-bottle-cap.parquet"

uv run python scripts/build_challenge_index.py \
  --task-root "$DATASET_ROOT/tower-of-hanoi-game" \
  --output "$CHALLENGE_ROOT/indexes/tower-of-hanoi-game.parquet"
```

The index builder scans `expert-data`, `success-and-hil-data`, and `failure-data`. It follows the same frame/action convention as the baseline and rejects chunks that cross `observation.commander_state` boundaries or contain large action discontinuities.

## Conversion Consistency

Weighted training must use the same conversion sequence as the baseline:

1. `repack_transforms`
2. `data_transforms`
3. normalization
4. `model_transforms`

This keeps YAM joint flips, gripper conversion, delta-action handling, image resizing, prompt tokenization, and normalization aligned with the official baseline.
