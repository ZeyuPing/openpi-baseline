# Challenge Weighted Post-Training

This document describes the experimental weighted post-training path. The official baseline path is unchanged:

```bash
bash train.sh pi05_insert-mouse-battery
```

The weighted path is additive. It uses new `pi05w_*` configs and `train_weighted.sh`, so baseline runs and weighted runs remain easy to compare.

## Method Summary

The current weighted post-training method keeps the official `pi05_*` baseline intact for comparison and adds task-specific `pi05w_*` configs for experiments. Each weighted run starts from a merged LeRobot root that combines the task's `expert-data`, `success-and-hil-data`, and `failure-data` sources.

For each merged root, build a frame-level/chunk-start weight index aligned to that root's `episode_index` and `frame_index`. The index keeps only chunks that are valid for actor imitation: it filters out chunks crossing `observation.commander_state` boundaries, chunks in drop modes (`restore`, `align`, `pre_teleop`), discontinuous action chunks, and short tails that cannot provide the configured action horizon. Any frame missing from the index receives zero actor weight during training.

Current actor imitation weights are:

| Source segment | Actor weight |
| --- | ---: |
| `expert-data` | 1.0 |
| `success-and-hil-data` teleop | 2.0 |
| `success-and-hil-data` inference success | 0.7 |
| `failure-data` | 0.0 |

Weighted training uses the same conversion sequence as the baseline: repack, data transforms, `Normalize`, then model transforms. The only intended difference is the sample weighting applied to actor imitation.

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

## Prepare Weighted Roots

Weighted `pi05w_*` configs do not point at the expert-only dataset. They expect cluster-generated merged LeRobot roots under:

```text
$CHALLENGE_ROOT/weighted-datasets/<task>-weighted-hil
```

For each `pi05w_*_hil` config, create one weighted root on the cluster from all three task sources before training:

```bash
uv run python scripts/merge_lerobot.py \
  --src_paths \
    "$DATASET_ROOT/insert-mouse-battery/expert-data" \
    "$DATASET_ROOT/insert-mouse-battery/success-and-hil-data" \
    "$DATASET_ROOT/insert-mouse-battery/failure-data" \
  --tgt_path "$CHALLENGE_ROOT/weighted-datasets/insert-mouse-battery-weighted-hil" \
  --repo_id "insert-mouse-battery/weighted-hil"
```

Repeat with matching task names for `seal-water-bottle-cap` and `tower-of-hanoi-game`.

Do not train a weighted config against `$DATASET_ROOT/<task>/expert-data`; that silently turns the run into expert-only post-training. The merged root and `repo_id` should match the `pi05w_*` config for that task.

## Build Weight Indexes

Build one index per weighted task root on the cluster:

```bash
uv run python scripts/build_challenge_index.py \
  --task-root "$CHALLENGE_ROOT/weighted-datasets/insert-mouse-battery-weighted-hil" \
  --output "$CHALLENGE_ROOT/indexes/insert-mouse-battery-weighted-hil.parquet"
```

Repeat for:

```bash
uv run python scripts/build_challenge_index.py \
  --task-root "$CHALLENGE_ROOT/weighted-datasets/seal-water-bottle-cap-weighted-hil" \
  --output "$CHALLENGE_ROOT/indexes/seal-water-bottle-cap-weighted-hil.parquet"

uv run python scripts/build_challenge_index.py \
  --task-root "$CHALLENGE_ROOT/weighted-datasets/tower-of-hanoi-game-weighted-hil" \
  --output "$CHALLENGE_ROOT/indexes/tower-of-hanoi-game-weighted-hil.parquet"
```

The index path must describe the same merged root that the weighted config reads. For example, `pi05w_insert-mouse-battery_hil` should use the merged root `$CHALLENGE_ROOT/weighted-datasets/insert-mouse-battery-weighted-hil` and index `$CHALLENGE_ROOT/indexes/insert-mouse-battery-weighted-hil.parquet`.

Before running `train_weighted.sh`, update the `/Your/path/to/...` placeholders for the matching `pi05w_*_hil` entry in `src/openpi/training/config.py`, or pass equivalent CLI overrides for the dataset root and `sample_weight_index_path`. The weighted training workflow is:

1. Merge `expert-data`, `success-and-hil-data`, and `failure-data` into `$CHALLENGE_ROOT/weighted-datasets/<task>-weighted-hil`.
2. Build the sample-weight index from that merged root.
3. Point the weighted config or CLI overrides at that same merged root and index.

The index builder follows the same frame/action convention as the baseline and rejects chunks that cross `observation.commander_state` boundaries or contain large action discontinuities.
Frames absent from the index receive zero sample weight during weighted training, so omitted entries are treated as filtered/rejected samples. Empty indexes are errors; if index construction produces no usable rows, check the filters, action horizon, and merged-root provenance before training.

## Conversion Consistency

Weighted training must use the same conversion sequence as the baseline:

1. `repack_transforms`
2. `data_transforms`
3. `Normalize`
4. `model_transforms`

This keeps YAM joint flips, gripper conversion, delta-action handling, image resizing, prompt tokenization, and normalization aligned with the official baseline.

## Iteration Guide

Start with one task before broadening the sweep, preferably `seal-water-bottle-cap` or `tower-of-hanoi-game`. Track baseline and weighted runs with the same seed/config wherever possible so changes are attributable to the weighting method rather than run setup.

During weighted runs, inspect `weighted_loss`, `sample_weight_sum`, `nonzero_sample_weight_count`, and `raw_sample_weight_mean`. If `nonzero_sample_weight_count` is low, inspect the index summary and filtering thresholds before spending more cluster time on longer training.

Suggested iteration knobs:

| Knob | What to vary |
| --- | --- |
| Source weights | Adjust expert, HIL teleop, inference-success, and failure actor weights. |
| Max action jump | Loosen or tighten the discontinuity filter. |
| Action horizon | Match chunk length to the task and policy-serving horizon. |
| Merged source composition | Include or exclude specific source subsets for a task. |
| Contact segment weighting | Upweight task-specific contact or manipulation phases once identified. |

Keep new method families under new config prefixes, such as `pi05awr_*`, so comparisons remain clean. A future stage should train a value/progress model and replace static source weights with clipped advantage weights, but keep that separate from `pi05w_*` results.

## Verification Matrix

Run these checks from the repository root unless a command says otherwise.

| Check | Command | Expected result |
| --- | --- | --- |
| Static tests | `uv run pytest src/openpi/training/challenge_weighting_test.py src/openpi/training/weighted_data_loader_test.py -q` | Weighting and weighted-loader tests pass. |
| Static compile | `uv run python -m py_compile scripts/build_challenge_index.py scripts/train_weighted.py` | Both weighted entry points compile. |
| Official baseline | `bash train.sh pi05_insert-mouse-battery` | Baseline still launches against the official `pi05_*` config. |
| Experimental weighted | `bash train_weighted.sh pi05w_insert-mouse-battery_hil` | Weighted run launches against the merged weighted root and matching index. |
| Policy server | `uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05w_insert-mouse-battery_hil --policy.dir=checkpoints/pi05w_insert-mouse-battery_hil/pi05w_insert-mouse-battery_hil/79999` | Server starts and loads the weighted checkpoint for an 80,000-step run. |
| Simulation compare | `cd ../policy_deployment && python sim/check_in_sim.py --mode compare --bundle sim/assets/example_slim.pkl --host 127.0.0.1 --port 8000 --prompt "Insert the battery to the mouse." --action-horizon 50 --output out/weighted_compare.mp4` | Sim client can compare the served weighted policy and write an output video. |

For cluster runs, complete the weighted-root merge and index build before the experimental weighted command. For local/laptop sanity work, run only static checks; do not fetch or create the challenge dataset locally.
