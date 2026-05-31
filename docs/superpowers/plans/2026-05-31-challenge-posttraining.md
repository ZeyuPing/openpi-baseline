# Challenge Post-Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental offline post-training pipeline for the RSS 2026 challenge while preserving the official baseline path for clean comparisons.

**Architecture:** Keep the existing `pi05_*` configs, `train.sh`, and `scripts/train.py` unchanged. Add new `pi05w_*` configs, an index builder, a weighted data loader, and `scripts/train_weighted.py`; experiments run side by side with baseline checkpoints and logs.

**Tech Stack:** Python 3.11, JAX/Flax NNX, LeRobot v2.1, pandas/pyarrow, existing openpi transforms and training stack.

---

## Baseline Preservation Contract

- Official configs remain unchanged: `pi05_insert-mouse-battery`, `pi05_seal-water-bottle-cap`, `pi05_tower-of-hanoi-game`.
- Official runner remains unchanged: `bash train.sh pi05_insert-mouse-battery`.
- Experimental runner is additive: `bash train_weighted.sh pi05w_insert-mouse-battery_hil`.
- Experimental assets use new config prefixes, so `assets/pi05_*` and `checkpoints/pi05_*` stay comparable to `assets/pi05w_*` and `checkpoints/pi05w_*`.
- Dataset indexes, merged datasets, logs, and checkpoints are generated outside git.
- Dataset processing must use the same conversion sequence as the baseline: `repack_transforms`, then `data_transforms`, then normalization, then `model_transforms`.
- Documentation must be clear enough to run on a cluster without requiring context from this thread.
- Do not download the challenge dataset on the laptop. Dataset download, index building on real data, normalization-stat computation on full data, and training are cluster-only operations.

## File Structure

- Create `src/openpi/training/challenge_weighting.py`: commander-state segmentation, chunk validity, source/mode sample weights.
- Create `scripts/build_challenge_index.py`: scans challenge LeRobot leaves and writes `episode_index`, `frame_index`, `sample_weight` parquet indexes.
- Create `src/openpi/training/weighted_data_loader.py`: mirrors the baseline LeRobot loader, attaches frame-aligned weights after transforms, yields `(Observation, actions, sample_weight)`.
- Create `scripts/train_weighted.py`: copy of `scripts/train.py` with weighted loss aggregation only.
- Modify `src/openpi/training/config.py`: add backward-compatible weighted-training fields and append new `pi05w_*` configs.
- Create `train_weighted.sh`: mirrors `train.sh` but calls `scripts/train_weighted.py`.
- Create tests: `src/openpi/training/challenge_weighting_test.py`, `src/openpi/training/weighted_data_loader_test.py`.
- Create docs: `docs/challenge_weighted_posttraining.md`.

---

### Task 1: Weighting Utilities

**Files:**
- Create: `src/openpi/training/challenge_weighting.py`
- Test: `src/openpi/training/challenge_weighting_test.py`

- [ ] **Step 1: Write failing tests**

Create `src/openpi/training/challenge_weighting_test.py`:

```python
import numpy as np

from openpi.training import challenge_weighting


def test_segments_split_on_commander_state_change():
    states = ["inference", "inference", "teleop", "teleop", "restore", "inference"]
    assert challenge_weighting.segment_commander_states(states) == [
        challenge_weighting.CommanderSegment("inference", 0, 2),
        challenge_weighting.CommanderSegment("teleop", 2, 4),
        challenge_weighting.CommanderSegment("restore", 4, 5),
        challenge_weighting.CommanderSegment("inference", 5, 6),
    ]


def test_chunk_rejected_when_it_crosses_mode_boundary():
    states = ["inference", "inference", "teleop", "teleop"]
    assert challenge_weighting.chunk_is_mode_pure(states, start=0, horizon=2)
    assert not challenge_weighting.chunk_is_mode_pure(states, start=1, horizon=2)


def test_sample_weight_policy():
    assert challenge_weighting.sample_weight("expert-data", "inference", success=True) == 1.0
    assert challenge_weighting.sample_weight("success-and-hil-data", "teleop", success=True) == 2.0
    assert challenge_weighting.sample_weight("success-and-hil-data", "inference", success=True) == 0.7
    assert challenge_weighting.sample_weight("failure-data", "inference", success=False) == 0.0
    assert challenge_weighting.sample_weight("success-and-hil-data", "restore", success=True) == 0.0


def test_large_action_jump_rejected():
    actions = np.zeros((4, 14), dtype=np.float32)
    actions[2, 3] = 0.25
    assert not challenge_weighting.chunk_has_smooth_actions(actions, start=1, horizon=2, max_abs_step=0.2)
    assert challenge_weighting.chunk_has_smooth_actions(actions, start=0, horizon=1, max_abs_step=0.2)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
uv run pytest src/openpi/training/challenge_weighting_test.py -q
```

Expected: fails because `openpi.training.challenge_weighting` is missing.

- [ ] **Step 3: Implement utilities**

Create `src/openpi/training/challenge_weighting.py`:

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommanderSegment:
    mode: str
    start: int
    end: int


DROP_MODES = frozenset({"restore", "align", "pre_teleop"})


def segment_commander_states(states: Sequence[str]) -> list[CommanderSegment]:
    if not states:
        return []
    segments: list[CommanderSegment] = []
    start = 0
    current = str(states[0])
    for idx, state in enumerate(states[1:], start=1):
        state = str(state)
        if state != current:
            segments.append(CommanderSegment(current, start, idx))
            start = idx
            current = state
    segments.append(CommanderSegment(current, start, len(states)))
    return segments


def chunk_is_mode_pure(states: Sequence[str], *, start: int, horizon: int) -> bool:
    stop = min(start + horizon, len(states))
    if start < 0 or start >= stop:
        return False
    mode = str(states[start])
    if mode in DROP_MODES:
        return False
    return all(str(state) == mode for state in states[start:stop])


def chunk_has_smooth_actions(actions: np.ndarray, *, start: int, horizon: int, max_abs_step: float = 0.2) -> bool:
    stop = min(start + horizon, len(actions))
    if stop - start <= 1:
        return True
    max_jump = float(np.max(np.abs(np.diff(np.asarray(actions[start:stop]), axis=0))))
    return max_jump <= max_abs_step


def sample_weight(source_name: str, commander_state: str, *, success: bool) -> float:
    commander_state = str(commander_state)
    if commander_state in DROP_MODES:
        return 0.0
    if source_name == "expert-data":
        return 1.0
    if source_name == "success-and-hil-data" and commander_state == "teleop":
        return 2.0
    if source_name == "success-and-hil-data" and commander_state == "inference" and success:
        return 0.7
    if source_name == "failure-data":
        return 0.0
    return 0.0
```

- [ ] **Step 4: Run the test and confirm it passes**

Run:

```bash
uv run pytest src/openpi/training/challenge_weighting_test.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/openpi/training/challenge_weighting.py src/openpi/training/challenge_weighting_test.py
git commit -m "feat: add challenge sample weighting utilities"
```

---

### Task 2: Challenge Index Builder

**Files:**
- Create: `scripts/build_challenge_index.py`
- Create: `docs/challenge_weighted_posttraining.md`

- [ ] **Step 1: Implement index builder**

Create `scripts/build_challenge_index.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from openpi.training import challenge_weighting

SOURCES = ("expert-data", "success-and-hil-data", "failure-data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action-horizon", default=50, type=int)
    parser.add_argument("--max-action-jump", default=0.2, type=float)
    return parser.parse_args()


def _episode_success(source_name: str) -> bool:
    return source_name in {"expert-data", "success-and-hil-data"}


def iter_parquet_files(task_root: Path):
    for source_name in SOURCES:
        data_root = task_root / source_name / "data"
        if not data_root.exists():
            continue
        for parquet_path in sorted(data_root.rglob("episode_*.parquet")):
            yield source_name, parquet_path


def _scalar(value) -> int:
    value = np.asarray(value)
    return int(value.reshape(-1)[0])


def build_rows(task_root: Path, *, action_horizon: int, max_action_jump: float) -> list[dict]:
    rows: list[dict] = []
    for source_name, parquet_path in iter_parquet_files(task_root):
        df = pd.read_parquet(parquet_path)
        if "observation.commander_state" not in df.columns:
            continue
        states = [str(x) for x in df["observation.commander_state"].tolist()]
        actions = np.stack(df["action"].to_numpy())
        for row_idx in range(len(df)):
            if not challenge_weighting.chunk_is_mode_pure(states, start=row_idx, horizon=action_horizon):
                continue
            if not challenge_weighting.chunk_has_smooth_actions(
                actions, start=row_idx, horizon=action_horizon, max_abs_step=max_action_jump
            ):
                continue
            mode = states[row_idx]
            rows.append(
                {
                    "source_name": source_name,
                    "episode_index": _scalar(df["episode_index"].iloc[row_idx]),
                    "frame_index": _scalar(df["frame_index"].iloc[row_idx]),
                    "commander_state": mode,
                    "sample_weight": challenge_weighting.sample_weight(
                        source_name, mode, success=_episode_success(source_name)
                    ),
                    "parquet_path": str(parquet_path),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    rows = build_rows(args.task_root, action_horizon=args.action_horizon, max_action_jump=args.max_action_jump)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    summary = {
        "task_root": str(args.task_root),
        "output": str(args.output),
        "rows": len(rows),
        "total_weight": float(sum(row["sample_weight"] for row in rows)),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with absent dataset**

Run:

```bash
uv run python scripts/build_challenge_index.py --task-root /tmp/missing --output /tmp/challenge-index.parquet
```

Expected: `/tmp/challenge-index.parquet` exists and `/tmp/challenge-index.json` contains `"rows": 0`.

- [ ] **Step 3: Add usage docs**

Create `docs/challenge_weighted_posttraining.md`:

````markdown
# Challenge Weighted Post-Training

The official baseline remains unchanged:

```bash
bash train.sh pi05_insert-mouse-battery
```

Build an experimental sample index on the cluster:

```bash
uv run python scripts/build_challenge_index.py \
  --task-root "$DATASET_ROOT/insert-mouse-battery" \
  --output "$CHALLENGE_ROOT/indexes/insert-mouse-battery.parquet"
```

Weighted experiments use `pi05w_*` configs and `train_weighted.sh`.
````

- [ ] **Step 4: Commit**

```bash
git add scripts/build_challenge_index.py docs/challenge_weighted_posttraining.md
git commit -m "feat: add challenge index builder"
```

---

### Task 3: Weighted Data Loader

**Files:**
- Create: `src/openpi/training/weighted_data_loader.py`
- Test: `src/openpi/training/weighted_data_loader_test.py`

- [ ] **Step 1: Write failing tests**

Create `src/openpi/training/weighted_data_loader_test.py`:

```python
import pandas as pd

from openpi.training import weighted_data_loader


def test_index_lookup_returns_default_for_missing_key(tmp_path):
    index_path = tmp_path / "index.parquet"
    pd.DataFrame([{"episode_index": 3, "frame_index": 7, "sample_weight": 2.0}]).to_parquet(index_path, index=False)
    lookup = weighted_data_loader.WeightLookup.from_parquet(index_path)
    assert lookup.get(episode_index=3, frame_index=7) == 2.0
    assert lookup.get(episode_index=3, frame_index=8) == 1.0


def test_weighted_transformed_dataset_attaches_weight(tmp_path):
    index_path = tmp_path / "index.parquet"
    pd.DataFrame([{"episode_index": 3, "frame_index": 7, "sample_weight": 2.0}]).to_parquet(index_path, index=False)
    lookup = weighted_data_loader.WeightLookup.from_parquet(index_path)
    raw_dataset = [{"episode_index": 3, "frame_index": 7, "actions": [1.0]}]
    dataset = weighted_data_loader.WeightedTransformedDataset(
        raw_dataset,
        transforms=[lambda item: {"actions": item["actions"]}],
        lookup=lookup,
    )
    assert dataset[0]["actions"] == [1.0]
    assert float(dataset[0]["sample_weight"]) == 2.0
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
uv run pytest src/openpi/training/weighted_data_loader_test.py -q
```

Expected: fails because `openpi.training.weighted_data_loader` is missing.

- [ ] **Step 3: Implement weighted loader primitives**

Create `src/openpi/training/weighted_data_loader.py`:

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pandas as pd

import openpi.models.model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


@dataclass(frozen=True)
class WeightLookup:
    weights: dict[tuple[int, int], float]
    default_weight: float = 1.0

    @classmethod
    def from_parquet(cls, path: str | Path, *, default_weight: float = 1.0) -> "WeightLookup":
        df = pd.read_parquet(path)
        weights = {
            (int(row.episode_index), int(row.frame_index)): float(row.sample_weight)
            for row in df.itertuples(index=False)
        }
        return cls(weights=weights, default_weight=default_weight)

    def get(self, *, episode_index: int, frame_index: int) -> float:
        return self.weights.get((int(episode_index), int(frame_index)), self.default_weight)


def _scalar_int(value) -> int:
    return int(np.asarray(value).reshape(-1)[0])


class WeightedTransformedDataset:
    def __init__(self, dataset, transforms: Sequence[_transforms.DataTransformFn], lookup: WeightLookup):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._lookup = lookup

    def __getitem__(self, index):
        raw = self._dataset[index]
        weight = self._lookup.get(
            episode_index=_scalar_int(raw["episode_index"]),
            frame_index=_scalar_int(raw["frame_index"]),
        )
        transformed = self._transform(raw)
        transformed["sample_weight"] = np.asarray(weight, dtype=np.float32)
        return transformed

    def __len__(self) -> int:
        return len(self._dataset)


class WeightedDataLoaderImpl:
    def __init__(self, data_config: _config.DataConfig, data_loader: _data_loader.TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"], batch["sample_weight"]
```

- [ ] **Step 4: Run the test and confirm it passes**

Run:

```bash
uv run pytest src/openpi/training/weighted_data_loader_test.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/openpi/training/weighted_data_loader.py src/openpi/training/weighted_data_loader_test.py
git commit -m "feat: add weighted challenge data loader primitives"
```

---

### Task 4: Weighted Loader Factory

**Files:**
- Modify: `src/openpi/training/weighted_data_loader.py`

- [ ] **Step 1: Add factory**

Append to `src/openpi/training/weighted_data_loader.py`:

```python
def create_weighted_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: str = "jax",
):
    if config.sample_weight_index_path is None:
        raise ValueError("sample_weight_index_path is required for weighted training")
    if framework != "jax":
        raise NotImplementedError("Weighted challenge training is currently implemented for JAX training only")

    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    norm_stats = {} if skip_norm_stats else data_config.norm_stats
    if norm_stats is None:
        raise ValueError("Normalization stats are required. Run scripts/compute_norm_stats.py first.")

    weighted_dataset = WeightedTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        WeightLookup.from_parquet(config.sample_weight_index_path),
    )
    local_batch_size = config.batch_size // jax.process_count()
    torch_loader = _data_loader.TorchDataLoader(
        weighted_dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework=framework,
    )
    return WeightedDataLoaderImpl(data_config, torch_loader)
```

- [ ] **Step 2: Compile**

Run:

```bash
uv run python -m py_compile src/openpi/training/weighted_data_loader.py
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add src/openpi/training/weighted_data_loader.py
git commit -m "feat: add weighted challenge data loader factory"
```

---

### Task 5: Weighted Training Entrypoint

**Files:**
- Modify: `src/openpi/training/config.py`
- Create: `scripts/train_weighted.py`

- [ ] **Step 1: Add backward-compatible config fields**

In `TrainConfig`, add these fields immediately after `data`:

```python
    # Optional parquet index used only by scripts/train_weighted.py.
    sample_weight_index_path: str | None = None
    # Samples with weights <= this threshold are ignored by weighted training.
    min_sample_weight: float = 0.0
```

Baseline behavior is unchanged because `scripts/train.py` never reads these fields.

- [ ] **Step 2: Create weighted training script**

Copy `scripts/train.py` to `scripts/train_weighted.py`, then make these exact edits:

```python
import openpi.training.weighted_data_loader as _weighted_data_loader
```

Add near `train_step`:

```python
def _weighted_mean_loss(chunked_loss, sample_weight, *, min_sample_weight: float):
    per_sample_loss = jnp.mean(chunked_loss, axis=-1)
    sample_weight = jnp.asarray(sample_weight, dtype=per_sample_loss.dtype)
    sample_weight = jnp.where(sample_weight > min_sample_weight, sample_weight, 0.0)
    denom = jnp.maximum(jnp.sum(sample_weight), 1.0)
    return jnp.sum(per_sample_loss * sample_weight) / denom
```

Change the weighted `loss_fn`:

```python
def loss_fn(model, rng, observation, actions, sample_weight):
    chunked_loss = model.compute_loss(rng, observation, actions, train=True)
    return _weighted_mean_loss(
        chunked_loss,
        sample_weight,
        min_sample_weight=config.min_sample_weight,
    )
```

Change batch unpacking:

```python
observation, actions, sample_weight = batch
```

Change gradient call:

```python
loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
    model, train_rng, observation, actions, sample_weight
)
```

Change data loader creation in `main`:

```python
if config.sample_weight_index_path is None:
    raise ValueError("Weighted training requires sample_weight_index_path.")

data_loader = _weighted_data_loader.create_weighted_data_loader(
    config,
    sharding=data_sharding,
    shuffle=True,
)
```

- [ ] **Step 3: Compile**

Run:

```bash
uv run python -m py_compile scripts/train_weighted.py src/openpi/training/config.py
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_weighted.py src/openpi/training/config.py
git commit -m "feat: add weighted challenge training entrypoint"
```

---

### Task 6: Add Experimental Configs and Launcher

**Files:**
- Modify: `src/openpi/training/config.py`
- Create: `train_weighted.sh`

- [ ] **Step 1: Append new configs**

Append after the three official challenge configs. Keep official configs unchanged.

```python
    TrainConfig(
        name="pi05w_insert-mouse-battery_hil",
        model=pi0_config.Pi0Config(pi05=True),
        data=DualYamDataConfig(
            repo_id="insert-mouse-battery/expert-data",
            base_config=DataConfig(
                prompt_from_task=True,
                local_files_path="/Your/path/to/Posttraining-RFM-RSS2026/Challenge-phase1-dataset/insert-mouse-battery/expert-data",
            ),
            use_delta_joint_actions=True,
            adapt_to_pi=True,
        ),
        sample_weight_index_path="/Your/path/to/indexes/insert-mouse-battery.parquet",
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80_000,
        batch_size=32,
        num_workers=64,
        save_interval=20_000,
    ),
```

Repeat with changed names and paths:

```python
"pi05w_seal-water-bottle-cap_hil"
"pi05w_tower-of-hanoi-game_hil"
```

- [ ] **Step 2: Add launcher**

Create `train_weighted.sh`:

```bash
source setup_env.sh

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

CONFIG=$1
EXP_ID=$(date "+%Y%m%d-%H%M%S")
EXP_NAME="${CONFIG}"

mkdir -p logs
LOG_FILE="logs/${EXP_NAME}_${EXP_ID}.log"

uv run scripts/compute_norm_stats.py --config-name "$CONFIG"

uv run scripts/train_weighted.py "$CONFIG" --exp-name="$EXP_NAME" --overwrite 2>&1 | tee -a "$LOG_FILE"
```

- [ ] **Step 3: Verify baseline and weighted configs resolve**

Run:

```bash
uv run python -c "from openpi.training import config; print(config.get_config('pi05_insert-mouse-battery').name)"
uv run python -c "from openpi.training import config; print(config.get_config('pi05w_insert-mouse-battery_hil').name)"
```

Expected:

```text
pi05_insert-mouse-battery
pi05w_insert-mouse-battery_hil
```

- [ ] **Step 4: Commit**

```bash
git add src/openpi/training/config.py train_weighted.sh
git commit -m "feat: add additive weighted challenge configs"
```

---

### Task 7: Verification Documentation

**Files:**
- Modify: `docs/challenge_weighted_posttraining.md`

- [ ] **Step 1: Add verification matrix**

Append:

````markdown
## Verification Matrix

Static checks:

```bash
uv run pytest src/openpi/training/challenge_weighting_test.py src/openpi/training/weighted_data_loader_test.py -q
uv run python -m py_compile scripts/build_challenge_index.py scripts/train_weighted.py
```

Official baseline:

```bash
bash train.sh pi05_insert-mouse-battery
```

Experimental weighted training:

```bash
bash train_weighted.sh pi05w_insert-mouse-battery_hil
```

Policy server check after training:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05w_insert-mouse-battery_hil \
  --policy.dir=checkpoints/pi05w_insert-mouse-battery_hil/pi05w_insert-mouse-battery_hil/80000
```

Simulation compare:

```bash
cd ../policy_deployment
python sim/check_in_sim.py --mode compare \
  --bundle sim/assets/example_slim.pkl \
  --host 127.0.0.1 --port 8000 \
  --prompt "Insert the battery to the mouse." \
  --action-horizon 50 \
  --output out/weighted_compare.mp4
```
````

- [ ] **Step 2: Run static checks**

Run:

```bash
uv run pytest src/openpi/training/challenge_weighting_test.py src/openpi/training/weighted_data_loader_test.py -q
uv run python -m py_compile scripts/build_challenge_index.py scripts/train_weighted.py
```

Expected: tests pass and compile commands print no errors.

- [ ] **Step 3: Commit**

```bash
git add docs/challenge_weighted_posttraining.md
git commit -m "docs: add weighted posttraining verification matrix"
```

---

## Second-Stage Plan After Static Weighted BC

After this plan lands and static weighted BC runs, add a separate `pi05awr_*` config family that replaces static weights with clipped offline advantage weights:

```python
weight = static_source_weight * clip(exp(advantage / beta), 0.2, 3.0)
```

Keep this as a separate plan so `pi05_*`, `pi05w_*`, and `pi05awr_*` remain independently comparable.

## Self-Review

- Spec coverage: baseline comparison is preserved by untouched official configs, untouched `train.sh`, untouched `scripts/train.py`, and new `pi05w_*` entrypoints.
- Placeholder scan: no `TBD`, `TODO`, `implement later`, `not implemented`, or "similar to" instructions remain.
- Type consistency: `sample_weight_index_path` is a `str | None`; `WeightLookup.from_parquet` accepts `str | Path`; weighted batches are `(Observation, actions, sample_weight)` throughout `weighted_data_loader.py` and `train_weighted.py`.
