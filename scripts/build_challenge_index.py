#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from openpi.training import challenge_weighting


SOURCES = ("expert-data", "success-and-hil-data", "failure-data")
INDEX_COLUMNS = (
    "source_name",
    "episode_index",
    "frame_index",
    "commander_state",
    "sample_weight",
    "parquet_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frame-level weighting index for a challenge task.")
    parser.add_argument("--task-root", required=True, type=Path, help="Task directory containing expert/failure/HIL leaves.")
    parser.add_argument("--output", required=True, type=Path, help="Output parquet index path.")
    parser.add_argument("--action-horizon", default=50, type=int, help="Action chunk horizon used by the policy.")
    parser.add_argument("--max-action-jump", default=0.2, type=float, help="Reject chunks with larger per-step jumps.")
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


def _scalar_int(value) -> int:
    value = np.asarray(value)
    return int(value.reshape(-1)[0])


def _stack_actions(action_values) -> np.ndarray:
    return np.stack([np.asarray(action, dtype=np.float32) for action in action_values])


def build_rows(task_root: Path, *, action_horizon: int, max_action_jump: float) -> list[dict]:
    rows: list[dict] = []
    for source_name, parquet_path in iter_parquet_files(task_root):
        df = pd.read_parquet(parquet_path)
        if "observation.commander_state" not in df.columns or "action" not in df.columns:
            continue

        states = [str(state) for state in df["observation.commander_state"].tolist()]
        actions = _stack_actions(df["action"].to_numpy())
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
                    "episode_index": _scalar_int(df["episode_index"].iloc[row_idx]),
                    "frame_index": _scalar_int(df["frame_index"].iloc[row_idx]),
                    "commander_state": mode,
                    "sample_weight": challenge_weighting.sample_weight(
                        source_name, mode, success=_episode_success(source_name)
                    ),
                    "parquet_path": str(parquet_path),
                }
            )
    return rows


def write_index(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=INDEX_COLUMNS).to_parquet(output, index=False)
    summary = {
        "output": str(output),
        "rows": len(rows),
        "total_weight": float(sum(row["sample_weight"] for row in rows)),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    rows = build_rows(args.task_root, action_horizon=args.action_horizon, max_action_jump=args.max_action_jump)
    write_index(rows, args.output)


if __name__ == "__main__":
    main()
