#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
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


def original_task_root_has_leaves(task_root: Path) -> bool:
    return any((task_root / source_name / "data").exists() for source_name in SOURCES)


def merged_root_has_provenance(task_root: Path) -> bool:
    return (task_root / "meta" / "sources.jsonl").is_file() and (task_root / "data").is_dir()


def _read_source_ranges(task_root: Path) -> list[tuple[int, int, str]]:
    source_ranges: list[tuple[int, int, str]] = []
    with (task_root / "meta" / "sources.jsonl").open("r", encoding="utf-8") as provenance:
        for line in provenance:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            source_ranges.append(
                (
                    int(entry["episode_index_start"]),
                    int(entry["episode_index_end"]),
                    Path(entry["source_path"]).name,
                )
            )
    return source_ranges


def _build_episode_source_resolver(task_root: Path) -> Callable[[int], str | None]:
    source_ranges = _read_source_ranges(task_root)

    def resolve(episode_index: int) -> str | None:
        for start, end, source_name in source_ranges:
            if start <= episode_index <= end:
                return source_name
        return None

    return resolve


def iter_merged_parquet_files(task_root: Path) -> Iterable[Path]:
    yield from sorted((task_root / "data").rglob("*.parquet"))


def _scalar_int(value) -> int:
    value = np.asarray(value)
    return int(value.reshape(-1)[0])


def _stack_actions(action_values) -> np.ndarray:
    return np.stack([np.asarray(action, dtype=np.float32) for action in action_values])


def _rows_for_parquet(
    parquet_path: Path,
    *,
    source_name_for_episode: Callable[[int], str | None],
    provenance_path: Path | None = None,
    action_horizon: int,
    max_action_jump: float,
) -> list[dict]:
    rows: list[dict] = []
    df = pd.read_parquet(parquet_path)
    if "observation.commander_state" not in df.columns or "action" not in df.columns:
        return rows

    states = [str(state) for state in df["observation.commander_state"].tolist()]
    actions = _stack_actions(df["action"].to_numpy())
    for row_idx in range(len(df)):
        if not challenge_weighting.chunk_is_mode_pure(states, start=row_idx, horizon=action_horizon):
            continue
        if not challenge_weighting.chunk_has_smooth_actions(
            actions, start=row_idx, horizon=action_horizon, max_abs_step=max_action_jump
        ):
            continue

        episode_index = _scalar_int(df["episode_index"].iloc[row_idx])
        source_name = source_name_for_episode(episode_index)
        if source_name is None:
            if provenance_path is not None:
                raise RuntimeError(
                    f"Episode {episode_index} in {parquet_path} is not covered by provenance file "
                    f"{provenance_path} for merged root {provenance_path.parent.parent}."
                )
            raise RuntimeError(f"Episode {episode_index} in {parquet_path} could not be resolved to a source.")

        mode = states[row_idx]
        rows.append(
            {
                "source_name": source_name,
                "episode_index": episode_index,
                "frame_index": _scalar_int(df["frame_index"].iloc[row_idx]),
                "commander_state": mode,
                "sample_weight": challenge_weighting.sample_weight(
                    source_name, mode, success=_episode_success(source_name)
                ),
                "parquet_path": str(parquet_path),
            }
        )
    return rows


def build_rows(task_root: Path, *, action_horizon: int, max_action_jump: float) -> list[dict]:
    rows: list[dict] = []
    if original_task_root_has_leaves(task_root):
        for source_name, parquet_path in iter_parquet_files(task_root):
            rows.extend(
                _rows_for_parquet(
                    parquet_path,
                    source_name_for_episode=lambda _episode_index, source_name=source_name: source_name,
                    action_horizon=action_horizon,
                    max_action_jump=max_action_jump,
                )
            )
        return rows

    if merged_root_has_provenance(task_root):
        provenance_path = task_root / "meta" / "sources.jsonl"
        source_name_for_episode = _build_episode_source_resolver(task_root)
        for parquet_path in iter_merged_parquet_files(task_root):
            rows.extend(
                _rows_for_parquet(
                    parquet_path,
                    source_name_for_episode=source_name_for_episode,
                    provenance_path=provenance_path,
                    action_horizon=action_horizon,
                    max_action_jump=max_action_jump,
                )
            )
        return rows

    raise RuntimeError(
        f"Unrecognized task root layout at {task_root}. Expected either an original task root with one of "
        f"{', '.join(f'{source}/data' for source in SOURCES)} or a merged LeRobot root with "
        "meta/sources.jsonl and data/."
    )


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
