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
TARGET_COLUMNS = (
    "source_name",
    "episode_index",
    "frame_index",
    "commander_state",
    "episode_success",
    "is_hil_episode",
    "is_drop_mode",
    "is_takeover_risk",
    "is_teleop",
    "is_terminal_active",
    "reward",
    "reward_target",
    "return_to_go",
    "value_target",
    "state",
    "parquet_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build takeover-aware Monte-Carlo value targets for the challenge dataset."
    )
    parser.add_argument("--task-root", required=True, type=Path, help="Original task root or merged LeRobot root.")
    parser.add_argument("--output", required=True, type=Path, help="Output parquet path.")
    parser.add_argument(
        "--risk-window-frames",
        default=60,
        type=int,
        help="Number of inference frames before takeover to label as risky.",
    )
    parser.add_argument("--step-penalty", default=-1.0, type=float, help="Per-frame non-terminal penalty.")
    parser.add_argument(
        "--failure-terminal-penalty",
        default=-50.0,
        type=float,
        help="Terminal reward for failed episodes.",
    )
    parser.add_argument(
        "--takeover-risk-penalty",
        default=-2.0,
        type=float,
        help="Additional reward penalty for pre-takeover risk frames.",
    )
    parser.add_argument(
        "--teleop-bonus",
        default=0.0,
        type=float,
        help="Optional reward bonus for teleop correction frames.",
    )
    return parser.parse_args()


def _episode_success(source_name: str) -> bool:
    return source_name in {"expert-data", "success-and-hil-data"}


def _scalar_int(value) -> int:
    value = np.asarray(value)
    return int(value.reshape(-1)[0])


def _scalar_str(value) -> str:
    value = np.asarray(value, dtype=object)
    return str(value.reshape(-1)[0])


def _state_list(value) -> list[float]:
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def original_task_root_has_leaves(task_root: Path) -> bool:
    return any((task_root / source_name / "data").exists() for source_name in SOURCES)


def merged_root_has_provenance(task_root: Path) -> bool:
    return (task_root / "meta" / "sources.jsonl").is_file() and (task_root / "data").is_dir()


def iter_original_parquet_files(task_root: Path) -> Iterable[tuple[str, Path]]:
    for source_name in SOURCES:
        data_root = task_root / source_name / "data"
        if not data_root.exists():
            continue
        for parquet_path in sorted(data_root.rglob("episode_*.parquet")):
            yield source_name, parquet_path


def iter_merged_parquet_files(task_root: Path) -> Iterable[Path]:
    yield from sorted((task_root / "data").rglob("*.parquet"))


def _read_source_ranges(task_root: Path) -> list[tuple[int, int, str]]:
    source_ranges: list[tuple[int, int, str]] = []
    provenance_path = task_root / "meta" / "sources.jsonl"
    with provenance_path.open("r", encoding="utf-8") as provenance:
        for line in provenance:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            source_name = Path(entry["source_path"]).name
            if source_name not in SOURCES:
                raise RuntimeError(f"Unknown source name {source_name!r} in provenance file {provenance_path}.")
            source_ranges.append(
                (
                    int(entry["episode_index_start"]),
                    int(entry["episode_index_end"]),
                    source_name,
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


def _rows_for_episode(
    episode_df: pd.DataFrame,
    *,
    source_name: str,
    parquet_path: Path,
    risk_window_frames: int,
    step_penalty: float,
    failure_terminal_penalty: float,
    takeover_risk_penalty: float,
    teleop_bonus: float,
) -> list[dict]:
    states = [_scalar_str(state) for state in episode_df["observation.commander_state"].tolist()]
    rewards = challenge_weighting.synthesize_rewards(
        source_name,
        states,
        step_penalty=step_penalty,
        failure_terminal_penalty=failure_terminal_penalty,
        takeover_risk_penalty=takeover_risk_penalty,
        teleop_bonus=teleop_bonus,
        risk_window_frames=risk_window_frames,
    )
    returns = challenge_weighting.return_to_go(rewards)
    risk = challenge_weighting.takeover_risk_mask(states, risk_window_frames=risk_window_frames)
    is_hil = challenge_weighting.hil_episode(states)
    terminal_index = challenge_weighting.active_terminal_index(states)
    episode_success = _episode_success(source_name)
    rows = []

    for local_index, (_, row) in enumerate(episode_df.iterrows()):
        mode = states[local_index]
        rows.append(
            {
                "source_name": source_name,
                "episode_index": _scalar_int(row["episode_index"]),
                "frame_index": _scalar_int(row["frame_index"]),
                "commander_state": mode,
                "episode_success": episode_success,
                "is_hil_episode": is_hil,
                "is_drop_mode": mode in challenge_weighting.DROP_MODES,
                "is_takeover_risk": bool(risk[local_index]),
                "is_teleop": mode == "teleop",
                "is_terminal_active": local_index == terminal_index,
                "reward": float(rewards[local_index]),
                "reward_target": 0.0,
                "return_to_go": float(returns[local_index]),
                "value_target": 0.0,
                "state": _state_list(row["observation.state"]) if "observation.state" in row else [],
                "parquet_path": str(parquet_path),
            }
        )
    return rows


def _rows_for_parquet(
    parquet_path: Path,
    *,
    source_name_for_episode: Callable[[int], str | None],
    risk_window_frames: int,
    step_penalty: float,
    failure_terminal_penalty: float,
    takeover_risk_penalty: float,
    teleop_bonus: float,
) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    required_columns = {"episode_index", "frame_index", "observation.commander_state"}
    missing = required_columns - set(df.columns)
    if missing:
        raise RuntimeError(f"{parquet_path} is missing required columns: {sorted(missing)}")

    episode_indices = np.asarray([_scalar_int(value) for value in df["episode_index"].to_numpy()])
    rows: list[dict] = []
    for episode_index in sorted(set(episode_indices.tolist())):
        source_name = source_name_for_episode(episode_index)
        if source_name is None:
            raise RuntimeError(f"Episode {episode_index} in {parquet_path} could not be resolved to a source.")
        episode_df = df.iloc[np.nonzero(episode_indices == episode_index)[0]]
        rows.extend(
            _rows_for_episode(
                episode_df,
                source_name=source_name,
                parquet_path=parquet_path,
                risk_window_frames=risk_window_frames,
                step_penalty=step_penalty,
                failure_terminal_penalty=failure_terminal_penalty,
                takeover_risk_penalty=takeover_risk_penalty,
                teleop_bonus=teleop_bonus,
            )
        )
    return rows


def build_rows(
    task_root: Path,
    *,
    risk_window_frames: int,
    step_penalty: float,
    failure_terminal_penalty: float,
    takeover_risk_penalty: float,
    teleop_bonus: float,
) -> list[dict]:
    rows: list[dict] = []
    if original_task_root_has_leaves(task_root):
        for source_name, parquet_path in iter_original_parquet_files(task_root):
            rows.extend(
                _rows_for_parquet(
                    parquet_path,
                    source_name_for_episode=lambda _episode_index, source_name=source_name: source_name,
                    risk_window_frames=risk_window_frames,
                    step_penalty=step_penalty,
                    failure_terminal_penalty=failure_terminal_penalty,
                    takeover_risk_penalty=takeover_risk_penalty,
                    teleop_bonus=teleop_bonus,
                )
            )
        return rows

    if merged_root_has_provenance(task_root):
        source_name_for_episode = _build_episode_source_resolver(task_root)
        for parquet_path in iter_merged_parquet_files(task_root):
            rows.extend(
                _rows_for_parquet(
                    parquet_path,
                    source_name_for_episode=source_name_for_episode,
                    risk_window_frames=risk_window_frames,
                    step_penalty=step_penalty,
                    failure_terminal_penalty=failure_terminal_penalty,
                    takeover_risk_penalty=takeover_risk_penalty,
                    teleop_bonus=teleop_bonus,
                )
            )
        return rows

    raise RuntimeError(
        f"Unrecognized task root layout at {task_root}. Expected an original task root or merged LeRobot root."
    )


def write_targets(rows: list[dict], output: Path) -> None:
    if not rows:
        raise RuntimeError("No rows were produced for takeover value targets.")

    frame = pd.DataFrame(rows, columns=TARGET_COLUMNS)
    scale = float(np.max(np.abs(frame["return_to_go"].to_numpy(dtype=np.float32))))
    if scale <= 0:
        scale = 1.0
    frame["reward_target"] = frame["reward"] / scale
    frame["value_target"] = frame["return_to_go"] / scale

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    summary = {
        "output": str(output),
        "rows": int(len(frame)),
        "episodes": int(frame["episode_index"].nunique()),
        "return_scale": scale,
        "source_counts": frame["source_name"].value_counts().to_dict(),
        "takeover_risk_frames": int(frame["is_takeover_risk"].sum()),
        "teleop_frames": int(frame["is_teleop"].sum()),
        "drop_mode_frames": int(frame["is_drop_mode"].sum()),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    rows = build_rows(
        args.task_root,
        risk_window_frames=args.risk_window_frames,
        step_penalty=args.step_penalty,
        failure_terminal_penalty=args.failure_terminal_penalty,
        takeover_risk_penalty=args.takeover_risk_penalty,
        teleop_bonus=args.teleop_bonus,
    )
    write_targets(rows, args.output)


if __name__ == "__main__":
    main()
