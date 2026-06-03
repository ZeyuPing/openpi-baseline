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
    "static_weight",
    "awr_multiplier",
    "advantage",
    "standardized_advantage",
    "value",
    "next_value",
    "reward_sum",
    "parquet_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a takeover-aware AWR sample-weight index.")
    parser.add_argument("--task-root", required=True, type=Path, help="Original task root or merged LeRobot root.")
    parser.add_argument(
        "--value-targets",
        required=True,
        type=Path,
        help="Parquet produced by build_takeover_value_targets.py or value-model predictions.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output parquet index path.")
    parser.add_argument("--value-column", default=None, help="Value column. Defaults to value_pred, then value_target.")
    parser.add_argument("--reward-column", default="reward_target", help="Reward column on the same scale as values.")
    parser.add_argument("--action-horizon", default=50, type=int, help="Action chunk horizon used by the policy.")
    parser.add_argument("--advantage-horizon", default=50, type=int, help="N-step advantage lookahead.")
    parser.add_argument("--max-action-jump", default=0.2, type=float, help="Reject chunks with larger per-step jumps.")
    parser.add_argument("--beta", default=1.0, type=float, help="AWR temperature.")
    parser.add_argument("--awr-min", default=0.25, type=float, help="Minimum clipped AWR multiplier.")
    parser.add_argument("--awr-max", default=3.0, type=float, help="Maximum clipped AWR multiplier.")
    parser.add_argument(
        "--rho",
        default=1.0,
        type=float,
        help="Blend from static weights to AWR weights: (1-rho)*static + rho*AWR.",
    )
    parser.add_argument(
        "--failure-actor-weight",
        default=0.0,
        type=float,
        help="Base actor weight for failure-data. Keep 0.0 for first Stage A runs.",
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


def _stack_actions(action_values) -> np.ndarray:
    return np.stack([np.asarray(action, dtype=np.float32) for action in action_values])


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


class ValueLookup:
    def __init__(self, frame: pd.DataFrame, *, value_column: str | None, reward_column: str):
        fallback_value_column = None
        if value_column is None:
            if "value_pred" in frame.columns and "value_target" in frame.columns:
                value_column = "value_pred"
                fallback_value_column = "value_target"
            elif "value_pred" in frame.columns:
                value_column = "value_pred"
            elif "value_target" in frame.columns:
                value_column = "value_target"
            else:
                raise ValueError("No value column found. Expected value_pred or value_target.")
        if value_column not in frame.columns:
            raise ValueError(f"Missing value column {value_column!r}.")
        if reward_column not in frame.columns:
            raise ValueError(f"Missing reward column {reward_column!r}.")

        if "source_name" in frame.columns:
            source_names = frame["source_name"].astype(str).tolist()
        else:
            source_names = [""] * len(frame)

        self.values = {}
        for source, episode_index, frame_index, value in zip(
            source_names, frame["episode_index"], frame["frame_index"], frame[value_column], strict=True
        ):
            value = float(value)
            if np.isfinite(value):
                self.values[(str(source), int(episode_index), int(frame_index))] = value
        if fallback_value_column is not None:
            for source, episode_index, frame_index, value in zip(
                source_names,
                frame["episode_index"],
                frame["frame_index"],
                frame[fallback_value_column],
                strict=True,
            ):
                key = (str(source), int(episode_index), int(frame_index))
                if key in self.values:
                    continue
                value = float(value)
                if np.isfinite(value):
                    self.values[key] = value
        self.rewards = {
            (str(source), int(episode_index), int(frame_index)): float(reward)
            for source, episode_index, frame_index, reward in zip(
                source_names, frame["episode_index"], frame["frame_index"], frame[reward_column], strict=True
            )
        }
        if "is_takeover_risk" in frame.columns:
            risk_values = frame["is_takeover_risk"]
        else:
            risk_values = [False] * len(frame)
        self.risk = {
            (str(source), int(episode_index), int(frame_index)): bool(is_risk)
            for source, episode_index, frame_index, is_risk in zip(
                source_names, frame["episode_index"], frame["frame_index"], risk_values, strict=True
            )
        }

    @classmethod
    def from_parquet(cls, path: Path, *, value_column: str | None, reward_column: str) -> "ValueLookup":
        return cls(pd.read_parquet(path), value_column=value_column, reward_column=reward_column)

    def value(self, source_name: str, episode_index: int, frame_index: int) -> float | None:
        val = self.values.get((source_name, episode_index, frame_index))
        if val is None:
            val = self.values.get(("", episode_index, frame_index))
        return val

    def reward(self, source_name: str, episode_index: int, frame_index: int) -> float:
        val = self.rewards.get((source_name, episode_index, frame_index))
        if val is None:
            val = self.rewards.get(("", episode_index, frame_index), 0.0)
        return val

    def is_risk(self, source_name: str, episode_index: int, frame_index: int) -> bool:
        val = self.risk.get((source_name, episode_index, frame_index))
        if val is None:
            val = self.risk.get(("", episode_index, frame_index), False)
        return val


def _rows_for_episode(
    episode_df: pd.DataFrame,
    *,
    source_name: str,
    parquet_path: Path,
    lookup: ValueLookup,
    action_horizon: int,
    advantage_horizon: int,
    max_action_jump: float,
    failure_actor_weight: float,
) -> list[dict]:
    states = [_scalar_str(state) for state in episode_df["observation.commander_state"].tolist()]
    actions = _stack_actions(episode_df["action"].to_numpy())
    episode_indices = [_scalar_int(value) for value in episode_df["episode_index"].to_numpy()]
    frame_indices = [_scalar_int(value) for value in episode_df["frame_index"].to_numpy()]
    reject_mask = [
        lookup.is_risk(source_name, episode_index, frame_index)
        for episode_index, frame_index in zip(episode_indices, frame_indices, strict=True)
    ]
    task_index = _scalar_int(episode_df["task_index"].iloc[0]) if "task_index" in episode_df.columns else None
    task_id = _scalar_str(episode_df["task_id"].iloc[0]) if "task_id" in episode_df.columns else None
    rows = []

    for row_idx, (episode_index, frame_index) in enumerate(zip(episode_indices, frame_indices, strict=True)):
        if not challenge_weighting.chunk_is_valid_actor_sample(
            states,
            actions,
            start=row_idx,
            horizon=action_horizon,
            max_action_jump=max_action_jump,
            reject_mask=reject_mask,
        ):
            continue

        value = lookup.value(source_name, episode_index, frame_index)
        if value is None:
            continue

        next_row_idx = min(row_idx + advantage_horizon, len(frame_indices) - 1)
        next_value = lookup.value(source_name, episode_indices[next_row_idx], frame_indices[next_row_idx])
        if next_value is None:
            continue

        reward_sum = 0.0
        for reward_row_idx in range(row_idx, next_row_idx):
            reward_sum += lookup.reward(source_name, episode_indices[reward_row_idx], frame_indices[reward_row_idx])
        advantage = reward_sum + next_value - value
        mode = states[row_idx]
        static_weight = challenge_weighting.actor_base_weight(
            source_name,
            mode,
            success=_episode_success(source_name),
            takeover_risk=False,
            failure_actor_weight=failure_actor_weight,
        )
        if static_weight <= 0:
            continue

        row = {
            "source_name": source_name,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "commander_state": mode,
            "sample_weight": 0.0,
            "static_weight": static_weight,
            "awr_multiplier": 1.0,
            "advantage": advantage,
            "standardized_advantage": 0.0,
            "value": value,
            "next_value": next_value,
            "reward_sum": reward_sum,
            "parquet_path": str(parquet_path),
        }
        if task_index is not None:
            row["task_index"] = task_index
        if task_id is not None:
            row["task_id"] = task_id
        rows.append(row)
    return rows


def _rows_for_parquet(
    parquet_path: Path,
    *,
    source_name_for_episode: Callable[[int], str | None],
    lookup: ValueLookup,
    action_horizon: int,
    advantage_horizon: int,
    max_action_jump: float,
    failure_actor_weight: float,
) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    required_columns = {"episode_index", "frame_index", "observation.commander_state", "action"}
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
                lookup=lookup,
                action_horizon=action_horizon,
                advantage_horizon=advantage_horizon,
                max_action_jump=max_action_jump,
                failure_actor_weight=failure_actor_weight,
            )
        )
    return rows


def build_rows(
    task_root: Path,
    *,
    lookup: ValueLookup,
    action_horizon: int,
    advantage_horizon: int,
    max_action_jump: float,
    failure_actor_weight: float,
) -> list[dict]:
    rows: list[dict] = []
    if original_task_root_has_leaves(task_root):
        for source_name, parquet_path in iter_original_parquet_files(task_root):
            rows.extend(
                _rows_for_parquet(
                    parquet_path,
                    source_name_for_episode=lambda _episode_index, source_name=source_name: source_name,
                    lookup=lookup,
                    action_horizon=action_horizon,
                    advantage_horizon=advantage_horizon,
                    max_action_jump=max_action_jump,
                    failure_actor_weight=failure_actor_weight,
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
                    lookup=lookup,
                    action_horizon=action_horizon,
                    advantage_horizon=advantage_horizon,
                    max_action_jump=max_action_jump,
                    failure_actor_weight=failure_actor_weight,
                )
            )
        return rows

    raise RuntimeError(
        f"Unrecognized task root layout at {task_root}. Expected an original task root or merged LeRobot root."
    )


def _standardize_advantages(frame: pd.DataFrame) -> pd.Series:
    standardized = pd.Series(np.zeros(len(frame), dtype=np.float32), index=frame.index)
    group_keys = ["source_name", "commander_state"]
    if "task_index" in frame.columns:
        group_keys.insert(0, "task_index")
    elif "task_id" in frame.columns:
        group_keys.insert(0, "task_id")
    for _, group in frame.groupby(group_keys, sort=False):
        values = group["advantage"].to_numpy(dtype=np.float32)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad
        if scale < 1e-6:
            scale = float(np.std(values))
        if scale < 1e-6:
            scale = 1.0
        standardized.loc[group.index] = (values - median) / scale
    return standardized


def write_index(
    rows: list[dict],
    output: Path,
    *,
    beta: float,
    awr_min: float,
    awr_max: float,
    rho: float,
) -> None:
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1].")
    if not rows:
        raise RuntimeError("No usable rows were produced for the takeover-aware AWR index.")

    columns = list(INDEX_COLUMNS)
    if "task_index" in rows[0]:
        columns.append("task_index")
    if "task_id" in rows[0]:
        columns.append("task_id")
    frame = pd.DataFrame(rows, columns=columns)
    frame["standardized_advantage"] = _standardize_advantages(frame)
    frame["awr_multiplier"] = np.clip(np.exp(frame["standardized_advantage"] / beta), awr_min, awr_max)
    awr_weight = frame["static_weight"] * frame["awr_multiplier"]
    frame["sample_weight"] = (1.0 - rho) * frame["static_weight"] + rho * awr_weight

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    weights = frame["sample_weight"].to_numpy(dtype=np.float32)
    ess = float(np.square(np.sum(weights)) / max(float(np.sum(np.square(weights))), 1e-6))
    summary = {
        "output": str(output),
        "rows": int(len(frame)),
        "total_weight": float(frame["sample_weight"].sum()),
        "effective_sample_size": ess,
        "rho": rho,
        "beta": beta,
        "awr_min": awr_min,
        "awr_max": awr_max,
        "source_counts": frame["source_name"].value_counts().to_dict(),
        "weight_by_source": frame.groupby("source_name")["sample_weight"].sum().to_dict(),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    lookup = ValueLookup.from_parquet(
        args.value_targets,
        value_column=args.value_column,
        reward_column=args.reward_column,
    )
    rows = build_rows(
        args.task_root,
        lookup=lookup,
        action_horizon=args.action_horizon,
        advantage_horizon=args.advantage_horizon,
        max_action_jump=args.max_action_jump,
        failure_actor_weight=args.failure_actor_weight,
    )
    write_index(rows, args.output, beta=args.beta, awr_min=args.awr_min, awr_max=args.awr_max, rho=args.rho)


if __name__ == "__main__":
    main()
