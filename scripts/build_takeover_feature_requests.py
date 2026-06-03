#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import build_takeover_awr_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a smaller frame request parquet for pi0.5 feature extraction. "
            "The full value-target file can keep all frames, while this request file selects the frames that need "
            "expensive frozen VLA features."
        )
    )
    parser.add_argument("--task-root", required=True, type=Path, help="Original task root or merged LeRobot root.")
    parser.add_argument("--value-targets", required=True, type=Path, help="Full value-target parquet.")
    parser.add_argument("--output", required=True, type=Path, help="Output feature-request parquet.")
    parser.add_argument("--action-horizon", default=50, type=int)
    parser.add_argument("--advantage-horizon", default=50, type=int)
    parser.add_argument("--max-action-jump", default=0.2, type=float)
    parser.add_argument("--failure-actor-weight", default=0.0, type=float)
    parser.add_argument(
        "--actor-stride",
        default=10,
        type=int,
        help="Keep every Nth valid actor chunk start for AWR/value prediction. Use 1 for all actor starts.",
    )
    parser.add_argument(
        "--critic-stride",
        default=30,
        type=int,
        help="Keep every Nth frame per episode for critic training coverage. Use 1 for all frames.",
    )
    return parser.parse_args()


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _key_frame(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["source_name"].astype(str)
        + "\0"
        + frame["episode_index"].astype(str)
        + "\0"
        + frame["frame_index"].astype(str)
    )


def _actor_and_next_keys(
    targets: pd.DataFrame,
    actor_rows: list[dict],
    *,
    advantage_horizon: int,
    actor_stride: int,
) -> tuple[set[tuple[str, int, int]], int]:
    if not actor_rows:
        return set(), 0

    actor_frame = pd.DataFrame(actor_rows).sort_values(["source_name", "episode_index", "frame_index"])
    kept_actor_rows = []
    for _, group in actor_frame.groupby(["source_name", "episode_index"], sort=False):
        kept_actor_rows.append(group.iloc[::actor_stride])
    actor_frame = pd.concat(kept_actor_rows, ignore_index=True) if kept_actor_rows else actor_frame.iloc[:0]

    grouped_targets = {
        (str(source), int(episode)): group.sort_values("frame_index").reset_index(drop=True)
        for (source, episode), group in targets.groupby(["source_name", "episode_index"], sort=False)
    }
    position_lookup = {
        key: {int(frame_index): position for position, frame_index in enumerate(group["frame_index"])}
        for key, group in grouped_targets.items()
    }

    keys: set[tuple[str, int, int]] = set()
    missing_next = 0
    for row in actor_frame.itertuples(index=False):
        source_name = str(row.source_name)
        episode_index = int(row.episode_index)
        frame_index = int(row.frame_index)
        keys.add((source_name, episode_index, frame_index))

        group_key = (source_name, episode_index)
        position = position_lookup.get(group_key, {}).get(frame_index)
        group = grouped_targets.get(group_key)
        if position is None or group is None:
            missing_next += 1
            continue
        next_position = min(position + advantage_horizon, len(group) - 1)
        keys.add((source_name, episode_index, int(group.iloc[next_position]["frame_index"])))
    return keys, missing_next


def _critic_keys(targets: pd.DataFrame, *, critic_stride: int) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    important_columns = ["is_takeover_risk", "is_teleop", "is_terminal_active", "is_drop_mode"]
    important_mask = np.zeros(len(targets), dtype=bool)
    for column in important_columns:
        if column in targets.columns:
            important_mask |= targets[column].to_numpy(dtype=bool)

    important_frame = targets.loc[important_mask]
    for row in important_frame.itertuples(index=False):
        keys.add((str(row.source_name), int(row.episode_index), int(row.frame_index)))

    for _, group in targets.groupby(["source_name", "episode_index"], sort=False):
        sampled = group.sort_values("frame_index").iloc[::critic_stride]
        for row in sampled.itertuples(index=False):
            keys.add((str(row.source_name), int(row.episode_index), int(row.frame_index)))
    return keys


def build_feature_requests(
    *,
    task_root: Path,
    value_targets: Path,
    action_horizon: int,
    advantage_horizon: int,
    max_action_jump: float,
    failure_actor_weight: float,
    actor_stride: int,
    critic_stride: int,
) -> tuple[pd.DataFrame, dict]:
    _require_positive("action_horizon", action_horizon)
    _require_positive("advantage_horizon", advantage_horizon)
    _require_positive("actor_stride", actor_stride)
    _require_positive("critic_stride", critic_stride)

    targets = pd.read_parquet(value_targets).sort_values(["source_name", "episode_index", "frame_index"])
    lookup = build_takeover_awr_index.ValueLookup(targets, value_column="value_target", reward_column="reward_target")
    actor_rows = build_takeover_awr_index.build_rows(
        task_root,
        lookup=lookup,
        action_horizon=action_horizon,
        advantage_horizon=advantage_horizon,
        max_action_jump=max_action_jump,
        failure_actor_weight=failure_actor_weight,
    )

    actor_keys, missing_next = _actor_and_next_keys(
        targets,
        actor_rows,
        advantage_horizon=advantage_horizon,
        actor_stride=actor_stride,
    )
    critic_keys = _critic_keys(targets, critic_stride=critic_stride)
    requested_keys = actor_keys | critic_keys
    key_strings = {f"{source}\0{episode}\0{frame}" for source, episode, frame in requested_keys}
    request_frame = targets.loc[_key_frame(targets).isin(key_strings)].copy()

    summary = {
        "rows": int(len(request_frame)),
        "full_target_rows": int(len(targets)),
        "valid_actor_rows": int(len(actor_rows)),
        "requested_actor_and_next_rows": int(len(actor_keys)),
        "requested_critic_rows": int(len(critic_keys)),
        "missing_next_rows": int(missing_next),
        "actor_stride": actor_stride,
        "critic_stride": critic_stride,
        "action_horizon": action_horizon,
        "advantage_horizon": advantage_horizon,
    }
    return request_frame, summary


def main() -> None:
    args = parse_args()
    request_frame, summary = build_feature_requests(
        task_root=args.task_root,
        value_targets=args.value_targets,
        action_horizon=args.action_horizon,
        advantage_horizon=args.advantage_horizon,
        max_action_jump=args.max_action_jump,
        failure_actor_weight=args.failure_actor_weight,
        actor_stride=args.actor_stride,
        critic_stride=args.critic_stride,
    )
    if request_frame.empty:
        raise RuntimeError("No feature-request rows were produced.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    request_frame.to_parquet(args.output, index=False)
    summary["output"] = str(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
