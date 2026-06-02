import dataclasses
import os
from collections.abc import Sequence

import numpy as np


DROP_MODES = {"restore", "align", "pre_teleop"}
TAKEOVER_TARGET_MODES = {"pre_teleop", "teleop"}


@dataclasses.dataclass(frozen=True)
class CommanderSegment:
    mode: str
    start: int
    end: int


def segment_commander_states(states: Sequence[str]) -> list[CommanderSegment]:
    if len(states) == 0:
        return []

    segments = []
    start = 0
    mode = states[0]
    for index, state in enumerate(states[1:], start=1):
        if state != mode:
            segments.append(CommanderSegment(mode, start, index))
            start = index
            mode = state
    segments.append(CommanderSegment(mode, start, len(states)))
    return segments


def chunk_is_mode_pure(states: Sequence[str], *, start: int, horizon: int) -> bool:
    if start < 0 or horizon <= 0 or start + horizon > len(states):
        return False

    chunk = states[start : start + horizon]
    return chunk[0] not in DROP_MODES and all(state == chunk[0] for state in chunk)


def chunk_has_smooth_actions(
    actions: np.ndarray, *, start: int, horizon: int, max_abs_step: float = 0.2
) -> bool:
    if start < 0 or horizon <= 0 or start + horizon > len(actions):
        return False

    chunk = actions[start : start + horizon]
    if len(chunk) <= 1:
        return True
    return bool(np.max(np.abs(np.diff(chunk, axis=0))) <= max_abs_step)


def hil_episode(states: Sequence[str]) -> bool:
    state_set = {str(state) for state in states}
    return "inference" in state_set and "teleop" in state_set


def takeover_risk_mask(states: Sequence[str], *, risk_window_frames: int = 60) -> np.ndarray:
    if risk_window_frames < 0:
        raise ValueError("risk_window_frames must be non-negative.")

    states = [str(state) for state in states]
    risk = np.zeros(len(states), dtype=bool)
    if risk_window_frames == 0:
        return risk

    segments = segment_commander_states(states)
    for index, segment in enumerate(segments[:-1]):
        if segment.mode != "inference":
            continue
        next_mode = segments[index + 1].mode
        if next_mode not in TAKEOVER_TARGET_MODES:
            continue
        risk_start = max(segment.start, segment.end - risk_window_frames)
        risk[risk_start : segment.end] = True
    return risk


def chunk_overlaps_mask(mask: Sequence[bool], *, start: int, horizon: int) -> bool:
    if start < 0 or horizon <= 0 or start + horizon > len(mask):
        return True
    return bool(np.any(np.asarray(mask, dtype=bool)[start : start + horizon]))


def chunk_is_valid_actor_sample(
    states: Sequence[str],
    actions: np.ndarray,
    *,
    start: int,
    horizon: int,
    max_action_jump: float = 0.2,
    reject_mask: Sequence[bool] | None = None,
) -> bool:
    if not chunk_is_mode_pure(states, start=start, horizon=horizon):
        return False
    if reject_mask is not None and chunk_overlaps_mask(reject_mask, start=start, horizon=horizon):
        return False
    return chunk_has_smooth_actions(actions, start=start, horizon=horizon, max_abs_step=max_action_jump)


def actor_base_weight(
    source_name: str,
    commander_state: str,
    *,
    success: bool,
    takeover_risk: bool = False,
    failure_actor_weight: float = 0.0,
) -> float:
    source_name = os.path.basename(source_name)
    if takeover_risk or commander_state in DROP_MODES:
        return 0.0
    if source_name == "failure-data":
        return failure_actor_weight
    return sample_weight(source_name, commander_state, success=success)


def active_terminal_index(states: Sequence[str]) -> int:
    active_indices = [index for index, state in enumerate(states) if str(state) not in DROP_MODES]
    if not active_indices:
        return max(len(states) - 1, 0)
    return active_indices[-1]


def synthesize_rewards(
    source_name: str,
    states: Sequence[str],
    *,
    step_penalty: float = -1.0,
    failure_terminal_penalty: float = -50.0,
    takeover_risk_penalty: float = -2.0,
    teleop_bonus: float = 0.0,
    risk_window_frames: int = 60,
) -> np.ndarray:
    states = [str(state) for state in states]
    rewards = np.zeros(len(states), dtype=np.float32)
    if not states:
        return rewards

    active = np.asarray([state not in DROP_MODES for state in states], dtype=bool)
    rewards[active] = step_penalty

    terminal_index = active_terminal_index(states)
    source_name = os.path.basename(source_name)
    if source_name == "failure-data":
        rewards[terminal_index] = failure_terminal_penalty
    else:
        rewards[terminal_index] = 0.0

    risk = takeover_risk_mask(states, risk_window_frames=risk_window_frames)
    pre_teleop = np.asarray([state == "pre_teleop" for state in states], dtype=bool)
    rewards[risk | pre_teleop] += takeover_risk_penalty

    if teleop_bonus:
        teleop = np.asarray([state == "teleop" for state in states], dtype=bool)
        rewards[teleop] += teleop_bonus

    return rewards


def return_to_go(rewards: Sequence[float]) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    return np.cumsum(rewards[::-1], dtype=np.float32)[::-1]


def sample_weight(source_name: str, commander_state: str, *, success: bool) -> float:
    source_name = os.path.basename(source_name)

    if commander_state in DROP_MODES:
        return 0.0
    if source_name == "expert-data":
        return 1.0
    if source_name == "success-and-hil-data" and success:
        if commander_state == "teleop":
            return 2.0
        if commander_state == "inference":
            return 0.7
    if source_name == "failure-data":
        return 0.0
    return 0.0
