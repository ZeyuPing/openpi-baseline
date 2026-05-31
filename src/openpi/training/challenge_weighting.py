import dataclasses
from collections.abc import Sequence

import numpy as np


DROP_MODES = {"restore", "align", "pre_teleop"}


@dataclasses.dataclass(frozen=True)
class CommanderSegment:
    mode: str
    start: int
    end: int


def segment_commander_states(states: Sequence[str]) -> list[CommanderSegment]:
    if not states:
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
    chunk = states[start : start + horizon]
    if not chunk:
        return False
    return chunk[0] not in DROP_MODES and all(state == chunk[0] for state in chunk)


def chunk_has_smooth_actions(
    actions: np.ndarray, *, start: int, horizon: int, max_abs_step: float = 0.2
) -> bool:
    chunk = actions[start : start + horizon]
    if len(chunk) <= 1:
        return True
    return bool(np.max(np.abs(np.diff(chunk, axis=0))) <= max_abs_step)


def sample_weight(source_name: str, commander_state: str, *, success: bool) -> float:
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
