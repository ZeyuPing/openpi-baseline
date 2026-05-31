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


def test_segments_accept_numpy_state_array():
    states = np.array(["inference", "teleop", "teleop"], dtype=object)
    assert challenge_weighting.segment_commander_states(states) == [
        challenge_weighting.CommanderSegment("inference", 0, 1),
        challenge_weighting.CommanderSegment("teleop", 1, 3),
    ]


def test_chunk_rejected_when_it_crosses_mode_boundary():
    states = ["inference", "inference", "teleop", "teleop"]
    assert challenge_weighting.chunk_is_mode_pure(states, start=0, horizon=2)
    assert not challenge_weighting.chunk_is_mode_pure(states, start=1, horizon=2)


def test_chunk_rejected_when_window_is_invalid():
    states = ["teleop"]
    assert not challenge_weighting.chunk_is_mode_pure(states, start=0, horizon=2)
    assert not challenge_weighting.chunk_is_mode_pure(states, start=-1, horizon=1)
    assert not challenge_weighting.chunk_is_mode_pure(states, start=0, horizon=0)
    assert not challenge_weighting.chunk_is_mode_pure(states, start=0, horizon=-1)


def test_sample_weight_policy():
    assert challenge_weighting.sample_weight("expert-data", "inference", success=True) == 1.0
    assert challenge_weighting.sample_weight("success-and-hil-data", "teleop", success=True) == 2.0
    assert challenge_weighting.sample_weight("success-and-hil-data", "inference", success=True) == 0.7
    assert challenge_weighting.sample_weight("failure-data", "inference", success=False) == 0.0
    assert challenge_weighting.sample_weight("success-and-hil-data", "restore", success=True) == 0.0


def test_sample_weight_accepts_path_like_source_names():
    assert challenge_weighting.sample_weight("insert-mouse-battery/expert-data", "inference", success=True) == 1.0
    assert challenge_weighting.sample_weight("/tmp/x/success-and-hil-data", "teleop", success=True) == 2.0


def test_large_action_jump_rejected():
    actions = np.zeros((4, 14), dtype=np.float32)
    actions[2, 3] = 0.25
    assert not challenge_weighting.chunk_has_smooth_actions(actions, start=1, horizon=2, max_abs_step=0.2)
    assert challenge_weighting.chunk_has_smooth_actions(actions, start=0, horizon=1, max_abs_step=0.2)


def test_action_chunk_rejected_when_window_is_invalid():
    actions = np.zeros((1, 14), dtype=np.float32)
    assert not challenge_weighting.chunk_has_smooth_actions(actions, start=0, horizon=2)
    assert not challenge_weighting.chunk_has_smooth_actions(actions, start=-1, horizon=1)
    assert not challenge_weighting.chunk_has_smooth_actions(actions, start=0, horizon=0)
    assert not challenge_weighting.chunk_has_smooth_actions(actions, start=0, horizon=-1)
