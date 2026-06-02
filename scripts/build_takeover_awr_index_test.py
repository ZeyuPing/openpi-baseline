from pathlib import Path

import numpy as np
import pandas as pd

from scripts import build_takeover_awr_index
from scripts import build_takeover_value_targets


def _write_episode(root: Path, source_name: str, episode_index: int, states: list[str]) -> Path:
    data_root = root / source_name / "data" / "chunk-000"
    data_root.mkdir(parents=True, exist_ok=True)
    parquet_path = data_root / f"episode_{episode_index:06d}.parquet"
    frame = pd.DataFrame(
        {
            "episode_index": [episode_index] * len(states),
            "frame_index": list(range(len(states))),
            "observation.commander_state": states,
            "observation.state": [np.zeros(14, dtype=np.float32) for _ in states],
            "action": [np.zeros(14, dtype=np.float32) for _ in states],
        }
    )
    frame.to_parquet(parquet_path, index=False)
    return parquet_path


def test_value_targets_mark_takeover_risk_and_failure_terminal(tmp_path):
    _write_episode(tmp_path, "success-and-hil-data", 0, ["inference", "inference", "teleop", "teleop"])
    _write_episode(tmp_path, "failure-data", 1, ["inference", "inference"])

    rows = build_takeover_value_targets.build_rows(
        tmp_path,
        risk_window_frames=1,
        step_penalty=-1.0,
        failure_terminal_penalty=-50.0,
        takeover_risk_penalty=-2.0,
        teleop_bonus=0.0,
    )
    output = tmp_path / "targets.parquet"
    build_takeover_value_targets.write_targets(rows, output)
    frame = pd.read_parquet(output)

    success_episode = frame[frame["episode_index"] == 0].sort_values("frame_index")
    failure_episode = frame[frame["episode_index"] == 1].sort_values("frame_index")

    assert success_episode["is_hil_episode"].tolist() == [True, True, True, True]
    assert success_episode["is_takeover_risk"].tolist() == [False, True, False, False]
    assert success_episode["reward"].tolist() == [-1.0, -3.0, -1.0, 0.0]
    assert failure_episode["reward"].tolist() == [-1.0, -50.0]
    assert frame["value_target"].between(-1.0, 0.0).all()


def test_awr_index_excludes_takeover_risk_and_failure_by_default(tmp_path):
    _write_episode(tmp_path, "success-and-hil-data", 0, ["inference", "inference", "teleop", "teleop"])
    _write_episode(tmp_path, "failure-data", 1, ["inference", "inference"])
    target_rows = build_takeover_value_targets.build_rows(
        tmp_path,
        risk_window_frames=1,
        step_penalty=-1.0,
        failure_terminal_penalty=-50.0,
        takeover_risk_penalty=-2.0,
        teleop_bonus=0.0,
    )
    targets_path = tmp_path / "targets.parquet"
    build_takeover_value_targets.write_targets(target_rows, targets_path)
    lookup = build_takeover_awr_index.ValueLookup.from_parquet(
        targets_path,
        value_column="value_target",
        reward_column="reward_target",
    )

    rows = build_takeover_awr_index.build_rows(
        tmp_path,
        lookup=lookup,
        action_horizon=1,
        advantage_horizon=1,
        max_action_jump=0.2,
        failure_actor_weight=0.0,
    )
    output = tmp_path / "awr.parquet"
    build_takeover_awr_index.write_index(rows, output, beta=1.0, awr_min=0.25, awr_max=3.0, rho=1.0)
    frame = pd.read_parquet(output)

    assert set(frame["episode_index"]) == {0}
    assert set(frame["frame_index"]) == {0, 2, 3}
    assert frame.loc[frame["frame_index"] == 0, "static_weight"].item() == 0.7
    assert frame.loc[frame["frame_index"] == 2, "static_weight"].item() == 2.0


def test_train_takeover_value_model(tmp_path):
    from scripts import train_takeover_value_model
    import argparse

    # Write dummy episodes
    _write_episode(tmp_path, "success-and-hil-data", 0, ["inference", "inference", "teleop", "teleop"])
    _write_episode(tmp_path, "failure-data", 1, ["inference", "inference"])

    target_rows = build_takeover_value_targets.build_rows(
        tmp_path,
        risk_window_frames=1,
        step_penalty=-1.0,
        failure_terminal_penalty=-50.0,
        takeover_risk_penalty=-2.0,
        teleop_bonus=0.0,
    )
    targets_path = tmp_path / "targets.parquet"
    build_takeover_value_targets.write_targets(target_rows, targets_path)

    predictions_path = tmp_path / "predictions.parquet"
    checkpoint_path = tmp_path / "checkpoint.pt"

    args = argparse.Namespace(
        targets=targets_path,
        predictions=predictions_path,
        checkpoint=checkpoint_path,
        metrics=None,
        hidden_dim=8,
        query_count=2,
        attention_heads=2,
        batch_size=2,
        epochs=2,
        lr=1e-3,
        weight_decay=1e-4,
        val_fraction=0.5,
        seed=42,
        device="cpu",
        fail_on_validation=False,
        features_dir=None,
        feature_key="pi05_prefix_tokens",
        allow_missing_features=False,
    )

    metrics = train_takeover_value_model.train(args)
    assert "validation_passed" in metrics
    assert predictions_path.exists()
    assert checkpoint_path.exists()

    # Now verify that it throws RuntimeError when fail_on_validation is True (since dummy weights might fail validation)
    args.fail_on_validation = True
    import pytest
    with pytest.raises(RuntimeError):
        train_takeover_value_model.train(args)


def test_train_takeover_value_model_uses_preextracted_features(tmp_path):
    from scripts import train_takeover_value_model
    import argparse

    _write_episode(tmp_path, "success-and-hil-data", 0, ["inference", "inference", "teleop", "teleop"])
    _write_episode(tmp_path, "failure-data", 1, ["inference", "inference"])

    target_rows = build_takeover_value_targets.build_rows(
        tmp_path,
        risk_window_frames=1,
        step_penalty=-1.0,
        failure_terminal_penalty=-50.0,
        takeover_risk_penalty=-2.0,
        teleop_bonus=0.0,
    )
    targets_path = tmp_path / "targets.parquet"
    build_takeover_value_targets.write_targets(target_rows, targets_path)

    features_dir = tmp_path / "features"
    (features_dir / "success-and-hil-data").mkdir(parents=True)
    (features_dir / "failure-data").mkdir(parents=True)
    np.save(features_dir / "success-and-hil-data" / "episode_000000_pi05_prefix.npy", np.ones((4, 3), dtype=np.float32))
    np.save(features_dir / "failure-data" / "episode_000001_pi05_prefix.npy", np.zeros((2, 3), dtype=np.float32))

    args = argparse.Namespace(
        targets=targets_path,
        predictions=tmp_path / "predictions_with_features.parquet",
        checkpoint=tmp_path / "checkpoint_with_features.pt",
        metrics=None,
        hidden_dim=8,
        query_count=2,
        attention_heads=2,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        val_fraction=0.5,
        seed=42,
        device="cpu",
        fail_on_validation=False,
        features_dir=features_dir,
        feature_key="pi05_prefix",
        allow_missing_features=False,
    )

    metrics = train_takeover_value_model.train(args)

    assert metrics["visual_feature_dim"] == 3
    assert metrics["visual_token_count"] == 1
    assert args.predictions.exists()
    assert args.checkpoint.exists()


def test_train_takeover_value_model_uses_token_features(tmp_path):
    from scripts import train_takeover_value_model
    import argparse

    _write_episode(tmp_path, "success-and-hil-data", 0, ["inference", "inference", "teleop", "teleop"])
    _write_episode(tmp_path, "failure-data", 1, ["inference", "inference"])

    target_rows = build_takeover_value_targets.build_rows(
        tmp_path,
        risk_window_frames=1,
        step_penalty=-1.0,
        failure_terminal_penalty=-50.0,
        takeover_risk_penalty=-2.0,
        teleop_bonus=0.0,
    )
    targets_path = tmp_path / "targets.parquet"
    build_takeover_value_targets.write_targets(target_rows, targets_path)

    features_dir = tmp_path / "features"
    (features_dir / "success-and-hil-data").mkdir(parents=True)
    (features_dir / "failure-data").mkdir(parents=True)
    np.savez_compressed(
        features_dir / "success-and-hil-data" / "episode_000000_pi05_prefix_tokens.npz",
        features=np.ones((4, 5, 3), dtype=np.float32),
        mask=np.ones((4, 5), dtype=bool),
    )
    np.savez_compressed(
        features_dir / "failure-data" / "episode_000001_pi05_prefix_tokens.npz",
        features=np.zeros((2, 5, 3), dtype=np.float32),
        mask=np.ones((2, 5), dtype=bool),
    )

    args = argparse.Namespace(
        targets=targets_path,
        predictions=tmp_path / "predictions_with_token_features.parquet",
        checkpoint=tmp_path / "checkpoint_with_token_features.pt",
        metrics=None,
        hidden_dim=8,
        query_count=2,
        attention_heads=2,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        val_fraction=0.5,
        seed=42,
        device="cpu",
        fail_on_validation=False,
        features_dir=features_dir,
        feature_key="pi05_prefix_tokens",
        allow_missing_features=False,
    )

    metrics = train_takeover_value_model.train(args)

    assert metrics["visual_feature_dim"] == 3
    assert metrics["visual_token_count"] == 5
    assert metrics["value_model_architecture"] == "token_cross_attention_value"
    assert args.predictions.exists()
    assert args.checkpoint.exists()


def test_train_takeover_value_model_rejects_missing_preextracted_features(tmp_path):
    from scripts import train_takeover_value_model
    import argparse
    import pytest

    _write_episode(tmp_path, "success-and-hil-data", 0, ["inference"])
    target_rows = build_takeover_value_targets.build_rows(
        tmp_path,
        risk_window_frames=1,
        step_penalty=-1.0,
        failure_terminal_penalty=-50.0,
        takeover_risk_penalty=-2.0,
        teleop_bonus=0.0,
    )
    targets_path = tmp_path / "targets.parquet"
    build_takeover_value_targets.write_targets(target_rows, targets_path)

    args = argparse.Namespace(
        targets=targets_path,
        predictions=tmp_path / "predictions.parquet",
        checkpoint=tmp_path / "checkpoint.pt",
        metrics=None,
        hidden_dim=8,
        query_count=2,
        attention_heads=2,
        batch_size=1,
        epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        val_fraction=0.5,
        seed=42,
        device="cpu",
        fail_on_validation=False,
        features_dir=tmp_path / "missing_features",
        feature_key="pi05_prefix_tokens",
        allow_missing_features=False,
    )

    with pytest.raises(FileNotFoundError):
        train_takeover_value_model.train(args)
