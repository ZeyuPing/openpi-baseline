#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


COMMANDER_MODES = ("inference", "teleop", "pre_teleop", "restore", "align")
DEFAULT_FEATURE_KEY = "pi05_prefix"


class ValueNet(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.value_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features)
        return self.value_head(hidden).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a takeover-aware value model.")
    parser.add_argument("--targets", required=True, type=Path, help="Parquet from build_takeover_value_targets.py.")
    parser.add_argument("--predictions", required=True, type=Path, help="Output parquet with value_pred.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Output torch checkpoint path.")
    parser.add_argument("--metrics", type=Path, default=None, help="Optional metrics JSON path.")
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--val-fraction", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda if available, else cpu.")
    parser.add_argument(
        "--fail-on-validation",
        action="store_true",
        help="Raise a RuntimeError and exit with non-zero if validation gates fail.",
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        type=Path,
        help="Optional directory of pre-extracted baseline pi0.5 features.",
    )
    parser.add_argument(
        "--feature-key",
        default=DEFAULT_FEATURE_KEY,
        help="Feature file suffix used by extract_takeover_visual_features.py.",
    )
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="Debug-only: replace missing cached features with zeros instead of failing.",
    )
    return parser.parse_args()


def _state_matrix(frame: pd.DataFrame) -> np.ndarray:
    states = []
    for value in frame.get("state", [[]] * len(frame)):
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if len(array) < 14:
            padded = np.zeros(14, dtype=np.float32)
            padded[: len(array)] = array
            array = padded
        states.append(array[:14])
    return np.stack(states).astype(np.float32)


def _frame_progress(frame: pd.DataFrame) -> np.ndarray:
    max_frame = frame.groupby("episode_index")["frame_index"].transform("max").to_numpy(dtype=np.float32)
    frame_index = frame["frame_index"].to_numpy(dtype=np.float32)
    denom = np.maximum(max_frame, 1.0)
    return (frame_index / denom).reshape(-1, 1).astype(np.float32)


def _mode_features(frame: pd.DataFrame) -> np.ndarray:
    modes = frame["commander_state"].astype(str).to_numpy()
    features = np.zeros((len(frame), len(COMMANDER_MODES)), dtype=np.float32)
    for index, mode in enumerate(COMMANDER_MODES):
        features[:, index] = modes == mode
    return features


def _task_features(frame: pd.DataFrame) -> np.ndarray:
    if "task_id" in frame.columns:
        tasks = frame["task_id"].astype(str).to_numpy()
    elif "task_index" in frame.columns:
        tasks = frame["task_index"].astype(str).to_numpy()
    else:
        return np.zeros((len(frame), 0), dtype=np.float32)

    unique_tasks = sorted(set(tasks.tolist()))
    features = np.zeros((len(frame), len(unique_tasks)), dtype=np.float32)
    for index, task in enumerate(unique_tasks):
        features[:, index] = tasks == task
    return features


def _bool_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        return np.zeros((len(frame), 1), dtype=np.float32)
    return frame[name].to_numpy(dtype=np.float32).reshape(-1, 1)


def build_state_features(frame: pd.DataFrame) -> np.ndarray:
    return np.concatenate(
        [
            _state_matrix(frame),
            _frame_progress(frame),
            _mode_features(frame),
            _task_features(frame),
            _bool_column(frame, "is_hil_episode"),
            _bool_column(frame, "is_teleop"),
            _bool_column(frame, "is_drop_mode"),
        ],
        axis=1,
    ).astype(np.float32)


def _feature_path(features_dir: Path, source_name: str, episode_index: int, feature_key: str) -> Path:
    return features_dir / source_name / f"episode_{episode_index:06d}_{feature_key}.npy"


def _load_visual_feature(
    cache: dict[tuple[str, int], np.ndarray | None],
    *,
    features_dir: Path,
    source_name: str,
    episode_index: int,
    frame_index: int,
    feature_key: str,
    allow_missing: bool,
    fallback_dim: int | None,
) -> np.ndarray:
    cache_key = (source_name, episode_index)
    if cache_key not in cache:
        path = _feature_path(features_dir, source_name, episode_index, feature_key)
        if path.exists():
            cache[cache_key] = np.load(path)
        elif allow_missing:
            cache[cache_key] = None
        else:
            raise FileNotFoundError(f"Missing pre-extracted feature file: {path}")

    array = cache[cache_key]
    if array is None:
        if fallback_dim is None:
            raise RuntimeError("fallback_dim must be known when allow_missing_features is enabled.")
        return np.zeros(fallback_dim, dtype=np.float32)
    if frame_index >= len(array):
        if allow_missing:
            return np.zeros(array.shape[-1], dtype=np.float32)
        raise IndexError(
            f"Feature file for source={source_name}, episode={episode_index} has {len(array)} rows; "
            f"cannot read frame_index={frame_index}."
        )
    feature = np.asarray(array[frame_index], dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(feature)):
        if allow_missing:
            return np.zeros(array.shape[-1], dtype=np.float32)
        raise RuntimeError(
            f"Feature for source={source_name}, episode={episode_index}, frame={frame_index} is missing or non-finite."
        )
    return feature


def _infer_visual_dim(frame: pd.DataFrame, features_dir: Path, feature_key: str) -> int:
    cache: dict[tuple[str, int], np.ndarray | None] = {}
    for row in frame.itertuples(index=False):
        feature = _load_visual_feature(
            cache,
            features_dir=features_dir,
            source_name=str(row.source_name),
            episode_index=int(row.episode_index),
            frame_index=int(row.frame_index),
            feature_key=feature_key,
            allow_missing=False,
            fallback_dim=None,
        )
        return int(feature.shape[-1])
    raise RuntimeError("Cannot infer visual feature dimension from an empty targets frame.")


def build_visual_features(
    frame: pd.DataFrame,
    *,
    features_dir: Path | None,
    feature_key: str,
    allow_missing: bool,
) -> np.ndarray:
    if features_dir is None:
        return np.zeros((len(frame), 0), dtype=np.float32)

    fallback_dim = _infer_visual_dim(frame, features_dir, feature_key)
    cache: dict[tuple[str, int], np.ndarray | None] = {}
    features = []
    for row in frame.itertuples(index=False):
        features.append(
            _load_visual_feature(
                cache,
                features_dir=features_dir,
                source_name=str(row.source_name),
                episode_index=int(row.episode_index),
                frame_index=int(row.frame_index),
                feature_key=feature_key,
                allow_missing=allow_missing,
                fallback_dim=fallback_dim,
            )
        )
    return np.stack(features).astype(np.float32)


def split_by_episode(frame: pd.DataFrame, *, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1).")
    rng = np.random.default_rng(seed)
    episodes = np.asarray(sorted(frame["episode_index"].unique().tolist()))
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * val_fraction)))
    val_episodes = set(episodes[:val_count].tolist())
    is_val = frame["episode_index"].isin(val_episodes).to_numpy()
    return ~is_val, is_val


def _batch_indices(indices: np.ndarray, *, batch_size: int, rng: np.random.Generator):
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]


def _evaluate(
    model: ValueNet,
    features: torch.Tensor,
    value_targets: torch.Tensor,
    indices: np.ndarray,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        value_pred = model(features[indices])
        value_loss = torch.mean(torch.square(value_pred - value_targets[indices]))
    return {
        "value_mse": float(value_loss.cpu()),
    }


def run_validation_gates(frame: pd.DataFrame) -> dict[str, Any]:
    success_values = frame.loc[frame["episode_success"], "value_pred"]
    failure_values = frame.loc[~frame["episode_success"], "value_pred"]
    if len(success_values) > 0 and len(failure_values) > 0:
        success_mean = float(success_values.mean())
        failure_mean = float(failure_values.mean())
        success_failure_gap = success_mean - failure_mean
        gap_pass = success_failure_gap > 0
    else:
        success_mean = 0.0
        failure_mean = 0.0
        success_failure_gap = 0.0
        gap_pass = True

    upward_trends = []
    for _, ep_df in frame[frame["episode_success"]].groupby("episode_index"):
        ep_df = ep_df.sort_values("frame_index")
        n = len(ep_df)
        if n >= 2:
            mid = n // 2
            first_half = ep_df.iloc[:mid]["value_pred"].mean()
            second_half = ep_df.iloc[mid:]["value_pred"].mean()
            upward_trends.append(float(second_half - first_half))

    if upward_trends:
        mean_upward_trend = float(np.mean(upward_trends))
        trend_pass = mean_upward_trend > 0
    else:
        mean_upward_trend = 0.0
        trend_pass = True

    hil_ep_risk = []
    pure_success_ep_risk = []
    for _, ep_df in frame.groupby("episode_index"):
        is_hil = ep_df["is_hil_episode"].any()
        has_risk = ep_df["is_takeover_risk"].any()
        is_success = ep_df["episode_success"].any()
        if is_hil:
            hil_ep_risk.append(has_risk)
        elif is_success:
            pure_success_ep_risk.append(has_risk)

    hil_risk_fraction = float(np.mean(hil_ep_risk)) if hil_ep_risk else 1.0
    pure_success_risk_fraction = float(np.mean(pure_success_ep_risk)) if pure_success_ep_risk else 0.0
    risk_pass = (hil_risk_fraction > 0.0) and (pure_success_risk_fraction == 0.0)

    value_std = float(frame["value_pred"].std()) if len(frame) > 1 else 0.0
    collapse_pass = value_std > 0.01

    advantages = frame["value_target"].to_numpy(dtype=np.float32) - frame["value_pred"].to_numpy(dtype=np.float32)
    median = float(np.median(advantages))
    mad = float(np.median(np.abs(advantages - median)))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = float(np.std(advantages))
    if scale < 1e-6:
        scale = 1.0
    std_adv = (advantages - median) / scale

    beta = 1.0
    awr_min = 0.25
    awr_max = 3.0
    awr_mult = np.clip(np.exp(std_adv / beta), awr_min, awr_max)

    from openpi.training import challenge_weighting

    base_weights = []
    for source, mode, success, risk_val in zip(
        frame["source_name"], frame["commander_state"], frame["episode_success"], frame["is_takeover_risk"], strict=True
    ):
        base_weights.append(challenge_weighting.actor_base_weight(source, mode, success=success, takeover_risk=risk_val))
    base_weights = np.asarray(base_weights, dtype=np.float32)

    rho = 0.25
    awr_weights = base_weights * awr_mult
    blended_weights = (1.0 - rho) * base_weights + rho * awr_weights

    weight_sum = float(blended_weights.sum())
    weight_sq_sum = float(np.sum(np.square(blended_weights)))
    ess = float((weight_sum**2) / weight_sq_sum) if weight_sq_sum > 1e-6 else 0.0

    active_count = int(np.sum(base_weights > 0))
    ess_fraction = ess / active_count if active_count > 0 else 0.0
    ess_pass = ess_fraction >= 0.1

    all_passed = bool(gap_pass and trend_pass and risk_pass and collapse_pass and ess_pass)

    results = {
        "success_mean": success_mean,
        "failure_mean": failure_mean,
        "success_failure_value_gap": success_failure_gap,
        "success_failure_gap_passed": gap_pass,
        "mean_upward_trend": mean_upward_trend,
        "upward_trend_passed": trend_pass,
        "hil_risk_fraction": hil_risk_fraction,
        "pure_success_risk_fraction": pure_success_risk_fraction,
        "risk_check_passed": risk_pass,
        "value_pred_std": value_std,
        "non_collapse_passed": collapse_pass,
        "simulated_ess": ess,
        "simulated_ess_fraction": ess_fraction,
        "ess_passed": ess_pass,
        "validation_passed": all_passed,
    }

    print("\n" + "=" * 50)
    print("OFFLINE VALUE VALIDATION REPORT")
    print("=" * 50)
    print(
        f"1. Success-vs-Failure Gap: {success_failure_gap:+.4f} "
        f"(Success: {success_mean:.4f}, Failure: {failure_mean:.4f}) -> {'PASS' if gap_pass else 'FAIL'}"
    )
    print(f"2. Success Upward Trend:   {mean_upward_trend:+.4f} -> {'PASS' if trend_pass else 'FAIL'}")
    print(
        f"3. Takeover Risk Check:    HIL: {hil_risk_fraction * 100:.1f}%, "
        f"Pure Success: {pure_success_risk_fraction * 100:.1f}% -> {'PASS' if risk_pass else 'FAIL'}"
    )
    print(f"4. Non-Collapse (Std Dev): {value_std:.4f} -> {'PASS' if collapse_pass else 'FAIL'}")
    print(
        f"5. AWR weight ESS fraction: {ess_fraction * 100:.1f}% "
        f"(ESS: {ess:.1f} / Active: {active_count}) -> {'PASS' if ess_pass else 'FAIL'}"
    )
    print("-" * 50)
    if all_passed:
        print("All validation gates passed. Ready for actor SFT training.")
    else:
        print("Validation gates failed. Do not run expensive actor training; fall back to static weighted SFT.")
    print("=" * 50 + "\n")

    return results


def train(args: argparse.Namespace) -> dict[str, float]:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    frame = pd.read_parquet(args.targets).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    state_features_np = build_state_features(frame)
    visual_features_np = build_visual_features(
        frame,
        features_dir=args.features_dir,
        feature_key=args.feature_key,
        allow_missing=args.allow_missing_features,
    )
    features_np = np.concatenate([state_features_np, visual_features_np], axis=1).astype(np.float32)
    value_targets_np = frame["value_target"].to_numpy(dtype=np.float32)
    train_mask, val_mask = split_by_episode(frame, val_fraction=args.val_fraction, seed=args.seed)
    train_indices = np.nonzero(train_mask)[0]
    val_indices = np.nonzero(val_mask)[0]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.as_tensor(features_np, device=device)
    value_targets = torch.as_tensor(value_targets_np, device=device)
    model = ValueNet(input_dim=features_np.shape[1], hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for _epoch in range(args.epochs):
        model.train()
        for batch in _batch_indices(train_indices, batch_size=args.batch_size, rng=rng):
            value_pred = model(features[batch])
            value_loss = torch.mean(torch.square(value_pred - value_targets[batch]))
            optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            optimizer.step()

    train_metrics = _evaluate(model, features, value_targets, train_indices)
    val_metrics = _evaluate(model, features, value_targets, val_indices)

    model.eval()
    with torch.no_grad():
        value_pred = model(features)
        frame["value_pred"] = value_pred.detach().cpu().numpy().astype(np.float32)

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.predictions, index=False)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": features_np.shape[1],
            "state_feature_dim": state_features_np.shape[1],
            "visual_feature_dim": visual_features_np.shape[1],
            "hidden_dim": args.hidden_dim,
            "feature_key": args.feature_key,
            "commander_modes": COMMANDER_MODES,
        },
        args.checkpoint,
    )

    validation_metrics = run_validation_gates(frame)
    metrics = {
        "rows": int(len(frame)),
        "train_rows": int(len(train_indices)),
        "val_rows": int(len(val_indices)),
        "input_dim": int(features_np.shape[1]),
        "state_feature_dim": int(state_features_np.shape[1]),
        "visual_feature_dim": int(visual_features_np.shape[1]),
        **validation_metrics,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"val_{key}": value for key, value in val_metrics.items()},
    }
    metrics_path = args.metrics or args.predictions.with_suffix(".json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    if args.fail_on_validation and not validation_metrics["validation_passed"]:
        raise RuntimeError("Offline value validation gates failed. Aborting SFT.")

    return metrics


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
