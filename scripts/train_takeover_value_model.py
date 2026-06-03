#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


COMMANDER_MODES = ("inference", "teleop", "pre_teleop", "restore", "align")
DEFAULT_FEATURE_KEY = "pi05_prefix_tokens"


@dataclasses.dataclass(frozen=True)
class VisualFeatureBatch:
    tokens: np.ndarray
    mask: np.ndarray


@dataclasses.dataclass(frozen=True)
class EpisodeFeatureFile:
    features: np.ndarray
    mask: np.ndarray
    frame_to_row: dict[int, int] | None = None


class TokenValueNet(torch.nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        visual_dim: int,
        hidden_dim: int,
        query_count: int,
        attention_heads: int,
    ):
        super().__init__()
        if query_count <= 0:
            raise ValueError("query_count must be positive.")
        if attention_heads <= 0:
            raise ValueError("attention_heads must be positive.")
        if visual_dim > 0 and hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads when visual features are used.")

        self.visual_dim = visual_dim
        self.state_proj = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        if visual_dim > 0:
            self.token_proj = torch.nn.Sequential(
                torch.nn.LayerNorm(visual_dim),
                torch.nn.Linear(visual_dim, hidden_dim),
            )
            self.query_tokens = torch.nn.Parameter(torch.empty(query_count, hidden_dim))
            torch.nn.init.normal_(self.query_tokens, std=0.02)
            self.cross_attn = torch.nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                batch_first=True,
            )
            self.visual_norm = torch.nn.LayerNorm(hidden_dim)
            fusion_dim = hidden_dim * 2
        else:
            self.token_proj = None
            self.query_tokens = None
            self.cross_attn = None
            self.visual_norm = None
            fusion_dim = hidden_dim

        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(fusion_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.value_head = torch.nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state_features: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state_hidden = self.state_proj(state_features)
        if self.visual_dim > 0:
            if visual_tokens is None or visual_mask is None:
                raise ValueError("visual_tokens and visual_mask are required when visual_dim > 0.")
            token_hidden = self.token_proj(visual_tokens)
            query_tokens = self.query_tokens.unsqueeze(0).expand(state_features.shape[0], -1, -1)
            attn_out, _ = self.cross_attn(
                query_tokens,
                token_hidden,
                token_hidden,
                key_padding_mask=~visual_mask.bool(),
                need_weights=False,
            )
            visual_hidden = self.visual_norm(attn_out).mean(dim=1)
            hidden = torch.cat([state_hidden, visual_hidden], dim=-1)
        else:
            hidden = state_hidden
        hidden = self.fusion(hidden)
        return self.value_head(hidden).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a takeover-aware value model.")
    parser.add_argument(
        "--targets",
        required=True,
        type=Path,
        help=(
            "Parquet rows to train/predict with cached visual features. This can be the full value-target parquet or a "
            "smaller feature-request parquet from build_takeover_feature_requests.py."
        ),
    )
    parser.add_argument(
        "--prediction-targets",
        default=None,
        type=Path,
        help=(
            "Optional full value-target parquet to preserve in the output. When set, value_pred is merged into this "
            "full frame by source/episode/frame and remains NaN for rows that were not feature-extracted."
        ),
    )
    parser.add_argument("--predictions", required=True, type=Path, help="Output parquet with value_pred.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Output torch checkpoint path.")
    parser.add_argument("--metrics", type=Path, default=None, help="Optional metrics JSON path.")
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument(
        "--query-count",
        default=8,
        type=int,
        help="Learnable query count for token feature aggregation.",
    )
    parser.add_argument("--attention-heads", default=8, type=int, help="Cross-attention heads for token aggregation.")
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
        help="Optional directory of pre-extracted baseline pi0.5 token features.",
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


def _row_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["source_name"].astype(str)
        + "\0"
        + frame["episode_index"].astype(str)
        + "\0"
        + frame["frame_index"].astype(str)
    )


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
    return features_dir / source_name / f"episode_{episode_index:06d}_{feature_key}.npz"


def _legacy_feature_path(features_dir: Path, source_name: str, episode_index: int, feature_key: str) -> Path:
    return features_dir / source_name / f"episode_{episode_index:06d}_{feature_key}.npy"


def _load_feature_file(path: Path) -> EpisodeFeatureFile:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            features = np.asarray(loaded["features"], dtype=np.float32)
            mask = np.asarray(loaded["mask"], dtype=bool)
            frame_indices = np.asarray(loaded["frame_index"], dtype=np.int64) if "frame_index" in loaded else None
        if features.ndim != 3 or mask.ndim != 2:
            raise RuntimeError(f"Expected token features [frames, tokens, dim] and mask [frames, tokens] in {path}.")
        if features.shape[:2] != mask.shape:
            raise RuntimeError(f"Feature/mask shape mismatch in {path}: features={features.shape}, mask={mask.shape}.")
        if frame_indices is not None:
            if frame_indices.ndim != 1 or len(frame_indices) != len(features):
                raise RuntimeError(
                    f"Sparse feature frame_index must have one row per feature row in {path}: "
                    f"frame_index={frame_indices.shape}, features={features.shape}."
                )
            return EpisodeFeatureFile(
                features=features,
                mask=mask,
                frame_to_row={int(frame_index): row for row, frame_index in enumerate(frame_indices)},
            )
        return EpisodeFeatureFile(features=features, mask=mask)

    features = np.asarray(loaded, dtype=np.float32)
    if features.ndim != 2:
        raise RuntimeError(f"Expected legacy pooled features [frames, dim] in {path}, got {features.shape}.")
    mask = np.ones((features.shape[0], 1), dtype=bool)
    return EpisodeFeatureFile(features=features[:, None, :], mask=mask)


def _load_visual_feature(
    cache: dict[tuple[str, int], EpisodeFeatureFile | None],
    *,
    features_dir: Path,
    source_name: str,
    episode_index: int,
    frame_index: int,
    feature_key: str,
    allow_missing: bool,
    fallback_shape: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    cache_key = (source_name, episode_index)
    if cache_key not in cache:
        path = _feature_path(features_dir, source_name, episode_index, feature_key)
        if path.exists():
            cache[cache_key] = _load_feature_file(path)
        else:
            legacy_path = _legacy_feature_path(features_dir, source_name, episode_index, feature_key)
            if legacy_path.exists():
                cache[cache_key] = _load_feature_file(legacy_path)
            elif allow_missing:
                cache[cache_key] = None
            else:
                raise FileNotFoundError(f"Missing pre-extracted feature file: {path}")

    episode_features = cache[cache_key]
    if episode_features is None:
        if fallback_shape is None:
            raise RuntimeError("fallback_shape must be known when allow_missing_features is enabled.")
        token_count, feature_dim = fallback_shape
        return np.zeros((token_count, feature_dim), dtype=np.float32), np.ones(token_count, dtype=bool)

    array = episode_features.features
    mask_array = episode_features.mask
    if episode_features.frame_to_row is not None:
        feature_row = episode_features.frame_to_row.get(frame_index)
    else:
        feature_row = frame_index
    if feature_row is None or feature_row >= len(array):
        if allow_missing:
            return np.zeros(array.shape[1:], dtype=np.float32), np.ones(array.shape[1], dtype=bool)
        raise IndexError(
            f"Feature file for source={source_name}, episode={episode_index} has {len(array)} rows; "
            f"cannot read frame_index={frame_index}."
        )
    feature = np.asarray(array[feature_row], dtype=np.float32)
    mask = np.asarray(mask_array[feature_row], dtype=bool)
    if feature.ndim != 2 or mask.ndim != 1 or feature.shape[0] != mask.shape[0]:
        raise RuntimeError(
            f"Invalid token feature shape for source={source_name}, episode={episode_index}, frame={frame_index}: "
            f"feature={feature.shape}, mask={mask.shape}."
        )
    if not np.all(np.isfinite(feature)) or not np.any(mask):
        if allow_missing:
            return np.zeros(array.shape[1:], dtype=np.float32), np.ones(array.shape[1], dtype=bool)
        raise RuntimeError(
            f"Feature for source={source_name}, episode={episode_index}, frame={frame_index} is missing or non-finite."
        )
    return feature, mask


def _infer_visual_shape(frame: pd.DataFrame, features_dir: Path, feature_key: str) -> tuple[int, int]:
    cache: dict[tuple[str, int], EpisodeFeatureFile | None] = {}
    for row in frame.itertuples(index=False):
        feature, _mask = _load_visual_feature(
            cache,
            features_dir=features_dir,
            source_name=str(row.source_name),
            episode_index=int(row.episode_index),
            frame_index=int(row.frame_index),
            feature_key=feature_key,
            allow_missing=False,
            fallback_shape=None,
        )
        return int(feature.shape[0]), int(feature.shape[-1])
    raise RuntimeError("Cannot infer visual feature shape from an empty targets frame.")


def build_visual_features(
    frame: pd.DataFrame,
    *,
    features_dir: Path | None,
    feature_key: str,
    allow_missing: bool,
) -> VisualFeatureBatch:
    if features_dir is None:
        return VisualFeatureBatch(
            tokens=np.zeros((len(frame), 0, 0), dtype=np.float32),
            mask=np.zeros((len(frame), 0), dtype=bool),
        )

    fallback_shape = _infer_visual_shape(frame, features_dir, feature_key)
    cache: dict[tuple[str, int], EpisodeFeatureFile | None] = {}
    features = []
    masks = []
    for row in frame.itertuples(index=False):
        feature, mask = _load_visual_feature(
            cache,
            features_dir=features_dir,
            source_name=str(row.source_name),
            episode_index=int(row.episode_index),
            frame_index=int(row.frame_index),
            feature_key=feature_key,
            allow_missing=allow_missing,
            fallback_shape=fallback_shape,
        )
        features.append(feature)
        masks.append(mask)
    return VisualFeatureBatch(tokens=np.stack(features).astype(np.float32), mask=np.stack(masks).astype(bool))


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
    model: TokenValueNet,
    state_features: torch.Tensor,
    visual_tokens: torch.Tensor | None,
    visual_mask: torch.Tensor | None,
    value_targets: torch.Tensor,
    indices: np.ndarray,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        batch_visual_tokens = visual_tokens[indices] if visual_tokens is not None else None
        batch_visual_mask = visual_mask[indices] if visual_mask is not None else None
        value_pred = model(state_features[indices], batch_visual_tokens, batch_visual_mask)
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
        base_weights.append(
            challenge_weighting.actor_base_weight(source, mode, success=success, takeover_risk=risk_val)
        )
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
    prediction_frame = (
        pd.read_parquet(args.prediction_targets).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        if args.prediction_targets is not None
        else None
    )
    state_features_np = build_state_features(frame)
    visual_feature_batch = build_visual_features(
        frame,
        features_dir=args.features_dir,
        feature_key=args.feature_key,
        allow_missing=args.allow_missing_features,
    )
    visual_tokens_np = visual_feature_batch.tokens
    visual_mask_np = visual_feature_batch.mask
    visual_token_count = int(visual_tokens_np.shape[1])
    visual_feature_dim = int(visual_tokens_np.shape[2]) if visual_tokens_np.ndim == 3 else 0
    value_targets_np = frame["value_target"].to_numpy(dtype=np.float32)
    train_mask, val_mask = split_by_episode(frame, val_fraction=args.val_fraction, seed=args.seed)
    train_indices = np.nonzero(train_mask)[0]
    val_indices = np.nonzero(val_mask)[0]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    state_features = torch.as_tensor(state_features_np, device=device)
    visual_tokens = torch.as_tensor(visual_tokens_np, device=device) if visual_feature_dim > 0 else None
    visual_mask = torch.as_tensor(visual_mask_np, device=device) if visual_feature_dim > 0 else None
    value_targets = torch.as_tensor(value_targets_np, device=device)
    model = TokenValueNet(
        state_dim=state_features_np.shape[1],
        visual_dim=visual_feature_dim,
        hidden_dim=args.hidden_dim,
        query_count=args.query_count,
        attention_heads=args.attention_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for _epoch in range(args.epochs):
        model.train()
        for batch in _batch_indices(train_indices, batch_size=args.batch_size, rng=rng):
            batch_visual_tokens = visual_tokens[batch] if visual_tokens is not None else None
            batch_visual_mask = visual_mask[batch] if visual_mask is not None else None
            value_pred = model(state_features[batch], batch_visual_tokens, batch_visual_mask)
            value_loss = torch.mean(torch.square(value_pred - value_targets[batch]))
            optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            optimizer.step()

    train_metrics = _evaluate(model, state_features, visual_tokens, visual_mask, value_targets, train_indices)
    val_metrics = _evaluate(model, state_features, visual_tokens, visual_mask, value_targets, val_indices)

    model.eval()
    with torch.no_grad():
        value_pred = model(state_features, visual_tokens, visual_mask)
        frame["value_pred"] = value_pred.detach().cpu().numpy().astype(np.float32)

    output_frame = frame
    if prediction_frame is not None:
        output_frame = prediction_frame.copy()
        output_frame["value_pred"] = np.nan
        pred_lookup = dict(zip(_row_key(frame), frame["value_pred"], strict=True))
        output_frame["value_pred"] = _row_key(output_frame).map(pred_lookup).astype(np.float32)

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_parquet(args.predictions, index=False)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "token_cross_attention_value",
            "input_dim": int(state_features_np.shape[1] + visual_token_count * visual_feature_dim),
            "state_feature_dim": state_features_np.shape[1],
            "visual_token_count": visual_token_count,
            "visual_feature_dim": visual_feature_dim,
            "hidden_dim": args.hidden_dim,
            "query_count": args.query_count,
            "attention_heads": args.attention_heads,
            "feature_key": args.feature_key,
            "commander_modes": COMMANDER_MODES,
        },
        args.checkpoint,
    )

    validation_metrics = run_validation_gates(frame)
    metrics = {
        "rows": int(len(frame)),
        "prediction_rows": int(len(output_frame)),
        "prediction_rows_with_value_pred": int(
            np.isfinite(output_frame["value_pred"].to_numpy(dtype=np.float32)).sum()
        ),
        "train_rows": int(len(train_indices)),
        "val_rows": int(len(val_indices)),
        "input_dim": int(state_features_np.shape[1] + visual_token_count * visual_feature_dim),
        "state_feature_dim": int(state_features_np.shape[1]),
        "visual_token_count": visual_token_count,
        "visual_feature_dim": visual_feature_dim,
        "value_model_architecture": "token_cross_attention_value",
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
