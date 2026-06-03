#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.transforms as _transforms
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


DEFAULT_FEATURE_KEY = "pi05_prefix_tokens"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract baseline-aligned pi0.5 prefix features for takeover value training. "
            "This intentionally reuses the configured LeRobot dataset, transforms, model construction, "
            "and weight loader instead of reading videos or loading a standalone vision tower."
        )
    )
    parser.add_argument("--config-name", required=True, help="Training config whose data/model path should be reused.")
    parser.add_argument(
        "--targets",
        required=True,
        type=Path,
        help="Value-target parquet. Provides the episode/frame/source keys that need cached features.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write feature .npz files.")
    parser.add_argument("--batch-size", default=32, type=int, help="Feature extraction batch size.")
    parser.add_argument(
        "--feature-key",
        default=DEFAULT_FEATURE_KEY,
        help="Suffix used in saved files: episode_000000_<feature-key>.npz.",
    )
    return parser.parse_args()


def _as_int_scalar(value: Any) -> int:
    value = np.asarray(value)
    return int(value.reshape(-1)[0])


def _target_source_lookup(targets_path: Path) -> dict[tuple[int, int], str]:
    frame = pd.read_parquet(targets_path, columns=["source_name", "episode_index", "frame_index"])
    lookup: dict[tuple[int, int], str] = {}
    conflicts: list[tuple[int, int]] = []
    for source_name, episode_index, frame_index in zip(
        frame["source_name"], frame["episode_index"], frame["frame_index"], strict=True
    ):
        key = (int(episode_index), int(frame_index))
        source = str(source_name)
        previous = lookup.get(key)
        if previous is not None and previous != source:
            conflicts.append(key)
        lookup[key] = source
    if conflicts:
        sample = ", ".join(f"(episode={ep}, frame={fr})" for ep, fr in conflicts[:5])
        raise RuntimeError(
            "Targets contain duplicate episode/frame keys with different sources. "
            "Use a merged LeRobot root with globally unique episode_index values before extracting features. "
            f"Examples: {sample}"
        )
    return lookup


def _make_data_transform(data_config: _config.DataConfig):
    norm_stats = data_config.norm_stats
    if data_config.repo_id != "fake" and norm_stats is None:
        raise ValueError(
            "Normalization stats not found. Run scripts/compute_norm_stats.py for the extraction config first, "
            "just as baseline training does."
        )
    return _transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )


def _collate_observations(samples: Sequence[Mapping[str, Any]]) -> _model.Observation:
    batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *samples)
    batch = jax.tree.map(jnp.asarray, batch)
    return _model.Observation.from_dict(batch)


def _load_baseline_model(config: _config.TrainConfig):
    rng = jax.random.key(config.seed)
    model = config.model.create(rng)
    graphdef, state = nnx.split(model)
    reference = state.to_pure_dict()
    loaded = config.weight_loader.load(reference)
    _assert_pytree_structure_and_shapes(reference, loaded)
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    model.eval()
    return model


def _assert_pytree_structure_and_shapes(reference: dict[str, Any], loaded: dict[str, Any]) -> None:
    flat_reference = traverse_util.flatten_dict(reference, sep="/")
    flat_loaded = traverse_util.flatten_dict(loaded, sep="/")
    missing = sorted(set(flat_reference) - set(flat_loaded))
    extra = sorted(set(flat_loaded) - set(flat_reference))
    if missing or extra:
        raise ValueError(
            "Loaded pi0.5 parameters do not match the initialized model structure. "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    shape_mismatches = [
        key
        for key, value in flat_loaded.items()
        if getattr(value, "shape", None) != getattr(flat_reference[key], "shape", None)
    ]
    if shape_mismatches:
        raise ValueError(f"Loaded pi0.5 parameters have shape mismatches: {shape_mismatches[:5]}")


def _extract_prefix_features(model, observation: _model.Observation) -> tuple[jnp.ndarray, jnp.ndarray]:
    observation = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    prefix_out, _ = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return prefix_out.astype(jnp.float32), prefix_mask


def _write_feature_groups(
    features_by_episode: dict[tuple[str, int], dict[int, tuple[np.ndarray, np.ndarray]]],
    output_dir: Path,
    *,
    feature_key: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "feature_key": feature_key,
        "episodes": 0,
        "frames": 0,
        "feature_shape": None,
        "token_count": None,
        "token_dim": None,
        "files": [],
    }
    for (source_name, episode_index), frame_features in sorted(features_by_episode.items()):
        if not frame_features:
            continue
        max_frame = max(frame_features)
        first_tokens, first_mask = next(iter(frame_features.values()))
        token_count = int(first_tokens.shape[0])
        token_dim = int(first_tokens.shape[-1])
        episode_features = np.full((max_frame + 1, token_count, token_dim), np.nan, dtype=np.float32)
        episode_masks = np.zeros((max_frame + 1, token_count), dtype=bool)
        for frame_index, (tokens, mask) in frame_features.items():
            if tokens.shape != first_tokens.shape or mask.shape != first_mask.shape:
                raise RuntimeError(
                    f"Inconsistent prefix feature shape in source={source_name}, episode={episode_index}: "
                    f"expected tokens={first_tokens.shape}, mask={first_mask.shape}; "
                    f"got tokens={tokens.shape}, mask={mask.shape}."
                )
            episode_features[frame_index] = tokens.astype(np.float32, copy=False)
            episode_masks[frame_index] = mask.astype(bool, copy=False)

        source_dir = output_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        path = source_dir / f"episode_{episode_index:06d}_{feature_key}.npz"
        np.savez_compressed(path, features=episode_features, mask=episode_masks)

        summary["episodes"] += 1
        summary["frames"] += len(frame_features)
        summary["feature_shape"] = [token_count, token_dim]
        summary["token_count"] = token_count
        summary["token_dim"] = token_dim
        summary["files"].append(str(path))
    return summary


def extract_features_for_config(
    *,
    config_name: str,
    targets: Path,
    output_dir: Path,
    batch_size: int,
    feature_key: str = DEFAULT_FEATURE_KEY,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transform = _make_data_transform(data_config)
    target_sources = _target_source_lookup(targets)
    pending = set(target_sources)

    model = _load_baseline_model(config)
    features_by_episode: dict[tuple[str, int], dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    batch_samples: list[Mapping[str, Any]] = []
    batch_keys: list[tuple[int, int, str]] = []

    def flush() -> None:
        if not batch_samples:
            return
        observation = _collate_observations(batch_samples)
        features, masks = jax.device_get(_extract_prefix_features(model, observation))
        for (episode_index, frame_index, source_name), feature, mask in zip(
            batch_keys, np.asarray(features), np.asarray(masks), strict=True
        ):
            features_by_episode.setdefault((source_name, episode_index), {})[frame_index] = (feature, mask)
            pending.discard((episode_index, frame_index))
        batch_samples.clear()
        batch_keys.clear()

    for index in range(len(raw_dataset)):
        raw_sample = raw_dataset[index]
        episode_index = _as_int_scalar(raw_sample["episode_index"])
        frame_index = _as_int_scalar(raw_sample["frame_index"])
        source_name = target_sources.get((episode_index, frame_index))
        if source_name is None:
            continue
        batch_samples.append(transform(raw_sample))
        batch_keys.append((episode_index, frame_index, source_name))
        if len(batch_samples) >= batch_size:
            flush()
    flush()

    if pending:
        sample = ", ".join(f"(episode={ep}, frame={fr})" for ep, fr in sorted(pending)[:10])
        raise RuntimeError(
            f"Could not find {len(pending)} target rows in the configured dataset {config_name!r}. "
            f"First missing keys: {sample}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _write_feature_groups(features_by_episode, output_dir, feature_key=feature_key)
    summary.update(
        {
            "config_name": config_name,
            "targets": str(targets),
            "output_dir": str(output_dir),
        }
    )
    (output_dir / f"{feature_key}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    extract_features_for_config(
        config_name=args.config_name,
        targets=args.targets,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        feature_key=args.feature_key,
    )


if __name__ == "__main__":
    main()
