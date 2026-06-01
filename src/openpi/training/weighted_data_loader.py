from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, SupportsIndex, TypeVar

import jax
import numpy as np

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


@dataclasses.dataclass(frozen=True)
class WeightLookup:
    weights: dict[tuple[int, int], float]
    default_weight: float = 0.0

    @classmethod
    def from_parquet(cls, path: str | Path, default_weight: float = 0.0) -> "WeightLookup":
        import pandas as pd

        frame = pd.read_parquet(path)
        weights = {
            (int(episode_index), int(frame_index)): float(sample_weight)
            for episode_index, frame_index, sample_weight in zip(
                frame["episode_index"],
                frame["frame_index"],
                frame["sample_weight"],
                strict=True,
            )
        }
        return cls(weights=weights, default_weight=default_weight)

    def get(self, episode_index: int, frame_index: int) -> float:
        return self.weights.get((episode_index, frame_index), self.default_weight)


class WeightedTransformedDataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset[Mapping[str, Any]], transforms: Sequence[Any], lookup: WeightLookup):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._lookup = lookup

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        raw_sample = self._dataset[index]
        episode_index = _as_int_scalar(raw_sample["episode_index"])
        frame_index = _as_int_scalar(raw_sample["frame_index"])
        sample_weight = self._lookup.get(episode_index=episode_index, frame_index=frame_index)

        transformed = dict(self._transform(raw_sample))
        transformed["sample_weight"] = np.float32(sample_weight)
        return transformed

    def __len__(self) -> int:
        return len(self._dataset)


class WeightedDataLoaderImpl:
    def __init__(self, data_config: _config.DataConfig, data_loader: _data_loader.TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"], batch["sample_weight"]


def create_weighted_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: str = "jax",
):
    sample_weight_index_path = getattr(config, "sample_weight_index_path", None)
    if sample_weight_index_path is None:
        raise ValueError("sample_weight_index_path is required for weighted training")
    if framework != "jax":
        raise NotImplementedError("Weighted challenge training is currently implemented for JAX training only")

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("Weighted challenge training only supports LeRobot datasets")
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)

    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    weighted_dataset = WeightedTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        WeightLookup.from_parquet(sample_weight_index_path),
    )
    torch_loader = _data_loader.TorchDataLoader(
        weighted_dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework=framework,
    )
    return WeightedDataLoaderImpl(data_config, torch_loader)


def _as_int_scalar(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.item())
    if isinstance(value, list | tuple):
        if len(value) != 1:
            raise ValueError(f"Expected scalar-like value, got {value!r}.")
        return _as_int_scalar(value[0])
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)
