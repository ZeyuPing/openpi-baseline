from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, SupportsIndex, TypeVar

import numpy as np


T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


@dataclasses.dataclass(frozen=True)
class WeightLookup:
    weights: dict[tuple[int, int], float]
    default_weight: float = 1.0

    @classmethod
    def from_parquet(cls, path: str | Path, default_weight: float = 1.0) -> "WeightLookup":
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
        import openpi.transforms as _transforms

        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._lookup = lookup

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        raw_sample = self._dataset[index]
        episode_index = _as_int_scalar(raw_sample["episode_index"])
        frame_index = _as_int_scalar(raw_sample["frame_index"])
        sample_weight = self._lookup.get(episode_index=episode_index, frame_index=frame_index)

        transformed = self._transform(raw_sample)
        transformed["sample_weight"] = np.float32(sample_weight)
        return transformed

    def __len__(self) -> int:
        return len(self._dataset)


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
