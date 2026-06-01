import numpy as np
import pandas as pd

from openpi.training import weighted_data_loader


def test_index_lookup_returns_default_for_missing_key(tmp_path):
    index_path = tmp_path / "index.parquet"
    pd.DataFrame([{"episode_index": 3, "frame_index": 7, "sample_weight": 2.0}]).to_parquet(index_path, index=False)

    lookup = weighted_data_loader.WeightLookup.from_parquet(index_path)

    assert lookup.get(episode_index=3, frame_index=7) == 2.0
    assert lookup.get(episode_index=3, frame_index=8) == 0.0


def test_index_lookup_allows_explicit_missing_weight_override(tmp_path):
    index_path = tmp_path / "index.parquet"
    pd.DataFrame([{"episode_index": 3, "frame_index": 7, "sample_weight": 2.0}]).to_parquet(index_path, index=False)

    lookup = weighted_data_loader.WeightLookup.from_parquet(index_path, default_weight=1.0)

    assert lookup.get(episode_index=3, frame_index=8) == 1.0


def test_weighted_transformed_dataset_attaches_weight(tmp_path):
    index_path = tmp_path / "index.parquet"
    pd.DataFrame([{"episode_index": 3, "frame_index": 7, "sample_weight": 2.0}]).to_parquet(index_path, index=False)
    lookup = weighted_data_loader.WeightLookup.from_parquet(index_path)
    raw_dataset = [{"episode_index": 3, "frame_index": 7, "actions": [1.0]}]
    dataset = weighted_data_loader.WeightedTransformedDataset(
        raw_dataset,
        transforms=[lambda item: {"actions": item["actions"]}],
        lookup=lookup,
    )

    assert dataset[0]["actions"] == [1.0]
    assert float(dataset[0]["sample_weight"]) == 2.0


def test_weighted_transformed_dataset_supports_scalar_like_indices():
    lookup = weighted_data_loader.WeightLookup({(3, 7): 2.0})
    raw_dataset = [{"episode_index": np.asarray(3), "frame_index": [7], "actions": [1.0]}]
    dataset = weighted_data_loader.WeightedTransformedDataset(
        raw_dataset,
        transforms=[lambda item: {"actions": item["actions"]}],
        lookup=lookup,
    )

    assert float(dataset[0]["sample_weight"]) == 2.0
