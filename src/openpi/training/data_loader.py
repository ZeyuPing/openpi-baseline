from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class WeightedLeRobotDataset(Dataset):
    """Adds a scalar sample weight to LeRobot samples using merge provenance metadata."""

    def __init__(self, dataset: Dataset, data_config: _config.DataConfig, dataset_root: pathlib.Path):
        if data_config.sample_weight_config is None:
            raise ValueError("sample_weight_config must be set to create a weighted dataset.")
        self._dataset = dataset
        self._config = data_config.sample_weight_config
        self._source_ranges = _load_source_ranges(dataset_root)
        self._episode_lengths = _load_episode_lengths(dataset_root)
        self._episode_start_indices = _episode_start_indices(self._episode_lengths)
        self._hil_episode_indices = _load_hil_episode_indices(dataset_root, self._config)

    def __getitem__(self, index: SupportsIndex) -> dict:
        sample = typing.cast(dict, self._dataset[index])
        episode_index = _scalar_int(sample.get("episode_index"))
        global_frame_index = _scalar_int(sample.get("index"))
        frame_index = _frame_index(sample, episode_index, global_frame_index, self._episode_start_indices)
        source_path = _source_for_episode(self._source_ranges, episode_index)
        source_weight = _source_weight(
            source_path,
            episode_index,
            frame_index,
            self._episode_lengths.get(episode_index),
            sample,
            self._hil_episode_indices,
            self._config,
        )
        progress_weight = _progress_weight(source_path, episode_index, frame_index, self._episode_lengths, self._config)
        return {**sample, "sample_weight": np.asarray(source_weight * progress_weight, dtype=np.float32)}

    def __len__(self) -> int:
        return len(self._dataset)


def _load_source_ranges(dataset_root: pathlib.Path) -> list[tuple[int, int, str]]:
    sources_path = dataset_root / "meta" / "sources.jsonl"
    if not sources_path.exists():
        logging.warning("No merge provenance found at %s; weighted dataset will use default weights.", sources_path)
        return []

    ranges = []
    with sources_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            source = json.loads(line)
            ranges.append(
                (
                    int(source["episode_index_start"]),
                    int(source["episode_index_end"]),
                    str(source.get("source_path", "")),
                )
            )
    return ranges


def _load_episode_lengths(dataset_root: pathlib.Path) -> dict[int, int]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return {}

    lengths = {}
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            episode = json.loads(line)
            lengths[int(episode["episode_index"])] = int(episode["length"])
    return lengths


def _load_hil_episode_indices(dataset_root: pathlib.Path, config: _config.SampleWeightConfig) -> set[int]:
    """Find success-and-hil episodes that actually contain teleop frames.

    The official dataset README defines an HIL episode as one whose
    `observation.commander_state` contains both `inference` and `teleop`.
    """
    data_root = dataset_root / "data"
    if not data_root.exists():
        return set()

    hil_episodes = set()
    try:
        import pandas as pd
    except ImportError:
        logging.warning("pandas unavailable; falling back to per-frame HIL mode weighting only.")
        return hil_episodes

    for parquet_path in data_root.rglob("episode_*.parquet"):
        try:
            df = pd.read_parquet(parquet_path, columns=["episode_index", config.commander_state_key])
        except Exception:
            continue

        if config.commander_state_key not in df:
            continue
        modes = {_string_value(mode) for mode in df[config.commander_state_key].to_numpy()}
        if modes.intersection(config.autonomous_modes) and modes.intersection(config.teleop_modes):
            episode_index = _scalar_int(df["episode_index"].iloc[0]) if "episode_index" in df else None
            if episode_index is not None:
                hil_episodes.add(episode_index)
    return hil_episodes


def _scalar_int(value) -> int | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return int(arr.reshape(-1)[0])


def _string_value(value) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    value = arr.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _sample_get(sample: dict, key: str):
    if key in sample:
        return sample[key]
    current = sample
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _episode_start_indices(episode_lengths: dict[int, int]) -> dict[int, int]:
    starts = {}
    next_start = 0
    for episode_index, length in sorted(episode_lengths.items()):
        starts[episode_index] = next_start
        next_start += length
    return starts


def _frame_index(
    sample: dict,
    episode_index: int | None,
    global_frame_index: int | None,
    episode_start_indices: dict[int, int],
) -> int | None:
    frame_index = _scalar_int(sample.get("frame_index"))
    if frame_index is not None:
        return frame_index
    if episode_index is None or global_frame_index is None:
        return None
    start_index = episode_start_indices.get(episode_index)
    if start_index is None:
        return None
    return global_frame_index - start_index


def _source_for_episode(source_ranges: list[tuple[int, int, str]], episode_index: int | None) -> str:
    if episode_index is None:
        return ""
    for start, end, source_path in source_ranges:
        if start <= episode_index <= end:
            return source_path
    return ""


def _source_weight(
    source_path: str,
    episode_index: int | None,
    frame_index: int | None,
    episode_length: int | None,
    sample: dict,
    hil_episode_indices: set[int],
    config: _config.SampleWeightConfig,
) -> float:
    source = source_path.lower()
    if any(pattern in source for pattern in config.success_and_hil_patterns):
        return _success_and_hil_weight(episode_index, frame_index, episode_length, sample, hil_episode_indices, config)
    if any(pattern in source for pattern in config.hil_patterns):
        return _hil_weight(frame_index, episode_length, sample, config, is_hil_episode=True)
    if any(pattern in source for pattern in config.success_patterns):
        return config.success_weight
    if any(pattern in source for pattern in config.failure_patterns):
        return config.failure_weight
    if any(pattern in source for pattern in config.expert_patterns):
        return config.expert_weight
    return config.default_weight


def _success_and_hil_weight(
    episode_index: int | None,
    frame_index: int | None,
    episode_length: int | None,
    sample: dict,
    hil_episode_indices: set[int],
    config: _config.SampleWeightConfig,
) -> float:
    mode = _commander_mode(sample, config)
    if mode in config.teleop_modes:
        return config.hil_correction_weight
    if mode in config.transition_modes:
        return config.hil_transition_weight
    if mode in config.restore_modes:
        return config.hil_restore_weight
    if mode in config.autonomous_modes:
        if episode_index in hil_episode_indices:
            return config.hil_pre_takeover_weight
        return config.success_weight
    return _hil_weight(frame_index, episode_length, sample, config, is_hil_episode=episode_index in hil_episode_indices)


def _hil_weight(
    frame_index: int | None,
    episode_length: int | None,
    sample: dict,
    config: _config.SampleWeightConfig,
    *,
    is_hil_episode: bool,
) -> float:
    mode = _commander_mode(sample, config)
    if mode in config.teleop_modes:
        return config.hil_correction_weight
    if mode in config.transition_modes:
        return config.hil_transition_weight
    if mode in config.restore_modes:
        return config.hil_restore_weight
    if mode in config.autonomous_modes:
        return config.hil_pre_takeover_weight if is_hil_episode else config.success_weight

    for key in config.hil_takeover_keys:
        value = _sample_get(sample, key)
        if value is not None:
            return config.hil_correction_weight if bool(np.asarray(value).reshape(-1)[0]) else config.hil_pre_takeover_weight

    if frame_index is None or not episode_length or episode_length <= 1:
        return config.hil_pre_takeover_weight

    progress = min(max(frame_index / float(episode_length - 1), 0.0), 1.0)
    if progress >= config.hil_correction_start_fraction:
        return config.hil_correction_weight
    return config.hil_pre_takeover_weight


def _commander_mode(sample: dict, config: _config.SampleWeightConfig) -> str:
    value = _sample_get(sample, config.commander_state_key)
    if value is None:
        return ""
    return _string_value(value)


def _progress_weight(
    source_path: str,
    episode_index: int | None,
    frame_index: int | None,
    episode_lengths: dict[int, int],
    config: _config.SampleWeightConfig,
) -> float:
    source = source_path.lower()
    if not any(pattern in source for pattern in config.failure_patterns):
        return 1.0
    if episode_index is None or frame_index is None:
        return 1.0

    length = episode_lengths.get(episode_index)
    if not length or length <= 1:
        return 1.0

    progress = min(max(frame_index / float(length - 1), 0.0), 1.0)
    if progress <= config.failure_prefix_keep_fraction:
        return 1.0
    if progress >= config.failure_tail_zero_fraction:
        return 0.0
    span = config.failure_tail_zero_fraction - config.failure_prefix_keep_fraction
    if span <= 0:
        return 0.0
    return 1.0 - (progress - config.failure_prefix_keep_fraction) / span


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=data_config.local_files_path)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        root=data_config.local_files_path
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    if data_config.sample_weight_config is not None:
        if data_config.local_files_path is None:
            raise ValueError("sample_weight_config requires local_files_path so merge provenance can be loaded.")
        dataset = WeightedLeRobotDataset(dataset, data_config, pathlib.Path(data_config.local_files_path))

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            sample_weight = batch.pop("sample_weight", None)
            if sample_weight is None:
                yield _model.Observation.from_dict(batch), batch["actions"]
            else:
                yield _model.Observation.from_dict(batch), batch["actions"], sample_weight
