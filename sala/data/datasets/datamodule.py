
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    DistributedSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from sala.data.bucketing import (
    AtomBucketBatchSampler,
    DenseAttentionBudgetBatchSampler,
    DistributedBatchSampler,
    ShardLocalAtomBucketBatchSampler,
    WeightedAtomBucketBatchSampler,
    build_bucket_index_lists,
    build_bucket_shard_index_lists,
)
from sala.data.collate import asu_collate_fn
from sala.data.datasets.augmentation import AsuSymmetryAugmenter
from sala.data.datasets.pt_dataset import AsuPtDataset

logger = logging.getLogger(__name__)


class AsuDataModule(LightningDataModule):

    def __init__(
        self,
        pt_root: str | Path,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
        index_path: Optional[str | Path] = None,
        c_count_min: Optional[int] = None,
        c_count_max: Optional[int] = None,
        classes: Optional[list[str]] = None,
        include_unknown: bool = True,
        spacegroups: Optional[list[int]] = None,
        is_neutral: Optional[bool] = None,
        has_metal: Optional[bool] = None,
        n_heavy_atoms_min: Optional[int] = None,
        n_heavy_atoms_max: Optional[int] = None,
        n_full_atoms_min: Optional[int] = None,
        n_full_atoms_max: Optional[int] = None,
        n_components_min: Optional[int] = None,
        n_components_max: Optional[int] = None,
        shard_cache_size: int = 1,
        shard_suffix: str = "",
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        sample_weight_no_metal: Optional[float] = None,
        sample_weight_has_metal: Optional[float] = None,
        atom_bucket_edges: Optional[Sequence[int]] = None,
        shard_local_batches: bool = False,
        max_dense_attention_cost: Optional[int] = None,
        sample_weights_path: Optional[str | Path] = None,
        asu_augmentation_enabled: bool = True,
        asu_augmentation_probability: float = 0.5,
        drop_last_train_batches: bool = False,
        distributed_drop_last_batches: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        self.pt_root = Path(pt_root)
        self.shard_suffix = shard_suffix if shard_suffix else ""
        self.index_path = self._resolve_index_path(index_path)
        self.c_count_min = c_count_min
        self.c_count_max = c_count_max
        self.classes = classes
        self.include_unknown = include_unknown
        self.spacegroups = spacegroups
        self.is_neutral = is_neutral
        self.has_metal = has_metal
        self.n_heavy_atoms_min = n_heavy_atoms_min
        self.n_heavy_atoms_max = n_heavy_atoms_max
        self.n_full_atoms_min = n_full_atoms_min
        self.n_full_atoms_max = n_full_atoms_max
        self.n_components_min = n_components_min
        self.n_components_max = n_components_max
        self.shard_cache_size = shard_cache_size
        self.pin_memory = pin_memory and num_workers > 0
        self.persistent_workers = persistent_workers and num_workers > 0
        self.sample_weight_no_metal = sample_weight_no_metal
        self.sample_weight_has_metal = sample_weight_has_metal
        self.atom_bucket_edges: Optional[List[int]] = (
            [int(x) for x in atom_bucket_edges] if atom_bucket_edges is not None else None
        )
        self.shard_local_batches = bool(shard_local_batches)
        self.max_dense_attention_cost = (
            int(max_dense_attention_cost)
            if max_dense_attention_cost is not None
            else None
        )
        self.sample_weights_path = (
            Path(sample_weights_path).expanduser().resolve()
            if sample_weights_path
            else None
        )
        self.asu_augmentation_enabled = bool(asu_augmentation_enabled)
        self.asu_augmentation_probability = float(asu_augmentation_probability)
        self.drop_last_train_batches = bool(drop_last_train_batches)
        self.distributed_drop_last_batches = bool(distributed_drop_last_batches)
        if self.max_dense_attention_cost is not None and not self.atom_bucket_edges:
            raise ValueError(
                "max_dense_attention_cost requires atom_bucket_edges so every "
                "training path uses an explicit batch sampler"
            )


        self.train_dataset: Optional[object] = None
        self._train_sampler: Optional[WeightedRandomSampler] = None
        self._train_batch_sampler: Optional[object] = None
        self.val_dataset: Optional[object] = None
        self.test_dataset: Optional[object] = None
        self._full_dataset: Optional[object] = None

        self.save_hyperparameters()

    def _resolve_index_path(self, index_path: Optional[str | Path]) -> Optional[Path]:
        if index_path:
            return Path(index_path)
        return self.pt_root / f"index{self.shard_suffix}.pt"

    def prepare_data(self) -> None:
        if not self.pt_root.exists():
            raise FileNotFoundError(f"PT root not found: {self.pt_root}")
        if self.index_path is None or not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")
        if not (self.pt_root / "shards").exists():
            raise FileNotFoundError(f"Shards directory not found: {self.pt_root / 'shards'}")
        logger.info("Using preprocessed dataset: %s", self.pt_root)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is None:
            train_augmenter = (
                AsuSymmetryAugmenter(
                    probability=self.asu_augmentation_probability,
                )
                if self.asu_augmentation_enabled
                else None
            )
            self.train_dataset = AsuPtDataset(
                root=self.pt_root,
                index_path=self.index_path,
                split="train",
                c_count_min=self.c_count_min,
                c_count_max=self.c_count_max,
                classes=self.classes,
                include_unknown=self.include_unknown,
                spacegroups=self.spacegroups,
                is_neutral=self.is_neutral,
                has_metal=self.has_metal,
                n_heavy_atoms_min=self.n_heavy_atoms_min,
                n_heavy_atoms_max=self.n_heavy_atoms_max,
                n_full_atoms_min=self.n_full_atoms_min,
                n_full_atoms_max=self.n_full_atoms_max,
                n_components_min=self.n_components_min,
                n_components_max=self.n_components_max,
                cache_shards=self.shard_cache_size,
                shard_suffix=self.shard_suffix,
                augmenter=train_augmenter,
            )
            shared_entries = self.train_dataset.raw_entries
            self.val_dataset = AsuPtDataset(
                root=self.pt_root,
                entries=shared_entries,
                index_path=self.index_path,
                split="val",
                c_count_min=self.c_count_min,
                c_count_max=self.c_count_max,
                classes=self.classes,
                include_unknown=self.include_unknown,
                spacegroups=self.spacegroups,
                is_neutral=self.is_neutral,
                has_metal=self.has_metal,
                n_heavy_atoms_min=self.n_heavy_atoms_min,
                n_heavy_atoms_max=self.n_heavy_atoms_max,
                n_full_atoms_min=self.n_full_atoms_min,
                n_full_atoms_max=self.n_full_atoms_max,
                n_components_min=self.n_components_min,
                n_components_max=self.n_components_max,
                cache_shards=self.shard_cache_size,
                shard_suffix=self.shard_suffix,
            )
            self.test_dataset = AsuPtDataset(
                root=self.pt_root,
                entries=shared_entries,
                index_path=self.index_path,
                split="test",
                c_count_min=self.c_count_min,
                c_count_max=self.c_count_max,
                classes=self.classes,
                include_unknown=self.include_unknown,
                spacegroups=self.spacegroups,
                is_neutral=self.is_neutral,
                has_metal=self.has_metal,
                n_heavy_atoms_min=self.n_heavy_atoms_min,
                n_heavy_atoms_max=self.n_heavy_atoms_max,
                n_full_atoms_min=self.n_full_atoms_min,
                n_full_atoms_max=self.n_full_atoms_max,
                n_components_min=self.n_components_min,
                n_components_max=self.n_components_max,
                cache_shards=self.shard_cache_size,
                shard_suffix=self.shard_suffix,
            )
            self._full_dataset = AsuPtDataset(
                root=self.pt_root,
                entries=shared_entries,
                index_path=self.index_path,
                split=None,
                c_count_min=self.c_count_min,
                c_count_max=self.c_count_max,
                classes=self.classes,
                include_unknown=self.include_unknown,
                spacegroups=self.spacegroups,
                is_neutral=self.is_neutral,
                has_metal=self.has_metal,
                n_heavy_atoms_min=self.n_heavy_atoms_min,
                n_heavy_atoms_max=self.n_heavy_atoms_max,
                n_full_atoms_min=self.n_full_atoms_min,
                n_full_atoms_max=self.n_full_atoms_max,
                n_components_min=self.n_components_min,
                n_components_max=self.n_components_max,
                cache_shards=self.shard_cache_size,
                shard_suffix=self.shard_suffix,
            )

        logger.info(
            f"PT dataset: train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, test={len(self.test_dataset)}"
            )

        self._train_sampler = None
        self._train_batch_sampler = None
        n_train = len(self.train_dataset) if self.train_dataset is not None else 0

        file_weights: Optional[torch.Tensor] = None
        if self.sample_weights_path is not None:
            if not self.sample_weights_path.is_file():
                raise FileNotFoundError(f"sample_weights_path not found: {self.sample_weights_path}")
            try:
                w = torch.load(self.sample_weights_path, map_location="cpu", weights_only=True)
            except TypeError:
                w = torch.load(self.sample_weights_path, map_location="cpu")
            if not torch.is_tensor(w):
                w = torch.tensor(w, dtype=torch.float32)
            else:
                w = w.float().view(-1)
            if w.numel() != n_train:
                raise ValueError(
                    f"sample_weights length {w.numel()} != train len {n_train} ({self.sample_weights_path})"
                )
            file_weights = w
            logger.info("Loaded train sample weights from %s", self.sample_weights_path)

        edges_sorted: Optional[List[int]] = None
        if self.atom_bucket_edges:
            edges_sorted = sorted(set(self.atom_bucket_edges))

        if edges_sorted and self.train_dataset is not None:
            entries = [self.train_dataset.get_entry_meta(i) for i in range(n_train)]
            n_list = [self._full_cell_count(entry) for entry in entries]
            buckets = build_bucket_index_lists(n_list, edges_sorted)
            if file_weights is not None:
                w_tensor = file_weights
            elif (
                self.sample_weight_no_metal is not None
                and self.sample_weight_has_metal is not None
            ):
                w_list = []
                for entry in entries:
                    has_metal = bool(entry.get("has_metal", False))
                    w_list.append(
                        self.sample_weight_has_metal
                        if has_metal
                        else self.sample_weight_no_metal
                    )
                w_tensor = torch.tensor(w_list, dtype=torch.float32)
            else:
                w_tensor = torch.ones(n_train, dtype=torch.float32)

            use_weighted_bucket = file_weights is not None or (
                self.sample_weight_no_metal is not None
                and self.sample_weight_has_metal is not None
            )
            if use_weighted_bucket:
                if self.shard_local_batches:
                    logger.info("shard_local_batches is ignored when weighted sampling is enabled")
                if (
                    self.sample_weight_no_metal is not None
                    and self.sample_weight_has_metal is not None
                    and file_weights is None
                ):
                    logger.info(
                        "Train batch sampler: atom buckets + metal weights (no_metal=%.2f, has_metal=%.2f)",
                        self.sample_weight_no_metal,
                        self.sample_weight_has_metal,
                    )
                else:
                    logger.info("Train batch sampler: weighted atom buckets (edges=%s)", edges_sorted)
                self._train_batch_sampler = WeightedAtomBucketBatchSampler(
                    buckets,
                    w_tensor,
                    batch_size=self.batch_size,
                    num_samples=n_train,
                    seed=self.seed,
                    drop_last=self.drop_last_train_batches,
                )
            elif self.shard_local_batches:
                shard_ids = [
                    entry.get("shard_id") if entry.get("shard_id") is not None else entry.get("shard_path")
                    for entry in entries
                ]
                bucket_groups = build_bucket_shard_index_lists(n_list, shard_ids, edges_sorted)
                estimated_full_cell_tokens = [
                    self._full_cell_count(entry) for entry in entries
                ]
                logger.info(
                    "Train batch sampler: shard-local full-cell buckets "
                    "(edges=%s, groups=%d, max_dense_attention_cost=%s)",
                    edges_sorted,
                    sum(len(bucket) for bucket in bucket_groups),
                    self.max_dense_attention_cost,
                )
                self._train_batch_sampler = ShardLocalAtomBucketBatchSampler(
                    bucket_groups,
                    batch_size=self.batch_size,
                    shuffle=True,
                    seed=self.seed,
                    drop_last=self.drop_last_train_batches,
                    estimated_full_cell_tokens=estimated_full_cell_tokens,
                    max_dense_attention_cost=self.max_dense_attention_cost,
                )
            else:
                logger.info("Train batch sampler: atom buckets (edges=%s)", edges_sorted)
                self._train_batch_sampler = AtomBucketBatchSampler(
                    buckets,
                    batch_size=self.batch_size,
                    shuffle=True,
                    seed=self.seed,
                    drop_last=self.drop_last_train_batches,
                )
        elif file_weights is not None:
            self._train_sampler = WeightedRandomSampler(
                weights=file_weights.tolist(),
                num_samples=n_train,
                replacement=True,
            )
            logger.info("Train sampler: weights from file (WeightedRandomSampler)")
        elif (
            self.train_dataset is not None
            and self.sample_weight_no_metal is not None
            and self.sample_weight_has_metal is not None
        ):
            weights = []
            for i in range(n_train):
                entry = self.train_dataset.get_entry_meta(i)
                has_metal = bool(entry.get("has_metal", False))
                w = (
                    self.sample_weight_has_metal
                    if has_metal
                    else self.sample_weight_no_metal
                )
                weights.append(w)
            self._train_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            logger.info(
                "Train sampler: weighted by metal (no_metal=%.2f, has_metal=%.2f)",
                self.sample_weight_no_metal,
                self.sample_weight_has_metal,
            )

        if self.max_dense_attention_cost is not None and self.train_dataset is not None:
            if self._train_batch_sampler is None:
                raise RuntimeError(
                    "dense attention budgeting requires a train batch sampler"
                )
            train_counts = self._dataset_full_cell_counts(self.train_dataset)
            self._train_batch_sampler = DenseAttentionBudgetBatchSampler(
                self._train_batch_sampler,
                estimated_full_cell_tokens=train_counts,
                max_dense_attention_cost=self.max_dense_attention_cost,
            )

    @staticmethod
    def _full_cell_count(entry: object) -> int:
        if not isinstance(entry, dict):
            entry = dict(entry)
        explicit = entry.get("n_full_atoms")
        if explicit is None:
            explicit = entry.get("n_heavy_cell")
        if explicit is None:
            explicit = int(entry.get("n_heavy_atoms") or 0) * max(
                1, int(entry.get("n_symops") or 1)
            )
        return max(0, int(explicit))

    def _dataset_full_cell_counts(self, dataset: object) -> list[int]:
        getter = getattr(dataset, "get_entry_meta", None)
        if not callable(getter):
            raise ValueError(
                "dense attention budgeting requires dataset.get_entry_meta(index)"
            )
        return [self._full_cell_count(getter(index)) for index in range(len(dataset))]

    def _distributed_context(self) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        trainer = getattr(self, "trainer", None)
        world_size = int(getattr(trainer, "world_size", 1)) if trainer is not None else 1
        global_rank = int(getattr(trainer, "global_rank", 0)) if trainer is not None else 0
        return max(1, world_size), global_rank

    def _gradient_accumulation_steps(self) -> int:
        """Read the single source of truth from the attached Lightning Trainer."""
        trainer = getattr(self, "trainer", None)
        steps = int(getattr(trainer, "accumulate_grad_batches", 1))
        if steps < 1:
            raise ValueError("trainer.accumulate_grad_batches must be >= 1")
        return steps

    def _distributed_sampler(
        self,
        dataset: object,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> Optional[DistributedSampler]:
        world_size, rank = self._distributed_context()
        if world_size <= 1:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=self.seed,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_batch_sampler is not None:
            batch_sampler = self._train_batch_sampler
            world_size, rank = self._distributed_context()
            if world_size > 1:
                gradient_accumulation_steps = self._gradient_accumulation_steps()
                batch_sampler = DistributedBatchSampler(
                    batch_sampler,
                    num_replicas=world_size,
                    rank=rank,
                    drop_last=self.distributed_drop_last_batches,
                    # The base sampler already applies its tail policy.  A
                    # memory-budget split is a complete, valid microbatch even
                    # when it is smaller than the nominal batch size.
                    require_full_batches=False,
                    num_batches_multiple=gradient_accumulation_steps,
                )
                logger.info(
                    "Distributed train batch sampler: rank=%d/%d, local_batches=%d, "
                    "drop_last_batches=%s, gradient_accumulation_steps=%d",
                    rank,
                    world_size,
                    len(batch_sampler),
                    self.distributed_drop_last_batches,
                    gradient_accumulation_steps,
                )
            kwargs = dict(
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=asu_collate_fn,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
            if self.num_workers > 0 and self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
            return DataLoader(self.train_dataset, **kwargs)

        distributed_sampler = self._distributed_sampler(
            self.train_dataset,
            shuffle=self._train_sampler is None,
            drop_last=True,
        )
        if distributed_sampler is not None and self._train_sampler is not None:
            raise ValueError(
                "Distributed training with a standalone weighted sampler is not supported; "
                "enable atom_bucket_edges so weights are applied by the distributed batch sampler"
            )
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=distributed_sampler is None and self._train_sampler is None,
            num_workers=self.num_workers,
            collate_fn=asu_collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )
        if distributed_sampler is not None:
            kwargs["sampler"] = distributed_sampler
        elif self._train_sampler is not None:
            kwargs["sampler"] = self._train_sampler
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self.train_dataset, **kwargs)

    def _evaluation_dataloader(self, dataset: object) -> DataLoader:
        if self.max_dense_attention_cost is not None:
            # Budget first on the identical global stream, then distribute whole
            # microbatches.  Budgeting independently after DistributedSampler
            # can produce a different number of validation collectives per rank.
            base_sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(
                base_sampler,
                batch_size=self.batch_size,
                drop_last=False,
            )
            budgeted = DenseAttentionBudgetBatchSampler(
                batch_sampler,
                estimated_full_cell_tokens=self._dataset_full_cell_counts(dataset),
                max_dense_attention_cost=self.max_dense_attention_cost,
                # Curated evaluation sets can intentionally contain rare large
                # cells. Preserve them as singleton batches without relaxing
                # the training sampler's strict memory budget.
                allow_oversized_singletons=True,
            )
            world_size, rank = self._distributed_context()
            if world_size > 1:
                budgeted = DistributedBatchSampler(
                    budgeted,
                    num_replicas=world_size,
                    rank=rank,
                    drop_last=False,
                    require_full_batches=False,
                )
            kwargs = dict(
                batch_sampler=budgeted,
                num_workers=self.num_workers,
                collate_fn=asu_collate_fn,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
            if self.num_workers > 0 and self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
            return DataLoader(dataset, **kwargs)
        sampler = self._distributed_sampler(
            dataset,
            shuffle=False,
            drop_last=False,
        )
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=asu_collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(dataset, **kwargs)

    def val_dataloader(self) -> DataLoader:
        return self._evaluation_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._evaluation_dataloader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._evaluation_dataloader(self._full_dataset)

    @property
    def num_train_samples(self) -> int:
        return len(self.train_dataset) if self.train_dataset else 0

    @property
    def num_val_samples(self) -> int:
        return len(self.val_dataset) if self.val_dataset else 0

    @property
    def num_test_samples(self) -> int:
        return len(self.test_dataset) if self.test_dataset else 0

    @property
    def total_samples(self) -> int:
        return len(self._full_dataset) if self._full_dataset else 0
