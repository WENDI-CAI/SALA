
from __future__ import annotations

import bisect
import math
from typing import Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Sampler


class DistributedBatchSampler(Sampler[list[int]]):
    """Shard a deterministic batch sampler across data-parallel ranks.

    Every rank must wrap an identically configured base sampler.  The base
    sampler creates the global batch stream and this wrapper assigns complete
    batches round-robin, so atom/shard bucketing is preserved.  Training uses
    ``drop_last=True`` to guarantee that all ranks execute the same number of
    optimizer microsteps without duplicating samples.  ``drop_last=False``
    pads with the first global batches and is intended for evaluation only.
    """

    def __init__(
        self,
        batch_sampler: Sampler[list[int]],
        *,
        num_replicas: int,
        rank: int,
        drop_last: bool = True,
        require_full_batches: bool = False,
        num_batches_multiple: int = 1,
    ) -> None:
        self.batch_sampler = batch_sampler
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.require_full_batches = bool(require_full_batches)
        self.num_batches_multiple = int(num_batches_multiple)
        self._cached_eligible_batch_count: Optional[int] = None
        if self.num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"rank must be in [0, {self.num_replicas}), got {self.rank}"
            )
        if self.num_batches_multiple < 1:
            raise ValueError("num_batches_multiple must be >= 1")

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.batch_sampler, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))
        self._cached_eligible_batch_count = None

    @property
    def sampler(self) -> Sampler[list[int]]:
        """Expose set_epoch through Lightning's batch_sampler.sampler hook."""
        return self

    def __len__(self) -> int:
        base_length = self._eligible_batch_count()
        if self.drop_last:
            local_length = base_length // self.num_replicas
            return (
                local_length // self.num_batches_multiple
            ) * self.num_batches_multiple
        return int(math.ceil(base_length / self.num_replicas)) if base_length else 0

    def __iter__(self) -> Iterator[list[int]]:
        base_length = self._eligible_batch_count()
        if base_length == 0:
            return

        if self.drop_last:
            usable = len(self) * self.num_replicas
            observed = 0
            for batch_index, batch in enumerate(self._iter_eligible_batches()):
                if batch_index >= usable:
                    break
                observed += 1
                if batch_index % self.num_replicas == self.rank:
                    yield list(batch)
            if observed != usable:
                raise RuntimeError(
                    "Base batch sampler yielded fewer batches than __len__ reported: "
                    f"expected at least {usable}, observed {observed}"
                )
            return

        total_size = int(math.ceil(base_length / self.num_replicas)) * self.num_replicas
        padding = total_size - base_length
        prefix: list[list[int]] = []
        observed = 0
        for batch_index, batch in enumerate(self._iter_eligible_batches()):
            batch_list = list(batch)
            if len(prefix) < max(1, padding):
                prefix.append(batch_list)
            if batch_index % self.num_replicas == self.rank:
                yield batch_list
            observed += 1
        if observed != base_length:
            raise RuntimeError(
                "Base batch sampler yielded a different number of batches than "
                f"__len__ reported: expected {base_length}, observed {observed}"
            )
        if padding and not prefix:
            raise RuntimeError("Cannot pad an empty distributed batch stream")
        for offset in range(padding):
            batch_index = base_length + offset
            if batch_index % self.num_replicas == self.rank:
                yield list(prefix[offset % len(prefix)])

    def _iter_eligible_batches(self) -> Iterator[list[int]]:
        expected_size = getattr(self.batch_sampler, "batch_size", None)
        if self.require_full_batches and expected_size is None:
            raise ValueError(
                "require_full_batches=True needs a base sampler with batch_size"
            )
        for batch in self.batch_sampler:
            batch_list = list(batch)
            if self.require_full_batches and len(batch_list) != int(expected_size):
                continue
            yield batch_list

    def _eligible_batch_count(self) -> int:
        if self._cached_eligible_batch_count is not None:
            return self._cached_eligible_batch_count
        if not self.require_full_batches:
            count = len(self.batch_sampler)
        else:
            count = sum(1 for _ in self._iter_eligible_batches())
        self._cached_eligible_batch_count = int(count)
        return int(count)


def bucket_id_for_n_atoms(n_atoms: Optional[int], edges: Sequence[int]) -> int:
    if not edges:
        return 0
    if n_atoms is None:
        return len(edges)
    n = int(n_atoms)
    return bisect.bisect_left(list(edges), n)


def build_bucket_index_lists(
    n_atoms_per_index: Sequence[Optional[int]],
    edges: Sequence[int],
) -> List[List[int]]:
    nb = len(edges) + 1
    buckets: List[List[int]] = [[] for _ in range(nb)]
    for idx, n in enumerate(n_atoms_per_index):
        b = bucket_id_for_n_atoms(n, edges)
        buckets[b].append(idx)
    return buckets


class AtomBucketBatchSampler(Sampler[list[int]]):

    def __init__(
        self,
        buckets: Sequence[Sequence[int]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        self._buckets: List[torch.Tensor] = [
            torch.tensor(list(b), dtype=torch.long) for b in buckets if len(b) > 0
        ]
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        n_batches = 0
        for bi in self._buckets:
            n = bi.numel()
            if drop_last:
                n_batches += n // self.batch_size
            else:
                n_batches += int(math.ceil(n / self.batch_size))
        self._len = n_batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def sampler(self) -> "AtomBucketBatchSampler":
        return self

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        order = list(range(len(self._buckets)))
        if self.shuffle:
            perm = torch.randperm(len(order), generator=g).tolist()
            order = [order[i] for i in perm]
        for bi in order:
            idx = self._buckets[bi]
            if self.shuffle:
                perm = torch.randperm(idx.numel(), generator=g)
                idx = idx[perm]
            n = idx.numel()
            step = self.batch_size
            for start in range(0, n, step):
                end = start + step
                if end > n:
                    if self.drop_last:
                        break
                    chunk = idx[start:n]
                else:
                    chunk = idx[start:end]
                yield chunk.tolist()


class DenseAttentionBudgetBatchSampler(Sampler[list[int]]):
    """Split arbitrary batches until ``B * (max(N_full) + 3)^2`` fits.

    The wrapper never truncates or discards a structure.  It is deliberately
    sampler-agnostic so the same safety boundary applies to weighted training
    and to validation/test/predict loaders, not only to shard-local batches.
    """

    def __init__(
        self,
        batch_sampler: Sampler[list[int]],
        *,
        estimated_full_cell_tokens: Sequence[int],
        max_dense_attention_cost: int,
        allow_oversized_singletons: bool = False,
    ) -> None:
        self.batch_sampler = batch_sampler
        self.batch_size = getattr(batch_sampler, "batch_size", None)
        self._estimated_full_cell_tokens = [
            max(0, int(value)) for value in estimated_full_cell_tokens
        ]
        self.max_dense_attention_cost = int(max_dense_attention_cost)
        self.allow_oversized_singletons = bool(allow_oversized_singletons)
        if self.max_dense_attention_cost < 16:
            raise ValueError("max_dense_attention_cost must be at least 16")
        oversized = [
            value
            for value in self._estimated_full_cell_tokens
            if (value + 3) ** 2 > self.max_dense_attention_cost
        ]
        if oversized and not self.allow_oversized_singletons:
            raise ValueError(
                "A single full-cell sample exceeds max_dense_attention_cost: "
                f"max N_full={max(oversized)}"
            )

    @property
    def sampler(self) -> "DenseAttentionBudgetBatchSampler":
        return self

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.batch_sampler, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))

    def _cost(self, indices: Sequence[int]) -> int:
        if not indices:
            return 0
        try:
            max_full = max(
                self._estimated_full_cell_tokens[int(index)] for index in indices
            )
        except IndexError as exc:
            raise IndexError(
                "batch sampler produced an index outside the full-cell metadata"
            ) from exc
        return len(indices) * (max_full + 3) ** 2

    def _split(self, batch: Sequence[int]) -> Iterator[list[int]]:
        pending: list[int] = []
        for raw_index in batch:
            index = int(raw_index)
            candidate = [*pending, index]
            if pending and self._cost(candidate) > self.max_dense_attention_cost:
                yield pending
                pending = [index]
            else:
                pending = candidate
        if pending:
            # Evaluation may explicitly retain an oversized sample as a
            # singleton; training keeps the strict construction-time guard.
            yield pending

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.batch_sampler:
            yield from self._split(list(batch))

    def __len__(self) -> int:
        return sum(1 for _ in self)



def build_bucket_shard_index_lists(
    n_atoms_per_index: Sequence[Optional[int]],
    shard_id_per_index: Sequence[object],
    edges: Sequence[int],
) -> List[List[List[int]]]:
    if len(n_atoms_per_index) != len(shard_id_per_index):
        raise ValueError("n_atoms_per_index and shard_id_per_index must have the same length")
    nb = len(edges) + 1
    groups_by_bucket: List[dict[object, List[int]]] = [dict() for _ in range(nb)]
    for idx, (n_atoms, shard_id) in enumerate(zip(n_atoms_per_index, shard_id_per_index)):
        bucket_id = bucket_id_for_n_atoms(n_atoms, edges)
        key = "__missing_shard__" if shard_id is None else shard_id
        groups_by_bucket[bucket_id].setdefault(key, []).append(idx)

    return [
        [group for group in shard_groups.values() if group]
        for shard_groups in groups_by_bucket
    ]


class ShardLocalAtomBucketBatchSampler(Sampler[list[int]]):

    def __init__(
        self,
        bucket_groups: Sequence[Sequence[Sequence[int]]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        estimated_full_cell_tokens: Optional[Sequence[int]] = None,
        max_dense_attention_cost: Optional[int] = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.max_dense_attention_cost = (
            int(max_dense_attention_cost)
            if max_dense_attention_cost is not None
            else None
        )
        self._epoch = 0
        self._bucket_groups: List[List[torch.Tensor]] = []
        for bucket in bucket_groups:
            groups = [
                torch.tensor(list(group), dtype=torch.long)
                for group in bucket
                if len(group) > 0
            ]
            if groups:
                self._bucket_groups.append(groups)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_dense_attention_cost is not None:
            if self.max_dense_attention_cost < 16:
                raise ValueError("max_dense_attention_cost must be at least 16")
            if estimated_full_cell_tokens is None:
                raise ValueError(
                    "estimated_full_cell_tokens is required when "
                    "max_dense_attention_cost is set"
                )
            self._estimated_full_cell_tokens = [
                max(0, int(value)) for value in estimated_full_cell_tokens
            ]
            oversized = [
                value
                for value in self._estimated_full_cell_tokens
                if (value + 3) ** 2 > self.max_dense_attention_cost
            ]
            if oversized:
                raise ValueError(
                    "A single full-cell sample exceeds max_dense_attention_cost: "
                    f"max N_full={max(oversized)}"
                )
        else:
            self._estimated_full_cell_tokens = None

        n_batches = 0
        for bucket in self._bucket_groups:
            n = sum(group.numel() for group in bucket)
            n_batches += n // self.batch_size if self.drop_last else int(math.ceil(n / self.batch_size))
        self._base_len = n_batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def sampler(self) -> "ShardLocalAtomBucketBatchSampler":
        return self

    def __len__(self) -> int:
        if self.max_dense_attention_cost is None:
            return self._base_len
        return sum(1 for _ in self._iter_batches())

    def _would_exceed_token_cap(self, pending: Sequence[int], candidate: int) -> bool:
        if not pending:
            return False
        proposed = [*pending, int(candidate)]
        if (
            self.max_dense_attention_cost is not None
            and self._estimated_full_cell_tokens is not None
        ):
            max_full = max(
                self._estimated_full_cell_tokens[int(index)] for index in proposed
            )
            cost = len(proposed) * (max_full + 3) ** 2
            return cost > self.max_dense_attention_cost
        return False

    def _iter_batches(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        bucket_order = list(range(len(self._bucket_groups)))
        if self.shuffle:
            perm = torch.randperm(len(bucket_order), generator=g).tolist()
            bucket_order = [bucket_order[i] for i in perm]

        for bucket_idx in bucket_order:
            groups = self._bucket_groups[bucket_idx]
            group_order = list(range(len(groups)))
            if self.shuffle:
                perm = torch.randperm(len(group_order), generator=g).tolist()
                group_order = [group_order[i] for i in perm]

            pending: list[int] = []
            for group_idx in group_order:
                idx = groups[group_idx]
                if self.shuffle:
                    idx = idx[torch.randperm(idx.numel(), generator=g)]
                for value in idx.tolist():
                    if pending and self._would_exceed_token_cap(pending, value):
                        yield pending
                        pending = []
                    pending.append(int(value))
                    if len(pending) == self.batch_size:
                        yield pending
                        pending = []
            if pending and not self.drop_last:
                yield pending

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._iter_batches()


class WeightedAtomBucketBatchSampler(Sampler[list[int]]):

    def __init__(
        self,
        buckets: Sequence[Sequence[int]],
        weights: torch.Tensor,
        batch_size: int,
        num_samples: int,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        self.weights = weights.float().clone()
        if self.weights.dim() != 1:
            raise ValueError("weights must be 1-D")
        self._buckets: List[torch.Tensor] = []
        self._bucket_w: List[torch.Tensor] = []
        bw_list = []
        for b in buckets:
            if len(b) == 0:
                continue
            t = torch.tensor(list(b), dtype=torch.long)
            wsub = self.weights[t].clamp(min=0.0)
            if float(wsub.sum().item()) <= 0.0:
                wsub = torch.ones_like(wsub)
            self._buckets.append(t)
            self._bucket_w.append(wsub)
            bw_list.append(float(wsub.sum().item()))
        self._bucket_weight = torch.tensor(bw_list, dtype=torch.float32)
        if (self._bucket_weight <= 0).all():
            raise ValueError("All bucket weights are zero; check sampler config")
        self._bucket_weight = self._bucket_weight / self._bucket_weight.sum().clamp(min=1e-8)
        self._len = max(1, num_samples // batch_size) if drop_last else max(1, int(math.ceil(num_samples / batch_size)))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def sampler(self) -> "WeightedAtomBucketBatchSampler":
        return self

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        for _ in range(self._len):
            b = int(torch.multinomial(self._bucket_weight, 1, generator=g).item())
            ids = self._buckets[b]
            w = self._bucket_w[b]
            if ids.numel() == 0:
                continue
            repl = ids.numel() < self.batch_size
            sub = torch.multinomial(w, self.batch_size, replacement=repl, generator=g)
            yield ids[sub].tolist()
