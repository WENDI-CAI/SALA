"""PyTorch dataset utilities for loading canonicalized ASU shard samples."""

from __future__ import annotations

import dataclasses
import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from sala.core.types import AsuInput
from sala.core.lattice_manifold import (
    lattice_invariance_residual,
    lattice_template_from_hall_number,
    resolve_lattice_template,
)
from sala.core.symmetry import (
    canonicalize_affine_symmetry_ops,
    spacegroup_number_from_symmetry_ops,
    symmetry_ops_match_hall,
)
from sala.data.schema import FULL_CELL_DATA_SCHEMA_VERSION
from sala.data.datasets.augmentation import AsuSymmetryAugmenter
from sala.data.datasets.asu_canonicalize import canonicalize_asu_input
from sala.data.strata import add_strata_fields


def _ensure_sala_asu_input(asu) -> AsuInput:
    if not isinstance(asu, AsuInput):
        raise ValueError(
            "Full-cell shard payload must contain the current AsuInput type; "
            "regenerate the dataset with the supported data schema"
        )
    if asu.symmetry_ops is None:
        raise ValueError("Full-cell shard payload is missing explicit symmetry_ops")
    operations = canonicalize_affine_symmetry_ops(
        torch.as_tensor(asu.symmetry_ops)
    )
    return dataclasses.replace(asu, symmetry_ops=operations)


def _hydrate_asu_strata(asu: AsuInput, entry: dict) -> AsuInput:
    entry = add_strata_fields(entry)
    return dataclasses.replace(
        asu,
        chem_group_id=torch.tensor([int(entry.get("chem_group_id", -1))], dtype=torch.long),
        crystal_system_id=torch.tensor([int(entry.get("crystal_system_id", -1))], dtype=torch.long),
    )


def _sample_scalar(asu: AsuInput, name: str) -> int:
    value = getattr(asu, name, None)
    if value is None or not value.numel():
        raise ValueError(f"Full-cell shard payload is missing {name}")
    return int(value.view(-1)[0].item())


def _validate_full_cell_sample(asu: AsuInput, entry: dict) -> None:
    """Cross-check a loaded payload against its already validated index row."""
    spacegroup = _sample_scalar(asu, "spacegroup")
    num_atoms = int(asu.atom_types.shape[0])
    if _sample_scalar(asu, "num_atoms") != num_atoms:
        raise ValueError("Full-cell shard payload num_atoms disagrees with atom_types")
    if int(entry["spacegroup"]) != spacegroup or int(entry["n_heavy_atoms"]) != num_atoms:
        raise ValueError("Full-cell shard payload disagrees with its index identity/counts")

    if asu.symmetry_ops is None:
        raise ValueError("Full-cell shard payload is missing explicit symmetry_ops")
    operations = asu.symmetry_ops.float()
    if operations.ndim != 3 or operations.shape[-2:] != (4, 4):
        raise ValueError("Full-cell shard symmetry_ops must have shape [G, 4, 4]")
    actual_ops = int(operations.shape[0])
    n_symops = _sample_scalar(asu, "n_symops")
    expected_full = num_atoms * actual_ops
    values = {
        "n_symops": n_symops,
        "n_heavy_cell": _sample_scalar(asu, "n_heavy_cell"),
        "n_full_atoms": _sample_scalar(asu, "n_full_atoms"),
        "hall_number": _sample_scalar(asu, "hall_number"),
        "setting_id": _sample_scalar(asu, "setting_id"),
        "lattice_template_id": _sample_scalar(asu, "lattice_template_id"),
    }
    if n_symops != actual_ops:
        raise ValueError("Full-cell shard n_symops disagrees with explicit operations")
    if values["n_heavy_cell"] != expected_full or values["n_full_atoms"] != expected_full:
        raise ValueError("Full-cell shard payload has inconsistent full-cell atom counts")
    for name, value in values.items():
        if int(entry[name]) != value:
            raise ValueError(f"Full-cell shard payload {name} disagrees with its index")

    hall_spacegroup, hall_template = lattice_template_from_hall_number(
        values["hall_number"]
    )
    if hall_spacegroup != spacegroup or values["setting_id"] != values["hall_number"]:
        raise ValueError("Full-cell shard Hall/setting metadata is inconsistent")
    if values["lattice_template_id"] != hall_template:
        raise ValueError("Full-cell shard lattice template disagrees with its Hall setting")
    resolve_lattice_template(
        asu.spacegroup.long(), asu.lattice_template_id.long()
    )

    rotations = operations[:, :3, :3]
    residual = lattice_invariance_residual(asu.lattice_gt, rotations)
    if residual.numel() and float(residual.max().item()) > 1.0e-3:
        raise ValueError("Full-cell shard lattice is incompatible with symmetry_ops")
    if spacegroup_number_from_symmetry_ops(operations) != spacegroup:
        raise ValueError("Full-cell shard symmetry operations disagree with IT spacegroup")
    if not symmetry_ops_match_hall(operations, values["hall_number"]):
        raise ValueError("Full-cell shard symmetry operations disagree with Hall setting")


BOND_ORDER_VALUES = [1.0, 1.5, 2.0, 3.0]
NUM_BOND_TYPES = len(BOND_ORDER_VALUES)


def bond_order_to_idx(edge_attr: Tensor, tol: float = 0.25) -> Tensor:
    device = edge_attr.device
    bo_values = torch.tensor(BOND_ORDER_VALUES, device=device, dtype=edge_attr.dtype)
    diffs = torch.abs(edge_attr.unsqueeze(-1) - bo_values.unsqueeze(0))
    indices = torch.argmin(diffs, dim=-1)
    return indices


class AsuPtDataset(Dataset):
    """Dataset that filters index entries and loads ASU samples from shard files."""

    def __init__(
        self,
        root: str | Path,
        index_path: Optional[str | Path] = None,
        entries: Optional[Sequence[dict]] = None,
        split: Optional[str] = None,
        c_count_min: Optional[int] = None,
        c_count_max: Optional[int] = None,
        classes: Optional[Sequence[str]] = None,
        include_unknown: bool = True,
        spacegroups: Optional[Sequence[int]] = None,
        is_neutral: Optional[bool] = None,
        has_metal: Optional[bool] = None,
        n_heavy_atoms_min: Optional[int] = None,
        n_heavy_atoms_max: Optional[int] = None,
        n_full_atoms_min: Optional[int] = None,
        n_full_atoms_max: Optional[int] = None,
        n_components_min: Optional[int] = None,
        n_components_max: Optional[int] = None,
        cache_shards: int = 1,
        return_meta: bool = False,
        shard_suffix: str = "",
        augmenter: AsuSymmetryAugmenter | None = None,
    ) -> None:
        self.root = Path(root)
        self.shards_dir = self.root / "shards"
        self.shard_suffix = shard_suffix if shard_suffix else ""
        self.index_path = (
            Path(index_path)
            if index_path
            else self.root / f"index{self.shard_suffix}.pt"
        )

        self.split = split
        self.c_count_min = c_count_min
        self.c_count_max = c_count_max
        self.classes = [c.strip() for c in classes] if classes else None
        self.include_unknown = include_unknown
        self.spacegroups = (
            {int(spacegroup) for spacegroup in spacegroups}
            if spacegroups is not None
            else None
        )
        self.is_neutral = is_neutral
        self.has_metal = has_metal
        self.n_heavy_atoms_min = n_heavy_atoms_min
        self.n_heavy_atoms_max = n_heavy_atoms_max
        self.n_full_atoms_min = n_full_atoms_min
        self.n_full_atoms_max = n_full_atoms_max
        self.n_components_min = n_components_min
        self.n_components_max = n_components_max
        if cache_shards is None:
            self.cache_shards = 0
        else:
            self.cache_shards = max(0, int(cache_shards))
        self.return_meta = return_meta
        self.augmenter = augmenter
        self._entries = list(entries) if entries is not None else self._load_index_entries()
        self._validate_full_cell_entries(self._entries)
        self._filtered_entries = self._filter_entries(self._entries)
        self._shard_cache: OrderedDict[str, list] = OrderedDict()
        # Scientific schema checks include Hall reconstruction and metric
        # invariance.  Validate each physical payload once per worker instead
        # of repeating spglib work on every epoch access.
        self._validated_full_cell_payloads: set[tuple[str, int]] = set()

    @staticmethod
    def _validate_full_cell_entries(entries: list[dict]) -> None:
        for index, entry in enumerate(entries):
            version = entry.get("coord_repr_version")
            if version != FULL_CELL_DATA_SCHEMA_VERSION:
                raise ValueError(
                    "FullCellCrystalDiT requires regenerated data schema "
                    f"{FULL_CELL_DATA_SCHEMA_VERSION!r}; entry {index} has {version!r}"
                )
            required = (
                "spacegroup",
                "n_heavy_atoms",
                "n_symops",
                "n_heavy_cell",
                "n_full_atoms",
                "hall_number",
                "setting_id",
                "lattice_template_id",
            )
            missing = [name for name in required if entry.get(name) is None]
            if missing:
                raise ValueError(
                    f"Full-cell index entry {index} is missing {missing}; regenerate shards"
                )
            spacegroup = int(entry["spacegroup"])
            n_asu = int(entry["n_heavy_atoms"])
            n_symops = int(entry["n_symops"])
            expected_full = n_asu * n_symops
            if int(entry["n_full_atoms"]) != expected_full or int(
                entry["n_heavy_cell"]
            ) != expected_full:
                raise ValueError(
                    f"Full-cell index entry {index} has inconsistent atom counts; "
                    f"expected {expected_full}"
                )
            if int(entry["hall_number"]) < 1 or int(entry["setting_id"]) < 1:
                raise ValueError(
                    f"Full-cell index entry {index} lacks authoritative Hall/setting metadata"
                )
            if int(entry["lattice_template_id"]) < 0:
                raise ValueError(
                    f"Full-cell index entry {index} lacks lattice_template_id"
                )
            hall_spacegroup, hall_template = lattice_template_from_hall_number(
                int(entry["hall_number"])
            )
            if hall_spacegroup != spacegroup or int(entry["setting_id"]) != int(
                entry["hall_number"]
            ):
                raise ValueError(
                    f"Full-cell index entry {index} has inconsistent Hall/setting metadata"
                )
            if int(entry["lattice_template_id"]) != hall_template:
                raise ValueError(
                    f"Full-cell index entry {index} template disagrees with Hall setting"
                )

    def _load_index_entries(self) -> list[dict]:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        if self.index_path.suffix == ".jsonl":
            entries: list[dict] = []
            with open(self.index_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entries.append(json.loads(line))
            return entries

        # Dataset indices contain trusted project metadata dictionaries rather
        # than a tensor-only state dict, so pickle loading is intentionally
        # explicit.  Never use this loader for untrusted external files.
        data = torch.load(
            self.index_path, map_location="cpu", weights_only=False
        )
        if isinstance(data, dict):
            if "entries" in data:
                return data["entries"]
            if "index" in data:
                return data["index"]
            raise ValueError(f"Unknown index dict keys: {list(data.keys())}")
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported index format: {type(data)}")

    def _filter_entries(self, entries: list[dict]) -> list[dict]:
        filtered: list[dict] = []
        for entry in entries:
            if self.split and entry.get("split") != self.split:
                continue

            class_label = entry.get("class", "unknown")
            if class_label == "unknown" and not self.include_unknown:
                continue
            if self.classes and class_label not in self.classes:
                continue

            c_count = entry.get("c_count")
            if c_count is None:
                if self.c_count_min is not None or self.c_count_max is not None:
                    continue
            else:
                if self.c_count_min is not None and c_count < self.c_count_min:
                    continue
                if self.c_count_max is not None and c_count > self.c_count_max:
                    continue

            if self.spacegroups is not None:
                spacegroup = entry.get("spacegroup")
                if spacegroup is None or int(spacegroup) not in self.spacegroups:
                    continue

            if self.is_neutral is not None:
                entry_neutral = entry.get("is_neutral")
                if entry_neutral is None or bool(entry_neutral) != self.is_neutral:
                    continue

            if self.has_metal is not None:
                entry_has_metal = entry.get("has_metal")
                if entry_has_metal is None or bool(entry_has_metal) != self.has_metal:
                    continue

            n_heavy_atoms = entry.get("n_heavy_atoms")
            if n_heavy_atoms is not None:
                n_heavy_atoms = int(n_heavy_atoms)
                if (
                    self.n_heavy_atoms_min is not None
                    and n_heavy_atoms < self.n_heavy_atoms_min
                ):
                    continue
                if (
                    self.n_heavy_atoms_max is not None
                    and n_heavy_atoms > self.n_heavy_atoms_max
                ):
                    continue
            elif (
                self.n_heavy_atoms_min is not None
                or self.n_heavy_atoms_max is not None
            ):
                continue

            n_full_atoms = entry.get("n_full_atoms")
            if n_full_atoms is not None:
                n_full_atoms = int(n_full_atoms)
                if (
                    self.n_full_atoms_min is not None
                    and n_full_atoms < self.n_full_atoms_min
                ):
                    continue
                if (
                    self.n_full_atoms_max is not None
                    and n_full_atoms > self.n_full_atoms_max
                ):
                    continue
            elif self.n_full_atoms_min is not None or self.n_full_atoms_max is not None:
                continue

            n_components = entry.get("n_components")
            if n_components is not None:
                n_components = int(n_components)
                if (
                    self.n_components_min is not None
                    and n_components < self.n_components_min
                ):
                    continue
                if (
                    self.n_components_max is not None
                    and n_components > self.n_components_max
                ):
                    continue
            elif (
                self.n_components_min is not None
                or self.n_components_max is not None
            ):
                continue

            filtered.append(entry)

        return filtered

    def __len__(self) -> int:
        return len(self._filtered_entries)

    @property
    def raw_entries(self) -> list[dict]:
        return self._entries

    def get_entry_meta(self, idx: int) -> dict:
        return self._filtered_entries[idx]

    def _get_shard_path(self, entry: dict) -> Path:
        shard_path = entry.get("shard_path")
        if shard_path:
            path = Path(shard_path)
            return path if path.is_absolute() else (self.root / path)

        shard_id = entry.get("shard_id")
        if shard_id is None:
            raise KeyError("Index entry missing shard_id")
        return self.shards_dir / f"shard-{int(shard_id):05d}{self.shard_suffix}.pt"

    def _load_shard(self, shard_path: Path) -> list:
        cache_key = str(shard_path)
        if self.cache_shards > 0 and cache_key in self._shard_cache:
            self._shard_cache.move_to_end(cache_key)
            return self._shard_cache[cache_key]

        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if not isinstance(shard, list):
            raise ValueError(f"Shard must be a list, got {type(shard)}: {shard_path}")

        if self.cache_shards > 0:
            self._shard_cache[cache_key] = shard
            while len(self._shard_cache) > self.cache_shards:
                self._shard_cache.popitem(last=False)

        return shard

    def __getitem__(self, idx: int):
        entry = self._filtered_entries[idx]
        shard_path = self._get_shard_path(entry)
        shard = self._load_shard(shard_path)

        item_idx = entry.get("item_idx")
        if item_idx is None:
            raise KeyError("Index entry missing item_idx")

        i = int(item_idx)
        if i < 0 or i >= len(shard):
            raise IndexError(
                f"Index entry item_idx={i} out of range for shard (len={len(shard)}). "
                f"Shard: {shard_path}. "
                "Index file may not match current shards (e.g. index and shards from different runs). "
                "Regenerate index for this shard set or use the correct index_path/shard_suffix."
            )
        item = shard[i]
        if isinstance(item, AsuInput):
            asu = item
            meta = None
        elif isinstance(item, dict) and "asu" in item:
            asu = item["asu"]
            meta = item.get("meta")
        else:
            raise ValueError(f"Unsupported shard item format: {type(item)}")

        asu = _ensure_sala_asu_input(asu)
        asu = _hydrate_asu_strata(asu, entry)
        validation_key = (str(shard_path.resolve()), i)
        if validation_key not in self._validated_full_cell_payloads:
            _validate_full_cell_sample(asu, entry)
            self._validated_full_cell_payloads.add(validation_key)
        asu = canonicalize_asu_input(asu)
        if self.augmenter is not None:
            asu = self.augmenter(asu)

        if self.return_meta:
            # The index is the authoritative view of mutable dataset metadata
            # such as split assignment and remapped shard identity.  This is
            # especially important for zero-copy dataset overlays that reuse
            # existing shard payloads after a deterministic re-split.
            merged_meta = dict(meta or {})
            merged_meta.update(entry)
            return asu, merged_meta
        return asu


__all__ = [
    "BOND_ORDER_VALUES",
    "NUM_BOND_TYPES",
    "bond_order_to_idx",
    "AsuPtDataset",
]
