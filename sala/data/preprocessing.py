from __future__ import annotations

import argparse
import csv
import io
import dataclasses
import json
import logging
import random
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Collection, Mapping, Optional, Sequence

import torch

from sala.core.constants import METAL_ATOMIC_NUMBERS
from sala.core.orbit import UnsupportedSpecialPositionError
from sala.data.datasets.cif_to_asu import cif_data_to_asu_input
from sala.data.io.cif_reader import (
    classify_chemical_formula_moiety,
    filter_hydrogens,
    read_cif,
)
from sala.data.datasets.quality import (
    MolecularCrystalQualityPolicy,
    assess_cif_quality,
    inclusive_complex_v1_policy,
)
from sala.data.hydrogenation_evidence import (
    HYDROGENATION_EVIDENCE_VERSION,
    build_hydrogenation_evidence,
)
from sala.data.schema import FULL_CELL_DATA_SCHEMA_VERSION

logger = logging.getLogger(__name__)

COORD_REPR_VERSION = FULL_CELL_DATA_SCHEMA_VERSION
GEOMETRY_VERSION = "row_lattice_v2"


def _build_split_map(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[int, str]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    split_map: dict[int, str] = {}
    for pos, idx in enumerate(indices):
        if pos < train_end:
            split_map[idx] = "train"
        elif pos < val_end:
            split_map[idx] = "val"
        else:
            split_map[idx] = "test"
    return split_map


def _process_cif_for_sharding(args: tuple) -> dict:
    (
        _index,
        cif_path,
        split,
        filter_h,
        skip_bad,
        quality_policy,
        max_cif_size_mb,
        max_heavy_cell_atoms,
    ) = args
    cif_path = Path(cif_path)
    try:
        if max_cif_size_mb is not None:
            size_mb = cif_path.stat().st_size / (1024 * 1024)
            if size_mb > float(max_cif_size_mb):
                return {
                    "status": "rejected",
                    "reject_reasons": ["cif_file_too_large"],
                    "reject_row": {
                        "cif_id": cif_path.stem,
                        "path": str(cif_path),
                        "reasons": ["cif_file_too_large"],
                        "metrics": {"file_size_mb": size_mb},
                    },
                }

        source_cif_data = read_cif(cif_path)
        cif_data = source_cif_data
        if filter_h:
            cif_data = filter_hydrogens(cif_data)

        n_heavy_atoms = int(len(cif_data.atom_types))
        n_symops = max(1, int(getattr(cif_data, "n_symops", 1)))
        n_heavy_cell = n_heavy_atoms * n_symops
        if (
            max_heavy_cell_atoms is not None
            and n_heavy_cell > int(max_heavy_cell_atoms)
        ):
            return {
                "status": "rejected",
                "reject_reasons": ["too_many_cell_heavy_atoms"],
                "reject_row": {
                    "cif_id": cif_data.cif_id,
                    "path": str(cif_path),
                    "split": split,
                    "reasons": ["too_many_cell_heavy_atoms"],
                    "metrics": {
                        "n_heavy_atoms": n_heavy_atoms,
                        "n_symops": n_symops,
                        "n_heavy_cell": n_heavy_cell,
                        "split_limit": int(max_heavy_cell_atoms),
                    },
                },
            }

        quality = None
        if quality_policy is not None:
            quality = assess_cif_quality(cif_data, quality_policy)
            if not quality.accepted:
                return {
                    "status": "rejected",
                    "reject_reasons": list(quality.reasons),
                    "reject_row": {
                        "cif_id": cif_data.cif_id,
                        "path": str(cif_path),
                        "reasons": list(quality.reasons),
                        "metrics": quality.metrics,
                    },
                }

        c_count = sum(1 for at in cif_data.atom_types if at == 6)
        class_label = classify_chemical_formula_moiety(cif_data.chemical_formula_moiety)
        spacegroup = int(cif_data.spacegroup)
        is_neutral = class_label != "salt"
        n_full_atoms = n_heavy_cell
        has_metal = any(at in METAL_ATOMIC_NUMBERS for at in cif_data.atom_types)

        asu = cif_data_to_asu_input(cif_data)
        asu = dataclasses.replace(
            asu,
            n_symops=torch.tensor([n_symops], dtype=torch.long),
            n_heavy_cell=torch.tensor([n_heavy_cell], dtype=torch.long),
            n_full_atoms=torch.tensor([n_full_atoms], dtype=torch.long),
        )
        hydrogenation_evidence = build_hydrogenation_evidence(
            source_cif_data, asu
        )
        n_components = int(asu.component_id.max().item()) + 1 if len(asu.component_id) > 0 else 0
        localized_atom_charge_count = int(asu.atom_charge_mask.sum().item())
        localized_nonzero_charge_count = int(
            ((asu.atom_formal_charge != 0) & asu.atom_charge_mask).sum().item()
        )
        unlocalized_component_count = 0
        for component_index in range(n_components):
            atom_mask = asu.component_id == component_index
            if atom_mask.any() and not bool(asu.atom_charge_mask[atom_mask].all()):
                unlocalized_component_count += 1
        n_component_types = int(torch.unique(asu.component_type_id).numel())
        chemistry_meta = {
            "localized_atom_charge_count": localized_atom_charge_count,
            "localized_nonzero_charge_count": localized_nonzero_charge_count,
            "unlocalized_component_count": unlocalized_component_count,
            "n_component_types": n_component_types,
        }
        item = {
            "asu": asu,
            "meta": {
                "cif_id": cif_data.cif_id,
                "c_count": c_count,
                "class": class_label,
                "split": split,
                "spacegroup": spacegroup,
                "is_neutral": is_neutral,
                "n_components": n_components,
                **chemistry_meta,
                "n_heavy_atoms": n_heavy_atoms,
                "n_symops": n_symops,
                "n_heavy_cell": n_heavy_cell,
                "n_full_atoms": n_full_atoms,
                "hall_number": int(getattr(cif_data, "hall_number", -1)),
                "setting_id": int(getattr(cif_data, "setting_id", -1)),
                "lattice_template_id": int(
                    getattr(asu, "lattice_template_id", torch.tensor([-1])).item()
                ),
                "has_metal": has_metal,
                "chemical_formula_moiety": cif_data.chemical_formula_moiety,
                "chemical_formula_sum": cif_data.chemical_formula_sum,
                "cell_formula_units_z": cif_data.cell_formula_units_z,
                "hydrogenation_evidence_version": HYDROGENATION_EVIDENCE_VERSION,
                "bond_order_confidence": hydrogenation_evidence[
                    "bond_order_confidence"
                ],
                "bond_order_authoritative_fraction": (
                    hydrogenation_evidence["authoritative_bond_count"]
                    / hydrogenation_evidence["heavy_bond_count"]
                    if hydrogenation_evidence["heavy_bond_count"]
                    else 0.0
                ),
                "has_disorder": hydrogenation_evidence["has_disorder"],
                "hydrogenation_evidence": hydrogenation_evidence,
                "quality_metrics": quality.metrics if quality is not None else None,
                "source_cif_path": str(cif_path),
                "coord_repr_version": COORD_REPR_VERSION,
                "geometry_version": GEOMETRY_VERSION,
            },
        }
        entry_base = {
            "cif_id": cif_data.cif_id,
            "c_count": c_count,
            "class": class_label,
            "split": split,
            "spacegroup": spacegroup,
            "is_neutral": is_neutral,
            "n_components": n_components,
            **chemistry_meta,
            "n_heavy_atoms": n_heavy_atoms,
            "n_symops": n_symops,
            "n_heavy_cell": n_heavy_cell,
            "n_full_atoms": n_full_atoms,
            "hall_number": int(getattr(cif_data, "hall_number", -1)),
            "setting_id": int(getattr(cif_data, "setting_id", -1)),
            "lattice_template_id": int(
                getattr(asu, "lattice_template_id", torch.tensor([-1])).item()
            ),
            "has_metal": has_metal,
            "chemical_formula_moiety": cif_data.chemical_formula_moiety,
            "chemical_formula_sum": cif_data.chemical_formula_sum,
            "cell_formula_units_z": cif_data.cell_formula_units_z,
            "hydrogenation_evidence_version": HYDROGENATION_EVIDENCE_VERSION,
            "bond_order_confidence": hydrogenation_evidence[
                "bond_order_confidence"
            ],
            "bond_order_authoritative_fraction": (
                hydrogenation_evidence["authoritative_bond_count"]
                / hydrogenation_evidence["heavy_bond_count"]
                if hydrogenation_evidence["heavy_bond_count"]
                else 0.0
            ),
            "has_disorder": hydrogenation_evidence["has_disorder"],
            "source_cif_path": str(cif_path),
            "quality_metrics": quality.metrics if quality is not None else None,
            "coord_repr_version": COORD_REPR_VERSION,
            "geometry_version": GEOMETRY_VERSION,
        }
        payload_buffer = io.BytesIO()
        torch.save(
            {
                "item": item,
                "entry_base": entry_base,
            },
            payload_buffer,
        )
        return {
            "status": "accepted",
            "payload": payload_buffer.getvalue(),
            "class_label": class_label,
            "source_cif_path": str(cif_path),
        }
    except UnsupportedSpecialPositionError as exc:
        if skip_bad:
            return {
                "status": "rejected",
                "reject_reasons": ["special_position_requires_stabilizer_metadata"],
                "reject_row": {
                    "cif_id": cif_path.stem,
                    "path": str(cif_path),
                    "reasons": ["special_position_requires_stabilizer_metadata"],
                    "error": str(exc),
                },
            }
        raise
    except ValueError as exc:
        if skip_bad:
            return {
                "status": "rejected",
                "reject_reasons": ["invalid_full_cell_crystallographic_contract"],
                "reject_row": {
                    "cif_id": cif_path.stem,
                    "path": str(cif_path),
                    "reasons": ["invalid_full_cell_crystallographic_contract"],
                    "error": str(exc),
                },
            }
        raise
    except Exception as exc:
        if skip_bad:
            return {
                "status": "skipped",
                "path": str(cif_path),
                "error": str(exc),
            }
        raise


def _iter_processed_results(worker_args: list[tuple], num_workers: int):
    if num_workers <= 1:
        for args in worker_args:
            yield _process_cif_for_sharding(args)
        return

    chunksize = max(1, min(128, len(worker_args) // max(1, num_workers * 4) or 1))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        yield from executor.map(_process_cif_for_sharding, worker_args, chunksize=chunksize)


def preprocess_to_shards(
    root: Path | Sequence[Path],
    output_dir: Path,
    shard_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    filter_h: bool,
    max_samples: Optional[int],
    skip_bad: bool,
    write_jsonl: bool,
    shard_suffix: str = "",
    quality_policy: Optional[MolecularCrystalQualityPolicy] = None,
    reject_jsonl: bool = True,
    accepted_cif_dir: Optional[Path] = None,
    max_cif_size_mb: Optional[float] = 2.0,
    num_workers: int = 1,
    split_by_cif_filename: Optional[Mapping[str, str]] = None,
    default_split: str = "train",
    excluded_cif_filenames: Optional[Collection[str]] = None,
    max_heavy_cell_atoms_by_split: Optional[Mapping[str, Optional[int]]] = None,
    quality_policy_name: Optional[str] = None,
) -> None:
    roots = [root] if isinstance(root, Path) else list(root)
    excluded = {
        _normalize_cif_filename(name) for name in (excluded_cif_filenames or ())
    }
    cif_files = _collect_cif_files(
        roots,
        max_samples=max_samples,
        excluded_cif_filenames=excluded,
    )

    if not cif_files:
        raise FileNotFoundError(f"No CIF files found in {roots}")

    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    num_workers = max(1, int(num_workers))

    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    explicit_split_map: dict[str, str] | None = None
    if split_by_cif_filename is not None:
        if default_split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported default split: {default_split!r}")
        explicit_split_map = {
            _normalize_cif_filename(name): str(split)
            for name, split in split_by_cif_filename.items()
        }
        invalid = sorted(
            {
                split
                for split in explicit_split_map.values()
                if split not in {"train", "val", "test"}
            }
        )
        if invalid:
            raise ValueError(f"Unsupported manifest split values: {invalid}")
        available = {path.name for path in cif_files}
        missing = sorted(set(explicit_split_map) - available)
        if missing:
            preview = ", ".join(missing[:10])
            raise FileNotFoundError(
                f"Split manifest contains {len(missing)} CIF files not found under roots: {preview}"
            )
        split_rank = {"train": 0, "val": 1, "test": 2}
        cif_files.sort(
            key=lambda path: (
                split_rank[explicit_split_map.get(path.name, default_split)],
                path.name,
            )
        )
        split_map = {
            index: explicit_split_map.get(path.name, default_split)
            for index, path in enumerate(cif_files)
        }
        split_mode = "manifest"
    else:
        split_map = _build_split_map(
            n=len(cif_files),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        split_mode = "random_ratio"

    split_limits = dict(max_heavy_cell_atoms_by_split or {})

    worker_args = [
        (
            i,
            cif_path,
            split_map[i],
            bool(filter_h),
            bool(skip_bad),
            quality_policy,
            max_cif_size_mb,
            split_limits.get(split_map[i]),
        )
        for i, cif_path in enumerate(cif_files)
    ]

    entries: list[dict] = []
    shard: list = []
    shard_split: str | None = None
    shard_id = 0
    skipped = 0
    rejected = 0
    class_counts: dict[str, int] = {}
    reject_rows: list[dict] = []
    reject_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    for processed_count, result in enumerate(
        _iter_processed_results(worker_args, num_workers),
        start=1,
    ):
        status = result["status"]
        if status == "skipped":
            skipped += 1
            logger.warning("Skip %s: %s", result.get("path"), result.get("error"))
        elif status == "rejected":
            rejected += 1
            for reason in result.get("reject_reasons", []):
                reject_counts[reason] += 1
            reject_row = result.get("reject_row")
            if reject_row is not None:
                reject_rows.append(reject_row)
        elif status == "accepted":
            class_label = result["class_label"]
            class_counts[class_label] = class_counts.get(class_label, 0) + 1

            payload = torch.load(
                io.BytesIO(result["payload"]),
                map_location="cpu",
                weights_only=False,
            )
            entry_split = str(payload["entry_base"]["split"])
            if (
                explicit_split_map is not None
                and shard
                and shard_split != entry_split
            ):
                shard_path = (
                    shards_dir / f"shard-{shard_id:05d}{shard_suffix}.pt"
                )
                torch.save(shard, shard_path)
                shard_id += 1
                shard = []
                shard_split = None
            if shard_split is None:
                shard_split = entry_split
            item_idx = len(shard)
            shard.append(payload["item"])

            entry = dict(payload["entry_base"])
            entry.update(
                {
                    "shard_id": shard_id,
                    "item_idx": item_idx,
                }
            )
            entries.append(entry)
            split_counts[entry["split"]] += 1

            if accepted_cif_dir is not None:
                _copy_accepted_cif(Path(result["source_cif_path"]), accepted_cif_dir)

            if len(shard) >= shard_size:
                shard_path = shards_dir / f"shard-{shard_id:05d}{shard_suffix}.pt"
                torch.save(shard, shard_path)
                shard_id += 1
                shard = []
                shard_split = None
        else:
            raise ValueError(f"Unsupported preprocessing result status: {status!r}")

        if processed_count % 1000 == 0:
            logger.info("Processed %d / %d", processed_count, len(cif_files))

    if shard:
        shard_path = shards_dir / f"shard-{shard_id:05d}{shard_suffix}.pt"
        torch.save(shard, shard_path)

    index_data = {
        "entries": entries,
        "meta": {
            "num_samples": len(entries),
            "num_skipped": skipped,
            "num_rejected": rejected,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "split_mode": split_mode,
            "split_counts": dict(split_counts),
            "default_split": default_split if explicit_split_map is not None else None,
            "manifest_split_counts": (
                dict(Counter(explicit_split_map.values()))
                if explicit_split_map is not None
                else None
            ),
            "excluded_manifest_count": len(excluded),
            "max_heavy_cell_atoms_by_split": split_limits,
            "seed": seed,
            "shard_size": shard_size,
            "filter_h": filter_h,
            "class_counts": class_counts,
            "reject_counts": dict(reject_counts),
            "quality_policy": (
                dataclasses.asdict(quality_policy)
                if quality_policy is not None
                else None
            ),
            "quality_policy_name": quality_policy_name,
            "max_cif_size_mb": max_cif_size_mb,
            "coord_repr_version": COORD_REPR_VERSION,
            "geometry_version": GEOMETRY_VERSION,
            "num_workers": num_workers,
        },
    }
    index_path = output_dir / f"index{shard_suffix}.pt"
    torch.save(index_data, index_path)

    if write_jsonl:
        jsonl_path = output_dir / f"index{shard_suffix}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    if reject_jsonl and reject_rows:
        reject_path = output_dir / f"rejected{shard_suffix}.jsonl"
        with open(reject_path, "w", encoding="utf-8") as f:
            for row in reject_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        logger.info("Rejected sample report: %s", reject_path)

    logger.info(
        "Preprocess complete: %d accepted, %d rejected, %d skipped",
        len(entries),
        rejected,
        skipped,
    )
    logger.info("Index: %s", index_path)


def _collect_cif_files(
    roots: Sequence[Path],
    max_samples: Optional[int],
    excluded_cif_filenames: Optional[Collection[str]] = None,
) -> list[Path]:
    excluded = set(excluded_cif_filenames or ())
    files: list[Path] = []
    for root in roots:
        files.extend(
            path
            for path in sorted(Path(root).glob("*.cif"))
            if path.name not in excluded
        )
        if max_samples is not None and len(files) >= max_samples:
            return files[:max_samples]
    return files


def _normalize_cif_filename(value: str) -> str:
    name = Path(str(value).strip()).name
    if not name:
        raise ValueError("Empty CIF filename in manifest")
    return name if name.lower().endswith(".cif") else f"{name}.cif"


def _read_cif_filename_manifest(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        field = (
            "cif_filename"
            if "cif_filename" in reader.fieldnames
            else reader.fieldnames[0]
        )
        values = {
            _normalize_cif_filename(row[field])
            for row in reader
            if row.get(field) and str(row[field]).strip()
        }
    if not values:
        raise ValueError(f"Manifest contains no CIF filenames: {path}")
    return values


def _copy_accepted_cif(cif_path: Path, accepted_cif_dir: Path) -> None:
    source_tag = cif_path.parent.name or "cif"
    target_dir = accepted_cif_dir / source_tag
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cif_path, target_dir / cif_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess CIF dataset into .pt shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        nargs="+",
        required=True,
        help="CIF directory or directories",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for shards and index",
    )
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-filter-h",
        action="store_true",
        help="Do not filter hydrogens",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-bad", action="store_true", help="Skip bad CIFs")
    parser.add_argument(
        "--write-jsonl",
        action="store_true",
        help="Also write index.jsonl",
    )
    parser.add_argument(
        "--shard-suffix",
        type=str,
        default="_fullcell_q",
        help="Suffix for shard filenames (e.g. _aug)",
    )
    parser.add_argument(
        "--single-shard",
        action="store_true",
        help="Output one .pt file: set shard_size = number of CIF files (ignores --shard-size)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes for CIF parsing/conversion; "
            "writing remains ordered in the parent process"
        ),
    )
    parser.add_argument(
        "--quality",
        choices=("high", "inclusive_complex_v1", "none"),
        default="high",
        help="Apply high-confidence molecular-crystal filtering before writing shards",
    )
    parser.add_argument("--min-carbon-atoms", type=int, default=None)
    parser.add_argument("--min-heavy-atoms", type=int, default=2)
    parser.add_argument("--max-heavy-atoms", type=int, default=100)
    parser.add_argument("--max-heavy-cell-atoms", type=int, default=500)
    parser.add_argument("--allow-metals", action="store_true")
    parser.add_argument("--allow-symmetry-bonds", action="store_true")
    parser.add_argument(
        "--max-cif-size-mb",
        type=float,
        default=2.0,
        help="Reject oversized CIF files before parsing; use null/negative to disable",
    )
    parser.add_argument(
        "--accepted-cif-dir",
        type=str,
        default=None,
        help="Optional directory to copy accepted high-confidence CIF files",
    )
    parser.add_argument(
        "--no-reject-jsonl",
        action="store_true",
        help="Do not write rejected*.jsonl quality report",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
        help="CSV whose CIF filenames are assigned to validation; all other files default to train",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=str,
        action="append",
        default=[],
        help="CSV of CIF filenames excluded before preprocessing; may be repeated",
    )
    parser.add_argument(
        "--train-max-heavy-cell-atoms",
        type=int,
        default=None,
        help="Optional full-cell atom limit applied only to the train split",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    roots = [Path(item) for item in args.root]
    excluded_filenames: set[str] = set()
    for manifest in args.exclude_manifest:
        excluded_filenames.update(_read_cif_filename_manifest(Path(manifest)))
    validation_filenames = (
        _read_cif_filename_manifest(Path(args.val_manifest))
        if args.val_manifest
        else set()
    )
    overlap = validation_filenames & excluded_filenames
    if overlap:
        preview = ", ".join(sorted(overlap)[:10])
        raise ValueError(f"Validation and exclusion manifests overlap: {preview}")
    cif_files = _collect_cif_files(
        roots,
        max_samples=args.max_samples,
        excluded_cif_filenames=excluded_filenames,
    )
    shard_size = args.shard_size
    if args.single_shard:
        if not cif_files:
            raise FileNotFoundError(f"No CIF files found in {roots}")
        shard_size = len(cif_files)

    quality_policy = None
    if args.quality == "high":
        quality_policy = MolecularCrystalQualityPolicy(
            min_heavy_atoms=args.min_heavy_atoms,
            max_heavy_atoms=args.max_heavy_atoms,
            min_carbon_atoms=(
                4 if args.min_carbon_atoms is None else args.min_carbon_atoms
            ),
            max_heavy_cell_atoms=args.max_heavy_cell_atoms,
            allow_metals=args.allow_metals,
            reject_symmetry_bonds=not args.allow_symmetry_bonds,
        )
    elif args.quality == "inclusive_complex_v1":
        quality_policy = inclusive_complex_v1_policy(
            min_heavy_atoms=args.min_heavy_atoms,
            max_heavy_atoms=args.max_heavy_atoms,
            min_carbon_atoms=(
                0 if args.min_carbon_atoms is None else args.min_carbon_atoms
            ),
            max_heavy_cell_atoms=None,
        )

    preprocess_to_shards(
        root=roots,
        output_dir=Path(args.output),
        shard_size=shard_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        filter_h=not args.no_filter_h,
        max_samples=args.max_samples,
        skip_bad=args.skip_bad,
        write_jsonl=args.write_jsonl,
        shard_suffix=args.shard_suffix,
        quality_policy=quality_policy,
        reject_jsonl=not args.no_reject_jsonl,
        accepted_cif_dir=Path(args.accepted_cif_dir) if args.accepted_cif_dir else None,
        max_cif_size_mb=args.max_cif_size_mb if args.max_cif_size_mb >= 0 else None,
        num_workers=args.num_workers,
        split_by_cif_filename=(
            {name: "val" for name in validation_filenames}
            if validation_filenames
            else None
        ),
        default_split="train",
        excluded_cif_filenames=excluded_filenames,
        max_heavy_cell_atoms_by_split=(
            {"train": args.train_max_heavy_cell_atoms}
            if args.train_max_heavy_cell_atoms is not None
            else None
        ),
        quality_policy_name=args.quality,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
