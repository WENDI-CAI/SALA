from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np
import torch

if TYPE_CHECKING:
    from sala.postprocess.geometry.context import GeometryEvaluationContext


CONTACT_FLOOR_SCHEMA = "sala-contact-floor-v3"
CONTACT_FLOOR_ALGORITHM = "geometry-v3"
CONTACT_FLOOR_MANIFEST_NAME = "contact_floor_manifest.json"
DEFAULT_UNKNOWN_VDW_RADIUS_A = 1.80

_QUANTILES = {
    "q001": 0.001,
    "q005": 0.005,
    "q01": 0.01,
    "q05": 0.05,
    "q10": 0.10,
    "q25": 0.25,
    "q50": 0.50,
    "q75": 0.75,
    "q90": 0.90,
    "q95": 0.95,
    "q99": 0.99,
}


class ContactFloorArtifactError(ValueError):
    """Raised when an empirical contact-floor artifact is absent or inconsistent."""


@dataclass(frozen=True)
class ElementPairContact:
    pair: str
    distance_A: float
    target_id: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contact_floor_manifest_path(csv_path: str | Path) -> Path:
    return Path(csv_path).resolve().parent / CONTACT_FLOOR_MANIFEST_NAME


def count_column_for_quantile(quantile_column: str) -> str:
    for suffix in _QUANTILES:
        marker = f"_{suffix}"
        if str(quantile_column).endswith(marker):
            return str(quantile_column)[: -len(marker)] + "_n"
    raise ContactFloorArtifactError(
        f"Unsupported contact-floor quantile column: {quantile_column!r}"
    )


def write_contact_floor_manifest(
    csv_path: str | Path,
    *,
    split: str = "train",
    quantile_column: str = "nearest_dist_q01",
    min_count: int = 100,
    margin_A: float = 0.05,
    cutoff_A: float = 5.0,
    extra: Mapping[str, object] | None = None,
) -> Path:
    csv_file = Path(csv_path).resolve()
    if not csv_file.is_file():
        raise ContactFloorArtifactError(f"Contact-floor CSV does not exist: {csv_file}")
    manifest = {
        "schema": CONTACT_FLOOR_SCHEMA,
        "algorithm": CONTACT_FLOOR_ALGORITHM,
        "csv_file": csv_file.name,
        "csv_sha256": _sha256(csv_file),
        "aggregation": "per_structure_minimum",
        "distance_scope": "intermolecular_physical_unordered_pairs",
        "cutoff_A": float(cutoff_A),
        "active": {
            "split": str(split),
            "quantile_column": str(quantile_column),
            "count_column": count_column_for_quantile(quantile_column),
            "min_count": int(min_count),
            "margin_A": float(margin_A),
        },
    }
    if extra:
        manifest["metadata"] = dict(extra)
    path = contact_floor_manifest_path(csv_file)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def validate_contact_floor_artifact(
    csv_path: str | Path,
    *,
    split: str,
    quantile_column: str,
    min_count: int,
    margin_A: float,
) -> dict[str, object]:
    csv_file = Path(csv_path).resolve()
    if not csv_file.is_file():
        raise ContactFloorArtifactError(f"Contact-floor CSV does not exist: {csv_file}")
    manifest_file = contact_floor_manifest_path(csv_file)
    if not manifest_file.is_file():
        raise ContactFloorArtifactError(
            f"Contact-floor manifest does not exist: {manifest_file}"
        )
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContactFloorArtifactError(
            f"Invalid contact-floor manifest JSON: {manifest_file}"
        ) from exc
    if manifest.get("schema") != CONTACT_FLOOR_SCHEMA:
        raise ContactFloorArtifactError("Contact-floor manifest schema mismatch.")
    if manifest.get("algorithm") != CONTACT_FLOOR_ALGORITHM:
        raise ContactFloorArtifactError("Contact-floor manifest algorithm mismatch.")
    if manifest.get("csv_file") != csv_file.name:
        raise ContactFloorArtifactError("Contact-floor manifest CSV filename mismatch.")
    if manifest.get("csv_sha256") != _sha256(csv_file):
        raise ContactFloorArtifactError("Contact-floor CSV SHA-256 mismatch.")
    if manifest.get("aggregation") != "per_structure_minimum":
        raise ContactFloorArtifactError("Contact-floor aggregation is not per-structure minimum.")
    if manifest.get("distance_scope") != "intermolecular_physical_unordered_pairs":
        raise ContactFloorArtifactError("Contact-floor physical-contact scope mismatch.")
    active = manifest.get("active")
    if not isinstance(active, dict):
        raise ContactFloorArtifactError("Contact-floor manifest has no active settings.")
    expected = {
        "split": str(split),
        "quantile_column": str(quantile_column),
        "count_column": count_column_for_quantile(quantile_column),
        "min_count": int(min_count),
    }
    for key, value in expected.items():
        if active.get(key) != value:
            raise ContactFloorArtifactError(
                f"Contact-floor manifest {key} mismatch: {active.get(key)!r} != {value!r}"
            )
    try:
        manifest_margin = float(active.get("margin_A"))
    except Exception as exc:
        raise ContactFloorArtifactError("Contact-floor manifest margin_A is invalid.") from exc
    if not math.isclose(manifest_margin, float(margin_A), rel_tol=0.0, abs_tol=1e-12):
        raise ContactFloorArtifactError(
            f"Contact-floor manifest margin_A mismatch: {manifest_margin} != {margin_A}"
        )
    return manifest


@lru_cache(maxsize=1)
def element_vdw_radii() -> tuple[tuple[float, ...], tuple[bool, ...]]:
    radii = [DEFAULT_UNKNOWN_VDW_RADIUS_A] * 119
    known = [False] * 119
    try:
        from pymatgen.core import Element
    except Exception:
        return tuple(radii), tuple(known)
    for atomic_number in range(1, 119):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                radius = Element.from_Z(atomic_number).van_der_waals_radius
            value = float(radius) if radius is not None else math.nan
        except Exception:
            value = math.nan
        if math.isfinite(value) and value > 0.0:
            radii[atomic_number] = value
            known[atomic_number] = True
    return tuple(radii), tuple(known)


@lru_cache(maxsize=118)
def element_symbol(atomic_number: int) -> str:
    try:
        from pymatgen.core import Element

        return str(Element.from_Z(int(atomic_number)).symbol)
    except Exception:
        return f"Z{int(atomic_number)}"


def element_pair_key(left: int, right: int) -> str:
    first, second = sorted((int(left), int(right)))
    return f"{element_symbol(first)}-{element_symbol(second)}"


def nearest_inter_contacts_by_element_pair(
    context: GeometryEvaluationContext,
    *,
    cutoff_A: float,
) -> dict[str, float]:
    """Return one exact nearest intermolecular distance per element pair."""
    cutoff = float(cutoff_A)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff_A must be finite and positive")
    from sala.postprocess.geometry.clash import (
        enumerate_physical_inter_component_pairs,
    )

    crystal = context.canonical
    search = enumerate_physical_inter_component_pairs(
        crystal,
        cutoff_A=cutoff,
        orbit_representative_left=True,
    )
    if not search.complete:
        raise RuntimeError("Physical intermolecular contact search was incomplete.")
    nearest: dict[str, float] = {}
    for record in search.pairs:
        left = crystal.unit_components[record.left_index]
        right = crystal.unit_components[record.right_index]
        shift = torch.tensor(
            record.relative_shift,
            dtype=right.cart_coords.dtype,
            device=right.cart_coords.device,
        )
        lattice = crystal.lattice.to(
            dtype=right.cart_coords.dtype,
            device=right.cart_coords.device,
        )
        right_cart = right.cart_coords + shift @ lattice
        distances = torch.cdist(left.cart_coords.float(), right_cart.float())
        active = torch.nonzero(distances <= cutoff + 1e-7, as_tuple=False)
        for left_index, right_index in active.detach().cpu().tolist():
            pair = element_pair_key(
                int(left.atom_types[left_index].item()),
                int(right.atom_types[right_index].item()),
            )
            distance = float(distances[left_index, right_index].item())
            nearest[pair] = min(nearest.get(pair, math.inf), distance)
    return nearest


def summarize_contact_minima(
    contacts: Iterable[ElementPairContact],
    *,
    split: str,
) -> list[dict[str, object]]:
    """Aggregate exact per-target minima before computing element-pair quantiles."""
    per_target: dict[tuple[str, str], float] = {}
    for contact in contacts:
        distance = float(contact.distance_A)
        if not math.isfinite(distance) or distance < 0.0:
            continue
        key = (str(contact.target_id), str(contact.pair))
        per_target[key] = min(per_target.get(key, math.inf), distance)
    by_pair: dict[str, list[float]] = {}
    for (_, pair), distance in per_target.items():
        by_pair.setdefault(pair, []).append(float(distance))
    rows: list[dict[str, object]] = []
    for pair in sorted(by_pair):
        values = np.asarray(by_pair[pair], dtype=np.float64)
        row: dict[str, object] = {
            "split": str(split),
            "pair": pair,
            "nearest_dist_n": int(values.size),
            "nearest_dist_min": float(values.min()),
            "nearest_dist_mean": float(values.mean()),
        }
        row.update(
            {
                f"nearest_dist_{name}": float(np.quantile(values, quantile))
                for name, quantile in _QUANTILES.items()
            }
        )
        rows.append(row)
    return rows


__all__ = [
    "CONTACT_FLOOR_ALGORITHM",
    "CONTACT_FLOOR_MANIFEST_NAME",
    "CONTACT_FLOOR_SCHEMA",
    "ContactFloorArtifactError",
    "ElementPairContact",
    "contact_floor_manifest_path",
    "count_column_for_quantile",
    "element_pair_key",
    "element_vdw_radii",
    "nearest_inter_contacts_by_element_pair",
    "summarize_contact_minima",
    "validate_contact_floor_artifact",
    "write_contact_floor_manifest",
]
