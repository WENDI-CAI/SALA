from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sala.postprocess.geometry.canonical import (
    CanonicalCrystal,
    prepare_canonical_crystal,
)
from sala.postprocess.geometry.component_topology import (
    ComponentTopology,
    build_component_topologies,
)
from sala.postprocess.geometry.periodic import PeriodicLattice, prepare_periodic_lattice
from sala.postprocess.geometry.symmetry_context import (
    SymmetryContext,
    resolve_symmetry_context,
)


@dataclass(frozen=True)
class GeometryEvaluationContext:
    """Immutable geometry shared by all metrics for one crystal structure."""

    asu: Any
    periodic: PeriodicLattice
    symmetry: SymmetryContext
    topologies: tuple[ComponentTopology, ...]
    canonical: CanonicalCrystal
    atom_scope: str
    dedup_tolerance_A: float


def build_geometry_context(
    asu: Any,
    *,
    meta: Mapping[str, Any],
    source_cif_root: str | Path | None,
    atom_scope: str = "heavy",
    max_symops: int | None = None,
    dedup_tolerance_A: float = 1e-3,
    symmetry_override: SymmetryContext | None = None,
) -> GeometryEvaluationContext:
    """Resolve all immutable geometry once for reuse across crystal metrics."""
    periodic = prepare_periodic_lattice(asu.lattice_gt)
    symmetry = resolve_symmetry_context(
        asu,
        meta=meta,
        source_cif_root=source_cif_root,
        max_symops=max_symops,
        override=symmetry_override,
    )
    topologies = build_component_topologies(asu, atom_scope=atom_scope)
    canonical = prepare_canonical_crystal(
        asu,
        atom_scope=atom_scope,
        max_symops=max_symops,
        supercell_radius=0,
        dedup_tolerance_A=dedup_tolerance_A,
        periodic=periodic,
        symmetry=symmetry,
        topologies=topologies,
    )
    return GeometryEvaluationContext(
        asu=asu,
        periodic=periodic,
        symmetry=symmetry,
        topologies=topologies,
        canonical=canonical,
        atom_scope=atom_scope,
        dedup_tolerance_A=float(dedup_tolerance_A),
    )


__all__ = ["GeometryEvaluationContext", "build_geometry_context"]
