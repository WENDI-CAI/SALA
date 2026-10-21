from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sala.core.constants import METAL_ATOMIC_NUMBERS
from sala.core.graph import build_covalent_graph
from sala.core.orbit import (
    UnsupportedSpecialPositionError,
    validate_general_position_orbits,
)
from sala.core.symmetry import parse_symops
from sala.core.types import CifData


@dataclass(frozen=True)
class MolecularCrystalQualityPolicy:
    min_heavy_atoms: int = 2
    max_heavy_atoms: int = 100
    min_carbon_atoms: int = 4
    max_heavy_cell_atoms: int | None = 500
    allow_metals: bool = False
    allow_charged_monatomic_components: bool = False
    allow_neutral_monatomic_components: bool = False
    require_charge_balance: bool = False
    require_bonds: bool = True
    reject_symmetry_bonds: bool = True
    max_isolated_heavy_atom_fraction: float = 0.0
    min_lattice_volume: float | None = 20.0
    max_lattice_volume: float | None = 20000.0
    require_general_position_orbits: bool = True
    special_position_tolerance: float = 1.0e-5


@dataclass(frozen=True)
class QualityDecision:
    accepted: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict[str, Any] = field(default_factory=dict)


def assess_cif_quality(
    cif_data: CifData,
    policy: MolecularCrystalQualityPolicy | None = None,
) -> QualityDecision:
    policy = policy or MolecularCrystalQualityPolicy()
    reasons: list[str] = []

    atom_types = list(cif_data.atom_types)
    heavy_types = [z for z in atom_types if int(z) != 1]
    n_heavy = len(heavy_types)
    n_carbon = sum(1 for z in heavy_types if int(z) == 6)
    has_metal = any(int(z) in METAL_ATOMIC_NUMBERS for z in heavy_types)
    n_symops = max(1, int(getattr(cif_data, "n_symops", 1) or 1))
    n_heavy_cell = n_heavy * n_symops

    volume = _safe_lattice_volume(cif_data.lattice)
    if volume is None:
        reasons.append("invalid_lattice")
    else:
        if (
            policy.min_lattice_volume is not None
            and volume < policy.min_lattice_volume
        ):
            reasons.append("lattice_volume_too_small")
        if (
            policy.max_lattice_volume is not None
            and volume > policy.max_lattice_volume
        ):
            reasons.append("lattice_volume_too_large")

    if n_heavy < policy.min_heavy_atoms:
        reasons.append("too_few_heavy_atoms")
    if n_heavy > policy.max_heavy_atoms:
        reasons.append("too_many_heavy_atoms")
    if n_carbon < policy.min_carbon_atoms:
        reasons.append("too_few_carbons")
    if has_metal and not policy.allow_metals:
        reasons.append("contains_metal")
    if policy.max_heavy_cell_atoms is not None and n_heavy_cell > policy.max_heavy_cell_atoms:
        reasons.append("too_many_cell_heavy_atoms")

    if policy.reject_symmetry_bonds and int(getattr(cif_data, "num_symmetry_bonds", 0) or 0) > 0:
        reasons.append("symmetry_covalent_bonds")

    if policy.require_general_position_orbits and cif_data.symop_xyz:
        operations = []
        for operation in parse_symops(cif_data.symop_xyz):
            affine = torch.eye(4, dtype=torch.float64)
            affine[:3, :3] = torch.as_tensor(operation.R, dtype=torch.float64)
            affine[:3, 3] = torch.as_tensor(operation.t, dtype=torch.float64)
            operations.append(affine)
        try:
            validate_general_position_orbits(
                torch.as_tensor(cif_data.frac_coords, dtype=torch.float64),
                torch.stack(operations),
                tolerance=policy.special_position_tolerance,
            )
        except UnsupportedSpecialPositionError:
            reasons.append("special_position_requires_stabilizer_metadata")

    if policy.require_bonds and not cif_data.bonds:
        reasons.append("missing_bonds")

    isolated_fraction = 1.0
    n_components = n_heavy
    n_neutral_monatomic_components = n_heavy
    n_charged_monatomic_components = 0
    total_component_formal_charge = 0.0
    charge_balanced = True
    if cif_data.bonds:
        edge_index, component_id, _ = build_covalent_graph(
            cif_data.bonds, cif_data.atom_labels
        )
        del edge_index
        if component_id.numel() > 0:
            n_components = int(component_id.max().item()) + 1
            counts = torch.bincount(component_id.long(), minlength=n_components)
            isolated_mask = counts == 1
            isolated = int(isolated_mask.sum().item())
            if (
                policy.allow_charged_monatomic_components
                or policy.allow_neutral_monatomic_components
                or policy.require_charge_balance
            ):
                # Import locally so the generic quality module does not make the
                # CIF-to-ASU conversion path a module-level dependency.
                from sala.data.datasets.cif_to_asu import (
                    _build_component_formal_charge,
                )

                component_charge = _build_component_formal_charge(
                    cif_data=cif_data,
                    component_id=component_id,
                    num_components=n_components,
                ).to(dtype=torch.float64)
                charged_mask = component_charge.abs() > 1.0e-8
                n_charged_monatomic_components = int(
                    (isolated_mask & charged_mask).sum().item()
                )
                n_neutral_monatomic_components = int(
                    (isolated_mask & ~charged_mask).sum().item()
                )
                total_component_formal_charge = float(component_charge.sum().item())
                charge_balanced = abs(total_component_formal_charge) <= 1.0e-8
                if policy.allow_neutral_monatomic_components:
                    isolated = 0
                elif policy.allow_charged_monatomic_components:
                    isolated = n_neutral_monatomic_components
            else:
                n_neutral_monatomic_components = isolated
            isolated_fraction = isolated / max(1, int(counts.numel()))
            if isolated_fraction > policy.max_isolated_heavy_atom_fraction:
                reasons.append("isolated_heavy_atoms")
            if (
                policy.require_charge_balance
                and bool((component_charge.abs() > 1.0e-8).any())
                and not charge_balanced
            ):
                reasons.append("unbalanced_component_charge")

    metrics = {
        "n_heavy_atoms": n_heavy,
        "n_carbon": n_carbon,
        "n_symops": n_symops,
        "n_heavy_cell": n_heavy_cell,
        "has_metal": has_metal,
        "n_bonds": len(cif_data.bonds),
        "n_components": n_components,
        "isolated_heavy_atom_fraction": isolated_fraction,
        "n_neutral_monatomic_components": n_neutral_monatomic_components,
        "n_charged_monatomic_components": n_charged_monatomic_components,
        "total_component_formal_charge": total_component_formal_charge,
        "charge_balanced": charge_balanced,
        "lattice_volume": volume,
        "num_symmetry_bonds": int(getattr(cif_data, "num_symmetry_bonds", 0) or 0),
        "general_position_orbits": (
            "special_position_requires_stabilizer_metadata" not in reasons
        ),
    }
    return QualityDecision(
        accepted=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        metrics=metrics,
    )


def inclusive_complex_v1_policy(
    *,
    min_heavy_atoms: int = 2,
    max_heavy_atoms: int = 100,
    min_carbon_atoms: int = 0,
    max_heavy_cell_atoms: int | None = None,
) -> MolecularCrystalQualityPolicy:
    """Training-domain policy for metal complexes, salts, and cocrystals.

    Charged and neutral monatomic components are retained, no minimum carbon
    count is imposed, and samples with explicit component charge must be balanced.
    The upper lattice-volume bound is deliberately disabled because the fixed
    evaluation manifest contains valid large full cells; train-time full-cell
    limits are applied per split by the preprocessing pipeline.
    """

    return MolecularCrystalQualityPolicy(
        min_heavy_atoms=min_heavy_atoms,
        max_heavy_atoms=max_heavy_atoms,
        min_carbon_atoms=min_carbon_atoms,
        max_heavy_cell_atoms=max_heavy_cell_atoms,
        allow_metals=True,
        allow_charged_monatomic_components=True,
        allow_neutral_monatomic_components=True,
        require_charge_balance=True,
        require_bonds=True,
        reject_symmetry_bonds=True,
        max_isolated_heavy_atom_fraction=0.0,
        min_lattice_volume=20.0,
        max_lattice_volume=None,
        require_general_position_orbits=True,
    )


def _safe_lattice_volume(lattice: torch.Tensor) -> float | None:
    try:
        lattice = torch.as_tensor(lattice, dtype=torch.float64)
        if lattice.shape != (3, 3) or not torch.isfinite(lattice).all():
            return None
        volume = float(torch.abs(torch.det(lattice)).item())
    except Exception:
        return None
    if not torch.isfinite(torch.tensor(volume)):
        return None
    return volume


__all__ = [
    "MolecularCrystalQualityPolicy",
    "QualityDecision",
    "assess_cif_quality",
    "inclusive_complex_v1_policy",
]
