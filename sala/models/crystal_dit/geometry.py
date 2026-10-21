"""Shared per-denoiser-call crystal geometry state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from sala.core.orbit import FullCellGeometry, OrbitPlan


@dataclass(frozen=True)
class CrystalGeometryState:
    lattice: Tensor
    lattice_inverse: Tensor
    reciprocal: Tensor
    lengths: Tensor
    log_volume: Tensor
    full: FullCellGeometry


def build_geometry_state(
    *,
    plan: OrbitPlan,
    asu_cart: Tensor,
    lattice: Tensor,
) -> CrystalGeometryState:
    work_lattice = lattice.float()
    lattice_inverse = torch.linalg.inv(work_lattice)
    # Reciprocal rows are columns of A^{-1} for row-vector cart = frac @ A.
    reciprocal = lattice_inverse.transpose(-1, -2)
    lengths = torch.linalg.vector_norm(work_lattice, dim=-1, keepdim=True)
    log_volume = torch.logdet(work_lattice).unsqueeze(-1).unsqueeze(-1)
    full = plan.expand_coordinates(
        asu_cart.float(), work_lattice, lattice_inverse=lattice_inverse
    )
    return CrystalGeometryState(
        lattice=work_lattice,
        lattice_inverse=lattice_inverse,
        reciprocal=reciprocal,
        lengths=lengths,
        log_volume=log_volume,
        full=full,
    )


__all__ = ["CrystalGeometryState", "build_geometry_state"]
