from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from torch import Tensor

from sala.core.lattice import frac_to_cart_coords
from sala.core.symmetry import (
    compute_cell_image_offset_torch,
    extract_explicit_symmetry_ops,
)
from sala.core.types import AsuInput


@dataclass(frozen=True)
class AsuAugmentationConfig:
    enabled: bool = True
    probability: float = 0.5


class AsuSymmetryAugmenter:
    def __init__(
        self,
        probability: float = 0.5,
    ) -> None:
        self.probability = float(probability)

    def __call__(self, asu: AsuInput) -> AsuInput:
        if self.probability <= 0.0:
            return asu
        if self.probability < 1.0 and float(torch.rand(()).item()) > self.probability:
            return asu
        return augment_asu_input(asu)


def augment_asu_input(
    asu: AsuInput,
    *,
    rotation: Tensor | None = None,
    translation: Tensor | None = None,
    component_cell_shifts: Tensor | None = None,
) -> AsuInput:
    device = asu.frac_coords_gt.device
    dtype = asu.frac_coords_gt.dtype
    local_component_id, num_components = _local_component_ids(asu.component_id)
    if rotation is None or translation is None:
        rotation, translation = _sample_component_symmetry_operations(
            asu, num_components=num_components, device=device, dtype=dtype
        )
    else:
        rotation = rotation.to(device=device, dtype=dtype)
        translation = translation.to(device=device, dtype=dtype)


    if rotation.dim() == 2:
        raw_frac_coords = asu.frac_coords_gt @ rotation.T + translation
    else:
        raw_frac_coords = asu.frac_coords_gt.clone()
        for component_idx in range(num_components):
            atom_mask = local_component_id == component_idx
            if not bool(atom_mask.any().item()):
                continue
            raw_frac_coords[atom_mask] = asu.frac_coords_gt[atom_mask] @ rotation[component_idx].T + translation[component_idx]
    if component_cell_shifts is None:
        component_cell_shifts = _deterministic_component_recentering_shifts(
            raw_frac_coords=raw_frac_coords,
            local_component_id=local_component_id,
            num_components=num_components,
        )
    else:
        component_cell_shifts = component_cell_shifts.to(device=device, dtype=dtype)
        if component_cell_shifts.shape != (num_components, 3):
            raise ValueError(
                f"component_cell_shifts must have shape ({num_components}, 3), "
                f"got {tuple(component_cell_shifts.shape)}"
            )

    shifts = component_cell_shifts[local_component_id] if num_components else asu.frac_coords_gt.new_zeros(asu.frac_coords_gt.shape)
    frac_coords = raw_frac_coords + shifts
    cart_coords = frac_to_cart_coords(frac_coords, asu.lattice_gt)
    return dataclasses.replace(
        asu,
        frac_coords_gt=frac_coords,
        cart_coords_gt=cart_coords,
    )


def _sample_symmetry_operation(
    asu: AsuInput,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    ops = _asu_symmetry_operations(asu, device=device, dtype=dtype)
    op_idx = int(torch.randint(0, int(ops.shape[0]), ()).item())
    return ops[op_idx, :3, :3], ops[op_idx, :3, 3]


def _asu_symmetry_operations(
    asu: AsuInput,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    explicit = extract_explicit_symmetry_ops(asu)
    if explicit is None:
        raise ValueError("ASU symmetry augmentation requires explicit symmetry_ops")
    return explicit.to(device=device, dtype=dtype)


def _sample_component_symmetry_operations(
    asu: AsuInput,
    *,
    num_components: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if num_components <= 0:
        return (
            torch.zeros((0, 3, 3), device=device, dtype=dtype),
            torch.zeros((0, 3), device=device, dtype=dtype),
        )
    ops = _asu_symmetry_operations(asu, device=device, dtype=dtype)
    op_idx = torch.randint(0, int(ops.shape[0]), (num_components,), device=device)
    selected = ops.index_select(0, op_idx.long())
    return selected[:, :3, :3], selected[:, :3, 3]


def _deterministic_component_recentering_shifts(
    *,
    raw_frac_coords: Tensor,
    local_component_id: Tensor,
    num_components: int,
) -> Tensor:
    if num_components <= 0:
        return raw_frac_coords.new_zeros((0, 3))

    shifts = raw_frac_coords.new_zeros((num_components, 3))
    for component_idx in range(num_components):
        atom_mask = local_component_id == component_idx
        if not bool(atom_mask.any().item()):
            continue
        offset = compute_cell_image_offset_torch(raw_frac_coords[atom_mask])
        shifts[component_idx] = -offset
    return shifts


def _local_component_ids(component_id: Tensor) -> tuple[Tensor, int]:
    if component_id.numel() == 0:
        return component_id.new_zeros((0,), dtype=torch.long), 0
    _, inverse = torch.unique(component_id.to(torch.long), sorted=True, return_inverse=True)
    return inverse, int(inverse.max().item()) + 1


__all__ = [
    "AsuAugmentationConfig",
    "AsuSymmetryAugmenter",
    "augment_asu_input",
]
