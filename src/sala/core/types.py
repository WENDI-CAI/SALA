"""Tensor batch container for the standalone SALA velocity field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch import Tensor


@dataclass
class AsuBatch:

    atom_types: Tensor
    spacegroup: Tensor
    component_id: Tensor
    atom_formal_charge: Tensor
    atom_charge_mask: Tensor
    component_formal_charge: Tensor
    component_type_id: Tensor
    num_atoms: Tensor
    edge_index_conv: Tensor
    edge_attr_conv: Tensor
    frac_coords_gt: Tensor
    lattice_gt: Tensor


    batch: Tensor
    ptr: Tensor
    component_batch: Tensor
    cart_coords_gt: Optional[Tensor] = None
    hybridization: Optional[Tensor] = None


    cif_ids: Optional[list[str]] = None
    n_symops: Optional[Tensor] = None
    n_heavy_cell: Optional[Tensor] = None
    chem_group_id: Optional[Tensor] = None
    crystal_system_id: Optional[Tensor] = None
    hall_number: Optional[Tensor] = None
    setting_id: Optional[Tensor] = None
    lattice_template_id: Optional[Tensor] = None
    n_full_atoms: Optional[Tensor] = None
    # Ragged source-CIF operations are padded to [B, G_max, 4, 4].  The
    # corresponding [B, G_max] mask is authoritative. Every graph in a
    # full-cell batch must have at least one explicit operation.
    symmetry_ops: Optional[Tensor] = None
    symmetry_mask: Optional[Tensor] = None
    # CPU DataLoader workers may prebuild the static OrbitPlan so the GPU hot
    # path never has to synchronize metadata through .item()/.cpu().
    orbit_plan: Optional[Any] = None

    def _map_tensors(self, transform: Callable[[Tensor], Tensor]) -> "AsuBatch":
        orbit_plan = getattr(self, "orbit_plan", None)
        if orbit_plan is not None:
            orbit_plan = type(orbit_plan)(
                **{
                    name: transform(getattr(orbit_plan, name))
                    for name in orbit_plan.__dataclass_fields__
                }
            )
        return AsuBatch(
            atom_types=transform(self.atom_types),
            spacegroup=transform(self.spacegroup),
            component_id=transform(self.component_id),
            atom_formal_charge=transform(self.atom_formal_charge),
            atom_charge_mask=transform(self.atom_charge_mask),
            component_formal_charge=transform(self.component_formal_charge),
            component_type_id=transform(self.component_type_id),
            num_atoms=transform(self.num_atoms),
            edge_index_conv=transform(self.edge_index_conv),
            edge_attr_conv=transform(self.edge_attr_conv),
            frac_coords_gt=transform(self.frac_coords_gt),
            lattice_gt=transform(self.lattice_gt),
            batch=transform(self.batch),
            ptr=transform(self.ptr),
            component_batch=transform(self.component_batch),
            cart_coords_gt=(
                transform(self.cart_coords_gt)
                if self.cart_coords_gt is not None
                else None
            ),
            hybridization=(
                transform(self.hybridization)
                if self.hybridization is not None
                else None
            ),
            cif_ids=self.cif_ids,
            n_symops=transform(self.n_symops) if self.n_symops is not None else None,
            n_heavy_cell=(
                transform(self.n_heavy_cell)
                if self.n_heavy_cell is not None
                else None
            ),
            chem_group_id=(
                transform(self.chem_group_id)
                if self.chem_group_id is not None
                else None
            ),
            crystal_system_id=(
                transform(self.crystal_system_id)
                if self.crystal_system_id is not None
                else None
            ),
            hall_number=(
                transform(self.hall_number)
                if self.hall_number is not None
                else None
            ),
            setting_id=(
                transform(self.setting_id)
                if self.setting_id is not None
                else None
            ),
            lattice_template_id=(
                transform(self.lattice_template_id)
                if self.lattice_template_id is not None
                else None
            ),
            n_full_atoms=(
                transform(self.n_full_atoms)
                if self.n_full_atoms is not None
                else None
            ),
            symmetry_ops=(
                transform(self.symmetry_ops)
                if getattr(self, "symmetry_ops", None) is not None
                else None
            ),
            symmetry_mask=(
                transform(self.symmetry_mask)
                if getattr(self, "symmetry_mask", None) is not None
                else None
            ),
            orbit_plan=orbit_plan,
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "AsuBatch":
        return self._map_tensors(
            lambda value: value.to(device=device, non_blocking=non_blocking)
        )

    def pin_memory(self) -> "AsuBatch":
        """Return a pinned copy suitable for asynchronous CUDA transfer."""
        return self._map_tensors(lambda value: value.pin_memory())

    @property
    def batch_size(self) -> int:
        return len(self.num_atoms)

    @property
    def total_atoms(self) -> int:
        return len(self.atom_types)

    @property
    def total_edges(self) -> int:
        return self.edge_index_conv.shape[1]

    @property
    def total_components(self) -> int:
        return int(self.component_id.max().item()) + 1 if len(self.component_id) > 0 else 0


__all__ = ["AsuBatch"]
