"""Graph-level selection and scatter maps for :class:`~sala.core.types.AsuBatch`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from sala.core.types import AsuBatch


@dataclass(frozen=True)
class GraphSubBatchMap:
    """Indices from a selected sub-batch back into its parent batch.

    Each index is ordered like the corresponding leading dimension in the
    returned sub-batch.  Consequently ``parent.index_select(0, index)`` yields
    sub-batch order, while the scatter helpers restore values to parent order.
    """

    graph_index: Tensor
    atom_index: Tensor
    component_index: Tensor
    atom_graph_index: Tensor
    component_graph_index: Tensor
    parent_num_graphs: int
    parent_num_atoms: int
    parent_num_components: int

    def __post_init__(self) -> None:
        sizes = {
            "graph": int(self.parent_num_graphs),
            "atom": int(self.parent_num_atoms),
            "component": int(self.parent_num_components),
        }
        if any(size < 0 for size in sizes.values()):
            raise ValueError("GraphSubBatchMap parent sizes must be non-negative")
        indices = {
            "graph": self.graph_index,
            "atom": self.atom_index,
            "component": self.component_index,
        }
        devices = {index.device for index in indices.values()}
        if len(devices) != 1:
            raise ValueError("GraphSubBatchMap indices must share one device")
        for name, index in indices.items():
            if index.ndim != 1 or index.dtype != torch.long:
                raise TypeError(
                    f"GraphSubBatchMap {name}_index must be a 1D torch.long tensor"
                )
            if index.numel() and (
                bool((index < 0).any().item())
                or bool((index >= sizes[name]).any().item())
            ):
                raise IndexError(
                    f"GraphSubBatchMap {name}_index contains an out-of-range value"
                )
            if torch.unique(index).numel() != index.numel():
                raise ValueError(
                    f"GraphSubBatchMap {name}_index must not contain duplicates"
                )
            object.__setattr__(self, f"{name}_index", index.detach().clone())
        owners = {
            "atom_graph_index": (self.atom_graph_index, self.atom_index.numel()),
            "component_graph_index": (
                self.component_graph_index,
                self.component_index.numel(),
            ),
        }
        for name, (owner, expected) in owners.items():
            if owner.device != self.graph_index.device:
                raise ValueError("GraphSubBatchMap indices must share one device")
            if (
                owner.ndim != 1
                or owner.dtype != torch.long
                or owner.numel() != expected
            ):
                raise TypeError(
                    f"GraphSubBatchMap {name} must be a 1D torch.long tensor "
                    f"with {expected} entries"
                )
            if owner.numel() and (
                bool((owner < 0).any().item())
                or bool((owner >= self.graph_index.numel()).any().item())
            ):
                raise IndexError(
                    f"GraphSubBatchMap {name} contains an out-of-range subgraph id"
                )
            object.__setattr__(self, name, owner.detach().clone())

    @property
    def num_graphs(self) -> int:
        return int(self.graph_index.numel())

    @property
    def num_atoms(self) -> int:
        return int(self.atom_index.numel())

    @property
    def num_components(self) -> int:
        return int(self.component_index.numel())

    def _select(
        self,
        values: Tensor,
        index: Tensor,
        parent_size: int,
        *,
        name: str,
    ) -> Tensor:
        if values.dim() == 0 or values.shape[0] != parent_size:
            raise ValueError(
                f"parent {name} values leading dimension must be {parent_size}, "
                f"got {tuple(values.shape)}"
            )
        return values.index_select(0, index.to(device=values.device))

    def select_graphs(self, values: Tensor) -> Tensor:
        return self._select(
            values, self.graph_index, self.parent_num_graphs, name="graph"
        )

    def select_atoms(self, values: Tensor) -> Tensor:
        return self._select(values, self.atom_index, self.parent_num_atoms, name="atom")

    def select_components(self, values: Tensor) -> Tensor:
        return self._select(
            values,
            self.component_index,
            self.parent_num_components,
            name="component",
        )

    def _scatter(
        self,
        values: Tensor,
        index: Tensor,
        parent_size: int,
        *,
        fill_value: float | int | bool = 0,
        name: str,
    ) -> Tensor:
        if values.dim() == 0:
            raise ValueError(
                f"{name} values must have a leading selected-item dimension"
            )
        if values.shape[0] != index.numel():
            raise ValueError(
                f"{name} values leading dimension must be {index.numel()}, "
                f"got {values.shape[0]}"
            )
        output = values.new_full((parent_size, *values.shape[1:]), fill_value)
        if index.numel():
            output.index_copy_(0, index.to(device=values.device), values)
        return output

    def _merge(
        self,
        parent: Tensor,
        values: Tensor,
        index: Tensor,
        parent_size: int,
        *,
        name: str,
    ) -> Tensor:
        if parent.dim() == 0 or parent.shape[0] != parent_size:
            raise ValueError(
                f"parent {name} values leading dimension must be {parent_size}, "
                f"got {tuple(parent.shape)}"
            )
        if values.shape != (index.numel(), *parent.shape[1:]):
            raise ValueError(
                f"selected {name} values must have shape "
                f"{(index.numel(), *parent.shape[1:])}, got {tuple(values.shape)}"
            )
        if values.device != parent.device or values.dtype != parent.dtype:
            raise ValueError(
                f"selected and parent {name} values must share device and dtype"
            )
        output = parent.clone()
        if index.numel():
            output.index_copy_(0, index.to(device=parent.device), values)
        return output

    def scatter_graphs(
        self, values: Tensor, *, fill_value: float | int | bool = 0
    ) -> Tensor:
        return self._scatter(
            values,
            self.graph_index,
            self.parent_num_graphs,
            fill_value=fill_value,
            name="graph",
        )

    def scatter_atoms(
        self, values: Tensor, *, fill_value: float | int | bool = 0
    ) -> Tensor:
        return self._scatter(
            values,
            self.atom_index,
            self.parent_num_atoms,
            fill_value=fill_value,
            name="atom",
        )

    def scatter_components(
        self, values: Tensor, *, fill_value: float | int | bool = 0
    ) -> Tensor:
        return self._scatter(
            values,
            self.component_index,
            self.parent_num_components,
            fill_value=fill_value,
            name="component",
        )

    def merge_graphs(self, parent: Tensor, values: Tensor) -> Tensor:
        return self._merge(
            parent,
            values,
            self.graph_index,
            self.parent_num_graphs,
            name="graph",
        )

    def merge_atoms(self, parent: Tensor, values: Tensor) -> Tensor:
        return self._merge(
            parent,
            values,
            self.atom_index,
            self.parent_num_atoms,
            name="atom",
        )

    def merge_components(self, parent: Tensor, values: Tensor) -> Tensor:
        return self._merge(
            parent,
            values,
            self.component_index,
            self.parent_num_components,
            name="component",
        )


def _require_leading_size(
    value: Optional[Tensor],
    expected: int,
    *,
    name: str,
) -> None:
    if value is None:
        return
    if value.dim() == 0 or value.shape[0] != expected:
        raise ValueError(
            f"batch.{name} must have leading dimension {expected}, got {tuple(value.shape)}"
        )


def _validate_batch_layout(batch: AsuBatch) -> tuple[torch.device, int, Tensor]:
    device = batch.atom_types.device
    batch_size = int(batch.batch_size)
    num_atoms = int(batch.total_atoms)

    tensor_fields = {
        "spacegroup": batch.spacegroup,
        "component_id": batch.component_id,
        "atom_formal_charge": batch.atom_formal_charge,
        "atom_charge_mask": batch.atom_charge_mask,
        "component_formal_charge": batch.component_formal_charge,
        "component_type_id": batch.component_type_id,
        "num_atoms": batch.num_atoms,
        "edge_index_conv": batch.edge_index_conv,
        "edge_attr_conv": batch.edge_attr_conv,
        "frac_coords_gt": batch.frac_coords_gt,
        "lattice_gt": batch.lattice_gt,
        "batch": batch.batch,
        "ptr": batch.ptr,
        "component_batch": batch.component_batch,
        "cart_coords_gt": batch.cart_coords_gt,
        "hybridization": batch.hybridization,
        "n_symops": batch.n_symops,
        "n_heavy_cell": batch.n_heavy_cell,
        "chem_group_id": batch.chem_group_id,
        "crystal_system_id": batch.crystal_system_id,
        "hall_number": batch.hall_number,
        "setting_id": batch.setting_id,
        "lattice_template_id": batch.lattice_template_id,
        "n_full_atoms": batch.n_full_atoms,
        "symmetry_ops": batch.symmetry_ops,
        "symmetry_mask": batch.symmetry_mask,
    }
    for name, value in tensor_fields.items():
        if value is not None and value.device != device:
            raise ValueError(
                f"All AsuBatch tensors must share device {device}; batch.{name} is on {value.device}"
            )

    if batch.num_atoms.dim() != 1:
        raise ValueError(
            f"batch.num_atoms must be 1D, got {tuple(batch.num_atoms.shape)}"
        )
    if batch.ptr.dim() != 1 or batch.ptr.shape[0] != batch_size + 1:
        raise ValueError(
            f"batch.ptr must have shape ({batch_size + 1},), got {tuple(batch.ptr.shape)}"
        )
    if batch.ptr.numel() == 0 or int(batch.ptr[0].item()) != 0:
        raise ValueError("batch.ptr must start at zero")
    ptr_counts = batch.ptr[1:].long() - batch.ptr[:-1].long()
    if not torch.equal(ptr_counts, batch.num_atoms.long()):
        raise ValueError("batch.ptr differences must equal batch.num_atoms")
    if int(batch.ptr[-1].item()) != num_atoms:
        raise ValueError("batch.ptr[-1] must equal the total atom count")
    if bool((batch.num_atoms < 0).any().item()):
        raise ValueError("batch.num_atoms must be non-negative")

    _require_leading_size(batch.atom_types, num_atoms, name="atom_types")
    _require_leading_size(batch.component_id, num_atoms, name="component_id")
    _require_leading_size(
        batch.atom_formal_charge, num_atoms, name="atom_formal_charge"
    )
    _require_leading_size(batch.atom_charge_mask, num_atoms, name="atom_charge_mask")
    _require_leading_size(batch.frac_coords_gt, num_atoms, name="frac_coords_gt")
    _require_leading_size(batch.batch, num_atoms, name="batch")
    _require_leading_size(batch.cart_coords_gt, num_atoms, name="cart_coords_gt")
    _require_leading_size(batch.hybridization, num_atoms, name="hybridization")

    _require_leading_size(batch.spacegroup, batch_size, name="spacegroup")
    _require_leading_size(batch.lattice_gt, batch_size, name="lattice_gt")
    _require_leading_size(batch.n_symops, batch_size, name="n_symops")
    _require_leading_size(batch.n_heavy_cell, batch_size, name="n_heavy_cell")
    _require_leading_size(batch.chem_group_id, batch_size, name="chem_group_id")
    _require_leading_size(batch.crystal_system_id, batch_size, name="crystal_system_id")
    _require_leading_size(batch.hall_number, batch_size, name="hall_number")
    _require_leading_size(batch.setting_id, batch_size, name="setting_id")
    _require_leading_size(
        batch.lattice_template_id, batch_size, name="lattice_template_id"
    )
    _require_leading_size(batch.n_full_atoms, batch_size, name="n_full_atoms")
    if batch.symmetry_ops is None:
        if batch.symmetry_mask is not None:
            raise ValueError("batch.symmetry_mask requires batch.symmetry_ops")
    else:
        if batch.symmetry_ops.dim() != 4 or batch.symmetry_ops.shape[0] != batch_size:
            raise ValueError(
                "batch.symmetry_ops must have shape (B, G_max, 4, 4), got "
                f"{tuple(batch.symmetry_ops.shape)}"
            )
        if batch.symmetry_ops.shape[2:] != (4, 4):
            raise ValueError("batch.symmetry_ops must contain 4x4 affine matrices")
        if batch.symmetry_mask is None:
            raise ValueError("batch.symmetry_ops requires batch.symmetry_mask")
        if batch.symmetry_mask.shape != batch.symmetry_ops.shape[:2]:
            raise ValueError(
                "batch.symmetry_mask must have shape (B, G_max) matching symmetry_ops"
            )
        if batch.symmetry_mask.dtype != torch.bool:
            raise TypeError("batch.symmetry_mask must have boolean dtype")

    expected_atom_batch = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=batch.batch.dtype),
        batch.num_atoms.long(),
    )
    if not torch.equal(batch.batch, expected_atom_batch):
        raise ValueError("batch.batch must be contiguous and agree with batch.ptr")

    if batch.edge_index_conv.dim() != 2 or batch.edge_index_conv.shape[0] != 2:
        raise ValueError(
            "batch.edge_index_conv must have shape (2, E), got "
            f"{tuple(batch.edge_index_conv.shape)}"
        )
    num_edges = int(batch.edge_index_conv.shape[1])
    _require_leading_size(batch.edge_attr_conv, num_edges, name="edge_attr_conv")
    if num_edges:
        if bool((batch.edge_index_conv < 0).any().item()) or bool(
            (batch.edge_index_conv >= num_atoms).any().item()
        ):
            raise ValueError(
                "batch.edge_index_conv contains an out-of-range atom index"
            )

    num_components = int(batch.component_batch.numel())
    if batch.component_batch.dim() != 1:
        raise ValueError("batch.component_batch must be 1D")
    if num_components:
        if bool((batch.component_batch < 0).any().item()) or bool(
            (batch.component_batch >= batch_size).any().item()
        ):
            raise ValueError(
                "batch.component_batch contains an out-of-range graph index"
            )
    component_counts = torch.bincount(
        batch.component_batch.long(), minlength=batch_size
    )
    expected_component_batch = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=batch.component_batch.dtype),
        component_counts,
    )
    if not torch.equal(batch.component_batch, expected_component_batch):
        raise ValueError(
            "batch.component_batch must be contiguous and grouped by graph"
        )
    if num_atoms:
        if bool((batch.component_id < 0).any().item()) or bool(
            (batch.component_id >= num_components).any().item()
        ):
            raise ValueError(
                "batch.component_id contains an out-of-range component index"
            )
        atom_component_graph = batch.component_batch.long().index_select(
            0, batch.component_id.long()
        )
        if not torch.equal(atom_component_graph, batch.batch.long()):
            raise ValueError(
                "batch.component_id and component_batch assign atoms to different graphs"
            )

    charge_size = int(batch.component_formal_charge.numel())
    if charge_size != num_components:
        raise ValueError(
            "batch.component_formal_charge must be per-component: "
            f"expected {num_components} values, got {charge_size}"
        )
    if batch.component_type_id.shape != (num_components,):
        raise ValueError(
            "batch.component_type_id must be per-component: "
            f"expected {num_components} values, got {tuple(batch.component_type_id.shape)}"
        )
    if batch.atom_charge_mask.dtype != torch.bool:
        raise TypeError("batch.atom_charge_mask must use torch.bool")
    if batch.cif_ids is not None and len(batch.cif_ids) != batch_size:
        raise ValueError(
            f"batch.cif_ids must contain {batch_size} entries, got {len(batch.cif_ids)}"
        )
    return device, num_components, component_counts


def _normalize_graph_index(
    graph_index: Tensor | Sequence[int] | Sequence[bool],
    *,
    batch_size: int,
    device: torch.device,
    allow_empty: bool,
) -> Tensor:
    if torch.is_tensor(graph_index):
        raw = graph_index.detach()
    else:
        values = list(graph_index)
        if not values:
            raw = torch.empty(0, dtype=torch.long)
        else:
            raw = torch.as_tensor(values)
    if raw.dim() != 1:
        raise ValueError(f"graph selection must be 1D, got shape {tuple(raw.shape)}")
    if raw.dtype == torch.bool:
        if raw.shape[0] != batch_size:
            raise ValueError(
                f"graph mask must have shape ({batch_size},), got {tuple(raw.shape)}"
            )
        selected = torch.nonzero(raw, as_tuple=False).view(-1).to(device=device)
    else:
        if raw.dtype.is_floating_point or raw.dtype.is_complex:
            raise TypeError("graph indices must use an integer dtype or a boolean mask")
        selected = raw.to(device=device, dtype=torch.long)
    if selected.numel() == 0:
        if not allow_empty:
            raise ValueError(
                "graph selection is empty; pass allow_empty=True to permit it"
            )
        return selected
    if bool((selected < 0).any().item()) or bool((selected >= batch_size).any().item()):
        bad = (
            selected[(selected < 0) | (selected >= batch_size)].detach().cpu().tolist()
        )
        raise IndexError(f"graph selection contains out-of-range ids: {bad[:8]}")
    if torch.unique(selected).numel() != selected.numel():
        raise ValueError("graph selection must not contain duplicate graph ids")
    return selected


def _ordered_indices_from_contiguous_groups(
    group_ptr: Tensor,
    selected_groups: Tensor,
) -> tuple[Tensor, Tensor]:
    ptr = group_ptr.long()
    starts = ptr.index_select(0, selected_groups)
    counts = ptr.index_select(0, selected_groups + 1) - starts
    total = int(counts.sum().item())
    if total == 0:
        return torch.empty(0, device=ptr.device, dtype=torch.long), counts
    selected_owner = torch.repeat_interleave(
        torch.arange(selected_groups.numel(), device=ptr.device, dtype=torch.long),
        counts,
    )
    selected_ptr = torch.zeros(
        selected_groups.numel() + 1, device=ptr.device, dtype=torch.long
    )
    selected_ptr[1:] = counts.cumsum(dim=0)
    position = torch.arange(total, device=ptr.device, dtype=torch.long)
    local_position = position - selected_ptr.index_select(0, selected_owner)
    return starts.index_select(0, selected_owner) + local_position, counts


def _select_optional(value: Optional[Tensor], index: Tensor) -> Optional[Tensor]:
    return None if value is None else value.index_select(0, index)


def select_graph_subbatch(
    batch: AsuBatch,
    graph_index: Tensor | Sequence[int] | Sequence[bool],
    *,
    allow_empty: bool = False,
) -> tuple[AsuBatch, GraphSubBatchMap]:
    """Select graphs from an ``AsuBatch`` and remap all flattened indices.

    Integer selections retain caller order. Boolean masks retain parent order.
    Duplicate and out-of-range graph ids are rejected because they make a
    one-to-one scatter map ambiguous.
    """

    device, parent_num_components, parent_component_counts = _validate_batch_layout(
        batch
    )
    selected_graphs = _normalize_graph_index(
        graph_index,
        batch_size=batch.batch_size,
        device=device,
        allow_empty=allow_empty,
    )
    atom_index, atom_counts = _ordered_indices_from_contiguous_groups(
        batch.ptr, selected_graphs
    )
    parent_component_ptr = torch.zeros(
        batch.batch_size + 1, device=device, dtype=torch.long
    )
    if batch.batch_size:
        parent_component_ptr[1:] = parent_component_counts.cumsum(dim=0)
    component_index, component_counts = _ordered_indices_from_contiguous_groups(
        parent_component_ptr, selected_graphs
    )

    new_num_graphs = int(selected_graphs.numel())
    new_num_atoms = int(atom_index.numel())
    new_num_components = int(component_index.numel())
    new_graph_ids = torch.arange(new_num_graphs, device=device, dtype=torch.long)
    new_atom_batch = torch.repeat_interleave(
        new_graph_ids.to(dtype=batch.batch.dtype), atom_counts
    )
    new_component_batch = torch.repeat_interleave(
        new_graph_ids.to(dtype=batch.component_batch.dtype), component_counts
    )
    new_ptr = batch.ptr.new_zeros(new_num_graphs + 1)
    if new_num_graphs:
        new_ptr[1:] = atom_counts.to(dtype=batch.ptr.dtype).cumsum(dim=0)

    component_lookup = torch.full(
        (parent_num_components,), -1, device=device, dtype=torch.long
    )
    if new_num_components:
        component_lookup[component_index] = torch.arange(
            new_num_components, device=device, dtype=torch.long
        )
    selected_component_id = batch.component_id.long().index_select(0, atom_index)
    remapped_component_id = component_lookup.index_select(0, selected_component_id)
    if bool((remapped_component_id < 0).any().item()):
        raise ValueError(
            "Selected atom refers to a component outside the selected graphs"
        )
    remapped_component_id = remapped_component_id.to(dtype=batch.component_id.dtype)

    atom_lookup = torch.full((batch.total_atoms,), -1, device=device, dtype=torch.long)
    if new_num_atoms:
        atom_lookup[atom_index] = torch.arange(
            new_num_atoms, device=device, dtype=torch.long
        )
    if batch.total_edges:
        remapped_edges = atom_lookup.index_select(
            0, batch.edge_index_conv.long().reshape(-1)
        ).view_as(batch.edge_index_conv)
        edge_mask = (remapped_edges >= 0).all(dim=0)
        new_edge_index = remapped_edges[:, edge_mask].to(
            dtype=batch.edge_index_conv.dtype
        )
        new_edge_attr = batch.edge_attr_conv[edge_mask]
    else:
        new_edge_index = batch.edge_index_conv.new_empty((2, 0))
        new_edge_attr = batch.edge_attr_conv[:0]

    new_component_formal_charge = batch.component_formal_charge.index_select(
        0, component_index
    )
    new_component_type_id = batch.component_type_id.index_select(0, component_index)

    new_cif_ids = (
        None
        if batch.cif_ids is None
        else [
            batch.cif_ids[int(index)]
            for index in selected_graphs.detach().cpu().tolist()
        ]
    )
    new_symmetry_ops = _select_optional(batch.symmetry_ops, selected_graphs)
    new_symmetry_mask = _select_optional(batch.symmetry_mask, selected_graphs)
    sub_batch = AsuBatch(
        atom_types=batch.atom_types.index_select(0, atom_index),
        spacegroup=batch.spacegroup.index_select(0, selected_graphs),
        component_id=remapped_component_id,
        atom_formal_charge=batch.atom_formal_charge.index_select(0, atom_index),
        atom_charge_mask=batch.atom_charge_mask.index_select(0, atom_index),
        component_formal_charge=new_component_formal_charge,
        component_type_id=new_component_type_id,
        num_atoms=atom_counts.to(dtype=batch.num_atoms.dtype),
        edge_index_conv=new_edge_index,
        edge_attr_conv=new_edge_attr,
        frac_coords_gt=batch.frac_coords_gt.index_select(0, atom_index),
        lattice_gt=batch.lattice_gt.index_select(0, selected_graphs),
        batch=new_atom_batch,
        ptr=new_ptr,
        component_batch=new_component_batch,
        cart_coords_gt=_select_optional(batch.cart_coords_gt, atom_index),
        hybridization=_select_optional(batch.hybridization, atom_index),
        cif_ids=new_cif_ids,
        n_symops=_select_optional(batch.n_symops, selected_graphs),
        n_heavy_cell=_select_optional(batch.n_heavy_cell, selected_graphs),
        chem_group_id=_select_optional(batch.chem_group_id, selected_graphs),
        crystal_system_id=_select_optional(batch.crystal_system_id, selected_graphs),
        hall_number=_select_optional(batch.hall_number, selected_graphs),
        setting_id=_select_optional(batch.setting_id, selected_graphs),
        lattice_template_id=_select_optional(
            batch.lattice_template_id, selected_graphs
        ),
        n_full_atoms=_select_optional(batch.n_full_atoms, selected_graphs),
        symmetry_ops=new_symmetry_ops,
        symmetry_mask=new_symmetry_mask,
    )
    index_map = GraphSubBatchMap(
        graph_index=selected_graphs,
        atom_index=atom_index,
        component_index=component_index,
        atom_graph_index=new_atom_batch.long(),
        component_graph_index=new_component_batch.long(),
        parent_num_graphs=batch.batch_size,
        parent_num_atoms=batch.total_atoms,
        parent_num_components=parent_num_components,
    )
    return sub_batch, index_map


__all__ = ["GraphSubBatchMap", "select_graph_subbatch"]
