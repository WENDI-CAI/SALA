
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from sala.core.lattice import frac_to_cart_coords
from sala.core.symmetry import extract_explicit_symmetry_ops
from sala.core.types import AsuBatch, AsuInput


def _pack_symmetry_operations(
    operation_sets: Sequence[Tensor | None],
    *,
    reference: Tensor,
) -> tuple[Tensor | None, Tensor | None]:
    available = [operations for operations in operation_sets if operations is not None]
    if not available:
        return None, None

    max_operations = max(int(operations.shape[0]) for operations in available)
    batch_size = len(operation_sets)
    identity = torch.eye(4, device=reference.device, dtype=reference.dtype)
    padded = identity.view(1, 1, 4, 4).expand(
        batch_size, max_operations, 4, 4
    ).clone()
    mask = torch.zeros(
        (batch_size, max_operations), device=reference.device, dtype=torch.bool
    )
    for graph_idx, operations in enumerate(operation_sets):
        if operations is None:
            continue
        count = int(operations.shape[0])
        padded[graph_idx, :count] = operations.to(
            device=reference.device, dtype=reference.dtype
        )
        mask[graph_idx, :count] = True
    return padded, mask


def asu_collate_fn(samples: Sequence[AsuInput]) -> AsuBatch:
    if not samples:
        raise ValueError("Cannot collate empty list of samples")

    batch_size = len(samples)


    atom_types_list = []
    spacegroup_list = []
    component_id_list = []
    atom_formal_charge_list = []
    atom_charge_mask_list = []
    component_formal_charge_list = []
    component_type_id_list = []
    num_atoms_list = []
    edge_index_list = []
    edge_attr_list = []
    frac_coords_list = []
    lattice_list = []
    cart_coords_list = []
    hybridization_list = []
    batch_list = []
    component_batch_list = []
    cif_ids = []
    n_symops_list: list[Tensor] = []
    n_heavy_cell_list: list[Tensor] = []
    chem_group_id_list: list[Tensor] = []
    crystal_system_id_list: list[Tensor] = []
    hall_number_list: list[Tensor] = []
    setting_id_list: list[Tensor] = []
    lattice_template_id_list: list[Tensor] = []
    n_full_atoms_list: list[Tensor] = []
    symmetry_ops_list: list[Tensor | None] = []


    atom_offset = 0
    component_offset = 0

    for i, sample in enumerate(samples):
        n_atoms = sample.num_atoms.item()
        n_components = int(sample.component_id.max().item()) + 1 if len(sample.component_id) > 0 else 0


        atom_types_list.append(sample.atom_types)
        frac_coords_list.append(sample.frac_coords_gt)
        if sample.hybridization is not None:
            hybridization_list.append(sample.hybridization)
        else:
            hybridization_list.append(torch.zeros(n_atoms, dtype=torch.long))


        if sample.atom_formal_charge.shape != (n_atoms,):
            raise ValueError(
                "AsuInput.atom_formal_charge must contain one value per atom"
            )
        if sample.atom_charge_mask.shape != (n_atoms,):
            raise ValueError("AsuInput.atom_charge_mask must contain one value per atom")
        if sample.atom_charge_mask.dtype != torch.bool:
            raise TypeError("AsuInput.atom_charge_mask must use torch.bool")
        if sample.component_formal_charge.numel() != n_components:
            raise ValueError(
                "AsuInput.component_formal_charge must contain one value per component"
            )
        if sample.component_type_id.shape != (n_components,):
            raise ValueError(
                "AsuInput.component_type_id must contain one value per component"
            )
        atom_formal_charge_list.append(sample.atom_formal_charge)
        atom_charge_mask_list.append(sample.atom_charge_mask)
        component_formal_charge_list.append(sample.component_formal_charge)
        component_type_id_list.append(sample.component_type_id)


        spacegroup_list.append(sample.spacegroup)
        num_atoms_list.append(sample.num_atoms)
        lattice_list.append(sample.lattice_gt)
        sample_cart = getattr(sample, "cart_coords_gt", None)
        if sample_cart is None:
            sample_cart = frac_to_cart_coords(sample.frac_coords_gt, sample.lattice_gt)
        cart_coords_list.append(sample_cart)


        adjusted_component_id = sample.component_id + component_offset
        component_id_list.append(adjusted_component_id)


        adjusted_edge_index = sample.edge_index_conv + atom_offset
        edge_index_list.append(adjusted_edge_index)
        edge_attr_list.append(sample.edge_attr_conv)


        batch_list.append(torch.full((n_atoms,), i, dtype=torch.long))
        component_batch_list.append(
            torch.full((n_components,), i, dtype=torch.long)
        )


        # Preserve graph alignment even when only part of a batch has IDs.
        cif_ids.append(sample.cif_id or "")

        explicit_ops = extract_explicit_symmetry_ops(sample)
        if explicit_ops is None:
            raise ValueError("AsuInput requires explicit Hall-setting symmetry_ops")
        ns = sample.n_symops
        if ns is None or ns.numel() != 1:
            raise ValueError("AsuInput.n_symops must contain exactly one value")
        ns = ns.view(-1).long()
        if int(ns.item()) != int(explicit_ops.shape[0]):
            raise ValueError(
                "AsuInput.n_symops does not match its explicit symmetry_ops: "
                f"{int(ns.item())} versus {int(explicit_ops.shape[0])}"
            )
        nh = sample.n_heavy_cell
        if nh is None:
            raise ValueError(
                "AsuInput.n_heavy_cell is required (per crystal: n_heavy_asu × n_symops). "
                "Regenerate PT shards with: python -m sala.data.preprocessing ..."
            )
        nh = nh.view(-1).long()
        if nh.numel() != 1:
            nh = nh[:1].clone()
        n_symops_list.append(ns)
        n_heavy_cell_list.append(nh)
        symmetry_ops_list.append(explicit_ops)

        cg = getattr(sample, "chem_group_id", None)
        if cg is None:
            cg = torch.tensor([-1], dtype=torch.long)
        else:
            cg = cg.view(-1).long()[:1].clone()
        cs = getattr(sample, "crystal_system_id", None)
        if cs is None:
            cs = torch.tensor([-1], dtype=torch.long)
        else:
            cs = cs.view(-1).long()[:1].clone()
        chem_group_id_list.append(cg)
        crystal_system_id_list.append(cs)
        hall = sample.hall_number
        setting = sample.setting_id
        template = sample.lattice_template_id
        n_full = sample.n_full_atoms
        for name, value in (
            ("hall_number", hall),
            ("setting_id", setting),
            ("lattice_template_id", template),
            ("n_full_atoms", n_full),
        ):
            if value is None or value.numel() != 1:
                raise ValueError(f"AsuInput.{name} must contain exactly one value")
        hall = hall.view(-1).long()
        setting = setting.view(-1).long()
        template = template.view(-1).long()
        n_full = n_full.view(-1).long()
        if int(hall.item()) <= 0 or int(setting.item()) <= 0:
            raise ValueError("AsuInput requires positive Hall/setting identifiers")
        if int(template.item()) < 0:
            raise ValueError("AsuInput requires non-negative lattice_template_id")
        expected_full = int(n_atoms) * int(ns.item())
        if int(nh.item()) != expected_full or int(n_full.item()) != expected_full:
            raise ValueError("AsuInput full-cell atom counts are inconsistent")
        hall_number_list.append(hall.clone())
        setting_id_list.append(setting.clone())
        lattice_template_id_list.append(template.clone())
        n_full_atoms_list.append(n_full.clone())


        atom_offset += n_atoms
        component_offset += n_components


    atom_types = torch.cat(atom_types_list, dim=0)
    spacegroup = torch.cat(spacegroup_list, dim=0)
    component_id = torch.cat(component_id_list, dim=0)
    atom_formal_charge = torch.cat(atom_formal_charge_list, dim=0)
    atom_charge_mask = torch.cat(atom_charge_mask_list, dim=0)
    component_formal_charge = torch.cat(component_formal_charge_list, dim=0)
    component_type_id = torch.cat(component_type_id_list, dim=0)
    num_atoms = torch.cat(num_atoms_list, dim=0)
    frac_coords_gt = torch.cat(frac_coords_list, dim=0)
    lattice_gt = torch.stack(lattice_list, dim=0)
    cart_coords_gt = torch.cat(cart_coords_list, dim=0)
    hybridization = torch.cat(hybridization_list, dim=0)
    batch = torch.cat(batch_list, dim=0)
    component_batch = torch.cat(component_batch_list, dim=0)


    if edge_index_list and any(e.shape[1] > 0 for e in edge_index_list):
        edge_index_conv = torch.cat(
            [e for e in edge_index_list if e.shape[1] > 0], dim=1
        )
    else:
        edge_index_conv = torch.zeros((2, 0), dtype=torch.long)


    if edge_attr_list and any(e.shape[0] > 0 for e in edge_attr_list):
        edge_attr_conv = torch.cat(
            [e for e in edge_attr_list if e.shape[0] > 0], dim=0
        )
    else:
        edge_attr_conv = torch.zeros(0, dtype=torch.long)


    ptr = torch.zeros(batch_size + 1, dtype=torch.long)
    ptr[1:] = num_atoms.cumsum(dim=0)

    n_symops_b = torch.cat(n_symops_list, dim=0)
    n_heavy_cell_b = torch.cat(n_heavy_cell_list, dim=0)
    chem_group_id_b = torch.cat(chem_group_id_list, dim=0)
    crystal_system_id_b = torch.cat(crystal_system_id_list, dim=0)
    hall_number_b = torch.cat(hall_number_list, dim=0)
    setting_id_b = torch.cat(setting_id_list, dim=0)
    lattice_template_id_b = torch.cat(lattice_template_id_list, dim=0)
    n_full_atoms_b = torch.cat(n_full_atoms_list, dim=0)
    symmetry_ops_b, symmetry_mask_b = _pack_symmetry_operations(
        symmetry_ops_list,
        reference=frac_coords_gt,
    )

    collated = AsuBatch(
        atom_types=atom_types,
        spacegroup=spacegroup,
        component_id=component_id,
        atom_formal_charge=atom_formal_charge,
        atom_charge_mask=atom_charge_mask,
        component_formal_charge=component_formal_charge,
        component_type_id=component_type_id,
        num_atoms=num_atoms,
        edge_index_conv=edge_index_conv,
        edge_attr_conv=edge_attr_conv,
        frac_coords_gt=frac_coords_gt,
        lattice_gt=lattice_gt,
        batch=batch,
        ptr=ptr,
        component_batch=component_batch,
        cart_coords_gt=cart_coords_gt,
        hybridization=hybridization,
        cif_ids=cif_ids if any(cif_ids) else None,
        n_symops=n_symops_b,
        n_heavy_cell=n_heavy_cell_b,
        chem_group_id=chem_group_id_b,
        crystal_system_id=crystal_system_id_b,
        hall_number=hall_number_b,
        setting_id=setting_id_b,
        lattice_template_id=lattice_template_id_b,
        n_full_atoms=n_full_atoms_b,
        symmetry_ops=symmetry_ops_b,
        symmetry_mask=symmetry_mask_b,
    )
    from sala.core.orbit import build_orbit_plan

    collated.orbit_plan = build_orbit_plan(
        collated, validate_general_position=True
    )
    return collated


def unbatch_asu(batch: AsuBatch) -> list[AsuInput]:
    samples = []
    batch_size = batch.batch_size

    for i in range(batch_size):

        atom_mask = batch.batch == i


        atom_types = batch.atom_types[atom_mask]
        atom_formal_charge = batch.atom_formal_charge[atom_mask]
        atom_charge_mask = batch.atom_charge_mask[atom_mask]
        frac_coords = batch.frac_coords_gt[atom_mask]
        cart_coords = batch.cart_coords_gt[atom_mask] if batch.cart_coords_gt is not None else None
        hybridization = batch.hybridization[atom_mask] if batch.hybridization is not None else None


        component_id = batch.component_id[atom_mask]
        if len(component_id) > 0:
            component_id = component_id - component_id.min()
        component_mask = batch.component_batch == i
        component_formal_charge = batch.component_formal_charge[component_mask]
        component_type_id = batch.component_type_id[component_mask]


        edge_index, edge_attr = _extract_sample_edges(
            batch.edge_index_conv, batch.edge_attr_conv, atom_mask, batch.ptr[i]
        )


        spacegroup = batch.spacegroup[i : i + 1]
        num_atoms = batch.num_atoms[i : i + 1]
        lattice = batch.lattice_gt[i]


        cif_id = (batch.cif_ids[i] or None) if batch.cif_ids else None
        n_symops_i = batch.n_symops[i : i + 1] if batch.n_symops is not None else None
        n_heavy_cell_i = batch.n_heavy_cell[i : i + 1] if batch.n_heavy_cell is not None else None
        chem_group_id_i = batch.chem_group_id[i : i + 1] if batch.chem_group_id is not None else None
        crystal_system_id_i = batch.crystal_system_id[i : i + 1] if batch.crystal_system_id is not None else None
        hall_number_i = batch.hall_number[i : i + 1] if batch.hall_number is not None else None
        setting_id_i = batch.setting_id[i : i + 1] if batch.setting_id is not None else None
        lattice_template_id_i = (
            batch.lattice_template_id[i : i + 1]
            if batch.lattice_template_id is not None
            else None
        )
        n_full_atoms_i = batch.n_full_atoms[i : i + 1] if batch.n_full_atoms is not None else None
        symmetry_ops_i = extract_explicit_symmetry_ops(batch, i)

        samples.append(
            AsuInput(
                atom_types=atom_types,
                spacegroup=spacegroup,
                component_id=component_id,
                atom_formal_charge=atom_formal_charge,
                atom_charge_mask=atom_charge_mask,
                component_formal_charge=component_formal_charge,
                component_type_id=component_type_id,
                num_atoms=num_atoms,
                edge_index_conv=edge_index,
                edge_attr_conv=edge_attr,
                frac_coords_gt=frac_coords,
                lattice_gt=lattice,
                cart_coords_gt=cart_coords,
                hybridization=hybridization,
                cif_id=cif_id,
                n_symops=n_symops_i,
                n_heavy_cell=n_heavy_cell_i,
                chem_group_id=chem_group_id_i,
                crystal_system_id=crystal_system_id_i,
                hall_number=hall_number_i,
                setting_id=setting_id_i,
                lattice_template_id=lattice_template_id_i,
                n_full_atoms=n_full_atoms_i,
                symmetry_ops=(
                    symmetry_ops_i.clone() if symmetry_ops_i is not None else None
                ),
            )
        )

    return samples


def _extract_sample_edges(
    edge_index: Tensor,
    edge_attr: Tensor,
    atom_mask: Tensor,
    ptr_start: int,
) -> tuple[Tensor, Tensor]:
    if edge_index.shape[1] == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.long)


    atom_indices = torch.where(atom_mask)[0]
    atom_set = set(atom_indices.tolist())


    src, dst = edge_index[0], edge_index[1]
    edge_mask = torch.tensor(
        [s.item() in atom_set and d.item() in atom_set for s, d in zip(src, dst)],
        dtype=torch.bool,
    )

    if not edge_mask.any():
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.long)


    sample_edges = edge_index[:, edge_mask]
    sample_edges = sample_edges - ptr_start


    sample_edge_attr = edge_attr[edge_mask]

    return sample_edges, sample_edge_attr
