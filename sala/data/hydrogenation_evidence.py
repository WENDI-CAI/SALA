"""Production chemical evidence for formula-constrained hydrogenation.

The builder intentionally ignores explicit H/D/T atom coordinates and their
bonds.  Those data are ground truth for recovery evaluation, not permissible
inputs to the production inference path.  The only direct attached-hydrogen
signal accepted here is the CIF ``_atom_site_attached_hydrogens`` field.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

HYDROGENATION_EVIDENCE_VERSION = "cif_hydrogenation_evidence_v3"


def build_hydrogenation_evidence(cif_evidence: Any, asu: Any) -> dict:
    """Align public CIF chemical evidence to a heavy-atom ASU ordering."""

    source_labels = [str(value) for value in cif_evidence.atom_labels]
    if len(set(source_labels)) != len(source_labels):
        raise ValueError("CIF atom labels must be unique for hydrogenation")
    source_index = {label: index for index, label in enumerate(source_labels)}
    source_types = [int(value) for value in cif_evidence.atom_types]
    asu_labels = list(getattr(asu, "atom_labels", None) or [])
    num_asu_atoms = int(asu.atom_types.numel())
    if len(asu_labels) != num_asu_atoms:
        raise ValueError(
            "Hydrogenation evidence requires one authoritative label per ASU atom"
        )
    all_asu_types = [int(value) for value in asu.atom_types.detach().cpu().tolist()]
    all_asu_charges = [
        int(value) for value in asu.atom_formal_charge.detach().cpu().tolist()
    ]
    all_asu_charge_mask = [
        bool(value) for value in asu.atom_charge_mask.detach().cpu().tolist()
    ]
    heavy_asu_indices = [
        index for index, atomic_number in enumerate(all_asu_types) if atomic_number != 1
    ]
    heavy_labels = [asu_labels[index] for index in heavy_asu_indices]
    missing = [label for label in heavy_labels if label not in source_index]
    if missing:
        raise ValueError(
            "CIF chemical evidence is missing heavy-atom labels: " + ", ".join(missing)
        )
    heavy_source_indices = [source_index[label] for label in heavy_labels]
    if any(source_types[index] == 1 for index in heavy_source_indices):
        raise ValueError(
            "Generation ASU labels map to hydrogen sites in the source CIF"
        )
    asu_types = [all_asu_types[index] for index in heavy_asu_indices]
    atom_formal_charges = [all_asu_charges[index] for index in heavy_asu_indices]
    atom_charge_mask = [all_asu_charge_mask[index] for index in heavy_asu_indices]
    if [source_types[index] for index in heavy_source_indices] != asu_types:
        raise ValueError("CIF chemical evidence atom types do not match the ASU")

    def atom_values(values: Sequence[Any], default: Any) -> list[Any]:
        source = list(values) if len(values) == len(source_labels) else None
        return [
            source[index] if source is not None else default
            for index in heavy_source_indices
        ]

    num_bonds = len(cif_evidence.bonds)

    def bond_values(values: Sequence[Any], default: Any) -> list[Any]:
        return list(values) if len(values) == num_bonds else [default] * num_bonds

    bond_order_mask = bond_values(cif_evidence.bond_order_authoritative, False)
    bond_order_sources = bond_values(cif_evidence.bond_order_sources, "default_single")
    bond_valences = bond_values(cif_evidence.bond_valences, None)
    bond_symmetry_1 = bond_values(cif_evidence.bond_symmetry_1, "")
    bond_symmetry_2 = bond_values(cif_evidence.bond_symmetry_2, "")
    source_type_by_label = dict(zip(source_labels, source_types))
    heavy_index = {label: index for index, label in enumerate(heavy_labels)}
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_orders: list[float] = []
    edge_order_mask: list[bool] = []
    edge_order_sources: list[str] = []
    edge_valences: list[float | None] = []
    edge_symmetry_refs: list[list[str]] = []
    num_symmetry_bonds = 0

    def is_nonidentity_bond_symmetry(value: str) -> bool:
        clean = str(value).strip().strip("'\"")
        if clean in {"", ".", "?"}:
            return False
        return clean not in {"1_555", "0", "1"}

    for bond_index, (atom1, atom2, _distance, order) in enumerate(cif_evidence.bonds):
        # Explicit hydrogen bonds are recovery ground truth and must not leak
        # parent-site answers into production evidence.
        if source_type_by_label.get(atom1) == 1 or source_type_by_label.get(atom2) == 1:
            continue
        if atom1 not in heavy_index or atom2 not in heavy_index:
            continue
        left = heavy_index[atom1]
        right = heavy_index[atom2]
        symmetry_1 = str(bond_symmetry_1[bond_index])
        symmetry_2 = str(bond_symmetry_2[bond_index])
        if is_nonidentity_bond_symmetry(symmetry_1) or is_nonidentity_bond_symmetry(
            symmetry_2
        ):
            num_symmetry_bonds += 1
        edge_sources.extend((left, right))
        edge_targets.extend((right, left))
        edge_orders.extend((float(order), float(order)))
        edge_order_mask.extend(
            (bool(bond_order_mask[bond_index]), bool(bond_order_mask[bond_index]))
        )
        edge_order_sources.extend(
            (str(bond_order_sources[bond_index]), str(bond_order_sources[bond_index]))
        )
        edge_valences.extend((bond_valences[bond_index], bond_valences[bond_index]))
        edge_symmetry_refs.extend(([symmetry_1, symmetry_2], [symmetry_2, symmetry_1]))

    site_hydrogens = atom_values(cif_evidence.atom_attached_hydrogens, 0)
    site_hydrogen_mask = atom_values(cif_evidence.atom_attached_hydrogen_mask, False)
    attached_hydrogens = [
        int(value) if bool(known) else 0
        for value, known in zip(site_hydrogens, site_hydrogen_mask)
    ]
    attached_hydrogen_mask = [bool(value) for value in site_hydrogen_mask]
    attached_hydrogen_sources = [
        "atom_site_attached_hydrogens" if known else "unknown"
        for known in attached_hydrogen_mask
    ]
    occupancies = atom_values(cif_evidence.atom_occupancies, None)
    disorder_assemblies = atom_values(cif_evidence.atom_disorder_assemblies, "")
    disorder_groups = atom_values(cif_evidence.atom_disorder_groups, "")
    has_disorder = any(
        str(value).strip() not in {"", "0", ".", "?"}
        for value in (*disorder_assemblies, *disorder_groups)
    ) or any(
        occupancy is not None and abs(float(occupancy) - 1.0) > 1.0e-6
        for occupancy in occupancies
    )

    heavy_bond_count = len(edge_orders) // 2
    authoritative_bond_count = sum(edge_order_mask) // 2
    if not heavy_bond_count or not authoritative_bond_count:
        bond_order_confidence = "unavailable"
    elif authoritative_bond_count == heavy_bond_count:
        bond_order_confidence = "authoritative"
    else:
        bond_order_confidence = "partial"
    n_symops_value = getattr(asu, "n_symops", None)
    n_symops = (
        int(n_symops_value.view(-1)[0].item()) if n_symops_value is not None else 1
    )
    component_ids = [
        int(asu.component_id[index].item()) for index in heavy_asu_indices
    ]
    num_components = int(asu.component_formal_charge.numel())
    component_formal_charges: list[int] = []
    component_formal_charge_mask: list[bool] = []
    component_formal_charge_sources: list[str] = []
    for component in range(num_components):
        indices = [
            index for index, value in enumerate(component_ids) if value == component
        ]
        # Component charge may disambiguate composition-identical moieties only
        # when independently localized atom-site evidence supports it. A neutral
        # zero-filled column does not establish a charge assignment.
        trusted = bool(indices) and all(atom_charge_mask[index] for index in indices)
        trusted = trusted and any(
            atom_formal_charges[index] != 0 for index in indices
        )
        component_formal_charges.append(
            sum(atom_formal_charges[index] for index in indices) if trusted else 0
        )
        component_formal_charge_mask.append(trusted)
        component_formal_charge_sources.append(
            "trusted_atom_site_sum" if trusted else "unknown"
        )
    return {
        "version": HYDROGENATION_EVIDENCE_VERSION,
        "formula_moiety": cif_evidence.chemical_formula_moiety,
        "formula_sum": cif_evidence.chemical_formula_sum,
        "formula_units_z": cif_evidence.cell_formula_units_z,
        "n_symops": n_symops,
        "atom_labels": heavy_labels,
        "atom_types": asu_types,
        "atom_formal_charges": atom_formal_charges,
        "atom_charge_mask": atom_charge_mask,
        "hybridization": atom_values(cif_evidence.atom_hybridizations, "unknown"),
        "occupancies": occupancies,
        "site_multiplicities": atom_values(cif_evidence.atom_site_multiplicities, None),
        "attached_hydrogens": attached_hydrogens,
        "attached_hydrogen_mask": attached_hydrogen_mask,
        "attached_hydrogen_sources": attached_hydrogen_sources,
        "attached_hydrogen_conflict_labels": [],
        "disorder_assemblies": disorder_assemblies,
        "disorder_groups": disorder_groups,
        "has_disorder": has_disorder,
        "component_id": component_ids,
        "component_formal_charges": component_formal_charges,
        "component_formal_charge_mask": component_formal_charge_mask,
        "component_formal_charge_sources": component_formal_charge_sources,
        "edge_index": [edge_sources, edge_targets],
        "bond_orders": edge_orders,
        "bond_order_mask": edge_order_mask,
        "bond_order_sources": edge_order_sources,
        "bond_valences": edge_valences,
        "bond_symmetry_refs": edge_symmetry_refs,
        "num_symmetry_bonds": num_symmetry_bonds,
        "has_symmetry_bonds": bool(num_symmetry_bonds),
        "bond_order_confidence": bond_order_confidence,
        "heavy_bond_count": heavy_bond_count,
        "authoritative_bond_count": authoritative_bond_count,
    }


__all__ = ["HYDROGENATION_EVIDENCE_VERSION", "build_hydrogenation_evidence"]
