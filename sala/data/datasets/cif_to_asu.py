
from __future__ import annotations

import re
from collections import Counter, defaultdict

import torch

from sala.core.graph import build_covalent_graph
from sala.core.lattice_manifold import (
    infer_lattice_template,
    lattice_invariance_residual,
    lattice_to_q,
    q_to_lattice,
)
from sala.core.orbit import validate_general_position_orbits
from sala.core.symmetry import canonicalize_affine_symmetry_ops, parse_symops
from sala.core.types import AsuInput, CifData, encode_hybridization, get_element_symbol
from sala.data.chemistry import assign_component_type_ids, build_charge_features
from sala.data.datasets.asu_canonicalize import canonicalize_asu_input
from sala.data.datasets.pt_dataset import bond_order_to_idx

_HYDROGEN_SYMBOLS = {"H", "D", "T"}


def cif_data_to_asu_input(cif_data: CifData) -> AsuInput:
    edge_index_conv, component_id, edge_attr_conv_float = build_covalent_graph(
        cif_data.bonds, cif_data.atom_labels
    )
    edge_attr_conv = bond_order_to_idx(edge_attr_conv_float).long()

    num_atoms = len(cif_data.atom_types)
    num_components = int(component_id.max().item()) + 1 if component_id.numel() > 0 else 0
    component_formal_charge = _build_component_formal_charge(
        cif_data=cif_data,
        component_id=component_id,
        num_components=num_components,
    )
    hybridization = _build_atom_hybridization(cif_data, num_atoms)
    atom_types = torch.tensor(cif_data.atom_types, dtype=torch.long)
    charge_features = build_charge_features(
        cif_data,
        component_id=component_id,
        component_formal_charge=component_formal_charge,
    )
    component_type_id = assign_component_type_ids(
        atom_types=atom_types,
        hybridization=hybridization,
        atom_formal_charge=charge_features.atom_formal_charge,
        atom_charge_mask=charge_features.atom_charge_mask,
        component_id=component_id,
        component_formal_charge=charge_features.component_formal_charge,
        edge_index=edge_index_conv,
        edge_attr=edge_attr_conv,
    )

    symmetry_ops = _affine_symmetry_ops(cif_data)
    n_symops = int(symmetry_ops.shape[0])
    declared_n_symops = max(1, int(getattr(cif_data, "n_symops", n_symops)))
    if declared_n_symops != n_symops:
        raise ValueError(
            "CIF symmetry metadata is inconsistent: "
            f"n_symops={declared_n_symops}, actual operations={n_symops}"
        )
    n_heavy_cell = int(num_atoms) * n_symops
    validate_general_position_orbits(cif_data.frac_coords, symmetry_ops)
    spacegroup = torch.tensor([cif_data.spacegroup], dtype=torch.long)
    hall_number = int(getattr(cif_data, "hall_number", -1))
    setting_id = int(getattr(cif_data, "setting_id", -1))
    if hall_number <= 0 or setting_id <= 0:
        raise ValueError(
            "Full-cell preprocessing requires authoritative positive Hall and "
            "setting identifiers"
        )
    explicit_template = int(getattr(cif_data, "lattice_template_id", -1))
    if explicit_template < 0:
        explicit_template = int(infer_lattice_template(spacegroup).item())
    template_id = torch.tensor([explicit_template], dtype=torch.long)
    source_residual = lattice_invariance_residual(
        cif_data.lattice, symmetry_ops[:, :3, :3]
    )
    if source_residual.numel() and float(source_residual.max().item()) > 1.0e-3:
        raise ValueError(
            "CIF lattice metric is incompatible with its symmetry operations "
            f"(max relative residual={float(source_residual.max().item()):.3e})"
        )
    projected_q = lattice_to_q(
        cif_data.lattice.unsqueeze(0),
        spacegroup=spacegroup,
        lattice_template_id=template_id,
    )
    constrained_lattice = q_to_lattice(
        projected_q,
        spacegroup=spacegroup,
        lattice_template_id=template_id,
    )[0]
    residual = lattice_invariance_residual(
        constrained_lattice, symmetry_ops[:, :3, :3]
    )
    if residual.numel() and float(residual.max().item()) > 1.0e-4:
        raise ValueError(
            "Resolved lattice template is incompatible with the CIF symmetry "
            f"operations (max relative metric residual={float(residual.max().item()):.3e})"
        )

    return canonicalize_asu_input(AsuInput(
        atom_types=atom_types,
        spacegroup=spacegroup,
        component_id=component_id,
        atom_formal_charge=charge_features.atom_formal_charge,
        atom_charge_mask=charge_features.atom_charge_mask,
        component_formal_charge=charge_features.component_formal_charge,
        component_type_id=component_type_id,
        num_atoms=torch.tensor([num_atoms], dtype=torch.long),
        edge_index_conv=edge_index_conv,
        edge_attr_conv=edge_attr_conv,
        frac_coords_gt=cif_data.frac_coords,
        lattice_gt=cif_data.lattice,
        hybridization=hybridization,
        atom_labels=cif_data.atom_labels,
        cif_id=cif_data.cif_id,
        n_symops=torch.tensor([n_symops], dtype=torch.long),
        n_heavy_cell=torch.tensor([n_heavy_cell], dtype=torch.long),
        n_full_atoms=torch.tensor([num_atoms * n_symops], dtype=torch.long),
        hall_number=torch.tensor([hall_number], dtype=torch.long),
        setting_id=torch.tensor([setting_id], dtype=torch.long),
        lattice_template_id=template_id,
        symmetry_ops=symmetry_ops,
    ))


def _affine_symmetry_ops(cif_data: CifData) -> torch.Tensor | None:
    if not cif_data.symop_xyz:
        from sala.core.symmetry import get_sym_ops, get_sym_ops_from_hall

        hall_number = int(getattr(cif_data, "hall_number", -1))
        operations = (
            get_sym_ops_from_hall(hall_number)
            if hall_number > 0
            else get_sym_ops(cif_data.spacegroup)
        )
        return canonicalize_affine_symmetry_ops(
            torch.as_tensor(operations, dtype=torch.float32)
        )
    matrices: list[torch.Tensor] = []
    for operation in parse_symops(cif_data.symop_xyz):
        matrix = torch.eye(4, dtype=torch.float32)
        matrix[:3, :3] = torch.as_tensor(operation.R, dtype=torch.float32)
        matrix[:3, 3] = torch.as_tensor(operation.t, dtype=torch.float32)
        matrices.append(matrix)
    return canonicalize_affine_symmetry_ops(torch.stack(matrices)) if matrices else None


def _build_component_formal_charge(
    cif_data: CifData,
    component_id: torch.Tensor,
    num_components: int,
) -> torch.Tensor:
    if num_components <= 0:
        return torch.zeros(0, dtype=torch.float32)

    moiety_charge = _component_charge_from_moiety(cif_data, component_id, num_components)
    if moiety_charge is not None:
        return moiety_charge

    if cif_data.atom_formal_charges and len(cif_data.atom_formal_charges) == len(cif_data.atom_types):
        atom_charge = torch.tensor(cif_data.atom_formal_charges, dtype=torch.float32)
        component_charge = torch.zeros(num_components, dtype=torch.float32)
        component_charge.index_add_(0, component_id.long(), atom_charge)
        return component_charge

    return torch.zeros(num_components, dtype=torch.float32)


def _build_atom_hybridization(cif_data: CifData, num_atoms: int) -> torch.Tensor:
    if cif_data.atom_hybridizations and len(cif_data.atom_hybridizations) == num_atoms:
        indices = [encode_hybridization(value) for value in cif_data.atom_hybridizations]
        return torch.tensor(indices, dtype=torch.long)
    return torch.zeros(num_atoms, dtype=torch.long)


def _component_charge_from_moiety(
    cif_data: CifData,
    component_id: torch.Tensor,
    num_components: int,
) -> torch.Tensor | None:
    moiety_entries = _parse_moiety_entries(cif_data.chemical_formula_moiety)
    if not moiety_entries:
        return None

    component_keys = [
        _component_formula_key(cif_data.atom_types, component_id, component_index)
        for component_index in range(num_components)
    ]
    moiety_keys = [entry["key"] for entry in moiety_entries]
    if Counter(component_keys) != Counter(moiety_keys):
        return None

    charges_by_key: dict[tuple[tuple[str, int], ...], list[int]] = defaultdict(list)
    for entry in moiety_entries:
        charges_by_key[entry["key"]].append(int(entry["charge"]))

    atom_charge_sum: torch.Tensor | None = None
    atom_charge_known: torch.Tensor | None = None
    if len(cif_data.atom_formal_charges) == len(cif_data.atom_types):
        raw_charge = torch.as_tensor(cif_data.atom_formal_charges, dtype=torch.long)
        atom_charge_sum = torch.zeros(num_components, dtype=torch.long)
        atom_charge_sum.index_add_(0, component_id.long(), raw_charge)
        if len(cif_data.atom_formal_charge_mask) == len(cif_data.atom_types):
            source_mask = torch.as_tensor(
                cif_data.atom_formal_charge_mask, dtype=torch.bool
            )
            known_count = torch.zeros(num_components, dtype=torch.long)
            known_count.index_add_(0, component_id.long(), source_mask.long())
            component_count = torch.bincount(
                component_id.long(), minlength=num_components
            )
            atom_charge_known = known_count == component_count

    assigned = []
    for component_index, key in enumerate(component_keys):
        candidates = charges_by_key[key]
        preferred = None
        if (
            atom_charge_sum is not None
            and atom_charge_known is not None
            and bool(atom_charge_known[component_index])
        ):
            raw_value = int(atom_charge_sum[component_index])
            if raw_value in candidates:
                preferred = raw_value
        if preferred is None:
            preferred = candidates[-1]
        candidates.remove(preferred)
        assigned.append(preferred)
    return torch.tensor(assigned, dtype=torch.float32)


def _component_formula_key(
    atom_types: list[int],
    component_id: torch.Tensor,
    component_index: int,
) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = defaultdict(int)
    mask = component_id == component_index
    for atom_index in mask.nonzero(as_tuple=False).view(-1).tolist():
        symbol = get_element_symbol(int(atom_types[atom_index]))
        if symbol in _HYDROGEN_SYMBOLS:
            continue
        counts[symbol] += 1
    return tuple(sorted((element, count) for element, count in counts.items() if count > 0))


def _parse_moiety_entries(moiety: str | None) -> list[dict[str, object]]:
    if not moiety:
        return []

    clean = " ".join(str(moiety).replace(";", " ").replace("\n", " ").split())
    if not clean:
        return []

    entries: list[dict[str, object]] = []
    for part in _split_top_level(clean, sep=","):
        match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*\((.*)\)\s*$", part)
        if match:
            repeat = float(match.group(1))
            body = match.group(2).strip()
        else:
            repeat = 1.0
            body = part.strip()

        rounded_repeat = int(round(repeat))
        if abs(repeat - rounded_repeat) > 1e-8 or rounded_repeat < 1:
            return []

        formula_text, charge = _split_formula_and_charge(body)
        key = _formula_key(formula_text)
        if not key:
            continue
        for _ in range(rounded_repeat):
            entries.append({"key": key, "charge": int(charge)})
    return entries


def _formula_key(formula_text: str) -> tuple[tuple[str, int], ...]:
    counts: dict[str, float] = defaultdict(float)
    for element, number in re.findall(r"([A-Z][a-z]?)([\d\.]*)", formula_text):
        if element in _HYDROGEN_SYMBOLS:
            continue
        amount = 1.0
        if number:
            try:
                amount = float(number)
            except ValueError:
                return ()
        counts[element] += amount

    normalized = []
    for element, amount in counts.items():
        rounded = int(round(amount))
        if abs(amount - rounded) > 1e-8 or rounded <= 0:
            return ()
        normalized.append((element, rounded))
    return tuple(sorted(normalized))


def _split_formula_and_charge(text: str) -> tuple[str, int]:
    text = " ".join(text.strip().split())
    if not text:
        return "", 0

    match = re.match(r"^(.*?)(?:\s|^)(\d+)\s*([+-])$", text)
    if match:
        charge = int(match.group(2))
        if match.group(3) == "-":
            charge = -charge
        return match.group(1).strip(), charge

    match = re.match(r"^(.*?)(?:\s|^)([+-]\d+)$", text)
    if match:
        return match.group(1).strip(), int(match.group(2))

    match = re.match(r"^(.*?)(?:\s|^)([+-])$", text)
    if match:
        return match.group(1).strip(), 1 if match.group(2) == "+" else -1

    return text, 0


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    level = 0
    for char in text:
        if char == "(":
            level += 1
        elif char == ")":
            level = max(0, level - 1)
        elif char == sep and level == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts
