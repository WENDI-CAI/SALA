
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from sala.core.lattice import lattice_params_to_matrix as _core_lattice_params_to_matrix
from sala.core.lattice_manifold import lattice_template_from_hall_number
from sala.core.symmetry import (
    get_sym_ops,
    get_sym_ops_from_hall,
    infer_hall_number_from_symmetry_setting,
    parse_symops,
    spacegroup_number_from_symmetry_ops,
    symmetry_ops_match_hall,
)
from sala.core.types import CifChemicalEvidence, CifData, get_atomic_number


@dataclass(frozen=True)
class _CifToken:
    value: str
    quoted: bool
    line: int


@dataclass(frozen=True)
class _CifLoop:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _CifDocument:
    tags: dict[str, str]
    loops: tuple[_CifLoop, ...]


@dataclass(frozen=True)
class _AtomSiteData:
    labels: list[str]
    atom_types: list[int]
    frac_coords: Tensor
    formal_charges: list[int]
    formal_charge_mask: list[bool]
    hybridizations: list[str]
    occupancies: list[Optional[float]]
    site_multiplicities: list[Optional[int]]
    attached_hydrogens: list[int]
    attached_hydrogen_mask: list[bool]
    disorder_assemblies: list[str]
    disorder_groups: list[str]
    has_disorder: bool


@dataclass(frozen=True)
class _BondData:
    bonds: list[tuple[str, str, float, float]]
    num_symmetry_bonds: int
    order_authoritative: list[bool]
    order_sources: list[str]
    valences: list[Optional[float]]
    symmetry_1: list[str]
    symmetry_2: list[str]


def _tokenize_cif(content: str) -> list[_CifToken]:
    """Tokenize the CIF 1.x constructs used by structural CIF files.

    Loop values are a token stream rather than a line-oriented table.  This is
    important because legal CIF writers freely wrap a row across physical
    lines. Comments are discarded only outside quoted/text values.
    """

    lines = content.splitlines()
    tokens: list[_CifToken] = []
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        line_number = line_idx + 1
        if line.startswith(";"):
            text_lines = [line[1:]] if len(line) > 1 else []
            line_idx += 1
            while line_idx < len(lines) and not lines[line_idx].startswith(";"):
                text_lines.append(lines[line_idx])
                line_idx += 1
            if line_idx >= len(lines):
                raise ValueError(
                    f"Unterminated semicolon-delimited CIF text field at line {line_number}"
                )
            tokens.append(_CifToken("\n".join(text_lines), True, line_number))
            line_idx += 1
            continue

        column = 0
        while column < len(line):
            while column < len(line) and line[column].isspace():
                column += 1
            if column >= len(line) or line[column] == "#":
                break
            if line[column] in {"'", '"'}:
                quote = line[column]
                column += 1
                start = column
                while column < len(line):
                    if line[column] == quote and (
                        column + 1 == len(line)
                        or line[column + 1].isspace()
                        or line[column + 1] == "#"
                    ):
                        break
                    column += 1
                if column >= len(line):
                    raise ValueError(
                        f"Unterminated quoted CIF value at line {line_number}"
                    )
                tokens.append(_CifToken(line[start:column], True, line_number))
                column += 1
                continue

            start = column
            while (
                column < len(line)
                and not line[column].isspace()
                and line[column] != "#"
            ):
                column += 1
            if column > start:
                tokens.append(_CifToken(line[start:column], False, line_number))
            if column < len(line) and line[column] == "#":
                break
        line_idx += 1
    return tokens


def _is_data_boundary(token: _CifToken) -> bool:
    if token.quoted:
        return False
    value = token.value.lower()
    return (
        value == "loop_"
        or value == "stop_"
        or value == "global_"
        or value.startswith("data_")
        or value.startswith("save_")
        or value.startswith("_")
    )


def _parse_cif_tokens(tokens: list[_CifToken]) -> _CifDocument:
    tags: dict[str, str] = {}
    loops: list[_CifLoop] = []
    token_idx = 0
    while token_idx < len(tokens):
        token = tokens[token_idx]
        lower = token.value.lower()
        if not token.quoted and lower == "loop_":
            token_idx += 1
            headers: list[str] = []
            while token_idx < len(tokens):
                header = tokens[token_idx]
                if header.quoted or not header.value.startswith("_"):
                    break
                headers.append(header.value.lower())
                token_idx += 1
            if not headers:
                raise ValueError(f"CIF loop at line {token.line} has no data names")

            values: list[str] = []
            while token_idx < len(tokens) and not _is_data_boundary(tokens[token_idx]):
                values.append(tokens[token_idx].value)
                token_idx += 1
            width = len(headers)
            remainder = len(values) % width
            if remainder:
                raise ValueError(
                    f"CIF loop beginning with {headers[0]} has {len(values)} values "
                    f"for {width} columns (incomplete final row)"
                )
            rows = tuple(
                tuple(values[start : start + width])
                for start in range(0, len(values), width)
            )
            loops.append(_CifLoop(tuple(headers), rows))
            if (
                token_idx < len(tokens)
                and not tokens[token_idx].quoted
                and tokens[token_idx].value.lower() == "stop_"
            ):
                token_idx += 1
            continue

        if not token.quoted and token.value.startswith("_"):
            tag = lower
            token_idx += 1
            if token_idx >= len(tokens) or _is_data_boundary(tokens[token_idx]):
                raise ValueError(f"CIF data name {token.value} has no value")
            tags[tag] = tokens[token_idx].value
            token_idx += 1
            continue
        token_idx += 1
    return _CifDocument(tags=tags, loops=tuple(loops))


def _contains_complete_structure(document: _CifDocument) -> bool:
    required_cell_tags = {
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    }
    required_atom_headers = {
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    }
    return required_cell_tags.issubset(document.tags) and any(
        required_atom_headers.issubset(set(loop.headers))
        for loop in document.loops
    )


def _parse_cif_document(content: str) -> _CifDocument:
    tokens = _tokenize_cif(content)
    block_tokens: list[tuple[str, list[_CifToken]]] = []
    current_name = "<preamble>"
    current_tokens: list[_CifToken] = []
    saw_data_block = False

    for token in tokens:
        lower = token.value.lower()
        if not token.quoted and lower.startswith("data_"):
            if current_tokens:
                block_tokens.append((current_name, current_tokens))
            current_name = token.value
            current_tokens = []
            saw_data_block = True
            continue
        current_tokens.append(token)

    if not saw_data_block:
        return _parse_cif_tokens(tokens)
    block_tokens.append((current_name, current_tokens))

    parsed_blocks = [
        (name, _parse_cif_tokens(tokens_for_block))
        for name, tokens_for_block in block_tokens
    ]
    structural_blocks = [
        (name, document)
        for name, document in parsed_blocks
        if _contains_complete_structure(document)
    ]
    if len(structural_blocks) == 1:
        return structural_blocks[0][1]
    if len(structural_blocks) > 1:
        names = ", ".join(name for name, _document in structural_blocks)
        raise ValueError(
            "CIF contains multiple complete structural data blocks; "
            f"one structure per file is required ({names})"
        )
    names = ", ".join(name for name, _document in parsed_blocks)
    raise ValueError(
        "No single CIF data block contains both complete cell parameters and "
        f"an atom-site loop ({names})"
    )


def _as_cif_document(value: str | _CifDocument) -> _CifDocument:
    return value if isinstance(value, _CifDocument) else _parse_cif_document(value)


def read_cif(path: str | Path) -> CifData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    document = _parse_cif_document(content)


    atom_sites = _parse_atom_site_data(document)
    bond_data = _parse_bond_data(document)
    evidence = _parse_chemical_evidence(
        document,
        cif_id=path.stem,
        atom_sites=atom_sites,
        bond_data=bond_data,
    )
    lattice = _parse_lattice_params(document)
    spacegroup, symop_xyz = _parse_spacegroup_and_symmetry_ops(document)
    hall_number, setting_id, lattice_template_id = _parse_hall_setting(
        document,
        spacegroup=spacegroup,
        symop_xyz=symop_xyz,
    )
    n_symops = (
        len(symop_xyz)
        if symop_xyz
        else int(len(get_sym_ops_from_hall(hall_number)))
    )

    return CifData(
        atom_labels=evidence.atom_labels,
        atom_types=evidence.atom_types,
        frac_coords=atom_sites.frac_coords,
        lattice=lattice,
        bonds=evidence.bonds,
        spacegroup=spacegroup,
        cif_id=evidence.cif_id,
        chemical_formula_moiety=evidence.chemical_formula_moiety,
        chemical_formula_sum=evidence.chemical_formula_sum,
        cell_formula_units_z=evidence.cell_formula_units_z,
        atom_formal_charges=evidence.atom_formal_charges,
        atom_formal_charge_mask=evidence.atom_formal_charge_mask,
        atom_hybridizations=evidence.atom_hybridizations,
        atom_occupancies=evidence.atom_occupancies,
        atom_site_multiplicities=evidence.atom_site_multiplicities,
        atom_attached_hydrogens=evidence.atom_attached_hydrogens,
        atom_attached_hydrogen_mask=evidence.atom_attached_hydrogen_mask,
        atom_disorder_assemblies=evidence.atom_disorder_assemblies,
        atom_disorder_groups=evidence.atom_disorder_groups,
        has_disorder=evidence.has_disorder,
        bond_order_authoritative=evidence.bond_order_authoritative,
        bond_order_sources=evidence.bond_order_sources,
        bond_valences=evidence.bond_valences,
        bond_symmetry_1=evidence.bond_symmetry_1,
        bond_symmetry_2=evidence.bond_symmetry_2,
        n_symops=n_symops,
        symop_xyz=symop_xyz,
        num_symmetry_bonds=evidence.num_symmetry_bonds,
        hall_number=hall_number,
        setting_id=setting_id,
        lattice_template_id=lattice_template_id,
    )


def read_cif_chemical_evidence(path: str | Path) -> CifChemicalEvidence:
    """Read formula, atom, and bond evidence without resolving crystallography.

    This intentionally avoids space-group and Hall-setting validation so raw
    source CIFs remain usable for hydrogen-ground-truth replay even when their
    crystallographic setting is ambiguous to the full structure reader.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        document = _parse_cif_document(f.read())
    return _parse_chemical_evidence(document, cif_id=path.stem)


def _parse_chemical_evidence(
    document: _CifDocument,
    *,
    cif_id: str,
    atom_sites: _AtomSiteData | None = None,
    bond_data: _BondData | None = None,
) -> CifChemicalEvidence:
    if atom_sites is None:
        atom_sites = _parse_atom_site_data(document)
    if bond_data is None:
        bond_data = _parse_bond_data(document)
    return CifChemicalEvidence(
        atom_labels=atom_sites.labels,
        atom_types=atom_sites.atom_types,
        frac_coords=atom_sites.frac_coords,
        bonds=bond_data.bonds,
        cif_id=cif_id,
        chemical_formula_moiety=_parse_chemical_formula_moiety(document),
        chemical_formula_sum=_parse_chemical_formula_sum(document),
        cell_formula_units_z=_parse_cell_formula_units_z(document),
        atom_formal_charges=atom_sites.formal_charges,
        atom_formal_charge_mask=atom_sites.formal_charge_mask,
        atom_hybridizations=atom_sites.hybridizations,
        atom_occupancies=atom_sites.occupancies,
        atom_site_multiplicities=atom_sites.site_multiplicities,
        atom_attached_hydrogens=atom_sites.attached_hydrogens,
        atom_attached_hydrogen_mask=atom_sites.attached_hydrogen_mask,
        atom_disorder_assemblies=atom_sites.disorder_assemblies,
        atom_disorder_groups=atom_sites.disorder_groups,
        has_disorder=atom_sites.has_disorder,
        bond_order_authoritative=bond_data.order_authoritative,
        bond_order_sources=bond_data.order_sources,
        bond_valences=bond_data.valences,
        bond_symmetry_1=bond_data.symmetry_1,
        bond_symmetry_2=bond_data.symmetry_2,
        num_symmetry_bonds=bond_data.num_symmetry_bonds,
    )


def _parse_lattice_params(content: str | _CifDocument) -> Tensor:
    document = _as_cif_document(content)

    def required_float(tag: str) -> float:
        value = document.tags.get(tag.lower())
        if value is None:
            raise ValueError(f"Could not find CIF data name: {tag}")
        return _parse_cif_float(value)

    a = required_float("_cell_length_a")
    b = required_float("_cell_length_b")
    c = required_float("_cell_length_c")

    alpha = required_float("_cell_angle_alpha")
    beta = required_float("_cell_angle_beta")
    gamma = required_float("_cell_angle_gamma")


    lengths = torch.tensor([a, b, c], dtype=torch.float32)
    angles = torch.tensor([alpha, beta, gamma], dtype=torch.float32)
    return _core_lattice_params_to_matrix(lengths, angles)


def _parse_atom_sites(
    content: str | _CifDocument,
) -> tuple[list[str], list[int], Tensor, list[int], list[bool], list[str]]:
    atom_sites = _parse_atom_site_data(content)
    return (
        atom_sites.labels,
        atom_sites.atom_types,
        atom_sites.frac_coords,
        atom_sites.formal_charges,
        atom_sites.formal_charge_mask,
        atom_sites.hybridizations,
    )


def _parse_atom_site_data(content: str | _CifDocument) -> _AtomSiteData:
    headers, rows = _parse_loop_rows(
        content,
        required_headers=(
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        ),
    )
    if headers is None or rows is None:
        raise ValueError("Could not find atom_site loop in CIF file")
    header_index = {header: idx for idx, header in enumerate(headers)}
    formal_charge_idx = header_index.get("_atom_site_formal_charge")
    hybridization_idx = header_index.get("_atom_site_hybridization")
    occupancy_idx = header_index.get("_atom_site_occupancy")
    multiplicity_idx = header_index.get("_atom_site_symmetry_multiplicity")
    if multiplicity_idx is None:
        multiplicity_idx = header_index.get("_atom_site_site_symmetry_multiplicity")
    attached_hydrogen_idx = header_index.get("_atom_site_attached_hydrogens")
    disorder_assembly_idx = header_index.get("_atom_site_disorder_assembly")
    disorder_group_idx = header_index.get("_atom_site_disorder_group")

    atom_labels: list[str] = []
    atom_types: list[int] = []
    coords: list[list[float]] = []
    atom_formal_charges: list[int] = []
    atom_formal_charge_mask: list[bool] = []
    atom_hybridizations: list[str] = []
    atom_occupancies: list[Optional[float]] = []
    atom_site_multiplicities: list[Optional[int]] = []
    atom_attached_hydrogens: list[int] = []
    atom_attached_hydrogen_mask: list[bool] = []
    atom_disorder_assemblies: list[str] = []
    atom_disorder_groups: list[str] = []

    for parts in rows:
        if len(parts) < len(headers):
            continue

        label = _normalize_loop_value(parts[header_index["_atom_site_label"]])
        element = _normalize_loop_value(parts[header_index["_atom_site_type_symbol"]])
        x = _parse_cif_float(parts[header_index["_atom_site_fract_x"]])
        y = _parse_cif_float(parts[header_index["_atom_site_fract_y"]])
        z = _parse_cif_float(parts[header_index["_atom_site_fract_z"]])

        try:
            formal_charge = _parse_optional_cif_integer(
                parts,
                formal_charge_idx,
                field_name="atom-site formal charge",
            )
        except ValueError:
            # Treat malformed optional charges as unavailable.
            formal_charge = None
        fc = formal_charge if formal_charge is not None else 0
        fc_valid = formal_charge is not None
        hybridization = "unknown"
        if hybridization_idx is not None and hybridization_idx < len(parts):
            hybridization = _normalize_loop_value(parts[hybridization_idx], default="unknown")
        occupancy = _parse_optional_cif_float(parts, occupancy_idx)
        site_multiplicity = _parse_optional_cif_integer(
            parts,
            multiplicity_idx,
            field_name="atom-site symmetry multiplicity",
            minimum=1,
        )
        attached_hydrogen = _parse_optional_cif_integer(
            parts,
            attached_hydrogen_idx,
            field_name="attached-hydrogen count",
            minimum=0,
        )
        disorder_assembly = _optional_loop_text(parts, disorder_assembly_idx)
        disorder_group = _optional_loop_text(parts, disorder_group_idx)

        atom_labels.append(label)
        atom_types.append(get_atomic_number(element))
        coords.append([x, y, z])
        atom_formal_charges.append(fc)
        atom_formal_charge_mask.append(fc_valid)
        atom_hybridizations.append(hybridization)
        atom_occupancies.append(occupancy)
        atom_site_multiplicities.append(site_multiplicity)
        atom_attached_hydrogens.append(attached_hydrogen or 0)
        atom_attached_hydrogen_mask.append(attached_hydrogen is not None)
        atom_disorder_assemblies.append(disorder_assembly)
        atom_disorder_groups.append(disorder_group)

    if not atom_labels:
        raise ValueError("No atoms found in CIF file")

    frac_coords = torch.tensor(coords, dtype=torch.float32)

    has_disorder = any(
        value not in {"", "0"}
        for value in (*atom_disorder_assemblies, *atom_disorder_groups)
    ) or any(
        occupancy is not None and not np.isclose(occupancy, 1.0, atol=1.0e-6)
        for occupancy in atom_occupancies
    )

    return _AtomSiteData(
        labels=atom_labels,
        atom_types=atom_types,
        frac_coords=frac_coords,
        formal_charges=atom_formal_charges,
        formal_charge_mask=atom_formal_charge_mask,
        hybridizations=atom_hybridizations,
        occupancies=atom_occupancies,
        site_multiplicities=atom_site_multiplicities,
        attached_hydrogens=atom_attached_hydrogens,
        attached_hydrogen_mask=atom_attached_hydrogen_mask,
        disorder_assemblies=atom_disorder_assemblies,
        disorder_groups=atom_disorder_groups,
        has_disorder=has_disorder,
    )


def _parse_bonds(
    content: str | _CifDocument,
) -> tuple[list[tuple[str, str, float, float]], int]:
    bond_data = _parse_bond_data(content)
    return bond_data.bonds, bond_data.num_symmetry_bonds


def _parse_bond_data(content: str | _CifDocument) -> _BondData:
    headers, rows = _parse_loop_rows(
        content,
        required_headers=(
            "_geom_bond_atom_site_label_1",
            "_geom_bond_atom_site_label_2",
            "_geom_bond_distance",
        ),
    )
    if headers is None or rows is None:
        return _BondData([], 0, [], [], [], [], [])

    header_index = {header: idx for idx, header in enumerate(headers)}
    atom1_idx = header_index["_geom_bond_atom_site_label_1"]
    atom2_idx = header_index["_geom_bond_atom_site_label_2"]
    distance_idx = header_index["_geom_bond_distance"]
    order_idx = header_index.get("_geom_bond_order")
    ccdc_type_idx = header_index.get("_ccdc_geom_bond_type")
    valence_idx = header_index.get("_geom_bond_valence")
    sym1_idx = header_index.get("_geom_bond_site_symmetry_1")
    sym2_idx = header_index.get("_geom_bond_site_symmetry_2")
    bonds: list[tuple[str, str, float, float]] = []
    order_authoritative: list[bool] = []
    order_sources: list[str] = []
    valences: list[Optional[float]] = []
    symmetry_1: list[str] = []
    symmetry_2: list[str] = []
    num_symmetry_bonds = 0

    for parts in rows:
        if len(parts) <= max(atom1_idx, atom2_idx, distance_idx):
            continue
        sym1 = _normalize_loop_value(parts[sym1_idx], default="") if sym1_idx is not None and sym1_idx < len(parts) else ""
        sym2 = _normalize_loop_value(parts[sym2_idx], default="") if sym2_idx is not None and sym2_idx < len(parts) else ""
        if _is_nonidentity_bond_symmetry(sym1) or _is_nonidentity_bond_symmetry(sym2):
            num_symmetry_bonds += 1

        atom1 = _normalize_loop_value(parts[atom1_idx], default=parts[atom1_idx])
        atom2 = _normalize_loop_value(parts[atom2_idx], default=parts[atom2_idx])
        distance = _parse_cif_float(parts[distance_idx])
        bond_order: Optional[float] = None
        order_source = "default_single"
        if order_idx is not None and order_idx < len(parts):
            bond_order = _try_parse_bond_order(parts[order_idx])
            if bond_order is not None:
                order_source = "geom_bond_order"
        if bond_order is None and ccdc_type_idx is not None and ccdc_type_idx < len(parts):
            bond_order = _try_parse_ccdc_bond_order(parts[ccdc_type_idx])
            if bond_order is not None:
                order_source = "ccdc_geom_bond_type"
        authoritative = bond_order is not None
        if bond_order is None:
            bond_order = 1.0
        valence = _parse_optional_cif_float(parts, valence_idx)

        bonds.append((atom1, atom2, distance, bond_order))
        order_authoritative.append(authoritative)
        order_sources.append(order_source)
        valences.append(valence)
        symmetry_1.append(sym1)
        symmetry_2.append(sym2)

    return _BondData(
        bonds=bonds,
        num_symmetry_bonds=num_symmetry_bonds,
        order_authoritative=order_authoritative,
        order_sources=order_sources,
        valences=valences,
        symmetry_1=symmetry_1,
        symmetry_2=symmetry_2,
    )


def _try_parse_bond_order(value: str) -> Optional[float]:
    try:
        parsed = _parse_cif_float(value)
    except ValueError:
        return _try_parse_ccdc_bond_order(value)
    return parsed if parsed > 0.0 else None


def _try_parse_ccdc_bond_order(value: str) -> Optional[float]:
    clean = _normalize_loop_value(value, default="").upper()
    return {
        "S": 1.0,
        "SINGLE": 1.0,
        "D": 2.0,
        "DOUBLE": 2.0,
        "T": 3.0,
        "TRIPLE": 3.0,
        "A": 1.5,
        "AROM": 1.5,
        "AROMATIC": 1.5,
        "DELOC": 1.5,
    }.get(clean)


def _parse_ccdc_bond_order(value: str) -> float:
    return _try_parse_ccdc_bond_order(value) or 1.0


def _is_nonidentity_bond_symmetry(value: str) -> bool:
    clean = _normalize_loop_value(value, default="")
    if not clean:
        return False
    # In a CIF symmetry code ``n_abc``, 555 is the zero cell translation.
    # ``1_666`` applies operation 1 after translating by (+1,+1,+1).
    return clean not in {"1_555", "0", "1"}


def _parse_chemical_formula_value(
    content: str | _CifDocument,
    tag: str,
) -> Optional[str]:
    value = _as_cif_document(content).tags.get(tag.lower())
    if value is None:
        return None
    normalized = _normalize_loop_value(value, default="")
    return " ".join(normalized.split()) if normalized else None


def _parse_chemical_formula_moiety(content: str | _CifDocument) -> Optional[str]:
    return _parse_chemical_formula_value(content, "_chemical_formula_moiety")


def _parse_chemical_formula_sum(content: str | _CifDocument) -> Optional[str]:
    return _parse_chemical_formula_value(content, "_chemical_formula_sum")


def _parse_cell_formula_units_z(content: str | _CifDocument) -> Optional[int]:
    value = _parse_tag_value(content, ("_cell_formula_units_Z",))
    if not value:
        return None
    parsed = _parse_cif_float(value)
    if not parsed.is_integer() or parsed <= 0:
        raise ValueError(
            f"Cell formula-units Z must be a positive integer, got {value!r}"
        )
    return int(parsed)


def classify_chemical_formula_moiety(moiety: Optional[str]) -> str:
    if not moiety:
        return "unknown"

    if "+" in moiety or "-" in moiety:
        return "salt"

    if "," in moiety:
        return "cocrystal"

    return "single"


def _parse_tag_value(
    content: str | _CifDocument,
    tags: tuple[str, ...],
) -> str | None:
    document = _as_cif_document(content)
    for tag in tags:
        value = document.tags.get(tag.lower())
        if value is not None:
            return _normalize_loop_value(value, default="")
    return None


def _parse_symop_xyz(content: str | _CifDocument) -> tuple[str, ...]:
    for symop_header in (
        "_symmetry_equiv_pos_as_xyz",
        "_space_group_symop_operation_xyz",
    ):
        headers, rows = _parse_loop_rows(content, (symop_header,))
        if headers and rows:
            symop_idx = headers.index(symop_header)
            values = [
                _normalize_loop_value(row[symop_idx], default="")
                for row in rows
                if len(row) > symop_idx
            ]
            values = [value for value in values if value]
            if values:
                return tuple(values)
        scalar = _parse_tag_value(content, (symop_header,))
        if scalar:
            return (scalar,)
    return ()


def _canonicalize_affine_ops(matrices: np.ndarray) -> tuple[tuple[float, ...], ...]:
    canonical = np.asarray(matrices, dtype=float).copy()
    canonical = np.round(canonical, decimals=6)
    return tuple(sorted(tuple(float(v) for v in matrix.reshape(-1)) for matrix in canonical))


@lru_cache(maxsize=256)
def _hm_symbol_to_spacegroup(symbol: str) -> int | None:
    symbol = _normalize_loop_value(symbol, default="")
    if not symbol:
        return None
    try:
        from pymatgen.symmetry.groups import SpaceGroup
    except Exception:
        return None

    candidates = [symbol]
    compact = symbol.replace(" ", "")
    if compact and compact != symbol:
        candidates.append(compact)
    for candidate in candidates:
        try:
            sg = SpaceGroup(candidate)
            return int(sg.int_number)
        except Exception:
            continue
        if hasattr(SpaceGroup, 'from_symbol'):
            try:
                sg = SpaceGroup.from_symbol(candidate)
                return int(sg.int_number)
            except Exception:
                continue
    return None


@lru_cache(maxsize=256)
def _infer_spacegroup_from_symops(symop_xyz: tuple[str, ...]) -> int | None:
    if not symop_xyz:
        return None
    matrices = []
    for symop in parse_symops(symop_xyz):
        affine = np.eye(4, dtype=float)
        affine[:3, :3] = symop.R
        affine[:3, 3] = symop.t
        matrices.append(affine)
    stacked = np.stack(matrices, axis=0)
    try:
        import spglib

        hall = spglib.get_hall_number_from_symmetry(
            np.rint(stacked[:, :3, :3]).astype(np.intc),
            stacked[:, :3, 3],
            symprec=1.0e-5,
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        return None
    if hall is None:
        return None
    return lattice_template_from_hall_number(int(hall))[0]


@lru_cache(maxsize=256)
def _hall_symbol_to_number(symbol: str) -> int | None:
    normalized = re.sub(r"\s+", "", _normalize_loop_value(symbol, default="")).lower()
    if not normalized:
        return None
    try:
        import spglib

        for hall in range(1, 531):
            sg_type = spglib.get_spacegroup_type(hall)
            candidate = re.sub(r"\s+", "", str(sg_type.hall_symbol)).lower()
            if candidate == normalized:
                return hall
    except (ImportError, RuntimeError, TypeError, ValueError):
        return None
    return None


def _parse_spacegroup_and_symmetry_ops(
    content: str | _CifDocument,
) -> tuple[int, tuple[str, ...]]:
    symop_xyz = _parse_symop_xyz(content)

    number = _parse_tag_value(
        content,
        (
            "_space_group_IT_number",
            "_symmetry_Int_Tables_number",
        ),
    )
    if number:
        parsed_number = _parse_cif_float(number)
        if not parsed_number.is_integer():
            raise ValueError(
                f"Space-group IT number must be an integer, got {number!r}"
            )
        spacegroup = int(parsed_number)
        if not 1 <= spacegroup <= 230:
            raise ValueError(f"Space-group IT number must be in [1, 230], got {spacegroup}")
        return spacegroup, symop_xyz

    hm_symbol = _parse_tag_value(
        content,
        (
            "_space_group_name_H-M_alt",
            "_symmetry_space_group_name_H-M",
        ),
    )
    hm_spacegroup = _hm_symbol_to_spacegroup(hm_symbol) if hm_symbol else None
    if hm_spacegroup is not None:
        return hm_spacegroup, symop_xyz

    hall_symbol = _parse_tag_value(
        content,
        ("_space_group_name_Hall", "_symmetry_space_group_name_Hall"),
    )
    hall_from_symbol = _hall_symbol_to_number(hall_symbol) if hall_symbol else None
    if hall_from_symbol is not None:
        return lattice_template_from_hall_number(hall_from_symbol)[0], symop_xyz

    inferred_spacegroup = _infer_spacegroup_from_symops(symop_xyz) if symop_xyz else None
    if inferred_spacegroup is not None:
        return inferred_spacegroup, symop_xyz

    raise ValueError(
        "CIF is missing a resolvable IT number, H-M/Hall symbol, or explicit "
        "symmetry-operation set; refusing to assume P1"
    )


def _parse_hall_setting(
    content: str | _CifDocument,
    *,
    spacegroup: int,
    symop_xyz: tuple[str, ...],
) -> tuple[int, int, int]:
    """Resolve a Hall setting from authoritative operations with spglib."""

    direct = _parse_tag_value(
        content,
        ("_space_group_hall_number", "_symmetry_space_group_hall_number"),
    )
    hall_number: int | None = None
    if direct:
        parsed = _parse_cif_float(direct)
        if not parsed.is_integer() or not 1 <= int(parsed) <= 530:
            raise ValueError(f"Hall number must be an integer in [1, 530], got {direct!r}")
        hall_number = int(parsed)
    if hall_number is None:
        hall_symbol = _parse_tag_value(
            content,
            ("_space_group_name_Hall", "_symmetry_space_group_name_Hall"),
        )
        if hall_symbol:
            hall_number = _hall_symbol_to_number(hall_symbol)
            if hall_number is None:
                raise ValueError(f"Could not resolve Hall symbol {hall_symbol!r}")

    if symop_xyz:
        parsed_ops = parse_symops(symop_xyz)
        rotations = np.stack([np.rint(op.R).astype(np.intc) for op in parsed_ops])
        translations = np.stack([np.asarray(op.t, dtype=np.float64) for op in parsed_ops])
    else:
        matrices = np.asarray(
            get_sym_ops_from_hall(hall_number)
            if hall_number is not None
            else get_sym_ops(spacegroup),
            dtype=np.float64,
        )
        rotations = np.rint(matrices[:, :3, :3]).astype(np.intc)
        translations = matrices[:, :3, 3]

    matrices = np.zeros((rotations.shape[0], 4, 4), dtype=np.float64)
    matrices[:, :3, :3] = rotations
    matrices[:, :3, 3] = translations
    matrices[:, 3, 3] = 1.0
    operations_spacegroup = spacegroup_number_from_symmetry_ops(matrices)
    if operations_spacegroup != int(spacegroup):
        raise ValueError(
            "CIF symmetry operations resolve to IT spacegroup "
            f"{operations_spacegroup}, not declared spacegroup {spacegroup}"
        )
    if hall_number is not None:
        if not symmetry_ops_match_hall(matrices, hall_number):
            raise ValueError(
                f"Declared Hall number {hall_number} does not match the symmetry "
                "operation basis/unique-axis setting"
            )
    else:
        hall_number = infer_hall_number_from_symmetry_setting(
            matrices, spacegroup=spacegroup
        )
    if hall_number is None:
        raise ValueError(
            "Could not resolve an authoritative Hall setting from the CIF "
            "symmetry operations"
        )

    hall_spacegroup, template = lattice_template_from_hall_number(hall_number)
    if hall_spacegroup != int(spacegroup):
        raise ValueError(
            f"Hall number {hall_number} is incompatible with IT spacegroup {spacegroup}"
        )
    return hall_number, hall_number, template


def filter_hydrogens(cif_data: CifData) -> CifData:

    non_h_indices = [i for i, at in enumerate(cif_data.atom_types) if at != 1]

    if len(non_h_indices) == len(cif_data.atom_types):
        return cif_data


    new_labels = [cif_data.atom_labels[i] for i in non_h_indices]
    new_types = [cif_data.atom_types[i] for i in non_h_indices]
    new_coords = cif_data.frac_coords[non_h_indices]

    def filter_atom_values(values: list) -> list:
        if len(values) != len(cif_data.atom_types):
            return []
        return [values[i] for i in non_h_indices]

    new_atom_formal_charges = filter_atom_values(cif_data.atom_formal_charges)
    new_atom_formal_charge_mask = filter_atom_values(cif_data.atom_formal_charge_mask)
    new_atom_hybridizations = filter_atom_values(cif_data.atom_hybridizations)
    new_atom_occupancies = filter_atom_values(cif_data.atom_occupancies)
    new_atom_site_multiplicities = filter_atom_values(
        cif_data.atom_site_multiplicities
    )
    new_atom_attached_hydrogens = filter_atom_values(
        cif_data.atom_attached_hydrogens
    )
    new_atom_attached_hydrogen_mask = filter_atom_values(
        cif_data.atom_attached_hydrogen_mask
    )
    new_atom_disorder_assemblies = filter_atom_values(
        cif_data.atom_disorder_assemblies
    )
    new_atom_disorder_groups = filter_atom_values(cif_data.atom_disorder_groups)


    non_h_label_set = set(new_labels)

    kept_bond_indices = [
        index
        for index, (atom1, atom2, _distance, _order) in enumerate(cif_data.bonds)
        if atom1 in non_h_label_set and atom2 in non_h_label_set
    ]
    new_bonds = [cif_data.bonds[index] for index in kept_bond_indices]

    def filter_bond_values(values: list) -> list:
        if len(values) != len(cif_data.bonds):
            return []
        return [values[index] for index in kept_bond_indices]

    new_bond_order_authoritative = filter_bond_values(
        cif_data.bond_order_authoritative
    )
    new_bond_order_sources = filter_bond_values(cif_data.bond_order_sources)
    new_bond_valences = filter_bond_values(cif_data.bond_valences)
    new_bond_symmetry_1 = filter_bond_values(cif_data.bond_symmetry_1)
    new_bond_symmetry_2 = filter_bond_values(cif_data.bond_symmetry_2)
    if new_bond_symmetry_1 or new_bond_symmetry_2:
        new_num_symmetry_bonds = sum(
            _is_nonidentity_bond_symmetry(symmetry_1)
            or _is_nonidentity_bond_symmetry(symmetry_2)
            for symmetry_1, symmetry_2 in zip(
                new_bond_symmetry_1 or [""] * len(new_bonds),
                new_bond_symmetry_2 or [""] * len(new_bonds),
            )
        )
    else:
        new_num_symmetry_bonds = (
            cif_data.num_symmetry_bonds
            if len(kept_bond_indices) == len(cif_data.bonds)
            else 0
        )

    if new_atom_occupancies or new_atom_disorder_assemblies or new_atom_disorder_groups:
        new_has_disorder = any(
            value not in {"", "0"}
            for value in (*new_atom_disorder_assemblies, *new_atom_disorder_groups)
        ) or any(
            occupancy is not None
            and not np.isclose(occupancy, 1.0, atol=1.0e-6)
            for occupancy in new_atom_occupancies
        )
    else:
        new_has_disorder = cif_data.has_disorder

    return CifData(
        atom_labels=new_labels,
        atom_types=new_types,
        frac_coords=new_coords,
        lattice=cif_data.lattice,
        bonds=new_bonds,
        spacegroup=cif_data.spacegroup,
        cif_id=cif_data.cif_id,
        chemical_formula_moiety=cif_data.chemical_formula_moiety,
        chemical_formula_sum=cif_data.chemical_formula_sum,
        cell_formula_units_z=cif_data.cell_formula_units_z,
        atom_formal_charges=new_atom_formal_charges,
        atom_formal_charge_mask=new_atom_formal_charge_mask,
        atom_hybridizations=new_atom_hybridizations,
        atom_occupancies=new_atom_occupancies,
        atom_site_multiplicities=new_atom_site_multiplicities,
        atom_attached_hydrogens=new_atom_attached_hydrogens,
        atom_attached_hydrogen_mask=new_atom_attached_hydrogen_mask,
        atom_disorder_assemblies=new_atom_disorder_assemblies,
        atom_disorder_groups=new_atom_disorder_groups,
        has_disorder=new_has_disorder,
        bond_order_authoritative=new_bond_order_authoritative,
        bond_order_sources=new_bond_order_sources,
        bond_valences=new_bond_valences,
        bond_symmetry_1=new_bond_symmetry_1,
        bond_symmetry_2=new_bond_symmetry_2,
        n_symops=cif_data.n_symops,
        symop_xyz=cif_data.symop_xyz,
        num_symmetry_bonds=new_num_symmetry_bonds,
        hall_number=cif_data.hall_number,
        setting_id=cif_data.setting_id,
        lattice_template_id=cif_data.lattice_template_id,
    )


def _count_symmetry_operations(content: str | _CifDocument) -> int:
    for headers in (
        ("_symmetry_equiv_pos_as_xyz",),
        ("_space_group_symop_operation_xyz",),
    ):
        _, rows = _parse_loop_rows(content, headers)
        if rows:
            return max(1, len(rows))
    return 1


def _parse_loop_rows(
    content: str | _CifDocument,
    required_headers: tuple[str, ...],
) -> tuple[Optional[list[str]], Optional[list[list[str]]]]:
    required = tuple(header.lower() for header in required_headers)
    for loop in _as_cif_document(content).loops:
        if all(header in loop.headers for header in required):
            return list(loop.headers), [list(row) for row in loop.rows]
    return None, None


def _normalize_loop_value(value: str, default: str = "") -> str:
    cleaned = str(value).strip().strip("'\"")
    if cleaned in {"?", ".", ""}:
        return default
    return cleaned


def _optional_loop_text(parts: list[str], index: Optional[int]) -> str:
    if index is None or index >= len(parts):
        return ""
    return _normalize_loop_value(parts[index], default="")


def _parse_optional_cif_float(
    parts: list[str],
    index: Optional[int],
) -> Optional[float]:
    value = _optional_loop_text(parts, index)
    return _parse_cif_float(value) if value else None


def _parse_optional_cif_integer(
    parts: list[str],
    index: Optional[int],
    *,
    field_name: str,
    minimum: Optional[int] = None,
) -> Optional[int]:
    value = _optional_loop_text(parts, index)
    if not value:
        return None
    parsed = _parse_cif_float(value)
    if not parsed.is_integer() or (minimum is not None and parsed < minimum):
        constraint = "an integer"
        if minimum is not None:
            constraint += f" >= {minimum}"
        raise ValueError(
            f"CIF {field_name} must be {constraint}, got {value!r}"
        )
    return int(parsed)


def _parse_cif_float(value: str) -> float:
    text = _normalize_loop_value(value, default="")
    if not text:
        raise ValueError(f"Missing CIF numeric value: {value!r}")
    # Standard uncertainties may precede an exponent: 1.234(5)e2 -> 123.4.
    text = re.sub(r"\(\d+\)(?=(?:[eE][+-]?\d+)?$)", "", text)
    if not re.fullmatch(
        r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?",
        text,
    ):
        raise ValueError(f"Invalid CIF numeric value: {value!r}")
    return float(text)
