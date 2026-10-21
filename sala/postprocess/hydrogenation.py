"""Deterministic, formula-constrained hydrogenation of heavy-atom ASUs.

This module deliberately separates the chemically important decision
(``which heavy atom owns how many hydrogens, in which formal-charge state``)
from the approximate coordinate initialization.  The former is a small
discrete optimization problem with hard formula/evidence constraints; the
latter only supplies non-pathological starting coordinates for subsequent
MLFF relaxation.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from sala.core.constants import METAL_ATOMIC_NUMBERS
from sala.core.types import (
    HYBRIDIZATION_TO_INDEX,
    encode_hybridization,
    get_element_symbol,
)
from sala.data.formula import (
    HYDROGEN_ISOTOPES,
    FormulaError,
    FormulaTargets,
    resolve_formula_targets,
)


class HydrogenationError(ValueError):
    """Base class for strict hydrogenation failures."""


class UnsupportedHydrogenationError(HydrogenationError):
    """Raised when the evidence lies outside the supported chemistry domain."""


class HydrogenationNoSolutionError(HydrogenationError):
    """Raised when formula, charge, and local valence constraints conflict."""


class AmbiguousHydrogenationError(HydrogenationError):
    """Raised when multiple equally preferred atom-level assignments remain."""


@dataclass(frozen=True)
class ResolvedValenceState:
    formal_charge: int
    hydrogens: int
    valence: float
    label: str
    penalty: float


@dataclass(frozen=True)
class HydrogenationPlan:
    """A resolved atom-level hydrogenation and formal-valence assignment."""

    parent_h_counts: tuple[int, ...]
    formal_charges: tuple[int, ...]
    formal_charge_resolved: tuple[bool, ...]
    resolved_states: tuple[ResolvedValenceState, ...]
    component_h_counts: tuple[int, ...]
    component_charge_totals: tuple[int, ...]
    parent_h_isotopes: tuple[tuple[str, ...], ...]
    isotope_counts: tuple[tuple[str, int], ...]
    target_hydrogen_count: int
    target_total_charge: int
    evidence: tuple[str, ...]
    charge_evidence_fallback_components: tuple[int, ...]
    formula_source: str | None
    status: str
    ambiguity: bool
    ambiguous_parent_atoms: tuple[int, ...]
    num_optimal_solutions: int
    score: float


@dataclass(frozen=True)
class HydrogenPlacement:
    hydrogen_frac_coords: Tensor
    parent_indices: Tensor
    isotopes: tuple[str, ...]
    bond_lengths: Tensor


@dataclass(frozen=True)
class HydrogenatedStructure:
    """Heavy atoms followed, without reordering, by newly placed hydrogens."""

    atom_types: Tensor
    frac_coords: Tensor
    atom_labels: tuple[str, ...]
    hydrogen_parent_indices: Tensor
    hydrogen_isotopes: tuple[str, ...]
    added_edge_index: Tensor
    plan: HydrogenationPlan


@dataclass(frozen=True)
class _ValenceTemplate:
    charge: int
    valences: tuple[int, ...]
    label: str
    penalty: float = 0.0


@dataclass
class _Best:
    score: float
    solution: tuple[ResolvedValenceState, ...]
    hydrogen_signature: tuple[int, ...]
    charge_signature: tuple[int, ...]
    hydrogen_ambiguous: tuple[bool, ...]
    charge_ambiguous: tuple[bool, ...]


_VALENCE_TEMPLATES: dict[int, tuple[_ValenceTemplate, ...]] = {
    5: (
        _ValenceTemplate(-1, (4,), "borate", 0.35),
        _ValenceTemplate(0, (3,), "boron"),
        _ValenceTemplate(1, (2,), "boron_cation", 0.55),
    ),
    6: (
        _ValenceTemplate(-1, (3,), "carbanion", 0.55),
        _ValenceTemplate(0, (4,), "carbon"),
        _ValenceTemplate(1, (3,), "carbocation", 0.55),
    ),
    7: (
        _ValenceTemplate(-1, (2,), "nitrogen_anion", 0.45),
        _ValenceTemplate(0, (3,), "nitrogen"),
        _ValenceTemplate(1, (4,), "nitrogen_cation", 0.35),
    ),
    8: (
        _ValenceTemplate(-1, (1,), "oxygen_anion", 0.30),
        _ValenceTemplate(0, (2,), "oxygen"),
        _ValenceTemplate(1, (3,), "oxygen_cation", 0.55),
    ),
    9: (
        _ValenceTemplate(-1, (0,), "fluoride", 0.20),
        _ValenceTemplate(0, (1,), "fluorine"),
    ),
    14: (
        _ValenceTemplate(-1, (3,), "silicon_anion", 0.60),
        _ValenceTemplate(0, (4,), "silicon"),
        _ValenceTemplate(1, (3,), "silicon_cation", 0.60),
    ),
    15: (
        _ValenceTemplate(-1, (2, 4), "phosphorus_anion", 0.50),
        _ValenceTemplate(0, (3, 5), "phosphorus", 0.0),
        _ValenceTemplate(1, (4, 6), "phosphorus_cation", 0.35),
    ),
    16: (
        _ValenceTemplate(-1, (1, 3, 5), "sulfur_anion", 0.45),
        _ValenceTemplate(0, (2, 4, 6), "sulfur", 0.0),
        _ValenceTemplate(1, (3, 5), "sulfur_cation", 0.50),
    ),
    17: (
        _ValenceTemplate(-1, (0,), "chloride", 0.20),
        _ValenceTemplate(0, (1,), "chlorine"),
    ),
    33: (
        _ValenceTemplate(-1, (2, 4), "arsenic_anion", 0.55),
        _ValenceTemplate(0, (3, 5), "arsenic"),
        _ValenceTemplate(1, (4,), "arsenic_cation", 0.45),
    ),
    34: (
        _ValenceTemplate(-1, (1, 3, 5), "selenium_anion", 0.50),
        _ValenceTemplate(0, (2, 4, 6), "selenium"),
        _ValenceTemplate(1, (3, 5), "selenium_cation", 0.55),
    ),
    35: (
        _ValenceTemplate(-1, (0,), "bromide", 0.20),
        _ValenceTemplate(0, (1,), "bromine"),
    ),
    53: (
        _ValenceTemplate(-1, (0,), "iodide", 0.20),
        _ValenceTemplate(0, (1,), "iodine"),
    ),
}

_MAX_HYDROGENS = {
    5: 4,
    6: 4,
    7: 4,
    8: 2,
    9: 1,
    14: 4,
    15: 4,
    16: 2,
    17: 1,
    33: 3,
    34: 2,
    35: 1,
    53: 1,
}

_METAL_CHARGES: dict[int, tuple[int, ...]] = {
    3: (0, 1),
    4: (0, 2),
    11: (0, 1),
    12: (0, 2),
    13: (0, 3),
    19: (0, 1),
    20: (0, 2),
    21: (0, 3),
    22: (0, 2, 3, 4),
    23: (0, 2, 3, 4, 5),
    24: (0, 2, 3, 6),
    25: (0, 2, 3, 4),
    26: (0, 2, 3),
    27: (0, 2, 3),
    28: (0, 2, 3),
    29: (0, 1, 2),
    30: (0, 2),
    31: (0, 3),
    37: (0, 1),
    38: (0, 2),
    39: (0, 3),
    40: (0, 4),
    41: (0, 3, 5),
    42: (0, 4, 6),
    43: (0, 4, 7),
    44: (0, 2, 3, 4),
    45: (0, 1, 3),
    46: (0, 2, 4),
    47: (0, 1),
    48: (0, 2),
    49: (0, 3),
    50: (0, 2, 4),
    55: (0, 1),
    56: (0, 2),
    57: (0, 3),
    58: (0, 3, 4),
    63: (0, 2, 3),
    64: (0, 3),
    70: (0, 2, 3),
    71: (0, 3),
    72: (0, 4),
    73: (0, 5),
    74: (0, 4, 6),
    75: (0, 4, 7),
    76: (0, 2, 3, 4),
    77: (0, 1, 3, 4),
    78: (0, 2, 4),
    79: (0, 1, 3),
    80: (0, 1, 2),
    81: (0, 1, 3),
    82: (0, 2, 4),
    83: (0, 3, 5),
}

_XH_BOND_LENGTH_A = {
    5: 1.19,
    6: 1.09,
    7: 1.01,
    8: 0.98,
    9: 0.92,
    14: 1.48,
    15: 1.42,
    16: 1.34,
    17: 1.27,
    33: 1.52,
    34: 1.46,
    35: 1.41,
    53: 1.61,
}

_BOND_ORDER_VALUES = (1.0, 1.5, 2.0, 3.0)
_SCORE_TOL = 1.0e-9
_PERIODIC_IMAGE_OFFSETS = torch.tensor(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=torch.int8,
)
_PERIODIC_DISTANCE_OTHER_CHUNK_SIZE = 256


def _one_dimensional(values: Sequence | Tensor, length: int, *, name: str) -> list:
    tensor = torch.as_tensor(values) if torch.is_tensor(values) else None
    if tensor is not None:
        if tensor.ndim != 1 or int(tensor.numel()) != length:
            raise HydrogenationError(f"{name} must have shape [{length}]")
        return tensor.detach().cpu().tolist()
    output = list(values)
    if len(output) != length:
        raise HydrogenationError(f"{name} must contain {length} values")
    return output


def _optional_integral_values(
    values: Sequence[int | float | None] | Tensor | None,
    length: int,
    *,
    name: str,
    nonnegative: bool = False,
) -> tuple[int | None, ...]:
    if values is None:
        return (None,) * length
    raw = _one_dimensional(values, length, name=name)
    output: list[int | None] = []
    for value in raw:
        if value is None:
            output.append(None)
            continue
        numeric = float(value)
        integer = round(numeric)
        if not math.isfinite(numeric) or abs(numeric - integer) > 1.0e-6:
            raise HydrogenationError(f"{name} must contain integral values")
        if nonnegative and integer < 0:
            raise HydrogenationError(f"{name} must be non-negative")
        output.append(integer)
    return tuple(output)


def _masked_atom_values(
    values: Sequence[int | float] | Tensor | None,
    mask: Sequence[bool] | Tensor | None,
    length: int,
    *,
    name: str,
    nonnegative: bool = False,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    if values is None:
        if mask is not None:
            raise HydrogenationError(f"{name}_mask was supplied without {name}")
        return (0,) * length, (False,) * length
    parsed = _optional_integral_values(
        values, length, name=name, nonnegative=nonnegative
    )
    if any(value is None for value in parsed):
        raise HydrogenationError(f"{name} cannot contain None; use its mask")
    if mask is None:
        parsed_mask = (True,) * length
    else:
        parsed_mask = tuple(
            bool(value) for value in _one_dimensional(mask, length, name=f"{name}_mask")
        )
    return tuple(int(value) for value in parsed if value is not None), parsed_mask


def _hybridization_values(
    values: Sequence[int | str] | Tensor | None, length: int
) -> tuple[int, ...]:
    if values is None:
        return (HYBRIDIZATION_TO_INDEX["unknown"],) * length
    raw = _one_dimensional(values, length, name="hybridization")
    output = []
    for value in raw:
        if isinstance(value, str):
            output.append(encode_hybridization(value))
        else:
            integer = int(value)
            if integer not in HYBRIDIZATION_TO_INDEX.values():
                raise HydrogenationError(f"Unknown hybridization index {integer}")
            output.append(integer)
    return tuple(output)


def _component_values(
    component_id: Sequence[int] | Tensor, num_atoms: int
) -> tuple[tuple[int, ...], int]:
    values = tuple(
        int(value)
        for value in _one_dimensional(component_id, num_atoms, name="component_id")
    )
    if not values and num_atoms == 0:
        return (), 0
    unique = sorted(set(values))
    if unique != list(range(len(unique))):
        raise HydrogenationError("component_id must be contiguous and start at zero")
    return values, len(unique)


def _canonical_bonds(
    *,
    edge_index: Sequence | Tensor,
    bond_orders: Sequence[float] | Tensor,
    bond_order_mask: Sequence[bool] | Tensor | None,
    bond_order_encoding: str,
    num_atoms: int,
    component_id: Sequence[int],
    allow_unknown: bool,
) -> tuple[tuple[tuple[int, int, float], ...], bool]:
    edges = torch.as_tensor(edge_index).detach().cpu()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise HydrogenationError("edge_index must have shape [2, E]")
    num_edges = int(edges.shape[1])
    orders = [
        float(value)
        for value in _one_dimensional(bond_orders, num_edges, name="bond_orders")
    ]
    if bond_order_mask is None:
        authoritative = [True] * num_edges
    else:
        authoritative = [
            bool(value)
            for value in _one_dimensional(
                bond_order_mask, num_edges, name="bond_order_mask"
            )
        ]
    if bond_order_encoding not in {"order", "index"}:
        raise HydrogenationError("bond_order_encoding must be 'order' or 'index'")

    unique: dict[tuple[int, int], float] = {}
    all_authoritative = True
    for position in range(num_edges):
        left = int(edges[0, position])
        right = int(edges[1, position])
        if not (0 <= left < num_atoms and 0 <= right < num_atoms):
            raise HydrogenationError("edge_index contains an out-of-range atom index")
        if left == right:
            raise UnsupportedHydrogenationError("Self bonds are not supported")
        if component_id[left] != component_id[right]:
            raise HydrogenationError(
                "A covalent edge connects different component_id values"
            )
        if not authoritative[position]:
            all_authoritative = False
            if not allow_unknown:
                raise UnsupportedHydrogenationError(
                    f"Bond order for edge ({left}, {right}) is not authoritative"
                )
        order = orders[position]
        if bond_order_encoding == "index":
            index = round(order)
            if abs(order - index) > 1.0e-6 or not 0 <= index < len(_BOND_ORDER_VALUES):
                raise UnsupportedHydrogenationError(f"Invalid bond-order index {order}")
            order = _BOND_ORDER_VALUES[index]
        nearest = min(_BOND_ORDER_VALUES, key=lambda candidate: abs(candidate - order))
        if not math.isfinite(order) or abs(nearest - order) > 0.05:
            if authoritative[position] or not allow_unknown:
                raise UnsupportedHydrogenationError(f"Unsupported bond order {order}")
            nearest = 1.0
        key = (min(left, right), max(left, right))
        previous = unique.get(key)
        if previous is not None and abs(previous - nearest) > 1.0e-6:
            raise HydrogenationError(f"Conflicting bond orders for edge {key}")
        unique[key] = nearest
    return (
        tuple((left, right, unique[(left, right)]) for left, right in sorted(unique)),
        all_authoritative,
    )


def _merge_targets(
    primary: Sequence[int | None],
    secondary: Sequence[int | None] | None,
    *,
    name: str,
    nonnegative: bool,
) -> tuple[int | None, ...]:
    if secondary is None:
        return tuple(primary)
    parsed = _optional_integral_values(
        secondary, len(primary), name=name, nonnegative=nonnegative
    )
    output = []
    for first, second in zip(primary, parsed):
        if first is not None and second is not None and first != second:
            raise HydrogenationError(f"Conflicting {name}: {first} versus {second}")
        output.append(first if first is not None else second)
    return tuple(output)


def _merge_scalar(
    primary: int | None,
    secondary: float | None,
    *,
    name: str,
    nonnegative: bool,
) -> int | None:
    if secondary is None:
        return primary
    numeric = float(secondary)
    value = round(numeric)
    if not math.isfinite(numeric) or abs(numeric - value) > 1.0e-6:
        raise HydrogenationError(f"{name} must be integral")
    if nonnegative and value < 0:
        raise HydrogenationError(f"{name} must be non-negative")
    if primary is not None and primary != value:
        raise HydrogenationError(f"Conflicting {name}: {primary} versus {value}")
    return value


def _merge_charge_targets(
    formula_targets: Sequence[int | None],
    input_targets: Sequence[int | None],
    *,
    mode: str,
) -> tuple[int | None, ...]:
    if mode not in {"strict", "formula", "input"}:
        raise HydrogenationError(
            "charge_evidence_mode must be 'strict', 'formula', or 'input'"
        )
    output: list[int | None] = []
    for formula_value, input_value in zip(formula_targets, input_targets):
        if mode == "strict" and (
            formula_value is not None
            and input_value is not None
            and formula_value != input_value
        ):
            raise HydrogenationError(
                "Component charge metadata conflicts with the chemical formula"
            )
        if mode == "input":
            output.append(input_value if input_value is not None else formula_value)
        else:
            output.append(formula_value if formula_value is not None else input_value)
    return tuple(output)


def _hybrid_penalty(
    hybridization: int,
    *,
    atomic_number: int,
    degree: int,
    hydrogens: int,
    aromatic: bool,
    max_order: float,
) -> float:
    total_degree = degree + hydrogens
    if hybridization in {
        HYBRIDIZATION_TO_INDEX["unknown"],
        HYBRIDIZATION_TO_INDEX["coord"],
    }:
        return 0.0
    if hybridization == HYBRIDIZATION_TO_INDEX["sp"]:
        return 0.0 if total_degree <= 2 and max_order >= 2.0 else 0.8
    if hybridization == HYBRIDIZATION_TO_INDEX["sp2"]:
        penalty = 0.0 if total_degree <= 3 else 0.8 * (total_degree - 3)
        if not aromatic and max_order < 1.5 and atomic_number in {5, 6, 7}:
            penalty += 0.15
        return penalty
    if hybridization == HYBRIDIZATION_TO_INDEX["sp3"]:
        penalty = 0.6 if aromatic or max_order > 1.5 else 0.0
        if total_degree > 4:
            penalty += float(total_degree - 4)
        return penalty
    return 0.0


def _atom_candidates(
    *,
    atomic_number: int,
    bond_order_sum: float,
    degree: int,
    aromatic: bool,
    max_order: float,
    hybridization: int,
    known_charge: int | None,
    known_hydrogens: int | None,
) -> tuple[ResolvedValenceState, ...]:
    if atomic_number == 1:
        raise UnsupportedHydrogenationError("Input graph already contains hydrogen")
    if atomic_number in METAL_ATOMIC_NUMBERS:
        if known_hydrogens not in {None, 0}:
            raise UnsupportedHydrogenationError(
                "Hydrogen bound to a metal is outside the supported domain"
            )
        charges = (
            (known_charge,)
            if known_charge is not None
            else _METAL_CHARGES.get(atomic_number, (0, 1, 2, 3))
        )
        return tuple(
            ResolvedValenceState(
                formal_charge=int(charge),
                hydrogens=0,
                valence=0.0,
                label="metal_no_hydrogen",
                penalty=0.25 if int(charge) == 0 else 0.35,
            )
            for charge in charges
        )

    templates = _VALENCE_TEMPLATES.get(atomic_number)
    if templates is None:
        if known_hydrogens not in {None, 0}:
            raise UnsupportedHydrogenationError(
                f"Hydrogenation templates do not cover {get_element_symbol(atomic_number)}"
            )
        charge = known_charge if known_charge is not None else 0
        return (
            ResolvedValenceState(
                formal_charge=charge,
                hydrogens=0,
                valence=bond_order_sum,
                label="unsupported_element_no_hydrogen",
                penalty=0.0,
            ),
        )

    candidates: dict[tuple[int, int], ResolvedValenceState] = {}
    for template in templates:
        if known_charge is not None and template.charge != known_charge:
            continue
        if known_charge is None:
            # Keep the hard domain chemically conservative. Ordinary closed-
            # shell C/Si ions are not inferred merely to make a global formula
            # balance; they require explicit localized charge evidence.
            if atomic_number in {6, 14} and template.charge != 0:
                continue
            if atomic_number == 5:
                if template.charge > 0:
                    continue
                if template.charge < 0 and degree < 4 and bond_order_sum < 3.5:
                    continue
            if atomic_number == 15:
                if template.charge > 0 and degree < 4 and bond_order_sum < 4.0:
                    continue
                if template.charge < 0 and degree > 2:
                    continue
            if atomic_number in {9, 17, 35, 53}:
                if degree > 0 and template.charge != 0:
                    continue
                if degree == 0 and template.charge not in {-1, 0}:
                    continue
        for valence_index, valence in enumerate(template.valences):
            difference = float(valence) - bond_order_sum
            hydrogens = round(difference)
            if abs(difference - hydrogens) > 1.0e-6 or hydrogens < 0:
                continue
            if hydrogens > _MAX_HYDROGENS.get(atomic_number, 4):
                continue
            if known_hydrogens is not None and hydrogens != known_hydrogens:
                continue
            penalty = template.penalty + 0.12 * valence_index
            penalty += _hybrid_penalty(
                hybridization,
                atomic_number=atomic_number,
                degree=degree,
                hydrogens=hydrogens,
                aromatic=aromatic,
                max_order=max_order,
            )
            state = ResolvedValenceState(
                formal_charge=template.charge,
                hydrogens=hydrogens,
                valence=float(valence),
                label=template.label,
                penalty=penalty,
            )
            key = (hydrogens, template.charge)
            previous = candidates.get(key)
            if previous is None or state.penalty < previous.penalty:
                candidates[key] = state

    # A 1.5 aromatic edge is a graph encoding, not an independently additive
    # integer valence. Use aromatic coordination degree for local H capacity;
    # this covers mixed 1.0/1.5 rings and fused sites with BOS 2.5/3.5/4.5.
    aromatic_special: list[ResolvedValenceState] = []
    if aromatic and atomic_number == 6 and degree <= 4:
        aromatic_h = max(0, 3 - degree)
        for template in templates:
            if known_charge is not None and template.charge != known_charge:
                continue
            aromatic_special.append(
                ResolvedValenceState(
                    template.charge,
                    aromatic_h,
                    bond_order_sum + aromatic_h,
                    "aromatic_carbon",
                    template.penalty,
                )
            )
    if aromatic and atomic_number == 5 and degree <= 4:
        for template in templates:
            if known_charge is not None and template.charge != known_charge:
                continue
            aromatic_h = max(0, (4 if template.charge == -1 else 3) - degree)
            aromatic_special.append(
                ResolvedValenceState(
                    template.charge,
                    aromatic_h,
                    bond_order_sum + aromatic_h,
                    "aromatic_boron",
                    template.penalty,
                )
            )
    if aromatic and atomic_number == 7 and degree in {2, 3}:
        if known_charge in {None, 0}:
            aromatic_special.append(
                ResolvedValenceState(0, 0, bond_order_sum, "aromatic_pyridine_n", 0.0)
            )
            if degree == 2:
                aromatic_special.append(
                    ResolvedValenceState(
                        0, 1, bond_order_sum + 1.0, "aromatic_pyrrolic_n", 0.08
                    )
                )
        if known_charge in {None, 1}:
            aromatic_h = max(0, 3 - degree)
            aromatic_special.append(
                ResolvedValenceState(
                    1,
                    aromatic_h,
                    bond_order_sum + aromatic_h,
                    "aromatic_nitrogen_cation",
                    0.35,
                )
            )
        if known_charge in {None, -1}:
            aromatic_special.append(
                ResolvedValenceState(
                    -1, 0, bond_order_sum, "aromatic_nitrogen_anion", 0.45
                )
            )
    if (
        aromatic
        and degree >= 2
        and atomic_number in {8, 16, 34}
        and known_charge in {None, 0}
    ):
        aromatic_special.append(
            ResolvedValenceState(
                0, 0, bond_order_sum, "aromatic_lone_pair_heteroatom", 0.0
            )
        )
    for state in aromatic_special:
        if known_hydrogens is not None and state.hydrogens != known_hydrogens:
            continue
        key = (state.hydrogens, state.formal_charge)
        previous = candidates.get(key)
        if previous is None or state.penalty < previous.penalty:
            candidates[key] = state

    return tuple(
        sorted(
            candidates.values(),
            key=lambda state: (
                state.penalty,
                state.hydrogens,
                state.formal_charge,
                state.label,
            ),
        )
    )


def _record_best(
    table: dict[tuple[int, int], _Best],
    key: tuple[int, int],
    score: float,
    solution: tuple[ResolvedValenceState, ...],
    hydrogen_signature: tuple[int, ...],
    charge_signature: tuple[int, ...],
    hydrogen_ambiguous: tuple[bool, ...],
    charge_ambiguous: tuple[bool, ...],
) -> None:
    current = table.get(key)
    if current is None:
        table[key] = _Best(
            score=score,
            solution=solution,
            hydrogen_signature=hydrogen_signature,
            charge_signature=charge_signature,
            hydrogen_ambiguous=hydrogen_ambiguous,
            charge_ambiguous=charge_ambiguous,
        )
        return
    merged_hydrogen_ambiguity = tuple(
        left_ambiguous or right_ambiguous or left_h != right_h
        for left_ambiguous, right_ambiguous, left_h, right_h in zip(
            current.hydrogen_ambiguous,
            hydrogen_ambiguous,
            current.hydrogen_signature,
            hydrogen_signature,
        )
    )
    if any(merged_hydrogen_ambiguity):
        merged_charge_ambiguity = (True,) * len(current.charge_signature)
    else:
        merged_charge_ambiguity = tuple(
            left_ambiguous or right_ambiguous or left_charge != right_charge
            for left_ambiguous, right_ambiguous, left_charge, right_charge in zip(
                current.charge_ambiguous,
                charge_ambiguous,
                current.charge_signature,
                charge_signature,
            )
        )

    replace_best = score < current.score - _SCORE_TOL
    if abs(score - current.score) <= _SCORE_TOL:
        signature = tuple(
            (state.hydrogens, state.formal_charge, state.label) for state in solution
        )
        current_signature = tuple(
            (state.hydrogens, state.formal_charge, state.label)
            for state in current.solution
        )
        replace_best = signature < current_signature
    table[key] = _Best(
        score=score if replace_best else current.score,
        solution=solution if replace_best else current.solution,
        hydrogen_signature=current.hydrogen_signature,
        charge_signature=current.charge_signature,
        hydrogen_ambiguous=merged_hydrogen_ambiguity,
        charge_ambiguous=merged_charge_ambiguity,
    )


def _component_options(
    atom_indices: Sequence[int],
    candidates: Sequence[tuple[ResolvedValenceState, ...]],
    target_h: int | None,
    target_q: int | None,
    max_h: int,
) -> list[tuple[int, int, _Best]]:
    table: dict[tuple[int, int], _Best] = {(0, 0): _Best(0.0, (), (), (), (), ())}
    suffix_min_h = [0] * (len(atom_indices) + 1)
    suffix_max_h = [0] * (len(atom_indices) + 1)
    suffix_min_q = [0] * (len(atom_indices) + 1)
    suffix_max_q = [0] * (len(atom_indices) + 1)
    for position in range(len(atom_indices) - 1, -1, -1):
        atom_states = candidates[atom_indices[position]]
        suffix_min_h[position] = suffix_min_h[position + 1] + min(
            state.hydrogens for state in atom_states
        )
        suffix_max_h[position] = suffix_max_h[position + 1] + max(
            state.hydrogens for state in atom_states
        )
        suffix_min_q[position] = suffix_min_q[position + 1] + min(
            state.formal_charge for state in atom_states
        )
        suffix_max_q[position] = suffix_max_q[position + 1] + max(
            state.formal_charge for state in atom_states
        )

    for position, atom_index in enumerate(atom_indices):
        updated: dict[tuple[int, int], _Best] = {}
        remaining = position + 1
        for (hydrogen_sum, charge_sum), best in table.items():
            for state in candidates[atom_index]:
                next_h = hydrogen_sum + state.hydrogens
                next_q = charge_sum + state.formal_charge
                if next_h > max_h:
                    continue
                if target_h is not None and not (
                    next_h + suffix_min_h[remaining]
                    <= target_h
                    <= next_h + suffix_max_h[remaining]
                ):
                    continue
                if target_q is not None and not (
                    next_q + suffix_min_q[remaining]
                    <= target_q
                    <= next_q + suffix_max_q[remaining]
                ):
                    continue
                _record_best(
                    updated,
                    (next_h, next_q),
                    best.score + state.penalty,
                    best.solution + (state,),
                    best.hydrogen_signature + (state.hydrogens,),
                    best.charge_signature + (state.formal_charge,),
                    best.hydrogen_ambiguous + (False,),
                    best.charge_ambiguous + (False,),
                )
        table = updated
        if not table:
            break
    output = []
    for (hydrogen_sum, charge_sum), best in table.items():
        if target_h is not None and hydrogen_sum != target_h:
            continue
        if target_q is not None and charge_sum != target_q:
            continue
        output.append((hydrogen_sum, charge_sum, best))
    return output


def _assign_isotopes(
    *,
    parent_h_counts: Sequence[int],
    component_id: Sequence[int],
    formula_targets: FormulaTargets | None,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, int], ...]]:
    assignments = [["H"] * int(count) for count in parent_h_counts]
    if formula_targets is None:
        return tuple(tuple(value) for value in assignments), (
            ("H", sum(parent_h_counts)),
        ) if sum(parent_h_counts) else ()

    locked_slots: set[tuple[int, int]] = set()
    for component, isotope_budget in enumerate(
        formula_targets.component_isotope_counts
    ):
        if isotope_budget is None:
            continue
        slots = [
            (atom_index, slot)
            for atom_index, atom_component in enumerate(component_id)
            if atom_component == component
            for slot in range(parent_h_counts[atom_index])
        ]
        expanded = [
            symbol for symbol, count in isotope_budget for _ in range(int(count))
        ]
        if len(expanded) != len(slots):
            raise HydrogenationNoSolutionError(
                "Component isotope budget disagrees with inferred hydrogens"
            )
        for location, symbol in zip(slots, expanded):
            assignments[location[0]][location[1]] = symbol
            locked_slots.add(location)

    target = Counter(dict(formula_targets.total_isotope_counts))
    assigned = Counter(symbol for values in assignments for symbol in values)
    # Reassign only unlocked ordinary-H slots to satisfy any global D/T budget.
    unlocked = [
        (atom_index, slot)
        for atom_index, values in enumerate(assignments)
        for slot, symbol in enumerate(values)
        if (atom_index, slot) not in locked_slots and symbol == "H"
    ]
    cursor = 0
    for symbol in ("D", "T"):
        needed = target[symbol] - assigned[symbol]
        if needed < 0 or cursor + needed > len(unlocked):
            raise HydrogenationNoSolutionError(
                "Global isotope budget disagrees with inferred hydrogens"
            )
        for atom_index, slot in unlocked[cursor : cursor + needed]:
            assignments[atom_index][slot] = symbol
        cursor += needed
    final_counts = Counter(symbol for values in assignments for symbol in values)
    if any(final_counts[symbol] != target[symbol] for symbol in HYDROGEN_ISOTOPES):
        raise HydrogenationNoSolutionError("Formula isotope counts were not preserved")
    return (
        tuple(tuple(value) for value in assignments),
        tuple(
            (symbol, final_counts[symbol])
            for symbol in HYDROGEN_ISOTOPES
            if final_counts[symbol]
        ),
    )


def _ambiguous_isotope_parents(
    *,
    parent_h_counts: Sequence[int],
    component_id: Sequence[int],
    formula_targets: FormulaTargets | None,
) -> tuple[int, ...]:
    """Find H-bearing parents whose isotope identity is not localized."""

    if formula_targets is None:
        return ()
    ambiguous: set[int] = set()
    localized_components: set[int] = set()
    localized_counts: Counter[str] = Counter()

    def inspect_scope(parents: list[int], counts: Counter[str]) -> None:
        species = [symbol for symbol in HYDROGEN_ISOTOPES if counts[symbol] > 0]
        if len(species) > 1 and len(parents) > 1:
            ambiguous.update(parents)

    for component, isotope_budget in enumerate(
        formula_targets.component_isotope_counts
    ):
        if isotope_budget is None:
            continue
        localized_components.add(component)
        counts = Counter(dict(isotope_budget))
        localized_counts.update(counts)
        parents = [
            atom_index
            for atom_index, atom_component in enumerate(component_id)
            if atom_component == component and parent_h_counts[atom_index] > 0
        ]
        inspect_scope(parents, counts)

    remaining_counts = Counter(dict(formula_targets.total_isotope_counts))
    remaining_counts.subtract(localized_counts)
    if any(count < 0 for count in remaining_counts.values()):
        raise HydrogenationNoSolutionError(
            "Component isotope budgets exceed the formula total"
        )
    remaining_parents = [
        atom_index
        for atom_index, atom_component in enumerate(component_id)
        if atom_component not in localized_components
        and parent_h_counts[atom_index] > 0
    ]
    inspect_scope(remaining_parents, remaining_counts)
    return tuple(sorted(ambiguous))


def infer_hydrogenation_plan(
    *,
    atom_types: Sequence[int] | Tensor,
    component_id: Sequence[int] | Tensor,
    edge_index: Sequence | Tensor,
    bond_orders: Sequence[float] | Tensor,
    bond_order_encoding: str = "order",
    bond_order_mask: Sequence[bool] | Tensor | None = None,
    formula_targets: FormulaTargets | None = None,
    formula_moiety: str | None = None,
    formula_sum: str | None = None,
    formula_units_z: float | str | None = None,
    n_symops: int | None = None,
    enforce_formula_units: bool = False,
    component_hydrogen_counts: Sequence[int | None] | Tensor | None = None,
    total_hydrogen_count: int | None = None,
    atom_formal_charges: Sequence[int] | Tensor | None = None,
    atom_charge_mask: Sequence[bool] | Tensor | None = None,
    component_formal_charges: Sequence[int | None] | Tensor | None = None,
    total_charge: int | None = None,
    hybridization: Sequence[int | str] | Tensor | None = None,
    attached_hydrogens: Sequence[int] | Tensor | None = None,
    attached_hydrogen_mask: Sequence[bool] | Tensor | None = None,
    charge_evidence_mode: str = "formula",
    strict: bool = True,
) -> HydrogenationPlan:
    """Infer a unique formula-constrained hydrogenation plan.

    Actual bond orders (1, 1.5, 2, 3) are expected by default. Set
    ``bond_order_encoding='index'`` only for SALA's categorical edge encoding.
    Unknown bond orders are a hard failure when ``bond_order_mask`` is false.
    """

    atom_values = tuple(
        int(value)
        for value in (
            atom_types.detach().cpu().tolist()
            if torch.is_tensor(atom_types)
            else atom_types
        )
    )
    num_atoms = len(atom_values)
    components, num_components = _component_values(component_id, num_atoms)
    if num_atoms == 0:
        raise HydrogenationError("Cannot hydrogenate an empty structure")
    atom_charge, atom_charge_known = _masked_atom_values(
        atom_formal_charges,
        atom_charge_mask,
        num_atoms,
        name="atom_formal_charges",
    )
    source_atom_charge_known = atom_charge_known
    attached_h, attached_known = _masked_atom_values(
        attached_hydrogens,
        attached_hydrogen_mask,
        num_atoms,
        name="attached_hydrogens",
        nonnegative=True,
    )
    bonds, _bond_orders_authoritative = _canonical_bonds(
        edge_index=edge_index,
        bond_orders=bond_orders,
        bond_order_mask=bond_order_mask,
        bond_order_encoding=bond_order_encoding,
        num_atoms=num_atoms,
        component_id=components,
        allow_unknown=False,
    )
    hybrid_values = _hybridization_values(hybridization, num_atoms)
    component_charge_input = _optional_integral_values(
        component_formal_charges,
        num_components,
        name="component_formal_charges",
    )

    if formula_targets is not None and any(
        value is not None for value in (formula_moiety, formula_sum)
    ):
        raise HydrogenationError("Supply formula_targets or formula text, not both")
    if formula_targets is None and any(
        value is not None for value in (formula_moiety, formula_sum)
    ):
        try:
            formula_targets = resolve_formula_targets(
                atom_types=atom_values,
                component_id=components,
                formula_moiety=formula_moiety,
                formula_sum=formula_sum,
                formula_units_z=formula_units_z,
                n_symops=n_symops,
                component_charge_evidence=component_charge_input,
                enforce_formula_units=enforce_formula_units,
            )
        except FormulaError as exc:
            raise HydrogenationError(str(exc)) from exc

    base_component_h: tuple[int | None, ...] = (
        formula_targets.component_hydrogen_counts
        if formula_targets is not None
        else (None,) * num_components
    )
    base_component_q: tuple[int | None, ...] = (
        formula_targets.component_charge_targets
        if formula_targets is not None
        else (None,) * num_components
    )
    component_h_targets = _merge_targets(
        base_component_h,
        component_hydrogen_counts,
        name="component_hydrogen_counts",
        nonnegative=True,
    )
    component_q_targets = _merge_charge_targets(
        base_component_q,
        component_charge_input,
        mode=charge_evidence_mode,
    )
    charge_fallback_components = [
        component
        for component, (formula_value, input_value) in enumerate(
            zip(base_component_q, component_charge_input)
        )
        if charge_evidence_mode == "formula"
        and formula_value is not None
        and input_value is not None
        and formula_value != input_value
    ]
    target_h = _merge_scalar(
        formula_targets.total_hydrogen_count if formula_targets is not None else None,
        total_hydrogen_count,
        name="total_hydrogen_count",
        nonnegative=True,
    )
    formula_total_q = (
        formula_targets.total_charge if formula_targets is not None else None
    )
    if charge_evidence_mode == "input" and total_charge is not None:
        target_q = _merge_scalar(
            None, total_charge, name="total_charge", nonnegative=False
        )
    elif charge_evidence_mode == "formula" and formula_total_q is not None:
        target_q = formula_total_q
    else:
        target_q = _merge_scalar(
            formula_total_q,
            total_charge,
            name="total_charge",
            nonnegative=False,
        )
    if target_h is None and all(value is not None for value in component_h_targets):
        target_h = sum(int(value) for value in component_h_targets if value is not None)
    if target_h is None and all(attached_known):
        target_h = sum(attached_h)
        component_h_targets = tuple(
            sum(
                attached_h[index]
                for index, component in enumerate(components)
                if component == component_index
            )
            for component_index in range(num_components)
        )
    if target_h is None:
        raise HydrogenationError(
            "An authoritative formula or explicit hydrogen target is required"
        )
    if target_q is None and all(value is not None for value in component_q_targets):
        target_q = sum(int(value) for value in component_q_targets if value is not None)
    if target_q is None and all(atom_charge_known):
        target_q = sum(atom_charge)
    if target_q is None:
        raise HydrogenationError(
            "An authoritative total/component charge target is required"
        )
    if (
        all(value is not None for value in component_h_targets)
        and sum(int(value) for value in component_h_targets if value is not None)
        != target_h
    ):
        raise HydrogenationError(
            "Component hydrogen targets disagree with total hydrogen target"
        )
    if (
        all(value is not None for value in component_q_targets)
        and sum(int(value) for value in component_q_targets if value is not None)
        != target_q
    ):
        if charge_evidence_mode == "formula" and formula_targets is not None:
            charge_fallback_components.extend(
                component
                for component, value in enumerate(component_q_targets)
                if value is not None and component not in charge_fallback_components
            )
            component_q_targets = (None,) * num_components
        else:
            raise HydrogenationError(
                "Component charge targets disagree with total charge target"
            )
    if (
        sum(attached_h[index] for index, known in enumerate(attached_known) if known)
        > target_h
    ):
        raise HydrogenationNoSolutionError(
            "Attached-hydrogen evidence exceeds the formula budget"
        )

    if charge_evidence_mode == "formula" and not all(attached_known):
        # Neutral zeroes on proton-active main-group atoms do not rule out a
        # zwitterionic/proton-transfer state. Explicit non-zero localization,
        # and neutral C/Si, remain hard unless a separate component-level
        # inconsistency below invalidates the whole localization.
        soft_zero_elements = {5, 7, 8, 15, 16, 33, 34}
        mutable_charge_mask = [
            known
            and (
                atom_charge[index] != 0 or atom_values[index] not in soft_zero_elements
            )
            for index, known in enumerate(atom_charge_known)
        ]
        for component, expected in enumerate(component_q_targets):
            indices = [
                index
                for index, atom_component in enumerate(components)
                if atom_component == component
            ]
            if (
                expected is not None
                and all(mutable_charge_mask[index] for index in indices)
                and sum(atom_charge[index] for index in indices) != expected
            ):
                for index in indices:
                    mutable_charge_mask[index] = False
                if component not in charge_fallback_components:
                    charge_fallback_components.append(component)
        atom_charge_known = tuple(mutable_charge_mask)

    order_sum = [0.0] * num_atoms
    degree = [0] * num_atoms
    aromatic = [False] * num_atoms
    max_order = [0.0] * num_atoms
    for left, right, order in bonds:
        # Coordination/metal-ligand edges locate components but do not consume
        # ordinary main-group covalent valence in this conservative model.
        if (
            atom_values[left] in METAL_ATOMIC_NUMBERS
            or atom_values[right] in METAL_ATOMIC_NUMBERS
        ):
            continue
        for atom in (left, right):
            order_sum[atom] += order
            degree[atom] += 1
            aromatic[atom] = aromatic[atom] or abs(order - 1.5) < 1.0e-6
            max_order[atom] = max(max_order[atom], order)

    candidates: list[tuple[ResolvedValenceState, ...]] = []
    for atom_index, atomic_number in enumerate(atom_values):
        states = _atom_candidates(
            atomic_number=atomic_number,
            bond_order_sum=order_sum[atom_index],
            degree=degree[atom_index],
            aromatic=aromatic[atom_index],
            max_order=max_order[atom_index],
            hybridization=hybrid_values[atom_index],
            known_charge=atom_charge[atom_index]
            if atom_charge_known[atom_index]
            else None,
            known_hydrogens=attached_h[atom_index]
            if attached_known[atom_index]
            else None,
        )
        if not states:
            symbol = get_element_symbol(atomic_number)
            raise HydrogenationNoSolutionError(
                f"No allowed valence state for atom {atom_index} ({symbol}); "
                f"heavy bond-order sum={order_sum[atom_index]:g}"
            )
        candidates.append(states)

    per_component: list[list[tuple[int, int, _Best]]] = []
    component_atoms: list[list[int]] = []
    for component in range(num_components):
        indices = [
            index for index, value in enumerate(components) if value == component
        ]
        component_atoms.append(indices)
        options = _component_options(
            indices,
            candidates,
            component_h_targets[component],
            component_q_targets[component],
            target_h,
        )
        if not options:
            raise HydrogenationNoSolutionError(
                f"No valence assignment satisfies component {component} hydrogen/charge constraints"
            )
        per_component.append(options)

    global_table: dict[tuple[int, int], _Best] = {
        (0, 0): _Best(0.0, (), (), (), (), ())
    }
    global_atom_order: list[int] = []
    for indices, options in zip(component_atoms, per_component):
        updated: dict[tuple[int, int], _Best] = {}
        for (current_h, current_q), best in global_table.items():
            for option_h, option_q, option in options:
                next_h = current_h + option_h
                next_q = current_q + option_q
                if next_h > target_h:
                    continue
                _record_best(
                    updated,
                    (next_h, next_q),
                    best.score + option.score,
                    best.solution + option.solution,
                    best.hydrogen_signature + option.hydrogen_signature,
                    best.charge_signature + option.charge_signature,
                    best.hydrogen_ambiguous + option.hydrogen_ambiguous,
                    best.charge_ambiguous + option.charge_ambiguous,
                )
        global_table = updated
        global_atom_order.extend(indices)
    final = global_table.get((target_h, target_q))
    if final is None:
        raise HydrogenationNoSolutionError(
            f"No valence assignment satisfies total H={target_h}, charge={target_q}"
        )

    by_atom: list[ResolvedValenceState | None] = [None] * num_atoms
    for atom_index, state in zip(global_atom_order, final.solution):
        by_atom[atom_index] = state
    if any(state is None for state in by_atom):
        raise RuntimeError("Internal hydrogenation assignment is incomplete")
    selected = tuple(state for state in by_atom if state is not None)
    ambiguous_parent_atoms = {
        index for index, value in enumerate(final.hydrogen_ambiguous) if value
    }
    ambiguity = bool(ambiguous_parent_atoms)
    domain_ambiguities: list[str] = []
    for component in range(num_components):
        indices = [
            index
            for index, atom_component in enumerate(components)
            if atom_component == component
        ]
        boron = [index for index in indices if atom_values[index] == 5]
        phosphorus = [index for index in indices if atom_values[index] == 15]
        if (
            boron
            and phosphorus
            and not any(attached_known[index] for index in boron + phosphorus)
            and any(
                state.hydrogens > 0
                for index in boron + phosphorus
                for state in candidates[index]
            )
        ):
            ambiguity = True
            ambiguous_parent_atoms.update(boron + phosphorus)
            domain_ambiguities.append(f"component {component}: B/P hydrogen allocation")

        terminal_sp_carbon = [
            index
            for index in indices
            if atom_values[index] == 6
            and hybrid_values[index] == HYBRIDIZATION_TO_INDEX["sp"]
            and degree[index] == 1
            and max_order[index] >= 2.5
        ]
        aromatic_nitrogen = [
            index for index in indices if atom_values[index] == 7 and aromatic[index]
        ]
        if (
            terminal_sp_carbon
            and aromatic_nitrogen
            and not any(
                attached_known[index]
                for index in terminal_sp_carbon + aromatic_nitrogen
            )
        ):
            ambiguity = True
            ambiguous_parent_atoms.update(terminal_sp_carbon + aromatic_nitrogen)
            domain_ambiguities.append(
                f"component {component}: terminal sp-C/aromatic-N tautomer"
            )

        if charge_evidence_mode == "formula":
            localized_charged_carbon = [
                index
                for index in indices
                if atom_values[index] in {6, 14}
                and source_atom_charge_known[index]
                and atom_charge[index] != 0
            ]
            protonation_competitors = [
                index
                for index in indices
                if atom_values[index] in {5, 7, 8, 15, 16, 33, 34}
                and len({state.hydrogens for state in candidates[index]}) > 1
            ]
            if localized_charged_carbon and protonation_competitors:
                ambiguity = True
                ambiguous_parent_atoms.update(
                    localized_charged_carbon + protonation_competitors
                )
                domain_ambiguities.append(
                    f"component {component}: localized C/Si charge competes with "
                    "heteroatom protonation"
                )
    selected_parent_h = tuple(state.hydrogens for state in selected)
    isotope_ambiguous_parents = _ambiguous_isotope_parents(
        parent_h_counts=selected_parent_h,
        component_id=components,
        formula_targets=formula_targets,
    )
    if isotope_ambiguous_parents:
        ambiguity = True
        ambiguous_parent_atoms.update(isotope_ambiguous_parents)
        domain_ambiguities.append("formula does not localize H/D/T isotopes")
    if ambiguity and strict:
        raise AmbiguousHydrogenationError(
            "Multiple chemically feasible hydrogenation assignments remain; "
            f"alternative_atoms={sorted(ambiguous_parent_atoms)}, "
            f"selected_parent_h={selected_parent_h}, "
            f"domain={domain_ambiguities}"
        )
    parent_h_counts = selected_parent_h
    formal_charges = tuple(state.formal_charge for state in selected)
    formal_charge_resolved = tuple(not value for value in final.charge_ambiguous)
    component_h_counts = tuple(
        sum(
            parent_h_counts[index]
            for index, value in enumerate(components)
            if value == component
        )
        for component in range(num_components)
    )
    component_charge_totals = tuple(
        sum(
            formal_charges[index]
            for index, value in enumerate(components)
            if value == component
        )
        for component in range(num_components)
    )
    parent_isotopes, isotope_counts = _assign_isotopes(
        parent_h_counts=parent_h_counts,
        component_id=components,
        formula_targets=formula_targets,
    )
    evidence = tuple(
        "attached_hydrogens" if attached_known[index] else "formula+bond_order+valence"
        for index in range(num_atoms)
    )
    charge_is_unique = all(formal_charge_resolved)
    return HydrogenationPlan(
        parent_h_counts=parent_h_counts,
        formal_charges=formal_charges,
        formal_charge_resolved=tuple(formal_charge_resolved),
        resolved_states=selected,
        component_h_counts=component_h_counts,
        component_charge_totals=component_charge_totals,
        parent_h_isotopes=parent_isotopes,
        isotope_counts=isotope_counts,
        target_hydrogen_count=target_h,
        target_total_charge=target_q,
        evidence=evidence,
        charge_evidence_fallback_components=tuple(charge_fallback_components),
        formula_source=formula_targets.source if formula_targets is not None else None,
        status=(
            "ambiguous"
            if ambiguity
            else (
                "authoritative"
                if all(attached_known) and charge_is_unique
                else (
                    "inferred_unique"
                    if charge_is_unique
                    else "inferred_unique_h_unresolved_charge"
                )
            )
        ),
        ambiguity=ambiguity,
        ambiguous_parent_atoms=tuple(sorted(ambiguous_parent_atoms)),
        num_optimal_solutions=2 if ambiguity else 1,
        score=float(final.score),
    )


def _minimum_image_cart(
    delta_frac: Tensor, lattice: Tensor, image_offsets: Tensor | None = None
) -> Tensor:
    base = -torch.round(delta_frac)
    offsets = (
        _PERIODIC_IMAGE_OFFSETS.to(device=delta_frac.device, dtype=delta_frac.dtype)
        if image_offsets is None
        else image_offsets
    )
    candidates = (delta_frac.unsqueeze(0) + base.unsqueeze(0) + offsets) @ lattice
    return candidates[torch.argmin(torch.sum(candidates * candidates, dim=1))]


def _unit(vector: Tensor, fallback: Tensor) -> Tensor:
    norm = torch.linalg.norm(vector)
    return vector / norm if float(norm) > 1.0e-10 else fallback


def _direction_candidates(device: torch.device, dtype: torch.dtype) -> Tensor:
    count = 192
    indices = torch.arange(count, device=device, dtype=dtype)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    angle = indices * (math.pi * (3.0 - math.sqrt(5.0)))
    points = torch.stack(
        (radius * torch.cos(angle), radius * torch.sin(angle), z), dim=1
    )
    axes = torch.eye(3, device=device, dtype=dtype)
    return torch.cat((points, axes, -axes), dim=0)


def _target_angle_cos(
    atomic_number: int, hybridization: int, total_neighbors: int
) -> float:
    if hybridization == HYBRIDIZATION_TO_INDEX["sp"]:
        return -1.0
    if hybridization == HYBRIDIZATION_TO_INDEX["sp2"]:
        return -0.5
    if hybridization == HYBRIDIZATION_TO_INDEX["sp3"]:
        return -1.0 / 3.0
    if atomic_number in {8, 16, 34} and total_neighbors == 2:
        return math.cos(math.radians(105.0))
    if atomic_number == 7 and total_neighbors in {3, 4}:
        return -1.0 / 3.0
    if total_neighbors <= 2:
        return -1.0
    if total_neighbors == 3:
        return -0.5
    return -1.0 / 3.0


def _minimum_periodic_distances(
    candidate_frac: Tensor,
    other_frac: Tensor,
    lattice: Tensor,
    image_offsets: Tensor,
) -> Tensor:
    """Return each candidate's nearest periodic distance to ``other_frac``.

    Candidate directions and comparison atoms are evaluated together.  The
    comparison-atom axis is chunked to bound temporary memory for large ASUs.
    """

    if candidate_frac.ndim != 2 or candidate_frac.shape[1] != 3:
        raise HydrogenationError("candidate_frac must have shape [C, 3]")
    if other_frac.ndim != 2 or other_frac.shape[1] != 3:
        raise HydrogenationError("other_frac must have shape [M, 3]")
    if not int(other_frac.shape[0]):
        return torch.full(
            (int(candidate_frac.shape[0]),),
            float("inf"),
            device=candidate_frac.device,
            dtype=candidate_frac.dtype,
        )

    minimum_squared = torch.full(
        (int(candidate_frac.shape[0]),),
        float("inf"),
        device=candidate_frac.device,
        dtype=candidate_frac.dtype,
    )
    for other_chunk in other_frac.split(_PERIODIC_DISTANCE_OTHER_CHUNK_SIZE):
        delta = other_chunk.unsqueeze(0) - candidate_frac.unsqueeze(1)
        reduced = delta - torch.round(delta)
        periodic_cart = (
            reduced.unsqueeze(2) + image_offsets.view(1, 1, -1, 3)
        ) @ lattice
        squared = torch.sum(periodic_cart * periodic_cart, dim=-1)
        minimum_squared = torch.minimum(minimum_squared, squared.amin(dim=(1, 2)))
    return torch.sqrt(torch.clamp_min(minimum_squared, 0.0))


def place_hydrogens(
    *,
    atom_types: Sequence[int] | Tensor,
    frac_coords: Tensor,
    lattice: Tensor,
    edge_index: Sequence | Tensor,
    plan: HydrogenationPlan,
    hybridization: Sequence[int | str] | Tensor | None = None,
) -> HydrogenPlacement:
    """Place approximate X-H coordinates without wrapping fractional images."""

    atom_tensor = torch.as_tensor(
        atom_types, device=frac_coords.device, dtype=torch.long
    )
    if atom_tensor.ndim != 1:
        raise HydrogenationError("atom_types must be one-dimensional")
    num_atoms = int(atom_tensor.numel())
    if frac_coords.shape != (num_atoms, 3):
        raise HydrogenationError("frac_coords must have shape [N, 3]")
    if lattice.shape != (3, 3):
        raise HydrogenationError("lattice must have shape [3, 3]")
    if len(plan.parent_h_counts) != num_atoms:
        raise HydrogenationError(
            "HydrogenationPlan atom count disagrees with coordinates"
        )
    if not torch.isfinite(frac_coords).all() or not torch.isfinite(lattice).all():
        raise HydrogenationError("Coordinates and lattice must be finite")
    determinant = torch.linalg.det(lattice.to(torch.float64))
    if abs(float(determinant)) < 1.0e-10:
        raise HydrogenationError("Lattice is singular")
    hybrids = _hybridization_values(hybridization, num_atoms)
    edges = torch.as_tensor(edge_index).detach().cpu()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise HydrogenationError("edge_index must have shape [2, E]")
    neighbors: list[set[int]] = [set() for _ in range(num_atoms)]
    for position in range(int(edges.shape[1])):
        left, right = int(edges[0, position]), int(edges[1, position])
        if 0 <= left < num_atoms and 0 <= right < num_atoms and left != right:
            neighbors[left].add(right)
            neighbors[right].add(left)

    dtype = frac_coords.dtype if frac_coords.dtype.is_floating_point else torch.float32
    frac = frac_coords.to(dtype=dtype)
    lat = lattice.to(device=frac.device, dtype=dtype)
    cart = frac @ lat
    inv_lattice = torch.linalg.inv(lat.to(torch.float64)).to(dtype=dtype)
    directions = _direction_candidates(frac.device, dtype)
    image_offsets = _PERIODIC_IMAGE_OFFSETS.to(device=frac.device, dtype=dtype)
    placed_frac: list[Tensor] = []
    parent_indices: list[int] = []
    isotope_labels: list[str] = []
    bond_lengths: list[float] = []

    for parent, hydrogen_count in enumerate(plan.parent_h_counts):
        if hydrogen_count <= 0:
            continue
        atomic_number = int(atom_tensor[parent])
        if atomic_number in METAL_ATOMIC_NUMBERS:
            raise UnsupportedHydrogenationError(
                "Plan attempts to place hydrogen on a metal"
            )
        length = _XH_BOND_LENGTH_A.get(atomic_number)
        if length is None:
            raise UnsupportedHydrogenationError(
                f"No X-H bond length is defined for {get_element_symbol(atomic_number)}"
            )
        occupied_directions: list[Tensor] = []
        for neighbor in sorted(neighbors[parent]):
            vector = _minimum_image_cart(
                frac[neighbor] - frac[parent], lat, image_offsets
            )
            fallback = torch.tensor([1.0, 0.0, 0.0], device=frac.device, dtype=dtype)
            occupied_directions.append(_unit(vector, fallback))
        total_neighbors = len(occupied_directions) + hydrogen_count
        target_cos = _target_angle_cos(atomic_number, hybrids[parent], total_neighbors)
        heavy_obstacles = torch.cat((frac[:parent], frac[parent + 1 :]), dim=0)

        for local_h in range(hydrogen_count):
            if occupied_directions:
                occupied = torch.stack(occupied_directions)
                dots = torch.sum(
                    directions.unsqueeze(1) * occupied.unsqueeze(0), dim=-1
                )
                angle_errors = torch.sum((dots - target_cos) ** 2, dim=1)
            else:
                angle_errors = torch.zeros(
                    int(directions.shape[0]), device=frac.device, dtype=dtype
                )

            candidate_cart = cart[parent].unsqueeze(0) + float(length) * directions
            candidate_frac = (
                frac[parent].unsqueeze(0)
                + (candidate_cart - cart[parent].unsqueeze(0)) @ inv_lattice
            )
            obstacles = (
                torch.cat((heavy_obstacles, torch.stack(placed_frac)), dim=0)
                if placed_frac
                else heavy_obstacles
            )
            minimum_distances = _minimum_periodic_distances(
                candidate_frac, obstacles, lat, image_offsets
            )
            clash_penalties = torch.clamp(0.72 - minimum_distances, min=0.0) ** 2 * 50.0
            scores = angle_errors + clash_penalties

            # Preserve the scalar implementation's lexicographic selection:
            # local geometry, then largest clearance, then candidate order.
            score_values = scores.detach().cpu().tolist()
            distance_values = minimum_distances.detach().cpu().tolist()
            best_index = min(
                range(int(directions.shape[0])),
                key=lambda index: (
                    score_values[index],
                    -distance_values[index],
                    index,
                ),
            )
            best_direction = directions[best_index]
            hydrogen_cart = cart[parent] + float(length) * best_direction
            hydrogen_frac = frac[parent] + (hydrogen_cart - cart[parent]) @ inv_lattice
            placed_frac.append(hydrogen_frac)
            parent_indices.append(parent)
            isotope_labels.append(plan.parent_h_isotopes[parent][local_h])
            bond_lengths.append(float(length))
            fallback = torch.tensor([0.0, 1.0, 0.0], device=frac.device, dtype=dtype)
            occupied_directions.append(
                _unit((hydrogen_frac - frac[parent]) @ lat, fallback)
            )

    if placed_frac:
        hydrogen_coords = torch.stack(placed_frac)
        parent_tensor = torch.tensor(
            parent_indices, device=frac.device, dtype=torch.long
        )
        length_tensor = torch.tensor(bond_lengths, device=frac.device, dtype=dtype)
    else:
        hydrogen_coords = torch.empty((0, 3), device=frac.device, dtype=dtype)
        parent_tensor = torch.empty(0, device=frac.device, dtype=torch.long)
        length_tensor = torch.empty(0, device=frac.device, dtype=dtype)
    return HydrogenPlacement(
        hydrogen_frac_coords=hydrogen_coords,
        parent_indices=parent_tensor,
        isotopes=tuple(isotope_labels),
        bond_lengths=length_tensor,
    )


def hydrogenate_structure(
    *,
    atom_types: Sequence[int] | Tensor,
    frac_coords: Tensor,
    lattice: Tensor,
    edge_index: Sequence | Tensor,
    plan: HydrogenationPlan,
    hybridization: Sequence[int | str] | Tensor | None = None,
    atom_labels: Sequence[str] | None = None,
) -> HydrogenatedStructure:
    """Append placed H/D/T atoms while preserving the heavy-atom ordering."""

    atom_tensor = torch.as_tensor(
        atom_types, device=frac_coords.device, dtype=torch.long
    )
    placement = place_hydrogens(
        atom_types=atom_tensor,
        frac_coords=frac_coords,
        lattice=lattice,
        edge_index=edge_index,
        plan=plan,
        hybridization=hybridization,
    )
    num_heavy = int(atom_tensor.numel())
    num_hydrogen = int(placement.parent_indices.numel())
    output_types = torch.cat(
        (
            atom_tensor,
            torch.ones(
                num_hydrogen, device=atom_tensor.device, dtype=atom_tensor.dtype
            ),
        )
    )
    output_coords = torch.cat((frac_coords, placement.hydrogen_frac_coords), dim=0)
    if atom_labels is None:
        heavy_labels = tuple(
            f"{get_element_symbol(int(atomic_number))}{index + 1}"
            for index, atomic_number in enumerate(atom_tensor.tolist())
        )
    else:
        if len(atom_labels) != num_heavy:
            raise HydrogenationError(
                "atom_labels must contain one label per heavy atom"
            )
        heavy_labels = tuple(str(label) for label in atom_labels)
    used = set(heavy_labels)
    counters: Counter[str] = Counter()
    hydrogen_labels: list[str] = []
    for isotope in placement.isotopes:
        while True:
            counters[isotope] += 1
            label = f"{isotope}{counters[isotope]}"
            if label not in used:
                break
        used.add(label)
        hydrogen_labels.append(label)
    if num_hydrogen:
        hydrogen_indices = torch.arange(
            num_heavy,
            num_heavy + num_hydrogen,
            device=atom_tensor.device,
            dtype=torch.long,
        )
        added_edges = torch.stack((placement.parent_indices, hydrogen_indices))
    else:
        added_edges = torch.empty((2, 0), device=atom_tensor.device, dtype=torch.long)
    return HydrogenatedStructure(
        atom_types=output_types,
        frac_coords=output_coords,
        atom_labels=heavy_labels + tuple(hydrogen_labels),
        hydrogen_parent_indices=placement.parent_indices,
        hydrogen_isotopes=placement.isotopes,
        added_edge_index=added_edges,
        plan=plan,
    )


__all__ = [
    "AmbiguousHydrogenationError",
    "HydrogenPlacement",
    "HydrogenatedStructure",
    "HydrogenationError",
    "HydrogenationNoSolutionError",
    "HydrogenationPlan",
    "ResolvedValenceState",
    "UnsupportedHydrogenationError",
    "hydrogenate_structure",
    "infer_hydrogenation_plan",
    "place_hydrogens",
]
