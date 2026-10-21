"""Strict chemical-formula parsing and ASU stoichiometry matching.

The parser intentionally keeps formula arithmetic exact.  CIF moiety formulae
often contain fractional *numbers of molecules* (for example ``0.5(H2 O)``),
while the corresponding general-position ASU contains an integral number of
those molecules.  ``fractions.Fraction`` prevents rounding that could silently
change a hydrogen budget.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from sala.core.types import ELEMENT_TO_ATOMIC_NUMBER, get_element_symbol

HYDROGEN_ISOTOPES = ("H", "D", "T")
_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d+)?|\.\d+|\d+/\d+)")


class FormulaError(ValueError):
    """Base class for formula parsing and stoichiometry errors."""


class FormulaMismatchError(FormulaError):
    """Raised when a formula cannot describe the supplied heavy-atom ASU."""


class FormulaAmbiguityError(FormulaError):
    """Raised when formula components have more than one distinct assignment."""


@dataclass(frozen=True)
class FormulaEntry:
    """One molecular/ionic entry in a CIF chemical formula.

    ``counts`` describes one entity. ``coefficient`` is the number of such
    entities per formula unit and may be fractional in ``formula_moiety``.
    ``charge`` is the integral charge of one entity, not coefficient-scaled.
    """

    counts: tuple[tuple[str, Fraction], ...]
    coefficient: Fraction = Fraction(1)
    charge: int = 0
    source: str = ""

    def count(self, symbol: str) -> Fraction:
        return dict(self.counts).get(symbol, Fraction(0))

    @property
    def hydrogen_count(self) -> Fraction:
        return sum((self.count(symbol) for symbol in HYDROGEN_ISOTOPES), Fraction(0))

    @property
    def heavy_counts(self) -> tuple[tuple[str, Fraction], ...]:
        return tuple(
            (symbol, count)
            for symbol, count in self.counts
            if symbol not in HYDROGEN_ISOTOPES
        )

    @property
    def isotope_counts(self) -> tuple[tuple[str, Fraction], ...]:
        return tuple(
            (symbol, self.count(symbol))
            for symbol in HYDROGEN_ISOTOPES
            if self.count(symbol) > 0
        )


@dataclass(frozen=True)
class FormulaTargets:
    """Hydrogen and charge constraints resolved for one heavy-atom ASU."""

    component_hydrogen_counts: tuple[int | None, ...]
    component_charge_targets: tuple[int | None, ...]
    component_isotope_counts: tuple[tuple[tuple[str, int], ...] | None, ...]
    total_hydrogen_count: int
    total_charge: int
    total_isotope_counts: tuple[tuple[str, int], ...]
    source: str
    formula_scale: Fraction
    formula_units_consistent: bool | None
    fallback_reason: str | None = None


def _clean_formula_text(text: str) -> str:
    clean = str(text).strip().strip("'\"").strip()
    if not clean or clean in {"?", "."}:
        raise FormulaError("Chemical formula is empty or unknown")
    return clean.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")


def _parse_fraction(token: str, *, context: str) -> Fraction:
    try:
        value = Fraction(token)
    except (ValueError, ZeroDivisionError) as exc:
        raise FormulaError(f"Invalid {context}: {token!r}") from exc
    if value <= 0:
        raise FormulaError(f"{context.capitalize()} must be positive, got {token!r}")
    return value


def _split_charge(text: str) -> tuple[str, int]:
    """Split a terminal CIF-style charge without confusing ``H4+`` with +4."""

    clean = text.strip()
    # CCDC/CIF convention normally separates a magnitude: ``... N1 2+``.
    match = re.match(r"^(.*\S)\s+(\d+)\s*([+-])\s*$", clean)
    if match:
        magnitude = int(match.group(2))
        return match.group(1).strip(), magnitude if match.group(
            3
        ) == "+" else -magnitude
    match = re.match(r"^(.*\S)\s+([+-])\s*(\d+)\s*$", clean)
    if match:
        magnitude = int(match.group(3))
        return match.group(1).strip(), magnitude if match.group(
            2
        ) == "+" else -magnitude
    # A bare sign is unambiguous: ``NH4+`` means NH4 with charge +1.
    match = re.match(r"^(.*\S)\s*([+-])\s*$", clean)
    if match:
        return match.group(1).strip(), 1 if match.group(2) == "+" else -1
    return clean, 0


def _matching_close(text: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return None
    return None


def _split_coefficient(text: str) -> tuple[Fraction, str]:
    stripped = text.strip()
    match = _NUMBER_RE.match(stripped)
    if match is None:
        return Fraction(1), stripped
    end = match.end()
    remainder = stripped[end:]
    if not remainder:
        raise FormulaError(f"Formula contains a coefficient without a body: {text!r}")
    # A leading number is a moiety coefficient only when followed by an atom
    # or a group. Atomic subscripts always follow an element symbol.
    remainder_lstrip = remainder.lstrip()
    if not remainder_lstrip or not (
        remainder_lstrip[0].isupper() or remainder_lstrip[0] == "("
    ):
        return Fraction(1), stripped
    coefficient = _parse_fraction(match.group(0), context="formula coefficient")
    if remainder_lstrip.startswith("("):
        close = _matching_close(remainder_lstrip, 0)
        if close is None:
            raise FormulaError(f"Unbalanced parentheses in formula: {text!r}")
        if not remainder_lstrip[close + 1 :].strip():
            remainder_lstrip = remainder_lstrip[1:close]
    return coefficient, remainder_lstrip


def _add_counts(
    target: dict[str, Fraction],
    source: Mapping[str, Fraction],
    scale: Fraction = Fraction(1),
) -> None:
    for symbol, count in source.items():
        target[symbol] += count * scale


def _parse_compact_group(
    compact: str,
    source: str,
    position: int,
    *,
    closing: bool = False,
) -> tuple[dict[str, Fraction], int]:
    counts: dict[str, Fraction] = defaultdict(Fraction)
    while position < len(compact):
        char = compact[position]
        if char == ")":
            if not closing:
                raise FormulaError(f"Unexpected ')' in formula {source!r}")
            return counts, position + 1
        if char == "(":
            nested, position = _parse_compact_group(
                compact, source, position + 1, closing=True
            )
            number = _NUMBER_RE.match(compact, position)
            multiplier = Fraction(1)
            if number is not None:
                multiplier = _parse_fraction(
                    number.group(0), context="group multiplier"
                )
                position = number.end()
            _add_counts(counts, nested, multiplier)
            continue
        if not char.isupper():
            raise FormulaError(
                f"Unexpected character {char!r} at position {position} "
                f"in formula {source!r}"
            )
        end = position + 1
        if end < len(compact) and compact[end].islower():
            end += 1
        symbol = compact[position:end]
        if symbol not in ELEMENT_TO_ATOMIC_NUMBER and symbol not in {"D", "T"}:
            raise FormulaError(f"Unknown element {symbol!r} in formula {source!r}")
        number = _NUMBER_RE.match(compact, end)
        amount = Fraction(1)
        if number is not None:
            amount = _parse_fraction(number.group(0), context=f"count for {symbol}")
            end = number.end()
        counts[symbol] += amount
        position = end
    if closing:
        raise FormulaError(f"Unbalanced parentheses in formula {source!r}")
    return counts, position


def _parse_composition(text: str) -> dict[str, Fraction]:
    # The middle dot is treated as an additive hydrate separator. A plain dot
    # remains part of decimal stoichiometry and is therefore not ambiguous.
    segments = text.split("·")
    total: dict[str, Fraction] = defaultdict(Fraction)
    for segment in segments:
        coefficient, body = _split_coefficient(segment)
        compact = "".join(body.split())
        if not compact:
            raise FormulaError(f"Empty formula segment in {text!r}")
        parsed, final_position = _parse_compact_group(compact, text, 0)
        if final_position != len(compact):
            raise FormulaError(f"Could not consume formula {text!r}")
        _add_counts(total, parsed, coefficient)
    return total


def parse_formula(text: str) -> FormulaEntry:
    """Parse one strict formula entry, retaining exact counts and D/T labels."""

    clean = _clean_formula_text(text)
    # Split the outer coefficient first: CCDC commonly writes charged repeated
    # moieties as ``3(Cl1 1-)``, with the charge inside the parentheses.
    coefficient, body = _split_coefficient(clean)
    body, charge = _split_charge(body)
    counts = _parse_composition(body)
    if not counts:
        raise FormulaError(f"Formula has no atoms: {text!r}")
    return FormulaEntry(
        counts=tuple(sorted(counts.items())),
        coefficient=coefficient,
        charge=charge,
        source=clean,
    )


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise FormulaError(f"Unbalanced parentheses in moiety formula {text!r}")
        elif char in ",;" and depth == 0:
            part = text[start:index].strip()
            if not part:
                raise FormulaError(f"Empty entry in moiety formula {text!r}")
            parts.append(part)
            start = index + 1
    if depth != 0:
        raise FormulaError(f"Unbalanced parentheses in moiety formula {text!r}")
    tail = text[start:].strip()
    if not tail:
        raise FormulaError(f"Empty entry in moiety formula {text!r}")
    parts.append(tail)
    return parts


def parse_formula_moiety(text: str) -> tuple[FormulaEntry, ...]:
    """Parse comma-separated ``_chemical_formula_moiety`` entries."""

    clean = _clean_formula_text(text)
    entries = tuple(parse_formula(part) for part in _split_top_level(clean))
    if not entries:
        raise FormulaError("Moiety formula has no entries")
    return entries


def _integer(value: Fraction, *, context: str, allow_zero: bool = True) -> int:
    if value.denominator != 1:
        raise FormulaMismatchError(f"{context} is non-integral ({value})")
    integer = value.numerator
    if integer < 0 or (integer == 0 and not allow_zero):
        raise FormulaMismatchError(
            f"{context} must be {'positive' if not allow_zero else 'non-negative'}"
        )
    return integer


def _counter_from_atom_types(
    atom_types: Sequence[int], indices: Iterable[int] | None = None
) -> Counter[str]:
    selected = range(len(atom_types)) if indices is None else indices
    counts: Counter[str] = Counter()
    for atom_index in selected:
        symbol = get_element_symbol(int(atom_types[atom_index]))
        if symbol in HYDROGEN_ISOTOPES:
            raise FormulaMismatchError(
                "Hydrogenation expects a hydrogen-free heavy-atom graph"
            )
        counts[symbol] += 1
    return counts


def component_heavy_compositions(
    atom_types: Sequence[int], component_id: Sequence[int]
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Return heavy-element compositions in compact component-id order."""

    atom_values = [int(value) for value in atom_types]
    component_values = [int(value) for value in component_id]
    if len(atom_values) != len(component_values):
        raise FormulaMismatchError("atom_types and component_id must have equal length")
    if not component_values:
        return ()
    unique = sorted(set(component_values))
    if unique != list(range(len(unique))):
        raise FormulaMismatchError("component_id must be contiguous and start at zero")
    output = []
    for component in unique:
        indices = (
            index for index, value in enumerate(component_values) if value == component
        )
        output.append(
            tuple(sorted(_counter_from_atom_types(atom_values, indices).items()))
        )
    return tuple(output)


def _aggregate_counts(
    entries: Sequence[FormulaEntry], *, heavy_only: bool
) -> dict[str, Fraction]:
    counts: dict[str, Fraction] = defaultdict(Fraction)
    for entry in entries:
        for symbol, amount in entry.counts:
            if heavy_only and symbol in HYDROGEN_ISOTOPES:
                continue
            counts[symbol] += entry.coefficient * amount
    return counts


def _infer_formula_scale(
    observed: Mapping[str, int], declared: Mapping[str, Fraction]
) -> Fraction:
    declared_nonzero = {
        symbol: count for symbol, count in declared.items() if count != 0
    }
    if not declared_nonzero:
        raise FormulaMismatchError("Chemical formula contains no heavy atoms")
    if set(observed) != set(declared_nonzero):
        raise FormulaMismatchError(
            "Formula/ASU heavy elements differ: "
            f"formula={dict(sorted(declared_nonzero.items()))}, ASU={dict(sorted(observed.items()))}"
        )
    ratios = {
        Fraction(int(observed[symbol]), count)
        for symbol, count in declared_nonzero.items()
    }
    if len(ratios) != 1:
        raise FormulaMismatchError(
            "Formula and ASU heavy-atom counts do not share one scale"
        )
    scale = ratios.pop()
    if scale <= 0:
        raise FormulaMismatchError("Formula/ASU scale must be positive")
    return scale


def _as_optional_ints(
    values: Sequence[int | None] | None, length: int, *, name: str
) -> tuple[int | None, ...]:
    if values is None:
        return (None,) * length
    if len(values) != length:
        raise FormulaMismatchError(f"{name} must contain one value per component")
    output: list[int | None] = []
    for value in values:
        if value is None:
            output.append(None)
            continue
        integer = int(value)
        if float(value) != integer:
            raise FormulaMismatchError(f"{name} must contain integral charges")
        output.append(integer)
    return tuple(output)


def _expand_moiety_entries(
    entries: Sequence[FormulaEntry], scale: Fraction
) -> list[tuple[tuple[tuple[str, int], ...], int, int, tuple[tuple[str, int], ...]]]:
    expanded: list[
        tuple[tuple[tuple[str, int], ...], int, int, tuple[tuple[str, int], ...]]
    ] = []
    for entry in entries:
        repetitions = _integer(
            entry.coefficient * scale,
            context=f"scaled coefficient for {entry.source!r}",
        )
        heavy_key = tuple(
            (
                symbol,
                _integer(
                    count,
                    context=f"{symbol} count in {entry.source!r}",
                    allow_zero=False,
                ),
            )
            for symbol, count in entry.heavy_counts
        )
        if not heavy_key:
            raise FormulaMismatchError(
                "Hydrogen-only formula components cannot be attached to a heavy-atom graph"
            )
        isotope_counts = tuple(
            (symbol, _integer(count, context=f"{symbol} count in {entry.source!r}"))
            for symbol, count in entry.isotope_counts
        )
        hydrogen_count = sum(count for _, count in isotope_counts)
        descriptor = (heavy_key, hydrogen_count, int(entry.charge), isotope_counts)
        expanded.extend([descriptor] * repetitions)
    return expanded


def _unique_descriptor_assignment(
    components: Sequence[int],
    descriptors: Sequence[tuple[int, int, tuple[tuple[str, int], ...]]],
    evidence: Sequence[int | None],
) -> tuple[tuple[int, int, tuple[tuple[str, int], ...]], ...] | None:
    descriptor_counts = Counter(descriptors)
    solutions: set[tuple[tuple[int, int, tuple[tuple[str, int], ...]], ...]] = set()

    def visit(
        position: int,
        current: list[tuple[int, int, tuple[tuple[str, int], ...]]],
    ) -> None:
        if len(solutions) > 1:
            return
        if position == len(components):
            solutions.add(tuple(current))
            return
        component = components[position]
        for descriptor in sorted(descriptor_counts):
            if descriptor_counts[descriptor] <= 0:
                continue
            if evidence[component] is not None and descriptor[1] != evidence[component]:
                continue
            descriptor_counts[descriptor] -= 1
            current.append(descriptor)
            visit(position + 1, current)
            current.pop()
            descriptor_counts[descriptor] += 1

    visit(0, [])
    if not solutions:
        return None
    if len(solutions) > 1:
        raise FormulaAmbiguityError(
            "Composition-identical formula components have different hydrogen/charge assignments; "
            "localized component charge evidence is required"
        )
    return next(iter(solutions))


def _match_expanded_components(
    component_keys: Sequence[tuple[tuple[str, int], ...]],
    expanded: Sequence[
        tuple[tuple[tuple[str, int], ...], int, int, tuple[tuple[str, int], ...]]
    ],
    charge_evidence: Sequence[int | None] | None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[tuple[str, int], ...], ...]]:
    if len(component_keys) != len(expanded):
        raise FormulaMismatchError(
            f"Formula expands to {len(expanded)} components but ASU has {len(component_keys)}"
        )
    evidence = _as_optional_ints(
        charge_evidence, len(component_keys), name="component charge evidence"
    )
    by_key_components: dict[tuple[tuple[str, int], ...], list[int]] = defaultdict(list)
    by_key_entries: dict[
        tuple[tuple[str, int], ...], list[tuple[int, int, tuple[tuple[str, int], ...]]]
    ] = defaultdict(list)
    for index, key in enumerate(component_keys):
        by_key_components[key].append(index)
    for key, hydrogen, charge, isotopes in expanded:
        by_key_entries[key].append((hydrogen, charge, isotopes))
    if {key: len(value) for key, value in by_key_components.items()} != {
        key: len(value) for key, value in by_key_entries.items()
    }:
        raise FormulaMismatchError(
            "Moiety heavy-atom components do not match the ASU components"
        )

    assigned_h = [0] * len(component_keys)
    assigned_q = [0] * len(component_keys)
    assigned_i: list[tuple[tuple[str, int], ...]] = [()] * len(component_keys)
    for key in sorted(by_key_components):
        components = by_key_components[key]
        solution = _unique_descriptor_assignment(
            components, by_key_entries[key], evidence
        )
        if solution is None:
            raise FormulaMismatchError(
                f"No formula entry matches component composition {dict(key)}"
            )
        for component, (hydrogen, charge, isotopes) in zip(components, solution):
            assigned_h[component] = hydrogen
            assigned_q[component] = charge
            assigned_i[component] = isotopes
    return tuple(assigned_h), tuple(assigned_q), tuple(assigned_i)


def _formula_units_consistency(
    scale: Fraction,
    formula_units_z: float | str | None,
    n_symops: int | None,
    *,
    enforce: bool,
) -> bool | None:
    if formula_units_z is None:
        return None
    if n_symops is None or int(n_symops) <= 0:
        raise FormulaMismatchError(
            "n_symops must be positive when formula_units_z is provided"
        )
    try:
        expected = Fraction(str(formula_units_z)) / int(n_symops)
    except (ValueError, ZeroDivisionError) as exc:
        raise FormulaMismatchError("Invalid formula_units_z/n_symops metadata") from exc
    consistent = expected == scale
    if enforce and not consistent:
        raise FormulaMismatchError(
            f"Formula/ASU scale {scale} disagrees with Z/n_symops={expected}"
        )
    return consistent


def resolve_formula_targets(
    *,
    atom_types: Sequence[int],
    component_id: Sequence[int],
    formula_moiety: str | None = None,
    formula_sum: str | None = None,
    formula_units_z: float | str | None = None,
    n_symops: int | None = None,
    component_charge_evidence: Sequence[int | None] | None = None,
    enforce_formula_units: bool = False,
) -> FormulaTargets:
    """Resolve exact ASU hydrogen/charge targets from CIF formula metadata.

    Moiety formulae are preferred and yield component-level constraints. A sum
    formula yields only ASU totals unless the ASU has one component. If moiety
    entries cannot be mapped uniquely, an available sum formula is used as an
    explicit global-budget fallback and the reason is retained in the result.
    """

    atom_values = [int(value) for value in atom_types]
    component_values = [int(value) for value in component_id]
    component_keys = component_heavy_compositions(atom_values, component_values)
    observed = _counter_from_atom_types(atom_values)
    num_components = len(component_keys)
    evidence = _as_optional_ints(
        component_charge_evidence, num_components, name="component charge evidence"
    )

    moiety_error: FormulaError | None = None
    if formula_moiety is not None and str(formula_moiety).strip().strip("'\"") not in {
        "",
        "?",
        ".",
    }:
        try:
            entries = parse_formula_moiety(formula_moiety)
            scale = _infer_formula_scale(
                observed, _aggregate_counts(entries, heavy_only=True)
            )
            formula_units_consistent = _formula_units_consistency(
                scale,
                formula_units_z,
                n_symops,
                enforce=enforce_formula_units,
            )
            expanded = _expand_moiety_entries(entries, scale)
            try:
                component_h, component_q, component_i = _match_expanded_components(
                    component_keys, expanded, evidence
                )
            except FormulaMismatchError:
                # Repeated ions can have incomplete localized charge evidence.
                # Retry composition matching without this optional charge hint.
                component_h, component_q, component_i = _match_expanded_components(
                    component_keys, expanded, (None,) * num_components
                )
            total_isotopes: Counter[str] = Counter()
            for isotope_counts in component_i:
                total_isotopes.update(dict(isotope_counts))
            return FormulaTargets(
                component_hydrogen_counts=component_h,
                component_charge_targets=component_q,
                component_isotope_counts=component_i,
                total_hydrogen_count=sum(component_h),
                total_charge=sum(component_q),
                total_isotope_counts=tuple(
                    (symbol, int(total_isotopes[symbol]))
                    for symbol in HYDROGEN_ISOTOPES
                    if total_isotopes[symbol]
                ),
                source="chemical_formula_moiety",
                formula_scale=scale,
                formula_units_consistent=formula_units_consistent,
            )
        except FormulaError as exc:
            moiety_error = exc

    if formula_sum is None or str(formula_sum).strip().strip("'\"") in {"", "?", "."}:
        if moiety_error is not None:
            raise moiety_error
        raise FormulaError(
            "Neither a usable moiety formula nor a sum formula was supplied"
        )
    entry = parse_formula(formula_sum)
    scale = _infer_formula_scale(observed, _aggregate_counts((entry,), heavy_only=True))
    formula_units_consistent = _formula_units_consistency(
        scale,
        formula_units_z,
        n_symops,
        enforce=enforce_formula_units,
    )
    scaled_counts = {
        symbol: amount * entry.coefficient * scale for symbol, amount in entry.counts
    }
    isotope_counts = tuple(
        (
            symbol,
            _integer(
                scaled_counts.get(symbol, Fraction(0)), context=f"ASU {symbol} count"
            ),
        )
        for symbol in HYDROGEN_ISOTOPES
        if scaled_counts.get(symbol, Fraction(0)) > 0
    )
    total_h = sum(count for _, count in isotope_counts)
    total_q = _integer(
        Fraction(entry.charge) * entry.coefficient * scale,
        context="ASU formula charge",
    )
    component_h: tuple[int | None, ...] = (
        (total_h,) if num_components == 1 else (None,) * num_components
    )
    component_i: tuple[tuple[tuple[str, int], ...] | None, ...] = (
        (isotope_counts,) if num_components == 1 else (None,) * num_components
    )
    fallback = moiety_error is not None
    component_q_list = [None] * num_components if fallback else list(evidence)
    if num_components == 1:
        component_q_list[0] = total_q
    elif (
        all(value is not None for value in component_q_list)
        and sum(int(value) for value in component_q_list if value is not None)
        != total_q
    ):
        component_q_list = [None] * num_components
    return FormulaTargets(
        component_hydrogen_counts=component_h,
        component_charge_targets=tuple(component_q_list),
        component_isotope_counts=component_i,
        total_hydrogen_count=total_h,
        total_charge=total_q,
        total_isotope_counts=isotope_counts,
        source=(
            "chemical_formula_sum_fallback_from_moiety"
            if fallback
            else "chemical_formula_sum"
        ),
        formula_scale=scale,
        formula_units_consistent=formula_units_consistent,
        fallback_reason=str(moiety_error) if moiety_error is not None else None,
    )


__all__ = [
    "HYDROGEN_ISOTOPES",
    "FormulaAmbiguityError",
    "FormulaEntry",
    "FormulaError",
    "FormulaMismatchError",
    "FormulaTargets",
    "component_heavy_compositions",
    "parse_formula",
    "parse_formula_moiety",
    "resolve_formula_targets",
]
