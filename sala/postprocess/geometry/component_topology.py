from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import lru_cache
from operator import index as integer_index

import networkx as nx
import torch

BondLabel = tuple[float, ...]


_ASSUME_ATOM_ORDER_PRESERVED: ContextVar[bool] = ContextVar(
    "sala_assume_atom_order_preserved",
    default=False,
)


def set_atom_order_preserved_mode(enabled: bool) -> Token:
    return _ASSUME_ATOM_ORDER_PRESERVED.set(bool(enabled))


def reset_atom_order_preserved_mode(token: Token) -> None:
    _ASSUME_ATOM_ORDER_PRESERVED.reset(token)


@dataclass(frozen=True)
class ComponentTopology:
    component_id: int
    atom_indices: tuple[int, ...]
    atomic_numbers: tuple[int, ...]
    edges: tuple[tuple[int, int, BondLabel], ...]
    component_formal_charge: float
    signature: str


AtomMapping = tuple[int, ...]
MappingPredicate = Callable[[AtomMapping], bool]


@dataclass(frozen=True)
class ComponentIsomorphismResult:
    """A bounded prefix of compatible graph-isomorphism mappings.

    ``truncated`` is false only when the underlying isomorphism iterator was
    exhausted.  It is true when either another accepted mapping was observed
    beyond ``limit`` or the raw-mapping inspection budget was exhausted before
    completeness could be established.
    """

    mappings: tuple[AtomMapping, ...]
    truncated: bool
    num_inspected: int


def _bond_label(edge_attr: torch.Tensor, index: int) -> BondLabel:
    value = torch.as_tensor(edge_attr[index]).detach().cpu().reshape(-1)
    return tuple(round(float(item), 8) for item in value.tolist())


def _as_graph(topology: ComponentTopology) -> nx.Graph:
    graph = nx.Graph()
    for index, atomic_number in enumerate(topology.atomic_numbers):
        graph.add_node(index, atomic_number=int(atomic_number))
    for left, right, label in topology.edges:
        graph.add_edge(int(left), int(right), bond_type=label)
    return graph


def _component_graph_matcher(
    left: ComponentTopology,
    right: ComponentTopology,
    *,
    ignore_bond_types: bool,
) -> nx.algorithms.isomorphism.GraphMatcher:
    edge_match = None
    if not ignore_bond_types:
        edge_match = nx.algorithms.isomorphism.categorical_edge_match(
            "bond_type",
            (),
        )
    return nx.algorithms.isomorphism.GraphMatcher(
        _as_graph(left),
        _as_graph(right),
        node_match=nx.algorithms.isomorphism.categorical_node_match(
            "atomic_number",
            0,
        ),
        edge_match=edge_match,
    )


def _component_topologies_may_match(
    left: ComponentTopology,
    right: ComponentTopology,
    *,
    ignore_bond_types: bool,
) -> bool:
    if len(left.atomic_numbers) != len(right.atomic_numbers):
        return False
    if not ignore_bond_types:
        return left.signature == right.signature
    return (
        Counter(left.atomic_numbers) == Counter(right.atomic_numbers)
        and len(left.edges) == len(right.edges)
        and f"{float(left.component_formal_charge):.8g}"
        == f"{float(right.component_formal_charge):.8g}"
    )


def _same_indexed_topology(
    left: ComponentTopology,
    right: ComponentTopology,
    *,
    ignore_bond_types: bool,
) -> bool:
    if left.atomic_numbers != right.atomic_numbers:
        return False
    if not ignore_bond_types:
        return left.edges == right.edges
    left_connectivity = {
        (min(int(u), int(v)), max(int(u), int(v))) for u, v, _ in left.edges
    }
    right_connectivity = {
        (min(int(u), int(v)), max(int(u), int(v))) for u, v, _ in right.edges
    }
    return left_connectivity == right_connectivity


def _validated_nonnegative_integer(value: int, *, name: str) -> int:
    try:
        result = integer_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _topology_signature(
    atomic_numbers: tuple[int, ...],
    edges: tuple[tuple[int, int, BondLabel], ...],
    component_formal_charge: float,
) -> str:
    temporary = ComponentTopology(
        component_id=-1,
        atom_indices=tuple(range(len(atomic_numbers))),
        atomic_numbers=atomic_numbers,
        edges=edges,
        component_formal_charge=float(component_formal_charge),
        signature="",
    )
    graph_hash = nx.weisfeiler_lehman_graph_hash(
        _as_graph(temporary),
        node_attr="atomic_number",
        edge_attr="bond_type",
    )
    formula = tuple(sorted(Counter(atomic_numbers).items()))
    return (
        f"formula={formula}|charge={float(component_formal_charge):.8g}|wl={graph_hash}"
    )


def build_component_topologies(
    asu, *, atom_scope: str = "heavy"
) -> tuple[ComponentTopology, ...]:
    if atom_scope not in {"heavy", "all"}:
        raise ValueError("atom_scope must be 'heavy' or 'all'")
    atom_types = asu.atom_types.long()
    component_ids = asu.component_id.long()
    edge_index = asu.edge_index_conv.long()
    edge_attr = asu.edge_attr_conv
    selected = torch.ones_like(atom_types, dtype=torch.bool)
    if atom_scope == "heavy":
        selected &= atom_types != 1
    topologies: list[ComponentTopology] = []
    for component_tensor in torch.unique(component_ids, sorted=True):
        component_id = int(component_tensor.item())
        atom_indices_tensor = torch.where(selected & (component_ids == component_id))[0]
        if atom_indices_tensor.numel() == 0:
            continue
        atom_indices = tuple(int(value) for value in atom_indices_tensor.tolist())
        global_to_local = {
            global_index: local_index
            for local_index, global_index in enumerate(atom_indices)
        }
        atomic_numbers = tuple(int(atom_types[index].item()) for index in atom_indices)
        unique_edges: dict[tuple[int, int], BondLabel] = {}
        for edge_position, (left_global, right_global) in enumerate(
            edge_index.detach().cpu().T.tolist()
        ):
            left_global = int(left_global)
            right_global = int(right_global)
            if (
                left_global not in global_to_local
                or right_global not in global_to_local
            ):
                continue
            left = global_to_local[left_global]
            right = global_to_local[right_global]
            if left == right:
                continue
            key = (min(left, right), max(left, right))
            label = _bond_label(edge_attr, edge_position)
            existing = unique_edges.get(key)
            if existing is not None and existing != label:
                raise ValueError(f"Conflicting bond labels for component edge {key}")
            unique_edges[key] = label
        edges = tuple(
            (left, right, unique_edges[(left, right)])
            for left, right in sorted(unique_edges)
        )
        charges = torch.as_tensor(asu.component_formal_charge).reshape(-1)
        component_formal_charge = (
            float(charges[component_id].item())
            if component_id < charges.numel()
            else 0.0
        )
        signature = _topology_signature(atomic_numbers, edges, component_formal_charge)
        topologies.append(
            ComponentTopology(
                component_id=component_id,
                atom_indices=atom_indices,
                atomic_numbers=atomic_numbers,
                edges=edges,
                component_formal_charge=component_formal_charge,
                signature=signature,
            )
        )
    return tuple(topologies)


@lru_cache(maxsize=4096)
def _exhaustive_component_isomorphisms(
    left: ComponentTopology,
    right: ComponentTopology,
) -> tuple[AtomMapping, ...]:
    if not _component_topologies_may_match(
        left,
        right,
        ignore_bond_types=False,
    ):
        return ()
    matcher = _component_graph_matcher(
        left,
        right,
        ignore_bond_types=False,
    )
    mappings = {
        tuple(int(mapping[index]) for index in range(len(left.atomic_numbers)))
        for mapping in matcher.isomorphisms_iter()
    }
    return tuple(sorted(mappings))


def bounded_component_isomorphisms(
    left: ComponentTopology,
    right: ComponentTopology,
    *,
    limit: int,
    max_inspected: int | None = None,
    mapping_predicate: MappingPredicate | None = None,
    ignore_bond_types: bool = False,
) -> ComponentIsomorphismResult:
    """Enumerate a deterministic, bounded prefix of graph isomorphisms.

    ``limit`` caps the mappings returned after ``mapping_predicate`` is
    applied. ``max_inspected`` independently caps raw mappings pulled from
    NetworkX.  By default at most ``limit + 1`` raw mappings are inspected, so
    the common unfiltered case can detect truncation without materializing all
    automorphisms.  When the raw inspection cap is reached, ``truncated`` is
    conservatively true because completeness has not been established.

    Setting ``ignore_bond_types`` retains atomic-number and connectivity
    matching while omitting edge-label comparison.
    :func:`component_isomorphisms` performs strict, exhaustive matching.
    """

    accepted_limit = _validated_nonnegative_integer(limit, name="limit")
    if max_inspected is None:
        inspection_limit = max(accepted_limit + 1, 1)
    else:
        inspection_limit = _validated_nonnegative_integer(
            max_inspected,
            name="max_inspected",
        )

    if not _component_topologies_may_match(
        left,
        right,
        ignore_bond_types=ignore_bond_types,
    ):
        return ComponentIsomorphismResult((), False, 0)

    if _ASSUME_ATOM_ORDER_PRESERVED.get() and _same_indexed_topology(
        left,
        right,
        ignore_bond_types=ignore_bond_types,
    ):
        if inspection_limit == 0:
            return ComponentIsomorphismResult((), True, 0)
        identity = tuple(range(len(left.atomic_numbers)))
        if mapping_predicate is not None and not mapping_predicate(identity):
            return ComponentIsomorphismResult((), False, 1)
        if accepted_limit == 0:
            return ComponentIsomorphismResult((), True, 1)
        return ComponentIsomorphismResult((identity,), False, 1)

    matcher = _component_graph_matcher(
        left,
        right,
        ignore_bond_types=ignore_bond_types,
    )
    iterator = matcher.isomorphisms_iter()
    mappings: list[AtomMapping] = []
    num_inspected = 0
    while num_inspected < inspection_limit:
        try:
            raw_mapping = next(iterator)
        except StopIteration:
            return ComponentIsomorphismResult(
                tuple(mappings),
                False,
                num_inspected,
            )
        num_inspected += 1
        mapping = tuple(
            int(raw_mapping[index]) for index in range(len(left.atomic_numbers))
        )
        if mapping_predicate is not None and not mapping_predicate(mapping):
            continue
        if len(mappings) == accepted_limit:
            return ComponentIsomorphismResult(
                tuple(mappings),
                True,
                num_inspected,
            )
        mappings.append(mapping)

    return ComponentIsomorphismResult(tuple(mappings), True, num_inspected)


def component_isomorphisms(
    left: ComponentTopology,
    right: ComponentTopology,
) -> tuple[AtomMapping, ...]:
    if left.signature != right.signature or len(left.atomic_numbers) != len(
        right.atomic_numbers
    ):
        return ()
    if (
        _ASSUME_ATOM_ORDER_PRESERVED.get()
        and left.atomic_numbers == right.atomic_numbers
        and left.edges == right.edges
    ):
        return (tuple(range(len(left.atomic_numbers))),)
    return _exhaustive_component_isomorphisms(left, right)


__all__ = [
    "ComponentIsomorphismResult",
    "ComponentTopology",
    "bounded_component_isomorphisms",
    "build_component_topologies",
    "component_isomorphisms",
    "reset_atom_order_preserved_mode",
    "set_atom_order_preserved_mode",
]
