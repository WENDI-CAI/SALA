"""Validated atom charges and permutation-invariant component identities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import torch
from torch import Tensor

if TYPE_CHECKING:
    from sala.core.types import CifData


MAX_LOCALIZED_FORMAL_CHARGE = 2


@dataclass(frozen=True)
class ChargeFeatures:
    """Two-level charge representation for one ASU.

    ``atom_formal_charge`` is only consumed where ``atom_charge_mask`` is true.
    The component charge remains authoritative even when localization is absent
    or chemically implausible.
    """

    atom_formal_charge: Tensor
    atom_charge_mask: Tensor
    component_formal_charge: Tensor


def sanitize_atom_formal_charges(
    *,
    raw_atom_charge: Tensor,
    raw_atom_charge_mask: Tensor | None = None,
    atom_types: Tensor,
    component_id: Tensor,
    component_formal_charge: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return conservative atom-localized charges and a validity mask.

    Cleaned CIFs occasionally store an entire polyatomic component charge on a
    single atom (for example C=-7 or I=-8).  Such assignments conserve charge
    but are not atom-resolved chemistry.  A component is accepted only when its
    atom charges sum to the authoritative component charge, all magnitudes are
    at most two, and a polyatomic charge with |q|>1 is not concentrated on one
    atom.  Rejected components retain their net component charge while their
    atom localization is represented as unknown.
    """

    num_atoms = int(atom_types.numel())
    if raw_atom_charge.shape != (num_atoms,):
        raise ValueError("raw_atom_charge must contain one value per atom")
    if component_id.shape != (num_atoms,):
        raise ValueError("component_id must contain one value per atom")
    num_components = int(component_formal_charge.numel())
    if num_atoms:
        valid_id = (component_id >= 0) & (component_id < max(num_components, 1))
        if not bool(valid_id.all()):
            raise ValueError("component_id contains an out-of-range value")

    charge = raw_atom_charge.to(dtype=torch.long).clone()
    if raw_atom_charge_mask is None:
        source_mask = torch.ones(num_atoms, dtype=torch.bool, device=charge.device)
    else:
        if raw_atom_charge_mask.shape != (num_atoms,):
            raise ValueError("raw_atom_charge_mask must contain one value per atom")
        source_mask = raw_atom_charge_mask.to(device=charge.device, dtype=torch.bool)
    mask = source_mask.clone()
    for component_index in range(num_components):
        atom_index = torch.where(component_id.long() == component_index)[0]
        if atom_index.numel() == 0:
            raise ValueError("component_formal_charge contains an empty component")
        values = charge.index_select(0, atom_index)
        values_known = source_mask.index_select(0, atom_index)
        expected = int(component_formal_charge[component_index].item())
        nonzero = values != 0
        charge_conserved = int(values.sum().item()) == expected
        bounded = bool((values.abs() <= MAX_LOCALIZED_FORMAL_CHARGE).all())
        concentrated_multicharge = (
            atom_index.numel() > 1
            and abs(expected) > 1
            and int(nonzero.sum().item()) == 1
        )
        localized = (
            bool(values_known.all())
            and charge_conserved
            and bounded
            and not concentrated_multicharge
        )
        if not localized:
            charge[atom_index] = 0
            mask[atom_index] = False
    return charge, mask


def build_charge_features(
    cif_data: "CifData",
    *,
    component_id: Tensor,
    component_formal_charge: Tensor,
) -> ChargeFeatures:
    num_atoms = len(cif_data.atom_types)
    source = cif_data.atom_formal_charges
    source_mask = cif_data.atom_formal_charge_mask
    if source and len(source) == num_atoms:
        raw = torch.as_tensor(source, dtype=torch.long, device=component_id.device)
        raw_mask = (
            torch.as_tensor(source_mask, dtype=torch.bool, device=component_id.device)
            if len(source_mask) == num_atoms
            else torch.zeros(num_atoms, dtype=torch.bool, device=component_id.device)
        )
        atom_charge, atom_mask = sanitize_atom_formal_charges(
            raw_atom_charge=raw,
            raw_atom_charge_mask=raw_mask,
            atom_types=torch.as_tensor(
                cif_data.atom_types, dtype=torch.long, device=component_id.device
            ),
            component_id=component_id.long(),
            component_formal_charge=component_formal_charge.long(),
        )
    else:
        atom_charge = torch.zeros(num_atoms, dtype=torch.long, device=component_id.device)
        atom_mask = torch.zeros(num_atoms, dtype=torch.bool, device=component_id.device)
    return ChargeFeatures(
        atom_formal_charge=atom_charge,
        atom_charge_mask=atom_mask,
        component_formal_charge=component_formal_charge.long(),
    )


def _component_graphs(
    *,
    atom_types: Tensor,
    hybridization: Tensor,
    atom_formal_charge: Tensor,
    atom_charge_mask: Tensor,
    component_id: Tensor,
    component_formal_charge: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
) -> list[nx.Graph]:
    num_atoms = int(atom_types.numel())
    fields = {
        "hybridization": hybridization,
        "atom_formal_charge": atom_formal_charge,
        "atom_charge_mask": atom_charge_mask,
        "component_id": component_id,
    }
    for name, value in fields.items():
        if value.shape != (num_atoms,):
            raise ValueError(f"{name} must contain one value per atom")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_attr.shape != (edge_index.shape[1],):
        raise ValueError("edge_attr must contain one value per directed edge")

    graphs: list[nx.Graph] = []
    for component_index in range(int(component_formal_charge.numel())):
        atom_index = torch.where(component_id.long() == component_index)[0]
        graph = nx.Graph(component_charge=int(component_formal_charge[component_index]))
        global_to_local = {
            int(global_index): local_index
            for local_index, global_index in enumerate(atom_index.tolist())
        }
        for global_index, local_index in global_to_local.items():
            charge_known = bool(atom_charge_mask[global_index])
            graph.add_node(
                local_index,
                label=(
                    int(atom_types[global_index]),
                    int(hybridization[global_index]),
                    int(atom_formal_charge[global_index]) if charge_known else 0,
                    charge_known,
                ),
            )
        for edge_position in range(int(edge_index.shape[1])):
            left = int(edge_index[0, edge_position])
            right = int(edge_index[1, edge_position])
            if left not in global_to_local or right not in global_to_local or left == right:
                continue
            local_left = global_to_local[left]
            local_right = global_to_local[right]
            bond_type = int(edge_attr[edge_position])
            previous = graph.get_edge_data(local_left, local_right)
            if previous is not None and int(previous["bond_type"]) != bond_type:
                raise ValueError("Conflicting bond types for one component edge")
            graph.add_edge(local_left, local_right, bond_type=bond_type)
        graphs.append(graph)
    return graphs


def assign_component_type_ids(
    *,
    atom_types: Tensor,
    hybridization: Tensor,
    atom_formal_charge: Tensor,
    atom_charge_mask: Tensor,
    component_id: Tensor,
    component_formal_charge: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
) -> Tensor:
    """Assign local ids to exactly isomorphic chemical components.

    Numeric ids are local and carry no ordinal semantics.  Only equality is
    consumed by the pair encoder.  A WL hash provides a fast candidate bucket;
    exact graph isomorphism prevents hash collisions or formula-only aliases.
    """

    graphs = _component_graphs(
        atom_types=atom_types,
        hybridization=hybridization,
        atom_formal_charge=atom_formal_charge,
        atom_charge_mask=atom_charge_mask,
        component_id=component_id,
        component_formal_charge=component_formal_charge,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    node_match = nx.algorithms.isomorphism.categorical_node_match("label", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("bond_type", -1)
    buckets: dict[tuple[int, int, int, str], list[tuple[int, nx.Graph]]] = defaultdict(list)
    type_ids = torch.empty(
        len(graphs), device=component_id.device, dtype=torch.long
    )
    next_type = 0
    for component_index, graph in enumerate(graphs):
        # NetworkX's WL implementation hashes the string form of the attributes.
        for _, attributes in graph.nodes(data=True):
            attributes["wl_label"] = repr(attributes["label"])
        signature = (
            int(graph.graph["component_charge"]),
            graph.number_of_nodes(),
            graph.number_of_edges(),
            nx.weisfeiler_lehman_graph_hash(
                graph, node_attr="wl_label", edge_attr="bond_type"
            ),
        )
        assigned = None
        for type_id, representative in buckets[signature]:
            if nx.is_isomorphic(
                graph,
                representative,
                node_match=node_match,
                edge_match=edge_match,
            ):
                assigned = type_id
                break
        if assigned is None:
            assigned = next_type
            next_type += 1
            buckets[signature].append((assigned, graph))
        type_ids[component_index] = assigned
    return type_ids


__all__ = [
    "ChargeFeatures",
    "MAX_LOCALIZED_FORMAL_CHARGE",
    "assign_component_type_ids",
    "build_charge_features",
    "sanitize_atom_formal_charges",
]
