"""Scatter utilities for normalized Cartesian/whitened-q self-conditioning."""

from __future__ import annotations

from typing import Optional

import torch

from sala.core.batch_ops import GraphSubBatchMap
from sala.flow_matching.types import FlowSelfConditioningState


def _validate_state_map(
    state: FlowSelfConditioningState,
    index_map: GraphSubBatchMap,
    *,
    subset: bool,
) -> None:
    expected_graphs = index_map.num_graphs if subset else index_map.parent_num_graphs
    expected_atoms = index_map.num_atoms if subset else index_map.parent_num_atoms
    if state.batch_size != expected_graphs or state.num_atoms != expected_atoms:
        scope = "subset" if subset else "parent"
        raise ValueError(
            f"flow self-conditioning {scope} state has "
            f"(B={state.batch_size}, N={state.num_atoms}); expected "
            f"(B={expected_graphs}, N={expected_atoms})"
        )


def scatter_flow_self_conditioning_state(
    subset: FlowSelfConditioningState,
    index_map: GraphSubBatchMap,
    *,
    base: Optional[FlowSelfConditioningState] = None,
) -> FlowSelfConditioningState:
    """Merge a selected normalized residual state into its parent batch."""
    _validate_state_map(subset, index_map, subset=True)
    if base is None:
        base = FlowSelfConditioningState(
            atom_residual=torch.zeros(
                (index_map.parent_num_atoms, 3),
                device=subset.atom_residual.device,
                dtype=subset.atom_residual.dtype,
            ),
            lattice_residual=torch.zeros(
                (index_map.parent_num_graphs, 6),
                device=subset.lattice_residual.device,
                dtype=subset.lattice_residual.dtype,
            ),
            graph_present=torch.zeros(
                (index_map.parent_num_graphs,),
                device=subset.graph_present.device,
                dtype=torch.bool,
            ),
        )
    _validate_state_map(base, index_map, subset=False)
    if base.atom_residual.device != subset.atom_residual.device:
        raise ValueError("subset and base self-conditioning states must share a device")
    atom_present = subset.graph_present.index_select(
        0, index_map.atom_graph_index.to(device=subset.graph_present.device)
    )
    atom_residual = torch.where(
        atom_present.unsqueeze(-1),
        subset.atom_residual,
        torch.zeros_like(subset.atom_residual),
    )
    lattice_residual = torch.where(
        subset.graph_present.unsqueeze(-1),
        subset.lattice_residual,
        torch.zeros_like(subset.lattice_residual),
    )
    return FlowSelfConditioningState._from_owned_detached(
        atom_residual=index_map.merge_atoms(base.atom_residual, atom_residual),
        lattice_residual=index_map.merge_graphs(
            base.lattice_residual, lattice_residual
        ),
        graph_present=index_map.merge_graphs(
            base.graph_present, subset.graph_present
        ),
    )


__all__ = ["scatter_flow_self_conditioning_state"]
