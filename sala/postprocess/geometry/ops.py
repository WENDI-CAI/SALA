"""Coordinate preparation used by periodic postprocessing geometry."""

from __future__ import annotations

import torch
from torch import Tensor

from sala.postprocess.geometry.periodic import (
    prepare_periodic_lattice,
    unwrap_component_graph,
)


def contiguous_fractional_coords(asu) -> Tensor:
    frac = asu.frac_coords_gt.float()
    component_id = asu.component_id.long().to(device=frac.device)
    edge_index = asu.edge_index_conv.long().to(device=frac.device)
    periodic = prepare_periodic_lattice(asu.lattice_gt.to(device=frac.device))
    contiguous = frac.clone()
    for component_value in torch.unique(component_id, sorted=True):
        indices = torch.where(component_id == component_value)[0]
        if indices.numel() == 0:
            continue
        global_to_local = torch.full(
            (frac.shape[0],), -1, dtype=torch.long, device=frac.device
        )
        global_to_local[indices] = torch.arange(indices.numel(), device=frac.device)
        if edge_index.numel():
            edge_mask = (component_id[edge_index[0]] == component_value) & (
                component_id[edge_index[1]] == component_value
            )
            component_edges = global_to_local[edge_index[:, edge_mask]]
        else:
            component_edges = torch.empty(
                (2, 0), dtype=torch.long, device=frac.device
            )
        contiguous[indices] = unwrap_component_graph(
            frac[indices], component_edges, periodic
        )
    return contiguous


__all__ = ["contiguous_fractional_coords"]
