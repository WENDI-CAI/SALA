from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from sala.core.graph import compute_connected_components


def closest_lattice_shift_torch(
    delta_frac: Tensor,
    lattice: Tensor,
    lattice_inverse: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the exact closest integer image shift for one fractional delta.

    SALA stores lattice vectors by row, so the Cartesian displacement for an
    integer image shift ``n`` is ``(delta_frac + n) @ lattice``. Componentwise
    rounding is exact only for orthogonal metrics; the reciprocal-norm bound
    below enumerates every shift that can improve upon the rounded candidate
    for a general triclinic cell.
    """

    delta = torch.as_tensor(delta_frac)
    if delta.shape != (3,):
        raise ValueError(f"delta_frac must have shape (3,), got {tuple(delta.shape)}")
    lat = torch.as_tensor(lattice, device=delta.device, dtype=torch.float64)
    if lat.shape != (3, 3):
        raise ValueError(f"lattice must have shape (3, 3), got {tuple(lat.shape)}")
    if not bool(torch.isfinite(delta).all().item()) or not bool(
        torch.isfinite(lat).all().item()
    ):
        raise ValueError("delta_frac and lattice must contain only finite values")
    determinant = torch.det(lat)
    if abs(float(determinant.item())) < 1.0e-12:
        raise ValueError("lattice must be nonsingular")

    inverse = (
        torch.linalg.inv(lat)
        if lattice_inverse is None
        else torch.as_tensor(
            lattice_inverse,
            device=delta.device,
            dtype=torch.float64,
        )
    )
    if inverse.shape != (3, 3):
        raise ValueError(
            f"lattice_inverse must have shape (3, 3), got {tuple(inverse.shape)}"
        )

    work_delta = delta.to(torch.float64)
    initial = -torch.round(work_delta).to(torch.long)
    initial_cart = (work_delta + initial.to(torch.float64)) @ lat
    upper_distance = torch.linalg.norm(initial_cart)
    reciprocal_column_norms = torch.linalg.norm(inverse, dim=0)
    bounds = reciprocal_column_norms * (upper_distance + 1.0e-12)
    low = torch.ceil(-work_delta - bounds - 1.0e-12).to(torch.long)
    high = torch.floor(-work_delta + bounds + 1.0e-12).to(torch.long)
    axes = [
        torch.arange(
            int(low[axis].item()),
            int(high[axis].item()) + 1,
            dtype=torch.long,
            device=delta.device,
        )
        for axis in range(3)
    ]
    candidates = torch.cartesian_prod(*axes).reshape(-1, 3)
    if candidates.numel() == 0:
        candidates = initial.view(1, 3)
    distances = torch.linalg.norm(
        (work_delta.unsqueeze(0) + candidates.to(torch.float64)) @ lat,
        dim=-1,
    )
    minimum = distances.min()
    tie_tolerance = max(1.0e-12, float(minimum.item()) * 1.0e-12)
    tied = torch.nonzero(
        distances <= minimum + tie_tolerance,
        as_tuple=False,
    ).view(-1)
    # torch.cartesian_prod over ascending axes is lexicographically ordered, so
    # the first tied candidate gives a deterministic representative.
    shift = candidates[tied[0]]
    return shift, minimum


def make_fractional_coords_contiguous(
    frac_coords: np.ndarray,
    edge_index_conv: np.ndarray,
    component_id: np.ndarray | None = None,
    *,
    lattice: np.ndarray,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(frac_coords, dtype=np.float32)
    continuous, shifts = make_fractional_coords_contiguous_torch(
        torch.as_tensor(coords),
        torch.as_tensor(edge_index_conv, dtype=torch.long),
        (
            None
            if component_id is None
            else torch.as_tensor(component_id, dtype=torch.long)
        ),
        lattice=torch.as_tensor(lattice, dtype=torch.float64),
        tol=tol,
        force_cpu=True,
    )
    return continuous.cpu().numpy(), shifts.cpu().numpy()


def make_fractional_coords_contiguous_torch(
    frac_coords: Tensor,
    edge_index_conv: Tensor,
    component_id: Tensor | None = None,
    *,
    lattice: Tensor,
    tol: float = 1e-3,
    force_cpu: bool = False,
) -> Tuple[Tensor, Tensor]:
    tolerance_A = float(tol)
    if not np.isfinite(tolerance_A) or tolerance_A < 0.0:
        raise ValueError("tol must be finite and non-negative")
    device = frac_coords.device
    dtype = frac_coords.dtype

    if force_cpu and device.type != "cpu":
        frac_coords = frac_coords.detach().cpu()
        edge_index_conv = edge_index_conv.detach().cpu()
        component_id = component_id.detach().cpu() if component_id is not None else None
        lattice = lattice.detach().cpu()

    coords = frac_coords
    num_atoms = coords.shape[0]
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"frac_coords must have shape (N, 3), got {tuple(coords.shape)}"
        )
    if edge_index_conv.ndim != 2 or edge_index_conv.shape[0] != 2:
        raise ValueError(
            "edge_index_conv must have shape (2, E), got "
            f"{tuple(edge_index_conv.shape)}"
        )
    work_lattice = torch.as_tensor(
        lattice,
        device=coords.device,
        dtype=torch.float64,
    )
    if work_lattice.shape != (3, 3):
        raise ValueError(
            f"lattice must have shape (3, 3), got {tuple(work_lattice.shape)}"
        )
    if not bool(torch.isfinite(work_lattice).all().item()):
        raise ValueError("lattice must contain only finite values")
    if abs(float(torch.det(work_lattice).item())) < 1.0e-12:
        raise ValueError("lattice must be nonsingular")
    lattice_inverse = torch.linalg.inv(work_lattice)
    if num_atoms == 0:
        empty = torch.zeros((0, 3), dtype=dtype, device=coords.device)
        return coords.clone(), empty

    if component_id is None:
        component_id = compute_connected_components(
            edge_index_conv.detach().cpu(),
            num_atoms,
        ).to(coords.device)
    component_id = component_id.to(torch.long)
    if component_id.shape != (num_atoms,):
        raise ValueError(
            "component_id must contain one value per atom, got "
            f"{tuple(component_id.shape)} for {num_atoms} atoms"
        )

    component_cpu = component_id.detach().cpu().tolist()
    undirected_edges: set[tuple[int, int]] = set()
    if edge_index_conv.numel() > 0:
        for raw_left, raw_right in edge_index_conv.detach().cpu().T.tolist():
            left, right = int(raw_left), int(raw_right)
            if not 0 <= left < num_atoms or not 0 <= right < num_atoms:
                raise ValueError("edge_index_conv contains an out-of-range atom index")
            if left == right:
                continue
            lower, upper = (left, right) if left < right else (right, left)
            if component_cpu[lower] != component_cpu[upper]:
                raise ValueError("covalent edge connects atoms from different components")
            undirected_edges.add((lower, upper))

    neighbors: list[list[tuple[int, Tensor]]] = [[] for _ in range(num_atoms)]
    for left, right in sorted(undirected_edges):
        relative_shift, _ = closest_lattice_shift_torch(
            coords[right] - coords[left],
            work_lattice,
            lattice_inverse=lattice_inverse,
        )
        neighbors[left].append((right, relative_shift))
        neighbors[right].append((left, -relative_shift))

    integer_shifts = torch.zeros(
        (num_atoms, 3),
        dtype=torch.long,
        device=coords.device,
    )
    visited = [False] * num_atoms
    component_nodes: dict[int, list[int]] = {}
    for atom_index, component in enumerate(component_cpu):
        component_nodes.setdefault(int(component), []).append(atom_index)

    for nodes in component_nodes.values():
        for anchor in sorted(nodes):
            if visited[anchor]:
                continue
            queue = [anchor]
            visited[anchor] = True
            cursor = 0
            while cursor < len(queue):
                left = queue[cursor]
                cursor += 1
                for right, relative_shift in neighbors[left]:
                    proposal = integer_shifts[left] + relative_shift
                    if visited[right]:
                        if not torch.equal(proposal, integer_shifts[right]):
                            mismatch = (
                                proposal - integer_shifts[right]
                            ).to(torch.float64) @ work_lattice
                            if float(torch.linalg.norm(mismatch).item()) > tolerance_A:
                                raise ValueError(
                                    "covalent component has inconsistent periodic image "
                                    "winding; the structure may be a periodic polymer or "
                                    "contain invalid bond topology"
                                )
                        continue
                    integer_shifts[right] = proposal
                    visited[right] = True
                    queue.append(right)

    shifts = integer_shifts.to(dtype=dtype)
    continuous = coords + shifts
    if force_cpu and device.type != "cpu":
        continuous = continuous.to(device)
        shifts = shifts.to(device)
    return continuous, shifts


def remove_graph_translation_gauge(
    coords: Tensor,
    batch_idx: Tensor,
    num_graphs: int | None = None,
) -> Tensor:
    if coords.numel() == 0:
        return coords
    if batch_idx.numel() != coords.shape[0]:
        raise ValueError(
            f"batch_idx must align with coords rows, got {batch_idx.numel()} vs {coords.shape[0]}"
        )
    graph_index = batch_idx.long()
    n_graphs = int(num_graphs) if num_graphs is not None else int(graph_index.max().item()) + 1
    sums = coords.new_zeros((n_graphs, coords.shape[-1]))
    sums.index_add_(0, graph_index, coords)
    counts = coords.new_zeros((n_graphs, 1))
    counts.index_add_(0, graph_index, torch.ones((coords.shape[0], 1), device=coords.device, dtype=coords.dtype))
    mean = sums / counts.clamp(min=1.0)
    return coords - mean[graph_index]


make_molecule_contiguous = make_fractional_coords_contiguous_torch


__all__ = [
    "closest_lattice_shift_torch",
    "make_fractional_coords_contiguous",
    "make_fractional_coords_contiguous_torch",
    "make_molecule_contiguous",
    "remove_graph_translation_gauge",
]
