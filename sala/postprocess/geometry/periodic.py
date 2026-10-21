from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from sala.core.coordinates import (
    closest_lattice_shift_torch,
    make_fractional_coords_contiguous_torch,
)


class PeriodicSearchBudgetExceeded(RuntimeError):
    """Raised before allocating a periodic translation box that is too large."""

    def __init__(self, *, candidate_count: int, budget: int) -> None:
        self.candidate_count = int(candidate_count)
        self.budget = int(budget)
        super().__init__(
            "periodic translation candidate budget exceeded: "
            f"candidate_count={self.candidate_count}, budget={self.budget}"
        )


@dataclass(frozen=True)
class PeriodicLattice:
    lattice: Tensor
    inverse: Tensor
    reciprocal_column_norms: Tensor
    reduced_lattice: Tensor
    reduced_transform: Tensor


@dataclass(frozen=True)
class PeriodicShiftResult:
    shift: Tensor
    distance_A: float
    complete: bool


@dataclass(frozen=True)
class PeriodicSearchResult:
    shifts: Tensor
    distances_A: Tensor
    complete: bool
    cutoff_A: float
    candidate_count: int


@dataclass(frozen=True)
class NearestPeriodicResult:
    base_indices: Tensor
    shifts: Tensor
    distances_A: Tensor
    complete: bool
    cutoff_A: float


def prepare_periodic_lattice(lattice: Tensor) -> PeriodicLattice:
    source = torch.as_tensor(lattice)
    if source.shape != (3, 3):
        raise ValueError("lattice must have shape [3, 3]")
    if not torch.isfinite(source).all().item():
        raise ValueError("lattice must contain only finite values")
    work = source.to(dtype=torch.float64)
    determinant = torch.det(work)
    if (
        not math.isfinite(float(determinant.item()))
        or abs(float(determinant.item())) < 1e-12
    ):
        raise ValueError("lattice must be nonsingular")
    inverse = torch.linalg.inv(work)
    reciprocal_norms = torch.linalg.norm(inverse, dim=0)
    reduced = work
    transform = torch.eye(3, dtype=torch.long, device=source.device)
    try:
        import spglib

        try:
            spglib.error.OLD_ERROR_HANDLING = False
        except AttributeError:
            pass

        reduced_np = spglib.niggli_reduce(work.detach().cpu().numpy(), eps=1e-8)
        if reduced_np is not None:
            candidate = torch.from_numpy(np.asarray(reduced_np)).to(
                device=source.device, dtype=torch.float64
            )
            candidate_transform = torch.round(candidate @ inverse).to(torch.long)
            reconstructed = candidate_transform.to(torch.float64) @ work
            det_transform = round(
                float(torch.det(candidate_transform.to(torch.float64)).item())
            )
            if abs(det_transform) == 1 and torch.allclose(
                candidate,
                reconstructed,
                atol=1e-7,
                rtol=1e-7,
            ):
                reduced = candidate
                transform = candidate_transform
    except Exception:  # A failed reduction leaves the input basis unchanged.
        reduced = work
        transform = torch.eye(3, dtype=torch.long, device=source.device)
    return PeriodicLattice(
        lattice=work,
        inverse=inverse,
        reciprocal_column_norms=reciprocal_norms,
        reduced_lattice=reduced,
        reduced_transform=transform,
    )


def _candidate_budget(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("max_candidates must be a positive integer or None")
    budget = int(value)
    if budget < 1:
        raise ValueError("max_candidates must be a positive integer or None")
    return budget


def _integer_box_candidate_count(low: Tensor, high: Tensor) -> int:
    if low.shape != (3,) or high.shape != (3,):
        raise ValueError("integer-box bounds must both have shape [3]")
    count = 1
    for axis in range(3):
        extent = int(high[axis].item()) - int(low[axis].item()) + 1
        if extent <= 0:
            return 0
        count *= extent
    return count


def _integer_box(low: Tensor, high: Tensor, *, device: torch.device) -> Tensor:
    ranges = [
        torch.arange(
            int(low[i].item()), int(high[i].item()) + 1, dtype=torch.long, device=device
        )
        for i in range(3)
    ]
    if any(values.numel() == 0 for values in ranges):
        return torch.empty((0, 3), dtype=torch.long, device=device)
    return torch.cartesian_prod(*ranges).reshape(-1, 3)


def _sort_candidates(shifts: Tensor, distances: Tensor) -> tuple[Tensor, Tensor]:
    if shifts.numel() == 0:
        return shifts, distances
    records = sorted(
        range(int(shifts.shape[0])),
        key=lambda index: (
            float(distances[index].item()),
            int(shifts[index, 0].item()),
            int(shifts[index, 1].item()),
            int(shifts[index, 2].item()),
        ),
    )
    order = torch.tensor(records, dtype=torch.long, device=shifts.device)
    return shifts[order], distances[order]


def closest_lattice_shift(
    delta_frac: Tensor, lattice: PeriodicLattice
) -> PeriodicShiftResult:
    delta = torch.as_tensor(delta_frac, device=lattice.lattice.device)
    shift, minimum = closest_lattice_shift_torch(
        delta,
        lattice.lattice,
        lattice_inverse=lattice.inverse,
    )
    return PeriodicShiftResult(
        shift=shift,
        distance_A=float(minimum.item()),
        complete=True,
    )


def translations_within_cutoff(
    delta_frac: Tensor,
    lattice: PeriodicLattice,
    *,
    cutoff_A: float,
    tolerance_A: float = 1e-9,
    max_candidates: int | None = None,
) -> PeriodicSearchResult:
    cutoff = float(cutoff_A)
    if not math.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("cutoff_A must be finite and non-negative")
    delta = torch.as_tensor(
        delta_frac, device=lattice.lattice.device, dtype=torch.float64
    )
    if delta.shape != (3,):
        raise ValueError("delta_frac must have shape [3]")
    candidate_budget = _candidate_budget(max_candidates)
    bounds = lattice.reciprocal_column_norms * (cutoff + float(tolerance_A))
    low = torch.ceil(-delta - bounds - 1e-12).to(torch.long)
    high = torch.floor(-delta + bounds + 1e-12).to(torch.long)
    candidate_count = _integer_box_candidate_count(low, high)
    if candidate_budget is not None and candidate_count > candidate_budget:
        raise PeriodicSearchBudgetExceeded(
            candidate_count=candidate_count,
            budget=candidate_budget,
        )
    candidates = _integer_box(low, high, device=delta.device)
    if candidates.numel() == 0:
        return PeriodicSearchResult(
            shifts=candidates,
            distances_A=torch.empty(0, dtype=torch.float64, device=delta.device),
            complete=True,
            cutoff_A=cutoff,
            candidate_count=candidate_count,
        )
    distances = torch.linalg.norm(
        (delta.unsqueeze(0) + candidates.to(torch.float64)) @ lattice.lattice,
        dim=-1,
    )
    keep = distances <= cutoff + float(tolerance_A)
    shifts, kept_distances = _sort_candidates(candidates[keep], distances[keep])
    return PeriodicSearchResult(
        shifts=shifts,
        distances_A=kept_distances,
        complete=True,
        cutoff_A=cutoff,
        candidate_count=candidate_count,
    )


def nearest_periodic_candidates(
    deltas_frac: Tensor,
    lattice: PeriodicLattice,
    *,
    count: int,
    tie_tolerance_A: float = 1e-8,
    initial_cutoff_A: float | None = None,
    max_candidates: int | None = None,
) -> NearestPeriodicResult:
    deltas = torch.as_tensor(
        deltas_frac, device=lattice.lattice.device, dtype=torch.float64
    )
    if deltas.ndim != 2 or deltas.shape[1] != 3 or deltas.shape[0] == 0:
        raise ValueError(
            "deltas_frac must have shape [num_bases, 3] with num_bases > 0"
        )
    requested = int(count)
    if requested < 1:
        raise ValueError("count must be positive")
    candidate_budget = _candidate_budget(max_candidates)
    row_norms = torch.linalg.norm(lattice.lattice, dim=-1)
    cutoff = (
        float(initial_cutoff_A)
        if initial_cutoff_A is not None
        else float(row_norms.min().item())
    )
    cutoff = max(cutoff, 1e-6)
    for _ in range(64):
        base_parts: list[Tensor] = []
        shift_parts: list[Tensor] = []
        distance_parts: list[Tensor] = []
        for base_index, delta in enumerate(deltas):
            local = translations_within_cutoff(
                delta,
                lattice,
                cutoff_A=cutoff,
                tolerance_A=tie_tolerance_A,
                max_candidates=candidate_budget,
            )
            if local.shifts.numel() == 0:
                continue
            base_parts.append(
                torch.full(
                    (local.shifts.shape[0],),
                    int(base_index),
                    dtype=torch.long,
                    device=deltas.device,
                )
            )
            shift_parts.append(local.shifts)
            distance_parts.append(local.distances_A)
        if distance_parts:
            base_indices = torch.cat(base_parts)
            shifts = torch.cat(shift_parts)
            distances = torch.cat(distance_parts)
            order_records = sorted(
                range(int(distances.numel())),
                key=lambda index: (
                    float(distances[index].item()),
                    int(base_indices[index].item()),
                    int(shifts[index, 0].item()),
                    int(shifts[index, 1].item()),
                    int(shifts[index, 2].item()),
                ),
            )
            order = torch.tensor(order_records, dtype=torch.long, device=deltas.device)
            base_indices = base_indices[order]
            shifts = shifts[order]
            distances = distances[order]
            if distances.numel() >= requested:
                boundary = float(distances[requested - 1].item())
                final_cutoff = boundary + float(tie_tolerance_A)
                if cutoff + 1e-12 < final_cutoff:
                    cutoff = final_cutoff
                    continue
                keep = distances <= final_cutoff + 1e-12
                return NearestPeriodicResult(
                    base_indices=base_indices[keep],
                    shifts=shifts[keep],
                    distances_A=distances[keep],
                    complete=True,
                    cutoff_A=final_cutoff,
                )
        cutoff *= 2.0
    raise RuntimeError("nearest periodic search did not converge after 64 expansions")


def unwrap_component_graph(
    frac_coords: Tensor,
    edge_index: Tensor,
    lattice: PeriodicLattice,
) -> Tensor:
    frac = torch.as_tensor(frac_coords)
    unwrapped, _ = make_fractional_coords_contiguous_torch(
        frac,
        torch.as_tensor(edge_index, dtype=torch.long, device=frac.device),
        lattice=lattice.lattice,
    )
    return unwrapped


__all__ = [
    "NearestPeriodicResult",
    "PeriodicLattice",
    "PeriodicSearchBudgetExceeded",
    "PeriodicSearchResult",
    "PeriodicShiftResult",
    "closest_lattice_shift",
    "nearest_periodic_candidates",
    "prepare_periodic_lattice",
    "translations_within_cutoff",
    "unwrap_component_graph",
]
