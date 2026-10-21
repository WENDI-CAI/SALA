from __future__ import annotations

import math

import torch
from torch import Tensor

from sala.core.types import AsuBatch
from sala.flow_matching.lattice_prior import LatticePriorBatchStats, VolumeLatticePrior
from sala.flow_matching.types import FlowPathSample


def flow_time_condition(time: Tensor, *, scale: float = 1000.0) -> Tensor:
    """Return the bounded scaled-time conditioner used by the Crystal-DiT."""

    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("time condition scale must be finite and positive")
    value = time.float()
    predicate = torch.isfinite(value).all() & (value >= 0.0).all() & (value <= 1.0).all()
    if predicate.device.type == "cpu" and not bool(predicate.item()):
        raise ValueError("flow time must be finite and in [0, 1]")
    return value * scale


def sample_logit_normal_times(
    batch_size: int,
    *,
    mean: float,
    std: float,
    clamp: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Sample the locked logit-normal rectified-flow training distribution."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    mean = float(mean)
    std = float(std)
    if not math.isfinite(mean):
        raise ValueError("time mean must be finite")
    if not math.isfinite(std) or std < 0.0:
        raise ValueError("time std must be finite and non-negative")
    eps = float(clamp)
    if not math.isfinite(eps) or not 0.0 < eps < 0.5:
        raise ValueError("time clamp must be finite and in (0, 0.5)")
    logits = torch.randn(int(batch_size), device=device, dtype=torch.float32)
    logits = logits * std + mean
    return torch.sigmoid(logits).clamp(eps, 1.0 - eps).to(dtype=dtype)


class RectifiedFlowPath:
    """Linear data-at-zero, noise-at-one path on normalized Cartesian/q state."""

    def __init__(
        self,
        *,
        coord_scale: float,
        lattice_prior: VolumeLatticePrior,
    ) -> None:
        coord_scale = float(coord_scale)
        if not math.isfinite(coord_scale) or coord_scale <= 0.0:
            raise ValueError("coord_scale must be finite and positive")
        self.coord_scale = coord_scale
        self.lattice_prior = lattice_prior

    def sample(
        self,
        *,
        batch: AsuBatch,
        clean_x: Tensor,
        clean_q: Tensor,
        time: Tensor,
        lattice_stats: LatticePriorBatchStats,
        noise_x: Tensor | None = None,
        noise_q: Tensor | None = None,
    ) -> FlowPathSample:
        if time.shape != (batch.batch_size,):
            raise ValueError("flow time must have shape [B]")
        clean_state_x = clean_x / self.coord_scale
        clean_state_q = self.lattice_prior.whiten_batch(
            clean_q, batch, lattice_stats
        )
        mask = lattice_stats.mask.to(
            device=clean_state_q.device, dtype=clean_state_q.dtype
        )
        clean_state_q = clean_state_q * mask
        noise_x = torch.randn_like(clean_state_x) if noise_x is None else noise_x
        noise_q = torch.randn_like(clean_state_q) if noise_q is None else noise_q
        if noise_x.shape != clean_state_x.shape:
            raise ValueError("coordinate base noise has an invalid shape")
        if noise_q.shape != clean_state_q.shape:
            raise ValueError("lattice base noise has an invalid shape")
        noise_q = noise_q * mask
        atom_time = time.index_select(0, batch.batch.long()).unsqueeze(-1)
        state_x = (1.0 - atom_time) * clean_state_x + atom_time * noise_x
        graph_time = time.unsqueeze(-1)
        state_q = (
            (1.0 - graph_time) * clean_state_q + graph_time * noise_q
        ) * mask
        return FlowPathSample(
            time=time,
            state_x=state_x,
            state_q=state_q,
            clean_x=clean_x,
            clean_q=clean_q,
            clean_state_x=clean_state_x,
            clean_state_q=clean_state_q,
            noise_x=noise_x,
            noise_q=noise_q,
            target_velocity_x=noise_x - clean_state_x,
            target_velocity_q=(noise_q - clean_state_q) * mask,
        )


__all__ = ["RectifiedFlowPath", "flow_time_condition", "sample_logit_normal_times"]
