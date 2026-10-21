from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor

from sala.core.lattice_manifold import (
    CrystalQFamily,
    q_active_mask,
    q_to_lattice,
    require_lattice_template,
)
from sala.core.types import AsuBatch


VOLUME_MEAN_COEFFICIENT = 17.784423
LENGTH_LOG_STD = 0.30
ANGLE_STD = 0.20


def _require_all(predicate: Tensor, message: str) -> None:
    """Validate tensors without forcing a CUDA host synchronization."""
    predicate = predicate.reshape(())
    if predicate.device.type == "cpu":
        if not bool(predicate.item()):
            raise ValueError(message)
        return
    assert_async = getattr(torch, "_assert_async", None)
    if assert_async is not None:
        assert_async(predicate, message)


@dataclass(frozen=True)
class LatticePriorBatchStats:
    mean: Tensor
    std: Tensor
    mask: Tensor
    bias: Tensor


class VolumeLatticePrior(nn.Module):
    """Analytic constrained-q prior centered on ``17.784423 * Ncell`` volume."""

    def __init__(
        self,
        *,
        length_log_std: float = LENGTH_LOG_STD,
        angle_std: float = ANGLE_STD,
    ) -> None:
        super().__init__()
        self.length_log_std = float(length_log_std)
        self.angle_std = float(angle_std)
        if self.length_log_std <= 0.0 or self.angle_std <= 0.0:
            raise ValueError("lattice prior standard deviations must be positive")
        self.crystal_family = CrystalQFamily()

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "VolumeLatticePrior":
        cfg = dict(config or {})
        unsupported = sorted(set(cfg) - {"type", "length_log_std", "angle_std"})
        if unsupported:
            raise ValueError(f"Unsupported volume lattice-prior config keys: {unsupported}")
        prior_type = str(cfg.get("type", "volume")).strip().lower()
        if prior_type != "volume":
            raise ValueError("flow.prior.lattice.type must be 'volume'")
        return cls(
            length_log_std=float(cfg.get("length_log_std", LENGTH_LOG_STD)),
            angle_std=float(cfg.get("angle_std", ANGLE_STD)),
        )

    @staticmethod
    def _validate_inputs(
        spacegroup: Tensor,
        n_heavy_cell: Tensor,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        sg_raw = spacegroup.to(device=device).view(-1)
        sg_float = sg_raw.float()
        _require_all(
            (
                torch.isfinite(sg_float)
                & (sg_float == sg_float.round())
                & (sg_float >= 1)
                & (sg_float <= 230)
            ).all(),
            "spacegroup must contain integers in [1, 230]",
        )
        sg = sg_raw.to(dtype=torch.long)
        ncell = n_heavy_cell.to(device=device, dtype=torch.float32).view(-1)
        if sg.shape != ncell.shape:
            raise ValueError("spacegroup and n_heavy_cell must both have shape (B,)")
        _require_all(
            (torch.isfinite(ncell) & (ncell > 0)).all(),
            "n_heavy_cell must contain positive finite values",
        )
        return sg, ncell

    def stats(
        self,
        *,
        spacegroup: Tensor,
        n_heavy_cell: Tensor,
        lattice_template_id: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> LatticePriorBatchStats:
        target_device = device or spacegroup.device
        sg, ncell = self._validate_inputs(
            spacegroup, n_heavy_cell, device=target_device
        )
        template = None
        if lattice_template_id is not None:
            template = lattice_template_id.to(
                device=target_device, dtype=torch.long
            ).view(-1)
            if template.shape != sg.shape:
                raise ValueError("lattice_template_id must have shape (B,)")

        family = self.crystal_family.to(target_device)
        mask = (
            family.masks.to(device=target_device, dtype=torch.float32)[sg]
            if template is None
            else q_active_mask(
                spacegroup=sg,
                lattice_template_id=template,
                dtype=torch.float32,
            )
        )
        bias = family.biass.to(device=target_device, dtype=torch.float32)[sg]

        # q=bias gives the unit-scale reference cell for every Hall-derived
        # template. Uniformly shifting each active log-length slot by log(s)
        # scales its volume by s^3 without changing shape or crystal system.
        reference_q = bias.clone()
        reference_lattice = q_to_lattice(
            reference_q,
            spacegroup=sg,
            lattice_template_id=template,
        )
        reference_volume = torch.linalg.det(reference_lattice.float()).abs()
        _require_all(
            (torch.isfinite(reference_volume) & (reference_volume > 0)).all(),
            "constrained-q reference lattice must have positive finite volume",
        )
        target_volume = ncell * VOLUME_MEAN_COEFFICIENT
        log_scale = (torch.log(target_volume) - torch.log(reference_volume)) / 3.0
        active_lengths = mask[:, :3]
        mean = bias.clone()
        mean[:, :3] = mean[:, :3] + active_lengths * log_scale.unsqueeze(-1)
        mean = mean * mask + bias

        slot_std = torch.full_like(mean, self.angle_std)
        slot_std[:, :3] = self.length_log_std
        std = torch.where(mask > 0.5, slot_std, torch.ones_like(slot_std))
        return LatticePriorBatchStats(
            mean=mean.to(dtype=dtype),
            std=std.to(dtype=dtype),
            mask=mask.to(dtype=dtype),
            bias=bias.to(dtype=dtype),
        )

    def stats_from_batch(
        self, batch: AsuBatch, *, device: torch.device, dtype: torch.dtype
    ) -> LatticePriorBatchStats:
        n_heavy_cell = getattr(batch, "n_heavy_cell", None)
        if n_heavy_cell is None:
            raise ValueError("Volume lattice prior requires batch.n_heavy_cell")
        template = require_lattice_template(
            batch.spacegroup.to(device),
            None
            if batch.lattice_template_id is None
            else batch.lattice_template_id.to(device),
        )
        return self.stats(
            spacegroup=batch.spacegroup,
            n_heavy_cell=n_heavy_cell,
            lattice_template_id=template,
            device=device,
            dtype=dtype,
        )

    def projected_noise_mask(
        self,
        *,
        spacegroup: Tensor,
        lattice_template_id: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        target_device = device or spacegroup.device
        sg, _ = self._validate_inputs(
            spacegroup,
            torch.ones_like(spacegroup, dtype=torch.float32),
            device=target_device,
        )
        if lattice_template_id is not None:
            return q_active_mask(
                spacegroup=sg,
                lattice_template_id=lattice_template_id.to(
                    device=target_device, dtype=torch.long
                ).view(-1),
                dtype=dtype,
            )
        return self.crystal_family.masks.to(device=target_device, dtype=dtype)[sg]

    @staticmethod
    def whiten_with_stats(k: Tensor, stats: LatticePriorBatchStats) -> Tensor:
        value = k.to(dtype=torch.float32)
        if value.shape != stats.mean.shape:
            raise ValueError(
                f"q must have shape {tuple(stats.mean.shape)}, got {tuple(value.shape)}"
            )
        mask = stats.mask.to(device=k.device, dtype=torch.float32)
        bias = stats.bias.to(device=k.device, dtype=torch.float32)
        projected = value * mask + bias
        return (((projected - stats.mean) / stats.std) * mask).to(dtype=k.dtype)

    @staticmethod
    def unwhiten_with_stats(z: Tensor, stats: LatticePriorBatchStats) -> Tensor:
        latent = z.to(dtype=torch.float32)
        if latent.shape != stats.mean.shape:
            raise ValueError(
                f"z must have shape {tuple(stats.mean.shape)}, got {tuple(latent.shape)}"
            )
        q = stats.mean + latent * stats.std * stats.mask
        return (q * stats.mask + stats.bias).to(dtype=z.dtype)

    def whiten(
        self,
        q: Tensor,
        *,
        spacegroup: Tensor,
        n_heavy_cell: Tensor,
        lattice_template_id: Optional[Tensor] = None,
    ) -> Tensor:
        stats = self.stats(
            spacegroup=spacegroup,
            n_heavy_cell=n_heavy_cell,
            lattice_template_id=lattice_template_id,
            device=q.device,
            dtype=torch.float32,
        )
        return self.whiten_with_stats(q, stats)

    def unwhiten(
        self,
        z: Tensor,
        *,
        spacegroup: Tensor,
        n_heavy_cell: Tensor,
        lattice_template_id: Optional[Tensor] = None,
    ) -> Tensor:
        stats = self.stats(
            spacegroup=spacegroup,
            n_heavy_cell=n_heavy_cell,
            lattice_template_id=lattice_template_id,
            device=z.device,
            dtype=torch.float32,
        )
        return self.unwhiten_with_stats(z, stats)

    def whiten_batch(
        self,
        q: Tensor,
        batch: AsuBatch,
        lattice_stats: Optional[LatticePriorBatchStats] = None,
    ) -> Tensor:
        stats = lattice_stats or self.stats_from_batch(
            batch, device=q.device, dtype=torch.float32
        )
        return self.whiten_with_stats(q, stats)

    def unwhiten_batch(
        self,
        z: Tensor,
        batch: AsuBatch,
        lattice_stats: Optional[LatticePriorBatchStats] = None,
    ) -> Tensor:
        stats = lattice_stats or self.stats_from_batch(
            batch, device=z.device, dtype=torch.float32
        )
        return self.unwhiten_with_stats(z, stats)

    def sample(
        self,
        *,
        spacegroup: Tensor,
        n_heavy_cell: Tensor,
        lattice_template_id: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        target_device = device or spacegroup.device
        stats = self.stats(
            spacegroup=spacegroup.to(target_device),
            n_heavy_cell=n_heavy_cell.to(target_device),
            lattice_template_id=(
                None
                if lattice_template_id is None
                else lattice_template_id.to(target_device)
            ),
            device=target_device,
            dtype=dtype,
        )
        z = torch.randn(
            stats.mask.shape,
            device=target_device,
            dtype=dtype,
            generator=generator,
        ) * stats.mask
        return self.unwhiten_with_stats(z, stats)


__all__ = [
    "ANGLE_STD",
    "LENGTH_LOG_STD",
    "LatticePriorBatchStats",
    "VOLUME_MEAN_COEFFICIENT",
    "VolumeLatticePrior",
]
