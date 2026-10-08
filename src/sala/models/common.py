from __future__ import annotations

import math
from typing import Type

import torch
import torch.nn as nn
from torch import Tensor


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
    dropout: float = 0.0,
    activation: Type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    layers = []
    current_dim = in_dim
    total_layers = max(int(num_layers), 1)
    for _layer_idx in range(total_layers - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)


def zero_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        nn.init.zeros_(param)
    return module


class RMSNorm(nn.Module):
    """RMSNorm with float32 reduction and optional affine weight."""

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1.0e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = (
            nn.Parameter(torch.ones(self.dim)) if elementwise_affine else None
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"RMSNorm expected trailing dimension {self.dim}, got {x.shape[-1]}"
            )
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        out = x * scale.to(dtype=x.dtype)
        if self.weight is not None:
            out = out * self.weight.to(device=x.device, dtype=x.dtype)
        return out


class MPFeatureMixer(nn.Module):
    """Magnitude-preserving, learnable fusion of same-width feature streams."""

    def __init__(self, num_inputs: int, dim: int, *, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.num_inputs = int(num_inputs)
        self.dim = int(dim)
        self.eps = float(eps)
        if self.num_inputs <= 0:
            raise ValueError("MPFeatureMixer requires at least one input")
        self.weights = nn.Parameter(torch.ones(self.num_inputs, self.dim))

    def forward(self, *features: Tensor) -> Tensor:
        if len(features) != self.num_inputs:
            raise ValueError(
                f"expected {self.num_inputs} feature streams, got {len(features)}"
            )
        reference = features[0]
        if any(feature.shape != reference.shape for feature in features):
            raise ValueError("all MPFeatureMixer inputs must have identical shapes")
        stacked = torch.stack(features, dim=-2)
        normalized = stacked * torch.rsqrt(
            stacked.float().square().mean(dim=-1, keepdim=True) + self.eps
        ).to(dtype=stacked.dtype)
        view_shape = (1,) * (normalized.ndim - 2) + self.weights.shape
        weights = self.weights.to(
            device=normalized.device, dtype=normalized.dtype
        ).view(view_shape)
        numerator = (normalized * weights).sum(dim=-2)
        denominator = torch.sqrt(
            weights.float().square().sum(dim=-2) + self.eps
        ).to(dtype=normalized.dtype)
        return numerator / denominator


def mp_residual(x: Tensor, branch: Tensor, gate: Tensor) -> Tensor:
    """Magnitude-preserving gated residual with identity at gate=0."""

    if x.shape != branch.shape:
        raise ValueError("residual stream and branch must have identical shapes")
    numerator = x + gate * branch
    denominator = torch.sqrt(1.0 + gate.float().square()).to(dtype=x.dtype)
    return numerator / denominator


class GaussianRBF(nn.Module):

    def __init__(
        self,
        num_rbf: int,
        cutoff: float = 12.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        num_rbf = max(int(num_rbf), 1)
        centers = torch.linspace(0.0, float(cutoff), num_rbf)
        if num_rbf > 1:
            spacing = float(cutoff) / float(num_rbf - 1)
            gamma = 1.0 / float(spacing * spacing + eps)
        else:
            gamma = 1.0
        self.register_buffer("centers", centers)
        self.gamma = float(gamma)

    def forward(self, distances: Tensor) -> Tensor:
        d = distances.unsqueeze(-1) - self.centers.to(device=distances.device, dtype=distances.dtype)
        return torch.exp(-self.gamma * d.square())


class TimeEmbedding(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_fourier_freqs: int = 16,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_fourier_freqs = max(int(num_fourier_freqs), 1)
        hidden_dim = int(hidden_dim or d_model)
        self.proj = build_mlp(
            in_dim=self.num_fourier_freqs * 2,
            hidden_dim=hidden_dim,
            out_dim=d_model,
            num_layers=2,
            dropout=dropout,
        )

    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 0:
            t = t.view(1)
        t = t.float().view(-1, 1)
        device = t.device
        freqs = torch.exp(
            torch.linspace(
                0.0,
                math.log(10000.0),
                self.num_fourier_freqs,
                device=device,
                dtype=t.dtype,
            )
            * (-1.0)
        )
        phases = t * freqs.view(1, -1)
        emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        return self.proj(emb)


class ResidualFFN(nn.Module):
    """Fused SwiGLU feed-forward block.

    ``hidden_dim`` is the actual width of each SwiGLU branch.  The old code
    treated it as a pre-gating width and silently multiplied by 2/3.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        dim = int(dim)
        hidden_dim = int(hidden_dim)
        inner_dim = max(1, hidden_dim)
        self.gate_up = nn.Linear(dim, 2 * inner_dim, bias=False)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.down_proj = nn.Linear(inner_dim, dim, bias=False)
        self.gate_up._sala_muon = True
        # CMuon orthogonalizes the SwiGLU gate and value projections
        # independently instead of coupling their momentum directions.
        self.gate_up._sala_muon_chunk_sizes = (inner_dim, inner_dim)
        self.down_proj._sala_muon = True

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        x_proj = torch.nn.functional.silu(gate) * value
        return self.down_proj(self.dropout(x_proj))


__all__ = [
    "GaussianRBF",
    "MPFeatureMixer",
    "RMSNorm",
    "ResidualFFN",
    "TimeEmbedding",
    "build_mlp",
    "mp_residual",
    "zero_module",
]
