"""Modern independently conditioned DiT blocks for full-cell crystals."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from sala.models.common import RMSNorm, ResidualFFN, mp_residual, zero_module
from sala.models.crystal_dit.attention import FullAttention


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale) + shift


class PairBiasLayerModulation(nn.Module):
    """Independent, magnitude-preserving head mixing for one DiT layer."""

    def __init__(self, *, num_heads: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.eps = float(eps)
        self.mixing_delta = nn.Parameter(
            torch.zeros(self.num_heads, self.num_heads)
        )
        self.log_scale = nn.Parameter(torch.zeros(self.num_heads))
        self.register_buffer(
            "identity",
            torch.eye(self.num_heads),
            persistent=False,
        )

    def forward(self, attention_bias: Tensor | None) -> Tensor | None:
        if attention_bias is None:
            return None
        if (
            attention_bias.ndim != 4
            or attention_bias.shape[1] != self.num_heads
            or attention_bias.shape[-2] != attention_bias.shape[-1]
        ):
            raise ValueError(
                "attention bias must have shape [B, H, L, L] with "
                f"H={self.num_heads}"
            )
        mixing = self.identity.to(
            device=self.mixing_delta.device,
            dtype=self.mixing_delta.dtype,
        ) + self.mixing_delta
        mixing = mixing * torch.rsqrt(
            mixing.float().square().sum(dim=-1, keepdim=True) + self.eps
        ).to(dtype=mixing.dtype)
        mixed = torch.einsum(
            "oh,bhij->boij",
            mixing.to(dtype=attention_bias.dtype),
            attention_bias,
        )
        scale = self.log_scale.clamp(-2.0, 2.0).exp().to(dtype=mixed.dtype)
        return mixed * scale[None, :, None, None]


class CrystalDiTBlock(nn.Module):
    """Affine-free RMS pre-norm with per-block AdaLN-Zero controls."""

    def __init__(
        self,
        *,
        d_model: int,
        d_cond: int,
        num_heads: int,
        d_ffn: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_flash: bool = True,
        qk_norm: bool = True,
        attention_backend: str = "auto",
        pair_bias_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(
            d_model, elementwise_affine=False, eps=1.0e-6
        )
        self.norm_ffn = RMSNorm(
            d_model, elementwise_affine=False, eps=1.0e-6
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), zero_module(nn.Linear(d_cond, 6 * d_model))
        )
        self.pair_bias_modulation = (
            PairBiasLayerModulation(num_heads=num_heads)
            if pair_bias_enabled
            else None
        )
        self.attention = FullAttention(
            d_model,
            num_heads,
            dropout=attn_dropout,
            use_flash=use_flash,
            qk_norm=qk_norm,
            backend=attention_backend,
        )
        self.ffn = ResidualFFN(d_model, d_ffn, dropout=dropout)

    def forward(
        self,
        hidden: Tensor,
        *,
        condition: Tensor,
        token_valid: Tensor,
        attention_bias: Tensor | None,
    ) -> Tensor:
        modulation = self.modulation(condition)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.chunk(
            6, dim=-1
        )
        attn_input = _modulate(
            self.norm_attn(hidden), shift_a[:, None], scale_a[:, None]
        )
        layer_attention_bias = (
            self.pair_bias_modulation(attention_bias)
            if self.pair_bias_modulation is not None
            else None
        )
        attn_branch = self.attention(
            attn_input,
            query_valid=token_valid,
            key_valid=token_valid,
            attention_bias=layer_attention_bias,
        )
        hidden = mp_residual(hidden, attn_branch, gate_a[:, None])
        ffn_input = _modulate(
            self.norm_ffn(hidden), shift_f[:, None], scale_f[:, None]
        )
        hidden = mp_residual(hidden, self.ffn(ffn_input), gate_f[:, None])
        return hidden * token_valid.unsqueeze(-1).to(dtype=hidden.dtype)


class FinalAdaNorm(nn.Module):
    def __init__(self, d_model: int, d_cond: int) -> None:
        super().__init__()
        self.norm = RMSNorm(
            d_model, elementwise_affine=False, eps=1.0e-6
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), zero_module(nn.Linear(d_cond, 2 * d_model))
        )

    def forward(self, hidden: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return _modulate(self.norm(hidden), shift[:, None], scale[:, None])


__all__ = ["CrystalDiTBlock", "FinalAdaNorm", "PairBiasLayerModulation"]
