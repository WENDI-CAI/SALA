"""Modern dense attention for full-cell Crystal-DiT."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sala.models.common import RMSNorm

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover
    SDPBackend = None
    sdpa_kernel = None

try:
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:  # pragma: no cover
    flex_attention = None

_COMPILED_FLEX_ATTENTION = None


def _flex_kernel(device: torch.device):
    global _COMPILED_FLEX_ATTENTION
    if flex_attention is None:
        return None
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return flex_attention
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(
            flex_attention, dynamic=False
        )
    return _COMPILED_FLEX_ATTENTION


class FullAttention(nn.Module):
    """Fused-QKV attention with per-head QK-Norm and full token-pair bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True,
        *,
        qk_norm: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = max(int(num_heads), 1)
        self.head_dim = self.dim // self.num_heads
        if self.head_dim * self.num_heads != self.dim:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.qkv_proj = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.qkv_proj._sala_muon = True
        # CMuon treats Q, K and V as functionally distinct row blocks.
        self.qkv_proj._sala_muon_chunk_sizes = (
            self.dim,
            self.dim,
            self.dim,
        )
        self.out_proj._sala_muon = True
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.logit_temperature = nn.Parameter(torch.zeros(self.num_heads))
        self.dropout = float(dropout)
        self.use_flash = bool(use_flash)
        self.backend = str(backend).strip().lower()
        if self.backend not in {"auto", "flex", "sdpa", "math"}:
            raise ValueError("attention backend must be auto, flex, sdpa, or math")
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

    def _qkv(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, length, _ = hidden.shape
        qkv = self.qkv_proj(hidden).view(
            batch_size, length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        temperature = self.logit_temperature.clamp(
            -math.log(4.0), math.log(4.0)
        ).exp()
        q = q * temperature[None, :, None, None].to(dtype=q.dtype)
        return q, k, v

    def _use_flex(self, hidden: Tensor) -> bool:
        if self.backend == "flex":
            return flex_attention is not None and self.dropout == 0.0
        if self.backend != "auto":
            return False
        return (
            flex_attention is not None
            and hidden.device.type == "cuda"
            and self.dropout == 0.0
        )

    def _flex_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        key_valid: Tensor,
        attention_bias: Tensor | None,
    ) -> Tensor:
        def score_mod(score, batch, head, q_idx, k_idx):
            key_ok = key_valid[batch, k_idx]
            modified = score
            if attention_bias is not None:
                modified = modified + attention_bias[batch, head, q_idx, k_idx]
            return torch.where(
                key_ok, modified, torch.full_like(modified, float("-inf"))
            )

        kernel = _flex_kernel(q.device)
        if kernel is None:  # pragma: no cover
            raise RuntimeError("FlexAttention is unavailable")
        return kernel(q, k, v, score_mod=score_mod, scale=self.scale)

    def _sdpa_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        key_valid: Tensor,
        attention_bias: Tensor | None,
    ) -> Tensor:
        length = int(q.shape[-2])
        key_mask = key_valid[:, None, None, :]
        if attention_bias is None:
            attn_mask: Tensor = key_mask
        else:
            expected = (
                q.shape[0],
                self.num_heads,
                length,
                length,
            )
            if tuple(attention_bias.shape) != expected:
                raise ValueError(
                    "attention bias must have shape "
                    f"{expected}, got {tuple(attention_bias.shape)}"
                )
            attn_mask = attention_bias.to(dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~key_mask, float("-inf"))

        force_math = self.backend == "math" or not self.use_flash
        if force_math and sdpa_kernel is not None and SDPBackend is not None:
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    scale=self.scale,
                )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        query_valid: Tensor,
        key_valid: Tensor,
        attention_bias: Tensor | None = None,
    ) -> Tensor:
        batch_size, length, _ = hidden.shape
        if length == 0:
            return hidden.new_zeros((batch_size, 0, self.dim))
        q, k, v = self._qkv(hidden)

        missing = ~key_valid.any(dim=-1)
        safe_key_valid = key_valid.clone()
        safe_key_valid[:, 0] |= missing
        missing_heads = missing[:, None, None, None]
        k = k.masked_fill(missing_heads, 0.0)
        v = v.masked_fill(missing_heads, 0.0)

        if self._use_flex(hidden):
            out = self._flex_forward(
                q,
                k,
                v,
                key_valid=safe_key_valid,
                attention_bias=attention_bias,
            )
        else:
            out = self._sdpa_forward(
                q,
                k,
                v,
                key_valid=safe_key_valid,
                attention_bias=attention_bias,
            )
        out = out.transpose(1, 2).reshape(batch_size, length, self.dim)
        out = self.out_proj(out)
        return out * query_valid.unsqueeze(-1).to(dtype=out.dtype)


__all__ = ["FullAttention"]
