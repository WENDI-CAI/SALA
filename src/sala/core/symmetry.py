"""Validate explicitly supplied fractional affine symmetry operations."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _validate_affine_operations(operations: Tensor, *, name: str) -> Tensor:
    if operations.ndim != 3 or operations.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [G, 4, 4], got {tuple(operations.shape)}")
    if operations.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one operation")
    if not bool(torch.isfinite(operations).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    expected_bottom = operations.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(
        operations.shape[0], -1
    )
    if not bool(torch.allclose(operations[:, 3], expected_bottom, atol=1e-6, rtol=0.0)):
        raise ValueError(f"{name} must contain homogeneous affine matrices")
    return operations

def canonicalize_affine_symmetry_ops(operations: Tensor) -> Tensor:
    """Canonicalize fractional translations to [0, 1) without changing R.

    CIF expressions may encode the same operation with translations differing
    by arbitrary lattice integers.  Full-cell Cartesian tokens must not depend
    on that textual choice, so every runtime operation crosses this boundary.
    """

    value = _validate_affine_operations(
        torch.as_tensor(operations), name="symmetry_ops"
    ).clone()
    if not value.is_floating_point():
        value = value.to(dtype=torch.float32)
    translation = torch.remainder(value[:, :3, 3], 1.0)
    tolerance = 8.0 * torch.finfo(translation.dtype).eps
    translation = torch.where(
        torch.isclose(translation, torch.ones_like(translation), atol=tolerance, rtol=0.0)
        | torch.isclose(translation, torch.zeros_like(translation), atol=tolerance, rtol=0.0),
        torch.zeros_like(translation),
        translation,
    )
    value[:, :3, 3] = translation
    return value

def extract_explicit_symmetry_ops(
    structure: Any,
    graph_idx: int | None = None,
) -> Tensor | None:
    """Return one graph's unpadded source-CIF affine operations.

    ``AsuInput`` stores ``[G, 4, 4]`` directly. Canonical ``AsuBatch`` stores
    padded ``[B, G_max, 4, 4]`` together with an authoritative
    ``[B, G_max]`` mask.
    """

    stored = getattr(structure, "symmetry_ops", None)
    if stored is None:
        return None
    operations = torch.as_tensor(stored)
    if operations.numel() == 0:
        return None

    if operations.ndim == 3:
        if hasattr(structure, "ptr"):
            raise ValueError(
                "AsuBatch symmetry_ops must be padded [B, G_max, 4, 4] with a mask"
            )
        return canonicalize_affine_symmetry_ops(operations)

    if operations.ndim != 4 or operations.shape[2:] != (4, 4):
        raise ValueError(
            "batched symmetry_ops must have shape [B, G_max, 4, 4], "
            f"got {tuple(operations.shape)}"
        )
    if graph_idx is None:
        if operations.shape[0] != 1:
            raise ValueError("graph_idx is required for a multi-graph symmetry batch")
        graph_idx = 0
    graph_idx = int(graph_idx)
    if graph_idx < 0 or graph_idx >= operations.shape[0]:
        raise IndexError(f"graph_idx {graph_idx} is outside [0, {operations.shape[0]})")

    stored_mask = getattr(structure, "symmetry_mask", None)
    if stored_mask is None:
        raise ValueError("batched symmetry_ops require an authoritative symmetry_mask")
    mask = torch.as_tensor(stored_mask, device=operations.device)
    if mask.shape != operations.shape[:2]:
        raise ValueError(
            "symmetry_mask must have shape [B, G_max] matching symmetry_ops, "
            f"got {tuple(mask.shape)} versus {tuple(operations.shape[:2])}"
        )
    active = operations[graph_idx, mask[graph_idx].bool()]
    if active.shape[0] == 0:
        return None
    return canonicalize_affine_symmetry_ops(active)


__all__ = ["canonicalize_affine_symmetry_ops", "extract_explicit_symmetry_ops"]
