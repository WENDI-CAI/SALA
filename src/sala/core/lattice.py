"""Small tensor operations used by SALA's periodic geometry."""

from __future__ import annotations

import torch
from torch import Tensor


def batched_inverse_3x3(matrix: Tensor) -> Tensor:
    """Return the analytic inverse of non-singular tensors ending in [3, 3].

    Uses adjugate / determinant without a device-to-host synchronization.
    Callers are responsible for supplying non-singular matrices.
    """

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"batched_inverse_3x3 expects trailing shape (3, 3), got "
            f"{tuple(matrix.shape)}"
        )
    a, b, c = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    d, e, f = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    g, h, i = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    # Cofactors, already transposed into adjugate layout.
    c00 = e * i - f * h
    c01 = -(b * i - c * h)
    c02 = b * f - c * e
    c10 = -(d * i - f * g)
    c11 = a * i - c * g
    c12 = -(a * f - c * d)
    c20 = d * h - e * g
    c21 = -(a * h - b * g)
    c22 = a * e - b * d
    determinant = a * c00 + b * c10 + c * c20
    adjugate = torch.stack(
        (
            torch.stack((c00, c01, c02), dim=-1),
            torch.stack((c10, c11, c12), dim=-1),
            torch.stack((c20, c21, c22), dim=-1),
        ),
        dim=-2,
    )
    return adjugate / determinant[..., None, None]


__all__ = ["batched_inverse_3x3"]
