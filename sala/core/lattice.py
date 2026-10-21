"""Lattice geometry utilities for crystal coordinate transforms and parameterizations."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from torch import Tensor

from sala.core.constants import EPS

MatrixLike = Union[np.ndarray, Tensor]

def _is_torch(x) -> bool:
    return torch.is_tensor(x)


def lattice_params_to_matrix(lengths, angles):
    if _is_torch(lengths):
        return _lattice_params_to_matrix_torch(lengths, angles)
    return _lattice_params_to_matrix_np(np.asarray(lengths), np.asarray(angles))


def _lattice_params_to_matrix_np(lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:
    lengths = lengths.reshape(-1, 3)
    angles = angles.reshape(-1, 3)
    ang = np.deg2rad(angles)
    a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
    alpha, beta, gamma = ang[:, 0], ang[:, 1], ang[:, 2]

    cos_alpha, cos_beta, cos_gamma = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_gamma = np.sin(gamma)

    v_x = a
    v_y = b * cos_gamma
    v_z = c * cos_beta

    w_y = b * sin_gamma
    w_z = c * (cos_alpha - cos_beta * cos_gamma) / np.maximum(sin_gamma, EPS)

    u_z_sq = np.maximum(c**2 - v_z**2 - w_z**2, 0.0)
    u_z = np.sqrt(u_z_sq)

    mats = np.stack(
        [
            np.stack([v_x, np.zeros_like(v_x), np.zeros_like(v_x)], axis=-1),
            np.stack([v_y, w_y, np.zeros_like(w_y)], axis=-1),
            np.stack([v_z, w_z, u_z], axis=-1),
        ],
        axis=-2,
    )
    return mats[0] if mats.shape[0] == 1 else mats


def _lattice_params_to_matrix_torch(lengths: Tensor, angles: Tensor) -> Tensor:
    lengths = lengths.reshape(-1, 3)
    angles = angles.reshape(-1, 3)
    ang = torch.deg2rad(angles)
    a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
    alpha, beta, gamma = ang[:, 0], ang[:, 1], ang[:, 2]

    cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    v_x = a
    v_y = b * cos_gamma
    v_z = c * cos_beta

    w_y = b * sin_gamma
    w_z = c * (cos_alpha - cos_beta * cos_gamma) / torch.clamp(sin_gamma, min=EPS)

    u_z_sq = torch.clamp(c**2 - v_z**2 - w_z**2, min=0.0)
    u_z = torch.sqrt(u_z_sq)

    mats = torch.stack(
        [
            torch.stack([v_x, torch.zeros_like(v_x), torch.zeros_like(v_x)], dim=-1),
            torch.stack([v_y, w_y, torch.zeros_like(w_y)], dim=-1),
            torch.stack([v_z, w_z, u_z], dim=-1),
        ],
        dim=-2,
    )
    return mats[0] if mats.shape[0] == 1 else mats


def lattice_matrix_to_params(lattice):
    if _is_torch(lattice):
        return _lattice_matrix_to_params_torch(lattice)
    return _lattice_matrix_to_params_np(np.asarray(lattice))


def _lattice_matrix_to_params_np(lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lattice = np.asarray(lattice, dtype=np.float64)
    if lattice.ndim == 2:
        lattice = lattice[None, ...]
    a = np.linalg.norm(lattice[:, 0], axis=-1)
    b = np.linalg.norm(lattice[:, 1], axis=-1)
    c = np.linalg.norm(lattice[:, 2], axis=-1)

    def angle(u, v):
        cosang = np.sum(u * v, axis=-1) / (
            np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + EPS
        )
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.rad2deg(np.arccos(cosang))

    alpha = angle(lattice[:, 1], lattice[:, 2])
    beta = angle(lattice[:, 0], lattice[:, 2])
    gamma = angle(lattice[:, 0], lattice[:, 1])

    lengths = np.stack([a, b, c], axis=-1).astype(np.float32)
    angles = np.stack([alpha, beta, gamma], axis=-1).astype(np.float32)
    return (lengths[0], angles[0]) if lengths.shape[0] == 1 else (lengths, angles)


def _lattice_matrix_to_params_torch(lattice: Tensor) -> tuple[Tensor, Tensor]:
    lattice = lattice.to(torch.float64)
    if lattice.ndim == 2:
        lattice = lattice.unsqueeze(0)
    a = torch.linalg.norm(lattice[:, 0], dim=-1)
    b = torch.linalg.norm(lattice[:, 1], dim=-1)
    c = torch.linalg.norm(lattice[:, 2], dim=-1)

    def angle(u, v):
        cosang = torch.sum(u * v, dim=-1) / (
            torch.linalg.norm(u, dim=-1) * torch.linalg.norm(v, dim=-1) + EPS
        )
        cosang = torch.clamp(cosang, -1.0, 1.0)
        return torch.rad2deg(torch.acos(cosang))

    alpha = angle(lattice[:, 1], lattice[:, 2])
    beta = angle(lattice[:, 0], lattice[:, 2])
    gamma = angle(lattice[:, 0], lattice[:, 1])

    lengths = torch.stack([a, b, c], dim=-1).to(torch.float32)
    angles = torch.stack([alpha, beta, gamma], dim=-1).to(torch.float32)
    return (lengths[0], angles[0]) if lengths.shape[0] == 1 else (lengths, angles)


def frac_to_cart_coords(frac_coords, lattice):
    if _is_torch(frac_coords):
        return frac_coords @ lattice
    return np.asarray(frac_coords) @ np.asarray(lattice)


def cart_to_frac_coords(cart_coords, lattice):
    if _is_torch(cart_coords):
        rhs = cart_coords.to(torch.float64).transpose(-1, -2)
        a = lattice.to(torch.float64).transpose(-1, -2)
        return torch.linalg.solve(a, rhs).transpose(-1, -2).to(dtype=cart_coords.dtype)
    cart_arr = np.asarray(cart_coords)
    out_dtype = cart_arr.dtype if np.issubdtype(cart_arr.dtype, np.floating) else np.float32
    cart = cart_arr.astype(np.float64, copy=False)
    lat = np.asarray(lattice, dtype=np.float64)
    frac = np.linalg.solve(lat.T, cart.T).T
    return frac.astype(out_dtype, copy=False)


def frac_to_cart_coords_batched(
    frac_coords: Tensor,
    lattice: Tensor,
    batch_idx: Tensor,
) -> Tensor:
    l_atom = lattice[batch_idx]
    return torch.bmm(frac_coords.unsqueeze(1), l_atom).squeeze(1)


def cart_to_frac_coords_batched(cart_coords: Tensor, lattice: Tensor, batch_idx: Tensor) -> Tensor:
    l_atom = lattice[batch_idx]
    rhs = cart_coords.to(torch.float64).unsqueeze(-1)
    a = l_atom.to(torch.float64).transpose(-1, -2)
    return torch.linalg.solve(a, rhs).squeeze(-1).to(dtype=cart_coords.dtype)


def min_plane_distances(lattice) -> tuple[float, float, float]:
    lattice = np.asarray(lattice, dtype=np.float64)
    a, b, c = lattice[0], lattice[1], lattice[2]
    volume = float(abs(np.dot(a, np.cross(b, c))))
    if volume < EPS:
        return 1.0, 1.0, 1.0
    bxc = np.cross(b, c)
    cxa = np.cross(c, a)
    axb = np.cross(a, b)
    d_a = 1.0 / (np.linalg.norm(bxc / volume) + EPS)
    d_b = 1.0 / (np.linalg.norm(cxa / volume) + EPS)
    d_c = 1.0 / (np.linalg.norm(axb / volume) + EPS)
    return float(d_a), float(d_b), float(d_c)


def batched_inverse_3x3(matrix: Tensor) -> Tensor:
    """Analytic inverse of stacked 3x3 matrices that does not block the CUDA queue.

    Returns ``adj(M) / det(M)``.  Callers must supply non-singular matrices; every
    lattice from the constrained-q decoder has strictly positive volume, so no
    singularity guard is applied here.
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


params_to_matrix = lattice_params_to_matrix
matrix_to_params = lattice_matrix_to_params


__all__ = [
    "MatrixLike",
    "batched_inverse_3x3",
    "cart_to_frac_coords",
    "cart_to_frac_coords_batched",
    "frac_to_cart_coords",
    "frac_to_cart_coords_batched",
    "lattice_matrix_to_params",
    "lattice_params_to_matrix",
    "matrix_to_params",
    "min_plane_distances",
    "params_to_matrix",
]
