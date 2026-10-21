from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from sala.core.coordinates import make_fractional_coords_contiguous_torch
from sala.core.lattice import frac_to_cart_coords
from sala.core.types import AsuInput


def canonicalize_frac_coords(
    frac_coords: Tensor,
    edge_index_conv: Tensor,
    component_id: Tensor,
    lattice: Tensor,
) -> Tensor:
    continuous, _ = make_fractional_coords_contiguous_torch(
        frac_coords=frac_coords,
        edge_index_conv=edge_index_conv,
        component_id=component_id,
        lattice=lattice,
        force_cpu=False,
    )
    return continuous


def canonicalize_asu_input(asu: AsuInput) -> AsuInput:
    frac_coords = canonicalize_frac_coords(
        frac_coords=asu.frac_coords_gt,
        edge_index_conv=asu.edge_index_conv,
        component_id=asu.component_id,
        lattice=asu.lattice_gt,
    )
    cart_coords = frac_to_cart_coords(frac_coords, asu.lattice_gt)
    if torch.equal(frac_coords, asu.frac_coords_gt) and (
        asu.cart_coords_gt is not None
        and torch.allclose(
            cart_coords,
            asu.cart_coords_gt,
            atol=1.0e-5,
            rtol=1.0e-5,
        )
    ):
        return asu
    return dataclasses.replace(
        asu,
        frac_coords_gt=frac_coords,
        cart_coords_gt=cart_coords,
    )


__all__ = [
    "canonicalize_asu_input",
    "canonicalize_frac_coords",
]
