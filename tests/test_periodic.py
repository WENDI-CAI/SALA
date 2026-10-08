"""Closest periodic image search checked against an independent small grid."""

import torch

from sala.models.crystal_dit.pair import PeriodicDisplacement


def _case():
    # The oblique basis makes simple component-wise fractional wrapping wrong.
    lattice = torch.tensor([[[5.0, 0.0, 0.0], [4.2, 1.7, 0.0], [0.6, 0.4, 4.1]]])
    frac = torch.tensor([[[0.10, 0.12, 0.13], [0.59, 0.60, 0.19],
                          [0.26, 0.74, 0.41], [2.3, -1.7, 0.2]]])
    valid = torch.tensor([[True, True, True, False]])
    return frac, lattice, valid


def test_skewed_minimum_image_matches_brute_force_and_not_naive_wrapping():
    frac, lattice, valid = _case()
    search = PeriodicDisplacement(pair_chunk_size=3, image_chunk_size=11)
    best_frac, best_cart = search(frac, lattice, valid)
    delta = frac[:, None, :, :] - frac[:, :, None, :]
    centered = delta - delta.round()
    grid = torch.cartesian_prod(*(torch.arange(-4, 5) for _ in range(3))).float()
    candidates = centered[..., None, :] + grid
    carts = torch.einsum("bijkd,bdl->bijkl", candidates, lattice)
    lengths2 = carts.square().sum(-1)
    minimum = lengths2.min(-1).values
    pair_valid = valid[:, :, None] & valid[:, None, :]
    torch.testing.assert_close(best_cart.square().sum(-1)[pair_valid], minimum[pair_valid])
    naive_cart = centered @ lattice
    assert best_cart[0, 0, 1].norm() < naive_cart[0, 0, 1].norm() - 1.0
    assert torch.count_nonzero(best_frac[~pair_valid]) == 0
    assert torch.count_nonzero(best_cart[~pair_valid]) == 0
    torch.testing.assert_close(best_cart, -best_cart.transpose(1, 2), rtol=1e-5, atol=1e-6)


def test_integer_image_shifts_preserve_displacements_and_gradients():
    frac, lattice, valid = _case()
    frac = frac.requires_grad_()
    lattice = lattice.requires_grad_()
    search = PeriodicDisplacement(pair_chunk_size=2, image_chunk_size=7)
    _, original = search(frac, lattice, valid)
    shifts = torch.tensor([[[2.0, -3.0, 1.0], [-1.0, 4.0, 2.0],
                            [3.0, 1.0, -2.0], [0.0, 0.0, 0.0]]])
    _, shifted = search(frac + shifts, lattice, valid)
    torch.testing.assert_close(original, shifted, rtol=2e-5, atol=3e-6)
    original.square().sum().backward()
    for gradient in [frac.grad, lattice.grad]:
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    assert torch.count_nonzero(frac.grad[~valid]) == 0
