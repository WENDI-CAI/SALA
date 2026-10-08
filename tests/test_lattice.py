"""Synthetic checks for constrained lattice geometry and its tensor inverse."""

import pytest
import torch

from sala.core.lattice import batched_inverse_3x3
from sala.core.lattice_manifold import (
    LatticeTemplate,
    lattice_metric,
    lattice_to_q,
    q_active_mask,
    q_to_lattice,
)


# IT numbers identify symmetry classes, not individual crystal structures.
TEMPLATE_CASES = [
    pytest.param(LatticeTemplate.TRICLINIC, 1, (0, 1, 2, 3, 4, 5), id="triclinic"),
    pytest.param(LatticeTemplate.MONOCLINIC_B, 3, (0, 1, 2, 4), id="monoclinic-b"),
    pytest.param(LatticeTemplate.ORTHORHOMBIC, 16, (0, 1, 2), id="orthorhombic"),
    pytest.param(LatticeTemplate.TETRAGONAL, 75, (0, 2), id="tetragonal"),
    pytest.param(LatticeTemplate.TRIGONAL_HEX, 143, (0, 2), id="trigonal-hex"),
    pytest.param(LatticeTemplate.HEXAGONAL, 168, (0, 2), id="hexagonal"),
    pytest.param(LatticeTemplate.CUBIC, 195, (0,), id="cubic"),
    pytest.param(LatticeTemplate.MONOCLINIC_A, 3, (0, 1, 2, 5), id="monoclinic-a"),
    pytest.param(LatticeTemplate.MONOCLINIC_C, 3, (0, 1, 2, 3), id="monoclinic-c"),
    pytest.param(LatticeTemplate.RHOMBOHEDRAL, 146, (0, 3), id="rhombohedral"),
]


def _metadata(template, spacegroup):
    return {
        "spacegroup": torch.tensor([spacegroup]),
        "lattice_template_id": torch.tensor([int(template)]),
    }


def _sample_q(dtype=torch.float64):
    # Nonzero angular coordinates expose derivatives that vanish at right angles.
    return torch.tensor([[1.1, 1.35, 1.6, 0.22, -0.31, 0.17]], dtype=dtype)


@pytest.mark.parametrize("template,spacegroup,active", TEMPLATE_CASES)
def test_template_roundtrip_and_rotated_metric(template, spacegroup, active):
    metadata = _metadata(template, spacegroup)
    q = _sample_q()
    mask = q_active_mask(**metadata, dtype=torch.bool)
    expected_mask = torch.zeros_like(q, dtype=torch.bool)
    expected_mask[:, list(active)] = True
    assert torch.equal(mask, expected_mask)
    assert int(mask.sum()) == len(active)

    lattice = q_to_lattice(q, **metadata)
    assert torch.equal(lattice, torch.tril(lattice))
    assert bool((torch.diagonal(lattice, dim1=-2, dim2=-1) > 0).all())
    assert bool((torch.linalg.det(lattice) > 0).all())

    recovered = lattice_to_q(lattice, **metadata)
    # Decoding evaluates in float32 even when the caller keeps float64 tensors.
    torch.testing.assert_close(recovered[mask], q[mask], rtol=2e-6, atol=2e-6)
    assert torch.count_nonzero(recovered[~mask]) == 0

    # A proper Cartesian rotation changes the basis matrix, but not its metric.
    rotation = torch.tensor(
        [[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]],
        dtype=q.dtype,
    )
    rotated = lattice @ rotation
    assert not torch.allclose(rotated, lattice)
    rotated_q = lattice_to_q(rotated, **metadata)
    canonical = q_to_lattice(rotated_q, **metadata)
    torch.testing.assert_close(rotated_q, recovered, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        lattice_metric(canonical), lattice_metric(rotated), rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("template,spacegroup,active", TEMPLATE_CASES)
def test_only_active_lattice_coordinates_affect_geometry_and_gradients(
    template, spacegroup, active
):
    metadata = _metadata(template, spacegroup)
    q = _sample_q().requires_grad_()
    mask = torch.zeros_like(q, dtype=torch.bool)
    mask[:, list(active)] = True
    lattice = q_to_lattice(q, **metadata)
    # An asymmetric linear observable also senses the three possible shears.
    weights = torch.tensor(
        [[1.0, 0.0, 0.0], [0.7, 1.3, 0.0], [0.4, 0.9, 1.7]], dtype=q.dtype
    )
    (gradient,) = torch.autograd.grad((lattice * weights).sum(), q)
    assert bool(torch.isfinite(gradient).all())
    assert torch.count_nonzero(gradient[~mask]) == 0
    assert bool((gradient[mask].abs() > 1e-5).all())

    changed_padding = q.detach().clone()
    changed_padding[~mask] = 100.0
    torch.testing.assert_close(
        q_to_lattice(changed_padding, **metadata), lattice.detach(), rtol=0, atol=0
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("coordinate", [0, 5], ids=["active", "inactive"])
def test_nonfinite_q_is_rejected_including_padding(value, coordinate):
    q = _sample_q()
    q[0, coordinate] = value
    with pytest.raises(ValueError, match="q contains NaN or Inf"):
        q_to_lattice(q, **_metadata(LatticeTemplate.CUBIC, 195))


@pytest.mark.parametrize(
    "template,spacegroup",
    [
        (LatticeTemplate.CUBIC, 3),
        (LatticeTemplate.MONOCLINIC_B, 1),
        (LatticeTemplate.RHOMBOHEDRAL, 143),
        (LatticeTemplate.HEXAGONAL, 146),
        (LatticeTemplate.RHOMBOHEDRAL, 195),
    ],
)
def test_incompatible_spacegroup_template_is_rejected(template, spacegroup):
    with pytest.raises(ValueError, match="incompatible with spacegroup"):
        q_to_lattice(_sample_q(), **_metadata(template, spacegroup))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_decoding_promotes_and_remains_finite(dtype):
    metadata = {
        "spacegroup": torch.tensor([case.values[1] for case in TEMPLATE_CASES]),
        "lattice_template_id": torch.tensor(
            [int(case.values[0]) for case in TEMPLATE_CASES]
        ),
    }
    # Finite but extreme coordinates exercise length and correlation safeguards.
    inputs = [
        _sample_q(dtype).expand(len(TEMPLATE_CASES), -1).clone(),
        torch.tensor([[20.0, -20.0, 20.0, 20.0, -20.0, 20.0]], dtype=dtype)
        .expand(len(TEMPLATE_CASES), -1)
        .clone(),
    ]
    for q in inputs:
        q.requires_grad_()
        lattice = q_to_lattice(q, **metadata)
        assert lattice.dtype == torch.float32
        assert bool(torch.isfinite(lattice).all())
        assert bool((torch.linalg.det(lattice) > 0).all())
        torch.testing.assert_close(
            lattice, q_to_lattice(q.detach().float(), **metadata), rtol=0, atol=0
        )
        (gradient,) = torch.autograd.grad(lattice.sum(), q)
        assert bool(torch.isfinite(gradient).all())

    # The encoder also promotes low precision input before forming its metric.
    quantized_lattice = q_to_lattice(inputs[0].detach(), **metadata).to(dtype)
    encoded = lattice_to_q(quantized_lattice, **metadata)
    assert encoded.dtype == torch.float32
    assert bool(torch.isfinite(encoded).all())
    torch.testing.assert_close(
        encoded, lattice_to_q(quantized_lattice.float(), **metadata), rtol=0, atol=0
    )


def test_analytic_inverse_matches_linalg_for_skewed_float64_batches():
    bases = torch.tensor(
        [
            [[2.4, 1.7, -0.8], [-0.4, 3.2, 1.1], [0.6, -1.3, 2.7]],
            [[1.2, -2.1, 0.3], [0.7, 2.8, -1.6], [-0.9, 0.4, 3.5]],
        ],
        dtype=torch.float64,
    )
    scales = torch.tensor([0.1, 1.0, 11.0], dtype=torch.float64)
    matrices = (bases[:, None] * scales[None, :, None, None]).transpose(-1, -2)
    assert not matrices.is_contiguous()
    inverse = batched_inverse_3x3(matrices)
    assert inverse.dtype == torch.float64
    assert inverse.shape == matrices.shape
    torch.testing.assert_close(inverse, torch.linalg.inv(matrices), rtol=1e-12, atol=1e-12)
    identity = torch.eye(3, dtype=torch.float64).expand_as(matrices)
    torch.testing.assert_close(matrices @ inverse, identity, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(inverse @ matrices, identity, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4)])
def test_analytic_inverse_rejects_wrong_trailing_shape(shape):
    with pytest.raises(ValueError, match="trailing shape"):
        batched_inverse_3x3(torch.zeros(shape))
