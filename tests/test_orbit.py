"""Explicit symmetry, padded orbit geometry and tangent consistency."""

from dataclasses import replace

import pytest
import torch

from sala.core.orbit import UnsupportedSpecialPositionError, build_orbit_plan
from sala.core.symmetry import canonicalize_affine_symmetry_ops
from sala.models.crystal_dit.pair import (
    DIFFERENT_COMPONENT_TYPE,
    SAME_BASE_COMPONENT_OTHER_INSTANCE,
    SAME_COMPONENT_INSTANCE,
    SAME_COMPONENT_TYPE_OTHER_BASE,
    component_relation_index,
)


def test_expansion_matches_hand_written_affine_operations(synthetic_inputs):
    inputs = synthetic_inputs
    batch = inputs.batch
    plan = build_orbit_plan(batch)
    expanded = plan.expand_coordinates(inputs.x, batch.lattice_gt)
    assert plan.full_counts.tolist() == [3, 8, 4]
    assert expanded.cart.shape == (3, 8, 3)
    assert plan.token_valid.sum(dim=1).tolist() == [6, 11, 7]
    for graph in range(3):
        start, end = batch.ptr[graph:graph + 2].tolist()
        frac = batch.frac_coords_gt[start:end]
        operations = batch.symmetry_ops[graph, batch.symmetry_mask[graph]]
        expected = torch.cat(
            [frac @ op[:3, :3].T + op[:3, 3] for op in operations]
        )
        n_full = len(expected)
        torch.testing.assert_close(expanded.frac[graph, :n_full], expected)
        torch.testing.assert_close(
            expanded.cart[graph, :n_full], expected @ batch.lattice_gt[graph]
        )
    assert torch.count_nonzero(expanded.frac[~plan.full_valid]) == 0
    assert torch.count_nonzero(expanded.cart[~plan.full_valid]) == 0


def test_vector_expansion_pullback_and_projection_respect_padding(synthetic_inputs):
    batch = synthetic_inputs.batch
    plan = build_orbit_plan(batch)
    vectors = torch.arange(27, dtype=torch.float32).reshape(9, 3) / 17 - 0.8
    expanded = plan.expand_vectors(vectors, batch.lattice_gt)
    torch.testing.assert_close(
        plan.pullback_vectors(expanded, batch.lattice_gt), vectors,
        rtol=2e-6, atol=2e-6,
    )
    arbitrary = torch.sin(torch.arange(72, dtype=torch.float32)).reshape(3, 8, 3)
    projected = plan.project_full_vectors(arbitrary, batch.lattice_gt)
    torch.testing.assert_close(
        plan.project_full_vectors(projected, batch.lattice_gt), projected,
        rtol=2e-6, atol=2e-6,
    )
    padded_poison = arbitrary.clone()
    padded_poison[~plan.full_valid] = 1e6
    torch.testing.assert_close(
        plan.pullback_vectors(padded_poison, batch.lattice_gt),
        plan.pullback_vectors(arbitrary, batch.lattice_gt),
    )
    assert torch.count_nonzero(projected[~plan.full_valid]) == 0


def test_inactive_operation_slots_are_not_read(synthetic_inputs):
    batch = synthetic_inputs.batch
    poisoned = batch.symmetry_ops.clone()
    poisoned[~batch.symmetry_mask] = torch.nan
    changed = replace(batch, symmetry_ops=poisoned)
    first = build_orbit_plan(batch).expand_coordinates(synthetic_inputs.x, batch.lattice_gt)
    second = build_orbit_plan(changed).expand_coordinates(synthetic_inputs.x, batch.lattice_gt)
    torch.testing.assert_close(first.cart, second.cart)


def test_component_relations_distinguish_bases_instances_and_types(synthetic_inputs):
    plan = build_orbit_plan(synthetic_inputs.batch)
    valid = plan.full_valid[:, :, None] & plan.full_valid[:, None, :]
    same_type = plan.full_component_type[:, :, None] == plan.full_component_type[:, None, :]
    same_base = plan.full_base_component[:, :, None] == plan.full_base_component[:, None, :]
    same_instance = plan.full_component_instance[:, :, None] == plan.full_component_instance[:, None, :]
    relations = component_relation_index(same_type & valid, same_base & valid, same_instance & valid)
    assert relations[1, 0, 1] == SAME_COMPONENT_INSTANCE
    assert relations[1, 0, 4] == SAME_BASE_COMPONENT_OTHER_INSTANCE
    assert relations[1, 0, 2] == SAME_COMPONENT_TYPE_OTHER_BASE
    assert relations[0, 0, 2] == DIFFERENT_COMPONENT_TYPE
    torch.testing.assert_close(relations, relations.transpose(-1, -2))


@pytest.mark.parametrize("missing", ["symmetry_ops", "symmetry_mask"])
def test_missing_explicit_symmetry_is_rejected(synthetic_inputs, missing):
    batch = replace(synthetic_inputs.batch, **{missing: None})
    with pytest.raises(ValueError, match="explicit|symmetry_mask"):
        build_orbit_plan(batch)


def test_all_masked_operations_and_inconsistent_counts_are_rejected(synthetic_inputs):
    batch = synthetic_inputs.batch
    mask = batch.symmetry_mask.clone()
    mask[1] = False
    with pytest.raises(ValueError, match="explicit"):
        build_orbit_plan(replace(batch, symmetry_mask=mask))
    with pytest.raises(ValueError, match="symmetry count disagrees"):
        build_orbit_plan(replace(batch, n_symops=torch.tensor([2, 2, 2])))
    with pytest.raises(ValueError, match="full-cell count disagrees"):
        build_orbit_plan(replace(batch, n_full_atoms=torch.tensor([4, 8, 4])))


def test_special_position_and_duplicate_orbit_representatives_are_rejected(synthetic_inputs):
    batch = synthetic_inputs.batch
    special = batch.frac_coords_gt.clone()
    special[3] = 0.0  # Inversion fixes the origin.
    with pytest.raises(UnsupportedSpecialPositionError, match="general-position"):
        build_orbit_plan(replace(batch, frac_coords_gt=special))
    duplicate = batch.frac_coords_gt.clone()
    duplicate[4] = 1.0 - duplicate[3]
    with pytest.raises(UnsupportedSpecialPositionError):
        build_orbit_plan(replace(batch, frac_coords_gt=duplicate))


def test_symmetry_translation_gauge_and_invalid_affine_rows(synthetic_inputs):
    ops = synthetic_inputs.batch.symmetry_ops[2].clone()
    shifted = ops.clone()
    shifted[:, :3, 3] += torch.tensor([3.0, -2.0, 1.0])
    torch.testing.assert_close(canonicalize_affine_symmetry_ops(shifted), ops)
    invalid = ops.clone()
    invalid[0, 3, 0] = 0.5
    with pytest.raises(ValueError, match="homogeneous affine"):
        canonicalize_affine_symmetry_ops(invalid)
