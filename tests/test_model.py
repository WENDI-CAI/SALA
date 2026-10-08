"""End-to-end architecture checks beyond zero-initialized output heads."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from sala import ModelConfig, SelfCondition
from sala.core.lattice_manifold import q_active_mask


def _nonzero_model(**overrides):
    """Activate deliberate zero-init paths only inside these gradient tests."""
    with torch.random.fork_rng():
        torch.manual_seed(19)
        model = ModelConfig.tiny(**overrides).build()
        # Fresh SALA heads and AdaLN gates intentionally make zero outputs.
        # A gradient test must open those paths before testing trunk behavior.
        modules = [model.atom_head, model.lattice_head[-1],
                   model.final_norm.modulation[-1],
                   model.token_embedder.component_context[-1]]
        modules.extend(block.modulation[-1] for block in model.blocks)
        if model.self_conditioning_enabled:
            modules.extend([model.token_embedder.atom_self_condition,
                            model.token_embedder.lattice_self_condition])
        for module in modules:
            nn.init.normal_(module.weight, std=0.025)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.03)
        for block in model.token_embedder.covalent_blocks:
            nn.init.constant_(block.gate, 0.2)
    return model


def test_fresh_tiny_model_has_intentional_zero_velocity_heads(synthetic_inputs):
    model = ModelConfig.tiny().build().eval()
    data = synthetic_inputs
    with torch.no_grad():
        atoms, lattice = model(data.batch, data.x, data.q, data.t)
    assert atoms.shape == (9, 3) and lattice.shape == (3, 6)
    assert torch.count_nonzero(atoms) == 0
    assert torch.count_nonzero(lattice) == 0


@pytest.mark.parametrize("checkpointing", [False, True])
def test_nonzero_forward_backward_reaches_geometry_attention_and_pair_bias(
    synthetic_inputs, checkpointing,
):
    model = _nonzero_model(use_gradient_checkpointing=checkpointing).train()
    data = synthetic_inputs
    x = data.x.clone().requires_grad_()
    q = data.q.clone().requires_grad_()
    mask = q_active_mask(spacegroup=data.batch.spacegroup,
                         lattice_template_id=data.batch.lattice_template_id).bool()
    # The model receives caller-prepared model_q. Its inactive components must
    # be projected before tokenization, separately from the physical decoder.
    atoms, lattice = model(
        data.batch, model_x=x / 8.0, physical_q=q, t=data.t,
        physical_x=x, model_q=q * mask,
        atom_input_scale=torch.full((3,), 1.0 / 8.0),
        flow_condition=1000.0 * data.t,
    )
    assert atoms.shape == x.shape and lattice.shape == q.shape
    assert torch.isfinite(atoms).all() and torch.isfinite(lattice).all()
    assert atoms.abs().max() > 1e-5 and lattice.abs().max() > 1e-5
    assert torch.count_nonzero(lattice[~mask]) == 0
    loss = atoms.square().sum() + lattice.square().sum()
    loss.backward()
    for gradient in [x.grad, q.grad]:
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 1e-7
    assert torch.count_nonzero(q.grad[~mask]) == 0
    exercised = {
        "atom geometry": model.token_embedder.atom_geometry[0].weight,
        "covalent messages": model.token_embedder.covalent_blocks[0].message[0].weight,
        "attention": model.blocks[0].attention.qkv_proj.weight,
        "feed-forward": model.blocks[0].ffn.gate_up.weight,
        "pair geometry": model.pair_embedding.atom_pair_mlp[0].weight,
        "lattice head": model.lattice_head[-1].weight,
    }
    for name, parameter in exercised.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 1e-9, name


def test_prepared_condition_matches_uncached_and_rejects_a_different_batch(synthetic_inputs):
    model = _nonzero_model().eval()
    data = synthetic_inputs
    condition = model.prepare_condition(data.batch)
    with torch.no_grad():
        direct = model(data.batch, data.x, data.q, data.t)
        cached = model(data.batch, data.x, data.q, data.t, prepared_condition=condition)
    for first, second in zip(direct, cached):
        torch.testing.assert_close(first, second)
    with pytest.raises(ValueError, match="different AsuBatch"):
        model(replace(data.batch), data.x, data.q, data.t, prepared_condition=condition)


def test_self_conditioning_respects_graph_presence_and_inactive_q(synthetic_inputs):
    model = _nonzero_model().eval()
    data = synthetic_inputs
    atom_residual = torch.linspace(-0.3, 0.4, 27).reshape(9, 3).requires_grad_()
    lattice_residual = torch.linspace(-0.2, 0.5, 18).reshape(3, 6).requires_grad_()
    present = torch.tensor([True, False, True])
    conditioning = SelfCondition(atom_residual, lattice_residual, present)
    baseline = model(data.batch, data.x, data.q, data.t)
    actual = model(data.batch, data.x, data.q, data.t, self_condition=conditioning)
    absent_atoms = ~present[data.batch.batch]
    torch.testing.assert_close(actual[0][absent_atoms], baseline[0][absent_atoms])
    torch.testing.assert_close(actual[1][~present], baseline[1][~present])
    assert (actual[0][~absent_atoms] - baseline[0][~absent_atoms]).abs().max() > 1e-6
    assert (actual[1][present] - baseline[1][present]).abs().max() > 1e-6
    (actual[0].square().sum() + actual[1].square().sum()).backward()
    assert atom_residual.grad is not None and lattice_residual.grad is not None
    assert torch.isfinite(atom_residual.grad).all() and torch.isfinite(lattice_residual.grad).all()
    assert atom_residual.grad[~absent_atoms].abs().sum() > 1e-8
    assert torch.count_nonzero(atom_residual.grad[absent_atoms]) == 0
    mask = q_active_mask(spacegroup=data.batch.spacegroup,
                         lattice_template_id=data.batch.lattice_template_id).bool()
    assert lattice_residual.grad[present[:, None] & mask].abs().sum() > 1e-8
    assert torch.count_nonzero(lattice_residual.grad[~(present[:, None] & mask)]) == 0


def test_all_absent_self_condition_is_equivalent_to_no_condition(synthetic_inputs):
    model = _nonzero_model().eval()
    data = synthetic_inputs
    absent = SelfCondition(torch.full_like(data.x, 1e4), torch.full_like(data.q, -1e4),
                           torch.zeros(3, dtype=torch.bool))
    with torch.no_grad():
        first = model(data.batch, data.x, data.q, data.t)
        second = model(data.batch, data.x, data.q, data.t, self_condition=absent)
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize("field", ["atom_residual", "lattice_residual", "graph_present"])
def test_malformed_self_conditioning_shapes_are_rejected(synthetic_inputs, field):
    model = ModelConfig.tiny().build()
    data = synthetic_inputs
    condition = SelfCondition(torch.zeros_like(data.x), torch.zeros_like(data.q),
                              torch.ones(3, dtype=torch.bool))
    condition = replace(condition, **{field: torch.zeros(1)})
    with pytest.raises(ValueError, match="self-conditioning.*shape"):
        model(data.batch, data.x, data.q, data.t, self_condition=condition)


def test_disabled_self_conditioning_rejects_supplied_values(synthetic_inputs):
    model = ModelConfig.tiny(self_conditioning_enabled=False).build()
    data = synthetic_inputs
    condition = SelfCondition(torch.zeros_like(data.x), torch.zeros_like(data.q),
                              torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="disabled velocity field"):
        model(data.batch, data.x, data.q, data.t, self_condition=condition)


def test_reference_configuration_has_expected_parameter_count_without_allocating_weights():
    with torch.device("meta"):
        model = ModelConfig().build()
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) == 486_282_617
    assert len(model.blocks) == 24
    assert model.self_conditioning_enabled
