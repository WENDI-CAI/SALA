"""Run one CPU forward pass without checkpoints or scientific input files.

From the repository root, after installation:
    python examples/minimal_forward.py
For a source checkout:
    PYTHONPATH=src python examples/minimal_forward.py
"""

from __future__ import annotations

import torch

from sala import ModelConfig
from sala.core.lattice_manifold import q_active_mask

if __package__:
    from .synthetic import make_synthetic_inputs
else:
    from synthetic import make_synthetic_inputs


def main() -> None:
    torch.manual_seed(7)
    inputs = make_synthetic_inputs()
    model = ModelConfig.tiny().build().eval()
    active = q_active_mask(
        spacegroup=inputs.batch.spacegroup,
        lattice_template_id=inputs.batch.lattice_template_id,
    )
    # These explicit toy scales demonstrate the tensor interface only. Masked
    # physical q is not the production model's whitened q representation.
    # This example implements no learned prior or production preconditioning.
    atom_scale = torch.full((inputs.batch.batch_size,), 1.0 / 8.0)
    with torch.no_grad():
        atom_velocity, lattice_velocity = model(
            inputs.batch,
            model_x=inputs.x / 8.0,
            physical_q=inputs.q,
            t=inputs.t,
            physical_x=inputs.x,
            model_q=inputs.q * active,
            atom_input_scale=atom_scale,
            flow_condition=1000.0 * inputs.t,
        )
    print("SALA tiny CPU architecture example")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"Graphs: {inputs.batch.batch_size}; ASU atoms: {inputs.batch.total_atoms}")
    print(f"Full-cell atoms per graph: {inputs.batch.n_full_atoms.tolist()}")
    print(f"Atom velocity shape: {tuple(atom_velocity.shape)}")
    print(f"Lattice velocity shape: {tuple(lattice_velocity.shape)}")
    print(f"Finite outputs: {bool(torch.isfinite(atom_velocity).all() and torch.isfinite(lattice_velocity).all())}")
    print("Toy scaling: x / 8 and masked physical q; not production whitening.")
    print("Randomly initialized weights; no checkpoint is loaded.")
    print("Output heads start at zero, so this fresh model returns zero velocities.")
    print("The inputs are synthetic tensors, not generated or validated crystals.")


if __name__ == "__main__":
    main()
