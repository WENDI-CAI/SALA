"""Small hand-written tensor inputs; no real crystal or dataset is included.

The three graphs use explicit P1, P-1, and unique-axis-b P2_1 operations.
Coordinates and chemical labels are chosen only to exercise the tensor API.
They are not measured structures, generated samples, or validated chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from sala import AsuBatch
from sala.core.lattice_manifold import LatticeTemplate, q_to_lattice


@dataclass(frozen=True)
class SyntheticInputs:
    """A mixed graph batch and one arbitrary Cartesian/q/time state."""

    batch: AsuBatch
    x: Tensor
    q: Tensor
    t: Tensor


def make_synthetic_inputs(device: str | torch.device = "cpu") -> SyntheticInputs:
    """Return nine synthetic ASU atoms in three graphs with 3/8/4 full atoms.

    Component ids are global and contiguous. Covalent edges are directed,
    with each undirected toy bond supplied in both directions. The padded
    operation mask is authoritative: P1 has one operation; the other two
    graphs have two. All coordinates are general-position representatives.
    """

    identity = torch.eye(4)
    inversion = torch.eye(4)
    inversion[:3, :3] = -torch.eye(3)
    screw = torch.eye(4)
    screw[:3, :3] = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
    screw[1, 3] = 0.5  # (x, y, z) -> (-x, y + 1/2, -z).
    operations = torch.zeros(3, 2, 4, 4)
    operations[:, 0] = identity
    operations[1, 1] = inversion
    operations[2, 1] = screw
    operation_mask = torch.tensor([[True, False], [True, True], [True, True]])

    fractional = torch.tensor(
        [
            [0.17, 0.23, 0.31], [0.27, 0.29, 0.36], [0.62, 0.53, 0.41],
            [0.13, 0.21, 0.32], [0.24, 0.28, 0.37],
            [0.49, 0.14, 0.23], [0.57, 0.19, 0.28],
            [0.19, 0.32, 0.27], [0.29, 0.39, 0.35],
        ],
        dtype=torch.float32,
    )
    spacegroup = torch.tensor([1, 2, 4])
    templates = torch.tensor(
        [LatticeTemplate.TRICLINIC, LatticeTemplate.TRICLINIC,
         LatticeTemplate.MONOCLINIC_B],
        dtype=torch.long,
    )
    q = torch.tensor(
        [[2.2, 2.1, 2.3, 0.10, 0.08, -0.05],
         [2.3, 2.4, 2.2, -0.10, 0.15, 0.08],
         [2.25, 2.35, 2.45, 0.0, -0.20, 0.0]],
        dtype=torch.float32,
    )
    lattice = q_to_lattice(q, spacegroup=spacegroup, lattice_template_id=templates)
    graph_index = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2])
    cartesian = torch.bmm(
        fractional.unsqueeze(1), lattice.index_select(0, graph_index)
    ).squeeze(1)
    batch = AsuBatch(
        atom_types=torch.tensor([6, 8, 7, 6, 7, 6, 8, 6, 8]),
        spacegroup=spacegroup,
        component_id=torch.tensor([0, 0, 1, 2, 2, 3, 3, 4, 4]),
        atom_formal_charge=torch.tensor([0, 0, 0, 1, 0, -1, 0, 0, 0]),
        atom_charge_mask=torch.ones(9, dtype=torch.bool),
        component_formal_charge=torch.tensor([0, 0, 1, -1, 0]),
        component_type_id=torch.tensor([0, 1, 2, 2, 3]),
        num_atoms=torch.tensor([3, 4, 2]),
        edge_index_conv=torch.tensor([[0, 1, 3, 4, 5, 6, 7, 8],
                                      [1, 0, 4, 3, 6, 5, 8, 7]]),
        edge_attr_conv=torch.zeros(8, dtype=torch.long),
        frac_coords_gt=fractional,
        lattice_gt=lattice,
        batch=graph_index,
        ptr=torch.tensor([0, 3, 7, 9]),
        component_batch=torch.tensor([0, 0, 1, 1, 2]),
        cart_coords_gt=cartesian,
        hybridization=torch.tensor([3, 2, 3, 3, 3, 3, 2, 3, 2]),
        n_symops=torch.tensor([1, 2, 2]),
        n_full_atoms=torch.tensor([3, 8, 4]),
        n_heavy_cell=torch.tensor([3, 8, 4]),
        lattice_template_id=templates,
        symmetry_ops=operations,
        symmetry_mask=operation_mask,
    )
    return SyntheticInputs(
        batch=batch.to(device),
        x=cartesian.to(device),
        q=q.to(device),
        t=torch.tensor([0.20, 0.50, 0.80], device=device),
    )


__all__ = ["SyntheticInputs", "make_synthetic_inputs"]
