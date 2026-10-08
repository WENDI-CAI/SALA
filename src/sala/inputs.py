"""Optional tensor inputs to the standalone velocity field."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class SelfCondition:
    """Previous-endpoint residuals in the model's input coordinate spaces.

    ``atom_residual`` has shape ``[N_asu, 3]`` in normalized Cartesian space;
    ``lattice_residual`` has shape ``[B, 6]`` in model q-space; and the Boolean
    ``graph_present`` mask has shape ``[B]``. The model masks absent graphs and
    inactive lattice components. This container does not calculate residuals
    or perform a sampling step.
    """

    atom_residual: Tensor
    lattice_residual: Tensor
    graph_present: Tensor


__all__ = ["SelfCondition"]
