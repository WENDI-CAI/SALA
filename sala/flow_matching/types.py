from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor


def _require_all(predicate: Tensor, message: str) -> None:
    """Validate a scalar predicate without synchronizing the CUDA hot path."""
    predicate = predicate.reshape(())
    if predicate.device.type != "cuda":
        if not bool(predicate):
            raise ValueError(message)
        return
    assert_async = getattr(torch, "_assert_async", None)
    if assert_async is not None:
        assert_async(predicate, message)
    else:  # pragma: no cover
        torch._assert(predicate, message)


@dataclass(frozen=True)
class FlowSelfConditioningState:
    """Detached clean-endpoint residuals in normalized Cartesian/whitened-q space."""

    atom_residual: Tensor
    lattice_residual: Tensor
    graph_present: Tensor

    @staticmethod
    def _validate_tensors(
        atom_residual: Tensor,
        lattice_residual: Tensor,
        graph_present: Tensor,
    ) -> None:
        if atom_residual.ndim != 2 or atom_residual.shape[-1] != 3:
            raise ValueError(
                "flow self-conditioning atom_residual must have shape (N, 3)"
            )
        if lattice_residual.ndim != 2 or lattice_residual.shape[-1] != 6:
            raise ValueError(
                "flow self-conditioning lattice_residual must have shape (B, 6)"
            )
        if (
            graph_present.ndim != 1
            or graph_present.shape[0] != lattice_residual.shape[0]
        ):
            raise ValueError(
                "flow self-conditioning graph_present must have shape (B,)"
            )
        if (
            not atom_residual.is_floating_point()
            or not lattice_residual.is_floating_point()
        ):
            raise TypeError(
                "flow self-conditioning residuals must be floating-point tensors"
            )
        if (
            atom_residual.device != lattice_residual.device
            or graph_present.device != lattice_residual.device
        ):
            raise ValueError(
                "flow self-conditioning tensors must share one device"
            )
        _require_all(
            torch.isfinite(atom_residual).all()
            & torch.isfinite(lattice_residual).all(),
            "flow self-conditioning residuals must be finite",
        )
        if graph_present.dtype != torch.bool:
            present_float = graph_present.float()
            _require_all(
                (
                    torch.isfinite(present_float)
                    & ((present_float == 0) | (present_float == 1))
                ).all(),
                "flow self-conditioning graph_present must contain boolean/0/1 values",
            )

    def __post_init__(self) -> None:
        self._validate_tensors(
            self.atom_residual, self.lattice_residual, self.graph_present
        )
        object.__setattr__(
            self, "atom_residual", self.atom_residual.detach().clone()
        )
        object.__setattr__(
            self, "lattice_residual", self.lattice_residual.detach().clone()
        )
        object.__setattr__(
            self,
            "graph_present",
            self.graph_present.detach().to(dtype=torch.bool).clone(),
        )

    @classmethod
    def _from_owned_detached(
        cls,
        *,
        atom_residual: Tensor,
        lattice_residual: Tensor,
        graph_present: Tensor,
    ) -> "FlowSelfConditioningState":
        """Adopt adapter-owned detached tensors without an additional large copy."""
        cls._validate_tensors(atom_residual, lattice_residual, graph_present)
        value = object.__new__(cls)
        object.__setattr__(value, "atom_residual", atom_residual.detach())
        object.__setattr__(value, "lattice_residual", lattice_residual.detach())
        object.__setattr__(
            value,
            "graph_present",
            graph_present.detach().to(dtype=torch.bool).clone(),
        )
        return value

    @property
    def num_atoms(self) -> int:
        return int(self.atom_residual.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.lattice_residual.shape[0])

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> "FlowSelfConditioningState":
        atom_dtype = dtype or self.atom_residual.dtype
        lattice_dtype = dtype or self.lattice_residual.dtype
        return FlowSelfConditioningState(
            atom_residual=self.atom_residual.to(
                device=device, dtype=atom_dtype
            ),
            lattice_residual=self.lattice_residual.to(
                device=device, dtype=lattice_dtype
            ),
            graph_present=self.graph_present.to(device=device),
        )

    @classmethod
    def zeros(
        cls,
        *,
        num_atoms: int,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "FlowSelfConditioningState":
        if int(num_atoms) < 0 or int(batch_size) < 0:
            raise ValueError(
                "flow self-conditioning zero-state sizes must be non-negative"
            )
        return cls(
            atom_residual=torch.zeros(
                (int(num_atoms), 3), device=device, dtype=dtype
            ),
            lattice_residual=torch.zeros(
                (int(batch_size), 6), device=device, dtype=dtype
            ),
            graph_present=torch.zeros(
                (int(batch_size),), device=device, dtype=torch.bool
            ),
        )


@dataclass(frozen=True)
class FlowPathSample:
    """One conditional rectified-flow training state in normalized coordinates."""

    time: Tensor
    state_x: Tensor
    state_q: Tensor
    clean_x: Tensor
    clean_q: Tensor
    clean_state_x: Tensor
    clean_state_q: Tensor
    noise_x: Tensor
    noise_q: Tensor
    target_velocity_x: Tensor
    target_velocity_q: Tensor


@dataclass(frozen=True)
class FlowPrediction:
    """Velocity prediction and its implied clean endpoint."""

    velocity_x: Tensor
    velocity_q: Tensor
    clean_x: Tensor
    clean_q: Tensor
    clean_state_x: Tensor
    clean_state_q: Tensor


@dataclass
class FlowTrainingOutput:
    loss: Tensor
    metrics: Dict[str, Tensor]
    path_sample: Optional[FlowPathSample] = None
    per_graph_loss: Optional[Tensor] = None


__all__ = [
    "FlowPathSample",
    "FlowPrediction",
    "FlowSelfConditioningState",
    "FlowTrainingOutput",
]
