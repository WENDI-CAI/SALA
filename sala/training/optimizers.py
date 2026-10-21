from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class OptimizerParameterAudit:
    """Deterministic accounting for every trainable optimizer parameter."""

    muon_names: tuple[str, ...]
    adamw_names: tuple[str, ...]
    muon_numel: int
    adamw_numel: int
    muon_chunk_specs: Dict[str, tuple[int, ...]]
    exclusion_reason_counts: Dict[str, int]

    @property
    def trainable_numel(self) -> int:
        return self.muon_numel + self.adamw_numel

    @property
    def muon_fraction(self) -> float:
        if self.trainable_numel == 0:
            return 0.0
        return float(self.muon_numel) / float(self.trainable_numel)

    @property
    def chunked_muon_tensor_count(self) -> int:
        return len(self.muon_chunk_specs)

    @property
    def muon_orthogonalization_count(self) -> int:
        return sum(
            len(self.muon_chunk_specs.get(name, ())) or 1
            for name in self.muon_names
        )

    def summary(self) -> str:
        return (
            f"Muon: {len(self.muon_names)} tensors/{self.muon_numel:,} params; "
            f"CMuon: {self.chunked_muon_tensor_count} chunked tensors/"
            f"{self.muon_orthogonalization_count} orthogonalizations; "
            f"AdamW: {len(self.adamw_names)} tensors/{self.adamw_numel:,} params; "
            f"Muon fraction: {self.muon_fraction:.2%}; "
            f"exclusions: {dict(sorted(self.exclusion_reason_counts.items()))}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON/checkpoint-safe, deterministically ordered summary."""

        return {
            "muon_names": list(self.muon_names),
            "adamw_names": list(self.adamw_names),
            "muon_tensor_count": len(self.muon_names),
            "chunked_muon_tensor_count": self.chunked_muon_tensor_count,
            "muon_orthogonalization_count": self.muon_orthogonalization_count,
            "muon_chunk_specs": {
                name: list(self.muon_chunk_specs[name])
                for name in sorted(self.muon_chunk_specs)
            },
            "adamw_tensor_count": len(self.adamw_names),
            "muon_numel": int(self.muon_numel),
            "adamw_numel": int(self.adamw_numel),
            "trainable_numel": int(self.trainable_numel),
            "muon_fraction": float(self.muon_fraction),
            "exclusion_reason_counts": dict(
                sorted(self.exclusion_reason_counts.items())
            ),
        }


@dataclass(frozen=True)
class OptimizerParameterGroups:
    muon: tuple[nn.Parameter, ...]
    muon_chunk_sizes: tuple[tuple[int, ...] | None, ...]
    adamw: tuple[nn.Parameter, ...]
    audit: OptimizerParameterAudit


def _normalize_muon_chunk_sizes(
    raw_sizes: Any,
    parameter: nn.Parameter,
    *,
    name: str,
) -> tuple[int, ...] | None:
    if raw_sizes is None:
        return None
    try:
        sizes = tuple(int(size) for size in raw_sizes)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Muon chunk sizes for {name} must be an integer sequence"
        ) from exc
    if len(sizes) < 2 or any(size <= 0 for size in sizes):
        raise ValueError(
            f"Muon chunk sizes for {name} must contain at least two positive values"
        )
    if parameter.ndim != 2 or sum(sizes) != int(parameter.shape[0]):
        raise ValueError(
            f"Muon chunk sizes for {name} must partition matrix rows "
            f"{tuple(parameter.shape)}, got {sizes}"
        )
    return sizes


def _muon_exclusion_reason(
    name: str,
    parameter: nn.Parameter,
    *,
    registered: bool,
    min_dim: int,
    max_aspect_ratio: float,
) -> str | None:
    if parameter.ndim != 2:
        return "not_2d"
    if not name.endswith(".weight"):
        return "not_weight"
    if not registered:
        in_transformer_block = ".blocks." in f".{name}"
        if in_transformer_block and (".attn_mod." in name or ".ffn_mod." in name):
            return "adaln_conditioning"
        if "embedding" in name.lower():
            return "embedding"
        if "geometry" in name.lower() or "readout" in name.lower():
            return "geometry_or_output"
        return "outside_transformer_hidden"
    rows, cols = (int(parameter.shape[0]), int(parameter.shape[1]))
    if min(rows, cols) < int(min_dim):
        return "below_min_dim"
    aspect_ratio = float(max(rows, cols)) / float(max(1, min(rows, cols)))
    if aspect_ratio > float(max_aspect_ratio):
        return "aspect_ratio"
    return None


def partition_optimizer_parameters(
    module: nn.Module,
    *,
    min_dim: int = 128,
    max_aspect_ratio: float = 8.0,
) -> OptimizerParameterGroups:
    """Partition trainable parameters into disjoint Muon and AdamW groups.

    Name matching is prefix agnostic, so the result is unchanged when the
    module is nested under Lightning or a DistributedDataParallel wrapper.
    """

    registered_ids: set[int] = set()
    registered_chunk_sizes: dict[int, tuple[int, ...]] = {}
    for module_name, submodule in module.named_modules():
        if not bool(getattr(submodule, "_sala_muon", False)):
            continue
        for local_name, parameter in submodule.named_parameters(recurse=False):
            if local_name != "weight":
                raise ValueError(
                    "Muon-tagged modules may expose only an eligible weight; "
                    f"found {local_name!r}"
                )
            parameter_id = id(parameter)
            registered_ids.add(parameter_id)
            chunk_sizes = _normalize_muon_chunk_sizes(
                getattr(submodule, "_sala_muon_chunk_sizes", None),
                parameter,
                name=f"{module_name}.{local_name}".lstrip("."),
            )
            if chunk_sizes is not None:
                registered_chunk_sizes[parameter_id] = chunk_sizes

    muon_parameters: list[nn.Parameter] = []
    muon_chunk_sizes: list[tuple[int, ...] | None] = []
    adamw_parameters: list[nn.Parameter] = []
    muon_names: list[str] = []
    muon_chunk_specs: dict[str, tuple[int, ...]] = {}
    adamw_names: list[str] = []
    exclusion_reasons: Counter[str] = Counter()
    seen: set[int] = set()

    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            raise ValueError(
                f"Trainable parameter is registered more than once: {name}"
            )
        seen.add(parameter_id)
        reason = _muon_exclusion_reason(
            name,
            parameter,
            registered=parameter_id in registered_ids,
            min_dim=min_dim,
            max_aspect_ratio=max_aspect_ratio,
        )
        if reason is None:
            chunk_sizes = registered_chunk_sizes.get(parameter_id)
            if chunk_sizes is not None:
                cols = int(parameter.shape[1])
                for chunk_rows in chunk_sizes:
                    if min(chunk_rows, cols) < int(min_dim):
                        raise ValueError(
                            f"Muon chunk for {name} falls below min_dim={min_dim}: "
                            f"{chunk_rows}x{cols}"
                        )
                    aspect_ratio = float(max(chunk_rows, cols)) / float(
                        max(1, min(chunk_rows, cols))
                    )
                    if aspect_ratio > float(max_aspect_ratio):
                        raise ValueError(
                            f"Muon chunk for {name} exceeds max_aspect_ratio="
                            f"{max_aspect_ratio}: {chunk_rows}x{cols}"
                        )
            muon_parameters.append(parameter)
            muon_chunk_sizes.append(chunk_sizes)
            muon_names.append(name)
            if chunk_sizes is not None:
                muon_chunk_specs[name] = chunk_sizes
        else:
            adamw_parameters.append(parameter)
            adamw_names.append(name)
            exclusion_reasons[reason] += 1

    if not muon_parameters:
        raise ValueError(
            "hybrid_muon selected no eligible Transformer matrices; expected "
            "explicitly Muon-tagged attention/FFN weights satisfying "
            f"min_dim={min_dim} and max_aspect_ratio={max_aspect_ratio}"
        )
    if len(seen) != len(muon_parameters) + len(adamw_parameters):
        raise RuntimeError("Optimizer parameter partition is not exhaustive")

    audit = OptimizerParameterAudit(
        muon_names=tuple(muon_names),
        adamw_names=tuple(adamw_names),
        muon_numel=sum(parameter.numel() for parameter in muon_parameters),
        adamw_numel=sum(parameter.numel() for parameter in adamw_parameters),
        muon_chunk_specs=muon_chunk_specs,
        exclusion_reason_counts=dict(exclusion_reasons),
    )
    return OptimizerParameterGroups(
        muon=tuple(muon_parameters),
        muon_chunk_sizes=tuple(muon_chunk_sizes),
        adamw=tuple(adamw_parameters),
        audit=audit,
    )


def newton_schulz_orthogonalize(matrix: Tensor, *, steps: int = 5) -> Tensor:
    """Approximate the polar factor of a two-dimensional update matrix.

    The quintic iteration is the one used by the reference Muon optimizer.  It
    deliberately runs in float32 (or float64 for double inputs), independent of
    autocast, and forms the Gram matrix along the smaller matrix dimension.
    """

    if matrix.ndim != 2:
        raise ValueError(f"Muon requires a 2D update matrix, got shape {matrix.shape}")
    if int(steps) < 1:
        raise ValueError("Newton-Schulz steps must be positive")
    compute_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    x = matrix.to(dtype=compute_dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.transpose(0, 1)
    eps = torch.finfo(compute_dtype).eps
    x = x / x.norm().clamp_min(eps)

    # Coefficients for a stable quintic iteration whose slope at zero is > 1.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = x @ x.transpose(0, 1)
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.transpose(0, 1)
    return x


class HybridMuon(torch.optim.Optimizer):
    """CMuon for Transformer matrices and AdamW for remaining parameters.

    This implementation has no process-global state or collectives.  Under DDP
    every rank applies the same deterministic update to already-synchronized
    gradients, just like a conventional AdamW optimizer.  Functionally
    distinct row blocks of fused matrices are orthogonalized and RMS-scaled
    independently before their updates are concatenated.
    """

    def __init__(
        self,
        *,
        muon_parameters: Iterable[nn.Parameter],
        muon_chunk_sizes: Iterable[Sequence[int] | None] | None = None,
        adamw_parameters: Iterable[nn.Parameter],
        muon_lr: float,
        adamw_lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        muon_weight_decay: float = 0.0,
        adamw_weight_decay: float = 0.0,
        adamw_betas: Sequence[float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adjust_lr_fn: str = "match_rms_adamw",
    ) -> None:
        muon_parameters = list(muon_parameters)
        adamw_parameters = list(adamw_parameters)
        if muon_chunk_sizes is None:
            resolved_chunk_sizes: list[tuple[int, ...] | None] = [
                None for _ in muon_parameters
            ]
        else:
            raw_chunk_sizes = list(muon_chunk_sizes)
            if len(raw_chunk_sizes) != len(muon_parameters):
                raise ValueError(
                    "Muon chunk metadata must align one-to-one with Muon parameters"
                )
            resolved_chunk_sizes = [
                _normalize_muon_chunk_sizes(
                    sizes,
                    parameter,
                    name=f"muon_parameters[{index}]",
                )
                for index, (parameter, sizes) in enumerate(
                    zip(muon_parameters, raw_chunk_sizes)
                )
            ]
        if not muon_parameters:
            raise ValueError("HybridMuon requires at least one Muon parameter")
        if float(muon_lr) <= 0.0 or float(adamw_lr) <= 0.0:
            raise ValueError("Optimizer learning rates must be positive")
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("Muon momentum must be in [0, 1)")
        if int(ns_steps) < 1:
            raise ValueError("Muon Newton-Schulz steps must be positive")
        if float(muon_weight_decay) < 0.0 or float(adamw_weight_decay) < 0.0:
            raise ValueError("Optimizer weight decay must be non-negative")
        if len(adamw_betas) != 2:
            raise ValueError("adamw_betas must contain exactly two values")
        beta1, beta2 = float(adamw_betas[0]), float(adamw_betas[1])
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("AdamW betas must be in [0, 1)")
        if float(adamw_eps) <= 0.0:
            raise ValueError("AdamW epsilon must be positive")
        adjust_lr_fn = str(adjust_lr_fn).strip().lower()
        if adjust_lr_fn not in {"original", "match_rms_adamw"}:
            raise ValueError(
                "Muon adjust_lr_fn must be original or match_rms_adamw"
            )
        for parameter in muon_parameters:
            if parameter.ndim != 2:
                raise ValueError(
                    f"Muon parameter must be 2D, got shape {tuple(parameter.shape)}"
                )

        self._muon_chunk_sizes = {
            id(parameter): chunk_sizes
            for parameter, chunk_sizes in zip(
                muon_parameters, resolved_chunk_sizes
            )
            if chunk_sizes is not None
        }

        parameter_groups: list[Dict[str, Any]] = [
            {
                "params": muon_parameters,
                "algorithm": "muon",
                "lr": float(muon_lr),
                "weight_decay": float(muon_weight_decay),
                "momentum": float(momentum),
                "nesterov": bool(nesterov),
                "ns_steps": int(ns_steps),
                "adjust_lr_fn": adjust_lr_fn,
            }
        ]
        if adamw_parameters:
            parameter_groups.append(
                {
                    "params": adamw_parameters,
                    "algorithm": "adamw",
                    "lr": float(adamw_lr),
                    "weight_decay": float(adamw_weight_decay),
                    "betas": (beta1, beta2),
                    "eps": float(adamw_eps),
                }
            )
        super().__init__(parameter_groups, defaults={})

    @staticmethod
    def _compute_dtype(tensor: Tensor) -> torch.dtype:
        return torch.float64 if tensor.dtype == torch.float64 else torch.float32

    @staticmethod
    def _muon_update_scale(
        rows: int,
        cols: int,
        *,
        adjust_lr_fn: str,
    ) -> float:
        if adjust_lr_fn == "match_rms_adamw":
            return 0.2 * math.sqrt(float(max(rows, cols)))
        return math.sqrt(max(1.0, float(rows) / float(max(cols, 1))))

    def _orthogonalized_muon_update(
        self,
        parameter: nn.Parameter,
        update: Tensor,
        group: Mapping[str, Any],
    ) -> Tensor:
        chunk_sizes = self._muon_chunk_sizes.get(id(parameter))
        chunks = (
            torch.split(update, chunk_sizes, dim=0)
            if chunk_sizes is not None
            else (update,)
        )
        adjust_lr_fn = str(group.get("adjust_lr_fn", "match_rms_adamw"))
        scaled_chunks: list[Tensor] = []
        for chunk in chunks:
            orthogonalized = newton_schulz_orthogonalize(
                chunk,
                steps=int(group["ns_steps"]),
            )
            scale = self._muon_update_scale(
                int(chunk.shape[0]),
                int(chunk.shape[1]),
                adjust_lr_fn=adjust_lr_fn,
            )
            scaled_chunks.append(orthogonalized * scale)
        return (
            scaled_chunks[0]
            if len(scaled_chunks) == 1
            else torch.cat(scaled_chunks, dim=0)
        )

    def _step_muon(self, parameter: nn.Parameter, group: Mapping[str, Any]) -> None:
        gradient = parameter.grad
        if gradient is None:
            return
        if gradient.is_sparse:
            raise RuntimeError("HybridMuon does not support sparse gradients")
        compute_dtype = self._compute_dtype(gradient)
        grad = gradient.detach().to(dtype=compute_dtype)
        state = self.state[parameter]
        momentum_buffer = state.get("momentum_buffer")
        if momentum_buffer is None:
            momentum_buffer = torch.zeros_like(grad)
        elif momentum_buffer.dtype != compute_dtype:
            momentum_buffer = momentum_buffer.to(dtype=compute_dtype)
        momentum = float(group["momentum"])
        momentum_buffer.lerp_(grad, 1.0 - momentum)
        state["momentum_buffer"] = momentum_buffer
        state["step"] = int(state.get("step", 0)) + 1
        update = (
            grad.lerp(momentum_buffer, momentum)
            if bool(group["nesterov"])
            else momentum_buffer
        )
        update = self._orthogonalized_muon_update(parameter, update, group)
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        if weight_decay:
            parameter.mul_(1.0 - learning_rate * weight_decay)
        parameter.add_(
            update.to(device=parameter.device, dtype=parameter.dtype),
            alpha=-learning_rate,
        )

    def _step_adamw(self, parameter: nn.Parameter, group: Mapping[str, Any]) -> None:
        gradient = parameter.grad
        if gradient is None:
            return
        if gradient.is_sparse:
            raise RuntimeError(
                "HybridMuon AdamW branch does not support sparse gradients"
            )
        compute_dtype = self._compute_dtype(gradient)
        grad = gradient.detach().to(dtype=compute_dtype)
        state = self.state[parameter]
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None:
            exp_avg = torch.zeros_like(grad)
            exp_avg_sq = torch.zeros_like(grad)
        else:
            exp_avg = exp_avg.to(dtype=compute_dtype)
            exp_avg_sq = exp_avg_sq.to(dtype=compute_dtype)
        step = int(state.get("step", 0)) + 1
        beta1, beta2 = group["betas"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        state["step"] = step
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq

        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        if weight_decay:
            parameter.mul_(1.0 - learning_rate * weight_decay)
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2))
        denominator.add_(float(group["eps"]))
        update = exp_avg.div(denominator).mul_(-learning_rate / bias_correction1)
        parameter.add_(update.to(device=parameter.device, dtype=parameter.dtype))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            algorithm = group.get("algorithm")
            if algorithm not in {"muon", "adamw"}:
                raise RuntimeError(
                    f"Unknown HybridMuon parameter algorithm: {algorithm}"
                )
            for parameter in group["params"]:
                if algorithm == "muon":
                    self._step_muon(parameter, group)
                else:
                    self._step_adamw(parameter, group)
        return loss


def build_hybrid_muon_optimizer(
    module: nn.Module,
    *,
    adamw_lr: float,
    adamw_weight_decay: float,
    config: Mapping[str, Any],
) -> tuple[HybridMuon, OptimizerParameterAudit]:
    groups = partition_optimizer_parameters(
        module,
        min_dim=int(config["min_dim"]),
        max_aspect_ratio=float(config["max_aspect_ratio"]),
    )
    optimizer = HybridMuon(
        muon_parameters=groups.muon,
        muon_chunk_sizes=groups.muon_chunk_sizes,
        adamw_parameters=groups.adamw,
        muon_lr=float(config["lr"]),
        adamw_lr=float(adamw_lr),
        momentum=float(config["momentum"]),
        nesterov=bool(config["nesterov"]),
        ns_steps=int(config["ns_steps"]),
        muon_weight_decay=float(config["weight_decay"]),
        adamw_weight_decay=float(adamw_weight_decay),
        adamw_betas=config["adamw_betas"],
        adamw_eps=float(config["adamw_eps"]),
        adjust_lr_fn=str(config.get("adjust_lr_fn", "match_rms_adamw")),
    )
    return optimizer, groups.audit


__all__ = [
    "HybridMuon",
    "OptimizerParameterAudit",
    "OptimizerParameterGroups",
    "build_hybrid_muon_optimizer",
    "newton_schulz_orthogonalize",
    "partition_optimizer_parameters",
]
