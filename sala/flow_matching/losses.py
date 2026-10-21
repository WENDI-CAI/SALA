from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from sala.core.lattice import batched_inverse_3x3
from sala.core.lattice_manifold import q_to_lattice, require_lattice_template
from sala.core.orbit import OrbitPlan, build_orbit_plan
from sala.core.types import AsuBatch
from sala.flow_matching.lattice_prior import (
    LatticePriorBatchStats,
    VolumeLatticePrior,
)
from sala.flow_matching.types import FlowPathSample, FlowPrediction
from sala.postprocess.geometry.contact_statistics import element_vdw_radii


DEFAULT_TIME_BUCKET_EDGES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

_CONTACT_CLASH_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "weight": 0.0,
    "max_time": 0.25,
    "vdw_offset_A": 1.0,
    "contact_cutoff_A": 5.0,
    "contact_match_weight": 1.0,
    "huber_delta_A": 0.50,
    "predicted_lattice_mode": "predicted_clean",
    "pair_chunk_size": 65536,
    "checkpoint_pair_distances": False,
    "periodic_image_radius": 1,
    "periodic_fallback_max_candidates": 65536,
}
_CONTACT_CLASH_EXECUTION_KEYS = frozenset(
    {
        "pair_chunk_size",
        "checkpoint_pair_distances",
        "periodic_image_radius",
        "periodic_fallback_max_candidates",
    }
)


def resolve_contact_clash_config(
    config: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    raw = dict(config or {})
    unsupported = sorted(set(raw) - set(_CONTACT_CLASH_DEFAULTS))
    if unsupported:
        raise ValueError(f"unsupported contact/clash loss keys: {unsupported}")
    merged = {**_CONTACT_CLASH_DEFAULTS, **raw}
    resolved = {
        "enabled": bool(merged["enabled"]),
        "weight": float(merged["weight"]),
        "max_time": float(merged["max_time"]),
        "vdw_offset_A": float(merged["vdw_offset_A"]),
        "contact_cutoff_A": float(merged["contact_cutoff_A"]),
        "contact_match_weight": float(merged["contact_match_weight"]),
        "huber_delta_A": float(merged["huber_delta_A"]),
        "predicted_lattice_mode": str(
            merged["predicted_lattice_mode"]
        ).strip().lower(),
        "pair_chunk_size": int(merged["pair_chunk_size"]),
        "checkpoint_pair_distances": bool(merged["checkpoint_pair_distances"]),
        "periodic_image_radius": int(merged["periodic_image_radius"]),
        "periodic_fallback_max_candidates": int(
            merged["periodic_fallback_max_candidates"]
        ),
    }
    float_keys = (
        "weight",
        "max_time",
        "vdw_offset_A",
        "contact_cutoff_A",
        "contact_match_weight",
        "huber_delta_A",
    )
    if any(
        not math.isfinite(resolved[key]) or resolved[key] < 0.0
        for key in float_keys
    ):
        raise ValueError("contact/clash loss values must be finite and non-negative")
    if not 0.0 < resolved["max_time"] <= 1.0:
        raise ValueError("contact/clash max_time must be in (0, 1]")
    if resolved["contact_cutoff_A"] <= 0.0 or resolved["huber_delta_A"] <= 0.0:
        raise ValueError("contact cutoff and Huber delta must be positive")
    if resolved["predicted_lattice_mode"] not in {
        "predicted_clean",
        "input",
    }:
        raise ValueError(
            "contact/clash predicted_lattice_mode must be "
            "predicted_clean or input"
        )
    if resolved["pair_chunk_size"] <= 0:
        raise ValueError("contact/clash pair_chunk_size must be positive")
    if resolved["periodic_image_radius"] < 1:
        raise ValueError("contact periodic_image_radius must be positive")
    fast_candidate_count = (2 * resolved["periodic_image_radius"] + 1) ** 3
    if resolved["periodic_fallback_max_candidates"] < fast_candidate_count:
        raise ValueError(
            "contact periodic_fallback_max_candidates must cover the fast "
            "periodic image grid"
        )
    return resolved


def contact_clash_objective_signature(
    config: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    resolved = resolve_contact_clash_config(config)
    return {
        key: value
        for key, value in resolved.items()
        if key not in _CONTACT_CLASH_EXECUTION_KEYS
    }


_BOND_GEOMETRY_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "weight": 0.0,
    "max_time": 1.0,
    "residual_mode": "relative",
    "huber_delta": 0.10,
}


def resolve_bond_geometry_config(
    config: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    raw = dict(config or {})
    unsupported = sorted(set(raw) - set(_BOND_GEOMETRY_DEFAULTS))
    if unsupported:
        raise ValueError(f"unsupported bond geometry loss keys: {unsupported}")
    merged = {**_BOND_GEOMETRY_DEFAULTS, **raw}
    mode = str(merged["residual_mode"]).strip().lower()
    if mode not in {"relative", "absolute"}:
        raise ValueError(
            "bond_geometry.residual_mode must be 'relative' or 'absolute'"
        )
    weight = float(merged["weight"])
    max_time = float(merged["max_time"])
    delta = float(merged["huber_delta"])
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("bond_geometry.weight must be finite and non-negative")
    if not 0.0 < max_time <= 1.0:
        raise ValueError("bond_geometry.max_time must lie in (0, 1]")
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("bond_geometry.huber_delta must be finite and positive")
    return {
        "enabled": bool(merged["enabled"]),
        "weight": weight,
        "max_time": max_time,
        "residual_mode": mode,
        "huber_delta": delta,
    }


def bond_geometry_objective_signature(
    config: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    return resolve_bond_geometry_config(config)


@dataclass(frozen=True)
class FlowMatchingLossOutput:
    loss: Tensor
    metrics: Dict[str, Tensor]
    per_graph_loss: Tensor
    coord_per_graph_loss: Tensor
    lattice_per_graph_loss: Tensor


@dataclass(frozen=True)
class _BondGeometryLossOutput:
    loss: Tensor
    signed_mean: Tensor
    absolute_mean: Tensor
    relative_signed_mean: Tensor
    bond_count: Tensor


@dataclass(frozen=True)
class _ContactClashLossOutput:
    combined: Tensor
    clash: Tensor
    contact: Tensor
    clash_normalizer_pair_count: Tensor
    contact_pair_count: Tensor
    target_capped_pair_count: Tensor
    periodic_same_instance_count: Tensor
    predicted_clash_pair_count: Tensor
    predicted_periodic_fallback_count: Tensor
    target_periodic_fallback_count: Tensor
    evaluated_pair_count: Tensor


def _require_finite(value: Tensor, name: str) -> None:
    predicate = torch.isfinite(value).all().reshape(())
    if predicate.device.type == "cpu":
        if not bool(predicate.item()):
            raise FloatingPointError(f"Non-finite {name} in flow-matching loss")
        return
    assert_async = getattr(torch, "_assert_async", None)
    if assert_async is not None:
        assert_async(predicate, f"Non-finite {name} in flow-matching loss")
    else:  # pragma: no cover
        torch._assert(predicate, f"Non-finite {name} in flow-matching loss")


def _mean_atoms_per_graph(values: Tensor, batch: AsuBatch) -> Tensor:
    if values.shape != (int(batch.atom_types.shape[0]),):
        raise ValueError("atom loss must contain one scalar per ASU atom")
    output = values.new_zeros(batch.batch_size)
    output.index_add_(0, batch.batch.long(), values)
    counts = torch.bincount(batch.batch.long(), minlength=batch.batch_size).to(
        device=values.device, dtype=values.dtype
    )
    return output / counts.clamp_min(1.0)


def _time_bucket_metrics(
    name: str,
    graph_loss: Tensor,
    time: Tensor,
    edges: Sequence[float],
) -> Dict[str, Tensor]:
    metrics: Dict[str, Tensor] = {}
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        upper = time <= right if index == len(edges) - 2 else time < right
        mask = (time >= left) & upper
        weight = mask.to(dtype=graph_loss.dtype)
        count = weight.sum()
        mean = (graph_loss * weight).sum() / count.clamp_min(1.0)
        metrics[f"metric/{name}_time_bucket_{index}"] = mean.detach()
        metrics[f"metric/{name}_time_bucket_{index}_count"] = count.detach()
    return metrics


def _periodic_minimum_distances(
    source_frac: Tensor,
    target_frac: Tensor,
    lattice: Tensor,
    translations: Tensor,
) -> Tensor:
    delta = source_frac[:, None, None, :] - target_frac[None, :, None, :]
    # Symmetry operations can use integer-equivalent translation notation.
    # Centering first makes the finite image search invariant to that choice.
    delta = delta - torch.round(delta)
    delta = delta + translations[None, None, :, :]
    cart = torch.matmul(delta, lattice)
    return torch.linalg.vector_norm(cart, dim=-1).amin(dim=-1)


class FlowMatchingLoss(nn.Module):
    """Velocity matching with retained full-cell endpoint supervision."""

    def __init__(
        self,
        *,
        coord_velocity_weight: float = 1.0,
        coord_velocity_mode: str = "asu",
        lattice_velocity_weight: float = 1.0,
        full_cell_endpoint_weight: float = 0.25,
        coord_endpoint_pred_lattice_mode: str = "predicted_clean",
        lattice_endpoint_weight: float = 0.0,
        coord_scale: float = 8.0,
        lattice_normalize_by_dof: bool = True,
        contact_clash_config: Optional[Mapping[str, Any]] = None,
        bond_geometry_config: Optional[Mapping[str, Any]] = None,
        time_bucket_edges: Sequence[float] = DEFAULT_TIME_BUCKET_EDGES,
    ) -> None:
        super().__init__()
        self.coord_velocity_weight = float(coord_velocity_weight)
        self.coord_velocity_mode = str(coord_velocity_mode).strip().lower()
        if self.coord_velocity_mode not in {"asu", "full_cell_orbit"}:
            raise ValueError(
                "coord_velocity_mode must be 'asu' or 'full_cell_orbit'"
            )
        self.lattice_velocity_weight = float(lattice_velocity_weight)
        self.full_cell_endpoint_weight = float(full_cell_endpoint_weight)
        self.coord_endpoint_pred_lattice_mode = str(
            coord_endpoint_pred_lattice_mode
        ).strip().lower()
        if self.coord_endpoint_pred_lattice_mode not in {
            "predicted_clean",
            "input",
        }:
            raise ValueError(
                "coord_endpoint_pred_lattice_mode must be "
                "'predicted_clean' or 'input'"
            )
        self.lattice_endpoint_weight = float(lattice_endpoint_weight)
        self.coord_scale = float(coord_scale)
        self.lattice_normalize_by_dof = bool(lattice_normalize_by_dof)
        weights = (
            self.coord_velocity_weight,
            self.lattice_velocity_weight,
            self.full_cell_endpoint_weight,
            self.lattice_endpoint_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("flow loss weights must be finite and non-negative")
        if self.coord_scale <= 0.0 or not math.isfinite(self.coord_scale):
            raise ValueError("coord_scale must be finite and positive")
        contact_cfg = resolve_contact_clash_config(contact_clash_config)
        self.contact_clash_enabled = bool(contact_cfg.get("enabled", False))
        self.contact_clash_weight = float(contact_cfg.get("weight", 0.0))
        self.contact_clash_max_time = float(contact_cfg.get("max_time", 0.25))
        self.contact_vdw_offset_A = float(contact_cfg.get("vdw_offset_A", 1.0))
        self.contact_cutoff_A = float(contact_cfg.get("contact_cutoff_A", 5.0))
        self.contact_match_weight = float(
            contact_cfg.get("contact_match_weight", 1.0)
        )
        self.contact_huber_delta_A = float(
            contact_cfg.get("huber_delta_A", 0.50)
        )
        self.contact_predicted_lattice_mode = str(
            contact_cfg.get(
                "predicted_lattice_mode", "predicted_clean"
            )
        )
        self.contact_pair_chunk_size = int(
            contact_cfg.get("pair_chunk_size", 65536)
        )
        self.contact_checkpoint_pair_distances = bool(
            contact_cfg.get("checkpoint_pair_distances", False)
        )
        self.contact_periodic_image_radius = int(
            contact_cfg.get("periodic_image_radius", 1)
        )
        self.contact_periodic_fallback_max_candidates = int(
            contact_cfg.get("periodic_fallback_max_candidates", 65536)
        )
        radius_values, _ = element_vdw_radii()
        self.register_buffer(
            "_vdw_radii",
            torch.tensor(radius_values, dtype=torch.float32),
            persistent=False,
        )
        axis = torch.arange(
            -self.contact_periodic_image_radius,
            self.contact_periodic_image_radius + 1,
            dtype=torch.float32,
        )
        translations = torch.cartesian_prod(axis, axis, axis)
        if translations.ndim == 1:
            translations = translations.view(1, 3)
        self.register_buffer(
            "_contact_translations",
            translations,
            persistent=False,
        )
        bond_cfg = resolve_bond_geometry_config(bond_geometry_config)
        self.bond_geometry_enabled = bool(bond_cfg["enabled"])
        self.bond_geometry_weight = float(bond_cfg["weight"])
        self.bond_geometry_max_time = float(bond_cfg["max_time"])
        self.bond_geometry_residual_mode = str(bond_cfg["residual_mode"])
        self.bond_geometry_huber_delta = float(bond_cfg["huber_delta"])
        self.time_bucket_edges = tuple(float(edge) for edge in time_bucket_edges)
        if (
            len(self.time_bucket_edges) < 2
            or self.time_bucket_edges[0] != 0.0
            or self.time_bucket_edges[-1] != 1.0
            or any(
                right <= left
                for left, right in zip(
                    self.time_bucket_edges[:-1], self.time_bucket_edges[1:]
                )
            )
        ):
            raise ValueError(
                "time_bucket_edges must increase strictly from 0.0 to 1.0"
            )

    @staticmethod
    def _best_image_translation(
        left_frac: Tensor,
        right_frac: Tensor,
        lattice: Tensor,
        exclude_intramolecular_image: Tensor,
        translations: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raw_delta = left_frac - right_frac
        center = torch.round(raw_delta)
        centered = raw_delta - center
        candidates = centered[:, None, :] + translations[None, :, :]
        cart = torch.einsum("ptd,pdk->ptk", candidates, lattice.float())
        distance = torch.linalg.vector_norm(cart, dim=-1)
        intramolecular = exclude_intramolecular_image[:, None] & (
            translations[None, :, :] == center[:, None, :]
        ).all(dim=-1)
        distance = distance.masked_fill(intramolecular, torch.inf)
        best_index = distance.argmin(dim=-1)
        row = torch.arange(distance.shape[0], device=distance.device)
        return (
            translations.index_select(0, best_index),
            distance[row, best_index],
            centered,
        )

    def _image_pair_chunk_size(self, image_count: int) -> int:
        # Bound each image-search work unit to roughly two million candidates.
        # Candidate selection is detached, so this limits forward workspace while
        # the final differentiable distance keeps only one image per pair.
        return min(
            self.contact_pair_chunk_size,
            max(1, 2_000_000 // max(int(image_count), 1)),
        )

    def _select_periodic_translations(
        self,
        *,
        left_frac: Tensor,
        right_frac: Tensor,
        lattice: Tensor,
        reciprocal_column_norms: Tensor,
        exclude_intramolecular_image: Tensor,
    ) -> tuple[Tensor, Tensor]:
        fast_translations = self._contact_translations.to(
            device=left_frac.device, dtype=torch.float32
        )
        selected_parts: list[Tensor] = []
        complete_parts: list[Tensor] = []
        low_parts: list[Tensor] = []
        high_parts: list[Tensor] = []
        fast_low = fast_translations.amin(dim=0).long()
        fast_high = fast_translations.amax(dim=0).long()
        fast_chunk_size = self._image_pair_chunk_size(
            int(fast_translations.shape[0])
        )
        pair_count = int(left_frac.shape[0])
        with torch.no_grad():
            for start in range(0, pair_count, fast_chunk_size):
                stop = min(start + fast_chunk_size, pair_count)
                selected, best_distance, centered = self._best_image_translation(
                    left_frac[start:stop].detach().float(),
                    right_frac[start:stop].detach().float(),
                    lattice[start:stop].detach().float(),
                    exclude_intramolecular_image[start:stop],
                    fast_translations,
                )
                bounds = reciprocal_column_norms[start:stop].float() * (
                    best_distance[:, None] + 1.0e-6
                )
                low = torch.ceil(-centered - bounds - 1.0e-6).long()
                high = torch.floor(-centered + bounds + 1.0e-6).long()
                complete = ((low >= fast_low) & (high <= fast_high)).all(dim=-1)
                selected_parts.append(selected)
                complete_parts.append(complete)
                low_parts.append(low)
                high_parts.append(high)

            selected = torch.cat(selected_parts, dim=0)
            complete = torch.cat(complete_parts, dim=0)
            low = torch.cat(low_parts, dim=0)
            high = torch.cat(high_parts, dim=0)
            fallback_index = torch.nonzero(~complete, as_tuple=False).view(-1)
            if fallback_index.numel() == 0:
                return selected, ~complete

            fallback_indices = fallback_index.detach().cpu().tolist()
            fallback_low = low.index_select(0, fallback_index).detach().cpu().tolist()
            fallback_high = high.index_select(0, fallback_index).detach().cpu().tolist()
            grouped_pairs: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
            for fallback_position, (pair_index, pair_low, pair_high) in enumerate(
                zip(fallback_indices, fallback_low, fallback_high)
            ):
                radii = tuple(
                    max(abs(int(low_value)), abs(int(high_value)))
                    for low_value, high_value in zip(pair_low, pair_high)
                )
                grouped_pairs.setdefault(radii, []).append(
                    (fallback_position, pair_index)
                )

            fallback_selected = selected.new_empty((len(fallback_indices), 3))
            for radii, entries in grouped_pairs.items():
                candidate_count = math.prod(
                    2 * radius + 1 for radius in radii
                )
                if candidate_count > self.contact_periodic_fallback_max_candidates:
                    raise RuntimeError(
                        "contact/clash periodic closest-image fallback requires "
                        f"{candidate_count} candidates, exceeding "
                        "periodic_fallback_max_candidates="
                        f"{self.contact_periodic_fallback_max_candidates}"
                    )
                axes = [
                    torch.arange(
                        -radius,
                        radius + 1,
                        device=left_frac.device,
                        dtype=torch.float32,
                    )
                    for radius in radii
                ]
                translations = torch.cartesian_prod(*axes).reshape(-1, 3)
                group_positions = torch.tensor(
                    [position for position, _ in entries],
                    device=left_frac.device,
                    dtype=torch.long,
                )
                pair_indices = torch.tensor(
                    [pair_index for _, pair_index in entries],
                    device=left_frac.device,
                    dtype=torch.long,
                )
                group_chunk_size = self._image_pair_chunk_size(candidate_count)
                group_selected_parts: list[Tensor] = []
                for start in range(0, int(pair_indices.shape[0]), group_chunk_size):
                    stop = min(
                        start + group_chunk_size, int(pair_indices.shape[0])
                    )
                    index = pair_indices[start:stop]
                    group_selected, _, _ = self._best_image_translation(
                        left_frac.index_select(0, index).detach().float(),
                        right_frac.index_select(0, index).detach().float(),
                        lattice.index_select(0, index).detach().float(),
                        exclude_intramolecular_image.index_select(0, index),
                        translations,
                    )
                    group_selected_parts.append(group_selected)
                fallback_selected.index_copy_(
                    0,
                    group_positions,
                    torch.cat(group_selected_parts, dim=0),
                )
            selected.index_copy_(
                0,
                fallback_index,
                fallback_selected,
            )
            return selected, ~complete

    @staticmethod
    def _distance_for_selected_translation(
        left_frac: Tensor,
        right_frac: Tensor,
        lattice: Tensor,
        selected_translation: Tensor,
    ) -> Tensor:
        raw_delta = left_frac - right_frac
        centered = raw_delta - torch.round(raw_delta)
        cart = torch.einsum(
            "pd,pdk->pk",
            centered + selected_translation,
            lattice.float(),
        )
        return torch.linalg.vector_norm(cart, dim=-1)

    @staticmethod
    def _distance_for_selected_pairs(
        frac: Tensor,
        lattice: Tensor,
        graph_index: Tensor,
        left_index: Tensor,
        right_index: Tensor,
        selected_translation: Tensor,
    ) -> Tensor:
        return FlowMatchingLoss._distance_for_selected_translation(
            frac[graph_index, left_index].float(),
            frac[graph_index, right_index].float(),
            lattice.index_select(0, graph_index).float(),
            selected_translation,
        )

    def _paired_periodic_distances(
        self,
        *,
        pred_frac: Tensor,
        true_frac: Tensor,
        pred_lattice: Tensor,
        true_lattice: Tensor,
        graph_index: Tensor,
        left_index: Tensor,
        right_index: Tensor,
        exclude_intramolecular_image: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pred_parts: list[Tensor] = []
        true_parts: list[Tensor] = []
        with torch.no_grad():
            pred_reciprocal_norms = torch.linalg.vector_norm(
                torch.linalg.inv(pred_lattice.detach().float()), dim=-2
            )
            true_reciprocal_norms = torch.linalg.vector_norm(
                torch.linalg.inv(true_lattice.detach().float()), dim=-2
            )
            pred_search_left = pred_frac.detach()[graph_index, left_index].float()
            pred_search_right = pred_frac.detach()[graph_index, right_index].float()
            pred_search_lattice = pred_lattice.detach().index_select(
                0, graph_index
            ).float()
            pred_translation, pred_fallback = self._select_periodic_translations(
                left_frac=pred_search_left,
                right_frac=pred_search_right,
                lattice=pred_search_lattice,
                reciprocal_column_norms=pred_reciprocal_norms.index_select(
                    0, graph_index
                ),
                exclude_intramolecular_image=exclude_intramolecular_image,
            )
            del pred_search_left, pred_search_right, pred_search_lattice

            true_search_left = true_frac.detach()[graph_index, left_index].float()
            true_search_right = true_frac.detach()[graph_index, right_index].float()
            true_search_lattice = true_lattice.detach().index_select(
                0, graph_index
            ).float()
            true_translation, true_fallback = self._select_periodic_translations(
                left_frac=true_search_left,
                right_frac=true_search_right,
                lattice=true_search_lattice,
                reciprocal_column_norms=true_reciprocal_norms.index_select(
                    0, graph_index
                ),
                exclude_intramolecular_image=exclude_intramolecular_image,
            )
            del true_search_left, true_search_right, true_search_lattice
        pair_count = int(graph_index.shape[0])
        for start in range(0, pair_count, self.contact_pair_chunk_size):
            stop = min(start + self.contact_pair_chunk_size, pair_count)
            graph = graph_index[start:stop]
            left = left_index[start:stop]
            right = right_index[start:stop]
            if (
                self.contact_checkpoint_pair_distances
                and torch.is_grad_enabled()
                and (pred_frac.requires_grad or pred_lattice.requires_grad)
            ):
                pred_distance = checkpoint(
                    self._distance_for_selected_pairs,
                    pred_frac,
                    pred_lattice,
                    graph,
                    left,
                    right,
                    pred_translation[start:stop],
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                pred_distance = self._distance_for_selected_pairs(
                    pred_frac,
                    pred_lattice,
                    graph,
                    left,
                    right,
                    pred_translation[start:stop],
                )
            with torch.no_grad():
                true_distance = self._distance_for_selected_pairs(
                    true_frac,
                    true_lattice,
                    graph,
                    left,
                    right,
                    true_translation[start:stop],
                )
            pred_parts.append(pred_distance)
            true_parts.append(true_distance)
        return (
            torch.cat(pred_parts, dim=0),
            torch.cat(true_parts, dim=0),
            pred_fallback,
            true_fallback,
        )

    def _contact_clash_losses(
        self,
        *,
        batch: AsuBatch,
        sample: FlowPathSample,
        plan: OrbitPlan,
        pred_full_frac: Tensor,
        true_full_frac: Tensor,
        pred_lattice: Tensor,
        true_lattice: Tensor,
    ) -> _ContactClashLossOutput:
        graph_zeros = pred_full_frac.sum(dim=(1, 2)).float() * 0.0
        count_zeros = graph_zeros.detach().clone()
        if not self.contact_clash_enabled or self.contact_clash_weight == 0.0:
            return _ContactClashLossOutput(
                combined=graph_zeros,
                clash=graph_zeros.clone(),
                contact=graph_zeros.clone(),
                clash_normalizer_pair_count=count_zeros,
                contact_pair_count=count_zeros.clone(),
                target_capped_pair_count=count_zeros.clone(),
                periodic_same_instance_count=count_zeros.clone(),
                predicted_clash_pair_count=count_zeros.clone(),
                predicted_periodic_fallback_count=count_zeros.clone(),
                target_periodic_fallback_count=count_zeros.clone(),
                evaluated_pair_count=count_zeros.clone(),
            )

        radii = self._vdw_radii.to(device=pred_full_frac.device, dtype=torch.float32)
        active = sample.time.float() <= self.contact_clash_max_time
        active_graph_index = torch.nonzero(active, as_tuple=False).view(-1)
        if active_graph_index.numel() == 0:
            return _ContactClashLossOutput(
                combined=graph_zeros,
                clash=graph_zeros.clone(),
                contact=graph_zeros.clone(),
                clash_normalizer_pair_count=count_zeros,
                contact_pair_count=count_zeros.clone(),
                target_capped_pair_count=count_zeros.clone(),
                periodic_same_instance_count=count_zeros.clone(),
                predicted_clash_pair_count=count_zeros.clone(),
                predicted_periodic_fallback_count=count_zeros.clone(),
                target_periodic_fallback_count=count_zeros.clone(),
                evaluated_pair_count=count_zeros.clone(),
            )
        pair_template = torch.triu_indices(
            plan.max_full_atoms,
            plan.max_full_atoms,
            offset=0,
            device=pred_full_frac.device,
        )
        pairs_per_graph = int(pair_template.shape[1])
        active_graph_count = int(active_graph_index.shape[0])
        graph_all = active_graph_index.repeat_interleave(pairs_per_graph)
        left_all = pair_template[0].repeat(active_graph_count)
        right_all = pair_template[1].repeat(active_graph_count)
        full_counts = plan.full_counts.to(device=pred_full_frac.device)
        pair_valid = (
            left_all < full_counts.index_select(0, graph_all)
        ) & (
            right_all < full_counts.index_select(0, graph_all)
        )
        source = plan.full_to_asu.to(device=pred_full_frac.device).clamp_min(0)
        left_source_all = source[graph_all, left_all]
        right_source_all = source[graph_all, right_all]
        atom_types = batch.atom_types.to(device=pred_full_frac.device).long()
        pair_valid = pair_valid & (
            atom_types.index_select(0, left_source_all) > 1
        ) & (
            atom_types.index_select(0, right_source_all) > 1
        )
        active_pair_index = torch.nonzero(pair_valid, as_tuple=False).view(-1)
        if active_pair_index.numel() == 0:
            return _ContactClashLossOutput(
                combined=graph_zeros,
                clash=graph_zeros.clone(),
                contact=graph_zeros.clone(),
                clash_normalizer_pair_count=count_zeros,
                contact_pair_count=count_zeros.clone(),
                target_capped_pair_count=count_zeros.clone(),
                periodic_same_instance_count=count_zeros.clone(),
                predicted_clash_pair_count=count_zeros.clone(),
                predicted_periodic_fallback_count=count_zeros.clone(),
                target_periodic_fallback_count=count_zeros.clone(),
                evaluated_pair_count=count_zeros.clone(),
            )
        graph_index = graph_all.index_select(0, active_pair_index)
        left_index = left_all.index_select(0, active_pair_index)
        right_index = right_all.index_select(0, active_pair_index)
        component_instance = plan.full_component_instance.to(
            device=pred_full_frac.device
        )
        periodic_same = component_instance[graph_index, left_index].eq(
            component_instance[graph_index, right_index]
        )
        (
            pred_distance,
            true_distance,
            pred_periodic_fallback,
            true_periodic_fallback,
        ) = self._paired_periodic_distances(
            pred_frac=pred_full_frac.float(),
            true_frac=true_full_frac.float(),
            pred_lattice=pred_lattice,
            true_lattice=true_lattice,
            graph_index=graph_index,
            left_index=left_index,
            right_index=right_index,
            exclude_intramolecular_image=periodic_same,
        )
        left_source = source[graph_index, left_index]
        right_source = source[graph_index, right_index]
        left_type = atom_types.index_select(0, left_source).clamp(
            0, radii.numel() - 1
        )
        right_type = atom_types.index_select(0, right_source).clamp(
            0, radii.numel() - 1
        )
        vdw_floor = (
            radii.index_select(0, left_type)
            + radii.index_select(0, right_type)
            - self.contact_vdw_offset_A
        ).clamp_min(0.0)
        target_capped = true_distance.detach() < vdw_floor
        clash_floor = torch.minimum(vdw_floor, true_distance.detach())
        penetration = torch.relu(clash_floor - pred_distance)
        contact_mask = true_distance.detach() <= self.contact_cutoff_A
        clash_normalizer_mask = contact_mask | target_capped
        contact_error = (pred_distance - true_distance).abs()
        quadratic = contact_error.clamp_max(self.contact_huber_delta_A)
        contact_value = (
            0.5 * quadratic.square()
            + self.contact_huber_delta_A * (contact_error - quadratic)
        )

        clash_sum = graph_zeros.clone()
        contact_sum = graph_zeros.clone()
        clash_normalizer_count_graph = count_zeros.clone()
        contact_count_graph = count_zeros.clone()
        target_capped_count_graph = count_zeros.clone()
        periodic_count_graph = count_zeros.clone()
        predicted_clash_count_graph = count_zeros.clone()
        predicted_fallback_count_graph = count_zeros.clone()
        target_fallback_count_graph = count_zeros.clone()
        evaluated_pair_count_graph = count_zeros.clone()
        clash_sum.index_add_(0, graph_index, penetration.square())
        contact_sum.index_add_(
            0, graph_index, contact_value * contact_mask.to(contact_value.dtype)
        )
        clash_normalizer_count_graph.index_add_(
            0, graph_index, clash_normalizer_mask.float()
        )
        contact_count_graph.index_add_(0, graph_index, contact_mask.float())
        target_capped_count_graph.index_add_(
            0, graph_index, target_capped.float()
        )
        periodic_count_graph.index_add_(
            0, graph_index, (periodic_same & clash_normalizer_mask).float()
        )
        predicted_clash_count_graph.index_add_(
            0, graph_index, (penetration.detach() > 0.0).float()
        )
        predicted_fallback_count_graph.index_add_(
            0, graph_index, pred_periodic_fallback.float()
        )
        target_fallback_count_graph.index_add_(
            0, graph_index, true_periodic_fallback.float()
        )
        evaluated_pair_count_graph.index_add_(
            0, graph_index, torch.ones_like(pred_distance, dtype=torch.float32)
        )
        clash_graph = clash_sum / clash_normalizer_count_graph.clamp_min(1.0)
        contact_graph = contact_sum / contact_count_graph.clamp_min(1.0)
        combined = clash_graph + self.contact_match_weight * contact_graph
        return _ContactClashLossOutput(
            combined=combined,
            clash=clash_graph,
            contact=contact_graph,
            clash_normalizer_pair_count=clash_normalizer_count_graph.detach(),
            contact_pair_count=contact_count_graph.detach(),
            target_capped_pair_count=target_capped_count_graph.detach(),
            periodic_same_instance_count=periodic_count_graph.detach(),
            predicted_clash_pair_count=predicted_clash_count_graph.detach(),
            predicted_periodic_fallback_count=(
                predicted_fallback_count_graph.detach()
            ),
            target_periodic_fallback_count=target_fallback_count_graph.detach(),
            evaluated_pair_count=evaluated_pair_count_graph.detach(),
        )

    def _bond_geometry_losses(
        self,
        *,
        batch: AsuBatch,
        pred_asu_cart: Tensor,
        true_asu_cart: Tensor,
        time: Tensor,
    ) -> _BondGeometryLossOutput:
        """Supervise ASU covalent bond lengths against the reference geometry.

        The endpoint objective is an L2 regression, so its minimizer is the
        posterior mean of the clean coordinates.  Averaging over plausible
        conformations shrinks every interatomic distance, and measurements on
        generated crystals show the effect concentrated in covalent geometry:
        heavy-atom bonds come out systematically ~4.5% short.  That bias is
        nearly invisible to superposition RMSD, because a near-isotropic
        contraction barely changes molecular shape, yet it is very expensive
        under an interatomic potential and forces long relaxations.

        Bond lengths are the one part of the clean endpoint that is essentially
        determined by the molecular graph rather than by the packing, so they are
        predictable at every noise level.  The term is therefore applied across
        the whole time range by default rather than only near the data end.

        ASU fractional coordinates are canonicalized to be contiguous along the
        bond graph, so a bond vector is a plain Cartesian difference and needs no
        periodic image search.  ``edge_index_conv`` stores both directions of each
        bond; counting a bond twice leaves every mean unchanged.
        """

        edge_index = batch.edge_index_conv.long()
        batch_size = int(batch.batch_size)
        device = pred_asu_cart.device
        zeros = torch.zeros(batch_size, device=device, dtype=torch.float32)
        if edge_index.numel() == 0:
            return _BondGeometryLossOutput(
                loss=zeros,
                signed_mean=zeros,
                absolute_mean=zeros,
                relative_signed_mean=zeros,
                bond_count=zeros,
            )

        source = edge_index[0]
        target = edge_index[1]
        pred = pred_asu_cart.float()
        true = true_asu_cart.float()
        pred_length = torch.linalg.vector_norm(
            pred.index_select(0, source) - pred.index_select(0, target), dim=-1
        )
        true_length = torch.linalg.vector_norm(
            true.index_select(0, source) - true.index_select(0, target), dim=-1
        )
        signed = pred_length - true_length
        safe_true = true_length.clamp_min(1.0e-3)
        relative = signed / safe_true
        residual = (
            relative
            if self.bond_geometry_residual_mode == "relative"
            else signed
        )
        # Huber keeps a rare pathological bond from dominating the batch while
        # still penalizing a small systematic offset quadratically.
        loss_per_bond = torch.nn.functional.huber_loss(
            residual,
            torch.zeros_like(residual),
            reduction="none",
            delta=self.bond_geometry_huber_delta,
        )

        edge_graph = batch.batch.long().index_select(0, source)
        counts = torch.zeros(batch_size, device=device, dtype=torch.float32)
        counts.index_add_(0, edge_graph, torch.ones_like(loss_per_bond))
        denominator = counts.clamp_min(1.0)

        def graph_mean(values: Tensor) -> Tensor:
            total = torch.zeros(batch_size, device=device, dtype=torch.float32)
            total.index_add_(0, edge_graph, values)
            return total / denominator

        loss_graph = graph_mean(loss_per_bond)
        if self.bond_geometry_max_time < 1.0:
            active = (
                time.float() <= self.bond_geometry_max_time
            ).to(dtype=loss_graph.dtype)
            loss_graph = loss_graph * active
        return _BondGeometryLossOutput(
            loss=loss_graph,
            signed_mean=graph_mean(signed.detach()),
            absolute_mean=graph_mean(signed.detach().abs()),
            relative_signed_mean=graph_mean(relative.detach()),
            bond_count=counts,
        )

    def forward(
        self,
        *,
        batch: AsuBatch,
        sample: FlowPathSample,
        prediction: FlowPrediction,
        lattice_stats: LatticePriorBatchStats,
        orbit_plan: Optional[OrbitPlan] = None,
        uncertainty_logvar: Optional[Tensor] = None,
    ) -> FlowMatchingLossOutput:
        _require_finite(prediction.velocity_x, "coordinate velocity")
        _require_finite(prediction.velocity_q, "lattice velocity")
        _require_finite(prediction.clean_x, "predicted clean coordinates")
        _require_finite(prediction.clean_q, "predicted clean lattice")

        coord_velocity_atom = (
            prediction.velocity_x.float() - sample.target_velocity_x.float()
        ).square().mean(dim=-1)
        coord_velocity_asu_graph = _mean_atoms_per_graph(
            coord_velocity_atom, batch
        )

        mask = lattice_stats.mask.to(
            device=prediction.velocity_q.device, dtype=prediction.velocity_q.dtype
        )
        lattice_velocity_sq = (
            prediction.velocity_q.float() - sample.target_velocity_q.float()
        ).square() * mask.float()
        if self.lattice_normalize_by_dof:
            lattice_velocity_graph = lattice_velocity_sq.sum(dim=-1) / mask.float().sum(
                dim=-1
            ).clamp_min(1.0)
        else:
            lattice_velocity_graph = lattice_velocity_sq.mean(dim=-1)

        template_id = require_lattice_template(
            batch.spacegroup, batch.lattice_template_id
        )
        plan = orbit_plan or build_orbit_plan(
            batch, validate_general_position=True
        )
        valid = plan.full_valid.to(device=prediction.velocity_x.device)

        # The full-cell atom head is orbit-pulled back by the denoiser before it
        # reaches FlowPrediction. Re-expand that projected ASU tangent, together
        # with the ASU target tangent, and evaluate the velocity objective over
        # every valid symmetry image in the current noisy cell.
        state_q = VolumeLatticePrior.unwhiten_with_stats(
            sample.state_q.float(), lattice_stats
        )
        state_lattice = q_to_lattice(
            state_q,
            spacegroup=batch.spacegroup,
            lattice_template_id=template_id,
        )
        # Each lattice is inverted once and shared with every orbit operator
        # below; the operators would otherwise each run their own inversion.
        state_lattice_inverse = batched_inverse_3x3(state_lattice)
        pred_full_velocity = plan.expand_vectors(
            prediction.velocity_x.float(),
            state_lattice,
            lattice_inverse=state_lattice_inverse,
        )
        target_full_velocity = plan.expand_vectors(
            sample.target_velocity_x.float(),
            state_lattice,
            lattice_inverse=state_lattice_inverse,
        )
        coord_velocity_full_sq = (
            pred_full_velocity - target_full_velocity
        ).square().sum(dim=-1)
        coord_velocity_full_graph = (
            coord_velocity_full_sq * valid
        ).sum(dim=-1) / (valid.sum(dim=-1).clamp_min(1) * 3)
        coord_velocity_graph = (
            coord_velocity_full_graph
            if self.coord_velocity_mode == "full_cell_orbit"
            else coord_velocity_asu_graph
        )

        pred_lattice = q_to_lattice(
            prediction.clean_q.float(),
            spacegroup=batch.spacegroup,
            lattice_template_id=template_id,
        )
        true_lattice = q_to_lattice(
            sample.clean_q.float(),
            spacegroup=batch.spacegroup,
            lattice_template_id=template_id,
        )
        pred_lattice_inverse = batched_inverse_3x3(pred_lattice)
        true_lattice_inverse = batched_inverse_3x3(true_lattice)
        use_state_endpoint_lattice = (
            self.coord_endpoint_pred_lattice_mode == "input"
        )
        pred_endpoint_lattice = (
            state_lattice if use_state_endpoint_lattice else pred_lattice
        )
        pred_endpoint_lattice_inverse = (
            state_lattice_inverse
            if use_state_endpoint_lattice
            else pred_lattice_inverse
        )
        pred_endpoint_geometry = plan.expand_coordinates(
            prediction.clean_x.float(),
            pred_endpoint_lattice,
            lattice_inverse=pred_endpoint_lattice_inverse,
        )
        true_endpoint_geometry = plan.expand_coordinates(
            sample.clean_x.float(),
            true_lattice,
            lattice_inverse=true_lattice_inverse,
        )
        pred_full = pred_endpoint_geometry.cart
        true_full = true_endpoint_geometry.cart
        coord_endpoint_sq = (pred_full - true_full).square().sum(dim=-1)
        coord_endpoint_graph = (coord_endpoint_sq * valid).sum(dim=-1) / (
            valid.sum(dim=-1).clamp_min(1) * 3 * self.coord_scale**2
        )
        use_state_contact_lattice = (
            self.contact_predicted_lattice_mode == "input"
        )
        pred_contact_lattice = (
            state_lattice if use_state_contact_lattice else pred_lattice
        )
        pred_contact_geometry = plan.expand_coordinates(
            prediction.clean_x.float(),
            pred_contact_lattice,
            lattice_inverse=(
                state_lattice_inverse
                if use_state_contact_lattice
                else pred_lattice_inverse
            ),
        )
        contact_losses = self._contact_clash_losses(
            batch=batch,
            sample=sample,
            plan=plan,
            pred_full_frac=pred_contact_geometry.frac,
            true_full_frac=true_endpoint_geometry.frac,
            pred_lattice=pred_contact_lattice,
            true_lattice=true_lattice,
        )
        contact_clash_graph = contact_losses.combined
        clash_graph = contact_losses.clash
        contact_match_graph = contact_losses.contact

        bond_losses = self._bond_geometry_losses(
            batch=batch,
            pred_asu_cart=prediction.clean_x,
            true_asu_cart=sample.clean_x,
            time=sample.time,
        )
        bond_geometry_graph = bond_losses.loss

        lattice_endpoint_sq = (
            prediction.clean_state_q.float() - sample.clean_state_q.float()
        ).square() * mask.float()
        if self.lattice_normalize_by_dof:
            lattice_endpoint_graph = lattice_endpoint_sq.sum(dim=-1) / mask.float().sum(
                dim=-1
            ).clamp_min(1.0)
        else:
            lattice_endpoint_graph = lattice_endpoint_sq.mean(dim=-1)

        coord_graph = (
            self.coord_velocity_weight * coord_velocity_graph
            + self.full_cell_endpoint_weight * coord_endpoint_graph
            + self.contact_clash_weight * contact_clash_graph
            + self.bond_geometry_weight * bond_geometry_graph
        )
        lattice_graph = (
            self.lattice_velocity_weight * lattice_velocity_graph
            + self.lattice_endpoint_weight * lattice_endpoint_graph
        )
        if uncertainty_logvar is None:
            per_graph = coord_graph + lattice_graph
        else:
            if uncertainty_logvar.shape != (batch.batch_size, 2):
                raise ValueError("uncertainty_logvar must have shape [B, 2]")
            logvar = uncertainty_logvar.to(
                device=coord_graph.device, dtype=coord_graph.dtype
            )
            coord_logvar, lattice_logvar = logvar.unbind(dim=-1)
            per_graph = (
                torch.exp(-coord_logvar) * coord_graph
                + coord_logvar
                + torch.exp(-lattice_logvar) * lattice_graph
                + lattice_logvar
            )
        total = per_graph.mean()
        _require_finite(total, "total loss")

        pred_lengths = torch.linalg.vector_norm(pred_lattice.detach(), dim=-1)
        true_lengths = torch.linalg.vector_norm(true_lattice.detach(), dim=-1)
        pair_index = ((0, 1), (0, 2), (1, 2))
        pred_angles = torch.stack(
            [
                torch.acos(
                    (
                        (pred_lattice.detach()[:, i] * pred_lattice.detach()[:, j]).sum(-1)
                        / (pred_lengths[:, i] * pred_lengths[:, j]).clamp_min(1.0e-12)
                    ).clamp(-1.0, 1.0)
                )
                for i, j in pair_index
            ],
            dim=-1,
        )
        true_angles = torch.stack(
            [
                torch.acos(
                    (
                        (true_lattice.detach()[:, i] * true_lattice.detach()[:, j]).sum(-1)
                        / (true_lengths[:, i] * true_lengths[:, j]).clamp_min(1.0e-12)
                    ).clamp(-1.0, 1.0)
                )
                for i, j in pair_index
            ],
            dim=-1,
        )
        pred_volume = torch.linalg.det(pred_lattice.detach()).abs()
        true_volume = torch.linalg.det(true_lattice.detach()).abs()
        metrics: Dict[str, Tensor] = {
            "loss/total": total.detach(),
            "loss/coord": coord_graph.mean().detach(),
            "loss/lattice": lattice_graph.mean().detach(),
            "loss/coord_velocity": coord_velocity_graph.mean().detach(),
            "loss/lattice_velocity": lattice_velocity_graph.mean().detach(),
            "loss/full_cell_endpoint": coord_endpoint_graph.mean().detach(),
            "loss/lattice_endpoint": lattice_endpoint_graph.mean().detach(),
            "metric/coord_velocity_asu_mse": (
                coord_velocity_asu_graph.mean().detach()
            ),
            "metric/coord_velocity_full_cell_mse": (
                coord_velocity_full_graph.mean().detach()
            ),
            "metric/full_cell_velocity_x_pred_rms": (
                (pred_full_velocity.square().sum(dim=-1) * valid).sum()
                / (valid.sum().clamp_min(1) * 3)
            ).sqrt().detach(),
            "metric/full_cell_velocity_x_target_rms": (
                (target_full_velocity.square().sum(dim=-1) * valid).sum()
                / (valid.sum().clamp_min(1) * 3)
            ).sqrt().detach(),
            "loss/bond_geometry": bond_geometry_graph.mean().detach(),
            "loss/bond_geometry_weighted": (
                self.bond_geometry_weight * bond_geometry_graph.mean()
            ).detach(),
            # The signed means are the diagnostic that motivated this term: a
            # negative value is systematic bond contraction in the predicted
            # endpoint.  Both should trend towards zero during training.
            "metric/bond_length_signed_mean_A": (
                bond_losses.signed_mean.mean().detach()
            ),
            "metric/bond_length_relative_signed_mean": (
                bond_losses.relative_signed_mean.mean().detach()
            ),
            "metric/bond_length_abs_mean_A": (
                bond_losses.absolute_mean.mean().detach()
            ),
            "metric/bond_count_mean": bond_losses.bond_count.mean().detach(),
            "loss/contact_clash": contact_clash_graph.mean().detach(),
            "loss/clash_repulsion": clash_graph.mean().detach(),
            "loss/contact_match": contact_match_graph.mean().detach(),
            "loss/contact_clash_weighted": (
                self.contact_clash_weight * contact_clash_graph.mean()
            ).detach(),
            "metric/contact_active_graph_fraction": (
                sample.time.float() <= self.contact_clash_max_time
            ).float().mean().detach(),
            "metric/clash_normalizer_pair_count_mean": (
                contact_losses.clash_normalizer_pair_count.mean().detach()
            ),
            "metric/contact_target_pair_count_mean": (
                contact_losses.contact_pair_count.mean().detach()
            ),
            "metric/predicted_clash_pair_count_mean": (
                contact_losses.predicted_clash_pair_count.mean().detach()
            ),
            "metric/contact_predicted_periodic_fallback_fraction": (
                contact_losses.predicted_periodic_fallback_count.sum()
                / contact_losses.evaluated_pair_count.sum().clamp_min(1.0)
            ).detach(),
            "metric/contact_target_periodic_fallback_fraction": (
                contact_losses.target_periodic_fallback_count.sum()
                / contact_losses.evaluated_pair_count.sum().clamp_min(1.0)
            ).detach(),
            "metric/contact_evaluated_pair_count_mean": (
                contact_losses.evaluated_pair_count.mean().detach()
            ),
            "metric/contact_predicted_periodic_fallback_count_mean": (
                contact_losses.predicted_periodic_fallback_count.mean().detach()
            ),
            "metric/contact_target_periodic_fallback_count_mean": (
                contact_losses.target_periodic_fallback_count.mean().detach()
            ),
            "metric/clash_target_capped_pair_fraction": (
                contact_losses.target_capped_pair_count.sum()
                / contact_losses.evaluated_pair_count.sum().clamp_min(1.0)
            ).detach(),
            "metric/clash_target_capped_pair_count_mean": (
                contact_losses.target_capped_pair_count.mean().detach()
            ),
            "metric/contact_periodic_same_instance_fraction": (
                contact_losses.periodic_same_instance_count.sum()
                / contact_losses.clash_normalizer_pair_count.sum().clamp_min(1.0)
            ).detach(),
            "metric/contact_periodic_same_instance_count_mean": (
                contact_losses.periodic_same_instance_count.mean().detach()
            ),
            "metric/time_mean": sample.time.float().mean().detach(),
            "metric/time_min": sample.time.float().min().detach(),
            "metric/time_max": sample.time.float().max().detach(),
            "metric/velocity_x_pred_rms": prediction.velocity_x.float().square().mean().sqrt().detach(),
            "metric/velocity_x_target_rms": sample.target_velocity_x.float().square().mean().sqrt().detach(),
            "metric/velocity_q_pred_rms": prediction.velocity_q.float().square().mean().sqrt().detach(),
            "metric/velocity_q_target_rms": sample.target_velocity_q.float().square().mean().sqrt().detach(),
            "metric/full_cell_endpoint_rms_angstrom": (
                (coord_endpoint_sq * valid).sum()
                / (valid.sum().clamp_min(1) * 3)
            ).sqrt().detach(),
            "metric/lattice_q_endpoint_rmse_whitened": lattice_endpoint_graph.mean().sqrt().detach(),
            "metric/lattice_length_mae_angstrom": (pred_lengths - true_lengths).abs().mean().detach(),
            "metric/lattice_angle_mae_degree": (
                (pred_angles - true_angles).abs().mean() * (180.0 / math.pi)
            ).detach(),
            "metric/lattice_relative_volume_error": (
                (pred_volume - true_volume).abs() / true_volume.clamp_min(1.0e-12)
            ).mean().detach(),
        }
        metrics.update(
            _time_bucket_metrics(
                "coord_velocity_mse",
                coord_velocity_graph.detach(),
                sample.time,
                self.time_bucket_edges,
            )
        )
        metrics.update(
            _time_bucket_metrics(
                "lattice_velocity_mse",
                lattice_velocity_graph.detach(),
                sample.time,
                self.time_bucket_edges,
            )
        )
        if uncertainty_logvar is not None:
            metrics["metric/uncertainty_coord_logvar_mean"] = (
                uncertainty_logvar[:, 0].mean().detach()
            )
            metrics["metric/uncertainty_lattice_logvar_mean"] = (
                uncertainty_logvar[:, 1].mean().detach()
            )
        return FlowMatchingLossOutput(
            loss=total,
            metrics=metrics,
            per_graph_loss=per_graph.detach(),
            coord_per_graph_loss=coord_graph.detach(),
            lattice_per_graph_loss=lattice_graph.detach(),
        )


__all__ = [
    "DEFAULT_TIME_BUCKET_EDGES",
    "FlowMatchingLoss",
    "FlowMatchingLossOutput",
    "bond_geometry_objective_signature",
    "resolve_bond_geometry_config",
]
