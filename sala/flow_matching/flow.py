from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from sala.core.batch_ops import GraphSubBatchMap, select_graph_subbatch
from sala.core.lattice import frac_to_cart_coords_batched
from sala.core.lattice_manifold import lattice_to_q, q_to_lattice, require_lattice_template
from sala.core.types import AsuBatch
from sala.flow_matching.lattice_prior import LatticePriorBatchStats, VolumeLatticePrior
from sala.flow_matching.losses import DEFAULT_TIME_BUCKET_EDGES, FlowMatchingLoss
from sala.flow_matching.path import RectifiedFlowPath, flow_time_condition, sample_logit_normal_times
from sala.flow_matching.self_conditioning import (
    scatter_flow_self_conditioning_state,
)
from sala.flow_matching.types import (
    FlowPrediction,
    FlowSelfConditioningState,
    FlowTrainingOutput,
)
from sala.models.common import TimeEmbedding, zero_module


def _autocast_without_weight_cache(device_type: str):
    """Preserve outer autocast dtype without caching no-grad weight casts."""
    if not torch.is_autocast_enabled(device_type):
        return nullcontext()
    return torch.autocast(
        device_type=device_type,
        dtype=torch.get_autocast_dtype(device_type),
        cache_enabled=False,
    )


def _select_lattice_stats(
    stats: LatticePriorBatchStats, index_map: GraphSubBatchMap
) -> LatticePriorBatchStats:
    return LatticePriorBatchStats(
        mean=index_map.select_graphs(stats.mean),
        std=index_map.select_graphs(stats.std),
        mask=index_map.select_graphs(stats.mask),
        bias=index_map.select_graphs(stats.bias),
    )


MODEL_FAMILY = "sala_full_cell_crystal_dit_rectified_flow"


class FlowUncertaintyWeighting(nn.Module):
    """Flow-time-only coordinate/lattice uncertainty weighting."""

    def __init__(
        self,
        *,
        channels: int = 128,
        clamp: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError("uncertainty channels must be positive")
        low, high = float(clamp[0]), float(clamp[1])
        if not low < high:
            raise ValueError("uncertainty clamp must satisfy low < high")
        self.low = low
        self.high = high
        self.embedding = TimeEmbedding(
            d_model=channels,
            num_fourier_freqs=16,
            hidden_dim=channels,
            dropout=0.0,
        )
        self.output = zero_module(nn.Linear(channels, 2))

    def forward(self, time: Tensor, *, time_condition_scale: float) -> Tensor:
        condition = flow_time_condition(time, scale=time_condition_scale)
        return self.output(self.embedding(condition)).clamp(self.low, self.high)


class FlowVectorFieldAdapter:
    """Bridge normalized rectified-flow state to physical Crystal-DiT geometry."""

    def __init__(
        self,
        denoiser: nn.Module,
        *,
        coord_scale: float,
        lattice_prior: VolumeLatticePrior,
        time_condition_scale: float,
        geometry_stabilization_start: float,
        geometry_latent_clip: float,
    ) -> None:
        self.denoiser = denoiser
        self.coord_scale = float(coord_scale)
        self.lattice_prior = lattice_prior
        self.time_condition_scale = float(time_condition_scale)
        self.geometry_stabilization_start = float(geometry_stabilization_start)
        self.geometry_latent_clip = float(geometry_latent_clip)
        if self.coord_scale <= 0.0 or not math.isfinite(self.coord_scale):
            raise ValueError("coord_scale must be finite and positive")
        if self.time_condition_scale <= 0.0 or not math.isfinite(
            self.time_condition_scale
        ):
            raise ValueError("time_condition_scale must be finite and positive")
        if not 0.0 <= self.geometry_stabilization_start < 1.0:
            raise ValueError("geometry_stabilization_start must be in [0, 1)")
        if self.geometry_latent_clip <= 0.0 or not math.isfinite(
            self.geometry_latent_clip
        ):
            raise ValueError("geometry_latent_clip must be finite and positive")
        if getattr(denoiser, "state_representation", None) != "cartesian_q":
            raise ValueError(
                "Flow Matching requires denoiser.state_representation='cartesian_q'"
            )
        if getattr(denoiser, "prediction_type", None) != "velocity":
            raise ValueError("Flow Matching requires denoiser.prediction_type='velocity'")

    def _ensure_device(self, device: torch.device) -> None:
        if self.lattice_prior.crystal_family.masks.device != device:
            self.lattice_prior.to(device)

    def prepare_condition(
        self,
        batch: AsuBatch,
        *,
        cache_trainable_features: Optional[bool] = None,
    ) -> Any:
        prepare = getattr(self.denoiser, "prepare_condition", None)
        if prepare is None:
            return None
        return prepare(batch, cache_trainable_features=cache_trainable_features)

    def predict(
        self,
        batch: AsuBatch,
        state_x: Tensor,
        state_q: Tensor,
        time: Tensor,
        *,
        lattice_stats: Optional[LatticePriorBatchStats] = None,
        prepared_condition: Any = None,
        self_condition: Optional[FlowSelfConditioningState] = None,
    ) -> FlowPrediction:
        self._ensure_device(state_x.device)
        if time.shape != (batch.batch_size,):
            raise ValueError("flow time must have shape [B]")
        stats = lattice_stats or self.lattice_prior.stats_from_batch(
            batch, device=state_x.device, dtype=torch.float32
        )
        mask = stats.mask.to(device=state_q.device, dtype=state_q.dtype)
        state_q = state_q * mask
        physical_x = state_x * self.coord_scale
        physical_q = self.lattice_prior.unwhiten_batch(state_q, batch, stats)
        clipped_q = state_q.float().clamp(
            -self.geometry_latent_clip, self.geometry_latent_clip
        )
        blend = (
            (time.float() - self.geometry_stabilization_start)
            / (1.0 - self.geometry_stabilization_start)
        ).clamp(0.0, 1.0)
        blend = blend.square() * (3.0 - 2.0 * blend)
        geometry_state_q = torch.lerp(
            state_q.float(), clipped_q, blend.unsqueeze(-1)
        ) * mask.float()
        geometry_q = self.lattice_prior.unwhiten_batch(
            geometry_state_q.to(dtype=state_q.dtype), batch, stats
        )
        atom_input_scale = torch.full(
            (batch.batch_size,),
            1.0 / self.coord_scale,
            device=state_x.device,
            dtype=torch.float32,
        )
        velocity_x, velocity_q = self.denoiser(
            batch,
            state_x,
            physical_q,
            time,
            flow_condition=flow_time_condition(
                time, scale=self.time_condition_scale
            ).to(
                device=state_x.device, dtype=state_x.dtype
            ),
            prepared_condition=prepared_condition,
            physical_x=physical_x,
            model_q=state_q,
            atom_input_scale=atom_input_scale,
            geometry_q=geometry_q,
            self_condition=self_condition,
        )
        velocity_q = velocity_q * mask.to(dtype=velocity_q.dtype)
        atom_time = time.index_select(0, batch.batch.long()).unsqueeze(-1).to(
            dtype=velocity_x.dtype
        )
        graph_time = time.unsqueeze(-1).to(dtype=velocity_q.dtype)
        clean_state_x = state_x - atom_time * velocity_x
        clean_state_q = (state_q - graph_time * velocity_q) * mask.to(
            dtype=velocity_q.dtype
        )
        clean_x = clean_state_x * self.coord_scale
        clean_q = self.lattice_prior.unwhiten_batch(clean_state_q, batch, stats)
        return FlowPrediction(
            velocity_x=velocity_x,
            velocity_q=velocity_q,
            clean_x=clean_x,
            clean_q=clean_q,
            clean_state_x=clean_state_x,
            clean_state_q=clean_state_q,
        )


class FlowMatchingModule(nn.Module):
    """Single-pass rectified-flow objective for Cartesian ASU and constrained q."""

    def __init__(
        self,
        denoiser: nn.Module,
        *,
        coord_scale: float = 8.0,
        time_mean: float = 0.0,
        time_std: float = 1.0,
        time_clamp: float = 1.0e-5,
        time_condition_scale: float = 1000.0,
        geometry_stabilization_start: float = 0.5,
        geometry_latent_clip: float = 1.5,
        coord_velocity_weight: float = 1.0,
        coord_velocity_mode: str = "asu",
        lattice_velocity_weight: float = 1.0,
        full_cell_endpoint_weight: float = 0.25,
        coord_endpoint_pred_lattice_mode: str = "predicted_clean",
        lattice_endpoint_weight: float = 0.0,
        lattice_normalize_by_dof: bool = True,
        contact_clash_config: Optional[Mapping[str, Any]] = None,
        bond_geometry_config: Optional[Mapping[str, Any]] = None,
        lattice_prior_config: Optional[Mapping[str, Any]] = None,
        diagnostics_config: Optional[Mapping[str, Any]] = None,
        uncertainty_weighting_config: Optional[Mapping[str, Any]] = None,
        self_conditioning_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.coord_scale = float(coord_scale)
        self.time_mean = float(time_mean)
        self.time_std = float(time_std)
        self.time_clamp = float(time_clamp)
        self.time_condition_scale = float(time_condition_scale)
        if self.coord_scale <= 0.0 or not math.isfinite(self.coord_scale):
            raise ValueError("coord_scale must be finite and positive")
        if not math.isfinite(self.time_mean):
            raise ValueError("time_mean must be finite")
        if self.time_std < 0.0 or not math.isfinite(self.time_std):
            raise ValueError("time_std must be finite and non-negative")
        if not 0.0 < self.time_clamp < 0.5:
            raise ValueError("time_clamp must be in (0, 0.5)")
        prior_config = dict(lattice_prior_config or {})
        self.lattice_prior = VolumeLatticePrior.from_config(prior_config)
        self.prior_cfg = {"lattice": prior_config}
        self.path = RectifiedFlowPath(
            coord_scale=self.coord_scale,
            lattice_prior=self.lattice_prior,
        )
        self.vector_field = FlowVectorFieldAdapter(
            denoiser,
            coord_scale=self.coord_scale,
            lattice_prior=self.lattice_prior,
            time_condition_scale=self.time_condition_scale,
            geometry_stabilization_start=geometry_stabilization_start,
            geometry_latent_clip=geometry_latent_clip,
        )
        diagnostic_cfg = dict(diagnostics_config or {})
        self.loss_fn = FlowMatchingLoss(
            coord_velocity_weight=coord_velocity_weight,
            coord_velocity_mode=coord_velocity_mode,
            lattice_velocity_weight=lattice_velocity_weight,
            full_cell_endpoint_weight=full_cell_endpoint_weight,
            coord_endpoint_pred_lattice_mode=(
                coord_endpoint_pred_lattice_mode
            ),
            lattice_endpoint_weight=lattice_endpoint_weight,
            coord_scale=self.coord_scale,
            lattice_normalize_by_dof=lattice_normalize_by_dof,
            contact_clash_config=contact_clash_config,
            bond_geometry_config=bond_geometry_config,
            time_bucket_edges=diagnostic_cfg.get(
                "time_bucket_edges", DEFAULT_TIME_BUCKET_EDGES
            ),
        )
        uncertainty_cfg = dict(uncertainty_weighting_config or {})
        self.uncertainty_weighting_enabled = bool(
            uncertainty_cfg.get("enabled", True)
        )
        clamp_raw = uncertainty_cfg.get("clamp", [-2.0, 2.0])
        if not isinstance(clamp_raw, (list, tuple)) or len(clamp_raw) != 2:
            raise ValueError("uncertainty_weighting.clamp must contain two values")
        self.uncertainty_weighting = (
            FlowUncertaintyWeighting(
                channels=int(uncertainty_cfg.get("channels", 128)),
                clamp=(float(clamp_raw[0]), float(clamp_raw[1])),
            )
            if self.uncertainty_weighting_enabled
            else None
        )
        self.self_conditioning_cfg = dict(self_conditioning_config or {})
        self.self_conditioning_enabled = bool(
            self.self_conditioning_cfg.get("enabled", False)
        )
        self.self_conditioning_train_probability = float(
            self.self_conditioning_cfg.get("train_probability", 0.5)
        )
        self.self_conditioning_prepass_selected_only = bool(
            self.self_conditioning_cfg.get("prepass_selected_only", True)
        )
        self.self_conditioning_sampling_enabled = bool(
            self.self_conditioning_cfg.get("sampling_enabled", True)
        )
        representation = str(
            self.self_conditioning_cfg.get(
                "representation", "normalized_clean_residual"
            )
        ).lower()
        if representation != "normalized_clean_residual":
            raise ValueError(
                "flow self-conditioning representation must be "
                "normalized_clean_residual"
            )
        if not bool(self.self_conditioning_cfg.get("detach", True)):
            raise ValueError("flow self-conditioning feedback must be detached")
        if not 0.0 <= self.self_conditioning_train_probability <= 1.0:
            raise ValueError(
                "flow self-conditioning train_probability must be in [0, 1]"
            )
        model_sc_enabled = getattr(denoiser, "self_conditioning_enabled", None)
        if self.self_conditioning_enabled and model_sc_enabled is not True:
            raise ValueError(
                "flow self-conditioning requires a self-conditioning-enabled denoiser"
            )

    def _clean_inputs(self, batch: AsuBatch) -> Tuple[Tensor, Tensor]:
        template_id = require_lattice_template(
            batch.spacegroup, batch.lattice_template_id
        )
        q_0 = lattice_to_q(
            batch.lattice_gt,
            spacegroup=batch.spacegroup,
            lattice_template_id=template_id,
        ).to(device=batch.frac_coords_gt.device, dtype=batch.frac_coords_gt.dtype)
        clean_lattice = q_to_lattice(
            q_0,
            spacegroup=batch.spacegroup,
            lattice_template_id=template_id,
        )
        x_0 = frac_to_cart_coords_batched(
            batch.frac_coords_gt,
            clean_lattice,
            batch.batch,
        )
        return x_0, q_0

    def _build_self_conditioning_state(
        self,
        batch: AsuBatch,
        *,
        prediction: FlowPrediction,
        state_x: Tensor,
        state_q: Tensor,
        lattice_stats: LatticePriorBatchStats,
        presence: Optional[Tensor] = None,
    ) -> FlowSelfConditioningState:
        if presence is None:
            presence = torch.ones(
                batch.batch_size, device=state_x.device, dtype=torch.bool
            )
        else:
            presence = presence.to(device=state_x.device, dtype=torch.bool)
        atom_present = presence.index_select(0, batch.batch.long())
        atom_residual = prediction.clean_state_x - state_x
        atom_residual = torch.where(
            atom_present.unsqueeze(-1),
            atom_residual,
            torch.zeros_like(atom_residual),
        )
        mask = lattice_stats.mask.to(
            device=state_q.device, dtype=state_q.dtype
        )
        lattice_residual = (prediction.clean_state_q - state_q) * mask
        lattice_residual = torch.where(
            presence.unsqueeze(-1),
            lattice_residual,
            torch.zeros_like(lattice_residual),
        )
        return FlowSelfConditioningState._from_owned_detached(
            atom_residual=atom_residual,
            lattice_residual=lattice_residual,
            graph_present=presence,
        )

    def _training_self_condition(
        self,
        *,
        batch: AsuBatch,
        sample,
        lattice_stats: LatticePriorBatchStats,
        prepared_condition: Any,
        enabled: bool,
    ) -> tuple[Optional[FlowSelfConditioningState], Dict[str, Tensor]]:
        batch_size = int(batch.batch_size)
        total_atoms = int(sample.state_x.shape[0])
        selected_count = 0
        selected_atoms = 0
        self_condition = None
        probability = self.self_conditioning_train_probability
        if enabled and probability > 0.0:
            selected_mask = torch.rand(batch_size, device="cpu") < probability
            selected_graphs_cpu = torch.nonzero(
                selected_mask, as_tuple=False
            ).view(-1)
            selected_count = int(selected_graphs_cpu.numel())
            if selected_count > 0:
                selected_graphs = selected_graphs_cpu.to(
                    device=sample.state_x.device, dtype=torch.long
                )
                use_subbatch = (
                    self.self_conditioning_prepass_selected_only
                    and selected_count < batch_size
                )
                with torch.no_grad(), _autocast_without_weight_cache(
                    sample.state_x.device.type
                ):
                    if use_subbatch:
                        sub_batch, index_map = select_graph_subbatch(
                            batch, selected_graphs
                        )
                        selected_atoms = index_map.num_atoms
                        sub_stats = _select_lattice_stats(
                            lattice_stats, index_map
                        )
                        sub_prepared = self.vector_field.prepare_condition(
                            sub_batch, cache_trainable_features=False
                        )
                        sub_prediction = self.vector_field.predict(
                            sub_batch,
                            index_map.select_atoms(sample.state_x),
                            index_map.select_graphs(sample.state_q),
                            index_map.select_graphs(sample.time),
                            lattice_stats=sub_stats,
                            prepared_condition=sub_prepared,
                        )
                        sub_state = self._build_self_conditioning_state(
                            sub_batch,
                            prediction=sub_prediction,
                            state_x=index_map.select_atoms(sample.state_x),
                            state_q=index_map.select_graphs(sample.state_q),
                            lattice_stats=sub_stats,
                        )
                        self_condition = (
                            scatter_flow_self_conditioning_state(
                                sub_state, index_map
                            )
                        )
                    else:
                        selected_atoms = total_atoms
                        full_prediction = self.vector_field.predict(
                            batch,
                            sample.state_x,
                            sample.state_q,
                            sample.time,
                            lattice_stats=lattice_stats,
                            prepared_condition=prepared_condition,
                        )
                        self_condition = self._build_self_conditioning_state(
                            batch,
                            prediction=full_prediction,
                            state_x=sample.state_x,
                            state_q=sample.state_q,
                            lattice_stats=lattice_stats,
                            presence=selected_mask.to(
                                device=sample.state_x.device
                            ),
                        )
                        if selected_count < batch_size:
                            selected_atoms = int(
                                batch.num_atoms.detach()
                                .cpu()
                                .index_select(0, selected_graphs_cpu)
                                .sum()
                                .item()
                            )
        metric_dtype = sample.state_x.dtype
        metric_device = sample.state_x.device
        metrics = {
            "metric/sc_selected_graph_fraction": torch.tensor(
                selected_count / max(batch_size, 1),
                device=metric_device,
                dtype=metric_dtype,
            ),
            "metric/sc_selected_atom_fraction": torch.tensor(
                selected_atoms / max(total_atoms, 1),
                device=metric_device,
                dtype=metric_dtype,
            ),
            "metric/sc_prepass_nfe": torch.tensor(
                float(selected_count > 0),
                device=metric_device,
                dtype=metric_dtype,
            ),
        }
        return self_condition, metrics

    def sample_times(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        return sample_logit_normal_times(
            batch_size,
            mean=self.time_mean,
            std=self.time_std,
            clamp=self.time_clamp,
            device=device,
            dtype=dtype,
        )

    def training_step(
        self,
        batch: AsuBatch,
        *,
        return_path_sample: bool = False,
        enable_self_conditioning: Optional[bool] = None,
    ) -> FlowTrainingOutput:
        clean_x, clean_q = self._clean_inputs(batch)
        lattice_stats = self.lattice_prior.stats(
            spacegroup=batch.spacegroup,
            n_heavy_cell=batch.n_heavy_cell,
            lattice_template_id=getattr(batch, "lattice_template_id", None),
            device=clean_x.device,
            dtype=torch.float32,
        )
        time = self.sample_times(batch.batch_size, clean_x.device, clean_x.dtype)
        sample = self.path.sample(
            batch=batch,
            clean_x=clean_x,
            clean_q=clean_q,
            time=time,
            lattice_stats=lattice_stats,
        )
        prepared_condition = self.vector_field.prepare_condition(
            batch, cache_trainable_features=False
        )
        use_self_conditioning = (
            self.self_conditioning_enabled and self.training
            if enable_self_conditioning is None
            else bool(enable_self_conditioning)
        )
        self_condition, sc_metrics = self._training_self_condition(
            batch=batch,
            sample=sample,
            lattice_stats=lattice_stats,
            prepared_condition=prepared_condition,
            enabled=use_self_conditioning,
        )
        prediction = self.vector_field.predict(
            batch,
            sample.state_x,
            sample.state_q,
            time,
            lattice_stats=lattice_stats,
            prepared_condition=prepared_condition,
            self_condition=self_condition,
        )
        uncertainty = (
            self.uncertainty_weighting(
                time, time_condition_scale=self.time_condition_scale
            )
            if self.uncertainty_weighting is not None
            else None
        )
        loss = self.loss_fn(
            batch=batch,
            sample=sample,
            prediction=prediction,
            lattice_stats=lattice_stats,
            orbit_plan=getattr(prepared_condition, "plan", None),
            uncertainty_logvar=uncertainty,
        )
        metrics = dict(loss.metrics)
        metrics.update(sc_metrics)
        return FlowTrainingOutput(
            loss=loss.loss,
            metrics=metrics,
            path_sample=sample if return_path_sample else None,
            per_graph_loss=loss.per_graph_loss,
        )

    def forward(
        self,
        batch: AsuBatch,
        state_x: Tensor,
        state_q: Tensor,
        time: Tensor,
        self_condition: Optional[FlowSelfConditioningState] = None,
    ) -> Tuple[Tensor, Tensor]:
        stats = self.lattice_prior.stats_from_batch(
            batch, device=state_x.device, dtype=torch.float32
        )
        prediction = self.vector_field.predict(
            batch,
            state_x,
            state_q,
            time,
            lattice_stats=stats,
            self_condition=self_condition,
        )
        return prediction.velocity_x, prediction.velocity_q


__all__ = [
    "FlowMatchingModule",
    "FlowUncertaintyWeighting",
    "FlowVectorFieldAdapter",
    "MODEL_FAMILY",
]
