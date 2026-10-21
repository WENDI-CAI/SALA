from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from tqdm.auto import tqdm

from sala.core.lattice import cart_to_frac_coords_batched
from sala.core.lattice_manifold import lattice_to_q, q_to_lattice
from sala.core.orbit import build_orbit_plan
from sala.core.types import AsuBatch
from sala.flow_matching.flow import FlowVectorFieldAdapter
from sala.flow_matching.lattice_prior import LatticePriorBatchStats, VolumeLatticePrior
from sala.flow_matching.types import FlowSelfConditioningState


@dataclass
class FlowSamplerOutput:
    frac_coords: Tensor
    lattice: Tensor
    cart_coords: Tensor
    lattice_q: Tensor
    full_frac_coords: Optional[Tensor] = None
    full_cart_coords: Optional[Tensor] = None
    full_atom_mask: Optional[Tensor] = None


@dataclass
class FlowSamplingTrajectory:
    coords_traj: List[Tensor]
    lattice_traj: List[Tensor]
    times: List[float]


class Flow2MSampler:
    """Second-order linear multistep sampler with one network evaluation per step."""

    def __init__(
        self,
        denoiser: nn.Module,
        *,
        lattice_prior: Optional[VolumeLatticePrior] = None,
        prior_cfg: Optional[dict[str, Any]] = None,
        coord_scale: float = 8.0,
        time_condition_scale: float = 1000.0,
        geometry_stabilization_start: float = 0.5,
        geometry_latent_clip: float = 1.5,
        count_nfe: bool = True,
        self_conditioning_cfg: Optional[dict[str, Any]] = None,
        coord_prior_temperature: float = 1.0,
        lattice_prior_temperature: float = 1.0,
    ) -> None:
        self.denoiser = denoiser
        effective_prior = dict((prior_cfg or {}).get("lattice") or {})
        self.lattice_prior = lattice_prior or VolumeLatticePrior.from_config(
            effective_prior
        )
        self.prior_cfg = dict(prior_cfg or {})
        self.coord_scale = float(coord_scale)
        self.time_condition_scale = float(time_condition_scale)
        self.coord_prior_temperature = float(coord_prior_temperature)
        self.lattice_prior_temperature = float(lattice_prior_temperature)
        for name, value in (
            ("coord_prior_temperature", self.coord_prior_temperature),
            ("lattice_prior_temperature", self.lattice_prior_temperature),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    f"{name} must lie in (0, 1]; values above 1 would sample "
                    "outside the trained prior support"
                )
        self.count_nfe = bool(count_nfe)
        self.last_nfe = 0
        self.self_conditioning_cfg = dict(self_conditioning_cfg or {})
        self.self_conditioning_enabled = bool(
            self.self_conditioning_cfg.get("enabled", False)
            and self.self_conditioning_cfg.get("sampling_enabled", True)
        )
        if (
            self.self_conditioning_enabled
            and getattr(denoiser, "self_conditioning_enabled", None) is not True
        ):
            raise ValueError(
                "flow sampling self-conditioning requires a "
                "self-conditioning-enabled denoiser"
            )
        self.vector_field = FlowVectorFieldAdapter(
            denoiser,
            coord_scale=self.coord_scale,
            lattice_prior=self.lattice_prior,
            time_condition_scale=self.time_condition_scale,
            geometry_stabilization_start=geometry_stabilization_start,
            geometry_latent_clip=geometry_latent_clip,
        )

    @staticmethod
    def _build_sampling_self_condition(
        batch: AsuBatch,
        *,
        clean_state_x: Tensor,
        clean_state_q: Tensor,
        next_state_x: Tensor,
        next_state_q: Tensor,
        lattice_mask: Tensor,
    ) -> FlowSelfConditioningState:
        """Condition the next solve step on the previous clean endpoint."""
        atom_residual = clean_state_x.to(dtype=next_state_x.dtype) - next_state_x
        lattice_residual = (
            clean_state_q.to(dtype=next_state_q.dtype) - next_state_q
        ) * lattice_mask.to(dtype=next_state_q.dtype)
        return FlowSelfConditioningState._from_owned_detached(
            atom_residual=atom_residual,
            lattice_residual=lattice_residual,
            graph_present=torch.ones(
                batch.batch_size,
                device=next_state_x.device,
                dtype=torch.bool,
            ),
        )

    def _ensure_device(self, device: torch.device) -> None:
        if self.lattice_prior.crystal_family.masks.device != device:
            self.lattice_prior.to(device)

    @staticmethod
    def build_time_sequence(
        num_steps: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        if int(num_steps) <= 0:
            raise ValueError("num_steps must be positive")
        return torch.linspace(
            1.0,
            0.0,
            int(num_steps) + 1,
            device=device,
            dtype=dtype,
        )

    def initialize_prior(
        self,
        batch: AsuBatch,
        *,
        device: torch.device,
        lattice_stats: LatticePriorBatchStats,
    ) -> Tuple[Tensor, Tensor]:
        """Draw the whitened prior state, optionally at reduced temperature.

        The Flow-2M trajectory is a deterministic ODE, so this draw is the only
        source of variation between samples for a fixed conditioning. Scaling it
        by ``t < 1`` therefore concentrates candidates near the prior mean and
        trades diversity for predictability, which is the standard low-temperature
        sampling knob. It changes only the starting distribution, not the
        whitening statistics, so train and sample stay on the same manifold.

        Coordinates and lattice are scaled independently: the lattice carries only
        ``k_G <= 6`` scalars whose prior is already calibrated from cell volume, so
        it is usually the one worth tightening first.
        """

        state_x = torch.randn(
            int(batch.atom_types.shape[0]), 3, device=device, dtype=torch.float32
        )
        state_q = torch.randn(
            batch.batch_size, 6, device=device, dtype=torch.float32
        )
        if self.coord_prior_temperature != 1.0:
            state_x = state_x * self.coord_prior_temperature
        if self.lattice_prior_temperature != 1.0:
            state_q = state_q * self.lattice_prior_temperature
        mask = lattice_stats.mask.to(device=device, dtype=state_q.dtype)
        return state_x, state_q * mask

    def _state_to_geometry(
        self,
        batch: AsuBatch,
        state_x: Tensor,
        state_q: Tensor,
        lattice_stats: LatticePriorBatchStats,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cart = state_x * self.coord_scale
        q = self.lattice_prior.unwhiten_batch(state_q, batch, lattice_stats)
        lattice = q_to_lattice(
            q,
            spacegroup=batch.spacegroup,
            lattice_template_id=getattr(batch, "lattice_template_id", None),
        )
        frac = cart_to_frac_coords_batched(cart, lattice, batch.batch)
        return frac, lattice, q

    @staticmethod
    def _flow_2m_update(
        *,
        state_x: Tensor,
        state_q: Tensor,
        velocity_x: Tensor,
        velocity_q: Tensor,
        previous_velocity_x: Tensor | None,
        previous_velocity_q: Tensor | None,
        time: float,
        time_next: float,
        previous_time: float | None,
        lattice_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        step = float(time_next) - float(time)
        if (
            previous_velocity_x is None
            or previous_velocity_q is None
            or previous_time is None
        ):
            corrected_x = velocity_x
            corrected_q = velocity_q
        else:
            previous_step = float(time) - float(previous_time)
            if abs(previous_step) < 1.0e-12:
                raise ValueError("flow time sequence contains a repeated step")
            ratio = step / previous_step
            corrected_x = (1.0 + 0.5 * ratio) * velocity_x - (
                0.5 * ratio
            ) * previous_velocity_x
            corrected_q = (1.0 + 0.5 * ratio) * velocity_q - (
                0.5 * ratio
            ) * previous_velocity_q
        next_x = state_x + step * corrected_x
        next_q = (state_q + step * corrected_q) * lattice_mask
        return next_x, next_q

    @torch.no_grad()
    def sample(
        self,
        batch: AsuBatch,
        num_steps: int = 24,
        return_trajectory: bool = False,
        trajectory_interval: int = 100,
        progress_bar: bool = True,
    ) -> Union[
        FlowSamplerOutput,
        Tuple[FlowSamplerOutput, FlowSamplingTrajectory],
    ]:
        self.last_nfe = 0
        device = batch.atom_types.device
        self._ensure_device(device)
        times = self.build_time_sequence(
            int(num_steps), torch.device("cpu"), torch.float32
        )
        time_values = [float(value) for value in times.tolist()]
        lattice_stats = self.lattice_prior.stats_from_batch(
            batch, device=device, dtype=torch.float32
        )
        lattice_mask = lattice_stats.mask.to(device=device, dtype=torch.float32)
        state_x, state_q = self.initialize_prior(
            batch, device=device, lattice_stats=lattice_stats
        )
        if self.denoiser.training:
            raise RuntimeError("Flow sampling requires denoiser.eval()")
        prepared_condition = self.vector_field.prepare_condition(
            batch, cache_trainable_features=True
        )

        trajectory = None
        if return_trajectory:
            frac, lattice, _ = self._state_to_geometry(
                batch, state_x, state_q, lattice_stats
            )
            trajectory = FlowSamplingTrajectory(
                coords_traj=[frac.clone()],
                lattice_traj=[lattice.clone()],
                times=[1.0],
            )

        previous_velocity_x: Tensor | None = None
        previous_velocity_q: Tensor | None = None
        previous_time: float | None = None
        self_condition: FlowSelfConditioningState | None = None
        iterator = enumerate(zip(time_values[:-1], time_values[1:]))
        if progress_bar:
            iterator = tqdm(
                list(iterator), desc="Flow-2M sampling", leave=False
            )
        for step_index, (time, time_next) in iterator:
            time_tensor = torch.full(
                (batch.batch_size,),
                time,
                device=device,
                dtype=state_x.dtype,
            )
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                prediction = self.vector_field.predict(
                    batch,
                    state_x,
                    state_q,
                    time_tensor,
                    lattice_stats=lattice_stats,
                    prepared_condition=prepared_condition,
                    self_condition=self_condition,
                )
            velocity_x = prediction.velocity_x.to(dtype=state_x.dtype)
            velocity_q = prediction.velocity_q.to(dtype=state_q.dtype)
            if self.count_nfe:
                self.last_nfe += 1
            next_x, next_q = self._flow_2m_update(
                state_x=state_x,
                state_q=state_q,
                velocity_x=velocity_x,
                velocity_q=velocity_q,
                previous_velocity_x=previous_velocity_x,
                previous_velocity_q=previous_velocity_q,
                time=time,
                time_next=time_next,
                previous_time=previous_time,
                lattice_mask=lattice_mask,
            )
            previous_velocity_x = velocity_x
            previous_velocity_q = velocity_q
            previous_time = time
            self_condition = (
                self._build_sampling_self_condition(
                    batch,
                    clean_state_x=prediction.clean_state_x,
                    clean_state_q=prediction.clean_state_q,
                    next_state_x=next_x,
                    next_state_q=next_q,
                    lattice_mask=lattice_mask,
                )
                if self.self_conditioning_enabled and time_next > 0.0
                else None
            )
            state_x, state_q = next_x, next_q
            if return_trajectory and (
                step_index % max(1, int(trajectory_interval)) == 0
                or time_next == 0.0
            ):
                frac, lattice, _ = self._state_to_geometry(
                    batch, state_x, state_q, lattice_stats
                )
                trajectory.coords_traj.append(frac.clone())
                trajectory.lattice_traj.append(lattice.clone())
                trajectory.times.append(time_next)

        output_x = state_x * self.coord_scale
        output_q = self.lattice_prior.unwhiten_batch(
            state_q * lattice_mask, batch, lattice_stats
        )
        output_lattice = q_to_lattice(
            output_q,
            spacegroup=batch.spacegroup,
            lattice_template_id=getattr(batch, "lattice_template_id", None),
        )
        output_q = lattice_to_q(
            output_lattice,
            spacegroup=batch.spacegroup,
            lattice_template_id=getattr(batch, "lattice_template_id", None),
        ).to(device=output_q.device, dtype=output_q.dtype)
        output_lattice = q_to_lattice(
            output_q,
            spacegroup=batch.spacegroup,
            lattice_template_id=getattr(batch, "lattice_template_id", None),
        )
        frac_asu = cart_to_frac_coords_batched(
            output_x, output_lattice, batch.batch
        )
        plan = build_orbit_plan(batch, validate_general_position=True)
        full = plan.expand_coordinates(output_x, output_lattice)
        output = FlowSamplerOutput(
            frac_coords=frac_asu,
            lattice=output_lattice,
            cart_coords=output_x,
            lattice_q=output_q,
            full_frac_coords=full.frac,
            full_cart_coords=full.cart,
            full_atom_mask=full.valid,
        )
        if return_trajectory:
            return output, trajectory
        return output


__all__ = ["Flow2MSampler", "FlowSamplerOutput", "FlowSamplingTrajectory"]
