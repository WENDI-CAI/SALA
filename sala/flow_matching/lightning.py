from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule

from sala.core.types import AsuBatch
from sala.data.strata import CHEM_GROUPS, CRYSTAL_SYSTEMS
from sala.flow_matching.flow import FlowMatchingModule, MODEL_FAMILY
from sala.flow_matching.lattice_prior import VolumeLatticePrior
from sala.flow_matching.losses import (
    bond_geometry_objective_signature,
    contact_clash_objective_signature,
    resolve_bond_geometry_config,
    resolve_contact_clash_config,
)

# Schema 16 adds an explicit pair-bias capability flag while retaining read
# compatibility with schema 15's always-enabled pair bias.
CHECKPOINT_SCHEMA_VERSION = 16
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = frozenset({15, CHECKPOINT_SCHEMA_VERSION})
_EVALUATION_NOISE_SEED = 0x5A1A
logger = logging.getLogger(__name__)


def _canonicalize_denoiser_config(
    denoiser_config: Dict[str, Any],
) -> Dict[str, Any]:
    canonical = dict(denoiser_config)
    pair = denoiser_config.get("pair") or {}
    if not isinstance(pair, dict):
        raise ValueError("denoiser_config.pair must be a mapping")
    pair = dict(pair)
    pair["enabled"] = bool(pair.get("enabled", True))
    canonical["pair"] = pair
    return canonical


def _pair_bias_enabled(denoiser_config: Dict[str, Any]) -> bool:
    return bool(_canonicalize_denoiser_config(denoiser_config)["pair"]["enabled"])


def _denoiser_contract_signature(
    denoiser_config: Dict[str, Any],
) -> Dict[str, Any]:
    return _canonicalize_denoiser_config(denoiser_config)


def validate_full_cell_checkpoint_contract(
    flow_config: Dict[str, Any],
    denoiser_config: Dict[str, Any],
    *,
    checkpoint_schema_version: int = CHECKPOINT_SCHEMA_VERSION,
) -> None:
    """Require the representation contract promised by the checkpoint schema."""
    architecture = str(
        denoiser_config.get("architecture", "")
    ).lower()
    if architecture != "full_cell_crystal_dit_flow":
        raise ValueError(
            "Flow Matching checkpoints require "
            "architecture=full_cell_crystal_dit_flow"
        )
    if str(denoiser_config.get("atom_coord_repr", "")).lower() != "canonical_cartesian":
        raise ValueError(
            "Flow Matching checkpoints require atom_coord_repr=canonical_cartesian"
        )
    if str(denoiser_config.get("lattice_parameterization", "")).lower() != "constrained_q":
        raise ValueError(
            "Flow Matching checkpoints require "
            "lattice_parameterization=constrained_q"
        )
    if (
        str(denoiser_config.get("component_representation", "")).lower()
        != "base_instance_type_v2"
    ):
        raise ValueError(
            "Flow Matching checkpoints require "
            "component_representation=base_instance_type_v2"
        )
    # Model dimensions are checked against the full denoiser signature on load.
    num_layers = int(denoiser_config.get("num_layers", 0))
    if num_layers <= 0:
        raise ValueError(
            "Flow Matching checkpoints require a positive num_layers, got "
            f"{num_layers}"
        )
    state = dict(flow_config.get("state") or {})
    expected = {
        "coordinate": "canonical_cartesian",
        "lattice": "constrained_q",
        "model": "normalized_cartesian_whitened_q",
        "output": "fractional_unwrapped",
        "coord_scale": 8.0,
    }
    if state != expected:
        raise ValueError(
            "Flow Matching checkpoints require the normalized Cartesian/whitened-q "
            f"flow state contract; got {state!r}"
        )
    path = dict(flow_config.get("path") or {})
    if path != {
        "type": "rectified_linear",
        "prediction": "velocity",
        "data_time": 0.0,
        "noise_time": 1.0,
    }:
        raise ValueError("Flow Matching checkpoints require the locked linear velocity path")

    loss = flow_config.get("loss") or {}
    if not isinstance(loss, dict):
        raise ValueError("flow_config.loss must be a mapping")
    resolve_contact_clash_config(loss.get("contact_clash"))
    resolve_bond_geometry_config(loss.get("bond_geometry"))

    flow_self_conditioning = flow_config.get("self_conditioning") or {}
    if not isinstance(flow_self_conditioning, dict):
        raise ValueError("flow_config.self_conditioning must be a mapping")
    unsupported_flow_sc = sorted(
        set(flow_self_conditioning)
        - {
            "enabled",
            "train_probability",
            "prepass_selected_only",
            "detach",
            "sampling_enabled",
            "representation",
        }
    )
    if unsupported_flow_sc:
        raise ValueError(
            "Unsupported checkpoint flow self-conditioning keys: "
            f"{unsupported_flow_sc}"
        )
    flow_sc_enabled = bool(flow_self_conditioning.get("enabled", False))
    train_probability = float(
        flow_self_conditioning.get("train_probability", 0.5)
    )
    if not math.isfinite(train_probability) or not 0.0 <= train_probability <= 1.0:
        raise ValueError(
            "Checkpoint flow self-conditioning probability must be in [0, 1]"
        )
    if not bool(flow_self_conditioning.get("detach", True)):
        raise ValueError("Checkpoint flow self-conditioning feedback must be detached")
    if (
        str(
            flow_self_conditioning.get(
                "representation", "normalized_clean_residual"
            )
        ).lower()
        != "normalized_clean_residual"
    ):
        raise ValueError(
            "Checkpoint flow self-conditioning must use "
            "normalized_clean_residual"
        )

    denoiser_self_conditioning = denoiser_config.get("self_conditioning") or {}
    if not isinstance(denoiser_self_conditioning, dict):
        raise ValueError("denoiser_config.self_conditioning must be a mapping")
    unsupported_model_sc = sorted(set(denoiser_self_conditioning) - {"enabled"})
    if unsupported_model_sc:
        raise ValueError(
            "Unsupported checkpoint denoiser self-conditioning keys: "
            f"{unsupported_model_sc}"
        )
    model_sc_enabled = bool(denoiser_self_conditioning.get("enabled", False))
    if flow_sc_enabled != model_sc_enabled:
        raise ValueError(
            "Flow and denoiser self-conditioning enabled flags must match"
        )

    pair = denoiser_config.get("pair") or {}
    if not isinstance(pair, dict):
        raise ValueError("denoiser_config.pair must be a mapping")
    if int(checkpoint_schema_version) >= 16 and "enabled" not in pair:
        raise ValueError("Checkpoint schema v16 requires denoiser_config.pair.enabled")
    pair_bias_enabled = bool(pair.get("enabled", True))
    if int(checkpoint_schema_version) < 16 and not pair_bias_enabled:
        raise ValueError("Checkpoint schema v15 requires full token-pair bias")


def validate_full_cell_denoiser_instance(
    denoiser: nn.Module,
    denoiser_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Bind checkpoint metadata to the runtime object's public semantics."""
    expected = {
        "state_representation": "cartesian_q",
        "atom_coord_repr": "canonical_cartesian",
        "lattice_parameterization": "constrained_q",
        "component_representation": "base_instance_type_v2",
        "prediction_type": "velocity",
    }
    mismatched = {
        name: getattr(denoiser, name, None)
        for name, value in expected.items()
        if str(getattr(denoiser, name, "")).lower() != value
    }
    if mismatched:
        raise ValueError(
            "Full-cell checkpoint configuration does not match the denoiser "
            f"instance: {mismatched!r}"
        )
    if getattr(denoiser, "architecture", None) == "full_cell_crystal_dit_flow":
        blocks = getattr(denoiser, "blocks", None)
        # Check the instance against its own configured depth rather than a
        # literal, so a reduced-capacity model is valid while a model whose block
        # count disagrees with its config is still rejected.
        expected_layers = int(denoiser_config.get("num_layers", 0))
        if blocks is None:
            raise ValueError(
                "Flow Matching checkpoints require a full-cell DiT block stack"
            )
        if len(blocks) != expected_layers:
            raise ValueError(
                "Full-cell denoiser block count does not match its "
                f"configuration: {len(blocks)} blocks against "
                f"num_layers={expected_layers}"
            )
        if hasattr(denoiser, "shared_modulation") or hasattr(denoiser, "pair_scale"):
            raise ValueError(
                "Flow Matching checkpoints require independent per-block AdaLN "
                "and no shared pair scale"
            )
        pair_bias_enabled = bool(getattr(denoiser, "pair_bias_enabled", True))
        if denoiser_config is not None:
            configured_pair_bias = _pair_bias_enabled(denoiser_config)
            if pair_bias_enabled != configured_pair_bias:
                raise ValueError(
                    "Denoiser pair-bias runtime flag does not match its "
                    "checkpoint configuration"
                )
        pair_embedding = getattr(denoiser, "pair_embedding", None)
        if pair_bias_enabled and not hasattr(pair_embedding, "token_pair_bias"):
            raise ValueError("Enabled pair bias requires full token-pair type bias")
        if not pair_bias_enabled and pair_embedding is not None:
            raise ValueError("Disabled pair bias must not retain a pair encoder")
        if any(
            not hasattr(block, "modulation")
            or not hasattr(block, "pair_bias_modulation")
            for block in blocks
        ):
            raise ValueError(
                "Flow Matching checkpoints require independent block modulation modules"
            )
        if pair_bias_enabled and any(
            block.pair_bias_modulation is None for block in blocks
        ):
            raise ValueError("Enabled pair bias requires per-block head modulation")
        if not pair_bias_enabled and any(
            block.pair_bias_modulation is not None for block in blocks
        ):
            raise ValueError("Disabled pair bias must omit per-block head modulation")
    if denoiser_config is not None:
        configured = bool(
            dict(denoiser_config.get("self_conditioning") or {}).get(
                "enabled", False
            )
        )
        runtime = bool(getattr(denoiser, "self_conditioning_enabled", False))
        if runtime != configured:
            raise ValueError(
                "Denoiser self-conditioning runtime flag does not match its "
                "checkpoint configuration"
            )


def _flow_contract_signature(flow_config: Dict[str, Any]) -> Dict[str, Any]:
    loss = dict(flow_config.get("loss") or {})
    loss["contact_clash"] = contact_clash_objective_signature(
        loss.get("contact_clash")
    )
    loss["bond_geometry"] = bond_geometry_objective_signature(
        loss.get("bond_geometry")
    )
    return {
        "state": dict(flow_config.get("state") or {}),
        "path": dict(flow_config.get("path") or {}),
        "time_sampling": dict(flow_config.get("time_sampling") or {}),
        "time_condition": dict(flow_config.get("time_condition") or {}),
        "geometry": dict(flow_config.get("geometry") or {}),
        "self_conditioning": dict(flow_config.get("self_conditioning") or {}),
        "loss": loss,
        "prior": dict(flow_config.get("prior") or {}),
        "uncertainty_weighting": dict(flow_config.get("uncertainty_weighting") or {}),
    }


def _flow_finetune_signature(flow_config: Dict[str, Any]) -> Dict[str, Any]:
    """State/conditioning contract required for weight-only fine-tuning.

    Fine-tuning is allowed to change only the training objective. Resume remains
    strict through :func:`_flow_contract_signature` so optimizer/scheduler state
    cannot silently continue under a different loss.
    """

    signature = _flow_contract_signature(flow_config)
    signature.pop("loss", None)
    return signature


class FlowMatchingLightningModule(LightningModule):
    """The single supported SALA training module: rectified-flow velocity prediction."""

    _TRAIN_SYNC_METRICS = frozenset({"loss/coord", "loss/lattice"})
    _GLOBAL_CONTACT_METRICS = frozenset(
        {
            "loss/contact_clash",
            "loss/clash_repulsion",
            "loss/contact_match",
            "loss/contact_clash_weighted",
            "metric/contact_active_graph_fraction",
            "metric/clash_normalizer_pair_count_mean",
            "metric/contact_target_pair_count_mean",
            "metric/predicted_clash_pair_count_mean",
            "metric/contact_evaluated_pair_count_mean",
            "metric/clash_target_capped_pair_fraction",
            "metric/clash_target_capped_pair_count_mean",
            "metric/contact_periodic_same_instance_fraction",
            "metric/contact_periodic_same_instance_count_mean",
            "metric/contact_predicted_periodic_fallback_fraction",
            "metric/contact_target_periodic_fallback_fraction",
            "metric/contact_predicted_periodic_fallback_count_mean",
            "metric/contact_target_periodic_fallback_count_mean",
        }
    )

    def __init__(
        self,
        denoiser: nn.Module,
        *,
        flow_config: Dict[str, Any],
        denoiser_config: Optional[Dict[str, Any]] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        optimizer_name: str = "adamw",
        muon_config: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[Dict[str, Any]] = None,
        warmup_steps: int = 10000,
        max_grad_norm: Optional[float] = 1.0,
        nominal_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        cfg = dict(flow_config)
        denoiser_config = _canonicalize_denoiser_config(
            dict(denoiser_config or {})
        )
        state = dict(cfg.get("state") or {})
        time_sampling = dict(cfg.get("time_sampling") or {})
        time_condition = dict(cfg.get("time_condition") or {})
        geometry = dict(cfg.get("geometry") or {})
        loss = dict(cfg.get("loss") or {})
        loss["contact_clash"] = resolve_contact_clash_config(
            loss.get("contact_clash")
        )
        loss["bond_geometry"] = resolve_bond_geometry_config(
            loss.get("bond_geometry")
        )
        cfg["loss"] = loss
        prior = dict(cfg.get("prior") or {})
        lattice_prior_cfg = dict(prior.get("lattice") or {})
        self.save_hyperparameters(
            {
                "model_family": MODEL_FAMILY,
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "flow_config": cfg,
                "denoiser_config": denoiser_config,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "optimizer_name": str(optimizer_name).lower(),
                "muon_config": dict(muon_config or {}),
                "lr_scheduler": dict(lr_scheduler or {}),
                "warmup_steps": int(warmup_steps),
                "max_grad_norm": (
                    None if max_grad_norm is None else float(max_grad_norm)
                ),
                "nominal_batch_size": (
                    None if nominal_batch_size is None else int(nominal_batch_size)
                ),
            }
        )
        self.flow = FlowMatchingModule(
            denoiser=denoiser,
            coord_scale=float(state.get("coord_scale", 8.0)),
            time_mean=float(time_sampling.get("mean", 0.0)),
            time_std=float(time_sampling.get("std", 1.0)),
            time_clamp=float(time_sampling.get("clamp", 1.0e-5)),
            time_condition_scale=float(time_condition.get("scale", 1000.0)),
            geometry_stabilization_start=float(
                geometry.get("stabilization_start", 0.5)
            ),
            geometry_latent_clip=float(geometry.get("latent_clip", 1.5)),
            coord_velocity_weight=float(loss.get("coord_velocity_weight", 1.0)),
            coord_velocity_mode=str(loss.get("coord_velocity_mode", "asu")),
            lattice_velocity_weight=float(loss.get("lattice_velocity_weight", 1.0)),
            full_cell_endpoint_weight=float(loss.get("full_cell_endpoint_weight", 0.25)),
            coord_endpoint_pred_lattice_mode=str(
                loss.get(
                    "coord_endpoint_pred_lattice_mode", "predicted_clean"
                )
            ),
            lattice_endpoint_weight=float(loss.get("lattice_endpoint_weight", 0.0)),
            lattice_normalize_by_dof=bool(loss.get("lattice_normalize_by_dof", True)),
            contact_clash_config=dict(loss.get("contact_clash") or {}),
            bond_geometry_config=dict(loss.get("bond_geometry") or {}),
            lattice_prior_config=lattice_prior_cfg,
            diagnostics_config=dict(cfg.get("diagnostics") or {}),
            uncertainty_weighting_config=dict(
                cfg.get("uncertainty_weighting") or {}
            ),
            self_conditioning_config=dict(
                cfg.get("self_conditioning") or {}
            ),
        )
        self._flow_config = cfg
        self._denoiser_config = denoiser_config
        validate_full_cell_checkpoint_contract(cfg, denoiser_config)
        validate_full_cell_denoiser_instance(denoiser, denoiser_config)
        self._lr = float(lr)
        self._weight_decay = float(weight_decay)
        self._optimizer_name = str(optimizer_name).lower()
        if self._optimizer_name not in {"adam", "adamw", "hybrid_muon"}:
            raise ValueError("optimizer_name must be one of: adam, adamw, hybrid_muon")
        self._muon_config = dict(muon_config or {})
        if self._optimizer_name == "hybrid_muon" and not self._muon_config:
            raise ValueError(
                "optimizer_name=hybrid_muon requires a resolved muon_config"
            )
        if self._optimizer_name != "hybrid_muon" and self._muon_config:
            raise ValueError("muon_config requires optimizer_name=hybrid_muon")
        self._lr_scheduler = dict(lr_scheduler or {})
        self._warmup_steps = int(warmup_steps)
        self._max_grad_norm = (
            None if max_grad_norm is None else float(max_grad_norm)
        )
        self._nominal_batch_size = (
            None if nominal_batch_size is None else int(nominal_batch_size)
        )
        if self._nominal_batch_size is not None and self._nominal_batch_size <= 0:
            raise ValueError("nominal_batch_size must be positive when provided")
        self._validation_strata_sums: Optional[Tensor] = None
        self._validation_strata_counts: Optional[Tensor] = None

    def forward(
        self,
        batch: AsuBatch,
        state_x: Tensor,
        state_q: Tensor,
        time: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return self.flow(batch, state_x, state_q, time)

    @staticmethod
    def _global_contact_metrics(
        metrics: Dict[str, Tensor], *, batch_size: int
    ) -> Dict[str, Tensor]:
        """Reduce contact/clash means with graph-count-correct DDP weighting."""

        required = (
            "loss/contact_clash",
            "loss/clash_repulsion",
            "loss/contact_match",
            "loss/contact_clash_weighted",
            "metric/contact_active_graph_fraction",
            "metric/clash_normalizer_pair_count_mean",
            "metric/contact_target_pair_count_mean",
            "metric/predicted_clash_pair_count_mean",
            "metric/contact_evaluated_pair_count_mean",
            "metric/clash_target_capped_pair_count_mean",
            "metric/contact_periodic_same_instance_count_mean",
            "metric/contact_predicted_periodic_fallback_count_mean",
            "metric/contact_target_periodic_fallback_count_mean",
        )
        if any(name not in metrics for name in required):
            return metrics
        reference = metrics[required[0]]
        graph_count = reference.new_tensor(float(batch_size))
        packed = torch.stack(
            [graph_count]
            + [metrics[name].to(reference) * graph_count for name in required]
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        total_graphs = packed[0].clamp_min(1.0)
        global_metrics = dict(metrics)
        for index, name in enumerate(required, 1):
            global_metrics[name] = (packed[index] / total_graphs).detach()

        clash_normalizer_pair_total = packed[6]
        evaluated_pair_total = packed[9]
        global_metrics["metric/clash_target_capped_pair_fraction"] = (
            packed[10] / evaluated_pair_total.clamp_min(1.0)
        ).detach()
        global_metrics["metric/contact_periodic_same_instance_fraction"] = (
            packed[11] / clash_normalizer_pair_total.clamp_min(1.0)
        ).detach()
        global_metrics["metric/contact_predicted_periodic_fallback_fraction"] = (
            packed[12] / evaluated_pair_total.clamp_min(1.0)
        ).detach()
        global_metrics["metric/contact_target_periodic_fallback_fraction"] = (
            packed[13] / evaluated_pair_total.clamp_min(1.0)
        ).detach()
        return global_metrics

    @property
    def denoiser(self) -> nn.Module:
        return self.flow.denoiser

    @property
    def lattice_prior(self):
        return self.flow.lattice_prior

    def validate_lattice_prior_state(self) -> None:
        """Validate the stateless analytic prior after weight-only bootstrap."""

        if not isinstance(self.flow.lattice_prior, VolumeLatticePrior):
            raise ValueError("Flow Matching requires the analytic VolumeLatticePrior")
        trainable = [
            name
            for name, parameter in self.flow.lattice_prior.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise ValueError(
                f"VolumeLatticePrior must not contain trainable state: {trainable}"
            )

    def validate_finetune_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Validate a compatible checkpoint while permitting a new loss.

        This is intentionally weight-only. Full-state resume continues to use
        :meth:`on_load_checkpoint` and therefore requires an identical objective.
        """

        checkpoint_schema_version = checkpoint.get("checkpoint_schema_version")
        if (
            checkpoint.get("model_family") != MODEL_FAMILY
            or checkpoint_schema_version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
        ):
            raise ValueError(
                f"Unsupported fine-tune checkpoint: expected {MODEL_FAMILY} "
                "with schema version 15 or 16"
            )
        checkpoint_flow_config = checkpoint.get("flow_config")
        checkpoint_denoiser_config = checkpoint.get("denoiser_config")
        if not isinstance(checkpoint_flow_config, dict) or not isinstance(
            checkpoint_denoiser_config, dict
        ):
            raise ValueError(
                "Fine-tune checkpoint requires flow_config and "
                "denoiser_config mappings"
            )
        validate_full_cell_checkpoint_contract(
            checkpoint_flow_config,
            checkpoint_denoiser_config,
            checkpoint_schema_version=int(checkpoint_schema_version),
        )
        validate_full_cell_checkpoint_contract(
            self._flow_config, self._denoiser_config
        )
        validate_full_cell_denoiser_instance(
            self.flow.denoiser, self._denoiser_config
        )
        if _denoiser_contract_signature(
            checkpoint_denoiser_config
        ) != _denoiser_contract_signature(self._denoiser_config):
            raise ValueError(
                "Fine-tune checkpoint denoiser_config does not match the "
                "instantiated model"
            )
        if _flow_finetune_signature(
            checkpoint_flow_config
        ) != _flow_finetune_signature(self._flow_config):
            raise ValueError(
                "Fine-tune checkpoint flow state/conditioning contract does "
                "not match the instantiated model"
            )

    @staticmethod
    def _validation_strata_layout() -> tuple[tuple[str, tuple[str, ...]], ...]:
        return (
            ("chem", tuple(CHEM_GROUPS)),
            ("crystal", tuple(CRYSTAL_SYSTEMS)),
        )

    def on_validation_epoch_start(self) -> None:
        total_groups = sum(len(names) for _, names in self._validation_strata_layout())
        self._validation_strata_sums = torch.zeros(
            total_groups,
            device=self.device,
            dtype=torch.float64,
        )
        self._validation_strata_counts = torch.zeros_like(
            self._validation_strata_sums
        )

    def _accumulate_validation_strata(
        self, stage: str, batch: AsuBatch, per_graph_loss: Optional[Tensor]
    ) -> None:
        if stage != "val" or per_graph_loss is None:
            return
        if (
            self._validation_strata_sums is None
            or self._validation_strata_counts is None
        ):
            self.on_validation_epoch_start()
        values = per_graph_loss.detach().float().view(-1)
        if values.numel() != int(batch.batch_size):
            return
        offset = 0
        for attr, names in (
            ("chem_group_id", CHEM_GROUPS),
            ("crystal_system_id", CRYSTAL_SYSTEMS),
        ):
            ids = getattr(batch, attr, None)
            group_count = len(names)
            if ids is None or ids.numel() != values.numel():
                offset += group_count
                continue
            ids = ids.to(device=values.device).view(-1).long()
            valid = (ids >= 0) & (ids < group_count)
            if valid.any():
                target_ids = ids[valid] + offset
                source_values = values[valid].to(dtype=torch.float64)
                self._validation_strata_sums.scatter_add_(
                    0,
                    target_ids,
                    source_values,
                )
                self._validation_strata_counts.scatter_add_(
                    0,
                    target_ids,
                    torch.ones_like(source_values),
                )
            offset += group_count

    def on_validation_epoch_end(self) -> None:
        sums = self._validation_strata_sums
        counts = self._validation_strata_counts
        if sums is None or counts is None:
            return
        packed = torch.stack((sums, counts), dim=0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        sums, counts = packed[0], packed[1]
        offset = 0
        for prefix, names in self._validation_strata_layout():
            for group_id, name in enumerate(names):
                count = counts[offset + group_id]
                if bool((count > 0).item()):
                    self.log(
                        f"val/{prefix}/{name}/loss",
                        (sums[offset + group_id] / count).float(),
                        sync_dist=False,
                    )
            offset += len(names)
        self._validation_strata_sums = None
        self._validation_strata_counts = None

    def _shared_step(self, stage: str, batch: AsuBatch, batch_idx: int) -> Tensor:
        if stage == "train":
            output = self.flow.training_step(batch)
        else:
            # Validation and test metrics must be comparable across epochs.
            # Isolate and deterministically seed their random corruption without
            # perturbing the training RNG stream.
            device = batch.atom_types.device
            cuda_devices = (
                list(range(torch.cuda.device_count())) if device.type == "cuda" else []
            )
            seed = (
                _EVALUATION_NOISE_SEED
                + int(batch_idx)
                + 1_000_003 * int(self.global_rank)
            )
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                output = self.flow.training_step(batch)
        self.log(
            f"{stage}/loss", output.loss, sync_dist=True, batch_size=batch.batch_size
        )
        logged_metrics = self._global_contact_metrics(
            output.metrics, batch_size=batch.batch_size
        )
        for key, value in logged_metrics.items():
            if key == "loss/total":
                # Identical to the stage loss logged above.
                continue
            global_contact_metric = key in self._GLOBAL_CONTACT_METRICS
            sync_metric = (
                not global_contact_metric
                and (stage != "train" or key in self._TRAIN_SYNC_METRICS)
            )
            self.log(
                f"{stage}/{key}",
                value,
                sync_dist=sync_metric,
                rank_zero_only=global_contact_metric
                or (stage == "train" and not sync_metric),
                batch_size=batch.batch_size,
            )
        self._accumulate_validation_strata(stage, batch, output.per_graph_loss)
        if stage == "train" and self._nominal_batch_size is not None:
            # Dense-attention budgeting may split a nominal batch into smaller
            # microbatches.  Lightning/DistributedDataParallel otherwise gives
            # every microbatch the same weight, over-weighting large crystals.
            # Scaling the mean loss by its sample count preserves a per-crystal
            # objective; the constant nominal denominator only fixes magnitude.
            return output.loss * (
                float(batch.batch_size) / float(self._nominal_batch_size)
            )
        return output.loss

    def training_step(self, batch: AsuBatch, batch_idx: int) -> Tensor:
        return self._shared_step("train", batch, batch_idx)

    def validation_step(self, batch: AsuBatch, batch_idx: int) -> Tensor:
        return self._shared_step("val", batch, batch_idx)

    def test_step(self, batch: AsuBatch, batch_idx: int) -> Tensor:
        return self._shared_step("test", batch, batch_idx)

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        if isinstance(batch, AsuBatch):
            return batch.to(device, non_blocking=True)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _lr_lambda_for_total_steps(self, total_steps: int):
        min_ratio = float(self._lr_scheduler.get("min_lr_ratio", 0.05))

        def lr_lambda(step: int) -> float:
            if step < self._warmup_steps:
                return step / max(1, self._warmup_steps)
            progress = (step - self._warmup_steps) / max(
                1, total_steps - self._warmup_steps
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return lr_lambda

    def configure_optimizers(self) -> Any:
        if self._optimizer_name == "hybrid_muon":
            from sala.training.optimizers import build_hybrid_muon_optimizer

            optimizer, audit = build_hybrid_muon_optimizer(
                self,
                adamw_lr=self._lr,
                adamw_weight_decay=self._weight_decay,
                config=self._muon_config,
            )
            self._optimizer_parameter_audit = audit
            logger.info("Hybrid Muon parameter audit: %s", audit.summary())
        else:
            cls = (
                torch.optim.Adam
                if self._optimizer_name == "adam"
                else torch.optim.AdamW
            )
            optimizer = cls(
                self.parameters(), lr=self._lr, weight_decay=self._weight_decay
            )
        scheduler_type = str(
            self._lr_scheduler.get("type", "cosine_warmup")
        ).lower()
        if scheduler_type != "cosine_warmup":
            raise ValueError(f"Unsupported lr scheduler type: {scheduler_type}")
        total_steps = 200_000
        try:
            if self.trainer.estimated_stepping_batches is not None:
                total_steps = int(self.trainer.estimated_stepping_batches)
        except RuntimeError:
            pass
        self._scheduler_total_steps = total_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, self._lr_lambda_for_total_steps(total_steps)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Clip once and reject non-finite gradients before optimizer state mutates."""

        algorithm = "norm" if gradient_clip_algorithm is None else str(
            gradient_clip_algorithm
        ).lower()
        if algorithm != "norm":
            raise ValueError("SALA supports norm gradient clipping only")
        max_norm = (
            self._max_grad_norm
            if gradient_clip_val is None
            else float(gradient_clip_val)
        )
        if max_norm is None or float(max_norm) <= 0.0:
            return
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if not parameters:
            return
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=max_norm,
            error_if_nonfinite=True,
            foreach=None,
        )
        self.log(
            "train/metric/grad_norm",
            grad_norm.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
            rank_zero_only=True,
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        finite_flags = [
            torch.isfinite(parameter.detach()).all()
            for parameter in self.parameters()
        ]
        if finite_flags and not bool(torch.stack(finite_flags).all().item()):
            raise FloatingPointError(
                "Refusing to save a checkpoint with non-finite model parameters"
            )
        validate_full_cell_checkpoint_contract(
            self._flow_config, self._denoiser_config
        )
        validate_full_cell_denoiser_instance(
            self.flow.denoiser, self._denoiser_config
        )
        checkpoint_metadata = {
            "model_family": MODEL_FAMILY,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "flow_config": self._flow_config,
            "denoiser_config": self._denoiser_config,
            "optimizer_name": self._optimizer_name,
        }
        if self._optimizer_name == "hybrid_muon":
            audit = getattr(self, "_optimizer_parameter_audit", None)
            if audit is None:
                from sala.training.optimizers import partition_optimizer_parameters

                audit = partition_optimizer_parameters(
                    self,
                    min_dim=int(self._muon_config["min_dim"]),
                    max_aspect_ratio=float(self._muon_config["max_aspect_ratio"]),
                ).audit
            if not any(".attention." in name for name in audit.muon_names):
                raise RuntimeError(
                    "Flow Matching optimizer audit assigned no attention matrices to Muon"
                )
            checkpoint_metadata["optimizer_parameter_audit"] = audit.to_dict()
        checkpoint.update(checkpoint_metadata)
        if hasattr(self, "_scheduler_total_steps"):
            checkpoint["scheduler_total_steps"] = self._scheduler_total_steps

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint_schema_version = checkpoint.get("checkpoint_schema_version")
        if (
            checkpoint.get("model_family") != MODEL_FAMILY
            or checkpoint_schema_version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
        ):
            raise ValueError(
                f"Unsupported checkpoint: expected {MODEL_FAMILY} with schema "
                "version 15 or 16. Earlier schemas use an incompatible "
                "Transformer, optimizer, EMA, or flow objective."
            )
        checkpoint_flow_config = checkpoint.get("flow_config")
        checkpoint_denoiser_config = checkpoint.get("denoiser_config")
        if not isinstance(checkpoint_flow_config, dict) or not isinstance(
            checkpoint_denoiser_config, dict
        ):
            raise ValueError(
                "Flow Matching checkpoints require flow_config and "
                "denoiser_config mappings"
            )
        validate_full_cell_checkpoint_contract(
            checkpoint_flow_config,
            checkpoint_denoiser_config,
            checkpoint_schema_version=int(checkpoint_schema_version),
        )
        validate_full_cell_checkpoint_contract(
            self._flow_config, self._denoiser_config
        )
        validate_full_cell_denoiser_instance(
            self.flow.denoiser, self._denoiser_config
        )
        if _denoiser_contract_signature(
            checkpoint_denoiser_config
        ) != _denoiser_contract_signature(self._denoiser_config):
            raise ValueError(
                "Checkpoint denoiser_config does not match the instantiated model"
            )
        if _flow_contract_signature(
            checkpoint_flow_config
        ) != _flow_contract_signature(self._flow_config):
            raise ValueError(
                "Checkpoint flow state/path contract does not match the "
                "instantiated model"
            )
        self._resumed_scheduler_total_steps = checkpoint.get("scheduler_total_steps")


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS",
    "FlowMatchingLightningModule",
    "validate_full_cell_denoiser_instance",
    "validate_full_cell_checkpoint_contract",
]
