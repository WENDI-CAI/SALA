from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from sala.training.checkpoint_compat import (
    install_torch_compile_state_dict_compatibility,
)
from sala.training.ema import EMACallback, select_ema_shadow_state_dict

try:
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf

    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict

from sala.core.config_resolve import (
    CHECKPOINT_CONFIG_SECTIONS,
    DATA_CONFIG_SECTIONS,
    TRAINER_CONFIG_SECTIONS,
    flatten_grouped_config,
)
from sala.flow_matching.losses import (
    resolve_bond_geometry_config,
    resolve_contact_clash_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _to_plain_dict(config: Any, name: str) -> Dict[str, Any]:
    if config is None:
        return {}
    if HYDRA_AVAILABLE and OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, dict):
        raise TypeError(f"{name} must be a mapping, got {type(config).__name__}")
    return dict(config)


def _merge_nested_config(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively merge a partial normalized model-config override."""

    merged = dict(base)
    for key, value in dict(override).items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_nested_config(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_optimizer_config(optim_cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = _to_plain_dict(optim_cfg, "optim")
    unsupported = sorted(
        set(raw)
        - {
            "optimizer",
            "lr",
            "weight_decay",
            "lr_scheduler",
            "warmup_steps",
            "muon",
        }
    )
    if unsupported:
        raise ValueError(f"Unsupported optim configuration keys: {unsupported}")

    optimizer_name = str(raw.get("optimizer", "adamw")).lower()
    if optimizer_name not in {"adam", "adamw", "hybrid_muon"}:
        raise ValueError("optim.optimizer must be one of: adam, adamw, hybrid_muon")
    lr = float(raw.get("lr", 1e-4))
    weight_decay = float(raw.get("weight_decay", 0.01))
    warmup_steps = int(raw.get("warmup_steps", 10000))
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("optim.lr must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("optim.weight_decay must be finite and non-negative")
    if warmup_steps < 0:
        raise ValueError("optim.warmup_steps must be non-negative")

    scheduler = _to_plain_dict(raw.get("lr_scheduler"), "optim.lr_scheduler")
    unsupported_scheduler = sorted(set(scheduler) - {"type", "min_lr_ratio"})
    if unsupported_scheduler:
        raise ValueError(
            f"Unsupported optim.lr_scheduler keys: {unsupported_scheduler}"
        )
    scheduler_type = str(scheduler.get("type", "cosine_warmup")).lower()
    if scheduler_type != "cosine_warmup":
        raise ValueError("optim.lr_scheduler.type must be cosine_warmup")
    resolved_scheduler: Dict[str, Any] = {"type": scheduler_type}
    min_lr_ratio = float(scheduler.get("min_lr_ratio", 0.05))
    if not math.isfinite(min_lr_ratio) or not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("optim.lr_scheduler.min_lr_ratio must be in [0, 1]")
    resolved_scheduler["min_lr_ratio"] = min_lr_ratio

    muon_raw = _to_plain_dict(raw.get("muon"), "optim.muon")
    unsupported_muon = sorted(
        set(muon_raw)
        - {
            "backend",
            "lr",
            "momentum",
            "nesterov",
            "ns_steps",
            "weight_decay",
            "min_dim",
            "max_aspect_ratio",
            "adamw_betas",
            "adamw_eps",
            "adjust_lr_fn",
        }
    )
    if unsupported_muon:
        raise ValueError(f"Unsupported optim.muon keys: {unsupported_muon}")
    if muon_raw and optimizer_name != "hybrid_muon":
        raise ValueError("optim.muon is only valid when optim.optimizer=hybrid_muon")

    resolved_muon: Dict[str, Any] = {}
    if optimizer_name == "hybrid_muon":
        backend = str(muon_raw.get("backend", "internal")).lower()
        if backend != "internal":
            raise ValueError(
                "optim.muon.backend must be 'internal'; SALA does not depend on "
                "an optional external Muon package"
            )
        muon_lr = float(muon_raw.get("lr", 5e-4))
        momentum = float(muon_raw.get("momentum", 0.95))
        ns_steps = int(muon_raw.get("ns_steps", 5))
        muon_weight_decay = float(muon_raw.get("weight_decay", weight_decay))
        min_dim = int(muon_raw.get("min_dim", 128))
        max_aspect_ratio = float(muon_raw.get("max_aspect_ratio", 8.0))
        adamw_betas_raw = muon_raw.get("adamw_betas", [0.9, 0.95])
        if not isinstance(adamw_betas_raw, (list, tuple)) or len(adamw_betas_raw) != 2:
            raise ValueError("optim.muon.adamw_betas must contain two values")
        adamw_betas = [float(value) for value in adamw_betas_raw]
        adamw_eps = float(muon_raw.get("adamw_eps", 1e-8))
        adjust_lr_fn = str(
            muon_raw.get("adjust_lr_fn", "match_rms_adamw")
        ).lower()
        if not math.isfinite(muon_lr) or muon_lr <= 0.0:
            raise ValueError("optim.muon.lr must be finite and positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("optim.muon.momentum must be in [0, 1)")
        if not 1 <= ns_steps <= 10:
            raise ValueError("optim.muon.ns_steps must be in [1, 10]")
        if not math.isfinite(muon_weight_decay) or muon_weight_decay < 0.0:
            raise ValueError("optim.muon.weight_decay must be non-negative")
        if min_dim < 2:
            raise ValueError("optim.muon.min_dim must be at least 2")
        if not math.isfinite(max_aspect_ratio) or max_aspect_ratio < 1.0:
            raise ValueError("optim.muon.max_aspect_ratio must be at least 1")
        if any(not 0.0 <= beta < 1.0 for beta in adamw_betas):
            raise ValueError("optim.muon.adamw_betas must be in [0, 1)")
        if not math.isfinite(adamw_eps) or adamw_eps <= 0.0:
            raise ValueError("optim.muon.adamw_eps must be finite and positive")
        if adjust_lr_fn not in {"original", "match_rms_adamw"}:
            raise ValueError(
                "optim.muon.adjust_lr_fn must be original or match_rms_adamw"
            )
        resolved_muon = {
            "backend": backend,
            "lr": muon_lr,
            "momentum": momentum,
            "nesterov": bool(muon_raw.get("nesterov", True)),
            "ns_steps": ns_steps,
            "weight_decay": muon_weight_decay,
            "min_dim": min_dim,
            "max_aspect_ratio": max_aspect_ratio,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
            "adjust_lr_fn": adjust_lr_fn,
        }

    return {
        "optimizer": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "lr_scheduler": resolved_scheduler,
        "warmup_steps": warmup_steps,
        "muon": resolved_muon,
    }


def _resolve_prior_temperature(raw: Any) -> Dict[str, float]:
    """Validate the low-temperature sampling scales.

    Flow-2M integrates a deterministic ODE, so the prior draw is the only source
    of variation between candidates for fixed conditioning. Scaling it below 1
    concentrates samples near the prior mean, trading diversity for
    predictability. Values above 1 would sample outside the trained prior support
    and are rejected.
    """

    config = _to_plain_dict(raw, "flow.sampler.prior_temperature")
    unsupported = sorted(set(config) - {"coord", "lattice"})
    if unsupported:
        raise ValueError(
            f"Unsupported flow.sampler.prior_temperature keys: {unsupported}"
        )
    resolved: Dict[str, float] = {}
    for key in ("coord", "lattice"):
        value = float(config.get(key, 1.0))
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(
                f"flow.sampler.prior_temperature.{key} must lie in (0, 1]"
            )
        resolved[key] = value
    return resolved


def _resolve_flow_config(flow_cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = _to_plain_dict(flow_cfg, "flow")
    allowed = {
        "state",
        "path",
        "time_sampling",
        "time_condition",
        "geometry",
        "self_conditioning",
        "loss",
        "prior",
        "sampler",
        "uncertainty_weighting",
        "diagnostics",
    }
    unsupported = sorted(set(raw) - allowed)
    if unsupported:
        raise ValueError(
            f"Unsupported Flow Matching configuration keys: {unsupported}"
        )

    state = _to_plain_dict(raw.get("state"), "flow.state")
    unsupported_state = sorted(
        set(state)
        - {"coordinate", "lattice", "model", "output", "coord_scale"}
    )
    if unsupported_state:
        raise ValueError(f"Unsupported flow.state keys: {unsupported_state}")
    resolved_state = {
        "coordinate": str(state.get("coordinate", "canonical_cartesian")).lower(),
        "lattice": str(state.get("lattice", "constrained_q")).lower(),
        "model": str(
            state.get("model", "normalized_cartesian_whitened_q")
        ).lower(),
        "output": str(state.get("output", "fractional_unwrapped")).lower(),
        "coord_scale": float(state.get("coord_scale", 8.0)),
    }
    if resolved_state != {
        "coordinate": "canonical_cartesian",
        "lattice": "constrained_q",
        "model": "normalized_cartesian_whitened_q",
        "output": "fractional_unwrapped",
        "coord_scale": 8.0,
    }:
        raise ValueError(
            "Flow Matching requires the locked canonical Cartesian/whitened-q "
            "state contract with coord_scale=8"
        )

    path_cfg = _to_plain_dict(raw.get("path"), "flow.path")
    unsupported_path = sorted(
        set(path_cfg) - {"type", "prediction", "data_time", "noise_time"}
    )
    if unsupported_path:
        raise ValueError(f"Unsupported flow.path keys: {unsupported_path}")
    resolved_path = {
        "type": str(path_cfg.get("type", "rectified_linear")).lower(),
        "prediction": str(path_cfg.get("prediction", "velocity")).lower(),
        "data_time": float(path_cfg.get("data_time", 0.0)),
        "noise_time": float(path_cfg.get("noise_time", 1.0)),
    }
    if resolved_path != {
        "type": "rectified_linear",
        "prediction": "velocity",
        "data_time": 0.0,
        "noise_time": 1.0,
    }:
        raise ValueError(
            "Flow Matching supports only data-at-zero linear rectified flow "
            "with direct velocity prediction"
        )

    time_sampling = _to_plain_dict(
        raw.get("time_sampling"), "flow.time_sampling"
    )
    unsupported_time = sorted(
        set(time_sampling) - {"type", "mean", "std", "clamp"}
    )
    if unsupported_time:
        raise ValueError(f"Unsupported flow.time_sampling keys: {unsupported_time}")
    resolved_time = {
        "type": str(time_sampling.get("type", "logit_normal")).lower(),
        "mean": float(time_sampling.get("mean", 0.0)),
        "std": float(time_sampling.get("std", 1.0)),
        "clamp": float(time_sampling.get("clamp", 1.0e-5)),
    }
    if resolved_time["type"] != "logit_normal":
        raise ValueError("flow.time_sampling.type must be logit_normal")
    if not math.isfinite(resolved_time["mean"]):
        raise ValueError("flow.time_sampling.mean must be finite")
    if not math.isfinite(resolved_time["std"]) or resolved_time["std"] < 0.0:
        raise ValueError("flow.time_sampling.std must be finite and non-negative")
    if not math.isfinite(resolved_time["clamp"]) or not 0.0 < resolved_time["clamp"] < 0.5:
        raise ValueError("flow.time_sampling.clamp must be in (0, 0.5)")

    time_condition = _to_plain_dict(
        raw.get("time_condition"), "flow.time_condition"
    )
    unsupported_condition = sorted(set(time_condition) - {"type", "scale"})
    if unsupported_condition:
        raise ValueError(
            f"Unsupported flow.time_condition keys: {unsupported_condition}"
        )
    resolved_condition = {
        "type": str(time_condition.get("type", "scaled_time")).lower(),
        "scale": float(time_condition.get("scale", 1000.0)),
    }
    if resolved_condition["type"] != "scaled_time":
        raise ValueError("flow.time_condition.type must be scaled_time")
    if not math.isfinite(resolved_condition["scale"]) or resolved_condition["scale"] <= 0.0:
        raise ValueError("flow.time_condition.scale must be finite and positive")

    geometry = _to_plain_dict(raw.get("geometry"), "flow.geometry")
    unsupported_geometry = sorted(
        set(geometry) - {"stabilization_start", "latent_clip"}
    )
    if unsupported_geometry:
        raise ValueError(f"Unsupported flow.geometry keys: {unsupported_geometry}")
    resolved_geometry = {
        "stabilization_start": float(
            geometry.get("stabilization_start", 0.5)
        ),
        "latent_clip": float(geometry.get("latent_clip", 1.5)),
    }
    if not 0.0 <= resolved_geometry["stabilization_start"] < 1.0:
        raise ValueError("flow.geometry.stabilization_start must be in [0, 1)")
    if not math.isfinite(resolved_geometry["latent_clip"]) or resolved_geometry["latent_clip"] <= 0.0:
        raise ValueError("flow.geometry.latent_clip must be finite and positive")

    self_conditioning = _to_plain_dict(
        raw.get("self_conditioning"), "flow.self_conditioning"
    )
    unsupported_self_conditioning = sorted(
        set(self_conditioning)
        - {
            "enabled",
            "train_probability",
            "prepass_selected_only",
            "detach",
            "sampling_enabled",
            "representation",
        }
    )
    if unsupported_self_conditioning:
        raise ValueError(
            "Unsupported flow.self_conditioning keys: "
            f"{unsupported_self_conditioning}"
        )
    resolved_self_conditioning = {
        "enabled": bool(self_conditioning.get("enabled", False)),
        "train_probability": float(
            self_conditioning.get("train_probability", 0.5)
        ),
        "prepass_selected_only": bool(
            self_conditioning.get("prepass_selected_only", True)
        ),
        "detach": bool(self_conditioning.get("detach", True)),
        "sampling_enabled": bool(
            self_conditioning.get("sampling_enabled", True)
        ),
        "representation": str(
            self_conditioning.get(
                "representation", "normalized_clean_residual"
            )
        ).lower(),
    }
    if not 0.0 <= resolved_self_conditioning["train_probability"] <= 1.0:
        raise ValueError(
            "flow.self_conditioning.train_probability must be in [0, 1]"
        )
    if not resolved_self_conditioning["detach"]:
        raise ValueError("flow.self_conditioning.detach must be true")
    if (
        resolved_self_conditioning["representation"]
        != "normalized_clean_residual"
    ):
        raise ValueError(
            "flow.self_conditioning.representation must be "
            "normalized_clean_residual"
        )

    loss = _to_plain_dict(raw.get("loss"), "flow.loss")
    unsupported_loss = sorted(
        set(loss)
        - {
            "coord_velocity_weight",
            "coord_velocity_mode",
            "lattice_velocity_weight",
            "full_cell_endpoint_weight",
            "coord_endpoint_pred_lattice_mode",
            "lattice_endpoint_weight",
            "lattice_normalize_by_dof",
            "contact_clash",
            "bond_geometry",
        }
    )
    if unsupported_loss:
        raise ValueError(f"Unsupported flow.loss keys: {unsupported_loss}")
    resolved_loss = {
        "coord_velocity_weight": float(loss.get("coord_velocity_weight", 1.0)),
        "coord_velocity_mode": str(
            loss.get("coord_velocity_mode", "asu")
        ).strip().lower(),
        "lattice_velocity_weight": float(
            loss.get("lattice_velocity_weight", 1.0)
        ),
        "full_cell_endpoint_weight": float(
            loss.get("full_cell_endpoint_weight", 0.25)
        ),
        "coord_endpoint_pred_lattice_mode": str(
            loss.get(
                "coord_endpoint_pred_lattice_mode", "predicted_clean"
            )
        ).strip().lower(),
        "lattice_endpoint_weight": float(
            loss.get("lattice_endpoint_weight", 0.0)
        ),
        "lattice_normalize_by_dof": bool(
            loss.get("lattice_normalize_by_dof", True)
        ),
        "contact_clash": resolve_contact_clash_config(
            _to_plain_dict(
                loss.get("contact_clash"), "flow.loss.contact_clash"
            )
        ),
        "bond_geometry": resolve_bond_geometry_config(
            _to_plain_dict(
                loss.get("bond_geometry"), "flow.loss.bond_geometry"
            )
        ),
    }
    if resolved_loss["coord_velocity_mode"] not in {
        "asu",
        "full_cell_orbit",
    }:
        raise ValueError(
            "flow.loss.coord_velocity_mode must be asu or full_cell_orbit"
        )
    if resolved_loss["coord_endpoint_pred_lattice_mode"] not in {
        "predicted_clean",
        "input",
    }:
        raise ValueError(
            "flow.loss.coord_endpoint_pred_lattice_mode must be "
            "predicted_clean or input"
        )
    if any(
        not math.isfinite(value) or value < 0.0
        for key, value in resolved_loss.items()
        if key
        not in {
            "coord_velocity_mode",
            "coord_endpoint_pred_lattice_mode",
            "lattice_normalize_by_dof",
            "contact_clash",
            "bond_geometry",
        }
    ):
        raise ValueError("flow.loss weights must be finite and non-negative")

    prior = _to_plain_dict(raw.get("prior"), "flow.prior")
    if sorted(set(prior) - {"lattice"}):
        raise ValueError("flow.prior supports only the lattice section")
    lattice_prior = _to_plain_dict(prior.get("lattice"), "flow.prior.lattice")
    if sorted(set(lattice_prior) - {"type", "length_log_std", "angle_std"}):
        raise ValueError(
            "flow.prior.lattice supports type, length_log_std, and angle_std"
        )
    resolved_prior = {
        "lattice": {
            "type": str(lattice_prior.get("type", "volume")).lower(),
            "length_log_std": float(lattice_prior.get("length_log_std", 0.30)),
            "angle_std": float(lattice_prior.get("angle_std", 0.20)),
        }
    }
    if resolved_prior["lattice"]["type"] != "volume":
        raise ValueError("flow.prior.lattice.type must be volume")
    if (
        not math.isfinite(resolved_prior["lattice"]["length_log_std"])
        or resolved_prior["lattice"]["length_log_std"] <= 0.0
        or not math.isfinite(resolved_prior["lattice"]["angle_std"])
        or resolved_prior["lattice"]["angle_std"] <= 0.0
    ):
        raise ValueError("flow.prior.lattice standard deviations must be positive")

    sampler = _to_plain_dict(raw.get("sampler"), "flow.sampler")
    unsupported_sampler = sorted(
        set(sampler)
        - {"solver", "num_steps", "time_grid", "count_nfe", "prior_temperature"}
    )
    if unsupported_sampler:
        raise ValueError(f"Unsupported flow.sampler keys: {unsupported_sampler}")
    resolved_sampler = {
        "solver": str(sampler.get("solver", "flow_2m")).lower(),
        "num_steps": int(sampler.get("num_steps", 24)),
        "time_grid": str(sampler.get("time_grid", "uniform")).lower(),
        "prior_temperature": _resolve_prior_temperature(
            sampler.get("prior_temperature")
        ),
        "count_nfe": bool(sampler.get("count_nfe", True)),
    }
    if resolved_sampler["solver"] != "flow_2m":
        raise ValueError("flow.sampler.solver must be flow_2m")
    if resolved_sampler["time_grid"] != "uniform":
        raise ValueError("flow.sampler.time_grid must be uniform")
    if resolved_sampler["num_steps"] <= 0:
        raise ValueError("flow.sampler.num_steps must be positive")

    uncertainty = _to_plain_dict(
        raw.get("uncertainty_weighting"), "flow.uncertainty_weighting"
    )
    unsupported_uncertainty = sorted(set(uncertainty) - {"enabled", "channels", "clamp"})
    if unsupported_uncertainty:
        raise ValueError(
            f"Unsupported flow.uncertainty_weighting keys: {unsupported_uncertainty}"
        )
    clamp = list(uncertainty.get("clamp", [-2.0, 2.0]))
    if len(clamp) != 2:
        raise ValueError("flow.uncertainty_weighting.clamp must contain two values")
    clamp = [float(clamp[0]), float(clamp[1])]
    if not all(math.isfinite(value) for value in clamp) or not clamp[0] < clamp[1]:
        raise ValueError(
            "flow.uncertainty_weighting.clamp must be finite and increasing"
        )
    resolved_uncertainty = {
        "enabled": bool(uncertainty.get("enabled", True)),
        "channels": int(uncertainty.get("channels", 128)),
        "clamp": clamp,
    }
    if resolved_uncertainty["channels"] <= 0:
        raise ValueError("flow.uncertainty_weighting.channels must be positive")

    diagnostics = _to_plain_dict(raw.get("diagnostics"), "flow.diagnostics")
    if sorted(set(diagnostics) - {"time_bucket_edges"}):
        raise ValueError("flow.diagnostics supports only time_bucket_edges")
    edges = [
        float(value)
        for value in diagnostics.get(
            "time_bucket_edges", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        )
    ]
    if (
        len(edges) < 2
        or edges[0] != 0.0
        or edges[-1] != 1.0
        or any(right <= left for left, right in zip(edges[:-1], edges[1:]))
    ):
        raise ValueError(
            "flow.diagnostics.time_bucket_edges must increase strictly from 0 to 1"
        )

    return {
        "state": resolved_state,
        "path": resolved_path,
        "time_sampling": resolved_time,
        "time_condition": resolved_condition,
        "geometry": resolved_geometry,
        "self_conditioning": resolved_self_conditioning,
        "loss": resolved_loss,
        "prior": resolved_prior,
        "sampler": resolved_sampler,
        "uncertainty_weighting": resolved_uncertainty,
        "diagnostics": {"time_bucket_edges": edges},
    }


def _require_full_cell_architecture(model_cfg: Dict[str, Any]) -> None:
    removed_aliases = sorted(set(model_cfg) & {"model_type", "depth", "pair_dim"})
    pair_cfg = model_cfg.get("pair")
    if isinstance(pair_cfg, dict) and "pair_dim" in pair_cfg:
        removed_aliases.append("pair.pair_dim")
    if isinstance(pair_cfg, dict) and "max_image_radius" in pair_cfg:
        removed_aliases.append("pair.max_image_radius")
    if removed_aliases:
        raise ValueError(
            "Removed model config aliases: " + ", ".join(removed_aliases)
        )
    architecture = model_cfg.get("architecture")
    if architecture is None:
        raise ValueError(
            "model.architecture is required and must be "
            "'full_cell_crystal_dit_flow'"
        )
    if str(architecture).lower() != "full_cell_crystal_dit_flow":
        raise ValueError(
            "Only model.architecture='full_cell_crystal_dit_flow' is supported; "
            f"received {architecture!r}"
        )


def _resolve_model_self_conditioning_config(
    model_cfg: Dict[str, Any],
) -> Dict[str, bool]:
    self_conditioning = _to_plain_dict(
        model_cfg.get("self_conditioning"), "model.self_conditioning"
    )
    unsupported = sorted(set(self_conditioning) - {"enabled"})
    if unsupported:
        raise ValueError(
            f"Unsupported model.self_conditioning keys: {unsupported}"
        )
    return {"enabled": bool(self_conditioning.get("enabled", False))}


def build_model_from_config(model_cfg: Dict[str, Any]):
    from sala.models.crystal_dit import FullCellCrystalDiT

    _require_full_cell_architecture(model_cfg)
    embedding = dict(model_cfg.get("embedding") or {})
    attention = dict(model_cfg.get("attention") or {})
    pair = dict(model_cfg.get("pair") or {})
    orbit = dict(model_cfg.get("orbit") or {})
    self_conditioning = _resolve_model_self_conditioning_config(model_cfg)
    return FullCellCrystalDiT(
        d_model=int(model_cfg.get("d_model", 512)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        num_layers=int(model_cfg.get("num_layers", 24)),
        d_ffn=int(model_cfg.get("d_ffn", 2816)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        attn_dropout=float(attention.get("attn_dropout", 0.0)),
        use_flash=bool(attention.get("use_flash", True)),
        qk_norm=bool(attention.get("qk_norm", True)),
        attention_backend=str(attention.get("backend", "auto")),
        num_elements=int(model_cfg.get("num_elements", 119)),
        num_hybridizations=int(embedding.get("num_hybridizations", 8)),
        formal_charge_range=int(embedding.get("formal_charge_range", 16)),
        num_covalent_layers=int(embedding.get("num_covalent_layers", 2)),
        num_fourier_freqs=int(embedding.get("num_fourier_freqs", 4)),
        d_pair=int(pair.get("d_pair", 128)),
        pair_num_rbf=int(pair.get("num_rbf", 32)),
        pair_rbf_cutoff=float(pair.get("rbf_cutoff", 12.0)),
        inter_component_cutoff=float(pair.get("inter_component_cutoff", 8.0)),
        max_bond_types=int(pair.get("max_bond_types", 16)),
        max_image_candidates=int(pair.get("max_image_candidates", 65536)),
        pair_chunk_size=int(pair.get("pair_chunk_size", 1024)),
        image_chunk_size=int(pair.get("image_chunk_size", 256)),
        pair_feature_chunk_size=int(pair.get("feature_chunk_size", 65536)),
        cuda_static_image_radius=int(pair.get("cuda_static_image_radius", 2)),
        cuda_dynamic_image_radius=bool(pair.get("cuda_dynamic_image_radius", False)),
        pair_bias_enabled=bool(pair.get("enabled", True)),
        self_conditioning_enabled=self_conditioning["enabled"],
        use_gradient_checkpointing=bool(
            model_cfg.get("use_gradient_checkpointing", True)
        ),
        validate_general_position=bool(orbit.get("require_general_position", True)),
        special_position_tolerance=float(
            orbit.get("special_position_tolerance", 1.0e-5)
        ),
    )


def build_denoiser_config(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    _require_full_cell_architecture(model_cfg)
    self_conditioning = _resolve_model_self_conditioning_config(model_cfg)
    pair = dict(model_cfg.get("pair") or {})
    pair["enabled"] = bool(pair.get("enabled", True))
    return {
        "architecture": "full_cell_crystal_dit_flow",
        "d_model": int(model_cfg.get("d_model", 512)),
        "num_heads": int(model_cfg.get("num_heads", 8)),
        "num_layers": int(model_cfg.get("num_layers", 24)),
        "d_ffn": int(model_cfg.get("d_ffn", 2816)),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        "num_elements": int(model_cfg.get("num_elements", 119)),
        "use_gradient_checkpointing": bool(
            model_cfg.get("use_gradient_checkpointing", True)
        ),
        "embedding": dict(model_cfg.get("embedding") or {}),
        "attention": dict(model_cfg.get("attention") or {}),
        "pair": pair,
        "orbit": dict(model_cfg.get("orbit") or {}),
        "self_conditioning": self_conditioning,
        "atom_coord_repr": "canonical_cartesian",
        "lattice_parameterization": "constrained_q",
        "component_representation": "base_instance_type_v2",
    }



def build_datamodule_from_config(
    data_cfg: Dict[str, Any],
):
    from sala.data.datasets.datamodule import AsuDataModule

    classes = data_cfg.get("classes")
    if isinstance(classes, str):
        classes = [item.strip() for item in classes.split(",") if item.strip()]
    spacegroups = data_cfg.get("spacegroups")
    if isinstance(spacegroups, str):
        spacegroups = [
            int(item.strip()) for item in spacegroups.split(",") if item.strip()
        ]

    pt_root = data_cfg.get("pt_root")
    if not pt_root:
        raise ValueError("data.pt_root is required")
    return AsuDataModule(
        pt_root=pt_root,
        batch_size=data_cfg.get("batch_size", 8),
        num_workers=data_cfg.get("num_workers", 0),
        seed=data_cfg.get("seed", 42),
        index_path=data_cfg.get("index_path"),
        c_count_min=data_cfg.get("c_count_min"),
        c_count_max=data_cfg.get("c_count_max"),
        classes=classes,
        include_unknown=data_cfg.get("include_unknown", True),
        spacegroups=spacegroups,
        is_neutral=data_cfg.get("is_neutral"),
        has_metal=data_cfg.get("has_metal"),
        n_heavy_atoms_min=data_cfg.get("n_heavy_atoms_min"),
        n_heavy_atoms_max=data_cfg.get("n_heavy_atoms_max"),
        n_full_atoms_min=data_cfg.get("n_full_atoms_min"),
        n_full_atoms_max=data_cfg.get("n_full_atoms_max"),
        n_components_min=data_cfg.get("n_components_min"),
        n_components_max=data_cfg.get("n_components_max"),
        shard_cache_size=data_cfg.get("shard_cache_size", 1),
        shard_suffix=data_cfg.get("shard_suffix", "") or "",
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=data_cfg.get("persistent_workers", False),
        prefetch_factor=data_cfg.get("prefetch_factor"),
        sample_weight_no_metal=data_cfg.get("sample_weight_no_metal"),
        sample_weight_has_metal=data_cfg.get("sample_weight_has_metal"),
        atom_bucket_edges=data_cfg.get("atom_bucket_edges"),
        shard_local_batches=data_cfg.get("shard_local_batches", False),
        max_dense_attention_cost=data_cfg.get("max_dense_attention_cost"),
        sample_weights_path=data_cfg.get("sample_weights_path"),
        drop_last_train_batches=bool(
            data_cfg.get("drop_last_train_batches", False)
        ),
        distributed_drop_last_batches=bool(
            data_cfg.get("distributed_drop_last_batches", True)
        ),
        asu_augmentation_enabled=data_cfg.get("asu_augmentation_enabled", True),
        asu_augmentation_probability=data_cfg.get("asu_augmentation_probability", 0.5),
    )


def build_lightning_module_from_config(
    denoiser,
    flow_cfg: Dict[str, Any],
    optim_cfg: Dict[str, Any],
    denoiser_config: Optional[Dict[str, Any]] = None,
    gradient_clip_val: Optional[float] = 1.0,
    nominal_batch_size: Optional[int] = None,
):
    from sala.flow_matching.lightning import FlowMatchingLightningModule

    resolved_flow = _resolve_flow_config(flow_cfg)
    resolved_optim = _resolve_optimizer_config(optim_cfg or {})

    return FlowMatchingLightningModule(
        denoiser=denoiser,
        flow_config=resolved_flow,
        denoiser_config=denoiser_config,
        lr=resolved_optim["lr"],
        weight_decay=resolved_optim["weight_decay"],
        optimizer_name=resolved_optim["optimizer"],
        muon_config=resolved_optim["muon"],
        lr_scheduler=resolved_optim["lr_scheduler"],
        warmup_steps=resolved_optim["warmup_steps"],
        max_grad_norm=(
            None if gradient_clip_val is None else float(gradient_clip_val)
        ),
        nominal_batch_size=nominal_batch_size,
    )


def build_callbacks(
    checkpoint_cfg: Dict[str, Any],
    trainer_cfg: Dict[str, Any],
    output_dir: str,
):
    # EMA must observe an optimizer update before a step-triggered checkpoint
    # serializes its shadow state.
    callbacks = [
        EMACallback(
            profile_stds=trainer_cfg.get("ema_profile_stds", [0.05, 0.10]),
            validation_profile_std=trainer_cfg.get("ema_validation_profile_std"),
        )
    ]
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    monitored = checkpoint_cfg.get("monitor", "val/loss")
    # Include the monitored metric in each checkpoint filename.
    monitor_label = str(monitored).replace("/", "_")
    checkpoint_kwargs = dict(
        dirpath=checkpoint_dir,
        filename=(
            "epoch={epoch:03d}-step={step}-"
            + monitor_label
            + "={"
            + str(monitored)
            + ":.4f}"
        ),
        save_top_k=checkpoint_cfg.get("save_top_k", 3),
        monitor=monitored,
        mode=checkpoint_cfg.get("mode", "min"),
        save_last=checkpoint_cfg.get("save_last", True),
        save_weights_only=checkpoint_cfg.get("save_weights_only", False),
        auto_insert_metric_name=False,
    )
    every_n_train_steps = checkpoint_cfg.get("every_n_train_steps")
    if every_n_train_steps is not None and int(every_n_train_steps) > 0:
        # Lightning requires mutually exclusive step and epoch save schedules.
        checkpoint_kwargs["every_n_train_steps"] = int(every_n_train_steps)
    else:
        every_n_epochs = checkpoint_cfg.get("every_n_epochs", 1)
        checkpoint_kwargs["every_n_epochs"] = (
            None if every_n_epochs is None else int(every_n_epochs)
        )
    callbacks.append(ModelCheckpoint(**checkpoint_kwargs))

    last_every_n_train_steps = checkpoint_cfg.get("last_every_n_train_steps")
    if (
        last_every_n_train_steps is not None
        and int(last_every_n_train_steps) > 0
    ):
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="last-step={step:09d}",
                monitor=None,
                save_top_k=0,
                save_last=True,
                every_n_train_steps=int(last_every_n_train_steps),
                every_n_epochs=None,
                save_on_train_epoch_end=False,
                save_weights_only=checkpoint_cfg.get("save_weights_only", False),
                auto_insert_metric_name=False,
                enable_version_counter=False,
            )
        )

    archive_every_n_train_steps = checkpoint_cfg.get("archive_every_n_train_steps")
    if (
        archive_every_n_train_steps is not None
        and int(archive_every_n_train_steps) > 0
    ):
        # Retain periodic snapshots independently of the monitored metric.
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="archive-step={step:09d}",
                monitor=None,
                save_top_k=-1,
                save_last=False,
                every_n_train_steps=int(archive_every_n_train_steps),
                every_n_epochs=None,
                save_on_train_epoch_end=False,
                save_weights_only=checkpoint_cfg.get("save_weights_only", False),
                auto_insert_metric_name=False,
                enable_version_counter=False,
            )
        )

    if trainer_cfg.get("early_stopping", False):
        callbacks.append(
            EarlyStopping(
                monitor=checkpoint_cfg.get("monitor", "val/loss"),
                mode=checkpoint_cfg.get("mode", "min"),
                patience=int(trainer_cfg.get("early_stopping_patience", 10)),
                min_delta=0.0,
                check_on_train_epoch_end=False,
            )
        )
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass
    return callbacks


def build_trainer(
    trainer_cfg: Dict[str, Any],
    checkpoint_cfg: Dict[str, Any],
    output_dir: str,
):
    loggers = [TensorBoardLogger(save_dir=output_dir, name="tensorboard")]
    if trainer_cfg.get("use_wandb", False):
        try:
            loggers.append(
                WandbLogger(
                    project=trainer_cfg.get("wandb_project", "sala"),
                    entity=trainer_cfg.get("wandb_entity"),
                    name=trainer_cfg.get("wandb_name"),
                    save_dir=output_dir,
                )
            )
        except Exception as exc:
            logger.warning("Failed to initialize WandB: %s", exc)

    return Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 1000),
        max_steps=trainer_cfg.get("max_steps", 250000),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        num_nodes=trainer_cfg.get("num_nodes", 1),
        strategy=trainer_cfg.get("strategy", "auto"),
        precision=trainer_cfg.get("precision", 32),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 25.0),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        val_check_interval=trainer_cfg.get("val_check_interval", 1000),
        check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
        limit_train_batches=trainer_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
        limit_test_batches=trainer_cfg.get("limit_test_batches", 1.0),
        num_sanity_val_steps=trainer_cfg.get("num_sanity_val_steps", 2),
        use_distributed_sampler=trainer_cfg.get("use_distributed_sampler", True),
        callbacks=build_callbacks(checkpoint_cfg, trainer_cfg, output_dir),
        logger=loggers,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir=output_dir,
    )


def train(cfg: Dict[str, Any]) -> None:
    seed = cfg.get("seed", 42)
    seed_everything(seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    data_cfg = flatten_grouped_config(cfg.get("data"), DATA_CONFIG_SECTIONS)
    model_cfg = _to_plain_dict(cfg.get("model"), "model")
    flow_cfg = _resolve_flow_config(cfg.get("flow", {}))
    optim_cfg = _to_plain_dict(cfg.get("optim"), "optim")
    trainer_cfg = flatten_grouped_config(cfg.get("trainer"), TRAINER_CONFIG_SECTIONS)
    checkpoint_cfg = flatten_grouped_config(
        cfg.get("checkpoint"), CHECKPOINT_CONFIG_SECTIONS
    )

    devices = trainer_cfg.get("devices", 1)
    multi_device = isinstance(devices, int) and devices > 1
    if (
        multi_device
        and data_cfg.get("atom_bucket_edges")
        and trainer_cfg.get("use_distributed_sampler", True)
    ):
        raise ValueError(
            "Multi-device training with SALA atom bucket batches requires "
            "trainer.use_distributed_sampler=false; AsuDataModule performs "
            "rank-aware whole-batch sharding explicitly"
        )
    finetune_from = cfg.get("finetune_from") or None
    resume_from = cfg.get("resume_from") or None
    if finetune_from and resume_from:
        raise ValueError("Specify only one of finetune_from or resume_from")
    bootstrap_checkpoint = None
    bootstrap_path = finetune_from or resume_from
    if bootstrap_path:
        bootstrap_path = str(Path(bootstrap_path).expanduser().resolve())
        bootstrap_checkpoint = torch.load(
            bootstrap_path, map_location="cpu", weights_only=False
        )
        release_metadata = bootstrap_checkpoint.get("release_metadata") or {}
        if (
            resume_from
            and isinstance(release_metadata, dict)
            and release_metadata.get("weights") == "ema"
            and not bootstrap_checkpoint.get("optimizer_states")
        ):
            raise ValueError(
                "The release checkpoint contains EMA model weights without "
                "optimizer state. Use finetune_from to initialize a new training "
                "run; resume_from requires a full training checkpoint."
            )
        from sala.flow_matching.flow import MODEL_FAMILY
        from sala.flow_matching.lightning import (
            SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS,
        )

        if (
            bootstrap_checkpoint.get("model_family") != MODEL_FAMILY
            or bootstrap_checkpoint.get("checkpoint_schema_version")
            not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
        ):
            raise ValueError(
                f"Checkpoint is not a supported {MODEL_FAMILY} "
                "schema version 15 or 16 checkpoint"
            )

    output_dir = cfg.get("output_dir", os.getcwd())
    logger.info("=" * 60)
    logger.info("SALA Rectified Flow Matching Training")
    logger.info("=" * 60)
    logger.info("Output directory: %s", output_dir)
    logger.info("Checkpoints: %s", os.path.join(output_dir, "checkpoints"))
    logger.info("Data pt_root: %s", data_cfg.get("pt_root") or "null")
    logger.info("Batch size: %s", data_cfg.get("batch_size", 8))
    logger.info("Max epochs: %s", trainer_cfg.get("max_epochs", 1000))
    logger.info("=" * 60)

    logger.info("Resolving model configuration...")
    denoiser_config = build_denoiser_config(model_cfg)

    logger.info("Building data module...")
    datamodule = build_datamodule_from_config(data_cfg)
    datamodule.setup(stage="fit")
    logger.info("Dataset: %s samples", datamodule.total_samples)
    logger.info("  Train: %s", datamodule.num_train_samples)
    logger.info("  Val: %s", datamodule.num_val_samples)
    logger.info("  Test: %s", datamodule.num_test_samples)

    logger.info("Building model...")
    denoiser = build_model_from_config(denoiser_config)
    if trainer_cfg.get("compile_denoiser", False):
        compile_mode = str(trainer_cfg.get("compile_mode", "reduce-overhead"))
        try:
            denoiser = torch.compile(denoiser, mode=compile_mode)
            logger.info("Applied torch.compile(mode=%r)", compile_mode)
        except Exception as exc:
            logger.warning("torch.compile not applied: %s", exc)

    num_params = sum(param.numel() for param in denoiser.parameters())
    num_trainable = sum(
        param.numel() for param in denoiser.parameters() if param.requires_grad
    )
    expected_trainable = model_cfg.get("expected_trainable_parameters")
    if expected_trainable is not None and num_trainable != int(expected_trainable):
        raise RuntimeError(
            "Trainable parameter count does not match the locked recipe: "
            f"expected {int(expected_trainable):,}, found {num_trainable:,}"
        )
    logger.info(
        "Denoiser parameters: %s (%s trainable)",
        f"{num_params:,}",
        f"{num_trainable:,}",
    )

    logger.info("Building lightning module...")
    model = build_lightning_module_from_config(
        denoiser,
        flow_cfg,
        optim_cfg,
        denoiser_config=denoiser_config,
        gradient_clip_val=(
            None
            if trainer_cfg.get("gradient_clip_val") is None
            else float(trainer_cfg.get("gradient_clip_val"))
        ),
        nominal_batch_size=int(data_cfg["batch_size"]),
    )
    # Lightning restores the module state itself for resume.  Keep that path,
    # finetune, and ordinary ``load_state_dict`` calls agnostic to whether the
    # denoiser was wrapped by torch.compile in either process.
    install_torch_compile_state_dict_compatibility(model)

    ckpt_path = None
    if finetune_from:
        finetune_path = str(bootstrap_path)
        logger.info("Finetune: loading model weights only from %s", finetune_path)
        assert bootstrap_checkpoint is not None
        if "state_dict" not in bootstrap_checkpoint:
            raise ValueError("Finetune checkpoint has no state_dict")
        model.validate_finetune_checkpoint(bootstrap_checkpoint)
        state_dict, used_ema = select_ema_shadow_state_dict(bootstrap_checkpoint)
        if used_ema:
            logger.info("Finetune: using EMA shadow parameters as initialization")
        model.load_state_dict(state_dict, strict=True)
        model.validate_lattice_prior_state()
    elif resume_from:
        ckpt_path = str(bootstrap_path)
        logger.info("Resuming from checkpoint: %s", ckpt_path)

    model_cfg_plain = _to_plain_dict(model_cfg, "model")
    if bool(model_cfg_plain.get("freeze_backbone", False)) and not resume_from:
        freeze_fn = getattr(model.denoiser, "freeze_backbone", None)
        if callable(freeze_fn):
            freeze_fn()
        else:
            logger.warning("Denoiser has no freeze_backbone(); skipping")

    logger.info("Building trainer...")
    trainer = build_trainer(trainer_cfg, checkpoint_cfg, output_dir)
    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if trainer_cfg.get("run_test", True):
        logger.info("Running test evaluation...")
        trainer.test(model, datamodule=datamodule)

    logger.info("Training complete!")
    ckpt_cb = trainer.checkpoint_callback
    if isinstance(ckpt_cb, list):
        ckpt_cb = ckpt_cb[0] if ckpt_cb else None
    if ckpt_cb and getattr(ckpt_cb, "best_model_path", None):
        logger.info("Best model saved to: %s", ckpt_cb.best_model_path)


if HYDRA_AVAILABLE:

    @hydra.main(version_base=None, config_path="../../conf", config_name="config")
    def main_hydra(cfg: DictConfig) -> None:
        logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
        config = OmegaConf.to_container(cfg, resolve=True)
        config["output_dir"] = HydraConfig.get().runtime.output_dir
        train(config)




def main() -> None:
    if not HYDRA_AVAILABLE:
        raise ImportError("Hydra is required for the SALA training CLI")
    main_hydra()


if __name__ == "__main__":
    main()
