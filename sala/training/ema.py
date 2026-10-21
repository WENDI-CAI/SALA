from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback

from sala.training.checkpoint_compat import normalize_compiled_state_dict_keys


def _std_to_exp(std: float) -> float:
    value = float(std)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("power-function EMA std must be positive and finite")
    # Largest real root of the EDM2 cubic from Equation 126.
    roots = torch.as_tensor(
        [1.0, 7.0, 16.0 - value**-2, 12.0 - value**-2],
        dtype=torch.float64,
    )
    # torch has no polynomial roots; companion eigenvalues keep this dependency-free.
    companion = torch.zeros((3, 3), dtype=torch.float64)
    companion[0] = -roots[1:] / roots[0]
    companion[1, 0] = 1.0
    companion[2, 1] = 1.0
    eigenvalues = torch.linalg.eigvals(companion)
    real = eigenvalues.real[torch.abs(eigenvalues.imag) < 1.0e-8]
    if real.numel() == 0:
        raise ValueError("could not resolve power-function EMA exponent")
    return float(real.max().item())


def power_function_beta(std: float, *, t_next: int, t_delta: int) -> float:
    if t_next <= 0 or t_delta <= 0 or t_delta > t_next:
        raise ValueError("invalid power-function EMA sample counters")
    exponent = _std_to_exp(float(std))
    return float((1.0 - float(t_delta) / float(t_next)) ** (exponent + 1.0))


def _profile_key(std: float) -> str:
    return f"{float(std):.6f}"


def select_ema_shadow_state_dict(
    checkpoint: Dict[str, Any],
    *,
    use_ema: bool = True,
    profile_std: Optional[float] = None,
) -> tuple[Dict[str, torch.Tensor], bool]:
    state_dict = dict(checkpoint.get("state_dict") or {})
    release_metadata = checkpoint.get("release_metadata") or {}
    if isinstance(release_metadata, dict) and release_metadata.get("weights") == "ema":
        stored_std = release_metadata.get("ema_profile_std")
        if stored_std is None:
            raise ValueError("Released EMA checkpoint is missing ema_profile_std")
        stored_std = float(stored_std)
        if not math.isfinite(stored_std) or stored_std <= 0.0:
            raise ValueError("Released EMA profile std must be positive and finite")
        if not use_ema:
            raise ValueError("This checkpoint contains EMA weights only")
        if profile_std is not None and not math.isclose(
            float(profile_std), stored_std, rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise ValueError(
                f"This checkpoint contains EMA profile {stored_std:g} only; "
                f"requested {float(profile_std):g}"
            )
        if not state_dict:
            raise ValueError("Released EMA checkpoint has no model state_dict")
        return state_dict, True
    if not use_ema:
        return state_dict, False

    state = checkpoint.get(EMACallback._CHECKPOINT_KEY)
    if not isinstance(state, dict):
        return state_dict, False
    profiles = state.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        return state_dict, False
    stds = [float(value) for value in state.get("profile_stds", [])]
    if not stds:
        stds = sorted(float(key) for key in profiles)
    selected = stds[0] if profile_std is None else float(profile_std)
    key = min(profiles, key=lambda candidate: abs(float(candidate) - selected))
    shadow = profiles.get(key)
    if not isinstance(shadow, dict):
        return state_dict, False

    applied = False
    for name, value in shadow.items():
        if name in state_dict and torch.is_tensor(value):
            state_dict[name] = value
            applied = True
    return state_dict, applied


class EMACallback(Callback):
    """Track multiple sample-count-based power-function EMA profiles."""

    _CHECKPOINT_KEY = "ema_callback"

    def __init__(
        self,
        profile_stds: Sequence[float] = (0.05, 0.10),
        *,
        validation_profile_std: Optional[float] = None,
    ) -> None:
        super().__init__()
        values = tuple(float(value) for value in profile_stds)
        if not values:
            raise ValueError("at least one EMA profile std is required")
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("EMA profile stds must be positive and finite")
        if len(set(values)) != len(values):
            raise ValueError("EMA profile stds must be unique")
        self.profile_stds = values
        self._profile_exponents = {
            _profile_key(std): _std_to_exp(std) for std in values
        }
        self.validation_profile_std = (
            values[0]
            if validation_profile_std is None
            else float(validation_profile_std)
        )
        self._ema_profiles: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        self._restore_params: Optional[Dict[str, torch.Tensor]] = None
        self._last_update_step: Optional[int] = None
        self._samples_seen = 0

    def _ensure_ema(self, pl_module: LightningModule) -> None:
        if self._ema_profiles is not None:
            return
        base = {
            name: parameter.detach().clone()
            for name, parameter in pl_module.named_parameters()
            if parameter.requires_grad
        }
        self._ema_profiles = {
            _profile_key(std): {
                name: value.detach().clone() for name, value in base.items()
            }
            for std in self.profile_stds
        }

    def on_train_start(self, trainer: Any, pl_module: LightningModule) -> None:
        self._ensure_ema(pl_module)
        if self._last_update_step is None:
            self._last_update_step = int(trainer.global_step)

    def _update_ema(
        self,
        pl_module: LightningModule,
        *,
        global_batch_size: int,
    ) -> None:
        self._ensure_ema(pl_module)
        assert self._ema_profiles is not None
        delta = max(1, int(global_batch_size))
        next_seen = self._samples_seen + delta
        with torch.no_grad():
            for std in self.profile_stds:
                key = _profile_key(std)
                exponent = self._profile_exponents[key]
                beta = float(
                    (1.0 - float(delta) / float(next_seen))
                    ** (exponent + 1.0)
                )
                profile = self._ema_profiles[key]
                for name, parameter in pl_module.named_parameters():
                    if not parameter.requires_grad or name not in profile:
                        continue
                    ema = profile[name].to(
                        device=parameter.device, dtype=parameter.dtype
                    )
                    profile[name] = ema
                    ema.lerp_(parameter.detach(), 1.0 - beta)
        self._samples_seen = next_seen

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch_idx
        current_step = int(getattr(trainer, "global_step", 0))
        if self._last_update_step is None:
            self._ensure_ema(pl_module)
            self._last_update_step = current_step
            return
        if current_step <= self._last_update_step:
            return
        local_batch = int(getattr(batch, "batch_size", 1))
        world_size = int(getattr(trainer, "world_size", 1))
        accumulation = int(getattr(trainer, "accumulate_grad_batches", 1))
        self._update_ema(
            pl_module,
            global_batch_size=max(
                1, local_batch * world_size * accumulation
            ),
        )
        self._last_update_step = current_step

    def on_save_checkpoint(
        self,
        trainer: Any,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        del trainer, pl_module
        if self._ema_profiles is None:
            return
        checkpoint[self._CHECKPOINT_KEY] = {
            "type": "power_function",
            "profile_stds": list(self.profile_stds),
            "validation_profile_std": float(self.validation_profile_std),
            "samples_seen": int(self._samples_seen),
            "last_update_step": self._last_update_step,
            "profiles": {
                key: {
                    name: value.detach().clone().cpu()
                    for name, value in profile.items()
                }
                for key, profile in self._ema_profiles.items()
            },
        }

    def on_load_checkpoint(
        self,
        trainer: Any,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        del trainer
        state = checkpoint.get(self._CHECKPOINT_KEY)
        if not isinstance(state, dict):
            return
        profiles = state.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            return
        expected_names = tuple(name for name, _ in pl_module.named_parameters())
        normalized_profiles: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, shadow in profiles.items():
            if not isinstance(shadow, dict):
                continue
            normalized, _ = normalize_compiled_state_dict_keys(
                shadow, expected_names
            )
            normalized_profiles[str(key)] = {
                str(name): value.detach().clone()
                for name, value in normalized.items()
                if torch.is_tensor(value)
            }
        if not normalized_profiles:
            return
        self.profile_stds = tuple(
            float(value) for value in state.get("profile_stds", normalized_profiles)
        )
        self._profile_exponents = {
            _profile_key(std): _std_to_exp(std) for std in self.profile_stds
        }
        self.validation_profile_std = float(
            state.get("validation_profile_std", self.profile_stds[0])
        )
        self._ema_profiles = normalized_profiles
        self._samples_seen = int(state.get("samples_seen", 0))
        last_update_step = state.get("last_update_step")
        self._last_update_step = (
            None if last_update_step is None else int(last_update_step)
        )

    def _selected_profile(self) -> Optional[Dict[str, torch.Tensor]]:
        if self._ema_profiles is None:
            return None
        key = min(
            self._ema_profiles,
            key=lambda candidate: abs(
                float(candidate) - float(self.validation_profile_std)
            ),
        )
        return self._ema_profiles[key]

    def _swap_in(self, pl_module: LightningModule) -> None:
        profile = self._selected_profile()
        if profile is None:
            return
        self._restore_params = {}
        for name, parameter in pl_module.named_parameters():
            if name in profile:
                self._restore_params[name] = parameter.data.clone()
                parameter.data.copy_(
                    profile[name].to(
                        device=parameter.device, dtype=parameter.dtype
                    )
                )

    def _restore(self, pl_module: LightningModule) -> None:
        if self._restore_params is None:
            return
        for name, parameter in pl_module.named_parameters():
            if name in self._restore_params:
                parameter.data.copy_(self._restore_params[name])
        self._restore_params = None

    def on_validation_start(self, trainer: Any, pl_module: LightningModule) -> None:
        del trainer
        self._swap_in(pl_module)

    def on_validation_end(self, trainer: Any, pl_module: LightningModule) -> None:
        del trainer
        self._restore(pl_module)

    def on_test_start(self, trainer: Any, pl_module: LightningModule) -> None:
        del trainer
        self._swap_in(pl_module)

    def on_test_end(self, trainer: Any, pl_module: LightningModule) -> None:
        del trainer
        self._restore(pl_module)


__all__ = [
    "EMACallback",
    "power_function_beta",
    "select_ema_shadow_state_dict",
]
