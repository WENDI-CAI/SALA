from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn


_ORIG_MOD_SEGMENT = re.compile(r"(^|\.)_orig_mod\.")


def canonical_state_dict_key(key: str) -> str:
    """Return the stable key used by both compiled and eager modules.

    ``torch.compile`` wraps modules in an ``OptimizedModule`` and inserts an
    ``_orig_mod`` segment into every state-dict key owned by that module.  The
    wrapper is an execution detail, so checkpoint compatibility must not
    depend on whether the current process happens to compile the denoiser.
    """

    normalized = str(key)
    while _ORIG_MOD_SEGMENT.search(normalized):
        normalized = _ORIG_MOD_SEGMENT.sub(r"\1", normalized)
    return normalized


def normalize_compiled_state_dict_keys(
    state_dict: Mapping[str, Any],
    expected_keys: Iterable[str],
) -> tuple[dict[str, Any], bool]:
    """Map eager/compiled checkpoint keys onto a module's expected keys.

    Non-module metadata in a state dict is left untouched.  Ambiguous
    canonical keys are rejected instead of silently choosing one tensor.
    """

    expected_by_canonical: dict[str, str] = {}
    for expected_key in expected_keys:
        expected_key = str(expected_key)
        canonical = canonical_state_dict_key(expected_key)
        previous = expected_by_canonical.get(canonical)
        if previous is not None and previous != expected_key:
            raise ValueError(
                "Ambiguous model state keys after removing torch.compile "
                f"wrappers: {previous!r} and {expected_key!r}"
            )
        expected_by_canonical[canonical] = expected_key

    normalized: dict[str, Any] = {}
    source_by_target: dict[str, str] = {}
    changed = False
    for source_key, value in state_dict.items():
        source_key = str(source_key)
        target_key = expected_by_canonical.get(
            canonical_state_dict_key(source_key), source_key
        )
        if target_key in normalized:
            raise ValueError(
                "Checkpoint contains duplicate state keys after normalizing "
                "torch.compile wrappers: "
                f"{source_by_target[target_key]!r} and {source_key!r} "
                f"both map to {target_key!r}"
            )
        normalized[target_key] = value
        source_by_target[target_key] = source_key
        changed = changed or target_key != source_key
    return normalized, changed


def install_torch_compile_state_dict_compatibility(
    module: nn.Module,
) -> torch.utils.hooks.RemovableHandle:
    """Install a root load hook that accepts compiled or eager checkpoints."""

    def _normalize_before_load(
        loaded_module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        if prefix:
            return
        normalized, changed = normalize_compiled_state_dict_keys(
            state_dict, loaded_module.state_dict().keys()
        )
        if changed:
            state_dict.clear()
            state_dict.update(normalized)

    return module.register_load_state_dict_pre_hook(_normalize_before_load)


__all__ = [
    "canonical_state_dict_key",
    "install_torch_compile_state_dict_compatibility",
    "normalize_compiled_state_dict_keys",
]
