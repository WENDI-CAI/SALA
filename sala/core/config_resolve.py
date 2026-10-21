
from __future__ import annotations

from typing import Any, Mapping


DATA_CONFIG_SECTIONS = ("loader", "paths", "filters", "sampling", "augmentation")
TRAINER_CONFIG_SECTIONS = ("compute", "training", "stopping_and_post", "wandb", "compile")
CHECKPOINT_CONFIG_SECTIONS = ("monitoring", "retention")
GENERATION_CONFIG_SECTIONS = ("run", "io", "sampler", "outputs", "eval")


def flatten_grouped_config(
    cfg: Any,
    section_keys: tuple[str, ...],
) -> dict[str, Any]:
    if cfg is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    if not isinstance(cfg, Mapping):
        return {}
    raw = dict(cfg)
    section_set = frozenset(section_keys)
    merged: dict[str, Any] = {}
    for key in section_keys:
        sub = raw.get(key)
        if isinstance(sub, Mapping):
            merged.update(dict(sub))
    for k, v in raw.items():
        if k in section_set:
            continue
        merged[k] = v
    return merged
