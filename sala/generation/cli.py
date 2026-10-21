from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

try:
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf

    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from sala.core.config_resolve import (
    DATA_CONFIG_SECTIONS,
    GENERATION_CONFIG_SECTIONS,
    flatten_grouped_config,
)
from sala.data.hydrogenation_evidence import (
    HYDROGENATION_EVIDENCE_VERSION,
    build_hydrogenation_evidence,
)
from sala.data.schema import FULL_CELL_DATA_SCHEMA_VERSION
from sala.postprocess.symmetry_rigid_corrector import (
    normalize_symmetry_corrector_config,
    symmetry_rigid_postprocess_batch,
)
from sala.training.checkpoint_compat import (
    install_torch_compile_state_dict_compatibility,
    normalize_compiled_state_dict_keys,
)

ATOMIC_NUMBER_TO_ELEMENT = None
DEFAULT_SHARD_SUFFIX = "_fullcell_q"
DEFAULT_HYDROGENATION_CONFIG = {
    "enabled": False,
    "failure_policy": "strict",
    "output_subdir": "hydrogenated",
}


def _temporary_output_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    return Path(temporary)


def _atomic_write_text(output_path: Path, content: str) -> None:
    temporary = _temporary_output_path(output_path)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(payload: Any, output_path: Path) -> None:
    temporary = _temporary_output_path(output_path)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class EvalTargetFilter:

    c_count_min: Optional[int] = None
    c_count_max: Optional[int] = None
    classes: Optional[list[str]] = None
    include_unknown: bool = True
    spacegroups: Optional[list[int]] = None
    is_neutral: Optional[bool] = None
    has_metal: Optional[bool] = None
    n_heavy_atoms_min: Optional[int] = None
    n_heavy_atoms_max: Optional[int] = None
    n_components_min: Optional[int] = None
    n_components_max: Optional[int] = None
    split: Optional[str] = None
    max_targets: Optional[int] = None
    target_selection: str = "first"
    target_selection_seed: Optional[int] = None
    shard_cache_size: int = 1
    shard_suffix: str = DEFAULT_SHARD_SUFFIX
    filter_h: bool = True


@dataclass
class GenerationTarget:

    target_id: str
    asu: Any
    source_path: Path
    source_type: str
    reference_cif_path: Optional[Path] = None
    meta: Optional[dict[str, Any]] = None


@dataclass
class GenerationConfig:

    checkpoint_path: str
    mode: str
    output_dir: Path
    device: str
    num_samples: int
    batch_size: int
    filter_h: bool
    sample_batch_size: int
    save_cif: bool
    save_pt: bool
    save_reference: bool
    seed: int = 42
    ema_profile_std: Optional[float] = None
    save_trajectory_cif: bool = False
    trajectory_interval: int = 100
    num_steps: Optional[int] = None
    input: Optional[Path] = None
    input_dir: Optional[Path] = None
    input_index: Optional[Path] = None
    pt_root: Optional[Path] = None
    input_format: str = "auto"
    eval_filter: EvalTargetFilter = None
    eval_num_targets_before_subset: Optional[int] = None
    eval_num_targets_after_subset: Optional[int] = None
    postprocess_cfg: Optional[dict[str, Any]] = None
    hydrogenation_cfg: Optional[dict[str, Any]] = None


def _ensure_atomic_number_mapping() -> Dict[int, str]:
    """Lazily initialize the atomic-number to element-symbol lookup table."""
    global ATOMIC_NUMBER_TO_ELEMENT
    if ATOMIC_NUMBER_TO_ELEMENT is None:
        from sala.core.types import ELEMENT_TO_ATOMIC_NUMBER

        ATOMIC_NUMBER_TO_ELEMENT = {v: k for k, v in ELEMENT_TO_ATOMIC_NUMBER.items()}
    return ATOMIC_NUMBER_TO_ELEMENT


def _get_element_symbol(atomic_number: int) -> str:
    symbol = _ensure_atomic_number_mapping().get(int(atomic_number))
    if symbol is None:
        raise ValueError(f"Unknown atomic number: {atomic_number}")
    return symbol


def _formula_from_atom_types(atom_types: List[int]) -> str:
    table = _ensure_atomic_number_mapping()
    return _formula_from_symbols([table.get(int(z), "X") for z in atom_types])


def _formula_from_symbols(atom_symbols: Sequence[str]) -> str:
    counts: Dict[str, int] = {}
    for symbol in atom_symbols:
        sym = str(symbol)
        counts[sym] = counts.get(sym, 0) + 1

    preferred = ("C", "H", "D", "T")
    order = [symbol for symbol in preferred if symbol in counts]
    order.extend(sorted(symbol for symbol in counts if symbol not in preferred))
    parts = []
    for sym in order:
        if sym not in counts:
            continue
        n = counts[sym]
        parts.append(f"{sym}{n}" if n > 1 else sym)
    return " ".join(parts) if parts else ""


def _replace_cif_value(content: str, key: str, value: float, digits: int = 6) -> str:
    pattern = rf"({re.escape(key)}\s+)([-\d.]+)"
    return re.sub(
        pattern,
        lambda match: f"{match.group(1)}{value:.{digits}f}",
        content,
    )


def _cif_float(val: float) -> float:
    import math

    if isinstance(val, (int, float)) and math.isfinite(val):
        return float(val)
    raise ValueError(f"CIF cell parameter must be finite, got {val!r}")


def _atom_site_block_minimal(
    atom_labels: List[str],
    atom_types: List[int],
    frac_coords: Tensor,
    *,
    atom_symbols: Optional[Sequence[str]] = None,
    formal_charges: Optional[Sequence[Optional[int]]] = None,
) -> str:
    if atom_symbols is None:
        symbols = [_get_element_symbol(int(value)) for value in atom_types]
    else:
        symbols = [str(value) for value in atom_symbols]
        if len(symbols) != len(atom_types):
            raise ValueError("atom_symbols must contain one value per atom")
    if formal_charges is not None and len(formal_charges) != len(atom_types):
        raise ValueError("formal_charges must contain one value per atom")
    headers = [
        "loop_",
        " _atom_site_type_symbol",
        " _atom_site_label",
        " _atom_site_fract_x",
        " _atom_site_fract_y",
        " _atom_site_fract_z",
    ]
    if formal_charges is not None:
        headers.append(" _atom_site_formal_charge")
    coords = frac_coords.detach().cpu().numpy()
    lines = []
    for idx, (label, symbol) in enumerate(zip(atom_labels, symbols)):
        x, y, z = coords[idx]
        line = f" {symbol:<3s} {label:<6s} {x:10.5f} {y:10.5f} {z:10.5f}"
        if formal_charges is not None:
            charge = formal_charges[idx]
            line += f" {'.' if charge is None else int(charge):>3}"
        lines.append(line)
    return "\n".join(headers + lines) + "\n"


def _geom_bond_block(
    atom_labels: List[str], bonds: List[tuple], frac_coords: Tensor, lattice: Tensor
) -> str:
    from sala.core.lattice import frac_to_cart_coords

    coords_cart = frac_to_cart_coords(frac_coords, lattice).detach().cpu()
    label_to_idx = {label: i for i, label in enumerate(atom_labels)}
    seen: set[tuple[str, str]] = set()
    lines: List[str] = []
    for b in bonds:
        atom1, atom2 = b[0], b[1]
        canonical = (min(atom1, atom2), max(atom1, atom2))
        if canonical in seen:
            continue
        seen.add(canonical)
        idx1 = label_to_idx.get(atom1)
        idx2 = label_to_idx.get(atom2)
        if idx1 is None or idx2 is None:
            continue
        dist = torch.linalg.norm(coords_cart[idx1] - coords_cart[idx2]).item()
        raw_order = None
        if len(b) >= 4:
            raw_order = b[3]
        elif len(b) >= 3:
            # Generated graph bonds use ``(label_1, label_2, order)`` while
            # parsed CIF bonds include the source distance before the order.
            raw_order = b[2]
        order_text = "?" if raw_order is None else f"{float(raw_order):.2f}"
        lines.append(f" {atom1:<6s} {atom2:<6s} {dist:7.3f} {order_text:>5s}")
    if not lines:
        return ""
    header = (
        "loop_\n"
        " _geom_bond_atom_site_label_1\n"
        " _geom_bond_atom_site_label_2\n"
        " _geom_bond_distance\n"
        " _geom_bond_order\n"
    )
    return header + "\n".join(lines) + "\n"


def _format_symmetry_axis(rotation: np.ndarray, translation: float) -> str:
    variables = ("x", "y", "z")
    terms: list[tuple[int, str]] = []
    for coefficient, variable in zip(rotation, variables):
        rounded = int(round(float(coefficient)))
        if abs(float(coefficient) - rounded) > 1.0e-6:
            raise ValueError("CIF symmetry rotations must be integral")
        if rounded:
            magnitude = "" if abs(rounded) == 1 else str(abs(rounded))
            terms.append((1 if rounded > 0 else -1, f"{magnitude}{variable}"))
    wrapped = float(translation) % 1.0
    if abs(wrapped - 1.0) < 1.0e-8 or abs(wrapped) < 1.0e-8:
        wrapped = 0.0
    if wrapped:
        fraction = Fraction(wrapped).limit_denominator(48)
        if abs(float(fraction) - wrapped) > 1.0e-6:
            raise ValueError("CIF symmetry translation is not a small rational")
        text = (
            str(fraction.numerator)
            if fraction.denominator == 1
            else f"{fraction.numerator}/{fraction.denominator}"
        )
        terms.append((1, text))
    if not terms:
        return "0"
    pieces: list[str] = []
    for index, (sign, text) in enumerate(terms):
        if index == 0:
            pieces.append(text if sign > 0 else f"-{text}")
        else:
            pieces.append(("+" if sign > 0 else "-") + text)
    return "".join(pieces)


def _resolve_cif_symmetry(
    space_group_number: int,
    hall_number: int,
    symmetry_ops: Tensor,
) -> tuple[int, str, str, np.ndarray]:
    from sala.core.lattice_manifold import lattice_template_from_hall_number
    from sala.core.symmetry import (
        spacegroup_number_from_symmetry_ops,
        symmetry_ops_match_hall,
    )

    sg_num = int(space_group_number)
    hall = int(hall_number)
    hall_spacegroup, _template = lattice_template_from_hall_number(hall)
    if hall_spacegroup != sg_num:
        raise ValueError(
            f"Hall number {hall} is incompatible with IT spacegroup {sg_num}"
        )
    matrices = np.asarray(
        torch.as_tensor(symmetry_ops).detach().cpu(), dtype=np.float64
    )
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not len(matrices):
        raise ValueError("symmetry_ops must have shape [G, 4, 4]")
    operations_spacegroup = spacegroup_number_from_symmetry_ops(matrices)
    if operations_spacegroup != sg_num:
        raise ValueError(
            f"Output symmetry operations resolve to IT spacegroup {operations_spacegroup}, "
            f"not requested spacegroup {sg_num}"
        )

    if not symmetry_ops_match_hall(matrices, hall):
        raise ValueError(
            f"Hall number {hall} disagrees with output symmetry operation "
            "basis/unique-axis setting"
        )
    import spglib

    sg_type = spglib.get_spacegroup_type(hall)
    hall_symbol = str(
        sg_type.hall_symbol if hasattr(sg_type, "hall_symbol") else sg_type["hall_symbol"]
    )
    hm_symbol = str(
        sg_type.international_full
        if hasattr(sg_type, "international_full")
        else sg_type["international_full"]
    )
    return hall, hall_symbol, hm_symbol, matrices


def _symmetry_operation_block(matrices: np.ndarray) -> str:
    lines = ["loop_", " _space_group_symop_id", " _space_group_symop_operation_xyz"]
    for index, matrix in enumerate(matrices, start=1):
        expression = ",".join(
            _format_symmetry_axis(matrix[axis, :3], matrix[axis, 3])
            for axis in range(3)
        )
        lines.append(f" {index} '{expression}'")
    return "\n".join(lines)


def _build_minimal_cif(
    atom_labels: List[str],
    atom_types: List[int],
    frac_coords: Tensor,
    lattice: Tensor,
    space_group_number: int,
    hall_number: int,
    symmetry_ops: Tensor,
    bonds: Optional[List[tuple]] = None,
    data_block_name: str = "generated",
    atom_symbols: Optional[Sequence[str]] = None,
    formula_override: Optional[str] = None,
    formal_charges: Optional[Sequence[Optional[int]]] = None,
) -> str:
    from sala.core.lattice import lattice_matrix_to_params

    lat = lattice.detach().cpu().float()
    if lat.dim() == 2:
        lat = lat.unsqueeze(0)
    lengths, angles = lattice_matrix_to_params(lat)
    if hasattr(lengths, "numpy"):
        lengths = lengths.numpy()
        angles = angles.numpy()
    a = _cif_float(float(lengths[0]))
    b = _cif_float(float(lengths[1]))
    c = _cif_float(float(lengths[2]))
    alpha = _cif_float(float(angles[0]))
    beta = _cif_float(float(angles[1]))
    gamma = _cif_float(float(angles[2]))
    sg_num = int(space_group_number)
    if not (1 <= sg_num <= 230):
        raise ValueError("space_group_number must be in [1, 230]")
    resolved_hall, hall_symbol, hm_symbol, resolved_ops = _resolve_cif_symmetry(
        sg_num, hall_number, symmetry_ops
    )
    formula = (
        str(formula_override)
        if formula_override is not None
        else (
            _formula_from_symbols(atom_symbols)
            if atom_symbols is not None
            else _formula_from_atom_types(atom_types)
        )
    )
    parts = [
        f"data_{data_block_name}",
        f"_symmetry_space_group_name_H-M   '{hm_symbol}'",
        f"_space_group_name_Hall   '{hall_symbol}'",
        f"_space_group_hall_number   {resolved_hall}",
        f"_cell_length_a   {a:.5f}",
        f"_cell_length_b   {b:.5f}",
        f"_cell_length_c   {c:.5f}",
        f"_cell_angle_alpha   {alpha:.5f}",
        f"_cell_angle_beta   {beta:.5f}",
        f"_cell_angle_gamma   {gamma:.5f}",
        f"_symmetry_Int_Tables_number   {sg_num}",
        "",
        _symmetry_operation_block(resolved_ops),
        "",
        _atom_site_block_minimal(
            atom_labels,
            atom_types,
            frac_coords,
            atom_symbols=atom_symbols,
            formal_charges=formal_charges,
        ),
    ]
    if formula:
        formula_line = f"_chemical_formula_moiety '{formula}'"
        parts.insert(len(parts) - 1, formula_line)
    if bonds:
        bond_block = _geom_bond_block(atom_labels, bonds, frac_coords, lattice)
        if bond_block:
            parts.append(bond_block)
    return "\n".join(parts) + "\n"


def _find_cif_loop_span(
    content: str,
    required_headers: set[str],
) -> tuple[list[str], int, int] | None:
    """Locate a loop by headers and return its exact line span.

    CIF loop rows are token streams, so blank lines and comments do not end a
    loop.  Reserved unquoted tokens do.  Keeping this one locator shared by
    atom and bond rewriting prevents an earlier symmetry/aniso loop from being
    mistaken for the requested table.
    """

    lines = content.splitlines(keepends=True)
    for start, line in enumerate(lines):
        if line.strip().lower() != "loop_":
            continue

        header_names: list[str] = []
        cursor = start + 1
        while cursor < len(lines):
            stripped = lines[cursor].strip()
            if not stripped or stripped.startswith("#"):
                cursor += 1
                continue
            if stripped.startswith("_"):
                header_names.append(stripped.split(None, 1)[0].lower())
                cursor += 1
                continue
            break

        if not required_headers.issubset(set(header_names)):
            continue

        end = cursor
        while end < len(lines):
            stripped = lines[end].strip()
            lowered = stripped.lower()
            if stripped and not stripped.startswith("#") and (
                stripped.startswith("_")
                or lowered in {"loop_", "stop_", "global_"}
                or lowered.startswith("data_")
                or lowered.startswith("save_")
            ):
                break
            end += 1
        return lines, start, end
    return None


def _replace_atom_site_loop_with_minimal(
    content: str,
    atom_labels: List[str],
    atom_types: List[int],
    frac_coords: Tensor,
) -> str:
    minimal_block = _atom_site_block_minimal(atom_labels, atom_types, frac_coords)
    located = _find_cif_loop_span(
        content,
        {
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        },
    )
    if located is None:
        raise ValueError("Could not find atom_site loop in CIF template")
    lines, start, end = located
    return "".join(lines[:start]) + minimal_block + "".join(lines[end:])


def _replace_bond_loop_with_minimal(
    content: str,
    atom_labels: List[str],
    bonds: List[tuple],
    frac_coords: Tensor,
    lattice: Tensor,
) -> str:
    bond_block = _geom_bond_block(atom_labels, bonds, frac_coords, lattice)
    if not bond_block:
        return content
    located = _find_cif_loop_span(
        content,
        {
            "_geom_bond_atom_site_label_1",
            "_geom_bond_atom_site_label_2",
            "_geom_bond_distance",
        },
    )
    if located is None:
        return content
    lines, start, end = located
    return "".join(lines[:start]) + bond_block.rstrip("\n") + "\n" + "".join(
        lines[end:]
    )


def _build_cif_from_template(
    template: str,
    atom_labels: List[str],
    atom_types: List[int],
    bonds: List[tuple[str, str, float, float]],
    frac_coords: Tensor,
    lattice: Tensor,
) -> str:
    from sala.core.lattice import lattice_matrix_to_params

    lattice_cpu = lattice.detach().cpu()
    lengths, angles = lattice_matrix_to_params(lattice_cpu)
    a = _cif_float(float(lengths[0]))
    b = _cif_float(float(lengths[1]))
    c = _cif_float(float(lengths[2]))
    alpha = _cif_float(float(angles[0]))
    beta = _cif_float(float(angles[1]))
    gamma = _cif_float(float(angles[2]))
    content = template
    content = _replace_cif_value(content, "_cell_length_a", a)
    content = _replace_cif_value(content, "_cell_length_b", b)
    content = _replace_cif_value(content, "_cell_length_c", c)
    content = _replace_cif_value(content, "_cell_angle_alpha", alpha)
    content = _replace_cif_value(content, "_cell_angle_beta", beta)
    content = _replace_cif_value(content, "_cell_angle_gamma", gamma)

    content = _replace_atom_site_loop_with_minimal(
        content, atom_labels, atom_types, frac_coords
    )

    if bonds:
        content = _replace_bond_loop_with_minimal(
            content, atom_labels, bonds, frac_coords, lattice_cpu
        )

    return content


def select_generation_state_dict(
    checkpoint: dict,
    *,
    use_ema: bool = True,
    profile_std: Optional[float] = None,
) -> tuple[dict, bool]:
    from sala.training.ema import select_ema_shadow_state_dict

    return select_ema_shadow_state_dict(
        checkpoint, use_ema=use_ema, profile_std=profile_std
    )


def load_checkpoint(
    ckpt_path: str,
    device: str = "cpu",
    *,
    ema_profile_std: Optional[float] = None,
):
    from sala.flow_matching.flow import MODEL_FAMILY
    from sala.flow_matching.lightning import (
        SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS,
        FlowMatchingLightningModule,
        validate_full_cell_checkpoint_contract,
    )
    from sala.training.cli import build_model_from_config, _resolve_flow_config

    path = Path(ckpt_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint path is not a file: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if checkpoint.get("model_family") != MODEL_FAMILY:
        raise ValueError(
            f"Unsupported checkpoint: generation requires model_family={MODEL_FAMILY!r}"
        )
    checkpoint_schema_version = checkpoint.get("checkpoint_schema_version")
    if checkpoint_schema_version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
        raise ValueError(
            "Unsupported checkpoint schema: generation requires schema version "
            "15 or 16; earlier schemas use an incompatible Transformer, optimizer, "
            "EMA, or flow objective"
        )
    flow_config = checkpoint.get("flow_config")
    denoiser_config = checkpoint.get("denoiser_config")
    if not isinstance(flow_config, dict) or not isinstance(denoiser_config, dict):
        raise ValueError(
            "Flow Matching checkpoint must contain flow_config and denoiser_config mappings"
        )
    flow_config = _resolve_flow_config(flow_config)
    denoiser_config = dict(denoiser_config)
    if "graph" in denoiser_config:
        raise ValueError(
            "denoiser_config.graph is unsupported; use denoiser_config.pair"
        )
    validate_full_cell_checkpoint_contract(
        flow_config,
        denoiser_config,
        checkpoint_schema_version=int(checkpoint_schema_version),
    )
    state_dict, used_ema_weights = select_generation_state_dict(
        checkpoint, profile_std=ema_profile_std
    )
    denoiser = build_model_from_config(denoiser_config)
    hparams = dict(checkpoint.get("hyper_parameters", {}) or {})
    model = FlowMatchingLightningModule(
        denoiser=denoiser,
        flow_config=flow_config,
        denoiser_config=denoiser_config,
        lr=float(hparams.get("lr", 1e-4)),
        weight_decay=float(hparams.get("weight_decay", 0.01)),
        optimizer_name=str(hparams.get("optimizer_name", "adamw")),
        muon_config=dict(hparams.get("muon_config") or {}),
        lr_scheduler=dict(hparams.get("lr_scheduler") or {}),
        warmup_steps=int(hparams.get("warmup_steps", 10000)),
        max_grad_norm=float(hparams.get("max_grad_norm", 1.0)),
        nominal_batch_size=hparams.get("nominal_batch_size"),
    )
    install_torch_compile_state_dict_compatibility(model)
    model.on_load_checkpoint(checkpoint)
    state_dict, normalized_compile_keys = normalize_compiled_state_dict_keys(
        state_dict, model.state_dict().keys()
    )
    if normalized_compile_keys:
        logger.info("Normalized torch.compile wrapper keys in generation checkpoint")
    model.load_state_dict(state_dict, strict=True)
    if used_ema_weights:
        logger.info("Using EMA parameters for Flow Matching generation")
    model.to(device)
    model.eval()
    return model, {
        "model_family": MODEL_FAMILY,
        "checkpoint_schema_version": int(checkpoint_schema_version),
        "flow_config": flow_config,
        "denoiser_config": denoiser_config,
    }


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def normalize_hydrogenation_config(
    config: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    raw = dict(config or {})
    unknown = sorted(set(raw) - set(DEFAULT_HYDROGENATION_CONFIG))
    if unknown:
        raise ValueError(f"Unknown hydrogenation config keys: {unknown}")
    merged = {**DEFAULT_HYDROGENATION_CONFIG, **raw}
    if not isinstance(merged["enabled"], bool):
        raise TypeError("hydrogenation.enabled must be a boolean")
    failure_policy = str(merged["failure_policy"])
    if failure_policy not in {"strict", "keep_heavy"}:
        raise ValueError(
            "hydrogenation.failure_policy must be 'strict' or 'keep_heavy'"
        )
    output_subdir = str(merged["output_subdir"]).strip()
    relative = Path(output_subdir)
    if (
        not output_subdir
        or relative.is_absolute()
        or relative == Path(".")
        or ".." in relative.parts
    ):
        raise ValueError(
            "hydrogenation.output_subdir must be a non-empty relative path without '..'"
        )
    if relative.parts[0] in {"generated", "reference"}:
        raise ValueError(
            "hydrogenation.output_subdir must not overlap generated or reference"
        )
    return {
        "enabled": merged["enabled"],
        "failure_policy": failure_policy,
        "output_subdir": relative.as_posix(),
    }


def _parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None or isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _split_csv_strings(
    value: Optional[Union[str, Sequence[str]]]
) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None
    items = [str(item).strip() for item in value if str(item).strip()]
    return items or None


def _split_csv_ints(value: Optional[Union[str, Sequence[int]]]) -> Optional[list[int]]:
    items = _split_csv_strings(value)
    if not items:
        return None
    return [int(item) for item in items]


def _make_target_id(name: str, fallback_index: Optional[int] = None) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    if cleaned:
        return cleaned
    if fallback_index is None:
        return "target"
    return f"target_{fallback_index:04d}"


def _validate_unique_target_ids(targets: Sequence[GenerationTarget]) -> None:
    sources_by_id: dict[str, list[str]] = {}
    for target in targets:
        sources_by_id.setdefault(str(target.target_id), []).append(
            str(target.source_path)
        )
    collisions = {
        target_id: sources
        for target_id, sources in sources_by_id.items()
        if len(sources) > 1
    }
    if not collisions:
        return
    details = "; ".join(
        f"{target_id!r}: {', '.join(sources)}"
        for target_id, sources in sorted(collisions.items())
    )
    raise ValueError(
        "Duplicate target_id values would overwrite generation artifacts. "
        f"Rename or de-duplicate the inputs: {details}"
    )


def _seed_generation_rngs(seed: int) -> int:
    sampling_seed = int(seed)
    random.seed(sampling_seed)
    np.random.seed(sampling_seed % (2**32))
    torch.manual_seed(sampling_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sampling_seed)
    return sampling_seed


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _candidate_index_names(shard_suffix: str = DEFAULT_SHARD_SUFFIX) -> list[str]:
    suffix = shard_suffix or ""
    names: list[str] = []
    if suffix:
        names.extend([f"index{suffix}.pt", f"index{suffix}.jsonl"])
    names.extend(["index.pt", "index.jsonl"])

    return list(dict.fromkeys(names))


def _is_index_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix in {".pt", ".jsonl"}
        and (path.stem == "index" or path.stem.startswith("index_"))
    )


def _resolve_pt_dataset_paths(
    input_path: Optional[Union[str, Path]],
    index_path: Optional[Union[str, Path]],
    shard_suffix: str = DEFAULT_SHARD_SUFFIX,
) -> tuple[Path, Path]:
    candidate_input = Path(input_path) if input_path is not None else None
    candidate_index = Path(index_path) if index_path is not None else None

    if candidate_index is not None:
        if candidate_index.is_dir():
            for name in _candidate_index_names(shard_suffix):
                possible = candidate_index / name
                if possible.exists():
                    return candidate_index, possible
            raise FileNotFoundError(
                f"Index not found under directory: {candidate_index}"
            )
        root = (
            candidate_input if candidate_input is not None else candidate_index.parent
        )
        return root, candidate_index

    if candidate_input is None:
        raise ValueError("PT dataset path is required")

    if candidate_input.is_file() and candidate_input.suffix in {".pt", ".jsonl"}:
        return candidate_input.parent, candidate_input

    if candidate_input.is_dir():
        for name in _candidate_index_names(shard_suffix):
            possible = candidate_input / name
            if possible.exists():
                return candidate_input, possible

    raise FileNotFoundError(
        f"Could not resolve PT dataset root/index from: {candidate_input}"
    )


def _validate_generation_asu(asu: Any, meta: Optional[dict[str, Any]] = None):
    from sala.core.types import AsuInput
    from sala.data.datasets.pt_dataset import (
        _ensure_sala_asu_input,
        _validate_full_cell_sample,
    )
    from sala.data.datasets.asu_canonicalize import canonicalize_asu_input

    asu = _ensure_sala_asu_input(asu)
    if not isinstance(asu, AsuInput):  # defensive for external pickle payloads
        raise ValueError(f"PT payload does not contain an AsuInput: {type(asu)}")
    if getattr(asu, "symmetry_ops", None) is None:
        raise ValueError(
            "Full-cell PT payload requires explicit symmetry_ops; regenerate data"
        )

    def scalar(name: str) -> int:
        value = getattr(asu, name, None)
        if value is None and isinstance(meta, dict):
            value = meta.get(name)
        if value is None:
            raise ValueError(f"Full-cell PT payload is missing {name}")
        tensor = torch.as_tensor(value)
        if not tensor.numel():
            raise ValueError(f"Full-cell PT payload has empty {name}")
        return int(tensor.view(-1)[0].item())

    entry = {
        "spacegroup": scalar("spacegroup"),
        "n_heavy_atoms": int(asu.atom_types.shape[0]),
        "n_symops": scalar("n_symops"),
        "n_heavy_cell": scalar("n_heavy_cell"),
        "n_full_atoms": scalar("n_full_atoms"),
        "hall_number": scalar("hall_number"),
        "setting_id": scalar("setting_id"),
        "lattice_template_id": scalar("lattice_template_id"),
    }
    _validate_full_cell_sample(asu, entry)
    return canonicalize_asu_input(asu)


def _dict_to_asu_input(payload: dict[str, Any]):
    from sala.core.types import AsuInput

    required = [
        "atom_types",
        "spacegroup",
        "component_id",
        "atom_formal_charge",
        "atom_charge_mask",
        "component_formal_charge",
        "component_type_id",
        "num_atoms",
        "edge_index_conv",
        "edge_attr_conv",
        "frac_coords_gt",
        "lattice_gt",
        "n_symops",
        "n_heavy_cell",
        "n_full_atoms",
        "hall_number",
        "setting_id",
        "lattice_template_id",
        "symmetry_ops",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"PT payload missing AsuInput fields: {missing}")
    asu = AsuInput(
        atom_types=torch.as_tensor(payload["atom_types"], dtype=torch.long),
        spacegroup=torch.as_tensor(payload["spacegroup"], dtype=torch.long),
        component_id=torch.as_tensor(payload["component_id"], dtype=torch.long),
        atom_formal_charge=torch.as_tensor(
            payload["atom_formal_charge"], dtype=torch.long
        ),
        atom_charge_mask=torch.as_tensor(payload["atom_charge_mask"], dtype=torch.bool),
        component_formal_charge=torch.as_tensor(
            payload["component_formal_charge"], dtype=torch.long
        ),
        component_type_id=torch.as_tensor(payload["component_type_id"], dtype=torch.long),
        num_atoms=torch.as_tensor(payload["num_atoms"], dtype=torch.long),
        edge_index_conv=torch.as_tensor(payload["edge_index_conv"], dtype=torch.long),
        edge_attr_conv=torch.as_tensor(payload["edge_attr_conv"], dtype=torch.long),
        frac_coords_gt=torch.as_tensor(payload["frac_coords_gt"], dtype=torch.float32),
        lattice_gt=torch.as_tensor(payload["lattice_gt"], dtype=torch.float32),
        cart_coords_gt=(
            torch.as_tensor(payload["cart_coords_gt"], dtype=torch.float32)
            if "cart_coords_gt" in payload else None
        ),
        hybridization=(
            torch.as_tensor(payload["hybridization"], dtype=torch.long)
            if "hybridization" in payload else None
        ),
        atom_labels=payload.get("atom_labels"),
        cif_id=payload.get("cif_id"),
        n_symops=torch.as_tensor(payload["n_symops"], dtype=torch.long),
        n_heavy_cell=torch.as_tensor(payload["n_heavy_cell"], dtype=torch.long),
        n_full_atoms=torch.as_tensor(payload["n_full_atoms"], dtype=torch.long),
        hall_number=torch.as_tensor(payload["hall_number"], dtype=torch.long),
        setting_id=torch.as_tensor(payload["setting_id"], dtype=torch.long),
        lattice_template_id=torch.as_tensor(payload["lattice_template_id"], dtype=torch.long),
        symmetry_ops=torch.as_tensor(payload["symmetry_ops"], dtype=torch.float32),
    )
    return _validate_generation_asu(asu, payload.get("meta"))


def _coerce_pt_item(
    payload: Any,
) -> tuple[Any, Optional[dict[str, Any]]]:
    from sala.core.types import AsuInput

    if isinstance(payload, AsuInput):
        return _validate_generation_asu(payload), None

    if isinstance(payload, dict):
        if "asu" in payload:
            meta = payload.get("meta")
            return _validate_generation_asu(payload["asu"], meta), meta
        return _dict_to_asu_input(payload), payload.get("meta")
    raise ValueError(f"Unsupported PT target item format: {type(payload)}")


def _load_pt_payloads(path: Path) -> list[tuple[Any, Optional[dict[str, Any]]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
        if not isinstance(samples, list):
            raise ValueError(f"'samples' must be a list in PT target file: {path}")
        return [_coerce_pt_item(item) for item in samples]

    if isinstance(payload, list):
        return [_coerce_pt_item(item) for item in payload]

    return [_coerce_pt_item(payload)]


def _build_targets_from_pt_file(
    path: Path,
    fallback_index: int = 0,
) -> list[GenerationTarget]:
    targets: list[GenerationTarget] = []
    for offset, (asu, meta) in enumerate(_load_pt_payloads(path)):
        target_name = (
            (meta or {}).get("target_id")
            or (meta or {}).get("cif_id")
            or getattr(asu, "cif_id", None)
            or path.stem
        )
        targets.append(
            GenerationTarget(
                target_id=_make_target_id(str(target_name), fallback_index + offset),
                asu=asu,
                source_path=path,
                source_type="pt",
                reference_cif_path=None,
                meta=meta,
            )
        )
    return targets


def _build_target_from_pt_file(path: Path, fallback_index: int = 0) -> GenerationTarget:
    targets = _build_targets_from_pt_file(path, fallback_index=fallback_index)
    if len(targets) != 1:
        raise ValueError(
            f"Expected exactly 1 PT target in {path}, found {len(targets)}. "
            "Use folder/eval mode for shard or multi-sample PT files."
        )
    return targets[0]


def _build_target_from_cif_file(
    path: Path,
    filter_h: bool = True,
    fallback_index: int = 0,
) -> GenerationTarget:
    from sala.data.datasets.cif_to_asu import cif_data_to_asu_input
    from sala.data.io.cif_reader import filter_hydrogens, read_cif

    cif_data = read_cif(path)
    if filter_h:
        heavy_cif_data = filter_hydrogens(cif_data)
    else:
        heavy_cif_data = cif_data
    asu = cif_data_to_asu_input(heavy_cif_data)
    hydrogenation_evidence = (
        build_hydrogenation_evidence(cif_data, asu)
        if filter_h
        else None
    )
    target_name = heavy_cif_data.cif_id or path.stem
    meta = {
        "cif_id": heavy_cif_data.cif_id,
        "spacegroup": int(heavy_cif_data.spacegroup),
        "chemical_formula_moiety": cif_data.chemical_formula_moiety,
        "chemical_formula_sum": cif_data.chemical_formula_sum,
        "cell_formula_units_z": cif_data.cell_formula_units_z,
    }
    if hydrogenation_evidence is not None:
        meta.update(
            {
                "hydrogenation_evidence_version": HYDROGENATION_EVIDENCE_VERSION,
                "bond_order_confidence": hydrogenation_evidence[
                    "bond_order_confidence"
                ],
                "has_disorder": hydrogenation_evidence["has_disorder"],
                "hydrogenation_evidence": hydrogenation_evidence,
            }
        )
    return GenerationTarget(
        target_id=_make_target_id(str(target_name), fallback_index),
        asu=asu,
        source_path=path,
        source_type="cif",
        reference_cif_path=path,
        meta=meta,
    )


def load_single_target(
    path: Union[str, Path],
    filter_h: bool = True,
    input_format: str = "auto",
) -> GenerationTarget:
    path = Path(path)
    fmt = input_format.lower()
    if not path.is_file():
        raise FileNotFoundError(f"Single target path must be a file: {path}")
    if fmt == "cif" or path.suffix.lower() == ".cif":
        return _build_target_from_cif_file(path, filter_h=filter_h)
    if fmt == "pt" or path.suffix.lower() == ".pt":
        return _build_target_from_pt_file(path)
    raise ValueError(f"Unsupported single-target input: {path}")


def load_targets_from_folder(
    folder: Union[str, Path],
    filter_h: bool = True,
    input_format: str = "auto",
) -> list[GenerationTarget]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder input does not exist: {folder}")

    fmt = input_format.lower()
    targets: list[GenerationTarget] = []
    files: list[Path] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".pt", ".cif"}:
            continue
        if _is_index_file(path):
            continue
        if fmt == "pt" and suffix != ".pt":
            continue
        if fmt == "cif" and suffix != ".cif":
            continue
        files.append(path)

    if not files:
        raise FileNotFoundError(f"No target files found in folder: {folder}")

    next_fallback_index = 0
    for path in files:
        if path.suffix.lower() == ".cif":
            targets.append(
                _build_target_from_cif_file(
                    path,
                    filter_h=filter_h,
                    fallback_index=next_fallback_index,
                )
            )
            next_fallback_index += 1
        else:
            file_targets = _build_targets_from_pt_file(
                path, fallback_index=next_fallback_index
            )
            targets.extend(file_targets)
            next_fallback_index += len(file_targets)
    return targets


def load_targets_from_eval_index(
    input_path: Optional[Union[str, Path]],
    index_path: Optional[Union[str, Path]],
    target_filter: EvalTargetFilter,
) -> tuple[list[GenerationTarget], int]:
    from sala.data.datasets.pt_dataset import AsuPtDataset

    pt_root, pt_index = _resolve_pt_dataset_paths(
        input_path,
        index_path,
        shard_suffix=target_filter.shard_suffix,
    )
    dataset = AsuPtDataset(
        root=pt_root,
        index_path=pt_index,
        split=target_filter.split,
        c_count_min=target_filter.c_count_min,
        c_count_max=target_filter.c_count_max,
        classes=target_filter.classes,
        include_unknown=target_filter.include_unknown,
        spacegroups=target_filter.spacegroups,
        is_neutral=target_filter.is_neutral,
        has_metal=target_filter.has_metal,
        n_heavy_atoms_min=target_filter.n_heavy_atoms_min,
        n_heavy_atoms_max=target_filter.n_heavy_atoms_max,
        n_components_min=target_filter.n_components_min,
        n_components_max=target_filter.n_components_max,
        cache_shards=target_filter.shard_cache_size,
        return_meta=True,
        shard_suffix=target_filter.shard_suffix,
    )

    targets: list[GenerationTarget] = []
    for idx in range(len(dataset)):
        asu, meta = dataset[idx]
        target_name = (
            (meta or {}).get("cif_id")
            or getattr(asu, "cif_id", None)
            or f"target_{idx:04d}"
        )
        targets.append(
            GenerationTarget(
                target_id=_make_target_id(str(target_name), idx),
                asu=asu,
                source_path=pt_index,
                source_type="pt_eval",
                reference_cif_path=None,
                meta=meta,
            )
        )
    num_targets_before_subset = len(targets)
    return targets, num_targets_before_subset


def _generation_target_id(target: Any) -> str:
    return str(getattr(target, "target_id", target))


def _iter_sample_batches(
    targets: Sequence[Any],
    *,
    num_samples: int,
    sample_batch_size: int,
) -> Iterable[tuple[list[Any], list[str], list[int]]]:
    sample_batch_size = max(1, int(sample_batch_size))
    target_list = list(targets)
    for sample_start in range(0, int(num_samples), sample_batch_size):
        sample_end = min(int(num_samples), sample_start + sample_batch_size)
        expanded_targets: list[Any] = []
        expanded_target_ids: list[str] = []
        expanded_sample_indices: list[int] = []
        for sample_idx in range(sample_start, sample_end):
            for target in target_list:
                expanded_targets.append(target)
                expanded_target_ids.append(_generation_target_id(target))
                expanded_sample_indices.append(sample_idx)
        yield expanded_targets, expanded_target_ids, expanded_sample_indices


def _select_eval_targets(
    targets: Sequence[GenerationTarget],
    max_targets: Optional[int],
    target_selection: str,
    seed: Optional[int],
) -> list[GenerationTarget]:
    if max_targets is None or len(targets) <= max_targets:
        return list(targets)
    if target_selection == "first":
        return list(targets[:max_targets])
    if target_selection == "random":
        indices = list(range(len(targets)))
        rng = random.Random(0 if seed is None else seed)
        rng.shuffle(indices)
        chosen = indices[:max_targets]
        return [targets[idx] for idx in chosen]
    raise ValueError(f"Unsupported target_selection: {target_selection}")


def _normalize_generation_config_from_args(
    args: argparse.Namespace,
) -> GenerationConfig:
    include_unknown = args.include_unknown if args.include_unknown is not None else True
    eval_filter = EvalTargetFilter(
        c_count_min=args.c_count_min,
        c_count_max=args.c_count_max,
        classes=_split_csv_strings(args.classes),
        include_unknown=include_unknown,
        spacegroups=_split_csv_ints(args.spacegroups),
        is_neutral=args.is_neutral,
        has_metal=args.has_metal,
        n_heavy_atoms_min=args.n_heavy_atoms_min,
        n_heavy_atoms_max=args.n_heavy_atoms_max,
        n_components_min=args.n_components_min,
        n_components_max=args.n_components_max,
        split=args.split,
        max_targets=args.max_targets,
        target_selection=args.target_selection,
        target_selection_seed=args.target_selection_seed,
        shard_cache_size=args.shard_cache_size,
        shard_suffix=args.shard_suffix,
        filter_h=not args.no_filter_h,
    )
    return GenerationConfig(
        checkpoint_path=args.checkpoint,
        mode=args.mode,
        output_dir=Path(args.output),
        device=args.device or _default_device(),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        filter_h=not args.no_filter_h,
        sample_batch_size=args.sample_batch_size,
        save_cif=not args.no_cif,
        save_pt=not args.no_pt,
        save_reference=not args.no_reference,
        seed=int(args.seed),
        ema_profile_std=args.ema_profile_std,
        save_trajectory_cif=args.save_trajectory_cif,
        trajectory_interval=args.trajectory_interval,
        num_steps=args.num_steps,
        input=Path(args.input) if args.input else None,
        input_dir=Path(args.input_dir) if args.input_dir else None,
        input_index=Path(args.input_index) if args.input_index else None,
        pt_root=Path(args.pt_root) if args.pt_root else None,
        input_format=args.input_format,
        eval_filter=eval_filter,
        postprocess_cfg=normalize_symmetry_corrector_config(None),
        hydrogenation_cfg=normalize_hydrogenation_config(
            {
                "enabled": bool(args.hydrogenation_enabled),
                "failure_policy": args.hydrogenation_failure_policy,
                "output_subdir": args.hydrogenation_output_subdir,
            }
        ),
    )


def _normalize_generation_config_from_hydra(cfg: DictConfig) -> GenerationConfig:
    gen_cfg = flatten_grouped_config(cfg.get("generation"), GENERATION_CONFIG_SECTIONS)
    data_cfg = flatten_grouped_config(cfg.get("data"), DATA_CONFIG_SECTIONS)
    use_data_filters = bool(gen_cfg.get("use_data_filters", True))

    def merged(name: str, fallback=None):
        base = data_cfg.get(name, fallback) if use_data_filters else fallback
        return _coalesce(gen_cfg.get(name, None), base)

    include_unknown = merged("include_unknown", True)
    eval_filter = EvalTargetFilter(
        c_count_min=merged("c_count_min"),
        c_count_max=merged("c_count_max"),
        classes=_split_csv_strings(merged("classes")),
        include_unknown=True if include_unknown is None else bool(include_unknown),
        spacegroups=_split_csv_ints(merged("spacegroups")),
        is_neutral=merged("is_neutral"),
        has_metal=merged("has_metal"),
        n_heavy_atoms_min=merged("n_heavy_atoms_min"),
        n_heavy_atoms_max=merged("n_heavy_atoms_max"),
        n_components_min=merged("n_components_min"),
        n_components_max=merged("n_components_max"),
        split=gen_cfg.get("split", "val"),
        max_targets=gen_cfg.get("max_targets"),
        target_selection=str(gen_cfg.get("target_selection", "first")),
        target_selection_seed=gen_cfg.get("target_selection_seed"),
        shard_cache_size=int(
            _coalesce(
                gen_cfg.get("shard_cache_size"), data_cfg.get("shard_cache_size"), 1
            )
        ),
        shard_suffix=str(
            _coalesce(
                gen_cfg.get("shard_suffix"),
                data_cfg.get("shard_suffix"),
                DEFAULT_SHARD_SUFFIX,
            )
            or ""
        ),
        filter_h=bool(gen_cfg.get("filter_h", True)),
    )
    output_dir = gen_cfg.get("output_dir")
    if not output_dir and HYDRA_AVAILABLE and HydraConfig.initialized():
        output_dir = HydraConfig.get().runtime.output_dir
    return GenerationConfig(
        checkpoint_path=str(gen_cfg.get("checkpoint_path")),
        mode=str(gen_cfg.get("mode", "single")),
        output_dir=Path(output_dir) if output_dir else Path(os.getcwd()),
        device=str(gen_cfg.get("device") or _default_device()),
        num_samples=int(gen_cfg.get("num_samples", 20)),
        batch_size=int(
            _coalesce(gen_cfg.get("batch_size"), data_cfg.get("batch_size"), 4)
        ),
        filter_h=eval_filter.filter_h,
        sample_batch_size=int(gen_cfg.get("sample_batch_size", 1)),
        save_cif=bool(gen_cfg.get("save_cif", True)),
        save_pt=bool(gen_cfg.get("save_pt", True)),
        save_reference=bool(gen_cfg.get("save_reference", True)),
        seed=int(_coalesce(gen_cfg.get("seed"), cfg.get("seed"), 42)),
        ema_profile_std=(
            None
            if gen_cfg.get("ema_profile_std") is None
            else float(gen_cfg.get("ema_profile_std"))
        ),
        save_trajectory_cif=bool(gen_cfg.get("save_trajectory_cif", False)),
        trajectory_interval=int(gen_cfg.get("trajectory_interval", 100)),
        num_steps=gen_cfg.get("num_steps"),
        input=Path(gen_cfg.get("input")) if gen_cfg.get("input") else None,
        input_dir=Path(gen_cfg.get("input_dir")) if gen_cfg.get("input_dir") else None,
        input_index=(
            Path(gen_cfg.get("input_index")) if gen_cfg.get("input_index") else None
        ),
        pt_root=(
            Path(_coalesce(gen_cfg.get("pt_root"), data_cfg.get("pt_root")))
            if _coalesce(gen_cfg.get("pt_root"), data_cfg.get("pt_root"))
            else None
        ),
        input_format=str(gen_cfg.get("input_format", "auto")),
        eval_filter=eval_filter,
        postprocess_cfg=_extract_postprocess_cfg_from_generation_config(gen_cfg),
        hydrogenation_cfg=normalize_hydrogenation_config(
            gen_cfg.get("hydrogenation")
            if isinstance(gen_cfg.get("hydrogenation"), Mapping)
            else None
        ),
    )


def _validate_generation_config(config: GenerationConfig) -> None:
    if config.eval_filter is None:
        config.eval_filter = EvalTargetFilter()
    config.hydrogenation_cfg = normalize_hydrogenation_config(config.hydrogenation_cfg)
    if not config.checkpoint_path or config.checkpoint_path == "None":
        raise ValueError("checkpoint path is required")
    if config.mode not in {"single", "folder", "eval"}:
        raise ValueError(f"Unsupported generation mode: {config.mode}")
    if config.mode == "single" and config.input is None:
        raise ValueError("mode=single requires --input / generation.input")
    if config.mode == "folder" and config.input_dir is None:
        raise ValueError("mode=folder requires --input-dir / generation.input_dir")
    if config.mode == "eval" and all(
        v is None for v in (config.input_index, config.pt_root, config.input)
    ):
        raise ValueError("mode=eval requires input_index, pt_root, or input")
    if config.num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if config.num_steps is not None and int(config.num_steps) < 1:
        raise ValueError("num_steps must be >= 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if getattr(config, "sample_batch_size", 1) < 1:
        raise ValueError("sample_batch_size must be >= 1")
    if config.trajectory_interval < 1:
        raise ValueError("trajectory_interval must be >= 1")
    if config.hydrogenation_cfg["enabled"] and not config.filter_h:
        raise ValueError("Hydrogenation requires filter_h=true / no --no-filter-h")
    if config.eval_filter.target_selection not in {"first", "random"}:
        raise ValueError(
            f"Unsupported target_selection: {config.eval_filter.target_selection}"
        )
    if (
        config.eval_filter.max_targets is not None
        and int(config.eval_filter.max_targets) < 1
    ):
        raise ValueError("max_targets must be >= 1")


def load_generation_targets(config: GenerationConfig) -> list[GenerationTarget]:
    if config.mode == "single":
        config.eval_num_targets_before_subset = 1
        config.eval_num_targets_after_subset = 1
        targets = [
            load_single_target(
                config.input, filter_h=config.filter_h, input_format=config.input_format
            )
        ]
        return targets
    if config.mode == "folder":
        targets = load_targets_from_folder(
            config.input_dir, filter_h=config.filter_h, input_format=config.input_format
        )
        config.eval_num_targets_before_subset = len(targets)
        config.eval_num_targets_after_subset = len(targets)
        return targets

    eval_input = config.pt_root or config.input
    targets, before_subset = load_targets_from_eval_index(
        input_path=eval_input,
        index_path=config.input_index,
        target_filter=config.eval_filter,
    )
    selected_targets = _select_eval_targets(
        targets,
        max_targets=config.eval_filter.max_targets,
        target_selection=config.eval_filter.target_selection,
        seed=config.eval_filter.target_selection_seed,
    )
    config.eval_num_targets_before_subset = before_subset
    config.eval_num_targets_after_subset = len(selected_targets)
    return selected_targets


def _build_generated_pt_entries(
    batch: Any,
    frac_coords: Tensor,
    lattice: Tensor,
    target_ids: Optional[Sequence[str]] = None,
    sample_idx: Optional[Union[int, Sequence[int]]] = None,
    artifact_stage: str = "generated",
) -> list[dict[str, Any]]:
    from sala.core.lattice import (
        frac_to_cart_coords_batched,
    )
    from sala.core.types import AsuBatch
    from sala.data.collate import unbatch_asu

    if artifact_stage not in {"generated", "raw", "postprocessed"}:
        raise ValueError(
            "artifact_stage must be 'generated', 'raw', or 'postprocessed'"
        )

    if lattice.ndim == 2:
        lattice = lattice.unsqueeze(0)

    cart_coords_gt = frac_to_cart_coords_batched(frac_coords, lattice, batch.batch)
    generated_batch = AsuBatch(
        atom_types=batch.atom_types,
        spacegroup=batch.spacegroup,
        component_id=batch.component_id,
        atom_formal_charge=batch.atom_formal_charge,
        atom_charge_mask=batch.atom_charge_mask,
        component_formal_charge=batch.component_formal_charge,
        component_type_id=batch.component_type_id,
        num_atoms=batch.num_atoms,
        edge_index_conv=batch.edge_index_conv,
        edge_attr_conv=batch.edge_attr_conv,
        frac_coords_gt=frac_coords,
        lattice_gt=lattice,
        batch=batch.batch,
        ptr=batch.ptr,
        component_batch=batch.component_batch,
        cart_coords_gt=cart_coords_gt,
        hybridization=batch.hybridization,
        cif_ids=batch.cif_ids,
        n_symops=getattr(batch, "n_symops", None),
        n_heavy_cell=getattr(batch, "n_heavy_cell", None),
        n_full_atoms=getattr(batch, "n_full_atoms", None),
        hall_number=getattr(batch, "hall_number", None),
        setting_id=getattr(batch, "setting_id", None),
        lattice_template_id=getattr(batch, "lattice_template_id", None),
        symmetry_ops=getattr(batch, "symmetry_ops", None),
        symmetry_mask=getattr(batch, "symmetry_mask", None),
    )

    entries: list[dict[str, Any]] = []
    for i, asu in enumerate(unbatch_asu(generated_batch)):
        base_target_id = (
            target_ids[i]
            if target_ids is not None and i < len(target_ids)
            else getattr(asu, "cif_id", None) or f"target_{i:04d}"
        )
        if isinstance(sample_idx, Sequence) and not isinstance(
            sample_idx, (str, bytes)
        ):
            graph_sample_idx = int(sample_idx[i]) if i < len(sample_idx) else None
        else:
            graph_sample_idx = int(sample_idx) if sample_idx is not None else None
        target_id = (
            f"{base_target_id}_sample_{graph_sample_idx:03d}"
            if graph_sample_idx is not None
            else str(base_target_id)
        )
        meta: dict[str, Any] = {
            "target_id": str(target_id),
            "source_target_id": str(base_target_id),
            "coord_repr_version": FULL_CELL_DATA_SCHEMA_VERSION,
            "artifact_stage": artifact_stage,
        }
        for field in (
            "n_symops",
            "n_heavy_cell",
            "n_full_atoms",
            "hall_number",
            "setting_id",
            "lattice_template_id",
        ):
            value = getattr(asu, field, None)
            if value is not None and value.numel():
                meta[field] = int(value.view(-1)[0].item())
        if getattr(asu, "cif_id", None):
            meta["cif_id"] = str(asu.cif_id)
        if graph_sample_idx is not None:
            meta["sample_idx"] = int(graph_sample_idx)
        entries.append(
            {
                "asu": asu.to("cpu"),
                "meta": meta,
            }
        )
    return entries


def _build_reference_pt_entries(targets: Sequence[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for idx, target in enumerate(targets):
        if isinstance(target, tuple) and len(target) >= 2:
            target_id = str(target[0])
            asu = target[1]
        else:
            asu = getattr(target, "asu", target)
            target_id = str(
                getattr(target, "target_id", None)
                or getattr(asu, "cif_id", None)
                or f"target_{idx:04d}"
            )
        if not hasattr(asu, "to"):
            continue
        meta: dict[str, Any] = {
            "target_id": target_id,
            "source_target_id": target_id,
        }
        cif_id = getattr(asu, "cif_id", None)
        if cif_id:
            meta["cif_id"] = str(cif_id)
        entries.append({"asu": asu.to("cpu"), "meta": meta})
    return entries


def save_results_pt(
    samples: Sequence[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    normalized = list(samples)
    asu_list = [
        item["asu"] for item in normalized if isinstance(item, dict) and "asu" in item
    ]
    results: dict[str, Any] = {
        "samples": normalized,
        "num_generated": len(normalized),
        "target_ids": [
            str((item.get("meta") or {}).get("target_id", f"target_{idx:04d}"))
            for idx, item in enumerate(normalized)
            if isinstance(item, dict)
        ],
    }
    if asu_list:
        results.update(
            {
                "frac_coords": torch.cat(
                    [asu.frac_coords_gt.cpu() for asu in asu_list], dim=0
                ),
                "atom_types": torch.cat(
                    [asu.atom_types.cpu() for asu in asu_list], dim=0
                ),
                "lattice": torch.stack(
                    [asu.lattice_gt.cpu() for asu in asu_list], dim=0
                ),
                "num_atoms": torch.cat(
                    [asu.num_atoms.cpu() for asu in asu_list], dim=0
                ),
            }
        )
    else:
        results.update(
            {
                "frac_coords": torch.empty((0, 3)),
                "atom_types": torch.empty((0,), dtype=torch.long),
                "lattice": torch.empty((0, 3, 3)),
                "num_atoms": torch.empty((0,), dtype=torch.long),
            }
        )
    _atomic_torch_save(results, output_path)
    logger.info(f"Saved results to {output_path}")
    return {
        key: results[key]
        for key in (
            "frac_coords",
            "atom_types",
            "lattice",
            "num_atoms",
            "num_generated",
            "target_ids",
        )
    }


def save_generation_manifest(
    manifest_entries: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> Path:
    payload = []
    for target_id in sorted(manifest_entries):
        entry = manifest_entries[target_id]
        record = {
            "target_id": target_id,
            "reference": entry["reference"],
            "samples": entry["samples"],
        }
        if "hydrogenated_samples" in entry:
            record["hydrogenated_samples"] = entry["hydrogenated_samples"]
        payload.append(record)
    _atomic_write_text(output_path, json.dumps(payload, indent=2))
    logger.info(f"Saved generation manifest to {output_path}")
    return output_path


def _atom_labels_from_asu(asu) -> list[str]:
    atom_labels = getattr(asu, "atom_labels", None)
    if atom_labels and len(atom_labels) == int(asu.num_atoms.item()):
        return list(atom_labels)
    labels: list[str] = []
    for idx, atom_type in enumerate(asu.atom_types.cpu().tolist()):
        labels.append(f"{_get_element_symbol(int(atom_type))}{idx}")
    return labels


def _bonds_from_asu(asu, atom_labels: list[str]) -> list[tuple]:
    from sala.data.datasets.pt_dataset import BOND_ORDER_VALUES

    bonds: list[tuple] = []
    edge_index = asu.edge_index_conv.detach().cpu()
    edge_attr = asu.edge_attr_conv.detach().cpu()
    seen: set[tuple[int, int]] = set()
    for idx in range(edge_index.shape[1]):
        src = int(edge_index[0, idx].item())
        dst = int(edge_index[1, idx].item())
        canonical = (min(src, dst), max(src, dst))
        if canonical in seen:
            continue
        seen.add(canonical)
        order_idx = int(edge_attr[idx].item()) if idx < edge_attr.shape[0] else 0
        bond_order = (
            BOND_ORDER_VALUES[order_idx]
            if 0 <= order_idx < len(BOND_ORDER_VALUES)
            else 1.0
        )
        bonds.append((atom_labels[src], atom_labels[dst], bond_order))
    return bonds


def _bonds_from_hydrogenation_evidence(
    evidence: Mapping[str, Any], atom_labels: Sequence[str]
) -> list[tuple[str, str, float | None]]:
    edge_index = torch.as_tensor(evidence["edge_index"], dtype=torch.long)
    bond_orders = list(evidence["bond_orders"])
    bond_order_mask = list(evidence["bond_order_mask"])
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Hydrogenation evidence edge_index must have shape [2, E]")
    if not (
        len(bond_orders) == len(bond_order_mask) == int(edge_index.shape[1])
    ):
        raise ValueError("Hydrogenation evidence bond-order length is inconsistent")
    bonds: list[tuple[str, str, float | None]] = []
    seen: set[tuple[int, int]] = set()
    for position, (left, right) in enumerate(edge_index.T.tolist()):
        left = int(left)
        right = int(right)
        canonical = (min(left, right), max(left, right))
        if left == right or canonical in seen:
            continue
        seen.add(canonical)
        bonds.append(
            (
                atom_labels[left],
                atom_labels[right],
                float(bond_orders[position])
                if bool(bond_order_mask[position])
                else None,
            )
        )
    return bonds


def _discover_target_reference_cif(target: GenerationTarget) -> Path | None:
    meta = target.meta if isinstance(target.meta, Mapping) else {}
    cif_id = str(meta.get("cif_id") or target.target_id)
    filenames = [f"{Path(cif_id).name}.cif"]
    source_cif_path = meta.get("source_cif_path")
    if source_cif_path:
        filenames.append(Path(str(source_cif_path)).name)
    roots = [
        target.source_path.parent / "cifs",
        target.source_path.parent.parent / "cifs",
    ]
    for root in roots:
        for filename in dict.fromkeys(filenames):
            candidate = root / filename
            if candidate.is_file():
                return candidate
    if source_cif_path:
        candidate = Path(str(source_cif_path)).expanduser()
        if candidate.is_file():
            return candidate
    return None


def _hydrogenation_evidence_for_target(
    target: GenerationTarget,
) -> Mapping[str, Any]:
    from sala.postprocess.hydrogenation import UnsupportedHydrogenationError

    meta = target.meta if isinstance(target.meta, Mapping) else {}
    evidence = meta.get("hydrogenation_evidence")
    evidence_version = (
        evidence.get("version") or meta.get("hydrogenation_evidence_version")
        if isinstance(evidence, Mapping)
        else None
    )
    if (
        not isinstance(evidence, Mapping)
        or evidence_version != HYDROGENATION_EVIDENCE_VERSION
    ):
        reference_path = target.reference_cif_path or _discover_target_reference_cif(
            target
        )
        if reference_path is None:
            if isinstance(evidence, Mapping):
                raise UnsupportedHydrogenationError(
                    f"Unsupported hydrogenation evidence version: {evidence_version!r}; "
                    "no source CIF was found to rebuild non-leaking evidence"
                )
            raise UnsupportedHydrogenationError(
                "Missing versioned CIF hydrogenation evidence and no matching "
                "source CIF was found"
            )
        from sala.data.io.cif_reader import read_cif_chemical_evidence

        try:
            evidence = build_hydrogenation_evidence(
                read_cif_chemical_evidence(reference_path), target.asu
            )
        except (OSError, TypeError, ValueError) as exc:
            raise UnsupportedHydrogenationError(
                f"Could not align source CIF evidence from {reference_path}: {exc}"
            ) from exc
        updated_meta = dict(meta)
        updated_meta.update(
            {
                "chemical_formula_moiety": evidence.get("formula_moiety"),
                "chemical_formula_sum": evidence.get("formula_sum"),
                "cell_formula_units_z": evidence.get("formula_units_z"),
                "hydrogenation_evidence_version": HYDROGENATION_EVIDENCE_VERSION,
                "bond_order_confidence": evidence.get("bond_order_confidence"),
                "has_disorder": evidence.get("has_disorder", False),
                "hydrogenation_source_cif_path": str(reference_path),
                "hydrogenation_evidence": evidence,
            }
        )
        target.meta = updated_meta
    version = evidence.get("version")
    if version != HYDROGENATION_EVIDENCE_VERSION:
        raise UnsupportedHydrogenationError(
            f"Unsupported hydrogenation evidence version: {version!r}"
        )
    conflicts = list(evidence.get("attached_hydrogen_conflict_labels") or [])
    if conflicts:
        raise UnsupportedHydrogenationError(
            "Conflicting explicit/attached hydrogen evidence for labels: "
            + ", ".join(str(value) for value in conflicts)
        )
    num_atoms = len(evidence.get("atom_labels") or [])
    attached_hydrogens = list(evidence.get("attached_hydrogens") or [])
    attached_mask = list(evidence.get("attached_hydrogen_mask") or [])
    attached_sources = list(evidence.get("attached_hydrogen_sources") or [])
    if not (
        len(attached_hydrogens)
        == len(attached_mask)
        == len(attached_sources)
        == num_atoms
    ):
        raise UnsupportedHydrogenationError(
            "Hydrogenation attached-site evidence lengths do not match atom labels"
        )
    for known, source in zip(attached_mask, attached_sources):
        expected = "atom_site_attached_hydrogens" if bool(known) else "unknown"
        if str(source) != expected:
            raise UnsupportedHydrogenationError(
                "Production attached-H evidence must come only from "
                "_atom_site_attached_hydrogens"
            )
    if bool(evidence.get("has_disorder", False)):
        raise UnsupportedHydrogenationError(
            "Disordered or partially occupied structures are not supported"
        )
    for occupancy in evidence.get("occupancies") or []:
        if occupancy is not None and abs(float(occupancy) - 1.0) > 1.0e-6:
            raise UnsupportedHydrogenationError(
                "Disordered or partially occupied structures are not supported"
            )
    n_symops = int(evidence.get("n_symops") or 0)
    for multiplicity in evidence.get("site_multiplicities") or []:
        if multiplicity is not None and n_symops > 0 and int(multiplicity) != n_symops:
            raise UnsupportedHydrogenationError(
                "Special-position atom sites are not supported"
            )
    if bool(evidence.get("has_symmetry_bonds", False)):
        raise UnsupportedHydrogenationError(
            "Covalent bonds across symmetry operations are not supported"
        )
    for key in (
        "edge_index",
        "bond_orders",
        "bond_order_mask",
        "bond_order_sources",
        "component_formal_charges",
        "component_formal_charge_mask",
        "component_formal_charge_sources",
        "num_symmetry_bonds",
        "has_symmetry_bonds",
    ):
        if key not in evidence:
            raise UnsupportedHydrogenationError(
                f"Hydrogenation evidence is missing bond provenance field {key!r}"
            )
    return evidence


def _canonical_edge_pairs(edge_index: Any) -> set[tuple[int, int]]:
    edges = torch.as_tensor(edge_index, dtype=torch.long).detach().cpu()
    if edges.ndim != 2 or edges.shape[0] != 2:
        return set()
    return {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in edges.T.tolist()
        if int(left) != int(right)
    }


def _infer_target_hydrogenation_plan(target: GenerationTarget):
    from sala.postprocess.hydrogenation import (
        HydrogenationError,
        UnsupportedHydrogenationError,
        infer_hydrogenation_plan,
    )

    asu = target.asu
    atom_types = asu.atom_types.detach().cpu().tolist()
    if any(int(value) == 1 for value in atom_types):
        raise UnsupportedHydrogenationError(
            "Hydrogenation expects a heavy-atom-only ASU"
        )
    evidence = _hydrogenation_evidence_for_target(target)
    labels = _atom_labels_from_asu(asu)
    evidence_labels = [str(value) for value in evidence.get("atom_labels") or []]
    if evidence_labels != labels:
        raise UnsupportedHydrogenationError(
            "Hydrogenation evidence atom labels do not match the generation ASU"
        )
    evidence_types = [int(value) for value in evidence.get("atom_types") or []]
    if evidence_types != atom_types:
        raise UnsupportedHydrogenationError(
            "Hydrogenation evidence atom types do not match the generation ASU"
        )
    components = asu.component_id.detach().cpu().tolist()
    evidence_components = [int(value) for value in evidence.get("component_id") or []]
    if evidence_components != components:
        raise UnsupportedHydrogenationError(
            "Hydrogenation evidence components do not match the generation ASU"
        )
    try:
        edge_index = torch.as_tensor(evidence["edge_index"], dtype=torch.long)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise UnsupportedHydrogenationError(
            f"Malformed hydrogenation edge_index: {exc}"
        ) from exc
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise UnsupportedHydrogenationError(
            "Hydrogenation evidence edge_index must have shape [2, E]"
        )
    num_edges = int(edge_index.shape[1])
    if any(
        len(evidence[key]) != num_edges
        for key in ("bond_orders", "bond_order_mask", "bond_order_sources")
    ):
        raise UnsupportedHydrogenationError(
            "Hydrogenation bond provenance lengths do not match edge_index"
        )
    if _canonical_edge_pairs(edge_index) != _canonical_edge_pairs(asu.edge_index_conv):
        raise UnsupportedHydrogenationError(
            "Hydrogenation evidence bonds do not match the generation ASU"
        )
    try:
        component_charge_values = list(
            evidence.get("component_formal_charges") or []
        )
        component_charge_mask = list(
            evidence.get("component_formal_charge_mask") or []
        )
        component_charge_sources = list(
            evidence.get("component_formal_charge_sources") or []
        )
        if not (
            len(component_charge_values)
            == len(component_charge_mask)
            == len(component_charge_sources)
        ):
            raise ValueError("component charge evidence lengths do not match")
        component_charge_evidence = [
            int(value) if bool(known) else None
            for value, known in zip(component_charge_values, component_charge_mask)
        ]
        if any(
            str(source) != ("trusted_atom_site_sum" if bool(known) else "unknown")
            for source, known in zip(
                component_charge_sources, component_charge_mask
            )
        ):
            raise ValueError("component charge evidence has invalid provenance")
        return infer_hydrogenation_plan(
            atom_types=atom_types,
            component_id=components,
            edge_index=edge_index,
            bond_orders=evidence["bond_orders"],
            bond_order_encoding="order",
            bond_order_mask=evidence["bond_order_mask"],
            formula_moiety=evidence.get("formula_moiety"),
            formula_sum=evidence.get("formula_sum"),
            formula_units_z=evidence.get("formula_units_z"),
            n_symops=int(evidence.get("n_symops") or 1),
            enforce_formula_units=False,
            atom_formal_charges=evidence.get("atom_formal_charges"),
            atom_charge_mask=evidence.get("atom_charge_mask"),
            component_formal_charges=component_charge_evidence,
            hybridization=evidence.get("hybridization"),
            attached_hydrogens=evidence.get("attached_hydrogens"),
            attached_hydrogen_mask=evidence.get("attached_hydrogen_mask"),
            charge_evidence_mode="formula",
            strict=True,
        )
    except HydrogenationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise UnsupportedHydrogenationError(
            f"Malformed hydrogenation evidence: {exc}"
        ) from exc


def _prepare_hydrogenation_plans(
    targets: Sequence[GenerationTarget],
    hydrogenation_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    from sala.postprocess.hydrogenation import HydrogenationError

    plans: dict[str, Any] = {}
    failures: dict[str, dict[str, str]] = {}
    for target in targets:
        try:
            plans[target.target_id] = _infer_target_hydrogenation_plan(target)
        except HydrogenationError as exc:
            message = f"Hydrogenation plan failed for {target.target_id}: {exc}"
            if hydrogenation_cfg["failure_policy"] == "strict":
                raise type(exc)(message) from exc
            logger.warning(message)
            failures[target.target_id] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    return plans, failures


def _write_hydrogenated_target_cif(
    target: GenerationTarget,
    frac_coords: Tensor,
    lattice: Tensor,
    plan: Any,
    output_path: Path,
):
    from sala.core.symmetry import extract_explicit_symmetry_ops
    from sala.postprocess.hydrogenation import hydrogenate_structure

    asu = target.asu
    heavy_labels = _atom_labels_from_asu(asu)
    evidence = _hydrogenation_evidence_for_target(target)
    # CIF emission is CPU-bound. Keeping the deterministic placement there
    # avoids thousands of tiny GPU kernels and host synchronizations.
    frac_coords = frac_coords.detach().cpu()
    lattice = lattice.detach().cpu()
    if lattice.ndim == 3:
        lattice = lattice[0]
    structure = hydrogenate_structure(
        atom_types=asu.atom_types.detach().cpu(),
        frac_coords=frac_coords,
        lattice=lattice,
        edge_index=asu.edge_index_conv.detach().cpu(),
        plan=plan,
        hybridization=asu.hybridization.detach().cpu(),
        atom_labels=heavy_labels,
    )
    bonds = _bonds_from_hydrogenation_evidence(evidence, heavy_labels)
    for parent, hydrogen in structure.added_edge_index.detach().cpu().T.tolist():
        bonds.append(
            (
                structure.atom_labels[int(parent)],
                structure.atom_labels[int(hydrogen)],
                1.0,
            )
        )
    atom_symbols = [
        _get_element_symbol(int(value))
        for value in asu.atom_types.detach().cpu().tolist()
    ] + list(structure.hydrogen_isotopes)
    heavy_charges: list[Optional[int]] = [
        int(charge) if resolved else None
        for charge, resolved in zip(plan.formal_charges, plan.formal_charge_resolved)
    ]
    formal_charges = heavy_charges + [0] * len(structure.hydrogen_isotopes)
    cif_content = _build_minimal_cif(
        atom_labels=list(structure.atom_labels),
        atom_types=structure.atom_types.detach().cpu().tolist(),
        atom_symbols=atom_symbols,
        formula_override=_formula_from_symbols(atom_symbols),
        formal_charges=formal_charges,
        frac_coords=structure.frac_coords,
        lattice=lattice,
        bonds=bonds,
        data_block_name=target.target_id,
        space_group_number=int(asu.spacegroup.view(-1)[0].item()),
        hall_number=int(asu.hall_number.view(-1)[0].item()),
        symmetry_ops=extract_explicit_symmetry_ops(asu),
    )
    _atomic_write_text(output_path, cif_content)
    return structure


def _hydrogenation_success_record(
    *,
    target: GenerationTarget,
    sample_idx: int,
    plan: Any,
    output_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    labels = _atom_labels_from_asu(target.asu)
    atom_types = target.asu.atom_types.detach().cpu().tolist()
    evidence = _hydrogenation_evidence_for_target(target)
    atoms = []
    for index, label in enumerate(labels):
        state = plan.resolved_states[index]
        atoms.append(
            {
                "atom_index": index,
                "atom_label": label,
                "element": _get_element_symbol(int(atom_types[index])),
                "attached_hydrogens": int(plan.parent_h_counts[index]),
                "formal_charge": int(plan.formal_charges[index]),
                "formal_charge_resolved": bool(plan.formal_charge_resolved[index]),
                "valence_state": state.label,
                "evidence": plan.evidence[index],
            }
        )
    return {
        "target_id": target.target_id,
        "sample_idx": int(sample_idx),
        "status": "hydrogenated",
        "sidecar": _relative_path(output_path, output_root),
        "plan_status": plan.status,
        "formula_source": plan.formula_source,
        "formula_moiety": evidence.get("formula_moiety"),
        "formula_sum": evidence.get("formula_sum"),
        "formula_units_z": evidence.get("formula_units_z"),
        "n_symops": evidence.get("n_symops"),
        "bond_order_confidence": evidence.get("bond_order_confidence"),
        "target_hydrogen_count": int(plan.target_hydrogen_count),
        "target_total_charge": int(plan.target_total_charge),
        "component_hydrogen_counts": list(plan.component_h_counts),
        "component_charge_totals": list(plan.component_charge_totals),
        "isotope_counts": dict(plan.isotope_counts),
        "charge_evidence_fallback_components": list(
            getattr(plan, "charge_evidence_fallback_components", ())
        ),
        "score": float(plan.score),
        "atoms": atoms,
    }


def _hydrogenation_failure_record(
    *,
    target: GenerationTarget,
    sample_idx: int,
    failure: Mapping[str, str],
    stage: str,
) -> dict[str, Any]:
    return {
        "target_id": target.target_id,
        "sample_idx": int(sample_idx),
        "status": "kept_heavy",
        "failure_stage": stage,
        "error_type": failure["error_type"],
        "error": failure["error"],
    }


def _write_target_cif(
    target: GenerationTarget,
    frac_coords: Tensor,
    lattice: Tensor,
    output_path: Path,
    filter_h: bool = True,
) -> Path:
    from sala.data.io.cif_reader import filter_hydrogens, read_cif

    asu = target.asu
    frac_coords = frac_coords.detach().cpu()
    lattice = lattice.detach().cpu()
    if lattice.dim() == 3:
        lattice = lattice[0]
    prepared_coords = frac_coords

    output_path.parent.mkdir(parents=True, exist_ok=True)

    template_path = target.reference_cif_path
    if template_path is not None and template_path.exists():
        cif_data = read_cif(template_path)
        if filter_h:
            cif_data = filter_hydrogens(cif_data)
        if len(cif_data.atom_labels) == prepared_coords.shape[0]:
            template = template_path.read_text(encoding="utf-8")
            cif_content = _build_cif_from_template(
                template=template,
                atom_labels=cif_data.atom_labels,
                atom_types=cif_data.atom_types,
                bonds=cif_data.bonds,
                frac_coords=prepared_coords,
                lattice=lattice,
            )
            _atomic_write_text(output_path, cif_content)
            return output_path

    atom_labels = _atom_labels_from_asu(asu)
    bonds = _bonds_from_asu(asu, atom_labels)
    from sala.core.symmetry import extract_explicit_symmetry_ops

    symmetry_ops = extract_explicit_symmetry_ops(asu)
    hall_number = int(asu.hall_number.view(-1)[0].item())
    formal_charges = None
    formula_override = None
    if target.source_type == "smiles":
        charges = asu.atom_formal_charge.detach().cpu().tolist()
        masks = asu.atom_charge_mask.detach().cpu().tolist()
        formal_charges = [int(charge) if known else None for charge, known in zip(charges, masks)]
        # Separate charged moieties preserve component charge on CIF reload.
        # The exported atom loop contains heavy atoms only.
        components = asu.component_id.detach().cpu()
        elements = asu.atom_types.detach().cpu()
        moieties = []
        for component, charge in enumerate(asu.component_formal_charge.detach().cpu().tolist()):
            formula = _formula_from_atom_types(elements[components == component].tolist())
            if int(charge):
                formula += f" {abs(int(charge))}{'+' if int(charge) > 0 else '-'}"
            moieties.append(formula)
        formula_override = ", ".join(moieties)
    cif_content = _build_minimal_cif(
        atom_labels=atom_labels,
        atom_types=asu.atom_types.detach().cpu().tolist(),
        frac_coords=prepared_coords,
        lattice=lattice,
        bonds=bonds,
        data_block_name=target.target_id,
        space_group_number=int(asu.spacegroup[0].item()),
        hall_number=hall_number,
        symmetry_ops=symmetry_ops,
        formal_charges=formal_charges,
        formula_override=formula_override,
    )
    _atomic_write_text(output_path, cif_content)
    return output_path


def _format_trajectory_timestep(timestep: Union[int, float]) -> str:
    if isinstance(timestep, int):
        return f"{timestep:04d}"
    return f"{float(timestep):.6f}".replace("-", "m").replace(".", "p")


def _write_trajectory_frames(
    target: GenerationTarget,
    coords_traj: Sequence[Tensor],
    lattice_traj: Sequence[Tensor],
    timesteps: Sequence[Union[int, float]],
    output_dir: Path,
    filter_h: bool = True,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for frame_idx, (coords, lattice, timestep) in enumerate(
        zip(coords_traj, lattice_traj, timesteps)
    ):
        filename = f"frame_{frame_idx:04d}_t{_format_trajectory_timestep(timestep)}.cif"
        output_path = output_dir / filename
        _write_target_cif(
            target,
            coords,
            lattice,
            output_path,
            filter_h=filter_h,
        )
        entries.append(
            {
                "frame_index": frame_idx,
                "timestep": (
                    float(timestep) if isinstance(timestep, float) else int(timestep)
                ),
                "path": str(output_path),
            }
        )
    return entries


def _extract_postprocess_cfg_from_generation_config(
    gen_cfg: dict[str, Any]
) -> dict[str, Any]:
    corrector_cfg = (
        gen_cfg.get("corrector") if isinstance(gen_cfg.get("corrector"), dict) else {}
    )
    symmetry_cfg = (
        corrector_cfg.get("symmetry_rigid")
        if isinstance(corrector_cfg.get("symmetry_rigid"), dict)
        else {}
    )
    cfg = dict(symmetry_cfg)
    return normalize_symmetry_corrector_config(
        {k: v for k, v in cfg.items() if v is not None}
    )


def _build_sampler_from_config(model, config: GenerationConfig, metadata: dict):
    from sala.flow_matching import Flow2MSampler

    denoiser = model.denoiser
    effective_num_steps = _resolve_num_steps(config.num_steps, model)
    flow_config = dict(metadata["flow_config"])
    state_cfg = dict(flow_config.get("state") or {})
    time_condition_cfg = dict(flow_config.get("time_condition") or {})
    geometry_cfg = dict(flow_config.get("geometry") or {})
    sampler_cfg = dict(flow_config.get("sampler") or {})
    prior_cfg = dict(flow_config.get("prior") or {})
    self_conditioning_cfg = dict(flow_config.get("self_conditioning") or {})
    temperature_cfg = dict(sampler_cfg.get("prior_temperature") or {})
    coord_temperature = float(temperature_cfg.get("coord", 1.0))
    lattice_temperature = float(temperature_cfg.get("lattice", 1.0))
    logger.info(
        "Flow sampler: solver=%s, steps=%s, grid=%s, "
        "prior_temperature coord=%s lattice=%s",
        sampler_cfg.get("solver", "flow_2m"),
        effective_num_steps,
        sampler_cfg.get("time_grid", "uniform"),
        coord_temperature,
        lattice_temperature,
    )
    sampler = Flow2MSampler(
        denoiser=denoiser,
        lattice_prior=model.lattice_prior,
        prior_cfg=prior_cfg,
        coord_scale=float(state_cfg.get("coord_scale", 8.0)),
        time_condition_scale=float(time_condition_cfg.get("scale", 1000.0)),
        geometry_stabilization_start=float(
            geometry_cfg.get("stabilization_start", 0.5)
        ),
        geometry_latent_clip=float(geometry_cfg.get("latent_clip", 1.5)),
        count_nfe=bool(sampler_cfg.get("count_nfe", True)),
        self_conditioning_cfg=self_conditioning_cfg,
        coord_prior_temperature=coord_temperature,
        lattice_prior_temperature=lattice_temperature,
    )
    return sampler, effective_num_steps


def _write_generation_summary(
    config: GenerationConfig,
    num_targets: int,
    num_generated: int,
    num_trajectory_frames: int,
    output_dir: Path,
    flow_config: dict | None = None,
    hydrogenation_stats: Optional[Mapping[str, Any]] = None,
) -> Path:
    postprocess_enabled = bool(
        config.postprocess_cfg
        and config.postprocess_cfg.get("postprocess_enabled", False)
    )
    checkpoint_path = Path(config.checkpoint_path).expanduser().resolve()
    hydrogenation_cfg = normalize_hydrogenation_config(config.hydrogenation_cfg)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": (
            _sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
        ),
        "mode": config.mode,
        "num_targets": num_targets,
        "num_generated": num_generated,
        "num_samples_per_target": config.num_samples,
        "sample_batch_size": int(getattr(config, "sample_batch_size", 1)),
        "sampling_seed": int(config.seed),
        "ema_profile_std": config.ema_profile_std,
        "model_family": "sala_full_cell_crystal_dit_rectified_flow",
        "num_steps": config.num_steps,
        "device": config.device,
        "save_cif": config.save_cif,
        "save_pt": config.save_pt,
        "save_reference": config.save_reference,
        "save_trajectory_cif": config.save_trajectory_cif,
        "trajectory_interval": config.trajectory_interval,
        "num_trajectory_frames": num_trajectory_frames,
        "artifact_stage": "postprocessed" if postprocess_enabled else "raw",
        "postprocess_enabled": postprocess_enabled,
        "postprocess_config": config.postprocess_cfg,
        "hydrogenation": {
            **hydrogenation_cfg,
            **dict(hydrogenation_stats or {}),
        },
        "raw_pt_artifact_saved": bool(
            config.save_pt and postprocess_enabled
        ),
        "raw_pt_artifact": (
            "generated_results_raw.pt"
            if config.save_pt and postprocess_enabled
            else None
        ),
        "num_targets_before_subset": config.eval_num_targets_before_subset,
        "num_targets_after_subset": config.eval_num_targets_after_subset or num_targets,
        "filters": {
            "split": config.eval_filter.split,
            "c_count_min": config.eval_filter.c_count_min,
            "c_count_max": config.eval_filter.c_count_max,
            "classes": config.eval_filter.classes,
            "include_unknown": config.eval_filter.include_unknown,
            "spacegroups": config.eval_filter.spacegroups,
            "is_neutral": config.eval_filter.is_neutral,
            "has_metal": config.eval_filter.has_metal,
            "n_heavy_atoms_min": config.eval_filter.n_heavy_atoms_min,
            "n_heavy_atoms_max": config.eval_filter.n_heavy_atoms_max,
            "n_components_min": config.eval_filter.n_components_min,
            "n_components_max": config.eval_filter.n_components_max,
        },
        "target_subset": {
            "max_targets": config.eval_filter.max_targets,
            "target_selection": config.eval_filter.target_selection,
            "target_selection_seed": config.eval_filter.target_selection_seed,
        },
    }
    if flow_config:
        summary["flow_config"] = flow_config
    output_path = output_dir / "summary.json"
    _atomic_write_text(output_path, json.dumps(summary, indent=2))
    return output_path


@torch.no_grad()
def generate_for_targets(
    model,
    targets: Sequence[GenerationTarget],
    config: GenerationConfig,
    metadata: dict,
) -> Dict[str, Any]:
    from sala.data.collate import asu_collate_fn

    _validate_unique_target_ids(targets)
    hydrogenation_cfg = normalize_hydrogenation_config(config.hydrogenation_cfg)
    config.hydrogenation_cfg = hydrogenation_cfg
    if hydrogenation_cfg["enabled"] and not config.filter_h:
        raise ValueError("Hydrogenation requires a heavy-atom-only target")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_root = output_dir / "generated"
    reference_root = output_dir / "reference"
    hydrogenated_root = output_dir / hydrogenation_cfg["output_subdir"]

    hydrogenation_plans: dict[str, Any] = {}
    hydrogenation_target_failures: dict[str, dict[str, str]] = {}
    if hydrogenation_cfg["enabled"]:
        hydrogenation_plans, hydrogenation_target_failures = (
            _prepare_hydrogenation_plans(targets, hydrogenation_cfg)
        )

    sampler, effective_num_steps = _build_sampler_from_config(model, config, metadata)
    logger.info("Starting generation for %d targets", len(targets))
    logger.info(
        "Sampler: Flow Matching %s",
        dict(metadata.get("flow_config", {}).get("sampler", {})).get(
            "solver", "flow_2m"
        ),
    )

    manifest_entries: Dict[str, Dict[str, Any]] = {}
    all_coords: list[Tensor] = []
    all_types: list[Tensor] = []
    all_lattices: list[Tensor] = []
    all_num_atoms: list[Tensor] = []
    all_paths: list[Path] = []
    hydrogenated_paths: list[Path] = []
    hydrogenation_report: list[dict[str, Any]] = []
    hydrogenation_stats = {
        "num_target_plans": len(hydrogenation_plans),
        "num_target_plan_failures": len(hydrogenation_target_failures),
        "num_hydrogenated_samples": 0,
        "num_kept_heavy_samples": 0,
        "num_added_hydrogens": 0,
        "report": (
            "hydrogenation_report.jsonl" if hydrogenation_cfg["enabled"] else None
        ),
    }
    pt_entries: list[dict[str, Any]] = []
    raw_pt_entries: list[dict[str, Any]] = []
    postprocess_enabled = bool(
        config.postprocess_cfg
        and config.postprocess_cfg.get("postprocess_enabled", False)
    )
    reference_pt_entries = (
        _build_reference_pt_entries(targets)
        if config.save_pt and config.save_reference
        else []
    )
    trajectory_entries: dict[str, dict[str, Any]] = {}
    num_trajectory_frames = 0
    num_generated = 0

    infer_bs = max(1, int(config.batch_size))
    n_targets = len(targets)
    for start_idx in range(0, n_targets, infer_bs):
        batch_targets = list(targets[start_idx : start_idx + infer_bs])
        batch = asu_collate_fn([target.asu for target in batch_targets]).to(
            config.device
        )
        batch_end = start_idx + len(batch_targets)
        logger.info(
            "Processing targets %d-%d / %d (batch_size=%d)",
            start_idx + 1,
            batch_end,
            n_targets,
            len(batch_targets),
        )

        if config.save_reference:
            for target in batch_targets:
                reference_path = reference_root / f"{target.target_id}.cif"
                _write_target_cif(
                    target,
                    target.asu.frac_coords_gt,
                    target.asu.lattice_gt,
                    reference_path,
                    filter_h=config.filter_h,
                )
                entry = manifest_entries.setdefault(
                    target.target_id,
                    {
                        "reference": _relative_path(reference_path, output_dir),
                        "samples": [],
                        **(
                            {"hydrogenated_samples": []}
                            if hydrogenation_cfg["enabled"]
                            else {}
                        ),
                    },
                )
                entry["reference"] = _relative_path(reference_path, output_dir)

        sample_batch_size = max(1, int(getattr(config, "sample_batch_size", 1)))
        for (
            expanded_targets,
            expanded_target_ids,
            expanded_sample_indices,
        ) in _iter_sample_batches(
            batch_targets,
            num_samples=config.num_samples,
            sample_batch_size=sample_batch_size,
        ):
            expanded_batch = asu_collate_fn(
                [target.asu for target in expanded_targets]
            ).to(config.device)
            logger.info(
                "  Sampling samples %03d-%03d / %03d (expanded_batch_size=%d)",
                min(expanded_sample_indices),
                max(expanded_sample_indices),
                config.num_samples - 1,
                len(expanded_targets),
            )
            sample_result = sampler.sample(
                batch=expanded_batch,
                num_steps=effective_num_steps,
                return_trajectory=config.save_trajectory_cif,
                trajectory_interval=config.trajectory_interval,
                progress_bar=True,
            )
            if config.save_trajectory_cif:
                output, trajectory = sample_result
            else:
                output = sample_result
                trajectory = None

            if config.save_pt and postprocess_enabled:
                raw_pt_entries.extend(
                    _build_generated_pt_entries(
                        batch=expanded_batch,
                        frac_coords=output.frac_coords.detach(),
                        lattice=output.lattice.detach(),
                        target_ids=expanded_target_ids,
                        sample_idx=expanded_sample_indices,
                        artifact_stage="raw",
                    )
                )
            if postprocess_enabled:
                correction = symmetry_rigid_postprocess_batch(
                    batch=expanded_batch,
                    frac_coords=output.frac_coords,
                    lattice=output.lattice,
                    postprocess_cfg=config.postprocess_cfg,
                )
                output.frac_coords = correction.frac_coords
                output.lattice = correction.lattice
                from sala.core.lattice import frac_to_cart_coords_batched
                from sala.core.lattice_manifold import lattice_to_q
                from sala.core.orbit import build_orbit_plan

                lattice_q = lattice_to_q(
                    output.lattice.float(),
                    spacegroup=expanded_batch.spacegroup,
                    lattice_template_id=getattr(
                        expanded_batch, "lattice_template_id", None
                    ),
                ).to(device=output.lattice_q.device, dtype=output.lattice_q.dtype)
                cart = frac_to_cart_coords_batched(
                    output.frac_coords.float(),
                    output.lattice.float(),
                    expanded_batch.batch,
                ).to(device=output.cart_coords.device, dtype=output.cart_coords.dtype)
                full = build_orbit_plan(
                    expanded_batch, validate_general_position=True
                ).expand_coordinates(cart, output.lattice)
                output.lattice_q = lattice_q
                output.cart_coords = cart
                output.full_frac_coords = full.frac
                output.full_cart_coords = full.cart
                output.full_atom_mask = full.valid

            if config.save_pt:
                pt_entries.extend(
                    _build_generated_pt_entries(
                        batch=expanded_batch,
                        frac_coords=output.frac_coords.detach(),
                        lattice=output.lattice.detach(),
                        target_ids=expanded_target_ids,
                        sample_idx=expanded_sample_indices,
                        artifact_stage=(
                            "postprocessed" if postprocess_enabled else "raw"
                        ),
                    )
                )
            else:
                all_coords.append(output.frac_coords.cpu())
                all_types.append(expanded_batch.atom_types.cpu())
                _lat = output.lattice.detach().cpu()
                if _lat.ndim == 2:
                    _lat = _lat.unsqueeze(0)
                all_lattices.append(_lat)
                all_num_atoms.append(expanded_batch.num_atoms.cpu())

            for local_idx, (target, sample_idx) in enumerate(
                zip(expanded_targets, expanded_sample_indices)
            ):
                sample_start = int(expanded_batch.ptr[local_idx].item())
                sample_end = int(expanded_batch.ptr[local_idx + 1].item())
                target_sample_dir = generated_root / target.target_id
                sample_path = target_sample_dir / f"sample_{sample_idx:03d}.cif"
                lattice_i = output.lattice
                if lattice_i.ndim == 3:
                    lattice_i = lattice_i[local_idx]
                if config.save_cif:
                    _write_target_cif(
                        target,
                        output.frac_coords[sample_start:sample_end],
                        lattice_i,
                        sample_path,
                        filter_h=config.filter_h,
                    )
                    all_paths.append(sample_path)

                entry = manifest_entries.setdefault(
                    target.target_id,
                    {
                        "reference": None,
                        "samples": [],
                        **(
                            {"hydrogenated_samples": []}
                            if hydrogenation_cfg["enabled"]
                            else {}
                        ),
                    },
                )
                if config.save_reference and entry["reference"] is None:
                    reference_path = reference_root / f"{target.target_id}.cif"
                    entry["reference"] = _relative_path(reference_path, output_dir)
                if config.save_cif:
                    entry["samples"].append(_relative_path(sample_path, output_dir))
                if hydrogenation_cfg["enabled"]:
                    plan = hydrogenation_plans.get(target.target_id)
                    target_failure = hydrogenation_target_failures.get(target.target_id)
                    if plan is None:
                        if target_failure is None:
                            raise RuntimeError(
                                f"Missing hydrogenation outcome for {target.target_id}"
                            )
                        hydrogenation_report.append(
                            _hydrogenation_failure_record(
                                target=target,
                                sample_idx=sample_idx,
                                failure=target_failure,
                                stage="plan",
                            )
                        )
                        hydrogenation_stats["num_kept_heavy_samples"] += 1
                    else:
                        from sala.postprocess.hydrogenation import (
                            HydrogenationError,
                        )

                        hydrogenated_path = (
                            hydrogenated_root
                            / target.target_id
                            / f"sample_{sample_idx:03d}.cif"
                        )
                        try:
                            structure = _write_hydrogenated_target_cif(
                                target=target,
                                frac_coords=output.frac_coords[sample_start:sample_end],
                                lattice=lattice_i,
                                plan=plan,
                                output_path=hydrogenated_path,
                            )
                        except HydrogenationError as exc:
                            if hydrogenation_cfg["failure_policy"] == "strict":
                                raise
                            failure = {
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                            logger.warning(
                                "Hydrogen placement failed for %s sample %03d: %s",
                                target.target_id,
                                sample_idx,
                                exc,
                            )
                            hydrogenation_report.append(
                                _hydrogenation_failure_record(
                                    target=target,
                                    sample_idx=sample_idx,
                                    failure=failure,
                                    stage="placement",
                                )
                            )
                            hydrogenation_stats["num_kept_heavy_samples"] += 1
                        else:
                            hydrogenated_paths.append(hydrogenated_path)
                            entry["hydrogenated_samples"].append(
                                _relative_path(hydrogenated_path, output_dir)
                            )
                            hydrogenation_report.append(
                                _hydrogenation_success_record(
                                    target=target,
                                    sample_idx=sample_idx,
                                    plan=plan,
                                    output_path=hydrogenated_path,
                                    output_root=output_dir,
                                )
                            )
                            hydrogenation_stats["num_hydrogenated_samples"] += 1
                            hydrogenation_stats["num_added_hydrogens"] += int(
                                structure.hydrogen_parent_indices.numel()
                            )
                if trajectory is not None:
                    traj_sample_dir = (
                        target_sample_dir / f"sample_{sample_idx:03d}_trajectory"
                    )
                    frame_entries = _write_trajectory_frames(
                        target=target,
                        coords_traj=[
                            coords[sample_start:sample_end].detach().cpu()
                            for coords in trajectory.coords_traj
                        ],
                        lattice_traj=[
                            (
                                lattice_batch[local_idx]
                                if lattice_batch.ndim == 3
                                else lattice_batch
                            )
                            .detach()
                            .cpu()
                            for lattice_batch in trajectory.lattice_traj
                        ],
                        timesteps=trajectory.times,
                        output_dir=traj_sample_dir,
                        filter_h=config.filter_h,
                    )
                    for frame_entry in frame_entries:
                        frame_entry["path"] = _relative_path(
                            Path(frame_entry["path"]), output_dir
                        )
                    trajectory_entries.setdefault(target.target_id, {})[
                        f"sample_{sample_idx:03d}"
                    ] = {
                        "final_sample": (
                            _relative_path(sample_path, output_dir)
                            if config.save_cif
                            else None
                        ),
                        "frames": frame_entries,
                    }
                    num_trajectory_frames += len(frame_entries)
                num_generated += 1

    results: dict[str, Any] = {
        "num_generated": num_generated,
        "saved_paths": all_paths,
    }
    if hydrogenation_cfg["enabled"]:
        report_path = output_dir / "hydrogenation_report.jsonl"
        report_content = "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in hydrogenation_report
        )
        _atomic_write_text(report_path, report_content)
        results["hydrogenated_paths"] = hydrogenated_paths
        results["hydrogenation_report_path"] = report_path

    if config.save_pt:
        pt_path = output_dir / "generated_results.pt"
        results.update(save_results_pt(pt_entries, pt_path))
        results["pt_path"] = pt_path
        if raw_pt_entries:
            raw_pt_path = output_dir / "generated_results_raw.pt"
            save_results_pt(raw_pt_entries, raw_pt_path)
            results["raw_pt_path"] = raw_pt_path
        if config.save_reference:
            reference_pt_path = output_dir / "reference_results.pt"
            save_results_pt(reference_pt_entries, reference_pt_path)
            results["reference_pt_path"] = reference_pt_path
    else:
        results.update(
            {
                "frac_coords": (
                    torch.cat(all_coords, dim=0)
                    if all_coords
                    else torch.empty((0, 3))
                ),
                "atom_types": (
                    torch.cat(all_types, dim=0)
                    if all_types
                    else torch.empty((0,), dtype=torch.long)
                ),
                "lattice": (
                    torch.cat(all_lattices, dim=0)
                    if all_lattices
                    else torch.empty((0, 3, 3))
                ),
                "num_atoms": (
                    torch.cat(all_num_atoms, dim=0)
                    if all_num_atoms
                    else torch.empty((0,), dtype=torch.long)
                ),
            }
        )

    manifest_entries = {
        key: value
        for key, value in manifest_entries.items()
        if value["samples"] or value.get("hydrogenated_samples")
    }
    if manifest_entries:
        manifest_path = output_dir / "manifest.json"
        save_generation_manifest(manifest_entries, manifest_path)
        results["manifest_path"] = manifest_path

    if trajectory_entries:
        trajectory_manifest_path = output_dir / "trajectory_manifest.json"
        _atomic_write_text(
            trajectory_manifest_path, json.dumps(trajectory_entries, indent=2)
        )
        results["trajectory_manifest_path"] = trajectory_manifest_path

    summary_path = _write_generation_summary(
        config,
        len(targets),
        num_generated,
        num_trajectory_frames,
        output_dir,
        flow_config=dict(metadata.get("flow_config") or {}),
        hydrogenation_stats=hydrogenation_stats,
    )
    results["summary_path"] = summary_path
    return results


def _resolve_num_steps(num_steps: Optional[int], model) -> int:
    if num_steps is not None:
        return int(num_steps)

    flow_config = getattr(model, "_flow_config", {})
    sampler_cfg = flow_config.get("sampler", {}) if isinstance(flow_config, dict) else {}
    if isinstance(sampler_cfg, dict) and sampler_cfg.get("num_steps") is not None:
        return int(sampler_cfg["num_steps"])

    return 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate crystal structures using a SALA Rectified Flow Matching model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--ema-profile-std",
        type=float,
        default=None,
        help="Power-function EMA relative std (default: checkpoint's first profile)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "folder", "eval"],
        help="Generation mode",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Single input path for mode=single or dataset root fallback for mode=eval",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing multiple .pt/.cif targets for mode=folder",
    )
    parser.add_argument(
        "--input-index",
        type=str,
        default=None,
        help="PT dataset index path for mode=eval",
    )
    parser.add_argument(
        "--pt-root", type=str, default=None, help="PT dataset root for mode=eval"
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["auto", "cif", "pt"],
        help="Input format filter for single/folder modes",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output directory"
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of ODE solver steps (default: use checkpoint config)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=20, help="Number of samples per target"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for Python, NumPy, Torch, and CUDA sampling RNGs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of targets per generation batch",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=1,
        help="Number of samples per target generated in one sampler call",
    )
    parser.add_argument(
        "--split", type=str, default="val", help="Dataset split for mode=eval"
    )
    parser.add_argument(
        "--c-count-min", type=int, default=None, help="Minimum carbon count (eval mode)"
    )
    parser.add_argument(
        "--c-count-max", type=int, default=None, help="Maximum carbon count (eval mode)"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class filters (eval mode)",
    )
    parser.add_argument(
        "--spacegroups",
        type=str,
        default=None,
        help="Comma-separated spacegroup filters (eval mode)",
    )
    parser.add_argument(
        "--is-neutral",
        type=_parse_optional_bool,
        default=None,
        help="Filter by neutral/non-neutral targets in eval mode",
    )
    parser.add_argument(
        "--has-metal",
        type=_parse_optional_bool,
        default=None,
        help="Filter by metal-containing targets in eval mode",
    )
    parser.add_argument(
        "--n-heavy-atoms-min",
        type=int,
        default=None,
        help="Minimum heavy atom count (eval mode)",
    )
    parser.add_argument(
        "--n-heavy-atoms-max",
        type=int,
        default=None,
        help="Maximum heavy atom count (eval mode)",
    )
    parser.add_argument(
        "--n-components-min",
        type=int,
        default=None,
        help="Minimum number of components (eval mode)",
    )
    parser.add_argument(
        "--n-components-max",
        type=int,
        default=None,
        help="Maximum number of components (eval mode)",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Maximum number of filtered eval targets to generate",
    )
    parser.add_argument(
        "--target-selection",
        type=str,
        default="first",
        choices=["first", "random"],
        help="How to choose a subset when max-targets is set",
    )
    parser.add_argument(
        "--target-selection-seed",
        type=int,
        default=42,
        help="Seed for target-selection=random",
    )
    parser.add_argument(
        "--shard-cache-size",
        type=int,
        default=1,
        help="Number of PT shards to cache (eval mode)",
    )
    parser.add_argument(
        "--shard-suffix",
        type=str,
        default=DEFAULT_SHARD_SUFFIX,
        help="Suffix for PT shard filenames in eval mode",
    )
    parser.add_argument(
        "--no-filter-h",
        action="store_true",
        help="Do not remove hydrogens when reading CIF inputs/templates",
    )

    include_unknown_group = parser.add_mutually_exclusive_group()
    include_unknown_group.add_argument(
        "--include-unknown",
        dest="include_unknown",
        action="store_true",
        help="Include unknown class entries in eval mode",
    )
    include_unknown_group.add_argument(
        "--exclude-unknown",
        dest="include_unknown",
        action="store_false",
        help="Exclude unknown class entries in eval mode",
    )
    parser.set_defaults(include_unknown=None)

    parser.add_argument(
        "--device", type=str, default=_default_device(), help="Device to run on"
    )
    parser.add_argument(
        "--no-cif", action="store_true", help="Don't save generated CIF files"
    )
    parser.add_argument(
        "--no-pt", action="store_true", help="Don't save combined .pt file"
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Don't write reference CIF files when a self-contained output is not needed",
    )
    parser.add_argument(
        "--save-trajectory-cif",
        action="store_true",
        help="Save intermediate CIF frames during denoising",
    )
    parser.add_argument(
        "--trajectory-interval",
        type=int,
        default=100,
        help="Save one trajectory CIF frame every N sampling steps",
    )
    parser.add_argument(
        "--hydrogenation-enabled",
        "--add-hydrogens",
        dest="hydrogenation_enabled",
        action="store_true",
        help="Write formula-constrained hydrogenated CIF sidecars",
    )
    parser.add_argument(
        "--hydrogenation-failure-policy",
        type=str,
        default="strict",
        choices=["strict", "keep_heavy"],
        help="Abort or keep only the heavy-atom sample when hydrogenation fails",
    )
    parser.add_argument(
        "--hydrogenation-output-subdir",
        type=str,
        default="hydrogenated",
        help="Output subdirectory for hydrogenated CIF sidecars",
    )
    return parser.parse_args()


def _should_route_to_hydra(argv: Sequence[str]) -> bool:
    """Route to Hydra only for true key=value overrides, not path arguments."""
    if not HYDRA_AVAILABLE:
        return False
    path_flags = {
        "-c",
        "--checkpoint",
        "-i",
        "--input",
        "-o",
        "--output",
        "--input-dir",
        "--input-index",
        "--pt-root",
        "--hydrogenation-output-subdir",
    }
    skip_value = False
    for arg in argv[1:]:
        if skip_value:
            skip_value = False
            continue
        if arg in path_flags:
            skip_value = True
            continue
        if "=" not in arg:
            continue
        if arg.startswith("--") and re.match(r"^--[a-zA-Z0-9_-]+=", arg):
            continue
        key, _, value = arg.partition("=")
        if not value:
            continue
        key_body = key[1:] if key.startswith("+") else key
        if key_body and re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", key_body):
            return True
    return False


def run_generation(config: GenerationConfig) -> Dict[str, Any]:
    _validate_generation_config(config)
    logger.info("Resolved generation config: %s", config)
    model, metadata = load_checkpoint(
        config.checkpoint_path,
        device=config.device,
        ema_profile_std=config.ema_profile_std,
    )
    targets = load_generation_targets(config)
    if not targets:
        raise ValueError(
            "No generation targets resolved from the provided configuration"
        )
    _validate_unique_target_ids(targets)
    sampling_seed = _seed_generation_rngs(config.seed)
    logger.info("Sampling seed: %d", sampling_seed)
    results = generate_for_targets(model, targets, config, metadata)
    logger.info(
        "Generation complete. Generated %d structures.", results["num_generated"]
    )
    return results


def main_argparse() -> int:
    args = parse_args()
    try:
        config = _normalize_generation_config_from_args(args)
        run_generation(config)
        return 0
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if HYDRA_AVAILABLE:

    @hydra.main(version_base=None, config_path="../../conf", config_name="config")
    def main_hydra(cfg: DictConfig) -> None:
        logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
        config = _normalize_generation_config_from_hydra(cfg)
        run_generation(config)


def main() -> None:
    if _should_route_to_hydra(sys.argv):
        main_hydra()
    else:
        sys.exit(main_argparse())


if __name__ == "__main__":
    main()
