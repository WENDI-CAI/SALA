from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from sala.core.symmetry import parse_symops
from sala.data.io.cif_reader import read_cif


@dataclass(frozen=True)
class SymmetryContext:
    operations: Tensor
    source: str
    operation_hash: str
    spacegroup: int


def _canonical_operation(operation: Tensor, *, decimals: int = 6) -> tuple[float, ...]:
    value = torch.as_tensor(operation, dtype=torch.float64).clone()
    value[:3, :3] = torch.round(value[:3, :3])
    translation = torch.remainder(value[:3, 3], 1.0)
    translation = torch.where(
        torch.isclose(translation, torch.ones_like(translation), atol=1e-7),
        0.0,
        translation,
    )
    value[:3, 3] = translation
    value[3] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], dtype=value.dtype, device=value.device
    )
    scale = float(10**decimals)
    value = torch.round(value * scale) / scale
    return tuple(float(item) for item in value.reshape(-1).tolist())


def _operation_hash(operations: Tensor) -> str:
    canonical = sorted(_canonical_operation(operation) for operation in operations)
    encoded = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contains_operation(operations: Tensor, candidate: Tensor) -> bool:
    target = _canonical_operation(candidate)
    return any(_canonical_operation(operation) == target for operation in operations)


def _validate_group(operations: Tensor) -> None:
    if (
        operations.ndim != 3
        or operations.shape[1:] != (4, 4)
        or operations.shape[0] == 0
    ):
        raise ValueError("symmetry operations must have shape [G, 4, 4]")
    if not torch.isfinite(operations).all().item():
        raise ValueError("symmetry operations must be finite")
    canonical_operations = {_canonical_operation(operation) for operation in operations}
    identity = torch.eye(4, dtype=operations.dtype, device=operations.device)
    if _canonical_operation(identity) not in canonical_operations:
        raise ValueError(
            "symmetry operation subset must contain identity and form a closed subgroup"
        )
    for operation in operations:
        inverse = torch.linalg.inv(operation.to(torch.float64)).to(operation.dtype)
        if _canonical_operation(inverse) not in canonical_operations:
            raise ValueError(
                "symmetry operation subset must contain inverses and form a closed subgroup"
            )
    for left in operations:
        for right in operations:
            if _canonical_operation(left @ right) not in canonical_operations:
                raise ValueError("symmetry operation subset must be a closed subgroup")


def _from_xyz(values: tuple[str, ...], *, device: torch.device) -> Tensor | None:
    if not values:
        return None
    matrices: list[Tensor] = []
    for operation in parse_symops(values):
        matrix = torch.eye(4, dtype=torch.float32, device=device)
        matrix[:3, :3] = torch.as_tensor(
            operation.R, dtype=torch.float32, device=device
        )
        matrix[:3, 3] = torch.as_tensor(operation.t, dtype=torch.float32, device=device)
        matrices.append(matrix)
    return torch.stack(matrices) if matrices else None


def _candidate_cif_paths(
    asu,
    meta: Mapping[str, Any],
    source_cif_root: str | Path | None,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for key in ("source_cif_path", "original_cif_path", "reference_cif_path"):
        value = meta.get(key)
        if value:
            paths.append(Path(str(value)).expanduser())
    target_id = str(
        meta.get("source_target_id")
        or meta.get("target_id")
        or getattr(asu, "cif_id", "")
    )
    if source_cif_root and target_id:
        paths.append(Path(source_cif_root).expanduser() / f"{target_id}.cif")
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return tuple(unique)


def _resolve_operations(
    asu,
    *,
    meta: Mapping[str, Any],
    source_cif_root: str | Path | None,
) -> tuple[Tensor, str]:
    device = asu.frac_coords_gt.device
    stored = getattr(asu, "symmetry_ops", None)
    if stored is not None and torch.as_tensor(stored).numel():
        return torch.as_tensor(
            stored, dtype=torch.float32, device=device
        ), "pt_affine_ops"
    for path in _candidate_cif_paths(asu, meta, source_cif_root):
        if not path.is_file():
            continue
        cif_data = read_cif(path)
        operations = _from_xyz(cif_data.symop_xyz, device=device)
        if operations is not None:
            return operations, "source_cif"
    raise ValueError(
        "symmetry evaluation requires explicit PT operations or a source CIF"
    )


def resolve_symmetry_context(
    asu,
    *,
    meta: Mapping[str, Any],
    source_cif_root: str | Path | None,
    max_symops: int | None,
    override: SymmetryContext | None = None,
) -> SymmetryContext:
    spacegroup = int(torch.as_tensor(asu.spacegroup).view(-1)[0].item())
    if override is not None:
        if int(override.spacegroup) != spacegroup:
            raise ValueError(
                "Reference/generated space groups differ for symmetry override"
            )
        operations = override.operations.to(device=asu.frac_coords_gt.device)
        source = f"reference_override:{override.source}"
    else:
        operations, source = _resolve_operations(
            asu,
            meta=meta,
            source_cif_root=source_cif_root,
        )
    if max_symops is not None:
        requested = int(max_symops)
        if requested < 1:
            raise ValueError("max_symops must be positive")
        operations = operations[:requested]
    _validate_group(operations)
    expected_count_tensor = getattr(asu, "n_symops", None)
    if max_symops is None and expected_count_tensor is not None:
        expected_count = int(torch.as_tensor(expected_count_tensor).view(-1)[0].item())
        if expected_count > 0 and expected_count != int(operations.shape[0]):
            raise ValueError(
                f"Symmetry operation count mismatch: PT declares {expected_count}, resolved {operations.shape[0]}"
            )
    return SymmetryContext(
        operations=operations,
        source=source,
        operation_hash=_operation_hash(operations),
        spacegroup=spacegroup,
    )


__all__ = ["SymmetryContext", "resolve_symmetry_context"]
