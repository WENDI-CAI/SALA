
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor


SPACE_GROUP_HM = (
    "P 1", "P -1", "P 2", "P 21", "C 2", "P m", "P c", "C m", "C c",
    "P 2/m", "P 21/m", "C 2/m", "P 2/c", "P 21/c", "C 2/c",
    "P 2 2 2", "P 2 2 21", "P 21 21 2", "P 21 21 21", "C 2 2 21", "C 2 2 2",
    "F 2 2 2", "I 2 2 2", "I 21 21 21",
    "P m m 2", "P m c 21", "P c c 2", "P m a 2", "P c a 21", "P n c 2",
    "P m n 21", "P b a 2", "P n a 21", "P n n 2", "C m m 2", "C m c 21",
    "C c c 2", "A m m 2", "A e m 2", "A m a 2", "A e a 2", "F m m 2",
    "F d d 2", "I m m 2", "I b a 2", "I m a 2",
    "P m m m", "P n n n", "P c c m", "P b a n", "P m m a", "P n n a",
    "P m n a", "P c c a", "P b a m", "P c c n", "P b c m", "P n n m",
    "P m m n", "P b c n", "P b c a", "P n m a", "C m c m", "C m c e",
    "C m m m", "C c c m", "C m m e", "C c c e", "F m m m", "F d d d",
    "I m m m", "I b a m", "I b c a", "I m m a",
    "P 4", "P 41", "P 42", "P 43", "I 4", "I 41", "P -4", "I -4",
    "P 4/m", "P 42/m", "P 4/n", "P 42/n", "I 4/m", "I 41/a",
    "P 4 2 2", "P 4 21 2", "P 41 2 2", "P 41 21 2", "P 42 2 2", "P 42 21 2",
    "P 43 2 2", "P 43 21 2", "I 4 2 2", "I 41 2 2",
    "P 4 m m", "P 4 b m", "P 42 c m", "P 42 n m", "P 4 c c", "P 4 n c",
    "P 42 m c", "P 42 b c", "I 4 m m", "I 4 c m", "I 41 m d", "I 41 c d",
    "P -4 2 m", "P -4 2 c", "P -4 21 m", "P -4 21 c", "P -4 m 2", "P -4 c 2",
    "P -4 b 2", "P -4 n 2", "I -4 m 2", "I -4 c 2", "I -4 2 m", "I -4 2 d",
    "P 4/m m m", "P 4/m c c", "P 4/n b m", "P 4/n n c", "P 4/m b m", "P 4/m n c",
    "P 4/n m m", "P 4/n c c", "P 42/m m c", "P 42/m c m", "P 42/n b c", "P 42/n n m",
    "P 42/m b c", "P 42/m n m", "P 42/n m c", "P 42/n c m",
    "I 4/m m m", "I 4/m c m", "I 41/a m d", "I 41/a c d",
    "P 3", "P 31", "P 32", "R 3", "P -3", "R -3",
    "P 3 1 2", "P 3 2 1", "P 31 1 2", "P 31 2 1", "P 32 1 2", "P 32 2 1", "R 3 2",
    "P 3 m 1", "P 3 1 m", "P 3 c 1", "P 3 1 c", "R 3 m", "R 3 c",
    "P -3 1 m", "P -3 1 c", "P -3 m 1", "P -3 c 1", "R -3 m", "R -3 c",
    "P 6", "P 61", "P 65", "P 62", "P 64", "P 63", "P -6", "P 6/m", "P 63/m",
    "P 6 2 2", "P 61 2 2", "P 65 2 2", "P 62 2 2", "P 64 2 2", "P 63 2 2",
    "P 6 m m", "P 6 c c", "P 63 c m", "P 63 m c",
    "P -6 m 2", "P -6 c 2", "P -6 2 m", "P -6 2 c",
    "P 6/m m m", "P 6/m c c", "P 63/m c m", "P 63/m m c",
    "P 2 3", "F 2 3", "I 2 3", "P 21 3", "I 21 3",
    "P m -3", "P n -3", "F m -3", "F d -3", "I m -3", "P a -3", "I a -3",
    "P 4 3 2", "P 42 3 2", "F 4 3 2", "F 41 3 2", "I 4 3 2", "P 43 3 2",
    "P 41 3 2", "I 41 3 2", "P -4 3 m", "F -4 3 m", "I -4 3 m", "P -4 3 n",
    "F -4 3 c", "I -4 3 d",
    "P m -3 m", "P n -3 n", "P m -3 n", "P n -3 m", "F m -3 m", "F m -3 c",
    "F d -3 m", "F d -3 c", "I m -3 m", "I a -3 d",
)


def spacegroup_number_to_hm(sg_num: int) -> str:
    try:
        n = int(sg_num)
        if 1 <= n <= 230:
            return SPACE_GROUP_HM[n - 1]
    except (TypeError, ValueError):
        pass
    return "P 1"


@lru_cache(maxsize=230)
def get_sym_ops(spacegroup: int, tol: float = 1e-6) -> np.ndarray:
    del tol
    try:
        from pymatgen.symmetry.groups import SpaceGroup
    except Exception as exc:
        raise ImportError("pymatgen is required for symmetry operations") from exc

    if hasattr(SpaceGroup, "from_int_number"):
        sg = SpaceGroup.from_int_number(int(spacegroup))
    else:
        sg = SpaceGroup(int(spacegroup))

    ops = sg.symmetry_ops
    mats = np.stack([op.affine_matrix for op in ops], axis=0).astype(np.float32)
    return mats


@lru_cache(maxsize=530)
def get_sym_ops_from_hall(hall_number: int) -> np.ndarray:
    """Return the exact affine operation set for one Hall setting.

    International space-group numbers do not identify the unique-axis,
    origin, or rhombohedral/hexagonal setting.  Hall numbers do, so all
    full-cell preprocessing and serialization paths use this function when
    authoritative Hall metadata is available.
    """

    hall = int(hall_number)
    if not 1 <= hall <= 530:
        raise ValueError(f"Hall number must be in [1, 530], got {hall}")
    try:
        import spglib
    except ImportError as exc:
        raise ImportError("spglib is required for Hall-setting symmetry") from exc
    symmetry = spglib.get_symmetry_from_database(hall)
    if symmetry is None:
        raise ValueError(f"spglib has no symmetry operations for Hall number {hall}")
    rotations = np.asarray(symmetry["rotations"], dtype=np.float32)
    translations = np.asarray(symmetry["translations"], dtype=np.float32)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"Invalid rotations returned for Hall number {hall}")
    if translations.shape != (rotations.shape[0], 3):
        raise ValueError(f"Invalid translations returned for Hall number {hall}")
    operations = np.zeros((rotations.shape[0], 4, 4), dtype=np.float32)
    operations[:, :3, :3] = rotations
    operations[:, :3, 3] = translations
    operations[:, 3, 3] = 1.0
    return operations


def _hall_setting_signature(
    operations: np.ndarray | Tensor, *, decimals: int = 6
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...]]:
    """Coordinate-setting signature invariant to operation order/origin shift.

    An origin shift changes translations of non-identity rotations but cannot
    change the rotation multiset or the pure-centering translations.  Those
    two pieces distinguish the Hall basis/unique-axis setting while accepting
    legitimate source-CIF origin choices.
    """

    if torch.is_tensor(operations):
        value = operations.detach().cpu().numpy()
    else:
        value = np.asarray(operations)
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (4, 4) or not len(value):
        raise ValueError("symmetry operations must have shape [G, 4, 4]")
    rotations = np.rint(value[:, :3, :3]).astype(np.int64)
    if not np.allclose(value[:, :3, :3], rotations, atol=10.0 ** (-decimals)):
        raise ValueError("symmetry rotations must be integral")
    rotation_signature = tuple(
        sorted(tuple(int(component) for component in rotation.reshape(-1)) for rotation in rotations)
    )
    identity = np.all(rotations == np.eye(3, dtype=np.int64), axis=(1, 2))
    translations = np.mod(value[identity, :3, 3], 1.0)
    translations[np.isclose(translations, 1.0, atol=10.0 ** (-decimals))] = 0.0
    centering_signature = tuple(
        sorted(
            tuple(float(component) for component in np.round(translation, decimals))
            for translation in translations
        )
    )
    return rotation_signature, centering_signature


def symmetry_ops_match_hall(
    operations: np.ndarray | Tensor, hall_number: int
) -> bool:
    """Whether operations use the basis/axis setting declared by a Hall id."""

    return _hall_setting_signature(operations) == _hall_setting_signature(
        get_sym_ops_from_hall(int(hall_number))
    )


@lru_cache(maxsize=1)
def _hall_setting_index() -> dict[
    tuple[int, tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...]]],
    tuple[int, ...],
]:
    try:
        import spglib
    except ImportError as exc:
        raise ImportError("spglib is required for Hall-setting symmetry") from exc
    mutable: dict[tuple[int, tuple], list[int]] = {}
    for hall in range(1, 531):
        sg_type = spglib.get_spacegroup_type(hall)
        spacegroup = int(
            sg_type.number if hasattr(sg_type, "number") else sg_type["number"]
        )
        key = (spacegroup, _hall_setting_signature(get_sym_ops_from_hall(hall)))
        mutable.setdefault(key, []).append(hall)
    return {key: tuple(values) for key, values in mutable.items()}


def infer_hall_number_from_symmetry_setting(
    operations: np.ndarray | Tensor, *, spacegroup: int | None = None
) -> int:
    """Infer Hall only when the current-basis setting signature is unique."""

    resolved_spacegroup = (
        spacegroup_number_from_symmetry_ops(operations)
        if spacegroup is None
        else int(spacegroup)
    )
    key = (resolved_spacegroup, _hall_setting_signature(operations))
    candidates = _hall_setting_index().get(key, ())
    if len(candidates) != 1:
        raise ValueError(
            "Symmetry operations do not uniquely identify an authoritative Hall "
            f"setting in IT spacegroup {resolved_spacegroup}; declare a Hall symbol/number"
        )
    return int(candidates[0])


def spacegroup_number_from_symmetry_ops(operations: np.ndarray | Tensor) -> int:
    """Infer only the IT number; unlike a Hall id this is canonicalizable."""

    if torch.is_tensor(operations):
        value = operations.detach().cpu().numpy()
    else:
        value = np.asarray(operations)
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (4, 4) or not len(value):
        raise ValueError("symmetry operations must have shape [G, 4, 4]")
    rotations = np.rint(value[:, :3, :3]).astype(np.intc)
    translations = value[:, :3, 3]
    try:
        import spglib

        resolver = getattr(spglib, "get_spacegroup_type_from_symmetry", None)
        if resolver is not None:
            sg_type = resolver(rotations, translations, symprec=1.0e-5)
            if sg_type is not None:
                return int(
                    sg_type.number if hasattr(sg_type, "number") else sg_type["number"]
                )
        hall = spglib.get_hall_number_from_symmetry(
            rotations, translations, symprec=1.0e-5
        )
        if hall is not None:
            sg_type = spglib.get_spacegroup_type(int(hall))
            return int(
                sg_type.number if hasattr(sg_type, "number") else sg_type["number"]
            )
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Could not infer spacegroup from symmetry operations") from exc
    raise ValueError("Could not infer spacegroup from symmetry operations")


def is_identity_op(sym_op: np.ndarray, tol: float = 1e-6) -> bool:
    rot = sym_op[:3, :3]
    trans = sym_op[:3, 3]
    return np.allclose(rot, np.eye(3), atol=tol) and np.allclose(trans, 0.0, atol=tol)


def _validate_affine_operations(operations: Tensor, *, name: str) -> Tensor:
    if operations.ndim != 3 or operations.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [G, 4, 4], got {tuple(operations.shape)}")
    if operations.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one operation")
    if not bool(torch.isfinite(operations).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    expected_bottom = operations.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(
        operations.shape[0], -1
    )
    if not bool(torch.allclose(operations[:, 3], expected_bottom, atol=1e-6, rtol=0.0)):
        raise ValueError(f"{name} must contain homogeneous affine matrices")
    return operations


def canonicalize_affine_symmetry_ops(operations: Tensor) -> Tensor:
    """Canonicalize fractional translations to [0, 1) without changing R.

    CIF expressions may encode the same operation with translations differing
    by arbitrary lattice integers.  Full-cell Cartesian tokens must not depend
    on that textual choice, so every runtime operation crosses this boundary.
    """

    value = _validate_affine_operations(
        torch.as_tensor(operations), name="symmetry_ops"
    ).clone()
    if not value.is_floating_point():
        value = value.to(dtype=torch.float32)
    translation = torch.remainder(value[:, :3, 3], 1.0)
    tolerance = 8.0 * torch.finfo(translation.dtype).eps
    translation = torch.where(
        torch.isclose(translation, torch.ones_like(translation), atol=tolerance, rtol=0.0)
        | torch.isclose(translation, torch.zeros_like(translation), atol=tolerance, rtol=0.0),
        torch.zeros_like(translation),
        translation,
    )
    value[:, :3, 3] = translation
    return value


def extract_explicit_symmetry_ops(
    structure: Any,
    graph_idx: int | None = None,
) -> Tensor | None:
    """Return one graph's unpadded source-CIF affine operations.

    ``AsuInput`` stores ``[G, 4, 4]`` directly. Canonical ``AsuBatch`` stores
    padded ``[B, G_max, 4, 4]`` together with an authoritative
    ``[B, G_max]`` mask.
    """

    stored = getattr(structure, "symmetry_ops", None)
    if stored is None:
        return None
    operations = torch.as_tensor(stored)
    if operations.numel() == 0:
        return None

    if operations.ndim == 3:
        if hasattr(structure, "ptr"):
            raise ValueError(
                "AsuBatch symmetry_ops must be padded [B, G_max, 4, 4] with a mask"
            )
        return canonicalize_affine_symmetry_ops(operations)

    if operations.ndim != 4 or operations.shape[2:] != (4, 4):
        raise ValueError(
            "batched symmetry_ops must have shape [B, G_max, 4, 4], "
            f"got {tuple(operations.shape)}"
        )
    if graph_idx is None:
        if operations.shape[0] != 1:
            raise ValueError("graph_idx is required for a multi-graph symmetry batch")
        graph_idx = 0
    graph_idx = int(graph_idx)
    if graph_idx < 0 or graph_idx >= operations.shape[0]:
        raise IndexError(f"graph_idx {graph_idx} is outside [0, {operations.shape[0]})")

    stored_mask = getattr(structure, "symmetry_mask", None)
    if stored_mask is None:
        raise ValueError("batched symmetry_ops require an authoritative symmetry_mask")
    mask = torch.as_tensor(stored_mask, device=operations.device)
    if mask.shape != operations.shape[:2]:
        raise ValueError(
            "symmetry_mask must have shape [B, G_max] matching symmetry_ops, "
            f"got {tuple(mask.shape)} versus {tuple(operations.shape[:2])}"
        )
    active = operations[graph_idx, mask[graph_idx].bool()]
    if active.shape[0] == 0:
        return None
    return canonicalize_affine_symmetry_ops(active)


@dataclass(frozen=True)
class SymOp:
    R: np.ndarray
    t: np.ndarray


IDENTITY_SYM_OP = SymOp(R=np.eye(3, dtype=float), t=np.zeros(3, dtype=float))


def parse_symop(op: str) -> SymOp:
    parts = [part.strip().lower() for part in op.strip().strip("'\"").split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid symmetry op: {op}")

    rows = []
    offsets = []
    for part in parts:
        coeffs, offset = _parse_axis_expr(part)
        rows.append([coeffs["x"], coeffs["y"], coeffs["z"]])
        offsets.append(offset)

    return SymOp(R=np.array(rows, dtype=float), t=np.array(offsets, dtype=float))


def parse_symops(ops: Iterable[str]) -> list[SymOp]:
    parsed = [parse_symop(op) for op in ops]
    return parsed if parsed else [IDENTITY_SYM_OP]


def apply_symop(symop: SymOp, frac_coords: np.ndarray) -> np.ndarray:
    return frac_coords @ symop.R.T + symop.t


def _parse_axis_expr(expr: str) -> tuple[dict[str, float], float]:
    expr = expr.replace(" ", "")
    expr = expr.replace("-", "+-")
    terms = [term for term in expr.split("+") if term]

    coeffs = {"x": 0.0, "y": 0.0, "z": 0.0}
    offset = 0.0
    for term in terms:
        if term in ("x", "y", "z"):
            coeffs[term] += 1.0
            continue
        if term in ("-x", "-y", "-z"):
            coeffs[term[1:]] -= 1.0
            continue
        if term.endswith(("x", "y", "z")):
            axis = term[-1]
            factor = term[:-1]
            if factor in ("", "+"):
                coeffs[axis] += 1.0
            elif factor == "-":
                coeffs[axis] -= 1.0
            else:
                coeffs[axis] += _parse_fraction(factor)
            continue
        offset += _parse_fraction(term)

    return coeffs, offset


def _parse_fraction(token: str) -> float:
    token = token.strip()
    if "/" in token:
        numerator, denominator = token.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(token)


def compute_cell_image_offset_torch(fracs: Tensor) -> Tensor:
    if fracs.numel() == 0:
        return torch.zeros(3, device=fracs.device, dtype=fracs.dtype)

    # An atom is inside [0, 1)^3 after subtracting integer offset m exactly
    # when floor(frac_i) == m. The most-populated integer cell therefore gives
    # the max-atoms-in-cell representative without a Python search or GPU sync.
    cell_index = torch.floor(fracs).to(torch.long)
    unique, counts = torch.unique(cell_index, dim=0, sorted=True, return_counts=True)
    candidates = unique[counts == counts.max()]
    norms = candidates.square().sum(dim=-1)
    return candidates[torch.argmin(norms)].to(device=fracs.device, dtype=fracs.dtype)


__all__ = [
    "IDENTITY_SYM_OP",
    "SPACE_GROUP_HM",
    "SymOp",
    "apply_symop",
    "compute_cell_image_offset_torch",
    "extract_explicit_symmetry_ops",
    "get_sym_ops",
    "is_identity_op",
    "parse_symop",
    "parse_symops",
    "spacegroup_number_to_hm",
]
