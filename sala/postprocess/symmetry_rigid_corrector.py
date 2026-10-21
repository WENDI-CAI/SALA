from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import torch
from torch import Tensor

from sala.core.lattice import (
    cart_to_frac_coords,
)
from sala.core.symmetry import (
    compute_cell_image_offset_torch,
    extract_explicit_symmetry_ops,
)
from sala.core.types import AsuBatch, AsuInput
from sala.postprocess.geometry.clash import (
    ClashConfig,
    ClashResult,
    evaluate_clashes,
)
from sala.postprocess.geometry.contact_statistics import element_vdw_radii

_PI_RING_ATOMIC_NUMBERS = {6, 7, 8, 15, 16}

_VDW_TABLE_CACHE: dict[tuple[str, Optional[int], str], Tensor] = {}
_TRANSLATION_GRID_CACHE: dict[tuple[tuple[int, ...], str, Optional[int], str], Tensor] = {}


def _vdw_table(device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (device.type, device.index, str(dtype))
    cached = _VDW_TABLE_CACHE.get(key)
    if cached is not None and cached.device == device and cached.dtype == dtype:
        return cached
    radius_values, _ = element_vdw_radii()
    values = torch.tensor(radius_values, device=device, dtype=dtype)
    _VDW_TABLE_CACHE[key] = values
    return values


def normalize_symmetry_corrector_config(cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    raw = dict(cfg or {})
    removed = sorted(
        set(raw)
        & {
            "num_steps",
            "anchor_weight",
            "translation_lr",
            "step_size",
            "volume_relaxation_enabled",
            "max_volume_rel_change",
            "volume_search_steps",
            "volume_pose_projection_steps",
        }
    )
    if removed:
        raise ValueError(
            "Removed symmetry-corrector configuration keys: " + ", ".join(removed)
        )
    allowed = {
        "postprocess_enabled", "postprocess_steps",
        "postprocess_projection_steps", "postprocess_refinement_steps",
        "translations", "max_symops", "heavy_only", "clash_supercell_radius",
        "clash_threshold_mode", "clash_distance_threshold_A",
        "clash_vdw_offset_A", "canonical_dedup_tolerance_A",
        "self_image_overlap_tolerance_A", "pi_pi_enabled", "pi_pi_weight",
        "pi_pi_max_rings", "pi_pi_center_min_A", "pi_pi_center_max_A",
        "pi_pi_center_low_A", "pi_pi_center_high_A", "pi_pi_vertical_min_A",
        "pi_pi_vertical_max_A", "pi_pi_vertical_low_A", "pi_pi_vertical_high_A",
        "pi_pi_slip_max_A", "pi_pi_slip_low_A", "pi_pi_slip_high_A",
        "pi_pi_angle_max_deg", "pi_pi_angle_sigma_deg", "pi_pi_vertical_weight",
        "pi_pi_slip_weight", "pi_pi_angle_weight", "pi_pi_center_weight",
        "clash_weight", "hard_tail_weight", "translation_weight",
        "rotation_weight", "translation_step_size_A", "rotation_step_size",
        "line_search_steps", "enable_rotation", "temperature",
        "max_component_pairs", "max_atom_pairs", "broadphase_extra_A",
    }
    unsupported = sorted(set(raw) - allowed)
    if unsupported:
        raise ValueError(
            "Unsupported symmetry-corrector configuration keys: "
            + ", ".join(unsupported)
        )
    clash_defaults = ClashConfig()
    translations = raw.get("translations", [-1, 0, 1])
    if isinstance(translations, str):
        translations = [int(x.strip()) for x in translations.split(",") if x.strip()]
    translations = [int(x) for x in translations]
    if not translations:
        translations = [-1, 0, 1]

    def _as_bool(name: str, default: bool) -> bool:
        value = raw.get(name, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _as_optional_int(name: str) -> Optional[int]:
        value = raw.get(name)
        return None if value is None else int(value)

    postprocess_steps = max(1, int(raw.get("postprocess_steps", 12)))
    postprocess_projection_steps = max(0, int(raw.get("postprocess_projection_steps", 2)))
    postprocess_refinement_steps = max(
        0,
        int(raw.get("postprocess_refinement_steps", max(1, postprocess_steps - postprocess_projection_steps))),
    )

    clash_threshold_mode = str(
        raw.get("clash_threshold_mode", clash_defaults.threshold_mode)
    ).lower()
    clash_distance_threshold_A = float(
        raw.get("clash_distance_threshold_A", clash_defaults.distance_threshold_A)
    )
    clash_vdw_offset_A = float(
        raw.get("clash_vdw_offset_A", clash_defaults.vdw_offset_A)
    )
    ClashConfig(
        threshold_mode=clash_threshold_mode,
        distance_threshold_A=clash_distance_threshold_A,
        vdw_offset_A=clash_vdw_offset_A,
    )

    return {
        "postprocess_enabled": _as_bool("postprocess_enabled", False),
        "postprocess_steps": postprocess_steps,
        "postprocess_projection_steps": postprocess_projection_steps,
        "postprocess_refinement_steps": postprocess_refinement_steps,
        "translations": translations,
        "max_symops": _as_optional_int("max_symops"),
        "heavy_only": _as_bool("heavy_only", True),
        "clash_supercell_radius": max(
            0,
            int(raw.get("clash_supercell_radius", clash_defaults.supercell_radius)),
        ),
        "clash_threshold_mode": clash_threshold_mode,
        "clash_distance_threshold_A": clash_distance_threshold_A,
        "clash_vdw_offset_A": clash_vdw_offset_A,
        "canonical_dedup_tolerance_A": float(
            raw.get("canonical_dedup_tolerance_A", clash_defaults.dedup_tolerance_A)
        ),
        "self_image_overlap_tolerance_A": float(raw.get("self_image_overlap_tolerance_A", 0.20)),
        "pi_pi_enabled": _as_bool("pi_pi_enabled", True),
        "pi_pi_weight": float(raw.get("pi_pi_weight", 0.05)),
        "pi_pi_max_rings": max(0, int(raw.get("pi_pi_max_rings", 32))),
        "pi_pi_center_min_A": float(raw.get("pi_pi_center_min_A", 3.0)),
        "pi_pi_center_max_A": float(raw.get("pi_pi_center_max_A", 5.8)),
        "pi_pi_center_low_A": float(raw.get("pi_pi_center_low_A", 3.4)),
        "pi_pi_center_high_A": float(raw.get("pi_pi_center_high_A", 5.2)),
        "pi_pi_vertical_min_A": float(raw.get("pi_pi_vertical_min_A", 2.6)),
        "pi_pi_vertical_max_A": float(raw.get("pi_pi_vertical_max_A", 4.6)),
        "pi_pi_vertical_low_A": float(raw.get("pi_pi_vertical_low_A", 3.1)),
        "pi_pi_vertical_high_A": float(raw.get("pi_pi_vertical_high_A", 4.3)),
        "pi_pi_slip_max_A": float(raw.get("pi_pi_slip_max_A", 3.6)),
        "pi_pi_slip_low_A": float(raw.get("pi_pi_slip_low_A", 0.8)),
        "pi_pi_slip_high_A": float(raw.get("pi_pi_slip_high_A", 3.4)),
        "pi_pi_angle_max_deg": float(raw.get("pi_pi_angle_max_deg", 45.0)),
        "pi_pi_angle_sigma_deg": float(raw.get("pi_pi_angle_sigma_deg", 25.0)),
        "pi_pi_vertical_weight": float(raw.get("pi_pi_vertical_weight", 1.0)),
        "pi_pi_slip_weight": float(raw.get("pi_pi_slip_weight", 0.5)),
        "pi_pi_angle_weight": float(raw.get("pi_pi_angle_weight", 0.25)),
        "pi_pi_center_weight": float(raw.get("pi_pi_center_weight", 0.10)),
        "clash_weight": float(raw.get("clash_weight", 1.0)),
        "hard_tail_weight": max(0.0, float(raw.get("hard_tail_weight", 0.25))),
        "translation_weight": float(raw.get("translation_weight", 0.02)),
        "rotation_weight": float(raw.get("rotation_weight", 0.002)),
        "translation_step_size_A": max(
            0.0,
            float(raw.get("translation_step_size_A", 0.05)),
        ),
        "rotation_step_size": max(
            0.0,
            float(raw.get("rotation_step_size", 0.02)),
        ),
        "line_search_steps": max(1, int(raw.get("line_search_steps", 6))),
        "enable_rotation": _as_bool("enable_rotation", True),
        "temperature": max(1e-4, float(raw.get("temperature", 0.20))),
        "max_component_pairs": max(1, int(raw.get("max_component_pairs", 256))),
        "max_atom_pairs": max(1, int(raw.get("max_atom_pairs", 8192))),
        "broadphase_extra_A": float(raw.get("broadphase_extra_A", 3.5)),
    }


@dataclass
class _GraphView:
    frac: Tensor
    lattice: Tensor
    atom_types: Tensor
    component_id: Tensor
    spacegroup: int
    edge_index: Optional[Tensor] = None
    hybridization: Optional[Tensor] = None
    symmetry_ops: Optional[Tensor] = None


@dataclass
class SymmetryRigidCorrectionResult:
    frac_coords: Tensor
    lattice: Tensor
    scale_factor: Tensor
    diagnostics: list[dict[str, float | int | bool | str | None]]


@dataclass
class _ClashEvaluation:
    energy: Tensor
    pair_count: int
    image_pair_count: int
    max_penetration_A: float
    identity_self_pair_count: int = 0

    def key(self) -> tuple[int, float, int, float]:
        return (
            int(self.image_pair_count),
            float(self.max_penetration_A),
            int(self.pair_count),
            float(self.energy.detach().item()),
        )


def _graph_view(batch: AsuBatch, frac_coords: Tensor, lattice: Tensor, graph_idx: int) -> _GraphView:
    start = int(batch.ptr[graph_idx].item())
    end = int(batch.ptr[graph_idx + 1].item())
    lat = lattice[graph_idx] if lattice.ndim == 3 else lattice
    edge_index = None
    if batch.edge_index_conv is not None and batch.edge_index_conv.numel() > 0:
        edge_mask = (
            (batch.edge_index_conv[0] >= start)
            & (batch.edge_index_conv[0] < end)
            & (batch.edge_index_conv[1] >= start)
            & (batch.edge_index_conv[1] < end)
        )
        edge_index = batch.edge_index_conv[:, edge_mask] - start
    hybridization = batch.hybridization[start:end] if batch.hybridization is not None else None
    symmetry_ops = extract_explicit_symmetry_ops(batch, graph_idx)
    return _GraphView(
        frac=frac_coords[start:end],
        lattice=lat,
        atom_types=batch.atom_types[start:end],
        component_id=batch.component_id[start:end],
        spacegroup=int(batch.spacegroup[graph_idx].view(-1)[0].item()),
        edge_index=edge_index,
        hybridization=hybridization,
        symmetry_ops=symmetry_ops,
    )


def _graph_asu_input(view: _GraphView, frac: Tensor, lattice: Tensor) -> AsuInput:
    edge_index = view.edge_index
    if edge_index is None:
        edge_index = torch.empty((2, 0), device=frac.device, dtype=torch.long)
    num_components = (
        int(view.component_id.max().item()) + 1
        if view.component_id.numel() > 0
        else 0
    )
    return AsuInput(
        atom_types=view.atom_types.to(device=frac.device),
        spacegroup=torch.tensor([view.spacegroup], device=frac.device, dtype=torch.long),
        component_id=view.component_id.to(device=frac.device, dtype=torch.long),
        atom_formal_charge=torch.zeros(
            frac.shape[0], device=frac.device, dtype=torch.long
        ),
        atom_charge_mask=torch.zeros(
            frac.shape[0], device=frac.device, dtype=torch.bool
        ),
        component_formal_charge=torch.zeros(num_components, device=frac.device, dtype=torch.float32),
        component_type_id=torch.arange(
            num_components, device=frac.device, dtype=torch.long
        ),
        num_atoms=torch.tensor([frac.shape[0]], device=frac.device, dtype=torch.long),
        edge_index_conv=edge_index.to(device=frac.device, dtype=torch.long),
        edge_attr_conv=torch.zeros(edge_index.shape[1], device=frac.device, dtype=torch.float32),
        frac_coords_gt=frac,
        lattice_gt=lattice.to(device=frac.device, dtype=frac.dtype),
        hybridization=(
            view.hybridization.to(device=frac.device)
            if view.hybridization is not None
            else None
        ),
        symmetry_ops=(
            view.symmetry_ops.to(device=frac.device, dtype=frac.dtype)
            if view.symmetry_ops is not None
            else None
        ),
    )


def _final_clash_config(cfg: dict[str, Any]) -> ClashConfig:
    return ClashConfig(
        atom_scope="heavy" if bool(cfg.get("heavy_only", True)) else "all",
        max_symops=cfg.get("max_symops"),
        supercell_radius=int(cfg["clash_supercell_radius"]),
        threshold_mode=str(cfg["clash_threshold_mode"]),
        distance_threshold_A=float(cfg["clash_distance_threshold_A"]),
        vdw_offset_A=float(cfg["clash_vdw_offset_A"]),
        dedup_tolerance_A=float(cfg["canonical_dedup_tolerance_A"]),
    )


def _evaluate_final_clashes(
    view: _GraphView,
    frac: Tensor,
    config: ClashConfig,
    *,
    differentiable: bool,
) -> ClashResult:
    return evaluate_clashes(
        _graph_asu_input(view, frac, view.lattice),
        config=config,
        differentiable=differentiable,
    )


def _ops_for_spacegroup(
    spacegroup: int,
    device: torch.device,
    dtype: torch.dtype,
    max_symops: Optional[int],
    *,
    symmetry_ops: Optional[Tensor] = None,
) -> Tensor:
    del spacegroup
    if symmetry_ops is None:
        raise ValueError("rigid postprocess requires explicit Hall-setting symmetry_ops")
    ops = torch.as_tensor(symmetry_ops, device=device, dtype=dtype)
    if ops.ndim != 3 or ops.shape[1:] != (4, 4) or ops.shape[0] == 0:
        raise ValueError("symmetry_ops must have shape [G, 4, 4]")
    if max_symops is not None and 0 < int(max_symops) < int(ops.shape[0]):
        identity_idx = _identity_op_index(ops)
        keep = [identity_idx] + [
            idx for idx in range(int(ops.shape[0])) if idx != identity_idx
        ]
        index = torch.tensor(keep[: int(max_symops)], device=device)
        ops = ops.index_select(0, index)
    return ops


def _translation_grid(values: list[int], device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (tuple(int(x) for x in values), device.type, device.index, str(dtype))
    cached = _TRANSLATION_GRID_CACHE.get(key)
    if cached is not None and cached.device == device and cached.dtype == dtype:
        return cached
    base = torch.tensor(values, device=device, dtype=dtype)
    mesh = torch.meshgrid(base, base, base, indexing="ij")
    out = torch.stack([x.reshape(-1) for x in mesh], dim=-1)
    _TRANSLATION_GRID_CACHE[key] = out
    return out


def _identity_op_index(ops: Tensor) -> int:
    eye = torch.eye(3, device=ops.device, dtype=ops.dtype)
    score = (ops[:, :3, :3] - eye).abs().sum(dim=(1, 2)) + ops[:, :3, 3].abs().sum(dim=-1)
    return int(torch.argmin(score).item())


def _component_indices(component_id: Tensor) -> list[Tensor]:
    return [
        torch.nonzero(component_id == comp, as_tuple=False).view(-1)
        for comp in torch.unique(component_id.to(torch.long), sorted=True)
    ]


def _canonicalize_component_in_cell(frac: Tensor) -> Tensor:
    if frac.numel() == 0:
        return frac
    return frac - compute_cell_image_offset_torch(frac)


def _canonicalize_components_in_cell(frac: Tensor, components: list[Tensor]) -> Tensor:
    if not components:
        return frac
    out = frac.clone()
    for atom_idx in components:
        out[atom_idx] = _canonicalize_component_in_cell(out[atom_idx])
    return out


def _component_centers_and_radii(frac: Tensor, lattice: Tensor, components: list[Tensor]) -> tuple[Tensor, Tensor]:
    cart = frac @ lattice
    centers = []
    radii = []
    for idx in components:
        c = cart[idx].mean(dim=0)
        centers.append(c)
        radii.append(torch.linalg.norm(cart[idx] - c, dim=-1).max())
    return torch.stack(centers, dim=0), torch.stack(radii, dim=0)


def _pair_lower_bounds(
    atom_types_a: Tensor,
    atom_types_b: Tensor,
    cfg: dict[str, Any],
    *,
    floor_kind: str = "hard",
) -> Tensor:
    del floor_kind
    dtype = torch.float32
    device = atom_types_a.device
    za = atom_types_a.clamp(0, 118).to(torch.long)
    zb = atom_types_b.clamp(0, 118).to(torch.long)
    if str(cfg["clash_threshold_mode"]) == "fixed":
        return torch.full(
            (int(za.numel()), int(zb.numel())),
            float(cfg["clash_distance_threshold_A"]),
            device=device,
            dtype=dtype,
        )
    table = _vdw_table(device, dtype)
    lower = table[za].unsqueeze(1) + table[zb].unsqueeze(0)
    return torch.clamp(lower - float(cfg["clash_vdw_offset_A"]), min=0.0)


@dataclass(frozen=True)
class _RingCycle:
    atoms: Tensor
    component: int


def _canonical_cycle(path: list[int]) -> tuple[int, ...]:
    m = min(path)
    i = path.index(m)
    fwd = tuple(path[i:] + path[:i])
    rev_path = list(reversed(path))
    j = rev_path.index(m)
    rev = tuple(rev_path[j:] + rev_path[:j])
    return min(fwd, rev)


def _ring_cycles(view: _GraphView, cfg: dict[str, Any]) -> list[_RingCycle]:
    max_rings = int(cfg.get("pi_pi_max_rings", 0))
    if max_rings <= 0 or view.edge_index is None or view.edge_index.numel() == 0:
        return []
    atom_types = view.atom_types.to(torch.long)
    component_id = view.component_id.to(torch.long)
    hybridization = view.hybridization.to(torch.long) if view.hybridization is not None else None
    allowed = torch.zeros_like(atom_types, dtype=torch.bool)
    for z in _PI_RING_ATOMIC_NUMBERS:
        allowed |= atom_types == int(z)
    if hybridization is not None:
        # SALA hybridization encoding: sp=2, sp2=3.
        allowed &= hybridization == 3
    adj: dict[int, set[int]] = {}
    for a, b in view.edge_index.t().detach().cpu().to(torch.long).tolist():
        if a == b or not bool(allowed[a]) or not bool(allowed[b]):
            continue
        adj.setdefault(int(a), set()).add(int(b))
        adj.setdefault(int(b), set()).add(int(a))
    cycles: set[tuple[int, ...]] = set()
    for start in sorted(adj):
        stack = [(start, [start])]
        while stack and len(cycles) < max_rings * 4:
            cur, path = stack.pop()
            if len(path) > 6:
                continue
            for nxt in adj.get(cur, ()):
                if nxt == start and 5 <= len(path) <= 6:
                    cycles.add(_canonical_cycle(path))
                elif nxt not in path and nxt >= start:
                    stack.append((nxt, path + [nxt]))
    out: list[_RingCycle] = []
    for cycle in sorted(cycles):
        cyc = torch.tensor(cycle, device=component_id.device, dtype=torch.long)
        comps = component_id[cyc]
        if torch.unique(comps).numel() != 1:
            continue
        out.append(_RingCycle(atoms=cyc.to(device=view.frac.device), component=int(comps[0].item())))
        if len(out) >= max_rings:
            break
    return out


def _ring_center_normal(frac: Tensor, lattice: Tensor, ring: _RingCycle) -> tuple[Tensor, Tensor]:
    pts = frac[ring.atoms] @ lattice
    center = pts.mean(dim=0)
    centered = pts - center
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / torch.linalg.norm(normal).clamp(min=1e-6)
    return center, normal


def _interval_penalty(x: Tensor, low: float, high: float) -> Tensor:
    return torch.relu(x.new_tensor(float(low)) - x).square() + torch.relu(x - x.new_tensor(float(high))).square()


def _graph_pi_energy(
    view: _GraphView,
    frac: Tensor,
    cfg: dict[str, Any],
    *,
    rings: Optional[list[_RingCycle]] = None,
) -> Tensor:
    if not bool(cfg.get("pi_pi_enabled", False)) or float(cfg.get("pi_pi_weight", 0.0)) <= 0.0:
        return frac.new_zeros(())
    rings = _ring_cycles(view, cfg) if rings is None else rings
    if len(rings) < 2:
        return frac.new_zeros(())
    device = frac.device
    dtype = torch.float32
    lattice = view.lattice.to(device=device, dtype=dtype)
    frac = frac.to(device=device, dtype=dtype)
    components = _component_indices(view.component_id.to(device=device))
    frac = _canonicalize_components_in_cell(frac, components)
    ops = _ops_for_spacegroup(
        view.spacegroup,
        device,
        dtype,
        cfg["max_symops"],
        symmetry_ops=view.symmetry_ops,
    )
    translations = _translation_grid(cfg["translations"], device, dtype)
    identity_idx = _identity_op_index(ops)
    zero_trans_mask = translations.abs().sum(dim=-1) == 0
    base = [_ring_center_normal(frac, lattice, ring) for ring in rings]
    losses: list[Tensor] = []
    angle_sigma = math.radians(max(1.0, float(cfg["pi_pi_angle_sigma_deg"])))
    for left_i, (ring_i, (center_i, normal_i)) in enumerate(zip(rings, base)):
        del ring_i
        for right_i, ring_j in enumerate(rings):
            frac_j = frac[ring_j.atoms]
            for op_idx in range(int(ops.shape[0])):
                R = ops[op_idx, :3, :3]
                tr = ops[op_idx, :3, 3]
                frac_j_op = _canonicalize_component_in_cell(frac_j @ R.transpose(0, 1) + tr)
                pts_j_base = frac_j_op @ lattice
                center_j_base = pts_j_base.mean(dim=0)
                centered_j = pts_j_base - center_j_base
                _u, _s, vh = torch.linalg.svd(centered_j, full_matrices=False)
                normal_j = vh[-1]
                normal_j = normal_j / torch.linalg.norm(normal_j).clamp(min=1e-6)
                for trans_idx, trans in enumerate(translations):
                    if left_i == right_i and op_idx == identity_idx and bool(zero_trans_mask[trans_idx].item()):
                        continue
                    center_j = center_j_base + trans @ lattice
                    delta = center_j - center_i
                    center_dist = torch.linalg.norm(delta).clamp(min=1e-6)
                    cosang = torch.abs(torch.sum(normal_i * normal_j)).clamp(max=1.0 - 1e-6)
                    angle = torch.acos(cosang)
                    vertical = torch.abs(torch.sum(delta * normal_i))
                    slip = torch.sqrt(torch.clamp(center_dist.square() - vertical.square(), min=0.0))
                    with torch.no_grad():
                        if float(center_dist.detach()) < float(cfg["pi_pi_center_min_A"]):
                            continue
                        if float(center_dist.detach()) > float(cfg["pi_pi_center_max_A"]):
                            continue
                        if float(vertical.detach()) < float(cfg["pi_pi_vertical_min_A"]):
                            continue
                        if float(vertical.detach()) > float(cfg["pi_pi_vertical_max_A"]):
                            continue
                        if float(slip.detach()) > float(cfg["pi_pi_slip_max_A"]):
                            continue
                        if math.degrees(float(angle.detach())) > float(cfg["pi_pi_angle_max_deg"]):
                            continue
                    loss = (
                        float(cfg["pi_pi_vertical_weight"]) * _interval_penalty(vertical, cfg["pi_pi_vertical_low_A"], cfg["pi_pi_vertical_high_A"])
                        + float(cfg["pi_pi_slip_weight"]) * _interval_penalty(slip, cfg["pi_pi_slip_low_A"], cfg["pi_pi_slip_high_A"])
                        + float(cfg["pi_pi_center_weight"]) * _interval_penalty(center_dist, cfg["pi_pi_center_low_A"], cfg["pi_pi_center_high_A"])
                        + float(cfg["pi_pi_angle_weight"]) * (angle / angle_sigma).square()
                    )
                    losses.append(loss)
    if not losses:
        return frac.new_zeros(())
    return torch.stack(losses).mean()


def _rotation_matrix_from_rotvec(rotvec: Tensor) -> Tensor:
    if rotvec.numel() == 0:
        return rotvec.new_zeros((0, 3, 3))
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = rotvec / theta
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    k = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).unsqueeze(0)
    sin_t = torch.sin(theta).view(-1, 1, 1)
    cos_t = torch.cos(theta).view(-1, 1, 1)
    return eye + sin_t * k + (1.0 - cos_t) * torch.bmm(k, k)


def _apply_component_pose(
    frac: Tensor,
    lattice: Tensor,
    components: list[Tensor],
    delta_cart: Tensor,
    rotvec: Tensor,
    enable_rotation: bool,
    *,
    canonicalize: bool = True,
) -> Tensor:
    result = frac.clone()
    cart = frac @ lattice
    if enable_rotation:
        rotations = _rotation_matrix_from_rotvec(rotvec)
    else:
        rotations = torch.eye(3, device=frac.device, dtype=frac.dtype).unsqueeze(0).repeat(len(components), 1, 1)
    for comp_idx, atom_idx in enumerate(components):
        center = cart[atom_idx].mean(dim=0, keepdim=True)
        rotated = (cart[atom_idx] - center) @ rotations[comp_idx].transpose(0, 1) + center
        translated = rotated + delta_cart[comp_idx].unsqueeze(0)
        frac_pose = cart_to_frac_coords(translated, lattice)
        result[atom_idx] = (
            _canonicalize_component_in_cell(frac_pose)
            if canonicalize
            else frac_pose
        )
    return result


@dataclass(frozen=True)
class _RigidPoseBasis:
    frac0: Tensor
    lattice: Tensor
    components: tuple[Tensor, ...]
    centers_cart: tuple[Tensor, ...]
    offsets_cart: tuple[Tensor, ...]

    @classmethod
    def from_fractional(
        cls,
        frac: Tensor,
        lattice: Tensor,
        components: list[Tensor],
    ) -> "_RigidPoseBasis":
        cart = frac @ lattice
        centers: list[Tensor] = []
        offsets: list[Tensor] = []
        for atom_index in components:
            center = cart[atom_index].mean(dim=0, keepdim=True)
            centers.append(center)
            offsets.append(cart[atom_index] - center)
        return cls(
            frac0=frac,
            lattice=lattice,
            components=tuple(components),
            centers_cart=tuple(centers),
            offsets_cart=tuple(offsets),
        )

    def reconstruct(
        self,
        delta_cart: Tensor,
        rotvec: Tensor,
        enable_rotation: bool,
    ) -> Tensor:
        output = self.frac0.clone()
        rotations = _rotation_matrix_from_rotvec(rotvec)
        identity = torch.eye(3, device=output.device, dtype=output.dtype)
        for index, atom_index in enumerate(self.components):
            rotation = rotations[index] if enable_rotation else identity
            cart = (
                self.centers_cart[index]
                + delta_cart[index].unsqueeze(0)
                + self.offsets_cart[index] @ rotation.transpose(0, 1)
            )
            output[atom_index] = cart_to_frac_coords(cart, self.lattice)
        return output


def _graph_clash_energy(
    view: _GraphView,
    frac: Tensor,
    cfg: dict[str, Any],
    *,
    return_counts: bool,
    floor_kind: str = "hard",
) -> _ClashEvaluation:
    device = frac.device
    dtype = torch.float32
    lattice = view.lattice.to(device=device, dtype=dtype)
    frac = frac.to(device=device, dtype=dtype)
    atom_types = view.atom_types.to(device=device)
    component_id = view.component_id.to(device=device)
    if bool(cfg["heavy_only"]):
        atom_mask = atom_types != 1
        if not bool(atom_mask.any().item()):
            zero = frac.new_zeros(())
            return _ClashEvaluation(zero, 0, 0, 0.0)
    else:
        atom_mask = torch.ones_like(atom_types, dtype=torch.bool)

    components = _component_indices(component_id)
    if not components:
        zero = frac.new_zeros(())
        return _ClashEvaluation(zero, 0, 0, 0.0)
    centers, radii = _component_centers_and_radii(frac, lattice, components)
    ops = _ops_for_spacegroup(
        view.spacegroup,
        device,
        dtype,
        cfg["max_symops"],
        symmetry_ops=view.symmetry_ops,
    )
    translations = _translation_grid(cfg["translations"], device, dtype)
    identity_idx = _identity_op_index(ops)
    zero_trans_mask = translations.abs().sum(dim=-1) == 0
    active_gaps: list[Tensor] = []
    pair_count = 0
    image_pair_count = 0
    identity_self_pair_count = 0
    broad_extra = float(cfg["broadphase_extra_A"])

    for left_i, left_atoms_all in enumerate(components):
        left_atoms = left_atoms_all[atom_mask[left_atoms_all]]
        if left_atoms.numel() == 0:
            continue
        left_frac = frac[left_atoms]
        left_types = atom_types[left_atoms]
        # All relative orbit configurations are represented with the lower ASU
        # component as the left anchor. This avoids directional duplication for
        # distinct ASU components while retaining self-image contacts.
        for right_i in range(left_i, len(components)):
            right_atoms_all = components[right_i]
            right_atoms = right_atoms_all[atom_mask[right_atoms_all]]
            if right_atoms.numel() == 0:
                continue
            right_frac_base = frac[right_atoms]
            right_types = atom_types[right_atoms]
            lower = _pair_lower_bounds(left_types, right_types, cfg, floor_kind=floor_kind).to(device=device, dtype=dtype)
            for op_idx in range(int(ops.shape[0])):
                R = ops[op_idx, :3, :3]
                tr = ops[op_idx, :3, 3]
                right_image_base = _canonicalize_component_in_cell(right_frac_base @ R.transpose(0, 1) + tr)
                image_center_frac = right_image_base.mean(dim=0)
                for trans_idx, trans in enumerate(translations):
                    is_identity_self = (
                        left_i == right_i
                        and op_idx == identity_idx
                        and bool(zero_trans_mask[trans_idx].item())
                    )
                    if is_identity_self:
                        continue
                    image_center = (image_center_frac + trans) @ lattice
                    center_dist = torch.linalg.norm(centers[left_i] - image_center)
                    if center_dist > radii[left_i] + radii[right_i] + broad_extra:
                        continue
                    right_image = right_image_base + trans
                    delta = (left_frac.unsqueeze(1) - right_image.unsqueeze(0)) @ lattice
                    dist = torch.linalg.norm(delta, dim=-1).clamp(min=1e-6)
                    gap = lower - dist
                    with torch.no_grad():
                        active = gap > 0.0
                        if left_i == right_i and float(cfg["self_image_overlap_tolerance_A"]) > 0.0:
                            active = active & (dist >= float(cfg["self_image_overlap_tolerance_A"]))
                        active_count = int(active.sum().item())
                        if active_count == 0:
                            continue
                    selected = gap[active]
                    active_gaps.append(selected)
                    if return_counts:
                        pair_count += active_count
                        image_pair_count += 1

    if not active_gaps:
        return _ClashEvaluation(
            frac.new_zeros(()),
            pair_count,
            image_pair_count,
            0.0,
            identity_self_pair_count,
        )
    gaps = torch.cat([value.reshape(-1) for value in active_gaps], dim=0)
    n_heavy = max(1, int(atom_mask.sum().item()))
    max_penetration = gaps.max()
    energy = gaps.square().sum() / float(n_heavy)
    energy = energy + float(cfg["hard_tail_weight"]) * max_penetration.square()
    return _ClashEvaluation(
        energy,
        int(gaps.numel()) if return_counts else 0,
        image_pair_count if return_counts else 0,
        float(max_penetration.detach().item()),
        identity_self_pair_count,
    )


def _view_with_geometry(view: _GraphView, frac: Tensor, lattice: Tensor) -> _GraphView:
    return _GraphView(
        frac=frac,
        lattice=lattice,
        atom_types=view.atom_types,
        component_id=view.component_id,
        spacegroup=view.spacegroup,
        edge_index=view.edge_index,
        hybridization=view.hybridization,
        symmetry_ops=view.symmetry_ops,
    )


def _pose_energy(
    view: _GraphView,
    frac_pose: Tensor,
    delta_cart: Tensor,
    rotvec: Tensor,
    cfg: dict[str, Any],
    *,
    in_projection: bool,
    rings: list[_RingCycle],
) -> tuple[Tensor, _ClashEvaluation]:
    pose_view = _view_with_geometry(view, frac_pose, view.lattice)
    hard = _graph_clash_energy(
        pose_view,
        frac_pose,
        cfg,
        return_counts=True,
        floor_kind="hard",
    )
    pi_energy = frac_pose.new_zeros(())
    trust = frac_pose.new_zeros(())
    if not in_projection:
        pi_energy = _graph_pi_energy(pose_view, frac_pose, cfg, rings=rings)
        trust = (
            float(cfg["translation_weight"]) * delta_cart.square().sum(dim=-1).mean()
            + float(cfg["rotation_weight"]) * rotvec.square().sum(dim=-1).mean()
        )
    total = (
        float(cfg["clash_weight"])
        * hard.energy
        + float(cfg["pi_pi_weight"]) * pi_energy
        + trust
    )
    return total, hard


def _correction_rank(hard: _ClashEvaluation, total_energy: Tensor) -> tuple[int, float, int, float, float]:
    return (*hard.key(), float(total_energy.detach().item()))


def _correct_graph_frac(
    view: _GraphView,
    cfg: dict[str, Any],
    *,
    projection_steps: int,
    refinement_steps: int,
) -> Tensor:
    frac0 = view.frac.detach().float()
    lattice = view.lattice.detach().float()
    components = _component_indices(view.component_id.to(frac0.device))
    if len(components) <= 0:
        return view.frac
    frac0 = _canonicalize_components_in_cell(frac0, components)
    delta = torch.zeros((len(components), 3), device=frac0.device, dtype=torch.float32, requires_grad=True)
    rotvec = torch.zeros_like(delta, requires_grad=True)
    enable_rotation = bool(cfg["enable_rotation"])
    translation_step = float(cfg["translation_step_size_A"])
    rotation_step = float(cfg["rotation_step_size"])
    line_search_steps = int(cfg["line_search_steps"])
    rings = _ring_cycles(_view_with_geometry(view, frac0, lattice), cfg) if refinement_steps > 0 else []
    best_frac = frac0
    initial_view = _view_with_geometry(view, frac0, lattice)
    initial_hard = _graph_clash_energy(initial_view, frac0, cfg, return_counts=True, floor_kind="hard")
    best_rank = _correction_rank(initial_hard, initial_hard.energy)
    total_steps = int(projection_steps) + int(refinement_steps)
    if total_steps <= 0:
        return frac0.to(dtype=view.frac.dtype, device=view.frac.device)
    for step_idx in range(total_steps):
        frac_pose = _apply_component_pose(
            frac0,
            lattice,
            components,
            delta,
            rotvec,
            enable_rotation,
            canonicalize=False,
        )
        in_projection = step_idx < int(projection_steps)
        energy, hard = _pose_energy(
            _view_with_geometry(view, frac_pose, lattice),
            frac_pose,
            delta,
            rotvec,
            cfg,
            in_projection=in_projection,
            rings=rings,
        )
        rank = _correction_rank(hard, energy)
        if rank < best_rank:
            best_rank = rank
            best_frac = frac_pose.detach()
        if in_projection and hard.pair_count == 0:
            continue
        if not energy.requires_grad:
            continue
        grad_delta, grad_rot = torch.autograd.grad(
            energy,
            (delta, rotvec),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        if grad_delta is None and (grad_rot is None or not enable_rotation):
            break
        base_energy = float(energy.detach().item())
        accepted = False
        trial_translation_step = translation_step
        trial_rotation_step = rotation_step
        with torch.no_grad():
            for _line_idx in range(line_search_steps):
                trial_delta = delta if grad_delta is None else delta - trial_translation_step * grad_delta
                trial_rot = rotvec
                if enable_rotation and grad_rot is not None:
                    trial_rot = rotvec - trial_rotation_step * grad_rot
                trial_frac = _apply_component_pose(
                    frac0,
                    lattice,
                    components,
                    trial_delta,
                    trial_rot,
                    enable_rotation,
                    canonicalize=False,
                )
                trial_energy, _trial_hard = _pose_energy(
                    _view_with_geometry(view, trial_frac, lattice),
                    trial_frac,
                    trial_delta,
                    trial_rot,
                    cfg,
                    in_projection=in_projection,
                    rings=rings,
                )
                if torch.isfinite(trial_energy) and float(trial_energy.item()) <= base_energy + 1e-10:
                    delta.copy_(trial_delta)
                    rotvec.copy_(trial_rot)
                    accepted = True
                    break
                trial_translation_step *= 0.5
                trial_rotation_step *= 0.5
        if not accepted:
            break
    return _canonicalize_components_in_cell(best_frac, components).to(
        dtype=view.frac.dtype,
        device=view.frac.device,
    )


def _pose_displacement(delta_cart: Tensor, rotvec: Tensor) -> float:
    value = delta_cart.square().sum() + rotvec.square().sum()
    return float(value.detach().item())


def _final_rank(
    result: ClashResult,
    pose_displacement: float,
) -> tuple[int, float, float, float]:
    return (
        int(result.pair_count),
        float(result.max_penetration_A),
        float(result.penalty.detach().item()),
        float(pose_displacement),
    )


def _final_diagnostics(
    initial: ClashResult,
    final: ClashResult,
    *,
    iterations: int,
    failure_reason: str | None,
) -> dict[str, float | int | bool | str | None]:
    resolved = not bool(final.clash)
    return {
        "resolved": resolved,
        "failure_reason": None if resolved else failure_reason,
        "iterations": int(iterations),
        "before_pair_count": int(initial.pair_count),
        "after_pair_count": int(final.pair_count),
        "before_intermolecular_pair_count": int(initial.intermolecular_pair_count),
        "after_intermolecular_pair_count": int(final.intermolecular_pair_count),
        "before_intramolecular_pair_count": int(initial.intramolecular_pair_count),
        "after_intramolecular_pair_count": int(final.intramolecular_pair_count),
        "before_max_penetration_A": float(initial.max_penetration_A),
        "after_max_penetration_A": float(final.max_penetration_A),
        "before_intermolecular_max_penetration_A": float(
            initial.intermolecular_max_penetration_A
        ),
        "after_intermolecular_max_penetration_A": float(
            final.intermolecular_max_penetration_A
        ),
        "before_intramolecular_max_penetration_A": float(
            initial.intramolecular_max_penetration_A
        ),
        "after_intramolecular_max_penetration_A": float(
            final.intramolecular_max_penetration_A
        ),
        "clash_neighborhood_complete": bool(final.neighborhood_complete),
        "clash_topology_complete": bool(final.topology_complete),
        "lattice_changed": False,
        "scale_factor": 1.0,
    }


def _solve_final_graph(
    view: _GraphView,
    cfg: dict[str, Any],
) -> tuple[Tensor, dict[str, float | int | bool | str | None]]:
    frac0 = view.frac.detach().float()
    lattice = view.lattice.detach().float()
    components = _component_indices(view.component_id.to(frac0.device))
    frac0 = _canonicalize_components_in_cell(frac0, components)
    clash_config = _final_clash_config(cfg)
    initial = _evaluate_final_clashes(
        view,
        frac0,
        clash_config,
        differentiable=False,
    )
    if not initial.clash:
        return (
            frac0.to(device=view.frac.device, dtype=view.frac.dtype),
            _final_diagnostics(
                initial,
                initial,
                iterations=0,
                failure_reason=None,
            ),
        )

    basis = _RigidPoseBasis.from_fractional(frac0, lattice, components)
    delta = torch.zeros(
        (len(components), 3),
        device=frac0.device,
        dtype=torch.float32,
        requires_grad=True,
    )
    rotvec = torch.zeros_like(delta, requires_grad=True)
    enable_rotation = bool(cfg["enable_rotation"])
    translation_step = float(cfg["translation_step_size_A"])
    rotation_step = float(cfg["rotation_step_size"])
    line_search_steps = int(cfg["line_search_steps"])
    max_steps = int(cfg["postprocess_steps"])

    best_frac = frac0
    best_result = initial
    best_rank = _final_rank(initial, 0.0)
    current_rank = best_rank
    iterations = 0
    failure_reason: str | None = "max_steps"

    for step_idx in range(max_steps):
        iterations = step_idx + 1
        pose = basis.reconstruct(delta, rotvec, enable_rotation)
        differentiable_result = _evaluate_final_clashes(
            view,
            pose,
            clash_config,
            differentiable=True,
        )
        trust = (
            float(cfg["translation_weight"]) * delta.square().sum(dim=-1).mean()
            + float(cfg["rotation_weight"]) * rotvec.square().sum(dim=-1).mean()
        )
        energy = float(cfg["clash_weight"]) * differentiable_result.penalty + trust
        if not bool(torch.isfinite(energy).item()) or not energy.requires_grad:
            failure_reason = "nonfinite" if not bool(torch.isfinite(energy).item()) else "stalled"
            break
        grad_delta, grad_rot = torch.autograd.grad(
            energy,
            (delta, rotvec),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        gradients = [value for value in (grad_delta, grad_rot) if value is not None]
        if not gradients:
            failure_reason = "stalled"
            break
        if any(not bool(torch.isfinite(value).all().item()) for value in gradients):
            failure_reason = "nonfinite"
            break
        if grad_delta is None or float(grad_delta.abs().sum().item()) == 0.0:
            if not enable_rotation or grad_rot is None or float(grad_rot.abs().sum().item()) == 0.0:
                failure_reason = "stalled"
                break

        accepted = False
        trial_translation_step = translation_step
        trial_rotation_step = rotation_step
        with torch.no_grad():
            for _ in range(line_search_steps):
                trial_delta = (
                    delta
                    if grad_delta is None
                    else delta - trial_translation_step * grad_delta
                )
                trial_rot = rotvec
                if enable_rotation and grad_rot is not None:
                    trial_rot = rotvec - trial_rotation_step * grad_rot
                trial_frac = basis.reconstruct(
                    trial_delta,
                    trial_rot,
                    enable_rotation,
                )
                if not bool(torch.isfinite(trial_frac).all().item()):
                    trial_translation_step *= 0.5
                    trial_rotation_step *= 0.5
                    continue
                trial_result = _evaluate_final_clashes(
                    view,
                    trial_frac,
                    clash_config,
                    differentiable=False,
                )
                displacement = _pose_displacement(trial_delta, trial_rot)
                trial_rank = _final_rank(trial_result, displacement)
                if trial_rank < current_rank:
                    delta.copy_(trial_delta)
                    rotvec.copy_(trial_rot)
                    current_rank = trial_rank
                    accepted = True
                    if trial_rank < best_rank:
                        best_rank = trial_rank
                        best_frac = trial_frac.detach()
                        best_result = trial_result
                    if not trial_result.clash:
                        failure_reason = None
                    break
                trial_translation_step *= 0.5
                trial_rotation_step *= 0.5
        if not accepted:
            failure_reason = "stalled"
            break
        if not best_result.clash:
            break

    best_frac = _canonicalize_components_in_cell(best_frac, components)
    final_result = _evaluate_final_clashes(
        view,
        best_frac,
        clash_config,
        differentiable=False,
    )
    if not final_result.clash:
        failure_reason = None
    return (
        best_frac.to(device=view.frac.device, dtype=view.frac.dtype),
        _final_diagnostics(
            initial,
            final_result,
            iterations=iterations,
            failure_reason=failure_reason,
        ),
    )


def count_symmetry_periodic_clashes(
    *,
    batch: AsuBatch,
    frac_coords: Tensor,
    lattice: Tensor,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    cfg = normalize_symmetry_corrector_config(config)
    total_penalty = 0.0
    total_pairs = 0
    total_inter_pairs = 0
    total_intra_pairs = 0
    total_inter_eligible = 0
    total_intra_eligible = 0
    total_image_pairs = 0
    total_intra_sources = 0
    total_unit_components = 0
    max_penetration = 0.0
    max_inter_penetration = 0.0
    max_intra_penetration = 0.0
    max_translation_abs = 0
    any_clash = False
    any_inter_clash = False
    any_intra_clash = False
    neighborhood_complete = True
    topology_complete = True
    frac = frac_coords.float()
    lat = lattice.float()
    clash_config = _final_clash_config(cfg)
    for graph_idx in range(int(batch.batch_size)):
        view = _graph_view(batch, frac, lat, graph_idx)
        components = _component_indices(view.component_id.to(view.frac.device))
        canonical_frac = _canonicalize_components_in_cell(view.frac, components)
        evaluation = _evaluate_final_clashes(
            view,
            canonical_frac,
            clash_config,
            differentiable=False,
        )
        total_penalty += float(evaluation.penalty.detach().item())
        total_pairs += int(evaluation.pair_count)
        total_inter_pairs += int(evaluation.intermolecular_pair_count)
        total_intra_pairs += int(evaluation.intramolecular_pair_count)
        total_inter_eligible += int(evaluation.intermolecular_eligible_pair_count)
        total_intra_eligible += int(evaluation.intramolecular_eligible_pair_count)
        total_image_pairs += int(evaluation.intermolecular_source_pair_count)
        total_intra_sources += int(evaluation.intramolecular_source_component_count)
        total_unit_components += int(evaluation.unit_component_count)
        max_penetration = max(max_penetration, float(evaluation.max_penetration_A))
        max_inter_penetration = max(
            max_inter_penetration,
            float(evaluation.intermolecular_max_penetration_A),
        )
        max_intra_penetration = max(
            max_intra_penetration,
            float(evaluation.intramolecular_max_penetration_A),
        )
        max_translation_abs = max(
            max_translation_abs,
            int(evaluation.max_translation_abs),
        )
        any_clash = any_clash or bool(evaluation.clash)
        any_inter_clash = any_inter_clash or bool(evaluation.intermolecular_clash)
        any_intra_clash = any_intra_clash or bool(evaluation.intramolecular_clash)
        neighborhood_complete = neighborhood_complete and bool(
            evaluation.neighborhood_complete
        )
        topology_complete = topology_complete and bool(evaluation.topology_complete)
    return {
        "clash": int(any_clash),
        "intermolecular_clash": int(any_inter_clash),
        "intramolecular_clash": int(any_intra_clash),
        "penalty": total_penalty,
        "pair_count": total_pairs,
        "intermolecular_pair_count": total_inter_pairs,
        "intramolecular_pair_count": total_intra_pairs,
        "intermolecular_eligible_pair_count": total_inter_eligible,
        "intramolecular_eligible_pair_count": total_intra_eligible,
        "image_pair_count": total_image_pairs,
        "intramolecular_source_component_count": total_intra_sources,
        "unit_component_count": total_unit_components,
        "max_penetration_A": max_penetration,
        "intermolecular_max_penetration_A": max_inter_penetration,
        "intramolecular_max_penetration_A": max_intra_penetration,
        "neighborhood_complete": bool(neighborhood_complete),
        "topology_complete": bool(topology_complete),
        "max_translation_abs": int(max_translation_abs),
        "identity_self_pair_count": 0,
    }


def symmetry_rigid_postprocess_batch(
    *,
    batch: AsuBatch,
    frac_coords: Tensor,
    lattice: Tensor,
    postprocess_cfg: Optional[dict[str, Any]] = None,
) -> SymmetryRigidCorrectionResult:
    cfg = normalize_symmetry_corrector_config(postprocess_cfg)
    lattice_was_unbatched = lattice.ndim == 2
    lattice_batch = lattice.unsqueeze(0) if lattice_was_unbatched else lattice
    scale_factor = torch.ones(
        (int(batch.batch_size),),
        device=lattice_batch.device,
        dtype=lattice_batch.dtype,
    )
    if not bool(cfg["postprocess_enabled"]):
        return SymmetryRigidCorrectionResult(
            frac_coords=frac_coords,
            lattice=lattice if lattice_was_unbatched else lattice_batch,
            scale_factor=scale_factor,
            diagnostics=[],
        )
    frac = frac_coords.detach()
    lat = lattice_batch.detach()
    corrected = frac.clone()
    diagnostics: list[dict[str, float | int | bool | str | None]] = []
    with torch.enable_grad():
        for graph_idx in range(int(batch.batch_size)):
            view = _graph_view(batch, corrected, lat, graph_idx)
            start = int(batch.ptr[graph_idx].item())
            end = int(batch.ptr[graph_idx + 1].item())
            final_frac, graph_diagnostics = _solve_final_graph(
                view,
                cfg,
            )
            corrected[start:end] = final_frac.to(device=corrected.device, dtype=corrected.dtype)
            diagnostics.append(graph_diagnostics)
    result_lattice = lattice if lattice_was_unbatched else lattice_batch
    return SymmetryRigidCorrectionResult(
        frac_coords=corrected.to(device=frac_coords.device, dtype=frac_coords.dtype),
        lattice=result_lattice,
        scale_factor=scale_factor,
        diagnostics=diagnostics,
    )


__all__ = [
    "SymmetryRigidCorrectionResult",
    "count_symmetry_periodic_clashes",
    "normalize_symmetry_corrector_config",
    "symmetry_rigid_postprocess_batch",
]
