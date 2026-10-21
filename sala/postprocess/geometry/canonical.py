from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import torch
from torch import Tensor

from sala.core.lattice import frac_to_cart_coords
from sala.postprocess.geometry.component_topology import (
    ComponentTopology,
)
from sala.postprocess.geometry.ops import contiguous_fractional_coords
from sala.postprocess.geometry.periodic import (
    PeriodicLattice,
    closest_lattice_shift,
    prepare_periodic_lattice,
)
from sala.postprocess.geometry.symmetry_context import SymmetryContext


@dataclass(frozen=True)
class ComponentImage:
    component_id: int
    symop_index: int
    unit_shift: tuple[int, int, int]
    supercell_shift: tuple[int, int, int]
    frac_coords: Tensor
    cart_coords: Tensor
    atom_types: Tensor
    base_atom_index: Tensor
    signature: tuple[tuple[int, int], ...]
    topology_signature: str
    centroid_frac: Tensor
    centroid_cart: Tensor
    radius_A: float
    central_cell: bool


@dataclass(frozen=True)
class CanonicalCrystal:
    lattice: Tensor
    contiguous_frac: Tensor
    unit_components: tuple[ComponentImage, ...]
    supercell_components: tuple[ComponentImage, ...]
    supercell_radius: int
    periodic: PeriodicLattice
    symmetry: SymmetryContext | None
    topologies: tuple[ComponentTopology, ...]
    orbit_multiplicity: tuple[int, ...]


def canonical_component_shift(frac_coords: Tensor) -> Tensor:
    """Choose an integer component shift with deterministic in-cell maximization."""
    frac = torch.as_tensor(frac_coords, dtype=torch.float32)
    if frac.numel() == 0:
        return torch.zeros(3, dtype=torch.float32, device=frac.device)

    candidates_per_axis: list[Tensor] = []
    for axis in range(3):
        x = frac[:, axis]
        base = torch.floor(-x).to(torch.long)
        near = torch.cat([base - 1, base, base + 1, torch.floor(1.0 - x).to(torch.long)])
        candidates_per_axis.append(torch.unique(near.cpu()).to(device=frac.device, dtype=torch.long))

    center = torch.full((3,), 0.5, dtype=frac.dtype, device=frac.device)
    best_shift: Tensor | None = None
    best_key: tuple[int, float, tuple[int, int, int]] | None = None
    for sx in candidates_per_axis[0].tolist():
        for sy in candidates_per_axis[1].tolist():
            for sz in candidates_per_axis[2].tolist():
                shift = torch.tensor([sx, sy, sz], dtype=frac.dtype, device=frac.device)
                shifted = frac + shift
                inside = ((shifted >= 0.0) & (shifted < 1.0)).all(dim=-1)
                key = (
                    -int(inside.sum().item()),
                    float(torch.linalg.norm(shifted.mean(dim=0) - center).item()),
                    (int(sx), int(sy), int(sz)),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_shift = shift
    if best_shift is None:
        raise RuntimeError("Unable to choose a canonical component shift.")
    return best_shift


def _eval_atom_mask(asu, atom_scope: str) -> Tensor:
    if atom_scope == "all":
        return torch.ones_like(asu.atom_types, dtype=torch.bool)
    if atom_scope == "heavy":
        return asu.atom_types.long() != 1
    raise ValueError("atom_scope must be 'heavy' or 'all'.")


@lru_cache(maxsize=8)
def _translation_grid(radius: int) -> tuple[tuple[int, int, int], ...]:
    r = max(0, int(radius))
    return tuple(product(range(-r, r + 1), repeat=3))


def _element_signature(atom_types: Tensor) -> tuple[tuple[int, int], ...]:
    values, counts = torch.unique(atom_types.long(), sorted=True, return_counts=True)
    return tuple((int(z), int(n)) for z, n in zip(values.tolist(), counts.tolist()))


def _periodic_cartesian_distances(
    left_frac: Tensor,
    right_frac: Tensor,
    periodic: PeriodicLattice,
) -> Tensor:
    """Return exact pairwise periodic distances for a general triclinic cell."""
    distances = torch.empty(
        (left_frac.shape[0], right_frac.shape[0]),
        dtype=torch.float64,
        device=periodic.lattice.device,
    )
    for left_index, left in enumerate(left_frac):
        for right_index, right in enumerate(right_frac):
            distances[left_index, right_index] = closest_lattice_shift(
                left.to(torch.float64) - right.to(torch.float64),
                periodic,
            ).distance_A
    return distances


def _linear_assignment(cost: Tensor) -> tuple[Tensor, Tensor]:
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(rows, dtype=torch.long, device=cost.device),
        torch.as_tensor(cols, dtype=torch.long, device=cost.device),
    )


def component_images_equivalent(
    left: ComponentImage,
    right: ComponentImage,
    lattice: Tensor,
    *,
    tolerance_A: float,
    periodic: PeriodicLattice | None = None,
    topology: ComponentTopology | None = None,
) -> bool:
    if left.component_id != right.component_id or left.signature != right.signature:
        return False
    exact_lattice = periodic or prepare_periodic_lattice(lattice)
    centroid_delta = (
        left.centroid_frac.detach().to(
            device=exact_lattice.lattice.device,
            dtype=torch.float64,
        )
        - right.centroid_frac.detach().to(
            device=exact_lattice.lattice.device,
            dtype=torch.float64,
        )
    )
    wrapped_centroid_delta = centroid_delta - torch.round(centroid_delta)
    coordinate_bounds = exact_lattice.reciprocal_column_norms * (
        float(tolerance_A) + 1e-12
    )
    if bool((wrapped_centroid_delta.abs() > coordinate_bounds).any().item()):
        return False
    if (
        closest_lattice_shift(centroid_delta, exact_lattice).distance_A
        > float(tolerance_A)
    ):
        return False
    if topology is not None and (
        left.topology_signature != topology.signature
        or right.topology_signature != topology.signature
    ):
        return False
    for atomic_number, _ in left.signature:
        left_idx = torch.where(left.atom_types.long() == int(atomic_number))[0]
        right_idx = torch.where(right.atom_types.long() == int(atomic_number))[0]
        if left_idx.numel() != right_idx.numel():
            return False
        cost = _periodic_cartesian_distances(
            left.frac_coords[left_idx],
            right.frac_coords[right_idx],
            exact_lattice,
        )
        rows, cols = _linear_assignment(cost)
        if rows.numel() != left_idx.numel():
            return False
        if float(cost[rows, cols].max().item()) > float(tolerance_A):
            return False
    return True


def _make_image(
    *,
    component_id: int,
    symop_index: int,
    unit_shift: tuple[int, int, int],
    supercell_shift: tuple[int, int, int],
    frac_coords: Tensor,
    atom_types: Tensor,
    base_atom_index: Tensor,
    lattice: Tensor,
    central_cell: bool,
    topology_signature: str = "",
) -> ComponentImage:
    cart = frac_to_cart_coords(frac_coords, lattice)
    centroid_frac = frac_coords.mean(dim=0)
    centroid_cart = cart.mean(dim=0)
    radius = float(torch.linalg.norm(cart - centroid_cart, dim=-1).max().item()) if cart.numel() else 0.0
    return ComponentImage(
        component_id=int(component_id),
        symop_index=int(symop_index),
        unit_shift=unit_shift,
        supercell_shift=supercell_shift,
        frac_coords=frac_coords,
        cart_coords=cart,
        atom_types=atom_types,
        base_atom_index=base_atom_index,
        signature=_element_signature(atom_types),
        topology_signature=str(topology_signature),
        centroid_frac=centroid_frac,
        centroid_cart=centroid_cart,
        radius_A=radius,
        central_cell=bool(central_cell),
    )


def build_unit_cell_components(
    asu,
    *,
    atom_scope: str = "heavy",
    max_symops: int | None = None,
    dedup_tolerance_A: float = 1e-3,
    periodic: PeriodicLattice | None = None,
    symmetry: SymmetryContext | None = None,
    topologies: tuple[ComponentTopology, ...] = (),
) -> tuple[Tensor, tuple[ComponentImage, ...]]:
    mask = _eval_atom_mask(asu, atom_scope)
    contiguous = contiguous_fractional_coords(asu).to(
        dtype=torch.float32, device=asu.frac_coords_gt.device
    )
    if not mask.any().item():
        return contiguous, ()

    frac = contiguous[mask]
    atom_types = asu.atom_types.long()[mask]
    component_ids = asu.component_id.long()[mask]
    base_atom_index = torch.where(mask)[0].long()
    lattice = asu.lattice_gt.to(device=frac.device, dtype=frac.dtype)
    exact_lattice = periodic or prepare_periodic_lattice(lattice)
    if symmetry is None:
        stored_ops = getattr(asu, "symmetry_ops", None)
        if stored_ops is None:
            raise ValueError("canonical geometry requires explicit symmetry_ops")
        ops = torch.as_tensor(stored_ops, device=frac.device, dtype=frac.dtype)
        if ops.ndim != 3 or ops.shape[1:] != (4, 4) or ops.shape[0] == 0:
            raise ValueError("symmetry_ops must have shape [G, 4, 4]")
        if max_symops is not None and int(max_symops) != int(ops.shape[0]):
            raise ValueError("max_symops must preserve the complete operation set")
    else:
        ops = symmetry.operations.to(device=frac.device, dtype=frac.dtype)
    topology_by_component = {topology.component_id: topology for topology in topologies}
    sym_frac = torch.einsum("gij,nj->gni", ops[:, :3, :3], frac) + ops[:, None, :3, 3]

    unique_images: list[ComponentImage] = []
    component_values = torch.unique(component_ids, sorted=True)
    for symop_index in range(int(sym_frac.shape[0])):
        for component_tensor in component_values:
            component_id = int(component_tensor.item())
            idx = torch.where(component_ids == component_id)[0]
            comp_frac = sym_frac[symop_index, idx]
            shift = canonical_component_shift(comp_frac)
            shift_tuple = tuple(int(v) for v in shift.round().long().tolist())
            candidate = _make_image(
                component_id=component_id,
                symop_index=symop_index,
                unit_shift=shift_tuple,
                supercell_shift=(0, 0, 0),
                frac_coords=comp_frac + shift,
                atom_types=atom_types[idx],
                base_atom_index=base_atom_index[idx],
                lattice=lattice,
                central_cell=True,
                topology_signature=(
                    topology_by_component[component_id].signature
                    if component_id in topology_by_component
                    else ""
                ),
            )
            if any(
                component_images_equivalent(
                    existing,
                    candidate,
                    lattice,
                    tolerance_A=dedup_tolerance_A,
                    periodic=exact_lattice,
                    topology=topology_by_component.get(component_id),
                )
                for existing in unique_images
                if existing.component_id == component_id and existing.signature == candidate.signature
            ):
                continue
            unique_images.append(candidate)
    return contiguous, tuple(unique_images)


def expand_unit_components(
    unit_components: tuple[ComponentImage, ...],
    lattice: Tensor,
    *,
    supercell_radius: int,
) -> tuple[ComponentImage, ...]:
    if not unit_components:
        return ()
    out: list[ComponentImage] = []
    for image in unit_components:
        for tx, ty, tz in _translation_grid(supercell_radius):
            shift_tuple = (int(tx), int(ty), int(tz))
            shift = torch.tensor(shift_tuple, dtype=image.frac_coords.dtype, device=image.frac_coords.device)
            frac = image.frac_coords + shift
            shift_cart = frac_to_cart_coords(shift, lattice)
            out.append(
                ComponentImage(
                    component_id=image.component_id,
                    symop_index=image.symop_index,
                    unit_shift=image.unit_shift,
                    supercell_shift=shift_tuple,
                    frac_coords=frac,
                    cart_coords=image.cart_coords + shift_cart,
                    atom_types=image.atom_types,
                    base_atom_index=image.base_atom_index,
                    signature=image.signature,
                    topology_signature=image.topology_signature,
                    centroid_frac=image.centroid_frac + shift,
                    centroid_cart=image.centroid_cart + shift_cart,
                    radius_A=image.radius_A,
                    central_cell=shift_tuple == (0, 0, 0),
                )
            )
    return tuple(out)


def prepare_canonical_crystal(
    asu,
    *,
    atom_scope: str = "heavy",
    max_symops: int | None = None,
    supercell_radius: int = 1,
    dedup_tolerance_A: float = 1e-3,
    periodic: PeriodicLattice | None = None,
    symmetry: SymmetryContext | None = None,
    topologies: tuple[ComponentTopology, ...] = (),
) -> CanonicalCrystal:
    lattice = asu.lattice_gt.to(dtype=torch.float32, device=asu.frac_coords_gt.device)
    exact_lattice = periodic or prepare_periodic_lattice(lattice)
    contiguous, unit = build_unit_cell_components(
        asu,
        atom_scope=atom_scope,
        max_symops=max_symops,
        dedup_tolerance_A=dedup_tolerance_A,
        periodic=exact_lattice,
        symmetry=symmetry,
        topologies=topologies,
    )
    component_order = tuple(topology.component_id for topology in topologies)
    if not component_order:
        component_order = tuple(sorted({image.component_id for image in unit}))
    orbit_multiplicity = tuple(
        sum(image.component_id == component_id for image in unit)
        for component_id in component_order
    )
    return CanonicalCrystal(
        lattice=lattice,
        contiguous_frac=contiguous,
        unit_components=unit,
        supercell_components=expand_unit_components(unit, lattice, supercell_radius=supercell_radius),
        supercell_radius=int(supercell_radius),
        periodic=exact_lattice,
        symmetry=symmetry,
        topologies=topologies,
        orbit_multiplicity=orbit_multiplicity,
    )
