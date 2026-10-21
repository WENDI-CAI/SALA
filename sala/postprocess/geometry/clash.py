from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

from sala.postprocess.geometry.canonical import (
    CanonicalCrystal,
    ComponentImage,
    prepare_canonical_crystal,
)
from sala.postprocess.geometry.component_topology import (
    ComponentTopology,
    build_component_topologies,
)
from sala.postprocess.geometry.contact_statistics import element_vdw_radii
from sala.postprocess.geometry.periodic import (
    closest_lattice_shift,
    nearest_periodic_candidates,
    translations_within_cutoff,
)

DEFAULT_VDW_OFFSET_A = 0.40


@dataclass(frozen=True)
class ClashConfig:
    atom_scope: str = "heavy"
    max_symops: int | None = None
    supercell_radius: int = 1
    threshold_mode: str = "vdw_offset"
    distance_threshold_A: float = 1.2
    vdw_offset_A: float = DEFAULT_VDW_OFFSET_A
    dedup_tolerance_A: float = 1e-3

    def __post_init__(self) -> None:
        if self.atom_scope not in {"heavy", "all"}:
            raise ValueError("atom_scope must be 'heavy' or 'all'.")
        if self.max_symops is not None and int(self.max_symops) < 1:
            raise ValueError("max_symops must be positive when provided.")
        if int(self.supercell_radius) < 0:
            raise ValueError("supercell_radius must be non-negative.")
        if self.threshold_mode not in {"vdw_offset", "fixed"}:
            raise ValueError("threshold_mode must be 'vdw_offset' or 'fixed'.")
        if float(self.distance_threshold_A) <= 0.0:
            raise ValueError("distance_threshold_A must be positive.")
        if (
            not math.isfinite(float(self.vdw_offset_A))
            or float(self.vdw_offset_A) < 0.0
        ):
            raise ValueError("vdw_offset_A must be finite and non-negative.")
        if float(self.dedup_tolerance_A) <= 0.0:
            raise ValueError("dedup_tolerance_A must be positive.")


@dataclass(frozen=True)
class ClashResult:
    clash: bool
    pair_count: int
    min_distance_A: float
    threshold_min_A: float
    threshold_max_A: float
    threshold_mean_A: float
    max_penetration_A: float
    penalty: Tensor
    intermolecular_clash: bool
    intramolecular_clash: bool
    intermolecular_pair_count: int
    intramolecular_pair_count: int
    intermolecular_eligible_pair_count: int
    intramolecular_eligible_pair_count: int
    intermolecular_min_distance_A: float
    intramolecular_min_distance_A: float
    intermolecular_max_penetration_A: float
    intramolecular_max_penetration_A: float
    intermolecular_source_pair_count: int
    intramolecular_source_component_count: int
    unit_component_count: int
    neighborhood_complete: bool
    topology_complete: bool
    max_translation_abs: int
    vdw_offset_A: float
    vdw_known_radius_pair_count: int
    vdw_unknown_radius_pair_count: int

    def to_metrics(self) -> dict[str, float]:
        return {
            "clash": 1.0 if self.clash else 0.0,
            "clash_pair_count": float(self.pair_count),
            "clash_min_distance_A": float(self.min_distance_A),
            "clash_threshold_min_A": float(self.threshold_min_A),
            "clash_threshold_max_A": float(self.threshold_max_A),
            "clash_threshold_mean_A": float(self.threshold_mean_A),
            "clash_max_penetration_A": float(self.max_penetration_A),
            "intermolecular_clash": 1.0 if self.intermolecular_clash else 0.0,
            "intramolecular_clash": 1.0 if self.intramolecular_clash else 0.0,
            "intermolecular_clash_pair_count": float(self.intermolecular_pair_count),
            "intramolecular_clash_pair_count": float(self.intramolecular_pair_count),
            "intermolecular_eligible_pair_count": float(
                self.intermolecular_eligible_pair_count
            ),
            "intramolecular_eligible_pair_count": float(
                self.intramolecular_eligible_pair_count
            ),
            "intermolecular_clash_pairs_per_1000_eligible": _rate_per_1000(
                self.intermolecular_pair_count,
                self.intermolecular_eligible_pair_count,
            ),
            "intramolecular_clash_pairs_per_1000_eligible": _rate_per_1000(
                self.intramolecular_pair_count,
                self.intramolecular_eligible_pair_count,
            ),
            "intermolecular_clash_min_distance_A": float(
                self.intermolecular_min_distance_A
            ),
            "intramolecular_clash_min_distance_A": float(
                self.intramolecular_min_distance_A
            ),
            "intermolecular_clash_max_penetration_A": float(
                self.intermolecular_max_penetration_A
            ),
            "intramolecular_clash_max_penetration_A": float(
                self.intramolecular_max_penetration_A
            ),
            "intermolecular_clash_source_pair_count": float(
                self.intermolecular_source_pair_count
            ),
            "intramolecular_clash_source_component_count": float(
                self.intramolecular_source_component_count
            ),
            "clash_unit_component_count": float(self.unit_component_count),
            "clash_neighborhood_complete": 1.0 if self.neighborhood_complete else 0.0,
            "clash_topology_complete": 1.0 if self.topology_complete else 0.0,
            "clash_max_translation_abs": float(self.max_translation_abs),
            "clash_vdw_offset_A": float(self.vdw_offset_A),
            "clash_vdw_known_radius_pair_count": float(
                self.vdw_known_radius_pair_count
            ),
            "clash_vdw_unknown_radius_pair_count": float(
                self.vdw_unknown_radius_pair_count
            ),
        }


def _nan() -> float:
    return float("nan")


def _rate_per_1000(count: int, eligible: int) -> float:
    if int(eligible) <= 0:
        return 0.0
    return 1000.0 * float(count) / float(eligible)


def _vdw_radius(z: int) -> float:
    radii, _ = element_vdw_radii()
    atomic_number = int(z)
    if 0 <= atomic_number < len(radii):
        return float(radii[atomic_number])
    return 1.80


def vdw_pair_threshold(
    z1: int,
    z2: int,
    *,
    offset_A: float = DEFAULT_VDW_OFFSET_A,
) -> float:
    offset = float(offset_A)
    if not math.isfinite(offset) or offset < 0.0:
        raise ValueError("offset_A must be finite and non-negative.")
    return max(0.0, _vdw_radius(int(z1)) + _vdw_radius(int(z2)) - offset)


@dataclass(frozen=True)
class _VdwOffsetLookup:
    thresholds: Tensor
    sources: Tensor


@lru_cache(maxsize=16)
def _vdw_offset_lookup(offset_A: float) -> _VdwOffsetLookup:
    radius_values, known_values = element_vdw_radii()
    radii = torch.tensor(radius_values, dtype=torch.float32)
    known = torch.tensor(known_values, dtype=torch.bool)
    thresholds = torch.clamp(radii[:, None] + radii[None, :] - float(offset_A), min=0.0)
    sources = torch.where(
        known[:, None] & known[None, :],
        torch.zeros((119, 119), dtype=torch.int8),
        torch.ones((119, 119), dtype=torch.int8),
    )
    return _VdwOffsetLookup(thresholds=thresholds, sources=sources)


def _zero_penalty(source) -> Tensor:
    coords = (
        source.contiguous_frac
        if isinstance(source, CanonicalCrystal)
        else source.frac_coords_gt
    )
    return coords.sum() * 0.0


@dataclass(frozen=True)
class PhysicalInterComponentPair:
    lower_bound_A: float
    max_floor_A: float
    left_index: int
    right_index: int
    relative_shift: tuple[int, int, int]


@dataclass(frozen=True)
class PhysicalInterComponentSearch:
    pairs: tuple[PhysicalInterComponentPair, ...]
    complete: bool
    max_translation_abs: int


@dataclass
class _ThresholdAccumulator:
    minimum: float = math.inf
    maximum: float = -math.inf
    total: float = 0.0
    count: int = 0

    def add(self, values: Tensor) -> None:
        if values.numel() == 0:
            return
        detached = values.detach()
        self.minimum = min(self.minimum, float(detached.min().item()))
        self.maximum = max(self.maximum, float(detached.max().item()))
        self.total += float(detached.sum().item())
        self.count += int(detached.numel())


def _empty_result(
    asu,
    *,
    config: ClashConfig,
    topology_complete: bool = True,
) -> ClashResult:
    penalty = _zero_penalty(asu)
    return ClashResult(
        clash=False,
        pair_count=0,
        min_distance_A=_nan(),
        threshold_min_A=_nan(),
        threshold_max_A=_nan(),
        threshold_mean_A=_nan(),
        max_penetration_A=0.0,
        penalty=penalty,
        intermolecular_clash=False,
        intramolecular_clash=False,
        intermolecular_pair_count=0,
        intramolecular_pair_count=0,
        intermolecular_eligible_pair_count=0,
        intramolecular_eligible_pair_count=0,
        intermolecular_min_distance_A=_nan(),
        intramolecular_min_distance_A=_nan(),
        intermolecular_max_penetration_A=0.0,
        intramolecular_max_penetration_A=0.0,
        intermolecular_source_pair_count=0,
        intramolecular_source_component_count=0,
        unit_component_count=0,
        neighborhood_complete=True,
        topology_complete=bool(topology_complete),
        max_translation_abs=0,
        vdw_offset_A=(
            float(config.vdw_offset_A)
            if config.threshold_mode == "vdw_offset"
            else _nan()
        ),
        vdw_known_radius_pair_count=0,
        vdw_unknown_radius_pair_count=0,
    )


def _maximum_floor(
    left: ComponentImage,
    right: ComponentImage,
    *,
    config: ClashConfig,
    lookup: _VdwOffsetLookup | None,
) -> float:
    if config.threshold_mode == "fixed":
        return float(config.distance_threshold_A)
    assert lookup is not None
    values = lookup.thresholds
    left_types = left.atom_types.detach().cpu().long().clamp(0, values.shape[0] - 1)
    right_types = right.atom_types.detach().cpu().long().clamp(0, values.shape[1] - 1)
    return float(values[left_types[:, None], right_types[None, :]].max().item())


def _cross_thresholds(
    left_types: Tensor,
    right_types: Tensor,
    *,
    config: ClashConfig,
    lookup: _VdwOffsetLookup | None,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if config.threshold_mode == "fixed":
        return torch.full(
            (int(left_types.numel()), int(right_types.numel())),
            float(config.distance_threshold_A),
            dtype=dtype,
            device=device,
        )
    assert lookup is not None
    local = lookup.thresholds.to(device=device, dtype=dtype)
    left = left_types.long().clamp(0, local.shape[0] - 1)
    right = right_types.long().clamp(0, local.shape[1] - 1)
    return local[left[:, None], right[None, :]]


def _paired_thresholds(
    left_types: Tensor,
    right_types: Tensor,
    *,
    config: ClashConfig,
    lookup: _VdwOffsetLookup | None,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if config.threshold_mode == "fixed":
        return torch.full(
            (int(left_types.numel()),),
            float(config.distance_threshold_A),
            dtype=dtype,
            device=device,
        )
    assert lookup is not None
    local = lookup.thresholds.to(device=device, dtype=dtype)
    left = left_types.long().clamp(0, local.shape[0] - 1)
    right = right_types.long().clamp(0, local.shape[1] - 1)
    return local[left, right]


def _cross_radius_sources(
    left_types: Tensor,
    right_types: Tensor,
    lookup: _VdwOffsetLookup,
) -> Tensor:
    local = lookup.sources.to(device=left_types.device)
    left = left_types.long().clamp(0, local.shape[0] - 1)
    right = right_types.long().clamp(0, local.shape[1] - 1)
    return local[left[:, None], right[None, :]]


def _paired_radius_sources(
    left_types: Tensor,
    right_types: Tensor,
    lookup: _VdwOffsetLookup,
) -> Tensor:
    local = lookup.sources.to(device=left_types.device)
    left = left_types.long().clamp(0, local.shape[0] - 1)
    right = right_types.long().clamp(0, local.shape[1] - 1)
    return local[left, right]


def _translated_cartesian(
    image: ComponentImage,
    shift: tuple[int, int, int],
    lattice: Tensor,
) -> Tensor:
    integer_shift = torch.tensor(
        shift,
        dtype=image.cart_coords.dtype,
        device=image.cart_coords.device,
    )
    local_lattice = lattice.to(
        dtype=image.cart_coords.dtype,
        device=image.cart_coords.device,
    )
    return image.cart_coords + integer_shift @ local_lattice


def _shortest_nonzero_shift(
    crystal: CanonicalCrystal,
    *,
    max_periodic_candidates: int | None = None,
) -> tuple[int, int, int]:
    result = nearest_periodic_candidates(
        torch.zeros(
            (1, 3), dtype=torch.float64, device=crystal.periodic.lattice.device
        ),
        crystal.periodic,
        count=2,
        max_candidates=max_periodic_candidates,
    )
    for shift in result.shifts:
        shift_tuple = tuple(int(value) for value in shift.tolist())
        if shift_tuple != (0, 0, 0):
            return shift_tuple
    raise RuntimeError("Unable to locate a nonzero lattice translation.")


def _candidate_distance_upper_bound(
    crystal: CanonicalCrystal,
    left: ComponentImage,
    right: ComponentImage,
    shift: tuple[int, int, int],
) -> float:
    left_cart = left.cart_coords.detach().to(
        device=crystal.periodic.lattice.device,
        dtype=torch.float64,
    )
    right_cart = right.cart_coords.detach().to(
        device=crystal.periodic.lattice.device,
        dtype=torch.float64,
    )
    shift_cart = (
        torch.tensor(
            shift,
            dtype=torch.float64,
            device=crystal.periodic.lattice.device,
        )
        @ crystal.periodic.lattice
    )
    return float(torch.cdist(left_cart, right_cart + shift_cart).min().item())


def _unordered_pair_records(
    crystal: CanonicalCrystal,
    *,
    config: ClashConfig,
    lookup: _VdwOffsetLookup | None,
    include_global_minimum: bool = True,
    left_indices: tuple[int, ...] | None = None,
    max_periodic_candidates: int | None = None,
) -> tuple[list[PhysicalInterComponentPair], bool, int]:
    components = crystal.unit_components
    if not components:
        return [], True, 0
    shortest_nonzero = (
        _shortest_nonzero_shift(
            crystal,
            max_periodic_candidates=max_periodic_candidates,
        )
        if include_global_minimum
        else None
    )
    max_floor_cache: dict[
        tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]],
        float,
    ] = {}
    records: list[PhysicalInterComponentPair] = []
    seen: set[tuple[int, int, tuple[int, int, int]]] = set()
    complete = True
    max_translation_abs = 0
    selected_left_indices = (
        tuple(range(len(components))) if left_indices is None else left_indices
    )
    for left_index in selected_left_indices:
        left = components[left_index]
        for right_index, right in enumerate(components):
            signature_key = tuple(sorted((left.signature, right.signature)))
            max_floor = max_floor_cache.get(signature_key)
            if max_floor is None:
                max_floor = _maximum_floor(left, right, config=config, lookup=lookup)
                max_floor_cache[signature_key] = max_floor
            delta = right.centroid_frac.detach().to(
                device=crystal.periodic.lattice.device,
                dtype=torch.float64,
            ) - left.centroid_frac.detach().to(
                device=crystal.periodic.lattice.device,
                dtype=torch.float64,
            )
            upper_bound = float(max_floor)
            if include_global_minimum:
                if left_index == right_index:
                    assert shortest_nonzero is not None
                    anchor_shift = shortest_nonzero
                else:
                    nearest = closest_lattice_shift(delta, crystal.periodic).shift
                    anchor_shift = tuple(int(value) for value in nearest.tolist())
                upper_bound = _candidate_distance_upper_bound(
                    crystal,
                    left,
                    right,
                    anchor_shift,
                )
            cutoff = (
                float(left.radius_A)
                + float(right.radius_A)
                + max(float(max_floor), float(upper_bound))
                + 1e-7
            )
            local = translations_within_cutoff(
                delta,
                crystal.periodic,
                cutoff_A=cutoff,
                tolerance_A=1e-8,
                max_candidates=max_periodic_candidates,
            )
            complete = complete and bool(local.complete)
            for shift, centroid_distance in zip(local.shifts, local.distances_A):
                shift_tuple = tuple(int(value) for value in shift.tolist())
                if left_index == right_index and shift_tuple == (0, 0, 0):
                    continue
                forward = (int(left_index), int(right_index), shift_tuple)
                reverse = (
                    int(right_index),
                    int(left_index),
                    tuple(-value for value in shift_tuple),
                )
                key = min(forward, reverse)
                if key in seen:
                    continue
                seen.add(key)
                max_translation_abs = max(
                    max_translation_abs,
                    max(abs(value) for value in shift_tuple),
                )
                records.append(
                    PhysicalInterComponentPair(
                        lower_bound_A=max(
                            0.0,
                            float(centroid_distance.item())
                            - float(left.radius_A)
                            - float(right.radius_A),
                        ),
                        max_floor_A=float(max_floor),
                        left_index=int(left_index),
                        right_index=int(right_index),
                        relative_shift=shift_tuple,
                    )
                )
    records.sort(
        key=lambda record: (
            record.lower_bound_A,
            record.left_index,
            record.right_index,
            record.relative_shift,
        )
    )
    return records, complete, max_translation_abs


def enumerate_physical_inter_component_pairs(
    crystal: CanonicalCrystal,
    *,
    cutoff_A: float,
    orbit_representative_left: bool = False,
    include_global_minimum: bool = False,
    max_periodic_candidates: int | None = None,
) -> PhysicalInterComponentSearch:
    """Enumerate complete unordered intermolecular image pairs relevant to a cutoff.

    ``include_global_minimum`` additionally retains the globally nearest
    component-image relation even when it lies beyond ``cutoff_A``.  This is
    useful for reporting a minimum intermolecular distance without weakening the
    cutoff used for clash-candidate enumeration. ``max_periodic_candidates`` is
    forwarded to every underlying integer-translation enumeration; exceeding it
    raises ``PeriodicSearchBudgetExceeded`` before the candidate box is allocated.
    """
    cutoff = float(cutoff_A)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff_A must be finite and positive")
    left_indices = None
    if orbit_representative_left:
        representatives: dict[int, int] = {}
        for index, image in enumerate(crystal.unit_components):
            representatives.setdefault(int(image.component_id), int(index))
        left_indices = tuple(representatives[key] for key in sorted(representatives))
    records, complete, max_translation_abs = _unordered_pair_records(
        crystal,
        config=ClashConfig(
            threshold_mode="fixed",
            distance_threshold_A=cutoff,
            supercell_radius=0,
        ),
        lookup=None,
        include_global_minimum=bool(include_global_minimum),
        left_indices=left_indices,
        max_periodic_candidates=max_periodic_candidates,
    )
    return PhysicalInterComponentSearch(
        pairs=tuple(
            record
            for record in records
            if include_global_minimum or record.lower_bound_A <= cutoff + 1e-6
        ),
        complete=bool(complete),
        max_translation_abs=int(max_translation_abs),
    )


@lru_cache(maxsize=4096)
def _eligible_intramolecular_pairs(
    topology: ComponentTopology,
) -> tuple[tuple[int, int], ...]:
    atom_count = len(topology.atomic_numbers)
    adjacency: list[set[int]] = [set() for _ in range(atom_count)]
    for left, right, _ in topology.edges:
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    eligible: list[tuple[int, int]] = []
    for start in range(atom_count):
        distances = [-1] * atom_count
        distances[start] = 0
        queue = [start]
        for left in queue:
            for right in sorted(adjacency[left]):
                if distances[right] >= 0:
                    continue
                distances[right] = distances[left] + 1
                queue.append(right)
        for right in range(start + 1, atom_count):
            if distances[right] < 0 or distances[right] > 3:
                eligible.append((start, right))
    return tuple(eligible)


@lru_cache(maxsize=4096)
def _topology_is_connected(topology: ComponentTopology) -> bool:
    atom_count = len(topology.atomic_numbers)
    if atom_count <= 1:
        return True
    adjacency: list[set[int]] = [set() for _ in range(atom_count)]
    for left, right, _ in topology.edges:
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    visited = {0}
    queue = [0]
    for left in queue:
        for right in sorted(adjacency[left]):
            if right in visited:
                continue
            visited.add(right)
            queue.append(right)
    return len(visited) == atom_count


def _finite_minimum(*values: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return min(finite) if finite else _nan()


def _normalization_atom_count(crystal: CanonicalCrystal) -> int:
    if crystal.topologies:
        return max(
            1, sum(len(topology.atomic_numbers) for topology in crystal.topologies)
        )
    indices = {
        int(index)
        for component in crystal.unit_components
        for index in component.base_atom_index.detach().cpu().tolist()
    }
    return max(1, len(indices))


def evaluate_clashes(
    asu,
    config: ClashConfig | None = None,
    *,
    differentiable: bool = False,
) -> ClashResult:
    config = config or ClashConfig()
    if isinstance(asu, CanonicalCrystal):
        crystal = asu
    else:
        topologies = build_component_topologies(asu, atom_scope=config.atom_scope)
        crystal = prepare_canonical_crystal(
            asu,
            atom_scope=config.atom_scope,
            max_symops=config.max_symops,
            supercell_radius=0,
            dedup_tolerance_A=config.dedup_tolerance_A,
            topologies=topologies,
        )
    if not crystal.unit_components:
        return _empty_result(asu, config=config)

    lookup = None
    if config.threshold_mode == "vdw_offset":
        lookup = _vdw_offset_lookup(float(config.vdw_offset_A))

    pair_records, neighborhood_complete, max_translation_abs = _unordered_pair_records(
        crystal,
        config=config,
        lookup=lookup,
    )

    intermolecular_count = 0
    intramolecular_count = 0
    intermolecular_eligible = 0
    intramolecular_eligible = 0
    intermolecular_source_pairs = 0
    intramolecular_source_components = 0
    intermolecular_minimum = math.inf
    intramolecular_minimum = math.inf
    intermolecular_maximum_penetration = 0.0
    intramolecular_maximum_penetration = 0.0
    radius_source_counts = [0, 0]
    thresholds = _ThresholdAccumulator()
    penalty_terms: list[Tensor] = []
    for record in pair_records:
        left = crystal.unit_components[record.left_index]
        right = crystal.unit_components[record.right_index]
        right_cart = _translated_cartesian(
            right,
            record.relative_shift,
            crystal.lattice,
        )
        dist = torch.cdist(left.cart_coords.float(), right_cart.float())
        if dist.numel() == 0:
            continue
        intermolecular_minimum = min(
            intermolecular_minimum,
            float(dist.detach().min().item()),
        )
        if record.lower_bound_A > record.max_floor_A + 1e-6:
            continue
        threshold = _cross_thresholds(
            left.atom_types,
            right.atom_types,
            config=config,
            lookup=lookup,
            dtype=dist.dtype,
            device=dist.device,
        )
        intermolecular_source_pairs += 1
        intermolecular_eligible += int(dist.numel())
        thresholds.add(threshold)
        if lookup is not None:
            source_codes = _cross_radius_sources(
                left.atom_types,
                right.atom_types,
                lookup,
            )
            for source_code in range(2):
                radius_source_counts[source_code] += int(
                    (source_codes == source_code).sum().item()
                )
        penetration = torch.relu(threshold - dist)
        penalty_terms.append(penetration.square().sum())
        intermolecular_count += int((penetration.detach() > 0.0).sum().item())
        intermolecular_maximum_penetration = max(
            intermolecular_maximum_penetration,
            float(penetration.detach().max().item()),
        )

    topology_by_component = {
        int(topology.component_id): topology for topology in crystal.topologies
    }
    topology_complete = True
    for image in crystal.unit_components:
        topology = topology_by_component.get(int(image.component_id))
        if (
            topology is None
            or len(topology.atomic_numbers) != int(image.atom_types.numel())
            or not _topology_is_connected(topology)
        ):
            topology_complete = False
            continue
        pairs = _eligible_intramolecular_pairs(topology)
        if not pairs:
            continue
        left_indices = torch.tensor(
            [left for left, _ in pairs],
            dtype=torch.long,
            device=image.cart_coords.device,
        )
        right_indices = torch.tensor(
            [right for _, right in pairs],
            dtype=torch.long,
            device=image.cart_coords.device,
        )
        distances = torch.linalg.norm(
            image.cart_coords[left_indices].float()
            - image.cart_coords[right_indices].float(),
            dim=-1,
        )
        if distances.numel() == 0:
            continue
        intramolecular_source_components += 1
        intramolecular_eligible += int(distances.numel())
        intramolecular_minimum = min(
            intramolecular_minimum,
            float(distances.detach().min().item()),
        )
        threshold = _paired_thresholds(
            image.atom_types[left_indices],
            image.atom_types[right_indices],
            config=config,
            lookup=lookup,
            dtype=distances.dtype,
            device=distances.device,
        )
        thresholds.add(threshold)
        if lookup is not None:
            source_codes = _paired_radius_sources(
                image.atom_types[left_indices],
                image.atom_types[right_indices],
                lookup,
            )
            for source_code in range(2):
                radius_source_counts[source_code] += int(
                    (source_codes == source_code).sum().item()
                )
        penetration = torch.relu(threshold - distances)
        penalty_terms.append(penetration.square().sum())
        intramolecular_count += int((penetration.detach() > 0.0).sum().item())
        intramolecular_maximum_penetration = max(
            intramolecular_maximum_penetration,
            float(penetration.detach().max().item()),
        )

    if thresholds.count == 0:
        if config.threshold_mode == "fixed":
            thresholds.minimum = thresholds.maximum = thresholds.total = float(
                config.distance_threshold_A
            )
            thresholds.count = 1
        else:
            assert lookup is not None
            present_types = torch.unique(
                torch.cat(
                    [
                        component.atom_types.long()
                        for component in crystal.unit_components
                    ],
                    dim=0,
                )
            ).clamp(0, lookup.thresholds.shape[0] - 1)
            present = lookup.thresholds[
                present_types[:, None],
                present_types[None, :],
            ]
            thresholds.add(present)

    if penalty_terms:
        penalty = torch.stack(penalty_terms).sum() / float(
            _normalization_atom_count(crystal)
        )
    else:
        penalty = _zero_penalty(asu)

    pair_count = int(intermolecular_count + intramolecular_count)
    maximum_penetration = max(
        intermolecular_maximum_penetration,
        intramolecular_maximum_penetration,
    )
    return ClashResult(
        clash=pair_count > 0,
        pair_count=pair_count,
        min_distance_A=_finite_minimum(
            intermolecular_minimum,
            intramolecular_minimum,
        ),
        threshold_min_A=float(thresholds.minimum) if thresholds.count else _nan(),
        threshold_max_A=float(thresholds.maximum) if thresholds.count else _nan(),
        threshold_mean_A=(
            float(thresholds.total / thresholds.count) if thresholds.count else _nan()
        ),
        max_penetration_A=float(maximum_penetration),
        penalty=penalty,
        intermolecular_clash=intermolecular_count > 0,
        intramolecular_clash=intramolecular_count > 0,
        intermolecular_pair_count=int(intermolecular_count),
        intramolecular_pair_count=int(intramolecular_count),
        intermolecular_eligible_pair_count=int(intermolecular_eligible),
        intramolecular_eligible_pair_count=int(intramolecular_eligible),
        intermolecular_min_distance_A=(
            float(intermolecular_minimum)
            if math.isfinite(intermolecular_minimum)
            else _nan()
        ),
        intramolecular_min_distance_A=(
            float(intramolecular_minimum)
            if math.isfinite(intramolecular_minimum)
            else _nan()
        ),
        intermolecular_max_penetration_A=float(intermolecular_maximum_penetration),
        intramolecular_max_penetration_A=float(intramolecular_maximum_penetration),
        intermolecular_source_pair_count=int(intermolecular_source_pairs),
        intramolecular_source_component_count=int(intramolecular_source_components),
        unit_component_count=len(crystal.unit_components),
        neighborhood_complete=bool(neighborhood_complete),
        topology_complete=bool(topology_complete),
        max_translation_abs=int(max_translation_abs),
        vdw_offset_A=(
            float(config.vdw_offset_A)
            if config.threshold_mode == "vdw_offset"
            else _nan()
        ),
        vdw_known_radius_pair_count=int(radius_source_counts[0]),
        vdw_unknown_radius_pair_count=int(radius_source_counts[1]),
    )


def compute_clash_metrics(
    asu,
    *,
    atom_scope: str = "heavy",
    max_symops: int | None = None,
    supercell_radius: int = 1,
    threshold_mode: str = "vdw_offset",
    distance_threshold_A: float = 1.2,
    vdw_offset_A: float = DEFAULT_VDW_OFFSET_A,
    dedup_tolerance_A: float = 1e-3,
) -> dict[str, float]:
    config = ClashConfig(
        atom_scope=atom_scope,
        max_symops=max_symops,
        supercell_radius=supercell_radius,
        threshold_mode=threshold_mode,
        distance_threshold_A=distance_threshold_A,
        vdw_offset_A=vdw_offset_A,
        dedup_tolerance_A=dedup_tolerance_A,
    )
    return evaluate_clashes(asu, config=config).to_metrics()


__all__ = [
    "DEFAULT_VDW_OFFSET_A",
    "ClashConfig",
    "ClashResult",
    "PhysicalInterComponentPair",
    "PhysicalInterComponentSearch",
    "compute_clash_metrics",
    "enumerate_physical_inter_component_pairs",
    "evaluate_clashes",
    "vdw_pair_threshold",
]
