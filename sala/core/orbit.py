"""Static general-position orbit plans for full-cell computation.

Only ASU Cartesian coordinates are independent generative variables.  An
``OrbitPlan`` deterministically expands them with the source-CIF symmetry
operations and pulls arbitrary full-cell tangent vectors back to the ASU by
orbit averaging.  Version 1 deliberately rejects special-position sites:
their stabilizer and molecular atom-permutation action must be supplied by a
future data schema and must never be inferred from noisy coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from sala.core.lattice import batched_inverse_3x3
from sala.core.symmetry import extract_explicit_symmetry_ops
from sala.core.types import AsuBatch


class UnsupportedSpecialPositionError(ValueError):
    """Raised when the v1 general-position contract is violated."""


@dataclass(frozen=True)
class FullCellGeometry:
    frac: Tensor
    cart: Tensor
    valid: Tensor


@dataclass(frozen=True)
class OrbitPlan:
    """Padded, operation-order explicit full-cell expansion metadata."""

    full_to_asu: Tensor
    full_symop_index: Tensor
    full_base_component: Tensor
    full_component_instance: Tensor
    full_component_type: Tensor
    full_valid: Tensor
    rotations: Tensor
    inverse_rotations: Tensor
    translations: Tensor
    symop_valid: Tensor
    atom_batch: Tensor
    atom_ptr: Tensor
    full_counts: Tensor
    bond_type: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.full_valid.shape[0])

    @property
    def max_full_atoms(self) -> int:
        return int(self.full_valid.shape[1])

    @property
    def num_asu_atoms(self) -> int:
        return int(self.atom_batch.shape[0])

    @property
    def token_valid(self) -> Tensor:
        lattice_valid = torch.ones(
            (self.batch_size, 3), device=self.full_valid.device, dtype=torch.bool
        )
        return torch.cat((lattice_valid, self.full_valid), dim=-1)

    def to(self, device: torch.device | str) -> "OrbitPlan":
        values = {
            name: getattr(self, name).to(device)
            for name in self.__dataclass_fields__
        }
        return OrbitPlan(**values)

    def _validate_state(self, asu_cart: Tensor, lattice: Tensor) -> None:
        if asu_cart.shape != (self.num_asu_atoms, 3):
            raise ValueError(
                f"asu_cart must have shape ({self.num_asu_atoms}, 3), got "
                f"{tuple(asu_cart.shape)}"
            )
        if lattice.shape != (self.batch_size, 3, 3):
            raise ValueError(
                f"lattice must have shape ({self.batch_size}, 3, 3), got "
                f"{tuple(lattice.shape)}"
            )

    @staticmethod
    def _resolve_inverse(
        lattice: Tensor, lattice_inverse: Tensor | None
    ) -> Tensor:
        """Reuse a caller-supplied inverse, else invert analytically.

        ``torch.linalg.inv`` synchronizes on CUDA to inspect its factorization
        info, so the cofactor form is used for these 3x3 lattices.
        """

        if lattice_inverse is None:
            return batched_inverse_3x3(lattice)
        return lattice_inverse.to(device=lattice.device, dtype=lattice.dtype)

    @staticmethod
    def _row_vector_transform(vectors: Tensor, matrices: Tensor) -> Tensor:
        """Row-vector product ``v @ M`` for per-element matrices.

        A broadcast multiply-reduce keeps this in one fused kernel; the
        equivalent ``bmm`` would launch one tiny 1x3 @ 3x3 GEMM per element.
        """

        return (vectors.unsqueeze(-1) * matrices).sum(dim=-2)

    def _asu_fractional(self, asu_cart: Tensor, inverse: Tensor) -> Tensor:
        atom_batch = self.atom_batch.to(device=asu_cart.device)
        return self._row_vector_transform(
            asu_cart, inverse.index_select(0, atom_batch)
        )

    def _gather_source(self, asu_values: Tensor) -> Tensor:
        source = self.full_to_asu.to(device=asu_values.device).clamp_min(0)
        return asu_values.index_select(0, source.reshape(-1)).reshape(
            self.batch_size, self.max_full_atoms, 3
        )

    def _operation_tensors(
        self, device: torch.device, dtype: torch.dtype, *, inverse: bool = False
    ) -> tuple[Tensor, Tensor]:
        op_index = self.full_symop_index.to(device=device).clamp_min(0)
        graph_index = torch.arange(self.batch_size, device=device)[:, None]
        source = self.inverse_rotations if inverse else self.rotations
        rotation = source.to(device=device, dtype=dtype)[graph_index, op_index]
        translation = self.translations.to(device=device, dtype=dtype)[
            graph_index, op_index
        ]
        return rotation, translation

    @staticmethod
    def _apply_rotation(rotation: Tensor, vectors: Tensor) -> Tensor:
        """Column-vector product ``R v`` fused over the padded cell."""

        return (rotation * vectors.unsqueeze(-2)).sum(dim=-1)

    def expand_coordinates(
        self,
        asu_cart: Tensor,
        lattice: Tensor,
        lattice_inverse: Tensor | None = None,
    ) -> FullCellGeometry:
        """Expand ASU coordinates without wrapping or target-derived shifts."""

        self._validate_state(asu_cart, lattice)
        work_dtype = torch.float32
        x = asu_cart.to(work_dtype)
        lat = lattice.to(device=x.device, dtype=work_dtype)
        inverse = self._resolve_inverse(lat, lattice_inverse)
        frac_asu = self._asu_fractional(x, inverse)
        source_frac = self._gather_source(frac_asu)
        rotation, translation = self._operation_tensors(x.device, work_dtype)
        full_frac = self._apply_rotation(rotation, source_frac) + translation
        # cart = frac @ lattice for row vectors.  One batched GEMM replaces
        # batch_size * max_full_atoms independent 1x3 @ 3x3 matmuls.
        full_cart = torch.bmm(full_frac, lat)
        valid = self.full_valid.to(device=x.device)
        mask = valid.unsqueeze(-1)
        return FullCellGeometry(
            frac=(full_frac * mask).to(dtype=asu_cart.dtype),
            cart=(full_cart * mask).to(dtype=asu_cart.dtype),
            valid=valid,
        )

    def expand_vectors(
        self,
        asu_vectors: Tensor,
        lattice: Tensor,
        lattice_inverse: Tensor | None = None,
    ) -> Tensor:
        """Apply each space-group tangent action to ASU Cartesian vectors."""

        self._validate_state(asu_vectors, lattice)
        work_dtype = torch.float32
        vec = asu_vectors.to(work_dtype)
        lat = lattice.to(device=vec.device, dtype=work_dtype)
        inverse = self._resolve_inverse(lat, lattice_inverse)
        frac_vec = self._asu_fractional(vec, inverse)
        source_vec = self._gather_source(frac_vec)
        rotation, _ = self._operation_tensors(vec.device, work_dtype)
        # Tangents carry the rotation but not the translation part.
        full_frac_vec = self._apply_rotation(rotation, source_vec)
        full_cart_vec = torch.bmm(full_frac_vec, lat)
        valid = self.full_valid.to(device=vec.device)
        return (full_cart_vec * valid.unsqueeze(-1)).to(dtype=asu_vectors.dtype)

    def pullback_vectors(
        self,
        full_cart_vectors: Tensor,
        lattice: Tensor,
        lattice_inverse: Tensor | None = None,
    ) -> Tensor:
        """Pull full-cell Cartesian tangents into ASU frames and average orbits."""

        expected = (self.batch_size, self.max_full_atoms, 3)
        if full_cart_vectors.shape != expected:
            raise ValueError(
                f"full_cart_vectors must have shape {expected}, got "
                f"{tuple(full_cart_vectors.shape)}"
            )
        if lattice.shape != (self.batch_size, 3, 3):
            raise ValueError("lattice shape does not match OrbitPlan")
        work_dtype = torch.float32
        vec = full_cart_vectors.to(work_dtype)
        lat = lattice.to(device=vec.device, dtype=work_dtype)
        inverse = self._resolve_inverse(lat, lattice_inverse)
        frac_vec = torch.bmm(vec, inverse)
        inverse_rotation, _ = self._operation_tensors(
            vec.device, work_dtype, inverse=True
        )
        candidate_frac = self._apply_rotation(inverse_rotation, frac_vec)

        # Boolean masking would need the selected element count on the host and
        # therefore synchronize on every call.  Padded images instead carry zero
        # weight into both accumulators, which leaves each orbit mean unchanged
        # while keeping the whole reduction on device.
        valid = self.full_valid.to(device=vec.device)
        weight = valid.to(dtype=work_dtype).unsqueeze(-1)
        target = self.full_to_asu.to(device=vec.device).clamp_min(0).reshape(-1)
        summed = torch.zeros(
            (self.num_asu_atoms, 3), device=vec.device, dtype=work_dtype
        )
        counts = torch.zeros(
            (self.num_asu_atoms, 1), device=vec.device, dtype=work_dtype
        )
        summed.index_add_(0, target, (candidate_frac * weight).reshape(-1, 3))
        counts.index_add_(0, target, weight.reshape(-1, 1))
        mean_frac = summed / counts.clamp_min(1.0)
        atom_lattice = lat.index_select(
            0, self.atom_batch.to(device=vec.device)
        )
        asu_cart = self._row_vector_transform(mean_frac, atom_lattice)
        return asu_cart.to(dtype=full_cart_vectors.dtype)

    def project_full_vectors(
        self,
        full_cart_vectors: Tensor,
        lattice: Tensor,
        lattice_inverse: Tensor | None = None,
    ) -> Tensor:
        return self.expand_vectors(
            self.pullback_vectors(full_cart_vectors, lattice, lattice_inverse),
            lattice,
            lattice_inverse,
        )


def _operation_set(batch: AsuBatch, graph_idx: int, reference: Tensor) -> Tensor:
    explicit = extract_explicit_symmetry_ops(batch, graph_idx)
    if explicit is None:
        raise ValueError(
            "OrbitPlan requires explicit Hall-setting symmetry_ops for every graph"
        )
    return explicit.to(device=reference.device, dtype=reference.dtype)


def _assert_general_position(
    frac: Tensor,
    operations: Tensor,
    *,
    tolerance: float,
    graph_idx: int,
) -> None:
    rotation = operations[:, :3, :3]
    translation = operations[:, :3, 3]
    transformed = torch.einsum("gij,nj->gni", rotation, frac) + translation[:, None]
    transformed = transformed - torch.floor(transformed)
    num_ops = int(operations.shape[0])
    for atom_idx in range(int(frac.shape[0])):
        positions = transformed[:, atom_idx]
        delta = positions[:, None] - positions[None, :]
        delta = delta - torch.round(delta)
        duplicate = delta.abs().amax(dim=-1) <= float(tolerance)
        duplicate.fill_diagonal_(False)
        if bool(duplicate.any().detach().cpu()):
            raise UnsupportedSpecialPositionError(
                "FullCell OrbitPlan v1 accepts general-position sites only: "
                f"graph {graph_idx}, ASU atom {atom_idx} has orbit multiplicity "
                f"smaller than {num_ops}. Regenerate data with stabilizer/permutation "
                "metadata or filter this sample explicitly."
            )
    # General-position multiplicity per site is not sufficient: an invalid ASU
    # may contain two different representatives of the same crystallographic
    # orbit.  Such intersecting orbits would create duplicate full-cell tokens.
    positions = transformed.permute(1, 0, 2).reshape(-1, 3).detach().cpu()
    owners = torch.arange(int(frac.shape[0]), dtype=torch.long).repeat_interleave(num_ops)
    # Periodic spatial hashing avoids an O((N_asu*|G|)^2) allocation for large
    # rejected CIFs.  A bin is at least tolerance wide, so 27 neighbor bins are
    # sufficient for the L-infinity duplicate criterion.
    num_bins = max(1, int(1.0 / float(tolerance)))
    bin_index = torch.floor(positions * num_bins).long().remainder(num_bins)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    offsets = (-1, 0, 1)
    for point_index, raw_key in enumerate(bin_index.tolist()):
        owner = int(owners[point_index].item())
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    key = (
                        (raw_key[0] + dx) % num_bins,
                        (raw_key[1] + dy) % num_bins,
                        (raw_key[2] + dz) % num_bins,
                    )
                    for candidate in buckets.get(key, ()):
                        other_owner = int(owners[candidate].item())
                        if owner == other_owner:
                            continue
                        delta = (positions[point_index] - positions[candidate]).abs()
                        delta = torch.minimum(delta, 1.0 - delta)
                        if bool((delta.max() <= float(tolerance)).item()):
                            raise UnsupportedSpecialPositionError(
                                "FullCell OrbitPlan v1 requires one ASU representative "
                                f"per orbit: graph {graph_idx}, ASU atoms {owner} and "
                                f"{other_owner} have intersecting symmetry orbits. "
                                "Regenerate or filter this sample."
                            )
        buckets.setdefault(tuple(raw_key), []).append(point_index)


def validate_general_position_orbits(
    frac_coords: Tensor,
    symmetry_ops: Tensor,
    *,
    tolerance: float = 1.0e-5,
) -> None:
    """Public preprocessing boundary for the v1 general-position policy."""

    _assert_general_position(
        frac_coords,
        symmetry_ops,
        tolerance=tolerance,
        graph_idx=0,
    )


def build_orbit_plan(
    batch: AsuBatch,
    *,
    validate_general_position: bool = True,
    special_position_tolerance: float = 1.0e-5,
) -> OrbitPlan:
    """Create a target-independent ``N_asu * |G|`` full-cell plan."""

    device = batch.atom_types.device
    reference = batch.frac_coords_gt
    batch_size = int(batch.batch_size)
    graph_ops: list[Tensor] = []
    full_counts: list[int] = []
    local_components: list[Tensor] = []
    local_component_types: list[Tensor] = []
    atom_ranges: list[tuple[int, int]] = []
    for graph_idx in range(batch_size):
        start = int(batch.ptr[graph_idx].detach().cpu().item())
        end = int(batch.ptr[graph_idx + 1].detach().cpu().item())
        atom_ranges.append((start, end))
        ops = _operation_set(batch, graph_idx, reference)
        num_ops = int(ops.shape[0])
        declared_ops = getattr(batch, "n_symops", None)
        if declared_ops is not None:
            declared = int(declared_ops[graph_idx].detach().cpu().item())
            if declared != num_ops:
                raise ValueError(
                    "OrbitPlan symmetry count disagrees with batch metadata: "
                    f"graph {graph_idx}, n_symops={declared}, actual={num_ops}. "
                    "Regenerate the full-cell data cache."
                )
        expected_full = (end - start) * num_ops
        declared_full = getattr(batch, "n_full_atoms", None)
        if declared_full is not None:
            declared = int(declared_full[graph_idx].detach().cpu().item())
            if declared != expected_full:
                raise ValueError(
                    "OrbitPlan full-cell count disagrees with batch metadata: "
                    f"graph {graph_idx}, n_full_atoms={declared}, "
                    f"expected={expected_full}. Regenerate the full-cell data cache."
                )
        if validate_general_position:
            _assert_general_position(
                reference[start:end],
                ops,
                tolerance=special_position_tolerance,
                graph_idx=graph_idx,
            )
        graph_ops.append(ops)
        full_counts.append(expected_full)
        component_values, comp_inverse = torch.unique(
            batch.component_id[start:end].long(), sorted=True, return_inverse=True
        )
        local_components.append(comp_inverse)
        local_component_types.append(
            batch.component_type_id.long().index_select(0, component_values)
        )

    max_ops = max((int(op.shape[0]) for op in graph_ops), default=1)
    max_full = max(full_counts, default=0)
    rotations = torch.eye(3, device=device, dtype=reference.dtype).view(1, 1, 3, 3)
    rotations = rotations.expand(batch_size, max_ops, -1, -1).clone()
    inverse_rotations = rotations.clone()
    translations = torch.zeros(
        (batch_size, max_ops, 3), device=device, dtype=reference.dtype
    )
    symop_valid = torch.zeros((batch_size, max_ops), device=device, dtype=torch.bool)
    full_to_asu = torch.zeros((batch_size, max_full), device=device, dtype=torch.long)
    full_symop = torch.zeros_like(full_to_asu)
    full_base_component = torch.zeros_like(full_to_asu)
    full_component_instance = torch.zeros_like(full_to_asu)
    full_component_type = torch.zeros_like(full_to_asu)
    full_valid = torch.zeros((batch_size, max_full), device=device, dtype=torch.bool)
    bond_type = torch.zeros((batch_size, max_full, max_full), device=device, dtype=torch.long)

    # Map global ASU edge indices to per-graph matrices once.
    edge_index = batch.edge_index_conv.long()
    edge_attr = batch.edge_attr_conv.long()
    for graph_idx, ((start, end), ops, comp, component_type) in enumerate(
        zip(atom_ranges, graph_ops, local_components, local_component_types)
    ):
        num_atoms = end - start
        num_ops = int(ops.shape[0])
        count = num_atoms * num_ops
        rotations[graph_idx, :num_ops] = ops[:, :3, :3]
        inverse_rotations[graph_idx, :num_ops] = torch.linalg.inv(
            ops[:, :3, :3].float()
        ).to(dtype=reference.dtype)
        translations[graph_idx, :num_ops] = ops[:, :3, 3]
        symop_valid[graph_idx, :num_ops] = True
        local_atom = torch.arange(start, end, device=device, dtype=torch.long)
        full_to_asu[graph_idx, :count] = local_atom.repeat(num_ops)
        full_symop[graph_idx, :count] = torch.arange(
            num_ops, device=device, dtype=torch.long
        ).repeat_interleave(num_atoms)
        num_components = max(int(comp.max().detach().cpu().item()) + 1, 1) if comp.numel() else 1
        full_base_component[graph_idx, :count] = comp.repeat(num_ops)
        full_component_instance[graph_idx, :count] = (
            full_base_component[graph_idx, :count]
            + torch.arange(num_ops, device=device).repeat_interleave(num_atoms)
            * num_components
        )
        full_component_type[graph_idx, :count] = component_type.index_select(
            0, comp
        ).repeat(num_ops)
        full_valid[graph_idx, :count] = True
        if edge_index.numel():
            in_graph = (
                (edge_index[0] >= start)
                & (edge_index[0] < end)
                & (edge_index[1] >= start)
                & (edge_index[1] < end)
            )
            local_edge = edge_index[:, in_graph] - start
            local_type = edge_attr[in_graph]
            for op_idx in range(num_ops):
                offset = op_idx * num_atoms
                bond_type[
                    graph_idx,
                    local_edge[0] + offset,
                    local_edge[1] + offset,
                ] = local_type.clamp_min(0) + 1

    return OrbitPlan(
        full_to_asu=full_to_asu,
        full_symop_index=full_symop,
        full_base_component=full_base_component,
        full_component_instance=full_component_instance,
        full_component_type=full_component_type,
        full_valid=full_valid,
        rotations=rotations,
        inverse_rotations=inverse_rotations,
        translations=translations,
        symop_valid=symop_valid,
        atom_batch=batch.batch.long(),
        atom_ptr=batch.ptr.long(),
        full_counts=torch.tensor(full_counts, device=device, dtype=torch.long),
        bond_type=bond_type,
    )


__all__ = [
    "FullCellGeometry",
    "OrbitPlan",
    "UnsupportedSpecialPositionError",
    "build_orbit_plan",
    "validate_general_position_orbits",
]
