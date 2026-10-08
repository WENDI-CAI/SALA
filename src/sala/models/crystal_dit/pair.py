"""Full token-pair bias with periodic atom geometry for Crystal-DiT."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from sala.core.lattice import batched_inverse_3x3
from sala.core.orbit import OrbitPlan
from sala.models.common import GaussianRBF
from sala.models.crystal_dit.geometry import CrystalGeometryState


SAME_COMPONENT_INSTANCE = 0
SAME_BASE_COMPONENT_OTHER_INSTANCE = 1
SAME_COMPONENT_TYPE_OTHER_BASE = 2
DIFFERENT_COMPONENT_TYPE = 3
NUM_COMPONENT_RELATIONS = 4


def component_relation_index(
    same_component_type: Tensor,
    same_base_component: Tensor,
    same_component_instance: Tensor,
) -> Tensor:
    relation = torch.full_like(
        same_base_component, DIFFERENT_COMPONENT_TYPE, dtype=torch.long
    )
    relation = torch.where(
        same_component_type,
        torch.full_like(relation, SAME_COMPONENT_TYPE_OTHER_BASE),
        relation,
    )
    relation = torch.where(
        same_base_component,
        torch.full_like(relation, SAME_BASE_COMPONENT_OTHER_INSTANCE),
        relation,
    )
    return torch.where(
        same_component_instance,
        torch.full_like(relation, SAME_COMPONENT_INSTANCE),
        relation,
    )


def quintic_cutoff(distance: Tensor, cutoff: float) -> Tensor:
    cutoff = float(cutoff)
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    ratio = distance / cutoff
    inside = ratio < 1.0
    ratio = ratio.clamp(min=0.0, max=1.0)
    envelope = 1.0 - 10.0 * ratio**3 + 15.0 * ratio**4 - 6.0 * ratio**5
    return torch.where(inside, envelope, torch.zeros_like(envelope))


class PeriodicDisplacement(nn.Module):
    """Closest periodic displacement between padded full-cell atom pairs.

    ``forward`` runs one batched search over the whole padded pair set: the image
    index is chosen under ``no_grad`` and the winning displacement is then rebuilt
    differentiably, so no candidate tensor enters the backward graph and there is
    no per-graph Python loop.  Translation grids are cached per radius triple.

    Two radius policies are supported.  With ``cuda_dynamic_image_radius`` the
    required per-axis radius is derived from the lattice and read back once as
    three integers.  Without it, the fixed ``cuda_static_image_radius`` grid is
    used and its sufficiency is asserted on device, which keeps the call entirely
    free of host synchronization.

    ``_one_graph`` is retained as the per-graph reference search.
    """

    def __init__(
        self,
        *,
        max_image_candidates: int = 65536,
        pair_chunk_size: int = 1024,
        image_chunk_size: int = 256,
        cuda_static_image_radius: int = 2,
        cuda_dynamic_image_radius: bool = False,
        pair_image_work_budget: int = 16777216,
    ) -> None:
        super().__init__()
        self.max_image_candidates = int(max_image_candidates)
        self.pair_chunk_size = max(int(pair_chunk_size), 1)
        self.image_chunk_size = max(int(image_chunk_size), 1)
        self.cuda_static_image_radius = max(int(cuda_static_image_radius), 1)
        self.cuda_dynamic_image_radius = bool(cuda_dynamic_image_radius)
        self.pair_image_work_budget = max(int(pair_image_work_budget), 1)
        if self.max_image_candidates < 27:
            raise ValueError("max_image_candidates must be at least 27")
        self._offset_cache: dict[tuple, Tensor] = {}

    def _offsets(
        self,
        radii: tuple[int, int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        key = (radii, device.type, device.index, str(dtype))
        cached = self._offset_cache.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached
        candidate_count = 1
        for value in radii:
            candidate_count *= 2 * int(value) + 1
        if candidate_count > self.max_image_candidates:
            raise ValueError(
                "Periodic closest-image search requires "
                f"{candidate_count} candidates for radii={radii}, exceeding "
                f"max_image_candidates={self.max_image_candidates}; the lattice "
                "is too ill-conditioned for the configured work budget"
            )
        axes = [
            torch.arange(-r, r + 1, device=device, dtype=dtype) for r in radii
        ]
        offsets = torch.cartesian_prod(*axes)
        if offsets.ndim == 1:
            offsets = offsets.unsqueeze(-1)
        self._offset_cache[key] = offsets
        return offsets

    def _one_graph(
        self,
        centered: Tensor,
        lattice: Tensor,
        radii: tuple[int, int, int],
    ) -> tuple[Tensor, Tensor]:
        n_atoms = int(centered.shape[0])
        if n_atoms == 0:
            empty = centered.new_zeros((0, 0, 3))
            return empty, empty
        offsets = self._offsets(
            radii, device=centered.device, dtype=centered.dtype
        )
        candidate_count = int(offsets.shape[0])
        flat = centered.reshape(-1, 3)
        best_frac_parts: list[Tensor] = []
        best_cart_parts: list[Tensor] = []
        for start in range(0, int(flat.shape[0]), self.pair_chunk_size):
            part = flat[start : start + self.pair_chunk_size]
            best_squared = torch.full(
                (part.shape[0],),
                torch.inf,
                device=centered.device,
                dtype=centered.dtype,
            )
            best_frac = torch.zeros_like(part)
            best_cart = torch.zeros_like(part)
            for image_start in range(0, candidate_count, self.image_chunk_size):
                image_offsets = offsets[
                    image_start : image_start + self.image_chunk_size
                ]
                candidates = part[:, None, :] + image_offsets[None, :, :]
                candidate_cart = candidates @ lattice
                squared = candidate_cart.square().sum(dim=-1)
                chunk_squared, chunk_best = squared.min(dim=-1)
                row = torch.arange(part.shape[0], device=centered.device)
                improve = chunk_squared < best_squared
                best_squared = torch.where(improve, chunk_squared, best_squared)
                best_frac = torch.where(
                    improve.unsqueeze(-1), candidates[row, chunk_best], best_frac
                )
                best_cart = torch.where(
                    improve.unsqueeze(-1), candidate_cart[row, chunk_best], best_cart
                )
            best_frac_parts.append(best_frac)
            best_cart_parts.append(best_cart)
        return (
            torch.cat(best_frac_parts, dim=0).reshape(n_atoms, n_atoms, 3),
            torch.cat(best_cart_parts, dim=0).reshape(n_atoms, n_atoms, 3),
        )

    def _closest_offsets(
        self,
        centered: Tensor,
        lattice: Tensor,
        radii: tuple[int, int, int],
    ) -> Tensor:
        """Return the integer lattice translation closest to each centered pair.

        Selecting an image is a discrete, piecewise-constant operation, so it is
        evaluated without autograd; the caller rebuilds the chosen displacement
        differentiably.  That keeps every candidate tensor out of the backward
        graph and lets the search use one ``argmin`` over all images per chunk
        instead of an iterative ``where`` chain.
        """

        batch_size, num_atoms = centered.shape[:2]
        offsets = self._offsets(
            radii, device=centered.device, dtype=centered.dtype
        )
        image_count = int(offsets.shape[0])
        flat = centered.reshape(-1, 3)
        pair_count = int(flat.shape[0])
        graph_index = torch.arange(
            batch_size, device=centered.device, dtype=torch.long
        ).repeat_interleave(num_atoms * num_atoms)
        # Chunk on the pair axis only, so the number of kernel launches scales
        # with the work budget rather than with the pair count.
        chunk = max(self.pair_image_work_budget // max(image_count, 1), 1)
        best = torch.empty_like(flat)
        for start in range(0, pair_count, chunk):
            stop = min(start + chunk, pair_count)
            part = flat[start:stop]
            part_lattice = lattice.index_select(0, graph_index[start:stop])
            candidates = part[:, None, :] + offsets[None, :, :]
            candidate_cart = torch.einsum(
                "pmd,pdk->pmk", candidates, part_lattice
            )
            winner = candidate_cart.square().sum(dim=-1).argmin(dim=-1)
            best[start:stop] = offsets.index_select(0, winner)
        return best.reshape(batch_size, num_atoms, num_atoms, 3)

    def _search_radii(
        self,
        centered: Tensor,
        lattice: Tensor,
        inverse: Tensor,
        pair_valid: Tensor,
        search_cutoff_A: float | None,
    ) -> tuple[int, int, int]:
        """Per-axis image radius that provably contains every relevant image.

        With row-vector fractional coordinates, a displacement of Cartesian
        length ``d`` has ``|f_k| <= d * ||A^{-1}[:, k]||``.  A centered pair
        already satisfies ``|f_k| <= 0.5``, so any translation ``n`` reaching a
        Cartesian length below ``d`` obeys ``|n_k| <= d * ||A^{-1}[:, k]|| + 0.5``.

        When ``search_cutoff_A`` is given, ``d`` is that cutoff: the pair encoder
        multiplies every geometry term by an envelope that is exactly zero at and
        beyond it, so images that can only produce longer displacements cannot
        change the bias.  Restricting the grid this way is exact for the encoder
        while making the radius independent of the largest pair in the cell.
        Without a cutoff the legacy bound from the largest pair distance is kept.
        """

        reciprocal_norm = torch.linalg.norm(inverse, dim=1)
        if search_cutoff_A is None:
            initial_cart = torch.einsum("bijd,bdk->bijk", centered, lattice)
            reach = (
                torch.linalg.norm(initial_cart, dim=-1)
                .masked_fill(~pair_valid, 0.0)
                .amax(dim=(1, 2))[:, None]
            )
        else:
            reach = torch.full_like(reciprocal_norm, float(search_cutoff_A))
        radius = torch.ceil(reach * reciprocal_norm + 0.5).long().clamp_min(1)
        if not self.cuda_dynamic_image_radius and lattice.device.type == "cuda":
            # Sync-free mode: keep a fixed grid and assert on device that it is
            # large enough rather than reading the required radius back.
            within_budget = (radius <= self.cuda_static_image_radius).all()
            message = (
                "periodic closest-image radius exceeds the synchronization-free "
                f"CUDA budget {self.cuda_static_image_radius}"
            )
            assert_async = getattr(torch, "_assert_async", None)
            if assert_async is not None:
                assert_async(within_budget, message)
            else:  # pragma: no cover
                torch._assert(within_budget, message)
            fixed = self.cuda_static_image_radius
            return (fixed, fixed, fixed)
        # One small host read of three values replaces the previous per-graph
        # work-spec transfer and the Python loop over graphs it drove.
        per_axis = radius.amax(dim=0).tolist()
        return (int(per_axis[0]), int(per_axis[1]), int(per_axis[2]))

    def forward(
        self,
        frac: Tensor,
        lattice: Tensor,
        valid: Tensor,
        *,
        lattice_inverse: Tensor | None = None,
        search_cutoff_A: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        if frac.ndim != 3 or frac.shape[-1] != 3:
            raise ValueError("frac must have shape [B, N, 3]")
        if lattice.shape != (frac.shape[0], 3, 3):
            raise ValueError("lattice must have shape [B, 3, 3]")
        if valid.shape != frac.shape[:2]:
            raise ValueError("valid must have shape [B, N]")
        delta = frac[:, None, :, :] - frac[:, :, None, :]
        centered = delta - torch.round(delta)
        pair_valid = valid[:, :, None] & valid[:, None, :]
        inverse = (
            batched_inverse_3x3(lattice)
            if lattice_inverse is None
            else lattice_inverse.to(device=lattice.device, dtype=lattice.dtype)
        )
        with torch.no_grad():
            radii = self._search_radii(
                centered.detach().float(),
                lattice.detach().float(),
                inverse.detach().float(),
                pair_valid,
                search_cutoff_A,
            )
            offsets = self._closest_offsets(
                centered.detach().float(), lattice.detach().float(), radii
            )
        best_frac = centered + offsets.to(dtype=centered.dtype)
        best_cart = torch.einsum("bijd,bdk->bijk", best_frac, lattice)
        mask = pair_valid.unsqueeze(-1)
        return best_frac * mask, best_cart * mask


class CrystalPairEmbedding(nn.Module):
    """Map token types and atom geometry/topology into additive head bias."""

    def __init__(
        self,
        *,
        num_heads: int,
        d_pair: int = 128,
        num_rbf: int = 32,
        rbf_cutoff: float = 12.0,
        num_fourier_frequencies: int = 4,
        inter_component_cutoff: float = 8.0,
        max_bond_types: int = 16,
        max_image_candidates: int = 65536,
        pair_chunk_size: int = 1024,
        image_chunk_size: int = 256,
        feature_chunk_size: int = 65536,
        cuda_static_image_radius: int = 2,
        cuda_dynamic_image_radius: bool = False,
        pair_image_work_budget: int = 16777216,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_fourier_frequencies = max(int(num_fourier_frequencies), 1)
        self.inter_component_cutoff = float(inter_component_cutoff)
        if self.inter_component_cutoff <= 0.0:
            raise ValueError("inter_component_cutoff must be positive")
        self.periodic = PeriodicDisplacement(
            max_image_candidates=max_image_candidates,
            pair_chunk_size=pair_chunk_size,
            image_chunk_size=image_chunk_size,
            cuda_static_image_radius=cuda_static_image_radius,
            cuda_dynamic_image_radius=cuda_dynamic_image_radius,
            pair_image_work_budget=pair_image_work_budget,
        )
        self.feature_chunk_size = max(int(feature_chunk_size), 1)
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=rbf_cutoff)
        feature_dim = int(num_rbf) + 6 * self.num_fourier_frequencies
        self.atom_pair_mlp = nn.Sequential(
            nn.Linear(feature_dim, int(d_pair)),
            nn.SiLU(),
            nn.Linear(int(d_pair), self.num_heads),
        )
        # Four directed token-pair classes: lattice-lattice, lattice-atom,
        # atom-lattice and atom-atom.  Atom-atom geometry is added below.
        self.token_pair_bias = nn.Embedding(4, self.num_heads)
        # Index 0 is the reference/no-bond relation and stays exactly zero.
        self.bond_delta = nn.Embedding(
            int(max_bond_types) + 2, self.num_heads, padding_idx=0
        )
        self.orbit_delta = nn.Parameter(torch.zeros(self.num_heads))
        # Different-component-type is the zero reference.
        self.component_relation_delta = nn.Embedding(
            DIFFERENT_COMPONENT_TYPE, self.num_heads
        )
        self.relation_geometry_log_scale = nn.Parameter(
            torch.zeros(NUM_COMPONENT_RELATIONS, self.num_heads)
        )
        nn.init.zeros_(self.bond_delta.weight)
        nn.init.zeros_(self.component_relation_delta.weight)

    def forward(
        self,
        state: CrystalGeometryState,
        plan: OrbitPlan,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        geometry = state.full
        frac_delta, cart_delta = self.periodic(
            geometry.frac.float(),
            state.lattice.float(),
            geometry.valid,
            lattice_inverse=state.lattice_inverse,
            # Every geometry term below is multiplied by an envelope that is
            # exactly zero at this distance, so the image search only has to be
            # exact inside it.
            search_cutoff_A=self.inter_component_cutoff,
        )
        raw_frac = geometry.frac[:, None, :, :] - geometry.frac[:, :, None, :]
        raw_cart = torch.einsum("bijd,bdk->bijk", raw_frac, state.lattice)
        same_instance = (
            plan.full_component_instance[:, :, None]
            == plan.full_component_instance[:, None, :]
        )
        same_base = (
            plan.full_base_component[:, :, None]
            == plan.full_base_component[:, None, :]
        )
        same_type = (
            plan.full_component_type[:, :, None]
            == plan.full_component_type[:, None, :]
        )
        same_orbit = plan.full_to_asu[:, :, None] == plan.full_to_asu[:, None, :]
        pair_valid = geometry.valid[:, :, None] & geometry.valid[:, None, :]
        same_instance = same_instance & pair_valid
        same_base = same_base & pair_valid
        same_type = same_type & pair_valid
        same_orbit = same_orbit & pair_valid
        # Within one molecular instance, ASU canonicalization already provides
        # an unwrapped connected geometry; periodic image search is unnecessary.
        frac_delta = torch.where(same_instance.unsqueeze(-1), raw_frac, frac_delta)
        cart_delta = torch.where(same_instance.unsqueeze(-1), raw_cart, cart_delta)

        frequencies = torch.arange(
            1,
            self.num_fourier_frequencies + 1,
            device=frac_delta.device,
            dtype=frac_delta.dtype,
        )
        flat_frac = frac_delta.reshape(-1, 3)
        flat_cart = cart_delta.reshape(-1, 3)
        flat_bond = plan.bond_type.to(frac_delta.device).reshape(-1)
        flat_same_base = same_base.reshape(-1)
        flat_same_instance = same_instance.reshape(-1)
        flat_same_type = same_type.reshape(-1)
        flat_same_orbit = same_orbit.reshape(-1)
        flat_valid = pair_valid.reshape(-1)
        bias_parts: list[Tensor] = []
        for start in range(0, int(flat_frac.shape[0]), self.feature_chunk_size):
            stop = min(start + self.feature_chunk_size, int(flat_frac.shape[0]))
            frac_part = flat_frac[start:stop]
            distance = torch.linalg.norm(flat_cart[start:stop], dim=-1)
            rbf = self.rbf(distance)
            phase = 2.0 * torch.pi * frac_part.unsqueeze(-1) * frequencies
            fourier = torch.cat(
                (torch.sin(phase), torch.cos(phase)), dim=-1
            ).flatten(-2)
            relation = component_relation_index(
                flat_same_type[start:stop],
                flat_same_base[start:stop], flat_same_instance[start:stop]
            )
            geometry_bias = self.atom_pair_mlp(torch.cat((rbf, fourier), dim=-1))
            relation_scale = self.relation_geometry_log_scale.index_select(
                0, relation
            ).clamp(-2.0, 2.0).exp()
            cutoff = torch.where(
                flat_same_instance[start:stop],
                torch.ones_like(distance),
                quintic_cutoff(distance, self.inter_component_cutoff),
            )
            relation_delta = torch.zeros_like(geometry_bias)
            non_reference = relation < DIFFERENT_COMPONENT_TYPE
            if relation_delta.numel():
                relation_delta = torch.where(
                    non_reference.unsqueeze(-1),
                    self.component_relation_delta(
                        relation.clamp(max=DIFFERENT_COMPONENT_TYPE - 1)
                    ),
                    relation_delta,
                )
            part = (
                cutoff.unsqueeze(-1) * relation_scale * geometry_bias
                + self.bond_delta(
                    flat_bond[start:stop].clamp(
                        max=self.bond_delta.num_embeddings - 1
                    )
                )
                + flat_same_orbit[start:stop].unsqueeze(-1).to(geometry_bias.dtype)
                * self.orbit_delta
                + relation_delta
            )
            bias_parts.append(part * flat_valid[start:stop].unsqueeze(-1))
        if bias_parts:
            atom_bias = torch.cat(bias_parts, dim=0).reshape(
                *frac_delta.shape[:3], self.num_heads
            )
        else:
            atom_bias = frac_delta.new_zeros(
                (*frac_delta.shape[:3], self.num_heads)
            )
        batch_size, num_atoms = geometry.valid.shape
        token_count = num_atoms + 3
        pair_type = torch.zeros(
            (batch_size, token_count, token_count),
            device=frac_delta.device,
            dtype=torch.long,
        )
        pair_type[:, :3, 3:] = 1
        pair_type[:, 3:, :3] = 2
        pair_type[:, 3:, 3:] = 3
        bias = self.token_pair_bias(pair_type)
        bias[:, 3:, 3:] = bias[:, 3:, 3:] + atom_bias
        return bias.permute(0, 3, 1, 2).contiguous().to(dtype=dtype)


__all__ = [
    "CrystalPairEmbedding",
    "PeriodicDisplacement",
    "SAME_COMPONENT_INSTANCE",
    "SAME_BASE_COMPONENT_OTHER_INSTANCE",
    "SAME_COMPONENT_TYPE_OTHER_BASE",
    "DIFFERENT_COMPONENT_TYPE",
    "component_relation_index",
    "quintic_cutoff",
]
