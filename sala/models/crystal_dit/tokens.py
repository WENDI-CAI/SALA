"""Permutation-equivariant atom tokens and three explicit lattice tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from sala.core.orbit import OrbitPlan
from sala.models.common import MPFeatureMixer, RMSNorm, mp_residual, zero_module
from sala.models.crystal_dit.geometry import CrystalGeometryState


class CovalentGraphBlock(nn.Module):
    """Lightweight ASU message passing over the static covalent graph."""

    def __init__(self, d_model: int, max_bond_types: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model, elementwise_affine=False, eps=1.0e-6)
        self.bond_embedding = nn.Embedding(int(max_bond_types) + 1, d_model)
        self.message = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.gate = nn.Parameter(torch.zeros(d_model))
        nn.init.zeros_(self.bond_embedding.weight[0])

    def forward(self, hidden: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index.numel() == 0 or hidden.shape[0] == 0:
            return hidden
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_attr.shape != (edge_index.shape[1],):
            raise ValueError("edge_attr must contain one value per directed edge")
        source, target = edge_index.long()
        normalized = self.norm(hidden)
        bond_index = edge_attr.long().clamp(0, self.bond_embedding.num_embeddings - 2) + 1
        message = self.message(
            normalized.index_select(0, source) + self.bond_embedding(bond_index)
        )
        aggregate = message.new_zeros(hidden.shape)
        aggregate.index_add_(0, target, message)
        degree = torch.bincount(target, minlength=hidden.shape[0]).to(message.dtype)
        aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
        update = self.update(torch.cat((normalized, aggregate), dim=-1))
        return mp_residual(hidden, update, self.gate)


class CrystalTokenEmbedder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_elements: int = 119,
        num_hybridizations: int = 8,
        formal_charge_range: int = 16,
        num_fractional_frequencies: int = 4,
        max_bond_types: int = 16,
        num_covalent_layers: int = 2,
        self_conditioning_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_fractional_frequencies = max(int(num_fractional_frequencies), 1)
        self.formal_charge_offset = int(formal_charge_range)
        self.self_conditioning_enabled = bool(self_conditioning_enabled)
        self.element_embedding = nn.Embedding(int(num_elements), self.d_model)
        self.hybridization_embedding = nn.Embedding(
            int(num_hybridizations), self.d_model
        )
        self.charge_embedding = nn.Embedding(
            2 * self.formal_charge_offset + 1, self.d_model
        )
        self.charge_valid_embedding = nn.Embedding(2, self.d_model)
        self.chemistry_mixer = MPFeatureMixer(4, self.d_model)
        self.covalent_blocks = nn.ModuleList(
            [
                CovalentGraphBlock(self.d_model, max_bond_types)
                for _ in range(max(int(num_covalent_layers), 0))
            ]
        )
        self.component_charge_embedding = nn.Embedding(
            2 * self.formal_charge_offset + 1, self.d_model
        )
        self.token_type_embedding = nn.Embedding(4, self.d_model)
        self.component_norm = RMSNorm(
            self.d_model, elementwise_affine=False, eps=1.0e-6
        )
        self.component_context = nn.Sequential(
            nn.Linear(4 * self.d_model + 1, self.d_model),
            nn.SiLU(),
            zero_module(nn.Linear(self.d_model, self.d_model, bias=False)),
        )
        self.component_gate = nn.Sequential(
            RMSNorm(self.d_model, elementwise_affine=False, eps=1.0e-6),
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid(),
        )
        atom_geom_dim = 3 + 6 * self.num_fractional_frequencies
        self.atom_geometry = nn.Sequential(
            nn.Linear(atom_geom_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.atom_self_condition = (
            zero_module(nn.Linear(3, self.d_model, bias=False))
            if self.self_conditioning_enabled
            else None
        )
        self.atom_mixer = MPFeatureMixer(
            5 if self.self_conditioning_enabled else 4, self.d_model
        )
        # row, reciprocal row, length, log-volume, q-model and active mask
        self.lattice_geometry = nn.Sequential(
            nn.Linear(3 + 3 + 1 + 1 + 6 + 6, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.lattice_self_condition = (
            zero_module(nn.Linear(6, self.d_model, bias=False))
            if self.self_conditioning_enabled
            else None
        )
        self.lattice_mixer = MPFeatureMixer(
            3 if self.self_conditioning_enabled else 2, self.d_model
        )

    def _fractional_fourier(self, frac: Tensor) -> Tensor:
        frequency = torch.arange(
            1,
            self.num_fractional_frequencies + 1,
            device=frac.device,
            dtype=frac.dtype,
        )
        phase = 2.0 * torch.pi * frac.unsqueeze(-1) * frequency
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(-2)

    def _charge_index(self, charge: Tensor) -> Tensor:
        return charge.long().clamp(
            -self.formal_charge_offset, self.formal_charge_offset
        ) + self.formal_charge_offset

    def _component_context(
        self,
        atom_chemical: Tensor,
        atom_to_component: Tensor,
        component_formal_charge: Tensor,
        num_components: int,
    ) -> Tensor:
        if atom_chemical.shape[0] == 0:
            return atom_chemical
        if atom_to_component.shape != (atom_chemical.shape[0],):
            raise ValueError("atom_to_component must contain one index per ASU atom")
        if component_formal_charge.shape != (int(num_components),):
            raise ValueError(
                "component_formal_charge must contain one value per component"
            )
        if int(num_components) <= 0:
            raise ValueError("non-empty atom input requires at least one component")

        component_index = atom_to_component.to(
            device=atom_chemical.device, dtype=torch.long
        )
        counts = torch.bincount(
            component_index, minlength=int(num_components)
        ).to(dtype=atom_chemical.dtype)
        component_sum = atom_chemical.new_zeros(
            (int(num_components), atom_chemical.shape[-1])
        )
        component_sum.index_add_(0, component_index, atom_chemical)
        component_mean = component_sum / counts.clamp_min(1.0).unsqueeze(-1)
        component_scaled_sum = component_sum / counts.clamp_min(1.0).sqrt().unsqueeze(-1)
        component_max = torch.full_like(component_sum, -torch.inf)
        component_max.scatter_reduce_(
            0,
            component_index[:, None].expand_as(atom_chemical),
            atom_chemical,
            reduce="amax",
            include_self=True,
        )
        component_max = torch.where(
            torch.isfinite(component_max), component_max, torch.zeros_like(component_max)
        )
        component_charge = self.component_charge_embedding(
            self._charge_index(component_formal_charge)
        )
        component_features = torch.cat(
            (
                self.component_norm(component_scaled_sum),
                self.component_norm(component_mean),
                self.component_norm(component_max),
                self.component_norm(component_charge),
                torch.log1p(counts).unsqueeze(-1),
            ),
            dim=-1,
        )
        context = self.component_context(component_features)
        atom_context = context.index_select(0, component_index)
        return atom_context * self.component_gate(atom_chemical)

    def forward(
        self,
        *,
        state: CrystalGeometryState,
        plan: OrbitPlan,
        model_full_cart: Tensor,
        q_model: Tensor,
        q_mask: Tensor,
        atom_types: Tensor,
        hybridization: Tensor,
        atom_formal_charge: Tensor,
        atom_charge_mask: Tensor,
        component_formal_charge: Tensor,
        atom_to_component: Tensor,
        num_components: int,
        edge_index: Tensor,
        edge_attr: Tensor,
        full_self_condition: Tensor | None = None,
        q_self_condition: Tensor | None = None,
    ) -> Tensor:
        geometry = state.full
        source = plan.full_to_asu.to(atom_types.device).clamp_min(0)
        z = atom_types.long().clamp(0, self.element_embedding.num_embeddings - 1)
        hyb = hybridization.long().clamp(
            0, self.hybridization_embedding.num_embeddings - 1
        )
        if atom_formal_charge.shape != atom_types.shape:
            raise ValueError("atom_formal_charge must contain one value per ASU atom")
        if atom_charge_mask.shape != atom_types.shape:
            raise ValueError("atom_charge_mask must contain one value per ASU atom")
        charge = torch.where(
            atom_charge_mask.bool(), atom_formal_charge.long(), torch.zeros_like(atom_types)
        )
        atom_chemical = self.chemistry_mixer(
            self.element_embedding(z),
            self.hybridization_embedding(hyb),
            self.charge_embedding(self._charge_index(charge)),
            self.charge_valid_embedding(atom_charge_mask.long()),
        )
        for block in self.covalent_blocks:
            atom_chemical = block(atom_chemical, edge_index, edge_attr)
        component_context = self._component_context(
            atom_chemical,
            atom_to_component,
            component_formal_charge,
            num_components,
        )
        full_chemical = atom_chemical.index_select(0, source.reshape(-1)).reshape(
            *source.shape, -1
        )
        full_component = component_context.index_select(
            0, source.reshape(-1)
        ).reshape(*source.shape, -1)
        atom_geom = torch.cat(
            (model_full_cart, self._fractional_fourier(geometry.frac.float())),
            dim=-1,
        )
        atom_geometry = self.atom_geometry(atom_geom.to(full_chemical.dtype))
        atom_type = self.token_type_embedding.weight[3].view(1, 1, -1).expand_as(
            full_chemical
        )
        atom_features = [
            full_chemical,
            full_component,
            atom_geometry,
            atom_type,
        ]
        if self.atom_self_condition is not None:
            if full_self_condition is None:
                full_self_condition = torch.zeros_like(model_full_cart)
            atom_features.append(
                self.atom_self_condition(
                    full_self_condition.to(dtype=full_chemical.dtype)
                )
            )
        atom_hidden = self.atom_mixer(*atom_features)
        atom_hidden = atom_hidden * geometry.valid.unsqueeze(-1).to(atom_hidden.dtype)

        q_common = q_model.float().unsqueeze(1).expand(-1, 3, -1)
        mask_common = q_mask.float().unsqueeze(1).expand(-1, 3, -1)
        lattice_features = torch.cat(
            (
                state.lattice,
                state.reciprocal,
                state.lengths,
                state.log_volume.expand(-1, 3, -1),
                q_common,
                mask_common,
            ),
            dim=-1,
        )
        lattice_geometry = self.lattice_geometry(lattice_features)
        axis = torch.arange(3, device=state.lattice.device)
        lattice_type = self.token_type_embedding(axis)[None].expand_as(
            lattice_geometry
        )
        lattice_features_list = [lattice_geometry, lattice_type]
        if self.lattice_self_condition is not None:
            if q_self_condition is None:
                q_self_condition = torch.zeros_like(q_model)
            lattice_features_list.append(
                self.lattice_self_condition(
                    q_self_condition.to(dtype=lattice_geometry.dtype)
                ).unsqueeze(1).expand(-1, 3, -1)
            )
        lattice_hidden = self.lattice_mixer(*lattice_features_list)
        return torch.cat((lattice_hidden, atom_hidden), dim=1)


__all__ = ["CovalentGraphBlock", "CrystalTokenEmbedder"]
