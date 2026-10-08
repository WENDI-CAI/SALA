"""Flow-matching full-cell, token-pair-biased Crystal DiT velocity field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from sala.core.lattice_manifold import (
    ConstrainedLatticeDecoder,
    NUM_LATTICE_TEMPLATES,
    q_active_mask,
    require_lattice_template,
)
from sala.core.orbit import OrbitPlan, build_orbit_plan
from sala.core.types import AsuBatch
from sala.models.common import MPFeatureMixer, RMSNorm, TimeEmbedding, zero_module
from sala.models.crystal_dit.blocks import CrystalDiTBlock, FinalAdaNorm
from sala.models.crystal_dit.geometry import build_geometry_state
from sala.models.crystal_dit.pair import CrystalPairEmbedding
from sala.models.crystal_dit.tokens import CrystalTokenEmbedder


@dataclass(frozen=True)
class PreparedCrystalCondition:
    batch_identity: int
    plan: OrbitPlan
    hybridization: Tensor
    atom_formal_charge: Tensor
    atom_charge_mask: Tensor
    component_formal_charge: Tensor
    atom_to_component: Tensor
    num_components: int

    def validate(self, batch: AsuBatch) -> None:
        if self.batch_identity != id(batch):
            raise ValueError("PreparedCrystalCondition belongs to a different AsuBatch")
        if self.plan.num_asu_atoms != int(batch.atom_types.shape[0]):
            raise ValueError("PreparedCrystalCondition atom count no longer matches batch")
        if self.atom_to_component.shape != batch.atom_types.shape:
            raise ValueError("PreparedCrystalCondition component map no longer matches batch")
        if self.atom_formal_charge.shape != batch.atom_types.shape:
            raise ValueError("PreparedCrystalCondition atom charge no longer matches batch")
        if self.atom_charge_mask.shape != batch.atom_types.shape:
            raise ValueError("PreparedCrystalCondition charge mask no longer matches batch")
        if self.component_formal_charge.shape != batch.component_batch.shape:
            raise ValueError("PreparedCrystalCondition component charge no longer matches batch")


class FullCellCrystalDiT(nn.Module):
    """Modern single-stream ``[3 lattice, full-cell atoms]`` velocity field."""

    accepts_physical_x = True
    accepts_model_q = True
    accepts_atom_input_scale = True
    accepts_self_conditioning = True
    state_representation = "cartesian_q"
    atom_coord_repr = "canonical_cartesian"
    lattice_parameterization = "constrained_q"
    component_representation = "base_instance_type_v2"
    prediction_type = "velocity"
    architecture = "full_cell_crystal_dit_flow"

    def __init__(
        self,
        *,
        d_model: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        d_ffn: int = 2816,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_flash: bool = True,
        qk_norm: bool = True,
        attention_backend: str = "auto",
        num_elements: int = 119,
        num_hybridizations: int = 8,
        formal_charge_range: int = 16,
        num_fourier_freqs: int = 4,
        d_pair: int = 128,
        pair_num_rbf: int = 32,
        pair_rbf_cutoff: float = 12.0,
        inter_component_cutoff: float = 8.0,
        max_bond_types: int = 16,
        num_covalent_layers: int = 2,
        max_image_candidates: int = 65536,
        pair_chunk_size: int = 1024,
        image_chunk_size: int = 256,
        pair_feature_chunk_size: int = 65536,
        cuda_static_image_radius: int = 2,
        cuda_dynamic_image_radius: bool = False,
        pair_bias_enabled: bool = True,
        self_conditioning_enabled: bool = False,
        use_gradient_checkpointing: bool = True,
        validate_general_position: bool = True,
        special_position_tolerance: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if int(d_model) % int(num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.d_ffn = int(d_ffn)
        self.inter_component_cutoff = float(inter_component_cutoff)
        self.pair_bias_enabled = bool(pair_bias_enabled)
        self.self_conditioning_enabled = bool(self_conditioning_enabled)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.validate_general_position = bool(validate_general_position)
        self.special_position_tolerance = float(special_position_tolerance)

        self.lattice_decoder = ConstrainedLatticeDecoder()
        self.token_embedder = CrystalTokenEmbedder(
            d_model=self.d_model,
            num_elements=num_elements,
            num_hybridizations=num_hybridizations,
            formal_charge_range=formal_charge_range,
            num_fractional_frequencies=num_fourier_freqs,
            max_bond_types=max_bond_types,
            num_covalent_layers=num_covalent_layers,
            self_conditioning_enabled=self.self_conditioning_enabled,
        )
        self.pair_embedding = (
            CrystalPairEmbedding(
                num_heads=self.num_heads,
                d_pair=d_pair,
                num_rbf=pair_num_rbf,
                rbf_cutoff=pair_rbf_cutoff,
                num_fourier_frequencies=num_fourier_freqs,
                inter_component_cutoff=self.inter_component_cutoff,
                max_bond_types=max_bond_types,
                max_image_candidates=max_image_candidates,
                pair_chunk_size=pair_chunk_size,
                image_chunk_size=image_chunk_size,
                feature_chunk_size=pair_feature_chunk_size,
                cuda_static_image_radius=cuda_static_image_radius,
                cuda_dynamic_image_radius=cuda_dynamic_image_radius,
            )
            if self.pair_bias_enabled
            else None
        )
        self.time_embedding = TimeEmbedding(
            d_model=self.d_model,
            num_fourier_freqs=max(16, num_fourier_freqs * 4),
            hidden_dim=self.d_model,
            dropout=0.0,
        )
        self.spacegroup_embedding = nn.Embedding(231, self.d_model)
        self.template_embedding = nn.Embedding(NUM_LATTICE_TEMPLATES, self.d_model)
        self.count_embedding = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.condition_mixer = MPFeatureMixer(4, self.d_model)
        self.blocks = nn.ModuleList(
            [
                CrystalDiTBlock(
                    d_model=self.d_model,
                    d_cond=self.d_model,
                    num_heads=self.num_heads,
                    d_ffn=self.d_ffn,
                    dropout=float(dropout),
                    attn_dropout=float(attn_dropout),
                    use_flash=bool(use_flash),
                    qk_norm=bool(qk_norm),
                    attention_backend=str(attention_backend),
                    pair_bias_enabled=self.pair_bias_enabled,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = FinalAdaNorm(self.d_model, self.d_model)
        self.atom_head = zero_module(nn.Linear(self.d_model, 3))
        self.lattice_head = nn.Sequential(
            RMSNorm(3 * self.d_model),
            nn.Linear(3 * self.d_model, self.d_model),
            nn.SiLU(),
            zero_module(nn.Linear(self.d_model, 6)),
        )

    def prepare_condition(
        self,
        batch: AsuBatch,
        cache_trainable_features: bool | None = None,
    ) -> PreparedCrystalCondition:
        del cache_trainable_features
        plan = getattr(batch, "orbit_plan", None)
        if plan is None:
            plan = build_orbit_plan(
                batch,
                validate_general_position=self.validate_general_position,
                special_position_tolerance=self.special_position_tolerance,
            )
        num_atoms = int(batch.atom_types.shape[0])
        hybridization = batch.hybridization
        if hybridization is None or hybridization.numel() == 0:
            hybridization = batch.atom_types.new_zeros(num_atoms)
        elif hybridization.numel() != num_atoms:
            raise ValueError("hybridization must contain one value per ASU atom")

        if batch.component_id.numel() != num_atoms:
            raise ValueError("component_id must contain one value per ASU atom")
        # Collation already assigns globally contiguous component ids. Reuse
        # them directly instead of a GPU torch.unique + host synchronization.
        atom_to_component = batch.component_id.long()
        num_components = int(batch.component_batch.shape[0])
        if num_atoms:
            valid_components = (
                (atom_to_component >= 0)
                & (atom_to_component < max(num_components, 1))
            ).all()
            if atom_to_component.device.type == "cuda":
                assert_async = getattr(torch, "_assert_async", None)
                if assert_async is not None:
                    assert_async(valid_components, "component ids are not contiguous")
            elif not bool(valid_components):
                raise ValueError("component ids are not contiguous within the batch")

        atom_formal_charge = batch.atom_formal_charge.long()
        atom_charge_mask = batch.atom_charge_mask.bool()
        if atom_formal_charge.shape != batch.atom_types.shape:
            raise ValueError("atom_formal_charge must contain one value per ASU atom")
        if atom_charge_mask.shape != batch.atom_types.shape:
            raise ValueError("atom_charge_mask must contain one value per ASU atom")
        component_formal_charge = batch.component_formal_charge.long()
        if component_formal_charge.numel() != num_components:
            raise ValueError(
                "component_formal_charge must contain one value per component"
            )
        return PreparedCrystalCondition(
            batch_identity=id(batch),
            plan=plan,
            hybridization=hybridization,
            atom_formal_charge=atom_formal_charge,
            atom_charge_mask=atom_charge_mask,
            component_formal_charge=component_formal_charge,
            atom_to_component=atom_to_component,
            num_components=num_components,
        )

    def _condition(
        self,
        *,
        batch: AsuBatch,
        plan: OrbitPlan,
        flow_condition: Tensor | None,
        t: Tensor,
        template_id: Tensor,
    ) -> Tensor:
        time_value = flow_condition if flow_condition is not None else t.float()
        if time_value.numel() == 1 and batch.batch_size != 1:
            time_value = time_value.expand(batch.batch_size)
        if time_value.shape != (batch.batch_size,):
            raise ValueError("flow condition must contain one value per graph")
        component_counts = torch.bincount(
            batch.component_batch.long(), minlength=batch.batch_size
        ).float()
        count_features = torch.stack(
            (
                torch.log1p(plan.full_counts.float()),
                torch.log1p(component_counts),
            ),
            dim=-1,
        )
        return self.condition_mixer(
            self.time_embedding(time_value),
            self.spacegroup_embedding(batch.spacegroup.long()),
            self.template_embedding(template_id.long()),
            self.count_embedding(count_features),
        )

    @staticmethod
    def _block_forward(
        block: nn.Module,
        hidden: Tensor,
        condition: Tensor,
        valid: Tensor,
        attention_bias: Tensor | None,
    ) -> Tensor:
        return block(
            hidden,
            condition=condition,
            token_valid=valid,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        batch: AsuBatch,
        model_x: Tensor,
        physical_q: Tensor,
        t: Tensor,
        flow_condition: Optional[Tensor] = None,
        prepared_condition: Optional[PreparedCrystalCondition] = None,
        physical_x: Optional[Tensor] = None,
        model_q: Optional[Tensor] = None,
        atom_input_scale: Optional[Tensor] = None,
        geometry_q: Optional[Tensor] = None,
        self_condition: Optional[object] = None,
    ) -> tuple[Tensor, Tensor]:
        condition_data = prepared_condition or self.prepare_condition(batch)
        condition_data.validate(batch)
        plan = condition_data.plan
        physical_x = model_x if physical_x is None else physical_x
        if physical_x.shape != (plan.num_asu_atoms, 3):
            raise ValueError("physical ASU Cartesian state has an invalid shape")
        template_id = require_lattice_template(
            batch.spacegroup, batch.lattice_template_id
        )
        geometry_q = physical_q if geometry_q is None else geometry_q
        lattice = self.lattice_decoder(geometry_q.float(), batch.spacegroup, template_id)
        q_mask = q_active_mask(
            spacegroup=batch.spacegroup,
            lattice_template_id=template_id,
            dtype=physical_q.dtype,
        )
        state = build_geometry_state(
            plan=plan, asu_cart=physical_x.float(), lattice=lattice
        )

        if atom_input_scale is None:
            c_in_atom = torch.ones(
                batch.batch_size, device=model_x.device, dtype=torch.float32
            )
        else:
            c_in_atom = atom_input_scale.to(
                device=model_x.device, dtype=torch.float32
            ).view(-1)
            if c_in_atom.shape != (batch.batch_size,):
                raise ValueError("atom_input_scale must contain one scale per graph")
        model_full_cart = state.full.cart.float() * c_in_atom[:, None, None]
        q_model = physical_q if model_q is None else model_q

        full_self_condition = None
        q_self_condition = None
        if self.self_conditioning_enabled:
            atom_residual = getattr(self_condition, "atom_residual", None)
            lattice_residual = getattr(self_condition, "lattice_residual", None)
            graph_present = getattr(self_condition, "graph_present", None)
            if atom_residual is not None or lattice_residual is not None:
                if atom_residual is None or lattice_residual is None or graph_present is None:
                    raise ValueError(
                        "flow self-conditioning requires atom, lattice, and presence tensors"
                    )
                if atom_residual.shape != (plan.num_asu_atoms, 3):
                    raise ValueError("self-conditioning atom residual shape is invalid")
                if lattice_residual.shape != (batch.batch_size, 6):
                    raise ValueError("self-conditioning lattice residual shape is invalid")
                graph_present = graph_present.to(
                    device=model_x.device, dtype=torch.bool
                ).view(-1)
                if graph_present.shape != (batch.batch_size,):
                    raise ValueError("self-conditioning presence shape is invalid")
                atom_present = graph_present.index_select(0, batch.batch.long())
                atom_residual = torch.where(
                    atom_present.unsqueeze(-1),
                    atom_residual.to(device=model_x.device, dtype=model_x.dtype),
                    torch.zeros_like(model_x),
                )
                lattice_residual = torch.where(
                    graph_present.unsqueeze(-1),
                    lattice_residual.to(device=q_model.device, dtype=q_model.dtype)
                    * q_mask.to(dtype=q_model.dtype),
                    torch.zeros_like(q_model),
                )
                full_self_condition = plan.expand_vectors(
                    atom_residual.float(),
                    state.lattice,
                    lattice_inverse=state.lattice_inverse,
                ).to(dtype=model_x.dtype)
                q_self_condition = lattice_residual
        elif self_condition is not None:
            raise ValueError(
                "self-conditioning input was provided to a disabled velocity field"
            )

        hidden = self.token_embedder(
            state=state,
            plan=plan,
            model_full_cart=model_full_cart,
            q_model=q_model,
            q_mask=q_mask,
            atom_types=batch.atom_types,
            hybridization=condition_data.hybridization,
            atom_formal_charge=condition_data.atom_formal_charge,
            atom_charge_mask=condition_data.atom_charge_mask,
            component_formal_charge=condition_data.component_formal_charge,
            atom_to_component=condition_data.atom_to_component,
            num_components=condition_data.num_components,
            edge_index=batch.edge_index_conv,
            edge_attr=batch.edge_attr_conv,
            full_self_condition=full_self_condition,
            q_self_condition=q_self_condition,
        )
        token_valid = plan.token_valid.to(hidden.device)
        global_condition = self._condition(
            batch=batch,
            plan=plan,
            flow_condition=flow_condition,
            t=t,
            template_id=template_id,
        )
        base_bias = (
            self.pair_embedding(state, plan, dtype=hidden.dtype)
            if self.pair_embedding is not None
            else None
        )
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(
                    self._block_forward,
                    block,
                    hidden,
                    global_condition,
                    token_valid,
                    base_bias,
                    use_reentrant=False,
                )
            else:
                hidden = block(
                    hidden,
                    condition=global_condition,
                    token_valid=token_valid,
                    attention_bias=base_bias,
                )
        hidden = self.final_norm(hidden, global_condition)
        raw_full = self.atom_head(hidden[:, 3:]) * plan.full_valid.unsqueeze(-1)
        raw_asu = plan.pullback_vectors(
            raw_full.float(),
            state.lattice,
            lattice_inverse=state.lattice_inverse,
        ).to(dtype=model_x.dtype)
        raw_q = self.lattice_head(hidden[:, :3].reshape(batch.batch_size, -1))
        raw_q = raw_q * q_mask.to(dtype=raw_q.dtype)
        return raw_asu, raw_q.to(dtype=physical_q.dtype)


__all__ = ["FullCellCrystalDiT", "PreparedCrystalCondition"]
