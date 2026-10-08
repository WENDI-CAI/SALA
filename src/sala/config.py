"""Explicit architecture configuration without a training-framework dependency."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from sala.models import FullCellCrystalDiT


@dataclass(frozen=True)
class ModelConfig:
    """Constructor settings matching the source model's formal default recipe.

    The default configuration includes self-conditioning and native PyTorch
    SDPA. It defines the architecture only; building it initializes fresh
    parameters and does not load trained weights.
    """

    d_model: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    d_ffn: int = 2816
    dropout: float = 0.0
    attn_dropout: float = 0.0
    use_flash: bool = True
    qk_norm: bool = True
    attention_backend: str = "sdpa"
    num_elements: int = 119
    num_hybridizations: int = 8
    formal_charge_range: int = 16
    num_fourier_freqs: int = 4
    d_pair: int = 128
    pair_num_rbf: int = 32
    pair_rbf_cutoff: float = 12.0
    inter_component_cutoff: float = 8.0
    max_bond_types: int = 16
    num_covalent_layers: int = 2
    max_image_candidates: int = 65536
    pair_chunk_size: int = 2048
    image_chunk_size: int = 256
    pair_feature_chunk_size: int = 131072
    cuda_static_image_radius: int = 2
    cuda_dynamic_image_radius: bool = True
    pair_bias_enabled: bool = True
    self_conditioning_enabled: bool = True
    use_gradient_checkpointing: bool = False
    validate_general_position: bool = True
    special_position_tolerance: float = 1.0e-5

    def build(self) -> FullCellCrystalDiT:
        """Build an untrained velocity field with these constructor settings."""

        return FullCellCrystalDiT(**asdict(self))

    @classmethod
    def tiny(cls, **overrides: object) -> ModelConfig:
        """Return a small CPU example profile with the same computation path.

        Overrides use the same field names as the full configuration. This
        profile is for examples and tests, not the reported trained model.
        """

        config = cls(
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ffn=192,
            d_pair=32,
            pair_num_rbf=8,
            use_flash=False,
            attention_backend="math",
            pair_chunk_size=128,
            image_chunk_size=64,
            pair_feature_chunk_size=2048,
            cuda_dynamic_image_radius=False,
        )
        return replace(config, **overrides)


__all__ = ["ModelConfig"]
