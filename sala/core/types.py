
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from torch import Tensor


@dataclass
class AsuInput:

    atom_types: Tensor
    spacegroup: Tensor
    component_id: Tensor
    atom_formal_charge: Tensor
    atom_charge_mask: Tensor
    component_formal_charge: Tensor
    component_type_id: Tensor
    num_atoms: Tensor
    edge_index_conv: Tensor
    edge_attr_conv: Tensor


    frac_coords_gt: Tensor
    lattice_gt: Tensor
    cart_coords_gt: Optional[Tensor] = None
    hybridization: Optional[Tensor] = None


    atom_labels: Optional[list[str]] = None
    cif_id: Optional[str] = None
    n_symops: Optional[Tensor] = None
    n_heavy_cell: Optional[Tensor] = None
    chem_group_id: Optional[Tensor] = None
    crystal_system_id: Optional[Tensor] = None
    # Authoritative space-group setting metadata for constrained q.
    hall_number: Optional[Tensor] = None
    setting_id: Optional[Tensor] = None
    lattice_template_id: Optional[Tensor] = None
    n_full_atoms: Optional[Tensor] = None
    # Explicit affine operations from the source CIF, shaped [G, 4, 4].
    # Full-cell consumers require this field; CIF parsing resolves it first.
    symmetry_ops: Optional[Tensor] = None

    def to(self, device: torch.device | str) -> "AsuInput":
        return AsuInput(
            atom_types=self.atom_types.to(device),
            spacegroup=self.spacegroup.to(device),
            component_id=self.component_id.to(device),
            atom_formal_charge=self.atom_formal_charge.to(device),
            atom_charge_mask=self.atom_charge_mask.to(device),
            component_formal_charge=self.component_formal_charge.to(device),
            component_type_id=self.component_type_id.to(device),
            num_atoms=self.num_atoms.to(device),
            edge_index_conv=self.edge_index_conv.to(device),
            edge_attr_conv=self.edge_attr_conv.to(device),
            frac_coords_gt=self.frac_coords_gt.to(device),
            lattice_gt=self.lattice_gt.to(device),
            cart_coords_gt=self.cart_coords_gt.to(device) if self.cart_coords_gt is not None else None,
            hybridization=self.hybridization.to(device) if self.hybridization is not None else None,
            atom_labels=self.atom_labels,
            cif_id=self.cif_id,
            n_symops=self.n_symops.to(device) if self.n_symops is not None else None,
            n_heavy_cell=self.n_heavy_cell.to(device) if self.n_heavy_cell is not None else None,
            chem_group_id=self.chem_group_id.to(device) if self.chem_group_id is not None else None,
            crystal_system_id=self.crystal_system_id.to(device) if self.crystal_system_id is not None else None,
            hall_number=self.hall_number.to(device) if self.hall_number is not None else None,
            setting_id=self.setting_id.to(device) if self.setting_id is not None else None,
            lattice_template_id=(
                self.lattice_template_id.to(device)
                if self.lattice_template_id is not None
                else None
            ),
            n_full_atoms=self.n_full_atoms.to(device) if self.n_full_atoms is not None else None,
            symmetry_ops=(
                getattr(self, "symmetry_ops", None).to(device)
                if getattr(self, "symmetry_ops", None) is not None
                else None
            ),
        )

    def clone(self) -> "AsuInput":
        return AsuInput(
            atom_types=self.atom_types.clone(),
            spacegroup=self.spacegroup.clone(),
            component_id=self.component_id.clone(),
            atom_formal_charge=self.atom_formal_charge.clone(),
            atom_charge_mask=self.atom_charge_mask.clone(),
            component_formal_charge=self.component_formal_charge.clone(),
            component_type_id=self.component_type_id.clone(),
            num_atoms=self.num_atoms.clone(),
            edge_index_conv=self.edge_index_conv.clone(),
            edge_attr_conv=self.edge_attr_conv.clone(),
            frac_coords_gt=self.frac_coords_gt.clone(),
            lattice_gt=self.lattice_gt.clone(),
            cart_coords_gt=self.cart_coords_gt.clone() if self.cart_coords_gt is not None else None,
            hybridization=self.hybridization.clone() if self.hybridization is not None else None,
            atom_labels=self.atom_labels.copy() if self.atom_labels else None,
            cif_id=self.cif_id,
            n_symops=self.n_symops.clone() if self.n_symops is not None else None,
            n_heavy_cell=self.n_heavy_cell.clone() if self.n_heavy_cell is not None else None,
            chem_group_id=self.chem_group_id.clone() if self.chem_group_id is not None else None,
            crystal_system_id=self.crystal_system_id.clone() if self.crystal_system_id is not None else None,
            hall_number=self.hall_number.clone() if self.hall_number is not None else None,
            setting_id=self.setting_id.clone() if self.setting_id is not None else None,
            lattice_template_id=(
                self.lattice_template_id.clone()
                if self.lattice_template_id is not None
                else None
            ),
            n_full_atoms=self.n_full_atoms.clone() if self.n_full_atoms is not None else None,
            symmetry_ops=(
                getattr(self, "symmetry_ops", None).clone()
                if getattr(self, "symmetry_ops", None) is not None
                else None
            ),
        )


@dataclass
class AsuBatch:

    atom_types: Tensor
    spacegroup: Tensor
    component_id: Tensor
    atom_formal_charge: Tensor
    atom_charge_mask: Tensor
    component_formal_charge: Tensor
    component_type_id: Tensor
    num_atoms: Tensor
    edge_index_conv: Tensor
    edge_attr_conv: Tensor
    frac_coords_gt: Tensor
    lattice_gt: Tensor


    batch: Tensor
    ptr: Tensor
    component_batch: Tensor
    cart_coords_gt: Optional[Tensor] = None
    hybridization: Optional[Tensor] = None


    cif_ids: Optional[list[str]] = None
    n_symops: Optional[Tensor] = None
    n_heavy_cell: Optional[Tensor] = None
    chem_group_id: Optional[Tensor] = None
    crystal_system_id: Optional[Tensor] = None
    hall_number: Optional[Tensor] = None
    setting_id: Optional[Tensor] = None
    lattice_template_id: Optional[Tensor] = None
    n_full_atoms: Optional[Tensor] = None
    # Ragged source-CIF operations are padded to [B, G_max, 4, 4].  The
    # corresponding [B, G_max] mask is authoritative. Every graph in a
    # full-cell batch must have at least one explicit operation.
    symmetry_ops: Optional[Tensor] = None
    symmetry_mask: Optional[Tensor] = None
    # CPU DataLoader workers may prebuild the static OrbitPlan so the GPU hot
    # path never has to synchronize metadata through .item()/.cpu().
    orbit_plan: Optional[Any] = None

    def _map_tensors(self, transform: Callable[[Tensor], Tensor]) -> "AsuBatch":
        orbit_plan = getattr(self, "orbit_plan", None)
        if orbit_plan is not None:
            orbit_plan = type(orbit_plan)(
                **{
                    name: transform(getattr(orbit_plan, name))
                    for name in orbit_plan.__dataclass_fields__
                }
            )
        return AsuBatch(
            atom_types=transform(self.atom_types),
            spacegroup=transform(self.spacegroup),
            component_id=transform(self.component_id),
            atom_formal_charge=transform(self.atom_formal_charge),
            atom_charge_mask=transform(self.atom_charge_mask),
            component_formal_charge=transform(self.component_formal_charge),
            component_type_id=transform(self.component_type_id),
            num_atoms=transform(self.num_atoms),
            edge_index_conv=transform(self.edge_index_conv),
            edge_attr_conv=transform(self.edge_attr_conv),
            frac_coords_gt=transform(self.frac_coords_gt),
            lattice_gt=transform(self.lattice_gt),
            batch=transform(self.batch),
            ptr=transform(self.ptr),
            component_batch=transform(self.component_batch),
            cart_coords_gt=(
                transform(self.cart_coords_gt)
                if self.cart_coords_gt is not None
                else None
            ),
            hybridization=(
                transform(self.hybridization)
                if self.hybridization is not None
                else None
            ),
            cif_ids=self.cif_ids,
            n_symops=transform(self.n_symops) if self.n_symops is not None else None,
            n_heavy_cell=(
                transform(self.n_heavy_cell)
                if self.n_heavy_cell is not None
                else None
            ),
            chem_group_id=(
                transform(self.chem_group_id)
                if self.chem_group_id is not None
                else None
            ),
            crystal_system_id=(
                transform(self.crystal_system_id)
                if self.crystal_system_id is not None
                else None
            ),
            hall_number=(
                transform(self.hall_number)
                if self.hall_number is not None
                else None
            ),
            setting_id=(
                transform(self.setting_id)
                if self.setting_id is not None
                else None
            ),
            lattice_template_id=(
                transform(self.lattice_template_id)
                if self.lattice_template_id is not None
                else None
            ),
            n_full_atoms=(
                transform(self.n_full_atoms)
                if self.n_full_atoms is not None
                else None
            ),
            symmetry_ops=(
                transform(self.symmetry_ops)
                if getattr(self, "symmetry_ops", None) is not None
                else None
            ),
            symmetry_mask=(
                transform(self.symmetry_mask)
                if getattr(self, "symmetry_mask", None) is not None
                else None
            ),
            orbit_plan=orbit_plan,
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "AsuBatch":
        return self._map_tensors(
            lambda value: value.to(device=device, non_blocking=non_blocking)
        )

    def pin_memory(self) -> "AsuBatch":
        """Return a pinned copy suitable for asynchronous CUDA transfer."""
        return self._map_tensors(lambda value: value.pin_memory())

    @property
    def batch_size(self) -> int:
        return len(self.num_atoms)

    @property
    def total_atoms(self) -> int:
        return len(self.atom_types)

    @property
    def total_edges(self) -> int:
        return self.edge_index_conv.shape[1]

    @property
    def total_components(self) -> int:
        return int(self.component_id.max().item()) + 1 if len(self.component_id) > 0 else 0


@dataclass
class CifData:

    atom_labels: list[str]
    atom_types: list[int]
    frac_coords: Tensor
    lattice: Tensor
    bonds: list[tuple[str, str, float, float]]
    spacegroup: int
    cif_id: str = ""
    chemical_formula_moiety: Optional[str] = None
    chemical_formula_sum: Optional[str] = None
    atom_formal_charges: list[int] = field(default_factory=list)
    atom_formal_charge_mask: list[bool] = field(default_factory=list)
    atom_hybridizations: list[str] = field(default_factory=list)
    n_symops: int = 1
    symop_xyz: tuple[str, ...] = field(default_factory=tuple)
    num_symmetry_bonds: int = 0
    hall_number: int = -1
    setting_id: int = -1
    lattice_template_id: int = -1
    # Chemical evidence fields are appended to preserve the positional layout
    # of the pre-existing CifData constructor.
    cell_formula_units_z: Optional[int] = None
    atom_occupancies: list[Optional[float]] = field(default_factory=list)
    atom_site_multiplicities: list[Optional[int]] = field(default_factory=list)
    atom_attached_hydrogens: list[int] = field(default_factory=list)
    atom_attached_hydrogen_mask: list[bool] = field(default_factory=list)
    atom_disorder_assemblies: list[str] = field(default_factory=list)
    atom_disorder_groups: list[str] = field(default_factory=list)
    has_disorder: bool = False
    bond_order_authoritative: list[bool] = field(default_factory=list)
    bond_order_sources: list[str] = field(default_factory=list)
    bond_valences: list[Optional[float]] = field(default_factory=list)
    bond_symmetry_1: list[str] = field(default_factory=list)
    bond_symmetry_2: list[str] = field(default_factory=list)


@dataclass
class CifChemicalEvidence:
    """Chemical CIF fields that can be read without resolving its Hall setting."""

    atom_labels: list[str]
    atom_types: list[int]
    frac_coords: Tensor
    bonds: list[tuple[str, str, float, float]]
    cif_id: str = ""
    chemical_formula_moiety: Optional[str] = None
    chemical_formula_sum: Optional[str] = None
    cell_formula_units_z: Optional[int] = None
    atom_formal_charges: list[int] = field(default_factory=list)
    atom_formal_charge_mask: list[bool] = field(default_factory=list)
    atom_hybridizations: list[str] = field(default_factory=list)
    atom_occupancies: list[Optional[float]] = field(default_factory=list)
    atom_site_multiplicities: list[Optional[int]] = field(default_factory=list)
    atom_attached_hydrogens: list[int] = field(default_factory=list)
    atom_attached_hydrogen_mask: list[bool] = field(default_factory=list)
    atom_disorder_assemblies: list[str] = field(default_factory=list)
    atom_disorder_groups: list[str] = field(default_factory=list)
    has_disorder: bool = False
    bond_order_authoritative: list[bool] = field(default_factory=list)
    bond_order_sources: list[str] = field(default_factory=list)
    bond_valences: list[Optional[float]] = field(default_factory=list)
    bond_symmetry_1: list[str] = field(default_factory=list)
    bond_symmetry_2: list[str] = field(default_factory=list)
    num_symmetry_bonds: int = 0


ELEMENT_TO_ATOMIC_NUMBER: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}

ATOMIC_NUMBER_TO_ELEMENT: dict[int, str] = {
    atomic_number: symbol for symbol, atomic_number in ELEMENT_TO_ATOMIC_NUMBER.items()
}

HYBRIDIZATION_TO_INDEX: dict[str, int] = {
    "unknown": 0,
    "s": 1,
    "sp": 2,
    "sp2": 3,
    "sp3": 4,
    "sp3d": 5,
    "sp3d2": 6,
    "coord": 7,
}


def get_atomic_number(symbol: str) -> int:
    import re

    raw = str(symbol).strip().strip("'\"")
    if raw in {"D", "T"}:
        return ELEMENT_TO_ATOMIC_NUMBER["H"]
    match = re.match(r"([A-Z][a-z]?)", raw)
    if not match:
        raise ValueError(f"Unknown element symbol: {symbol}")
    normalized = match.group(1)
    if normalized in ELEMENT_TO_ATOMIC_NUMBER:
        return ELEMENT_TO_ATOMIC_NUMBER[normalized]
    raise ValueError(f"Unknown element symbol: {symbol}")


def get_element_symbol(atomic_number: int) -> str:
    if atomic_number in ATOMIC_NUMBER_TO_ELEMENT:
        return ATOMIC_NUMBER_TO_ELEMENT[atomic_number]
    raise ValueError(f"Unknown atomic number: {atomic_number}")


def encode_hybridization(value: str) -> int:
    key = str(value or "unknown").strip().lower()
    return HYBRIDIZATION_TO_INDEX.get(key, HYBRIDIZATION_TO_INDEX["unknown"])
