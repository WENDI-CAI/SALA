"""Convert a molecular SMILES into SALA's heavy-atom conditioning graph.

The geometry fields in :class:`AsuInput` are required by the shared data schema.
This adapter fills them with nonphysical placeholders, without constructing a
molecular conformer or reading a reference crystal. Generation draws its initial
coordinates and lattice from the sampler's prior.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from numbers import Integral
from typing import Any

import torch

from sala.core.graph import compute_connected_components
from sala.core.lattice_manifold import lattice_template_from_hall_number, q_to_lattice
from sala.core.orbit import UnsupportedSpecialPositionError, validate_general_position_orbits
from sala.core.symmetry import canonicalize_affine_symmetry_ops, get_sym_ops_from_hall
from sala.core.types import AsuInput, encode_hybridization
from sala.data.chemistry import assign_component_type_ids
from sala.data.datasets.pt_dataset import bond_order_to_idx


MAX_FULL_CELL_HEAVY_ATOMS = 1997


def _integer(value: Any, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {bound}, got {result}")
    return result


@lru_cache(maxsize=230)
def _default_hall_number(spacegroup: int) -> int:
    import spglib

    for hall in range(1, 531):
        group = spglib.get_spacegroup_type(hall)
        if group is not None and int(group.number) == spacegroup:
            return hall
    raise ValueError(f"spglib could not resolve space group {spacegroup}")


def _placeholder_geometry(
    num_atoms: int,
    spacegroup: torch.Tensor,
    template_id: torch.Tensor,
    symmetry_ops: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # A private generator keeps preprocessing independent of the sampling RNG.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(16):
        # Keep every bond inside one periodic image, including molecular rings.
        # These arbitrary points have no molecular geometry or packing meaning.
        frac = torch.tensor([0.217, 0.361, 0.493]) + 0.08 * torch.rand(
            (num_atoms, 3), generator=generator, dtype=torch.float32
        )
        try:
            validate_general_position_orbits(frac, symmetry_ops)
        except UnsupportedSpecialPositionError:
            continue
        break
    else:
        raise ValueError("Could not construct general-position schema placeholders")
    q = torch.zeros((1, 6), dtype=torch.float32)
    q[:, :3] = math.log(10.0)
    lattice = q_to_lattice(
        q, spacegroup=spacegroup, lattice_template_id=template_id
    )[0]
    return frac, lattice


def prepare_smiles(
    smiles: str,
    *,
    spacegroup: int = 1,
    hall_number: int | None = None,
    copies: int = 1,
    seed: int = 42,
    target_id: str = "smiles_target",
) -> tuple[AsuInput, dict[str, Any]]:
    """Build conditioning and JSON-serializable provenance from one SMILES.

    ``spacegroup`` is a user-selected international number, defaulting to P1.
    SALA does not predict or search for the space group. If ``hall_number`` is
    omitted, the first matching spglib database setting is used and recorded.
    An explicit Hall number must match ``spacegroup``.

    Each dot-separated fragment is a separate chemical component. ``copies``
    repeats the entire formula in the asymmetric unit; symmetry expansion then
    produces ``copies * heavy_atoms * symmetry_operations`` full-cell atoms.
    Only general positions are supported.

    The released model has no stereochemical, isotope, radical, or dative-bond
    channel. Explicit stereochemistry, isotopes, radicals, unsupported bond
    orders, and hydrogen-only fragments are rejected instead of being silently
    discarded. Attached hydrogens are recorded as chemical metadata; the model
    itself generates heavy atoms.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    smiles = smiles.strip()
    if any(character.isspace() for character in smiles):
        raise ValueError("Pass one plain SMILES without a molecule name or whitespace")
    if not isinstance(target_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", target_id
    ):
        raise ValueError("target_id must start with a letter/digit and use only letters, digits, _, -, or .")
    sg_number = _integer(spacegroup, "spacegroup", 1, 230)
    copies = _integer(copies, "copies", 1, MAX_FULL_CELL_HEAVY_ATOMS)
    seed = _integer(seed, "seed", 0, 2**63 - 1)
    if hall_number is not None:
        hall_number = _integer(hall_number, "hall_number", 1, 530)
    try:
        import rdkit
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except ImportError as exc:
        raise ImportError("SMILES input requires RDKit: pip install 'sala-crystal[inference]'") from exc
    try:
        import spglib
    except ImportError as exc:
        raise ImportError("SMILES input requires spglib: pip install spglib") from exc

    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    parameters.parseName = False
    parameters.allowCXSMILES = False
    molecule = Chem.MolFromSmiles(smiles, parameters)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("Invalid SMILES: RDKit could not parse a molecule")
    if "@" in smiles or any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in molecule.GetAtoms()
    ) or any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        or bond.GetBondDir() != Chem.BondDir.NONE
        for bond in molecule.GetBonds()
    ):
        raise ValueError(
            "Stereochemical SMILES are not supported: this checkpoint has no "
            "chirality or E/Z conditioning channel and cannot guarantee the "
            "requested stereochemistry. Supply an achiral target or explicitly "
            "remove stereochemical annotations only if that loss is acceptable."
        )
    if any(atom.GetIsotope() for atom in molecule.GetAtoms()):
        raise ValueError("Isotopic SMILES are not supported by this checkpoint")
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        raise ValueError("Radical SMILES are not supported by this checkpoint")
    if any(not 1 <= atom.GetAtomicNum() <= 118 for atom in molecule.GetAtoms()):
        raise ValueError("SMILES must contain real elements; wildcard/query atoms are unsupported")
    for fragment in Chem.GetMolFrags(molecule, asMols=False):
        if not any(molecule.GetAtomWithIdx(index).GetAtomicNum() > 1 for index in fragment):
            raise ValueError(
                "Hydrogen-only fragments are unsupported; specify protonation "
                "on a heavy atom, for example [NH4+]"
            )
    if any(
        atom.GetAtomicNum() == 1 and (
            atom.GetDegree() != 1 or atom.GetFormalCharge() != 0
        )
        for atom in molecule.GetAtoms()
    ):
        raise ValueError("Only ordinary singly attached hydrogen atoms are supported")

    heavy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    num_base_atoms = len(heavy_atoms)
    if not num_base_atoms:
        raise ValueError("SMILES must contain at least one heavy atom")
    resolved_hall = hall_number if hall_number is not None else _default_hall_number(sg_number)
    hall_spacegroup, template = lattice_template_from_hall_number(resolved_hall)
    if hall_spacegroup != sg_number:
        raise ValueError(
            f"Hall number {resolved_hall} belongs to space group {hall_spacegroup}, "
            f"not the requested space group {sg_number}"
        )
    symmetry_ops = canonicalize_affine_symmetry_ops(
        torch.tensor(get_sym_ops_from_hall(resolved_hall), dtype=torch.float32)
    )
    n_symops = int(symmetry_ops.shape[0])
    num_atoms = num_base_atoms * copies
    n_full_atoms = num_atoms * n_symops
    if n_full_atoms > MAX_FULL_CELL_HEAVY_ATOMS:
        raise ValueError(
            f"Requested full cell has {n_full_atoms} heavy atoms "
            f"({num_base_atoms} per SMILES × {copies} copies × {n_symops} operations); "
            f"this adapter limits full cells to {MAX_FULL_CELL_HEAVY_ATOMS}, "
            "matching the released configuration"
        )

    heavy_index = {atom.GetIdx(): index for index, atom in enumerate(heavy_atoms)}
    supported_bonds = {
        Chem.BondType.SINGLE: 1.0,
        Chem.BondType.AROMATIC: 1.5,
        Chem.BondType.DOUBLE: 2.0,
        Chem.BondType.TRIPLE: 3.0,
    }
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    bond_orders: list[float] = []
    for bond in molecule.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type not in supported_bonds:
            raise ValueError(
                f"Unsupported SMILES bond type {bond_type}; use single, aromatic, double, or triple bonds"
            )
        left = heavy_index.get(bond.GetBeginAtomIdx())
        right = heavy_index.get(bond.GetEndAtomIdx())
        if left is None or right is None:
            continue
        edge_sources.extend((left, right))
        edge_targets.extend((right, left))
        bond_orders.extend((supported_bonds[bond_type],) * 2)
    base_edges = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    base_components = compute_connected_components(base_edges, num_base_atoms)
    base_component_count = int(base_components.max().item()) + 1
    edge_index = torch.cat([base_edges + copy * num_base_atoms for copy in range(copies)], dim=1)
    edge_orders = torch.tensor(bond_orders * copies, dtype=torch.float32)
    edge_attr = bond_order_to_idx(edge_orders).long()
    component_id = torch.cat([
        base_components + copy * base_component_count for copy in range(copies)
    ])
    atom_types = torch.tensor([atom.GetAtomicNum() for atom in heavy_atoms] * copies, dtype=torch.long)
    atom_charge = torch.tensor([atom.GetFormalCharge() for atom in heavy_atoms] * copies, dtype=torch.long)
    if bool((atom_charge.abs() > 2).any()):
        raise ValueError(
            "SMILES atom charges must lie in [-2, 2] to match the checkpoint's "
            "known-charge training representation; use the CIF preprocessing "
            "pipeline for other charge-localization cases"
        )
    atom_charge_mask = torch.ones(num_atoms, dtype=torch.bool)
    hybridization = torch.tensor([
        encode_hybridization(str(atom.GetHybridization())) for atom in heavy_atoms
    ] * copies, dtype=torch.long)
    component_charge = torch.zeros(base_component_count * copies, dtype=torch.long)
    component_charge.scatter_add_(0, component_id, atom_charge)
    if bool((component_charge.abs() > 16).any()):
        raise ValueError("Component formal charges must lie in [-16, 16]; larger charges cannot be encoded")
    for component in range(base_component_count):
        charges = atom_charge[component_id == component]
        if charges.numel() > 1 and abs(int(charges.sum())) > 1 and int(torch.count_nonzero(charges)) == 1:
            raise ValueError(
                "A polyatomic component with charge magnitude above one localized "
                "on a single atom is outside the checkpoint's known-charge training "
                "representation; use the CIF preprocessing pipeline for this case"
            )
    component_type_id = assign_component_type_ids(
        atom_types=atom_types,
        hybridization=hybridization,
        atom_formal_charge=atom_charge,
        atom_charge_mask=atom_charge_mask,
        component_id=component_id,
        component_formal_charge=component_charge,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    labels = [
        f"{atom.GetSymbol()}{copy * num_base_atoms + index + 1}"
        for copy in range(copies) for index, atom in enumerate(heavy_atoms)
    ]
    sg_tensor = torch.tensor([sg_number], dtype=torch.long)
    template_id = torch.tensor([template], dtype=torch.long)
    frac, lattice = _placeholder_geometry(num_atoms, sg_tensor, template_id, symmetry_ops, seed)
    asu = AsuInput(
        atom_types=atom_types,
        spacegroup=sg_tensor,
        component_id=component_id,
        atom_formal_charge=atom_charge,
        atom_charge_mask=atom_charge_mask,
        component_formal_charge=component_charge,
        component_type_id=component_type_id,
        num_atoms=torch.tensor([num_atoms], dtype=torch.long),
        edge_index_conv=edge_index,
        edge_attr_conv=edge_attr,
        frac_coords_gt=frac,
        lattice_gt=lattice,
        cart_coords_gt=frac @ lattice,
        hybridization=hybridization,
        atom_labels=labels,
        cif_id=target_id,
        n_symops=torch.tensor([n_symops], dtype=torch.long),
        n_heavy_cell=torch.tensor([n_full_atoms], dtype=torch.long),
        n_full_atoms=torch.tensor([n_full_atoms], dtype=torch.long),
        hall_number=torch.tensor([resolved_hall], dtype=torch.long),
        setting_id=torch.tensor([resolved_hall], dtype=torch.long),
        lattice_template_id=template_id,
        symmetry_ops=symmetry_ops,
    )
    group_type = spglib.get_spacegroup_type(resolved_hall)
    attached_hydrogens = [int(atom.GetTotalNumHs(includeNeighbors=True)) for atom in heavy_atoms] * copies
    metadata: dict[str, Any] = {
        "input_type": "smiles",
        "target_id": target_id,
        "smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        "molecular_formula": rdMolDescriptors.CalcMolFormula(molecule),
        "copies_in_asymmetric_unit": copies,
        "spacegroup": sg_number,
        "hall_number": resolved_hall,
        "hall_symbol": str(group_type.hall_symbol),
        "spacegroup_setting_choice": str(group_type.choice),
        "hall_selection": "explicit" if hall_number is not None else "first_spglib_setting",
        "spacegroup_source": "user_selected_or_default_P1",
        "n_heavy_atoms": num_atoms,
        "n_symops": n_symops,
        "n_heavy_cell": n_full_atoms,
        "n_full_atoms": n_full_atoms,
        "setting_id": resolved_hall,
        "lattice_template_id": template,
        "n_components": base_component_count * copies,
        "n_component_types": int(component_type_id.unique().numel()),
        "total_formal_charge_asu": int(component_charge.sum().item()),
        "position_policy": "general_positions_only",
        "geometry_source": "nonphysical_schema_placeholder_not_a_reference_structure",
        "generation_initialization": "sampled_from_prior",
        "placeholder_seed": seed,
        "rdkit_version": str(rdkit.__version__),
        "spglib_version": str(spglib.__version__),
        "smiles_chemistry": {
            "source": "rdkit_smiles",
            "atom_labels": labels,
            "atom_types": atom_types.tolist(),
            "component_id": component_id.tolist(),
            "edge_index": edge_index.tolist(),
            "bond_orders": edge_orders.tolist(),
            "bond_order_mask": [True] * edge_orders.numel(),
            "atom_formal_charges": atom_charge.tolist(),
            "atom_charge_mask": atom_charge_mask.tolist(),
            "component_formal_charges": component_charge.tolist(),
            "component_formal_charge_mask": [True] * component_charge.numel(),
            "hybridization": hybridization.tolist(),
            "attached_hydrogens": attached_hydrogens,
            "attached_hydrogen_mask": [True] * num_atoms,
            "n_symops": n_symops,
        },
    }
    return asu, metadata


__all__ = ["MAX_FULL_CELL_HEAVY_ATOMS", "prepare_smiles"]
