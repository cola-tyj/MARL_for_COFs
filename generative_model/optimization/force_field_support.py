"""Strict conversion of the canonical graph to RDKit force-field inputs.

Unsupported force-field parameters are an explicit scientific result.  Graph,
elements, charges, radicals, bonds and coordinates are never repaired or
substituted here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


SCHEMA_VERSION = "canonical-rdkit-force-field-input-v1"
BOND_TYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}


def build_strict_rdkit_molecule(sample: dict[str, Any]) -> Chem.Mol:
    """Construct an explicit-H molecule without changing canonical semantics."""

    atomic_numbers = np.asarray(sample["atomic_numbers"], dtype=np.int64)
    formal_charges = np.asarray(sample["formal_charges"], dtype=np.int64)
    radicals = np.asarray(sample["radical_electrons"], dtype=np.int64)
    positions = np.asarray(sample["positions"], dtype=np.float64)
    aromaticity = np.asarray(sample["aromaticity"], dtype=np.bool_)
    bond_index = np.asarray(sample["bond_index"], dtype=np.int64)
    bond_types = np.asarray(sample["bond_types"], dtype=np.int64)
    atom_count = len(atomic_numbers)
    if any(len(values) != atom_count for values in (formal_charges, radicals, aromaticity)):
        raise ValueError("canonical atom fields 长度不一致")
    if positions.shape != (atom_count, 3) or not np.isfinite(positions).all():
        raise ValueError("canonical positions 非法")
    if bond_index.shape != (2, len(bond_types)):
        raise ValueError("canonical sparse bonds shape 非法")
    if np.any(atomic_numbers <= 0):
        raise ValueError("canonical atomic number 非法")

    editable = Chem.RWMol()
    for atomic_number, charge, radical, aromatic in zip(
        atomic_numbers, formal_charges, radicals, aromaticity
    ):
        atom = Chem.Atom(int(atomic_number))
        atom.SetFormalCharge(int(charge))
        atom.SetNumRadicalElectrons(int(radical))
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(bool(aromatic))
        editable.AddAtom(atom)
    seen: set[tuple[int, int]] = set()
    for (begin, end), category in zip(bond_index.T, bond_types):
        begin, end, category = int(begin), int(end), int(category)
        if not (0 <= begin < end < atom_count):
            raise ValueError("canonical bond index 必须为有序无向 pair")
        if (begin, end) in seen:
            raise ValueError("canonical graph 含重复 bond")
        if category not in BOND_TYPES:
            raise ValueError(f"canonical bond type 非法: {category}")
        seen.add((begin, end))
        editable.AddBond(begin, end, BOND_TYPES[category])
        if category == 4:
            editable.GetBondBetweenAtoms(begin, end).SetIsAromatic(True)

    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    conformer = Chem.Conformer(atom_count)
    for index, coordinate in enumerate(positions):
        conformer.SetAtomPosition(index, tuple(map(float, coordinate)))
    molecule.AddConformer(conformer, assignId=True)

    observed_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int64
    )
    observed_charges = np.asarray(
        [atom.GetFormalCharge() for atom in molecule.GetAtoms()], dtype=np.int64
    )
    observed_radicals = np.asarray(
        [atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()], dtype=np.int64
    )
    if not np.array_equal(observed_numbers, atomic_numbers):
        raise RuntimeError("RDKit conversion 修改了 atomic numbers")
    if not np.array_equal(observed_charges, formal_charges):
        raise RuntimeError("RDKit conversion 修改了 formal charges")
    if not np.array_equal(observed_radicals, radicals):
        raise RuntimeError("RDKit conversion 修改了 radical electrons")
    if molecule.GetNumBonds() != len(bond_types):
        raise RuntimeError("RDKit conversion 修改了 bond count")
    return molecule


def force_field_support(molecule: Chem.Mol) -> dict[str, Any]:
    """Return strict setup coverage for MMFF94s and UFF without optimization."""

    mmff_has_all = bool(AllChem.MMFFHasAllMoleculeParams(molecule))
    mmff_setup = False
    if mmff_has_all:
        properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94s")
        force_field = (
            None
            if properties is None
            else AllChem.MMFFGetMoleculeForceField(molecule, properties, confId=0)
        )
        mmff_setup = force_field is not None

    uff_has_all = bool(AllChem.UFFHasAllMoleculeParams(molecule))
    uff_setup = False
    if uff_has_all:
        force_field = AllChem.UFFGetMoleculeForceField(molecule, confId=0)
        uff_setup = force_field is not None
    return {
        "mmff94s_has_all_parameters": mmff_has_all,
        "mmff94s_setup_success": mmff_setup,
        "uff_has_all_parameters": uff_has_all,
        "uff_setup_success": uff_setup,
    }
