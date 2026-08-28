"""Deterministic ETKDG initial conformers for a known graph and orbit action.

The canonical explicit-H graph and atom order are immutable.  ETKDG coordinates
are oriented without reference XYZ, projected into the stored target group
action, and returned for the orbit-constrained force-field backend.
"""

from __future__ import annotations

from itertools import permutations, product
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from generative_model.models.graph_pg_3d_projection import (
    center_positions,
    coordinate_rmsd,
    operation_errors,
    project_reynolds,
)
from .force_field_support import build_strict_rdkit_molecule


SCHEMA_VERSION = "etkdg-orbit-initializer-v1"


def validate_typed_graph_automorphisms(sample: dict[str, Any]) -> None:
    """Require every stored target permutation to preserve typed graph edges."""

    atom_count = len(sample["atomic_numbers"])
    atomic_numbers = np.asarray(sample["atomic_numbers"], dtype=np.int64)
    permutations_array = np.asarray(sample["target_permutation_index"], dtype=np.int64)
    bond_index = np.asarray(sample["bond_index"], dtype=np.int64)
    bond_types = np.asarray(sample["bond_types"], dtype=np.int64)
    if bond_index.shape != (2, len(bond_types)):
        raise ValueError("canonical typed graph shape 非法")
    expected_atoms = np.arange(atom_count, dtype=np.int64)
    expected_edges = {
        (min(int(left), int(right)), max(int(left), int(right)), int(kind))
        for (left, right), kind in zip(bond_index.T, bond_types)
    }
    if len(expected_edges) != len(bond_types):
        raise ValueError("canonical typed graph 含重复键")
    for permutation in permutations_array:
        if not np.array_equal(np.sort(permutation), expected_atoms):
            raise ValueError("target permutation 不是原子双射")
        if not np.array_equal(atomic_numbers, atomic_numbers[permutation]):
            raise ValueError("target permutation 跨元素映射")
        mapped_edges = {
            (
                min(int(permutation[left]), int(permutation[right])),
                max(int(permutation[left]), int(permutation[right])),
                int(kind),
            )
            for (left, right), kind in zip(bond_index.T, bond_types)
        }
        if mapped_edges != expected_edges:
            raise ValueError("target permutation 不是 typed-graph automorphism")


def build_coordinate_free_rdkit_graph(sample: dict[str, Any]) -> Chem.Mol:
    """Build the canonical explicit-H graph without reading reference positions."""

    atom_count = len(sample["atomic_numbers"])
    molecule = build_strict_rdkit_molecule(
        {**sample, "positions": np.zeros((atom_count, 3), dtype=np.float64)}
    )
    molecule.RemoveAllConformers()
    if molecule.GetNumConformers() != 0:
        raise RuntimeError("coordinate-free RDKit graph 仍含 conformer")
    observed_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int64
    )
    observed_charges = np.asarray(
        [atom.GetFormalCharge() for atom in molecule.GetAtoms()], dtype=np.int64
    )
    observed_radicals = np.asarray(
        [atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()], dtype=np.int64
    )
    if not np.array_equal(observed_numbers, np.asarray(sample["atomic_numbers"])):
        raise RuntimeError("ETKDG graph 修改 atomic number/order")
    if not np.array_equal(observed_charges, np.asarray(sample["formal_charges"])):
        raise RuntimeError("ETKDG graph 修改 formal charge/order")
    if not np.array_equal(observed_radicals, np.asarray(sample["radical_electrons"])):
        raise RuntimeError("ETKDG graph 修改 radical/order")
    return molecule


def embed_etkdg(
    sample: dict[str, Any],
    *,
    seed: int,
    maximum_iterations: int,
    use_random_coordinates: bool,
) -> np.ndarray:
    """Embed one explicit-H conformer while preserving canonical atom indices."""

    if not 0 <= int(seed) < 2**31:
        raise ValueError("ETKDG seed 必须位于 signed 32-bit 非负范围")
    if maximum_iterations <= 0:
        raise ValueError("ETKDG maximum_iterations 必须为正")
    validate_typed_graph_automorphisms(sample)
    molecule = build_coordinate_free_rdkit_graph(sample)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed)
    parameters.maxIterations = int(maximum_iterations)
    parameters.useRandomCoords = bool(use_random_coordinates)
    parameters.enforceChirality = True
    parameters.useSmallRingTorsions = True
    parameters.useMacrocycleTorsions = True
    parameters.clearConfs = True
    result = int(AllChem.EmbedMolecule(molecule, parameters))
    if result != 0 or molecule.GetNumConformers() != 1:
        raise RuntimeError(f"ETKDGv3 strict embedding 失败: return_code={result}")
    coordinates = np.asarray(molecule.GetConformer(0).GetPositions(), dtype=np.float64)
    if coordinates.shape != (len(sample["atomic_numbers"]), 3):
        raise RuntimeError("ETKDG coordinates/atom order shape 不一致")
    return center_positions(coordinates)


def proper_axis_orientation_matrices() -> np.ndarray:
    """Return the 24 deterministic orientation-preserving signed permutations."""

    matrices: list[np.ndarray] = []
    for order in permutations(range(3)):
        base = np.eye(3, dtype=np.float64)[:, order]
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = base @ np.diag(signs)
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    matrices.sort(key=lambda value: tuple(value.reshape(-1).tolist()))
    result = np.stack(matrices)
    if result.shape != (24, 3, 3):
        raise RuntimeError("proper axis orientation 数量不为 24")
    return result


def orient_and_project_to_orbits(
    positions: np.ndarray,
    sample: dict[str, Any],
    *,
    match_radius_of_gyration: bool,
) -> dict[str, Any]:
    """PCA-align and Reynolds-project without reference-coordinate alignment."""

    values = center_positions(positions)
    covariance = values.T @ values / len(values)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    frame = eigenvectors[:, order]
    for column in range(3):
        pivot = int(np.argmax(np.abs(frame[:, column])))
        if frame[pivot, column] < 0:
            frame[:, column] *= -1
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1
    principal = values @ frame
    source_radius = float(np.sqrt(np.mean(np.sum(values**2, axis=1))))
    candidates: list[dict[str, Any]] = []
    for orientation_id, matrix in enumerate(proper_axis_orientation_matrices()):
        oriented = center_positions(principal @ matrix)
        projected = project_reynolds(
            oriented,
            sample["target_operation_matrices"],
            sample["target_permutation_index"],
            iterations=1,
        )
        projected_radius = float(np.sqrt(np.mean(np.sum(projected**2, axis=1))))
        if projected_radius <= 1e-8:
            continue
        radius_scale = source_radius / projected_radius if match_radius_of_gyration else 1.0
        projected = center_positions(projected * radius_scale)
        errors = operation_errors(
            projected,
            sample["target_operation_matrices"],
            sample["target_permutation_index"],
        )
        candidates.append(
            {
                "orientation_id": orientation_id,
                "positions": projected,
                "projection_rmsd_angstrom": coordinate_rmsd(oriented, projected),
                "radius_scale": float(radius_scale),
                "source_radius_of_gyration_angstrom": source_radius,
                "projected_radius_of_gyration_angstrom": float(
                    np.sqrt(np.mean(np.sum(projected**2, axis=1)))
                ),
                "maximum_operation_error_angstrom": errors["max_atom_error_angstrom"],
            }
        )
    if not candidates:
        raise RuntimeError("24 个 PCA orientation 均投影为退化坐标")
    selected = min(candidates, key=lambda item: (item["projection_rmsd_angstrom"], item["orientation_id"]))
    return {
        "schema_version": SCHEMA_VERSION,
        **selected,
        "orientation_candidates": len(candidates),
        "pca_eigenvalues": eigenvalues[order].astype(np.float64),
    }
