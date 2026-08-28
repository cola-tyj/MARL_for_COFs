"""Explicit resonance-normalized graph action for ETKDG orbit initialization.

The only normalization allowed is permutation of electronic Lewis labels among
same-element degree-1 neighbors of the same mapped center.  Connectivity,
elements, center-atom features, and the multiset of terminal bond/charge/radical
labels remain exact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit.Chem import AllChem

from generative_model.models.graph_pg_3d_projection import center_positions
from .etkdg_orbit_initializer import build_coordinate_free_rdkit_graph


SCHEMA_VERSION = "terminal-resonance-compatible-graph-action-v1"


def validate_resonance_compatible_graph_automorphisms(
    sample: dict[str, Any],
) -> dict[str, Any]:
    atom_count = len(sample["atomic_numbers"])
    numbers = np.asarray(sample["atomic_numbers"], dtype=np.int64)
    charges = np.asarray(sample["formal_charges"], dtype=np.int64)
    radicals = np.asarray(sample["radical_electrons"], dtype=np.int64)
    aromaticity = np.asarray(sample["aromaticity"], dtype=np.bool_)
    permutations = np.asarray(sample["target_permutation_index"], dtype=np.int64)
    bonds = np.asarray(sample["bond_index"], dtype=np.int64)
    bond_types = np.asarray(sample["bond_types"], dtype=np.int64)
    if any(len(values) != atom_count for values in (charges, radicals, aromaticity)):
        raise ValueError("canonical electronic atom fields 长度不一致")
    if bonds.shape != (2, len(bond_types)):
        raise ValueError("canonical bond fields shape 不一致")
    edge_types = {
        (min(int(left), int(right)), max(int(left), int(right))): int(kind)
        for (left, right), kind in zip(bonds.T, bond_types)
    }
    if len(edge_types) != len(bond_types):
        raise ValueError("canonical graph 含重复边")
    neighbors: list[list[int]] = [[] for _ in range(atom_count)]
    for left, right in edge_types:
        neighbors[left].append(right)
        neighbors[right].append(left)
    expected_atoms = np.arange(atom_count, dtype=np.int64)
    normalized_operations = 0
    normalized_atom_assignments = 0
    normalized_bond_assignments = 0
    for permutation in permutations:
        if not np.array_equal(np.sort(permutation), expected_atoms):
            raise ValueError("target permutation 不是原子双射")
        if not np.array_equal(numbers, numbers[permutation]):
            raise ValueError("target permutation 跨元素映射")
        mapped_connectivity = {
            tuple(sorted((int(permutation[left]), int(permutation[right]))))
            for left, right in edge_types
        }
        if mapped_connectivity != set(edge_types):
            raise ValueError("target permutation 修改 canonical connectivity")

        allowed_atoms: set[int] = set()
        allowed_bonds: set[tuple[int, int]] = set()
        for center in range(atom_count):
            mapped_center = int(permutation[center])
            groups: dict[int, list[int]] = {}
            for neighbor in neighbors[center]:
                if len(neighbors[neighbor]) == 1:
                    groups.setdefault(int(numbers[neighbor]), []).append(neighbor)
            for atomic_number, terminals in groups.items():
                if len(terminals) < 2:
                    continue
                mapped_terminals = {int(permutation[terminal]) for terminal in terminals}
                target_terminals = {
                    neighbor
                    for neighbor in neighbors[mapped_center]
                    if len(neighbors[neighbor]) == 1
                    and int(numbers[neighbor]) == atomic_number
                }
                if mapped_terminals != target_terminals:
                    continue
                source_signatures = sorted(
                    (
                        edge_types[tuple(sorted((center, terminal)))],
                        int(charges[terminal]),
                        int(radicals[terminal]),
                        bool(aromaticity[terminal]),
                    )
                    for terminal in terminals
                )
                target_signatures = sorted(
                    (
                        edge_types[tuple(sorted((mapped_center, terminal)))],
                        int(charges[terminal]),
                        int(radicals[terminal]),
                        bool(aromaticity[terminal]),
                    )
                    for terminal in target_terminals
                )
                if source_signatures == target_signatures:
                    allowed_atoms.update(terminals)
                    allowed_bonds.update(tuple(sorted((center, terminal))) for terminal in terminals)

        operation_normalized = False
        for atom in range(atom_count):
            mapped = int(permutation[atom])
            source_features = (int(charges[atom]), int(radicals[atom]), bool(aromaticity[atom]))
            target_features = (int(charges[mapped]), int(radicals[mapped]), bool(aromaticity[mapped]))
            if source_features != target_features:
                if atom not in allowed_atoms:
                    raise ValueError("非末端共振等价原子的 electronic labels 被 permutation 修改")
                normalized_atom_assignments += 1
                operation_normalized = True
        for pair, kind in edge_types.items():
            left, right = pair
            mapped_pair = tuple(sorted((int(permutation[left]), int(permutation[right]))))
            if edge_types[mapped_pair] != kind:
                if pair not in allowed_bonds:
                    raise ValueError("非末端共振等价键的 bond type 被 permutation 修改")
                normalized_bond_assignments += 1
                operation_normalized = True
        normalized_operations += int(operation_normalized)
    return {
        "schema_version": SCHEMA_VERSION,
        "operation_count": len(permutations),
        "resonance_normalized_operation_count": normalized_operations,
        "resonance_normalized_atom_assignments": normalized_atom_assignments,
        "resonance_normalized_bond_assignments": normalized_bond_assignments,
    }


def embed_etkdg_resonance_compatible(
    sample: dict[str, Any],
    *,
    seed: int,
    maximum_iterations: int,
    use_random_coordinates: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0 <= int(seed) < 2**31:
        raise ValueError("ETKDG seed 必须位于 signed 32-bit 非负范围")
    if maximum_iterations <= 0:
        raise ValueError("ETKDG maximum_iterations 必须为正")
    audit = validate_resonance_compatible_graph_automorphisms(sample)
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
    return center_positions(coordinates), audit
