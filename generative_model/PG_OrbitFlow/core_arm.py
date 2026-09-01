"""Strict atom-level contract for the future core/representative-arm generator.

Canonical v2 does not contain these atom labels.  This module therefore only
validates an explicitly supplied decomposition; it never infers one from a
template name or silently chooses a substructure match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .group import validate_group_action


CORE_ARM_SCHEMA_VERSION = "pg-orbitflow-core-arm-decomposition-v1"


@dataclass(frozen=True)
class CoreArmDecomposition:
    core_atom_indices: np.ndarray
    arm_atom_indices: tuple[np.ndarray, ...]
    attachment_bonds: np.ndarray
    copy_permutation: np.ndarray
    representative_arm_copy: int


def validate_core_arm_decomposition(
    *,
    atomic_numbers: np.ndarray,
    bond_index: np.ndarray,
    operation_matrices: np.ndarray,
    permutation_index: np.ndarray,
    atom_role: np.ndarray,
    arm_copy_id: np.ndarray,
    representative_arm_copy: int = 0,
) -> CoreArmDecomposition:
    """Validate a one-core, one-attachment-per-arm orbit decomposition.

    ``atom_role`` uses 0=core and 1=arm. ``arm_copy_id`` is -1 on core atoms
    and contiguous from zero on arm atoms.  More general fused/cage structures
    are intentionally outside this first contract and fail explicitly.
    """

    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    edges = np.asarray(bond_index, dtype=np.int64)
    roles = np.asarray(atom_role, dtype=np.int64)
    copies = np.asarray(arm_copy_id, dtype=np.int64)
    atom_count = len(numbers)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("bond_index must be [2,M]")
    if roles.shape != (atom_count,) or copies.shape != (atom_count,):
        raise ValueError("core/arm atom labels shape mismatch")
    if np.any((roles < 0) | (roles > 1)):
        raise ValueError("atom_role must use exactly 0=core, 1=arm")
    core = np.flatnonzero(roles == 0)
    arm = np.flatnonzero(roles == 1)
    if not len(core) or not len(arm):
        raise ValueError("decomposition requires non-empty core and arms")
    if np.any(copies[core] != -1) or np.any(copies[arm] < 0):
        raise ValueError("arm_copy_id semantics mismatch atom_role")
    unique_copies = np.unique(copies[arm])
    if not np.array_equal(unique_copies, np.arange(len(unique_copies), dtype=np.int64)):
        raise ValueError("arm_copy_id must be contiguous from zero")
    if not 0 <= representative_arm_copy < len(unique_copies):
        raise ValueError("representative_arm_copy out of range")
    validate_group_action(
        operation_matrices,
        permutation_index,
        atomic_numbers=numbers,
    )
    permutations = np.asarray(permutation_index, dtype=np.int64)
    core_mask = roles == 0
    arm_indices = tuple(np.flatnonzero(copies == copy) for copy in unique_copies)
    copy_permutation = np.zeros((len(permutations), len(unique_copies)), dtype=np.int64)
    for operation, row in enumerate(permutations):
        if not np.array_equal(core_mask, core_mask[row]):
            raise ValueError("group action maps between core and arm atoms")
        for copy, indices in enumerate(arm_indices):
            mapped = row[indices]
            mapped_copies = np.unique(copies[mapped])
            if len(mapped_copies) != 1:
                raise ValueError("one arm copy maps into multiple arm copies")
            target_copy = int(mapped_copies[0])
            if not np.array_equal(np.sort(mapped), arm_indices[target_copy]):
                raise ValueError("group action does not map complete arm copies")
            copy_permutation[operation, copy] = target_copy
    for row in copy_permutation:
        if not np.array_equal(np.sort(row), np.arange(len(unique_copies))):
            raise ValueError("group action on arm copies is not a bijection")
    reached = np.unique(copy_permutation[:, representative_arm_copy])
    if not np.array_equal(reached, np.arange(len(unique_copies))):
        raise ValueError("arm copies do not form one transitive attachment orbit")

    attachment = []
    per_copy_count = np.zeros(len(unique_copies), dtype=np.int64)
    for left, right in edges.T:
        left, right = int(left), int(right)
        if roles[left] == roles[right]:
            continue
        core_atom, arm_atom = (left, right) if roles[left] == 0 else (right, left)
        attachment.append((core_atom, arm_atom))
        per_copy_count[int(copies[arm_atom])] += 1
    if not attachment or np.any(per_copy_count != 1):
        raise ValueError("first core-arm contract requires exactly one attachment bond per arm copy")
    attachment_array = np.asarray(sorted(attachment), dtype=np.int64).T
    return CoreArmDecomposition(
        core_atom_indices=core,
        arm_atom_indices=arm_indices,
        attachment_bonds=attachment_array,
        copy_permutation=copy_permutation,
        representative_arm_copy=int(representative_arm_copy),
    )
