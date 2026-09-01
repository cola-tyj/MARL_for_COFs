"""Finite point-group actions used by PG-OrbitFlow.

The frozen convention is ``X @ R_g.T == X[p_g]``.  All functions fail on an
invalid action; none repairs atom assignments or substitutes an element.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GROUP_SCHEMA_VERSION = "pg-orbitflow-group-action-v1"


@dataclass(frozen=True)
class GroupAudit:
    order: int
    atom_count: int
    orbit_count: int
    identity_index: int
    improper_fraction: float
    closure_max_matrix_error: float


def _as_action(
    matrices: np.ndarray, permutations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.asarray(matrices, dtype=np.float64)
    perm = np.asarray(permutations, dtype=np.int64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or not len(rotations):
        raise ValueError("operation_matrices must have shape [G,3,3]")
    if perm.ndim != 2 or perm.shape[0] != len(rotations) or perm.shape[1] == 0:
        raise ValueError("permutation_index must have shape [G,N]")
    if not np.isfinite(rotations).all():
        raise ValueError("operation_matrices contains NaN/Inf")
    expected = np.arange(perm.shape[1], dtype=np.int64)
    if any(not np.array_equal(np.sort(row), expected) for row in perm):
        raise ValueError("permutation_index contains a non-bijection")
    return rotations, perm


def validate_group_action(
    matrices: np.ndarray,
    permutations: np.ndarray,
    *,
    atomic_numbers: np.ndarray | None = None,
    orbit_id: np.ndarray | None = None,
    orthogonality_tolerance: float = 2e-5,
    closure_tolerance: float = 2e-4,
) -> GroupAudit:
    """Validate orthogonality, identity, paired closure and atom/orbit semantics."""

    rotations, perm = _as_action(matrices, permutations)
    atom_count = perm.shape[1]
    eye = np.eye(3, dtype=np.float64)
    orth_error = np.max(
        np.abs(np.swapaxes(rotations, 1, 2) @ rotations - eye), axis=(1, 2)
    )
    if float(orth_error.max()) > orthogonality_tolerance:
        raise ValueError("operation_matrices contains a non-orthogonal matrix")
    determinants = np.linalg.det(rotations)
    if float(np.max(np.abs(np.abs(determinants) - 1.0))) > orthogonality_tolerance:
        raise ValueError("operation determinant is not +/-1")

    identity_errors = np.max(np.abs(rotations - eye), axis=(1, 2))
    identity = np.arange(atom_count, dtype=np.int64)
    identity_candidates = np.flatnonzero(
        (identity_errors <= closure_tolerance)
        & np.asarray([np.array_equal(row, identity) for row in perm])
    )
    if len(identity_candidates) != 1:
        raise ValueError("group action must contain exactly one paired identity")

    numbers = None if atomic_numbers is None else np.asarray(atomic_numbers, dtype=np.int64)
    if numbers is not None:
        if numbers.shape != (atom_count,):
            raise ValueError("atomic_numbers shape does not match group action")
        for row in perm:
            if not np.array_equal(numbers, numbers[row]):
                raise ValueError("group permutation maps across elements")

    orbits = None if orbit_id is None else np.asarray(orbit_id, dtype=np.int64)
    if orbits is not None:
        if orbits.shape != (atom_count,):
            raise ValueError("orbit_id shape does not match group action")
        unique = np.unique(orbits)
        if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
            raise ValueError("orbit_id must be contiguous from zero")
        for row in perm:
            if not np.array_equal(orbits, orbits[row]):
                raise ValueError("group permutation maps across atomic orbits")

    closure_max = 0.0
    for left in range(len(rotations)):
        for right in range(len(rotations)):
            # With X R.T = X[p], applying left and then right yields
            # R_product = R_right R_left and p_product = p_right[p_left].
            product_matrix = rotations[right] @ rotations[left]
            product_perm = perm[right][perm[left]]
            matching_perm = np.asarray(
                [np.array_equal(row, product_perm) for row in perm], dtype=bool
            )
            if not matching_perm.any():
                raise ValueError("permutation action is not closed")
            errors = np.max(np.abs(rotations - product_matrix), axis=(1, 2))
            paired_error = float(errors[matching_perm].min())
            closure_max = max(closure_max, paired_error)
            if paired_error > closure_tolerance:
                raise ValueError("matrix and permutation actions are not paired-closed")

    orbit_count = atom_count if orbits is None else len(np.unique(orbits))
    return GroupAudit(
        order=len(rotations),
        atom_count=atom_count,
        orbit_count=orbit_count,
        identity_index=int(identity_candidates[0]),
        improper_fraction=float(np.mean(determinants < 0.0)),
        closure_max_matrix_error=closure_max,
    )


def project_vectors_numpy(
    vectors: np.ndarray, matrices: np.ndarray, permutations: np.ndarray
) -> np.ndarray:
    """Reynolds average a full atom-vector field under the paired action."""

    rotations, perm = _as_action(matrices, permutations)
    values = np.asarray(vectors, dtype=np.float64)
    if values.shape != (perm.shape[1], 3) or not np.isfinite(values).all():
        raise ValueError("vectors must be finite [N,3]")
    result = np.zeros_like(values)
    for rotation, row in zip(rotations, perm, strict=True):
        result[row] += values @ rotation.T
    result /= len(rotations)
    return result - result.mean(axis=0, keepdims=True)


def operation_error_numpy(
    vectors: np.ndarray, matrices: np.ndarray, permutations: np.ndarray
) -> dict[str, float]:
    rotations, perm = _as_action(matrices, permutations)
    values = np.asarray(vectors, dtype=np.float64)
    if values.shape != (perm.shape[1], 3) or not np.isfinite(values).all():
        raise ValueError("vectors must be finite [N,3]")
    atom_errors = np.stack(
        [np.linalg.norm(values @ r.T - values[p], axis=1) for r, p in zip(rotations, perm)]
    )
    operation_rms = np.sqrt(np.mean(np.square(atom_errors), axis=1))
    return {
        "mean_operation_rms_angstrom": float(operation_rms.mean()),
        "max_operation_rms_angstrom": float(operation_rms.max()),
        "max_atom_error_angstrom": float(atom_errors.max()),
    }


def group_features(
    matrices: np.ndarray, permutations: np.ndarray, orbit_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return global character-like and per-atom orbit/stabilizer features.

    The feature map deliberately includes parity and trace moments so improper
    operations are distinguishable from pure rotations.  Per-atom features are
    constant on each orbit and therefore do not break the supplied action.
    """

    rotations, perm = _as_action(matrices, permutations)
    orbits = np.asarray(orbit_id, dtype=np.int64)
    audit = validate_group_action(rotations, perm, orbit_id=orbits)
    determinants = np.linalg.det(rotations)
    traces = np.trace(rotations, axis1=1, axis2=2)
    fixed_fraction = np.mean(perm == np.arange(audit.atom_count)[None, :], axis=1)
    global_values = np.asarray(
        [
            np.log1p(audit.order),
            np.mean(determinants < 0.0),
            np.mean(traces) / 3.0,
            np.std(traces) / 3.0,
            np.min(traces) / 3.0,
            np.max(traces) / 3.0,
            np.mean(fixed_fraction),
            np.std(fixed_fraction),
        ],
        dtype=np.float32,
    )
    atom_values = np.zeros((audit.atom_count, 4), dtype=np.float32)
    for current_orbit in range(audit.orbit_count):
        members = np.flatnonzero(orbits == current_orbit)
        representative = int(members[0])
        stabilizer_size = int(np.sum(perm[:, representative] == representative))
        atom_values[members] = np.asarray(
            [
                len(members) / audit.order,
                stabilizer_size / audit.order,
                np.log1p(len(members)),
                np.log1p(stabilizer_size),
            ],
            dtype=np.float32,
        )
    return global_values, atom_values


def project_vectors_torch(vectors, matrices, permutations):
    """Differentiable full-atom Reynolds average used only for target/noise prep.

    PG-OrbitFlow's model output is not passed through this function.  The model
    uses an equivariant complete-pair vector field so that an invariant input
    stays in the invariant subspace by construction.
    """

    import torch

    if vectors.ndim != 2 or vectors.shape[-1] != 3:
        raise ValueError("vectors must have shape [N,3]")
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("matrices must have shape [G,3,3]")
    if permutations.shape != (len(matrices), len(vectors)):
        raise ValueError("permutations shape mismatch")
    result = torch.zeros_like(vectors)
    for rotation, row in zip(matrices, permutations):
        result.index_add_(0, row, vectors @ rotation.T)
    result = result / len(matrices)
    return result - result.mean(dim=0, keepdim=True)
