"""Hard-symmetry coordinate projection for known-graph + Target_PG baselines."""

from __future__ import annotations

from typing import Any

import numpy as np


PROJECTION_SCHEMA_VERSION = "graph-pg-3d-reynolds-projection-v1"


def center_positions(positions: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("positions 必须为 [N,3]")
    if not np.isfinite(values).all():
        raise ValueError("positions 含 NaN/Inf")
    return values - values.mean(axis=0, keepdims=True)


def add_centered_gaussian_noise(
    positions: np.ndarray, *, sigma_angstrom: float, seed: int
) -> np.ndarray:
    if sigma_angstrom <= 0:
        raise ValueError("noise sigma 必须为正数")
    values = center_positions(positions)
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, float(sigma_angstrom), size=values.shape)
    return center_positions(values + noise)


def validate_group_action(
    operation_matrices: np.ndarray,
    permutation_index: np.ndarray,
    *,
    atom_count: int,
    orthogonality_tolerance: float = 1e-6,
) -> None:
    matrices = np.asarray(operation_matrices, dtype=np.float64)
    permutations = np.asarray(permutation_index, dtype=np.int64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or not len(matrices):
        raise ValueError("operation matrices 必须为 [G,3,3]")
    if permutations.shape != (len(matrices), atom_count):
        raise ValueError("permutation index shape 与 operations/atom_count 不一致")
    if not np.isfinite(matrices).all():
        raise ValueError("operation matrices 含 NaN/Inf")
    identity = np.arange(atom_count, dtype=np.int64)
    for permutation in permutations:
        if not np.array_equal(np.sort(permutation), identity):
            raise ValueError("permutation 不是原子双射")
    errors = np.max(
        np.abs(np.swapaxes(matrices, 1, 2) @ matrices - np.eye(3)), axis=(1, 2)
    )
    if float(errors.max()) > orthogonality_tolerance:
        raise ValueError("operation matrix 不是正交矩阵")


def project_reynolds(
    positions: np.ndarray,
    operation_matrices: np.ndarray,
    permutation_index: np.ndarray,
    *,
    iterations: int = 1,
) -> np.ndarray:
    """Apply the Reynolds average for ``X @ R.T = X[permutation]``.

    Each transformed source atom is accumulated at its operation-mapped target
    atom.  For an exact finite group representation one pass is the orthogonal
    projection onto the invariant coordinate subspace.  ``iterations`` is
    explicit and frozen so non-idempotent annotations cannot be silently fixed.
    """

    values = center_positions(positions)
    matrices = np.asarray(operation_matrices, dtype=np.float64)
    permutations = np.asarray(permutation_index, dtype=np.int64)
    validate_group_action(matrices, permutations, atom_count=len(values))
    if iterations <= 0:
        raise ValueError("projection iterations 必须为正数")
    projected = values
    for _ in range(iterations):
        accumulator = np.zeros_like(projected)
        for matrix, permutation in zip(matrices, permutations):
            accumulator[permutation] += projected @ matrix.T
        projected = center_positions(accumulator / len(matrices))
    if not np.isfinite(projected).all():
        raise RuntimeError("projected coordinates 含 NaN/Inf")
    return projected


def operation_errors(
    positions: np.ndarray,
    operation_matrices: np.ndarray,
    permutation_index: np.ndarray,
) -> dict[str, Any]:
    values = center_positions(positions)
    matrices = np.asarray(operation_matrices, dtype=np.float64)
    permutations = np.asarray(permutation_index, dtype=np.int64)
    validate_group_action(matrices, permutations, atom_count=len(values))
    rms_values = []
    max_values = []
    for matrix, permutation in zip(matrices, permutations):
        distances = np.linalg.norm(values @ matrix.T - values[permutation], axis=1)
        rms_values.append(float(np.sqrt(np.mean(np.square(distances)))))
        max_values.append(float(distances.max()))
    rms = np.asarray(rms_values, dtype=np.float64)
    maximum = np.asarray(max_values, dtype=np.float64)
    return {
        "operation_rms": rms,
        "operation_max": maximum,
        "mean_operation_rms_angstrom": float(rms.mean()),
        "max_operation_rms_angstrom": float(rms.max()),
        "max_atom_error_angstrom": float(maximum.max()),
    }


def coordinate_rmsd(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_centered = center_positions(reference)
    candidate_centered = center_positions(candidate)
    if reference_centered.shape != candidate_centered.shape:
        raise ValueError("coordinate RMSD shape 不一致")
    return float(
        np.sqrt(np.mean(np.sum(np.square(reference_centered - candidate_centered), axis=1)))
    )


def bond_length_mae(
    reference: np.ndarray, candidate: np.ndarray, bond_index: np.ndarray
) -> float:
    reference_centered = center_positions(reference)
    candidate_centered = center_positions(candidate)
    edges = np.asarray(bond_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2 or not edges.shape[1]:
        raise ValueError("bond_index 必须为非空 [2,M]")
    ref_lengths = np.linalg.norm(
        reference_centered[edges[0]] - reference_centered[edges[1]], axis=1
    )
    candidate_lengths = np.linalg.norm(
        candidate_centered[edges[0]] - candidate_centered[edges[1]], axis=1
    )
    return float(np.mean(np.abs(candidate_lengths - ref_lengths)))


def minimum_pair_distance(positions: np.ndarray) -> float:
    values = center_positions(positions)
    if len(values) < 2:
        raise ValueError("minimum pair distance 至少需要两个原子")
    delta = values[:, None, :] - values[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    upper = distances[np.triu_indices(len(values), k=1)]
    return float(upper.min())
