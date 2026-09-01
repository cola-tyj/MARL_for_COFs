"""Model-independent raw-coordinate metrics for PG-OrbitFlow gates."""

from __future__ import annotations

import numpy as np

from .group import operation_error_numpy


METRICS_SCHEMA_VERSION = "pg-orbitflow-raw-geometry-metrics-v1"

_COVALENT_RADII_ANGSTROM = {
    1: 0.31,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    35: 1.20,
    50: 1.40,
    53: 1.39,
}


def _center(values: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(values, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not len(coordinates):
        raise ValueError("coordinates must be non-empty [N,3]")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates contains NaN/Inf")
    return coordinates - coordinates.mean(axis=0, keepdims=True)


def kabsch_rmsd(reference: np.ndarray, candidate: np.ndarray) -> float:
    """SO(3)-aligned RMSD; reflections are not allowed."""

    target = _center(reference)
    predicted = _center(candidate)
    if target.shape != predicted.shape:
        raise ValueError("Kabsch coordinate shape mismatch")
    covariance = predicted.T @ target
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    aligned = predicted @ rotation
    return float(np.sqrt(np.mean(np.sum(np.square(aligned - target), axis=1))))


def pair_distance_mae(reference: np.ndarray, candidate: np.ndarray) -> float:
    target = _center(reference)
    predicted = _center(candidate)
    if target.shape != predicted.shape or len(target) < 2:
        raise ValueError("pair-distance coordinate shape mismatch")
    upper = np.triu_indices(len(target), k=1)
    target_distances = np.linalg.norm(target[:, None] - target[None, :], axis=-1)[upper]
    predicted_distances = np.linalg.norm(
        predicted[:, None] - predicted[None, :], axis=-1
    )[upper]
    return float(np.mean(np.abs(predicted_distances - target_distances)))


def bond_length_mae(
    reference: np.ndarray, candidate: np.ndarray, bond_index: np.ndarray
) -> float:
    target = _center(reference)
    predicted = _center(candidate)
    edges = np.asarray(bond_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2 or not edges.shape[1]:
        raise ValueError("bond_index must be non-empty [2,M]")
    if int(edges.min()) < 0 or int(edges.max()) >= len(target):
        raise ValueError("bond_index out of range")
    target_lengths = np.linalg.norm(target[edges[0]] - target[edges[1]], axis=1)
    predicted_lengths = np.linalg.norm(
        predicted[edges[0]] - predicted[edges[1]], axis=1
    )
    return float(np.mean(np.abs(predicted_lengths - target_lengths)))


def collision_audit(
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    bond_index: np.ndarray,
    *,
    radius_fraction: float = 0.60,
) -> dict[str, float | bool]:
    values = _center(coordinates)
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    edges = np.asarray(bond_index, dtype=np.int64)
    if numbers.shape != (len(values),):
        raise ValueError("atomic_numbers shape mismatch")
    radii = []
    for number in numbers:
        if int(number) not in _COVALENT_RADII_ANGSTROM:
            raise ValueError(f"no collision radius for Z={int(number)}")
        radii.append(_COVALENT_RADII_ANGSTROM[int(number)])
    radii = np.asarray(radii, dtype=np.float64)
    bonded = np.zeros((len(values), len(values)), dtype=bool)
    bonded[edges[0], edges[1]] = True
    bonded[edges[1], edges[0]] = True
    upper = np.triu_indices(len(values), k=1)
    keep = ~bonded[upper]
    left, right = upper[0][keep], upper[1][keep]
    if not len(left):
        raise ValueError("collision audit has no non-bonded pairs")
    distances = np.linalg.norm(values[left] - values[right], axis=1)
    ratios = distances / (radii[left] + radii[right])
    minimum_index = int(np.argmin(ratios))
    return {
        "collision_free": bool(ratios[minimum_index] >= radius_fraction),
        "minimum_nonbonded_distance_angstrom": float(distances[minimum_index]),
        "minimum_covalent_radius_ratio": float(ratios[minimum_index]),
        "collision_threshold_ratio": float(radius_fraction),
    }


def raw_geometry_metrics(sample, coordinates: np.ndarray) -> dict[str, float | bool]:
    candidate = _center(coordinates)
    reference = _center(sample.symmetric_target_angstrom)
    if candidate.shape != reference.shape:
        raise ValueError("raw generated atom count changed")
    operation = operation_error_numpy(
        candidate, sample.operation_matrices, sample.permutation_index
    )
    collision = collision_audit(
        candidate, sample.atomic_numbers, sample.bond_index
    )
    return {
        "kabsch_rmsd_angstrom": kabsch_rmsd(reference, candidate),
        "pair_distance_mae_angstrom": pair_distance_mae(reference, candidate),
        "bond_length_mae_angstrom": bond_length_mae(
            reference, candidate, sample.bond_index
        ),
        **operation,
        **collision,
    }
