"""
Point group computation for molecules.

Determines the molecular point group from 3D atomic coordinates.
Used for:
1. Labeling QM9 molecules with symmetry types for conditional training
2. Verifying that generated COF building blocks have the correct symmetry

Algorithm:
1. Compute center of mass and shift to origin
2. Compute inertia tensor and diagonalize to get principal axes
3. For each candidate point group, test all symmetry operations
4. Return the highest-symmetry group that matches within tolerance

Alternative: Use pymatgen's PointGroupAnalyzer when available.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# Atomic masses for inertia tensor computation (most common elements)
ATOMIC_MASSES = {
    1: 1.008,    # H
    6: 12.011,   # C
    7: 14.007,   # N
    8: 15.999,   # O
    9: 18.998,   # F
    16: 32.065,  # S
    17: 35.453,  # Cl
    35: 79.904,  # Br
    14: 28.085,  # Si
}

# Candidate point groups ordered by symmetry (high to low)
# We test from highest symmetry to lowest, stopping at first match
CANDIDATE_POINT_GROUPS = [
    "D6h", "D4h", "D3h", "D2h",
    "C6v", "C4v", "C3v", "C2v",
    "C6", "C4", "C3", "C2",
    "C2h", "C1",
]


def _get_mass(atomic_num: int) -> float:
    """Get atomic mass for a given atomic number."""
    return ATOMIC_MASSES.get(atomic_num, 12.0)


def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Generate a rotation matrix for rotation around axis by angle in degrees."""
    angle = np.radians(angle_deg)
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    c = np.cos(angle)
    s = np.sin(angle)
    v = 1 - c

    x, y, z = axis
    return np.array([
        [c + x*x*v,     x*y*v - z*s,   x*z*v + y*s],
        [y*x*v + z*s,   c + y*y*v,     y*z*v - x*s],
        [z*x*v - y*s,   z*y*v + x*s,   c + z*z*v],
    ])


def _reflection_matrix(axis: np.ndarray) -> np.ndarray:
    """Generate a reflection matrix across plane perpendicular to axis."""
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    return np.eye(3) - 2 * np.outer(axis, axis)


def _inversion_matrix() -> np.ndarray:
    """Generate an inversion matrix."""
    return -np.eye(3)


def compute_point_group(
    atomic_numbers: Tensor,
    positions: Tensor,
    tolerance: float = 0.3,
) -> str:
    """
    Compute the molecular point group.

    Args:
        atomic_numbers: (N,) atomic numbers
        positions: (N, 3) cartesian coordinates in Angstroms
        tolerance: RMSD tolerance for symmetry match (Angstroms)

    Returns:
        point_group: string label (e.g., "D3h", "C2v", "C1")
    """
    # Convert to numpy
    if isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = atomic_numbers.cpu().numpy()
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()

    n_atoms = len(atomic_numbers)
    if n_atoms <= 1:
        return "C1"

    # Get masses
    masses = np.array([_get_mass(z) for z in atomic_numbers])

    # Center of mass
    com = np.average(positions, axis=0, weights=masses)
    positions_centered = positions - com

    # Inertia tensor
    I = np.zeros((3, 3))
    for i in range(n_atoms):
        r = positions_centered[i]
        I += masses[i] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    I /= masses.sum()

    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(I)
    # Sort by eigenvalue (I_zz >= I_yy >= I_xx)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    # Principal axes
    axis_z = eigenvectors[:, 0]  # Principal axis (highest moment)
    axis_y = eigenvectors[:, 1]
    axis_x = eigenvectors[:, 2]

    principal_axes = [axis_z, axis_y, axis_x]

    # Test each candidate point group
    for pg in CANDIDATE_POINT_GROUPS:
        if _check_point_group(
            positions_centered, atomic_numbers, pg, principal_axes, tolerance
        ):
            return pg

    return "C1"


def _check_point_group(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    point_group: str,
    principal_axes: List[np.ndarray],
    tolerance: float,
) -> bool:
    """
    Check if a molecule matches a specific point group.

    Tests all symmetry operations of the group and checks whether
    the transformed coordinates match the original.
    """
    axis_z, axis_y, axis_x = principal_axes
    operations = _get_symmetry_operations(point_group, axis_z, axis_y, axis_x)

    all_match = True
    for op_matrix in operations:
        transformed = positions @ op_matrix.T

        # Find nearest-neighbor match for each atom
        max_dist = 0.0
        matched = set()
        for i in range(len(positions)):
            z_i = atomic_numbers[i]
            best_dist = float("inf")
            for j in range(len(positions)):
                if j in matched or atomic_numbers[j] != z_i:
                    continue
                dist = np.linalg.norm(transformed[i] - positions[j])
                if dist < best_dist:
                    best_dist = dist
            if best_dist < tolerance:
                matched.add(i)
                max_dist = max(max_dist, best_dist)
            else:
                all_match = False
                break
        if not all_match:
            break

    return all_match


def _get_symmetry_operations(
    point_group: str,
    axis_z: np.ndarray,
    axis_y: np.ndarray,
    axis_x: np.ndarray,
) -> List[np.ndarray]:
    """
    Generate all symmetry operation matrices for a point group.

    Returns a list of 3x3 transformation matrices.
    """
    ops = [np.eye(3)]  # Always include identity

    if point_group == "C1":
        pass  # Only identity

    elif point_group == "C2":
        ops.append(_rotation_matrix(axis_z, 180))

    elif point_group == "C2v":
        ops.append(_rotation_matrix(axis_z, 180))
        ops.append(_reflection_matrix(axis_x))
        ops.append(_reflection_matrix(axis_y))

    elif point_group == "C2h":
        ops.append(_rotation_matrix(axis_z, 180))
        ops.append(_reflection_matrix(axis_z))
        ops.append(_inversion_matrix())

    elif point_group == "C3":
        ops.append(_rotation_matrix(axis_z, 120))
        ops.append(_rotation_matrix(axis_z, 240))

    elif point_group == "C3v":
        for angle in [120, 240]:
            ops.append(_rotation_matrix(axis_z, angle))
        # 3 vertical mirror planes
        for angle in [0, 120, 240]:
            v = np.cos(np.radians(angle)) * axis_x + np.sin(np.radians(angle)) * axis_y
            ops.append(_reflection_matrix(v))

    elif point_group == "D3h":
        # C3 rotations
        for angle in [120, 240]:
            ops.append(_rotation_matrix(axis_z, angle))
        # C2' rotations (3 perpendicular C2 axes)
        for angle in [0, 120, 240]:
            axis = np.cos(np.radians(angle)) * axis_x + np.sin(np.radians(angle)) * axis_y
            ops.append(_rotation_matrix(axis, 180))
        # sigma_h
        ops.append(_reflection_matrix(axis_z))
        # sigma_v (3 vertical mirrors)
        for angle in [30, 150, 270]:
            v = np.cos(np.radians(angle)) * axis_x + np.sin(np.radians(angle)) * axis_y
            ops.append(_reflection_matrix(v))

    elif point_group == "C4":
        for angle in [90, 180, 270]:
            ops.append(_rotation_matrix(axis_z, angle))

    elif point_group == "C4v":
        for angle in [90, 180, 270]:
            ops.append(_rotation_matrix(axis_z, angle))
        for angle in [0, 45, 90, 135]:
            v = np.cos(np.radians(angle)) * axis_x + np.sin(np.radians(angle)) * axis_y
            ops.append(_reflection_matrix(v))

    elif point_group == "D4h":
        for angle in [90, 180, 270]:
            ops.append(_rotation_matrix(axis_z, angle))
        for angle in [0, 45, 90, 135]:
            axis = np.cos(np.radians(angle)) * axis_x + np.sin(np.radians(angle)) * axis_y
            ops.append(_rotation_matrix(axis, 180))
        ops.append(_reflection_matrix(axis_z))
        for angle in [22.5, 67.5, 112.5, 157.5]:
            v = np.cos(np.radians(angle)) * axis_x + np.sin(np.radians(angle)) * axis_y
            ops.append(_reflection_matrix(v))
        ops.append(_inversion_matrix())

    elif point_group == "C6":
        for angle in [60, 120, 180, 240, 300]:
            ops.append(_rotation_matrix(axis_z, angle))

    elif point_group == "C6v":
        for angle in [60, 120, 180, 240, 300]:
            ops.append(_rotation_matrix(axis_z, angle))
        for angle in [0, 30, 60, 90, 120, 150]:
            v = np.cos(np.radians(angle)) * axis_x + np.sin(np.radians(angle)) * axis_y
            ops.append(_reflection_matrix(v))

    elif point_group == "D6h":
        for angle in [60, 120, 180, 240, 300]:
            ops.append(_rotation_matrix(axis_z, angle))
        for angle in [0, 30, 60, 90, 120, 150]:
            axis = np.cos(np.radians(angle)) * axis_x + np.sin(np.radians(angle)) * axis_y
            ops.append(_rotation_matrix(axis, 180))
        ops.append(_reflection_matrix(axis_z))
        for angle in [15, 45, 75, 105, 135, 165]:
            v = np.cos(np.radians(angle)) * axis_x + np.sin(np.radians(angle)) * axis_y
            ops.append(_reflection_matrix(v))
        ops.append(_inversion_matrix())

    elif point_group == "D2h":
        ops.append(_rotation_matrix(axis_z, 180))
        ops.append(_rotation_matrix(axis_y, 180))
        ops.append(_rotation_matrix(axis_x, 180))
        ops.append(_reflection_matrix(axis_z))
        ops.append(_reflection_matrix(axis_y))
        ops.append(_reflection_matrix(axis_x))
        ops.append(_inversion_matrix())

    return ops


def compute_point_group_batch(
    atomic_numbers_batch: List[Tensor],
    positions_batch: List[Tensor],
    tolerance: float = 0.3,
) -> List[str]:
    """Compute point groups for a batch of molecules."""
    results = []
    for z, pos in zip(atomic_numbers_batch, positions_batch):
        results.append(compute_point_group(z, pos, tolerance))
    return results
