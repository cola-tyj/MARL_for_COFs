"""Reference-free inertia guard for stable S4 point-group recognition."""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem


def mass_weighted_inertia_separation(
    atomic_numbers: np.ndarray, positions: np.ndarray
) -> float:
    """Largest adjacent principal-moment gap, normalized by max moment."""

    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    coordinates = np.asarray(positions, dtype=np.float64)
    if coordinates.shape != (len(numbers), 3) or not len(numbers):
        raise ValueError("atomic_numbers/positions shape 非法")
    if not np.isfinite(coordinates).all():
        raise ValueError("positions 含 NaN/Inf")
    table = Chem.GetPeriodicTable()
    masses = np.asarray([
        float(table.GetAtomicWeight(int(number))) for number in numbers
    ], dtype=np.float64)
    if not np.isfinite(masses).all() or bool(np.any(masses <= 0)):
        raise ValueError("atomic mass 非法")
    centered = coordinates - np.average(coordinates, axis=0, weights=masses)
    inertia = (
        np.eye(3) * np.sum(masses * np.sum(centered * centered, axis=1))
        - np.einsum("n,ni,nj->ij", masses, centered, centered)
    )
    eigenvalues = np.linalg.eigvalsh(inertia)
    if eigenvalues[-1] <= 0:
        raise ValueError("惯量张量非正")
    return float(np.max(np.diff(eigenvalues)) / eigenvalues[-1])


def select_s4_inertia_stable_candidate(
    records: list[dict[str, Any]],
    coordinates: list[dict[str, np.ndarray]],
    atomic_numbers: np.ndarray,
    *,
    minimum_separation: float,
) -> tuple[int, list[float]]:
    """Select lowest objective among candidates outside the spherical boundary."""

    if minimum_separation <= 0 or len(records) != len(coordinates) or not records:
        raise ValueError("S4 inertia selection 输入非法")
    separations = [
        mass_weighted_inertia_separation(atomic_numbers, value["final"])
        for value in coordinates
    ]
    eligible = [
        index for index, separation in enumerate(separations)
        if separation >= minimum_separation
    ]
    if not eligible:
        raise RuntimeError(
            "所有通过几何 Gate 的 S4 candidates 均落在惯量近简并边界"
        )
    selected = min(eligible, key=lambda index: (
        float(records[index]["final_total_objective_kcal_mol"]),
        int(records[index]["candidate_id"]),
    ))
    return selected, separations
