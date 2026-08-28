"""Pure geometry helpers for the frozen our_ET_Flow GFN2-xTB audit."""

from __future__ import annotations

from typing import Any

import numpy as np

from generative_model.conformer.etflow_ef1_projection import (
    kabsch_rmsd,
    pair_distance_mae,
)
from generative_model.models.graph_pg_3d_projection import (
    bond_length_mae,
    minimum_pair_distance,
)


def center_positions(positions: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("positions must be finite [N, 3]")
    return values - values.mean(axis=0, keepdims=True)


def maximum_force_norm(forces: np.ndarray) -> float:
    values = np.asarray(forces, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("forces must be finite [N, 3]")
    return float(np.linalg.norm(values, axis=1).max(initial=0.0))


def relaxation_geometry_metrics(
    before: np.ndarray,
    after: np.ndarray,
    canonical: np.ndarray,
    bond_index: np.ndarray,
) -> dict[str, Any]:
    """Return rigid-motion-invariant pre/post geometry diagnostics."""

    source = center_positions(before)
    relaxed = center_positions(after)
    reference = center_positions(canonical)
    bonds = np.asarray(bond_index, dtype=np.int64)
    return {
        "kabsch_rmsd_pre_to_post_angstrom": kabsch_rmsd(source, relaxed),
        "pair_distance_mae_pre_to_post_angstrom": pair_distance_mae(source, relaxed),
        "minimum_pair_distance_before_angstrom": minimum_pair_distance(source),
        "minimum_pair_distance_after_angstrom": minimum_pair_distance(relaxed),
        "collision_free_before_at_0p6": bool(minimum_pair_distance(source) >= 0.6),
        "collision_free_after_at_0p6": bool(minimum_pair_distance(relaxed) >= 0.6),
        "bond_length_mae_pre_to_post_angstrom": bond_length_mae(source, relaxed, bonds),
        "bond_length_mae_before_to_canonical_angstrom": bond_length_mae(
            reference, source, bonds
        ),
        "bond_length_mae_after_to_canonical_angstrom": bond_length_mae(
            reference, relaxed, bonds
        ),
    }


def electronic_state(formal_charges: np.ndarray, radical_electrons: np.ndarray) -> dict[str, int]:
    """Map canonical charge/radical ground truth to tblite inputs strictly."""

    charges = np.asarray(formal_charges, dtype=np.int64)
    radicals = np.asarray(radical_electrons, dtype=np.int64)
    if charges.shape != radicals.shape or np.any(radicals < 0):
        raise ValueError("invalid canonical charge/radical arrays")
    unpaired = int(radicals.sum())
    return {
        "charge": int(charges.sum()),
        "unpaired_electrons": unpaired,
        "multiplicity": unpaired + 1,
    }
