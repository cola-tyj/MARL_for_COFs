"""EF1 symmetry projection with the frozen terminal-resonance contract.

EF0 files remain immutable after their official preflight.  EF1 adds only the
already validated E2/E3 terminal-resonance semantics needed for an unbiased
IID-test panel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from generative_model.models.graph_pg_3d_projection import operation_errors
from generative_model.optimization.etkdg_orbit_initializer import orient_and_project_to_orbits
from generative_model.optimization.etkdg_orbit_resonance import (
    validate_resonance_compatible_graph_automorphisms,
)


SCHEMA_VERSION = "etflow-ef1-resonance-compatible-projection-v1"


def project_etflow_ef1(
    positions: np.ndarray,
    sample: dict[str, Any],
    *,
    match_radius_of_gyration: bool,
) -> dict[str, Any]:
    values = np.asarray(positions, dtype=np.float64)
    if values.shape != (len(sample["atomic_numbers"]), 3) or not np.isfinite(values).all():
        raise ValueError("EF1 ET-Flow conformer 必须为 finite [N,3]")
    resonance_audit = validate_resonance_compatible_graph_automorphisms(sample)
    result = orient_and_project_to_orbits(
        values, sample, match_radius_of_gyration=match_radius_of_gyration
    )
    errors = operation_errors(
        result["positions"], sample["target_operation_matrices"],
        sample["target_permutation_index"],
    )
    return {
        **result,
        "schema_version": SCHEMA_VERSION,
        "resonance_audit": resonance_audit,
        "operation_errors": errors,
        "reference_coordinates_used": False,
        "stored_symmetry_action_used": False,
        "etkdg_coordinates_used": False,
    }


def kabsch_rmsd(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Kabsch inputs 必须为同 shape [N,3]")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Kabsch inputs 含 NaN/Inf")
    target = reference - reference.mean(axis=0, keepdims=True)
    source = candidate - candidate.mean(axis=0, keepdims=True)
    left, _, right_t = np.linalg.svd(source.T @ target)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    aligned = source @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))


def pair_distance_mae(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or len(reference) < 2:
        raise ValueError("pair-distance inputs shape 非法")
    indices = np.triu_indices(len(reference), k=1)
    reference_distances = np.linalg.norm(
        reference[:, None, :] - reference[None, :, :], axis=-1
    )[indices]
    candidate_distances = np.linalg.norm(
        candidate[:, None, :] - candidate[None, :, :], axis=-1
    )[indices]
    return float(np.mean(np.abs(candidate_distances - reference_distances)))
