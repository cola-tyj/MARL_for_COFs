"""Model-independent symmetry layer applied after an ET-Flow conformer sample."""

from __future__ import annotations

from typing import Any

import numpy as np

from generative_model.models.graph_pg_3d_projection import operation_errors
from generative_model.optimization.etkdg_orbit_initializer import (
    orient_and_project_to_orbits,
    validate_typed_graph_automorphisms,
)


SCHEMA_VERSION = "etflow-to-e3-symmetry-projection-v1"


def project_etflow_conformer(
    positions: np.ndarray,
    sample: dict[str, Any],
    *,
    match_radius_of_gyration: bool = True,
) -> dict[str, Any]:
    """Orient and hard-project an ET-Flow sample into a known E3 action.

    No target/reference coordinates are accepted or read.  This deliberately
    reuses only the frozen E2/E3 group-action projection layer, not ETKDG.
    """

    values = np.asarray(positions, dtype=np.float64)
    if values.shape != (len(sample["atomic_numbers"]), 3):
        raise ValueError("ET-Flow conformer 与 canonical atom count 不一致")
    if not np.isfinite(values).all():
        raise ValueError("ET-Flow conformer 含 NaN/Inf")
    validate_typed_graph_automorphisms(sample)
    projected = orient_and_project_to_orbits(
        values, sample, match_radius_of_gyration=match_radius_of_gyration
    )
    errors = operation_errors(
        projected["positions"],
        sample["target_operation_matrices"],
        sample["target_permutation_index"],
    )
    return {
        **projected,
        "schema_version": SCHEMA_VERSION,
        "conformer_provider": "ET-Flow",
        "symmetry_provider": "recovered E3 graph action + Reynolds projection",
        "reference_coordinates_used": False,
        "etkdg_coordinates_used": False,
        "operation_errors": errors,
    }
