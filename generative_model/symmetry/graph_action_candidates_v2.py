"""Graph-only candidate actions for ambiguous S4 and D6h molecular graphs.

This module is versioned separately so the frozen D6h-only v3 inference
evidence remains byte-identical.  Candidate enumeration never reads XYZ or a
stored symmetry annotation.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from generative_model.symmetry.graph_action import (
    _as_graph_fields,
    _build_normalized_heavy_graph,
    _extract_orbits,
    _lift_hydrogens,
    _permutation_power,
)
from generative_model.symmetry.graph_action_candidates import (
    recover_d6h_graph_action_candidates,
)
from generative_model.symmetry.graph_action_extended import (
    SCHEMA_VERSION,
    _permutation_order,
    _s4_matrices,
    _score_full_permutations,
    _validate_graph_preservation,
)


def recover_s4_graph_action_candidates(
    atomic_numbers: np.ndarray,
    formal_charges: np.ndarray,
    radical_electrons: np.ndarray,
    bond_index: np.ndarray,
    bond_types: np.ndarray,
    *,
    maximum_candidates: int = 64,
    maximum_search_matches: int = 100_000,
) -> list[dict[str, Any]]:
    """Enumerate deterministic exact-order-4 S4 graph actions."""

    if maximum_candidates <= 0 or maximum_search_matches <= 0:
        raise ValueError("candidate/match limits 必须为正")
    fields = _as_graph_fields(
        atomic_numbers, formal_charges, radical_electrons, bond_index, bond_types
    )
    graph, heavy_atoms, _, normalized_terminal_count = _build_normalized_heavy_graph(
        fields
    )
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph,
        graph,
        node_match=lambda left, right: left["label"] == right["label"],
        edge_match=lambda left, right: left["label"] == right["label"],
    )
    candidate_values: list[tuple[tuple[Any, ...], np.ndarray, int]] = []
    seen: set[tuple[int, ...]] = set()
    match_count = 0
    for match_count, mapping in enumerate(matcher.isomorphisms_iter(), start=1):
        if match_count > maximum_search_matches:
            raise RuntimeError("S4 action-candidate search 超过 match 上限")
        heavy = np.asarray(
            [mapping[index] for index in range(len(heavy_atoms))], dtype=np.int64
        )
        if _permutation_order(heavy) != 4:
            continue
        full = _lift_hydrogens(heavy, heavy_atoms, fields)
        permutations = np.stack([
            _permutation_power(full, power) for power in range(4)
        ]).astype(np.int64)
        key = tuple(permutations.ravel().tolist())
        if key in seen:
            continue
        seen.add(key)
        candidate_values.append((
            _score_full_permutations(permutations), permutations, match_count
        ))
    if not candidate_values:
        raise ValueError("canonical graph 未恢复任何 S4 action candidate")

    actions: list[dict[str, Any]] = []
    for _, permutations, found_at in sorted(
        candidate_values, key=lambda value: value[0]
    )[:maximum_candidates]:
        orbit_id, orbit_size = _extract_orbits(permutations)
        action: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "target_pg": "S4",
            "operation_matrices": _s4_matrices(),
            "permutation_index": permutations,
            "orbit_id": orbit_id,
            "orbit_size": orbit_size,
            "recovery_audit": {
                "candidate_id": len(actions),
                "selection_rule": (
                    "exact-order-4 actions sorted by group moved-atom support, "
                    "single-operation support, then lexicographic permutation"
                ),
                "search_match_count_when_found": int(found_at),
                "search_match_count_total": int(match_count),
                "exact_order_candidate_count": len(candidate_values),
                "heavy_atom_count": len(heavy_atoms),
                "normalized_terminal_atom_count": normalized_terminal_count,
                "coordinates_read": False,
                "stored_symmetry_action_read": False,
                "maximum_candidates": maximum_candidates,
                "maximum_search_matches": maximum_search_matches,
            },
        }
        action["validation"] = _validate_graph_preservation(action, fields)
        actions.append(action)
    return actions


__all__ = [
    "recover_s4_graph_action_candidates",
    "recover_d6h_graph_action_candidates",
]
