"""Deterministic graph-only candidate actions for non-unique D6h graphs."""

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
from generative_model.symmetry.graph_action_extended import (
    SCHEMA_VERSION,
    _d6h_matrices_and_permutations,
    _permutation_order,
    _validate_graph_preservation,
)


def recover_d6h_graph_action_candidates(
    atomic_numbers: np.ndarray,
    formal_charges: np.ndarray,
    radical_electrons: np.ndarray,
    bond_index: np.ndarray,
    bond_types: np.ndarray,
    *,
    maximum_candidates: int = 64,
    maximum_search_matches: int = 100_000,
) -> list[dict[str, Any]]:
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
    rotations: list[np.ndarray] = []
    involutions: list[np.ndarray] = []
    actions: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()

    def valid_pair(rotation: np.ndarray, secondary: np.ndarray) -> bool:
        inverse = _permutation_power(rotation, 5)
        if not np.array_equal(secondary[rotation[secondary]], inverse):
            return False
        generated = {
            tuple(_permutation_power(rotation, power).tolist())
            for power in range(6)
        }
        generated.update(
            tuple(_permutation_power(rotation, power)[secondary].tolist())
            for power in range(6)
        )
        return len(generated) == 12

    def append_pair(
        heavy_rotation: np.ndarray,
        heavy_secondary: np.ndarray,
        match_count: int,
    ) -> None:
        full_rotation = _lift_hydrogens(heavy_rotation, heavy_atoms, fields)
        full_secondary = _lift_hydrogens(heavy_secondary, heavy_atoms, fields)
        matrices, permutations = _d6h_matrices_and_permutations(
            full_rotation, full_secondary
        )
        key = tuple(permutations.ravel().tolist())
        if key in seen:
            return
        seen.add(key)
        orbit_id, orbit_size = _extract_orbits(permutations)
        action: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "target_pg": "D6h",
            "operation_matrices": matrices,
            "permutation_index": permutations,
            "orbit_id": orbit_id,
            "orbit_size": orbit_size,
            "recovery_audit": {
                "candidate_id": len(actions),
                "selection_rule": "GraphMatcher order of valid D6 generator pairs",
                "search_match_count_when_found": match_count,
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

    for match_count, mapping in enumerate(matcher.isomorphisms_iter(), start=1):
        if match_count > maximum_search_matches:
            raise RuntimeError("D6h action-candidate search 超过 match 上限")
        permutation = np.asarray(
            [mapping[index] for index in range(len(heavy_atoms))], dtype=np.int64
        )
        order = _permutation_order(permutation)
        if order == 6:
            for secondary in involutions:
                if valid_pair(permutation, secondary):
                    append_pair(permutation, secondary, match_count)
                    if len(actions) >= maximum_candidates:
                        return actions
            rotations.append(permutation)
        elif order == 2:
            for rotation in rotations:
                if valid_pair(rotation, permutation):
                    append_pair(rotation, permutation, match_count)
                    if len(actions) >= maximum_candidates:
                        return actions
            involutions.append(permutation)
    if not actions:
        raise ValueError("canonical graph 未恢复任何 D6h action candidate")
    return actions

