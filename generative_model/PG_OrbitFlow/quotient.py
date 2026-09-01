"""Quotient molecular graph construction under an atomic group action."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .group import validate_group_action


QUOTIENT_SCHEMA_VERSION = "pg-orbitflow-quotient-graph-v1"


@dataclass(frozen=True)
class QuotientGraph:
    orbit_count: int
    representative_indices: np.ndarray
    orbit_sizes: np.ndarray
    edge_orbit_index: np.ndarray
    edge_multiplicity: np.ndarray
    bond_type_histogram: np.ndarray
    typed_action_mismatch_count: int


def _typed_edges(bond_index: np.ndarray, bond_types: np.ndarray) -> set[tuple[int, int, int]]:
    return {
        (min(int(i), int(j)), max(int(i), int(j)), int(kind))
        for (i, j), kind in zip(bond_index.T, bond_types, strict=True)
    }


def build_quotient_graph(
    *,
    atom_count: int,
    bond_index: np.ndarray,
    bond_types: np.ndarray,
    orbit_id: np.ndarray,
    operation_matrices: np.ndarray,
    permutation_index: np.ndarray,
) -> QuotientGraph:
    """Compress atom and topological edge orbits without hiding Kekule mismatch.

    The untyped bond topology must be invariant.  Typed single/double resonance
    assignments are allowed to vary inside one edge orbit, but the exact type
    histogram and mismatch count are retained explicitly.
    """

    edges = np.asarray(bond_index, dtype=np.int64)
    types = np.asarray(bond_types, dtype=np.int64)
    orbits = np.asarray(orbit_id, dtype=np.int64)
    if edges.shape != (2, len(types)) or np.any((types < 1) | (types > 4)):
        raise ValueError("bond arrays must be [2,M] with types in 1..4")
    if orbits.shape != (atom_count,):
        raise ValueError("orbit_id shape mismatch")
    if len(types) and (int(edges.min()) < 0 or int(edges.max()) >= atom_count):
        raise ValueError("bond index out of range")
    validate_group_action(operation_matrices, permutation_index, orbit_id=orbits)

    topology = {
        (min(int(left), int(right)), max(int(left), int(right))) for left, right in edges.T
    }
    typed = _typed_edges(edges, types)
    typed_mismatch = 0
    for permutation in np.asarray(permutation_index, dtype=np.int64):
        mapped_topology = {
            (min(int(permutation[i]), int(permutation[j])), max(int(permutation[i]), int(permutation[j])))
            for i, j in topology
        }
        if mapped_topology != topology:
            raise ValueError("target group action is not an automorphism of bond topology")
        mapped_typed = {
            (
                min(int(permutation[i]), int(permutation[j])),
                max(int(permutation[i]), int(permutation[j])),
                kind,
            )
            for i, j, kind in typed
        }
        typed_mismatch += len(typed.symmetric_difference(mapped_typed)) // 2

    orbit_count = len(np.unique(orbits))
    representative_indices = np.asarray(
        [np.flatnonzero(orbits == current)[0] for current in range(orbit_count)],
        dtype=np.int64,
    )
    orbit_sizes = np.bincount(orbits, minlength=orbit_count).astype(np.int64)

    # Edge orbits are quotient endpoint pairs.  Self-edges represent bonds
    # between distinct atoms belonging to the same atomic orbit.
    buckets: dict[tuple[int, int], list[int]] = {}
    for (left, right), kind in zip(edges.T, types, strict=True):
        q_left, q_right = int(orbits[left]), int(orbits[right])
        key = (min(q_left, q_right), max(q_left, q_right))
        buckets.setdefault(key, []).append(int(kind))
    keys = sorted(buckets)
    quotient_edges = np.asarray(keys, dtype=np.int64).T if keys else np.zeros((2, 0), dtype=np.int64)
    multiplicity = np.asarray([len(buckets[key]) for key in keys], dtype=np.int64)
    histogram = np.zeros((len(keys), 4), dtype=np.int64)
    for index, key in enumerate(keys):
        for kind in buckets[key]:
            histogram[index, kind - 1] += 1
    return QuotientGraph(
        orbit_count=orbit_count,
        representative_indices=representative_indices,
        orbit_sizes=orbit_sizes,
        edge_orbit_index=quotient_edges,
        edge_multiplicity=multiplicity,
        bond_type_histogram=histogram,
        typed_action_mismatch_count=int(typed_mismatch),
    )


def symmetry_invariant_bond_order(
    atom_count: int,
    bond_index: np.ndarray,
    bond_types: np.ndarray,
    permutation_index: np.ndarray,
) -> np.ndarray:
    """Average canonical bond order over the group action for model features."""

    adjacency = np.zeros((atom_count, atom_count), dtype=np.float32)
    for (left, right), kind in zip(bond_index.T, bond_types, strict=True):
        adjacency[int(left), int(right)] = float(kind)
        adjacency[int(right), int(left)] = float(kind)
    accumulated = np.zeros_like(adjacency)
    permutations = np.asarray(permutation_index, dtype=np.int64)
    for row in permutations:
        inverse = np.argsort(row)
        accumulated += adjacency[np.ix_(inverse, inverse)]
    result = accumulated / len(permutations)
    if float(np.max(np.abs(result - result.T))) > 1e-6:
        raise RuntimeError("symmetrized bond order is not symmetric")
    return result
