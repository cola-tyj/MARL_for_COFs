"""Recover a deterministic C2/C3 action from a canonical molecular graph only.

Coordinates and saved symmetry permutations are deliberately absent from this
interface.  The output is a finite cyclic action that preserves the canonical
typed graph, with the same explicit terminal-resonance contract used by E2.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np


SCHEMA_VERSION = "graph-only-cyclic-action-v1"
SUPPORTED_TARGETS = {"C2": 2, "C3": 3}


@dataclass(frozen=True)
class _GraphFields:
    atomic_numbers: np.ndarray
    formal_charges: np.ndarray
    radical_electrons: np.ndarray
    aromaticity: np.ndarray
    bond_index: np.ndarray
    bond_types: np.ndarray
    edge_types: dict[tuple[int, int], int]
    neighbors: tuple[tuple[int, ...], ...]


def _as_graph_fields(
    atomic_numbers: np.ndarray,
    formal_charges: np.ndarray,
    radical_electrons: np.ndarray,
    bond_index: np.ndarray,
    bond_types: np.ndarray,
) -> _GraphFields:
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    charges = np.asarray(formal_charges, dtype=np.int64)
    radicals = np.asarray(radical_electrons, dtype=np.int64)
    bonds = np.asarray(bond_index, dtype=np.int64)
    kinds = np.asarray(bond_types, dtype=np.int64)
    atom_count = len(numbers)
    if numbers.ndim != 1 or atom_count == 0:
        raise ValueError("atomic_numbers 必须为非空 [N]")
    if charges.shape != (atom_count,) or radicals.shape != (atom_count,):
        raise ValueError("canonical atom feature shape 不一致")
    if np.any(numbers <= 0):
        raise ValueError("atomic number 非法；不允许元素 fallback")
    if bonds.shape != (2, len(kinds)):
        raise ValueError("canonical bond_index/bond_types shape 不一致")
    if len(kinds) and (
        int(bonds.min()) < 0 or int(bonds.max()) >= atom_count
        or np.any((kinds < 1) | (kinds > 4))
    ):
        raise ValueError("canonical bond index/type 非法")
    edge_types: dict[tuple[int, int], int] = {}
    neighbors: list[list[int]] = [[] for _ in range(atom_count)]
    aromaticity = np.zeros(atom_count, dtype=np.bool_)
    for (left_raw, right_raw), kind_raw in zip(bonds.T, kinds, strict=True):
        left, right, kind = int(left_raw), int(right_raw), int(kind_raw)
        if left == right:
            raise ValueError("canonical graph 含 self-loop")
        pair = (min(left, right), max(left, right))
        if pair in edge_types:
            raise ValueError("canonical graph 含重复键")
        edge_types[pair] = kind
        neighbors[left].append(right); neighbors[right].append(left)
        if kind == 4:
            aromaticity[left] = True; aromaticity[right] = True
    if atom_count > 1:
        graph = nx.Graph()
        graph.add_nodes_from(range(atom_count)); graph.add_edges_from(edge_types)
        if not nx.is_connected(graph):
            raise ValueError("canonical graph 非连通")
    return _GraphFields(
        atomic_numbers=numbers,
        formal_charges=charges,
        radical_electrons=radicals,
        aromaticity=aromaticity,
        bond_index=bonds,
        bond_types=kinds,
        edge_types=edge_types,
        neighbors=tuple(tuple(sorted(values)) for values in neighbors),
    )


def _terminal_resonance_groups(
    fields: _GraphFields,
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[Any, ...]]]:
    """Return terminal->(center,Z) and deterministic center descriptors."""

    terminal_to_group: dict[int, tuple[int, int]] = {}
    center_descriptors: dict[int, tuple[Any, ...]] = {}
    for center, neighbors in enumerate(fields.neighbors):
        grouped: dict[int, list[int]] = defaultdict(list)
        for neighbor in neighbors:
            if (
                fields.atomic_numbers[neighbor] != 1
                and len(fields.neighbors[neighbor]) == 1
            ):
                grouped[int(fields.atomic_numbers[neighbor])].append(neighbor)
        descriptors = []
        for atomic_number, terminals in sorted(grouped.items()):
            if len(terminals) < 2:
                continue
            signatures = tuple(sorted(
                (
                    fields.edge_types[(min(center, terminal), max(center, terminal))],
                    int(fields.formal_charges[terminal]),
                    int(fields.radical_electrons[terminal]),
                    bool(fields.aromaticity[terminal]),
                )
                for terminal in terminals
            ))
            descriptors.append((atomic_number, len(terminals), signatures))
            for terminal in terminals:
                terminal_to_group[terminal] = (center, atomic_number)
        if descriptors:
            center_descriptors[center] = tuple(descriptors)
    return terminal_to_group, center_descriptors


def _build_normalized_heavy_graph(
    fields: _GraphFields,
) -> tuple[nx.Graph, np.ndarray, dict[int, int], int]:
    heavy_atoms = np.flatnonzero(fields.atomic_numbers != 1).astype(np.int64)
    if not len(heavy_atoms):
        raise ValueError("E3 C2/C3 recovery 不支持无重原子图")
    original_to_heavy = {int(atom): index for index, atom in enumerate(heavy_atoms)}
    terminal_to_group, center_descriptors = _terminal_resonance_groups(fields)
    graph = nx.Graph()
    for original in heavy_atoms:
        atom = int(original); hydrogen_count = sum(
            fields.atomic_numbers[neighbor] == 1 for neighbor in fields.neighbors[atom]
        )
        if atom in terminal_to_group:
            atom_label: tuple[Any, ...] = (
                "terminal-resonance", int(fields.atomic_numbers[atom])
            )
        else:
            atom_label = (
                "canonical", int(fields.atomic_numbers[atom]),
                int(fields.formal_charges[atom]),
                int(fields.radical_electrons[atom]), bool(fields.aromaticity[atom]),
            )
        graph.add_node(
            original_to_heavy[atom],
            label=(atom_label, int(hydrogen_count), center_descriptors.get(atom, ())),
        )
    for pair, kind in fields.edge_types.items():
        left, right = pair
        if left not in original_to_heavy or right not in original_to_heavy:
            continue
        if left in terminal_to_group and terminal_to_group[left][0] == right:
            edge_label: tuple[Any, ...] = (
                "terminal-resonance", terminal_to_group[left][1]
            )
        elif right in terminal_to_group and terminal_to_group[right][0] == left:
            edge_label = ("terminal-resonance", terminal_to_group[right][1])
        else:
            edge_label = ("canonical", int(kind))
        graph.add_edge(original_to_heavy[left], original_to_heavy[right], label=edge_label)
    return graph, heavy_atoms, original_to_heavy, len(terminal_to_group)


def _permutation_power(permutation: np.ndarray, exponent: int) -> np.ndarray:
    result = np.arange(len(permutation), dtype=np.int64)
    for _ in range(exponent):
        result = permutation[result]
    return result


def _stable_refinement_colors(graph: nx.Graph) -> dict[int, int]:
    """Deterministic labeled 1-WL colors; automorphic nodes share a color."""

    signatures: dict[int, Any] = {
        node: (graph.nodes[node]["label"], graph.degree[node]) for node in graph
    }
    colors: dict[int, int] = {}
    previous_partition: tuple[tuple[int, ...], ...] | None = None
    for _ in range(len(graph) + 1):
        palette = {
            signature: index
            for index, signature in enumerate(sorted(set(signatures.values()), key=repr))
        }
        updated = {node: palette[signature] for node, signature in signatures.items()}
        grouped: dict[int, list[int]] = defaultdict(list)
        for node, color in updated.items():
            grouped[color].append(node)
        partition = tuple(sorted(tuple(sorted(group)) for group in grouped.values()))
        if partition == previous_partition:
            return updated
        previous_partition = partition
        colors = updated
        signatures = {
            node: (
                colors[node],
                tuple(sorted(
                    (graph.edges[node, neighbor]["label"], colors[neighbor])
                    for neighbor in graph[node]
                )),
            )
            for node in graph
        }
    raise RuntimeError("graph color refinement 未在有限步内稳定")


def _cycle_anchored_graphs(
    graph: nx.Graph, cycle: tuple[int, ...]
) -> tuple[nx.Graph, nx.Graph]:
    source = graph.copy(); target = graph.copy()
    nx.set_node_attributes(source, -1, "anchor")
    nx.set_node_attributes(target, -1, "anchor")
    for marker, source_node in enumerate(cycle):
        target_node = cycle[(marker + 1) % len(cycle)]
        source.nodes[source_node]["anchor"] = marker
        target.nodes[target_node]["anchor"] = marker
    return source, target


def _candidate_anchor_cycles(group: list[int], order: int):
    for members in combinations(group, order):
        if order == 2:
            yield tuple(members)
        else:
            first, second, third = members
            yield (first, second, third)
            yield (first, third, second)


def _lift_hydrogens(
    heavy_permutation: np.ndarray,
    heavy_atoms: np.ndarray,
    fields: _GraphFields,
) -> np.ndarray:
    atom_count = len(fields.atomic_numbers)
    result = np.full(atom_count, -1, dtype=np.int64)
    for source_local, target_local in enumerate(heavy_permutation):
        source = int(heavy_atoms[source_local]); target = int(heavy_atoms[target_local])
        result[source] = target
        source_h = sorted(
            neighbor for neighbor in fields.neighbors[source]
            if fields.atomic_numbers[neighbor] == 1
        )
        target_h = sorted(
            neighbor for neighbor in fields.neighbors[target]
            if fields.atomic_numbers[neighbor] == 1
        )
        if len(source_h) != len(target_h):
            raise RuntimeError("heavy automorphism 未保持显式 H 数量")
        for source_atom, target_atom in zip(source_h, target_h, strict=True):
            if result[source_atom] not in (-1, target_atom):
                raise RuntimeError("显式 H lifting 产生冲突")
            result[source_atom] = target_atom
    if np.any(result < 0) or not np.array_equal(
        np.sort(result), np.arange(atom_count, dtype=np.int64)
    ):
        raise RuntimeError("显式 H lifting 未得到完整原子双射")
    return result


def _cyclic_operation_matrices(order: int) -> np.ndarray:
    matrices = []
    for power in range(order):
        angle = 2.0 * np.pi * power / order
        cosine, sine = np.cos(angle), np.sin(angle)
        matrices.append(np.asarray([
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64))
    result = np.stack(matrices)
    result[np.abs(result) < 1e-15] = 0.0
    return result


def _extract_orbits(permutations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    atom_count = permutations.shape[1]
    unassigned = set(range(atom_count)); orbit_id = np.full(atom_count, -1, dtype=np.int64)
    orbit_size = np.zeros(atom_count, dtype=np.int64); current = 0
    while unassigned:
        representative = min(unassigned)
        members = sorted({int(permutation[representative]) for permutation in permutations})
        for atom in members:
            orbit_id[atom] = current; orbit_size[atom] = len(members)
        unassigned.difference_update(members); current += 1
    return orbit_id, orbit_size


def validate_recovered_graph_action(
    action: dict[str, Any],
    atomic_numbers: np.ndarray,
    formal_charges: np.ndarray,
    radical_electrons: np.ndarray,
    bond_index: np.ndarray,
    bond_types: np.ndarray,
) -> dict[str, Any]:
    fields = _as_graph_fields(
        atomic_numbers, formal_charges, radical_electrons, bond_index, bond_types
    )
    target_pg = str(action["target_pg"])
    if target_pg not in SUPPORTED_TARGETS:
        raise ValueError(f"E3 仅支持 C2/C3，收到 {target_pg}")
    order = SUPPORTED_TARGETS[target_pg]; atom_count = len(fields.atomic_numbers)
    matrices = np.asarray(action["operation_matrices"], dtype=np.float64)
    permutations = np.asarray(action["permutation_index"], dtype=np.int64)
    if matrices.shape != (order, 3, 3) or permutations.shape != (order, atom_count):
        raise ValueError("recovered operation/permutation shape 非法")
    expected = np.arange(atom_count, dtype=np.int64)
    if not np.array_equal(permutations[0], expected):
        raise ValueError("recovered action 第一项必须为 identity")
    if any(not np.array_equal(np.sort(value), expected) for value in permutations):
        raise ValueError("recovered permutation 不是原子双射")
    if not np.array_equal(fields.atomic_numbers[permutations], np.broadcast_to(
        fields.atomic_numbers, permutations.shape
    )):
        raise ValueError("recovered permutation 跨元素")
    for power in range(order):
        if not np.array_equal(permutations[power], _permutation_power(permutations[1], power)):
            raise ValueError("recovered permutations 不满足 cyclic group relation")
    if np.array_equal(permutations[1], expected):
        raise ValueError("recovered cyclic generator 不能为 identity")
    edge_set = set(fields.edge_types)
    normalized_operation_count = 0
    terminal_to_group, _ = _terminal_resonance_groups(fields)
    for permutation in permutations:
        mapped_connectivity = {
            tuple(sorted((int(permutation[left]), int(permutation[right]))))
            for left, right in edge_set
        }
        if mapped_connectivity != edge_set:
            raise ValueError("recovered permutation 修改 canonical connectivity")
        operation_normalized = False
        for atom in range(atom_count):
            mapped = int(permutation[atom])
            features = (
                int(fields.formal_charges[atom]), int(fields.radical_electrons[atom]),
                bool(fields.aromaticity[atom]),
            )
            mapped_features = (
                int(fields.formal_charges[mapped]), int(fields.radical_electrons[mapped]),
                bool(fields.aromaticity[mapped]),
            )
            if features != mapped_features:
                if atom not in terminal_to_group:
                    raise ValueError("recovered permutation 修改非共振 atom labels")
                operation_normalized = True
        for pair, kind in fields.edge_types.items():
            mapped_pair = tuple(sorted((
                int(permutation[pair[0]]), int(permutation[pair[1]])
            )))
            if fields.edge_types[mapped_pair] != kind:
                terminal = pair[0] if pair[0] in terminal_to_group else pair[1]
                if terminal not in terminal_to_group:
                    raise ValueError("recovered permutation 修改非共振 bond type")
                operation_normalized = True
        normalized_operation_count += int(operation_normalized)
    orbit_id, orbit_size = _extract_orbits(permutations)
    if not np.array_equal(orbit_id, np.asarray(action["orbit_id"], dtype=np.int64)):
        raise ValueError("recovered orbit_id 与 permutation action 不一致")
    if not np.array_equal(orbit_size, np.asarray(action["orbit_size"], dtype=np.int64)):
        raise ValueError("recovered orbit_size 与 permutation action 不一致")
    return {
        "atom_count": atom_count,
        "operation_count": order,
        "orbit_count": int(orbit_id.max()) + 1,
        "moved_atom_count": int(np.sum(permutations[1] != expected)),
        "resonance_normalized_operation_count": normalized_operation_count,
    }


def recover_cyclic_graph_action(
    atomic_numbers: np.ndarray,
    formal_charges: np.ndarray,
    radical_electrons: np.ndarray,
    bond_index: np.ndarray,
    bond_types: np.ndarray,
    target_pg: str,
    *,
    maximum_search_matches: int = 10_000,
) -> dict[str, Any]:
    """Recover one maximal-support cyclic action in canonical atom order."""

    if target_pg not in SUPPORTED_TARGETS:
        raise ValueError(f"E3 仅支持 C2/C3，收到 {target_pg}")
    if maximum_search_matches <= 0:
        raise ValueError("maximum_search_matches 必须为正")
    order = SUPPORTED_TARGETS[target_pg]
    fields = _as_graph_fields(
        atomic_numbers, formal_charges, radical_electrons, bond_index, bond_types
    )
    graph, heavy_atoms, _, normalized_terminal_count = _build_normalized_heavy_graph(fields)
    identity = np.arange(len(heavy_atoms), dtype=np.int64)
    colors = _stable_refinement_colors(graph)
    color_groups: dict[int, list[int]] = defaultdict(list)
    for node, color in colors.items():
        color_groups[color].append(node)
    movable_upper_bound = sum(
        (len(group) // order) * order for group in color_groups.values()
    )
    candidate_count = 0; match_yield_count = 0; anchor_cycle_count = 0
    best: tuple[Any, ...] | None = None; reached_upper_bound = False
    groups = sorted(
        (sorted(group) for group in color_groups.values() if len(group) >= order),
        key=lambda group: (-len(group), tuple(group)),
    )
    for group in groups:
        if reached_upper_bound:
            break
        for cycle in _candidate_anchor_cycles(group, order):
            if reached_upper_bound:
                break
            anchor_cycle_count += 1
            source_graph, target_graph = _cycle_anchored_graphs(graph, cycle)
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                source_graph, target_graph,
                node_match=lambda left, right: (
                    left["label"] == right["label"]
                    and left["anchor"] == right["anchor"]
                ),
                edge_match=lambda left, right: left["label"] == right["label"],
            )
            for mapping in matcher.isomorphisms_iter():
                match_yield_count += 1
                if match_yield_count > maximum_search_matches:
                    raise RuntimeError(
                        "anchored graph-action search 超过冻结 match 上限 "
                        f"{maximum_search_matches}"
                    )
                generator = np.asarray(
                    [mapping[index] for index in range(len(heavy_atoms))],
                    dtype=np.int64,
                )
                if not np.array_equal(
                    _permutation_power(generator, order), identity
                ):
                    continue
                candidate_count += 1
                full = _lift_hydrogens(generator, heavy_atoms, fields)
                moved_heavy = int(np.sum(generator != identity))
                moved_all = int(np.sum(
                    full != np.arange(len(full), dtype=np.int64)
                ))
                score = (-moved_heavy, -moved_all, tuple(full.tolist()), full)
                if best is None or score[:-1] < best[:-1]:
                    best = score
                if moved_heavy == movable_upper_bound:
                    reached_upper_bound = True
                    break
    if best is None:
        raise ValueError(f"canonical graph 不存在非平凡 {target_pg} cyclic automorphism")
    generator = np.asarray(best[-1], dtype=np.int64)
    permutations = np.stack([
        _permutation_power(generator, power) for power in range(order)
    ])
    orbit_id, orbit_size = _extract_orbits(permutations)
    action: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "target_pg": target_pg,
        "operation_matrices": _cyclic_operation_matrices(order),
        "permutation_index": permutations,
        "orbit_id": orbit_id,
        "orbit_size": orbit_size,
        "recovery_audit": {
            "selection_rule": (
                "maximum moved heavy atoms, then maximum moved all atoms, "
                "then lexicographically minimum full permutation"
            ),
            "heavy_atom_count": len(heavy_atoms),
            "isomorphism_match_yield_count": match_yield_count,
            "anchor_cycle_count": anchor_cycle_count,
            "cyclic_candidate_count": candidate_count,
            "selected_moved_heavy_atom_count": int(-best[0]),
            "selected_moved_atom_count": int(-best[1]),
            "wl_movable_heavy_atom_upper_bound": movable_upper_bound,
            "reached_wl_movable_upper_bound": reached_upper_bound,
            "normalized_terminal_atom_count": normalized_terminal_count,
            "maximum_search_matches": maximum_search_matches,
            "coordinates_read": False,
            "stored_symmetry_action_read": False,
        },
    }
    action["validation"] = validate_recovered_graph_action(
        action, fields.atomic_numbers, fields.formal_charges,
        fields.radical_electrons, fields.bond_index, fields.bond_types,
    )
    return action
