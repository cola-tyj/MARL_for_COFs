"""Scalable graph-witnessed factorized branch automorphisms.

The full centralizer enumerator is exact but its Cartesian product of
independent branch swaps is unsuitable for dataset-scale preprocessing.  This
module keeps explicit graph-isomorphism witnesses for each local articulation
branch swap and works only with their induced torsion-orbit generators.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import signal
import threading

import numpy as np


@dataclass(frozen=True)
class FactorizedAutomorphismContract:
    groups: tuple[tuple[int, ...], ...]
    representative_tuples: np.ndarray
    allowed_torsion_permutations: np.ndarray
    induced_generator_count: int
    allowed_torsion_permutation_count: int
    branch_swap_witness_count: int
    anchored_witness_count: int


def _canonical_graph(sample):
    import networkx as nx

    graph = nx.Graph()
    for atom, (kind, charge, radical) in enumerate(
        zip(
            sample.atom_types,
            sample.formal_charges,
            sample.radical_electrons,
            strict=True,
        )
    ):
        graph.add_node(
            atom,
            atom_type=int(kind),
            charge=int(charge),
            radical=int(radical),
        )
    for edge, bond_type in zip(sample.bond_index.T, sample.bond_types, strict=True):
        graph.add_edge(int(edge[0]), int(edge[1]), bond_type=int(bond_type))
    if not nx.is_connected(graph):
        raise RuntimeError("factorized automorphism requires a connected graph")
    return graph


def _representatives(geometry) -> np.ndarray:
    identifiers = np.asarray(geometry.torsion_orbit_id, dtype=np.int64)
    return np.asarray(
        [
            geometry.torsion_index[
                :, np.flatnonzero(identifiers == current)[0]
            ]
            for current in range(int(identifiers.max()) + 1)
        ],
        dtype=np.int64,
    )


def _branch_swap_permutations(graph) -> list[np.ndarray]:
    import networkx as nx

    node_match = nx.algorithms.isomorphism.categorical_node_match(
        ("atom_type", "charge", "radical", "branch_root"),
        (None, None, None, False),
    )
    edge_match = nx.algorithms.isomorphism.categorical_edge_match(
        "bond_type", None
    )
    permutations = []
    atom_count = len(graph)
    for center in sorted(graph.nodes):
        detached = graph.copy()
        detached.remove_node(center)
        components = list(nx.connected_components(detached))
        component_by_atom = {
            atom: component_id
            for component_id, component in enumerate(components)
            for atom in component
        }
        neighbors = sorted(graph.neighbors(center))
        for left_position, left in enumerate(neighbors):
            for right in neighbors[left_position + 1 :]:
                if component_by_atom[left] == component_by_atom[right]:
                    continue
                if graph.edges[center, left]["bond_type"] != graph.edges[center, right]["bond_type"]:
                    continue
                left_nodes = components[component_by_atom[left]]
                right_nodes = components[component_by_atom[right]]
                if len(left_nodes) != len(right_nodes):
                    continue
                left_graph = graph.subgraph(left_nodes).copy()
                right_graph = graph.subgraph(right_nodes).copy()
                nx.set_node_attributes(left_graph, False, "branch_root")
                nx.set_node_attributes(right_graph, False, "branch_root")
                left_graph.nodes[left]["branch_root"] = True
                right_graph.nodes[right]["branch_root"] = True
                matcher = nx.algorithms.isomorphism.GraphMatcher(
                    left_graph,
                    right_graph,
                    node_match=node_match,
                    edge_match=edge_match,
                )
                mapping = next(matcher.isomorphisms_iter(), None)
                if mapping is None:
                    continue
                inverse = {target: source for source, target in mapping.items()}
                permutation = np.arange(atom_count, dtype=np.int64)
                for source, target in mapping.items():
                    permutation[source] = target
                for target, source in inverse.items():
                    permutation[target] = source
                if len(np.unique(permutation)) != atom_count:
                    raise RuntimeError("branch-swap witness is not bijective")
                permutations.append(permutation)
    return permutations


def _is_graph_automorphism(graph, permutation: np.ndarray) -> bool:
    if len(np.unique(permutation)) != len(graph):
        return False
    for atom, attributes in graph.nodes(data=True):
        if attributes != graph.nodes[int(permutation[atom])]:
            return False
    for left, right, attributes in graph.edges(data=True):
        mapped = (int(permutation[left]), int(permutation[right]))
        if not graph.has_edge(*mapped) or attributes != graph.edges[mapped]:
            return False
    return True


def _centralized_branch_witnesses(sample, graph, branch_witnesses):
    """Synchronize a local branch swap over its full Target-PG conjugacy orbit."""

    actions = np.asarray(sample.permutation_index, dtype=np.int64)
    inverses = np.argsort(actions, axis=1)
    atom_count = len(graph)
    result = {}
    for witness in branch_witnesses:
        conjugates = []
        for action, inverse in zip(actions, inverses, strict=True):
            current = action[witness[inverse]]
            if not any(np.array_equal(current, row) for row in conjugates):
                conjugates.append(current)
        product = np.arange(atom_count, dtype=np.int64)
        for current in conjugates:
            product = current[product]
        if any(
            not np.array_equal(product[action], action[product])
            for action in actions
        ):
            continue
        if not _is_graph_automorphism(graph, product):
            raise RuntimeError("centralized branch witness lost canonical graph")
        result.setdefault(tuple(product.tolist()), product)
    return list(result.values())


def _augmented_action_graph(sample, graph):
    """Encode graph bonds and each labelled PG action as a directed gadget."""

    import networkx as nx

    result = nx.DiGraph()
    for atom, attributes in graph.nodes(data=True):
        result.add_node(
            ("atom", int(atom)),
            kind="atom",
            label=(
                int(attributes["atom_type"]),
                int(attributes["charge"]),
                int(attributes["radical"]),
            ),
            anchor=-1,
        )
    for edge_id, (left, right, attributes) in enumerate(graph.edges(data=True)):
        node = ("bond", edge_id)
        result.add_node(
            node,
            kind="bond",
            label=(int(attributes["bond_type"]),),
            anchor=-1,
        )
        for atom in (left, right):
            result.add_edge(("atom", int(atom)), node)
            result.add_edge(node, ("atom", int(atom)))
    for operation, permutation in enumerate(
        np.asarray(sample.permutation_index, dtype=np.int64)
    ):
        for source, target in enumerate(permutation):
            node = ("action", operation, source)
            result.add_node(
                node,
                kind="action",
                label=(operation,),
                anchor=-1,
            )
            result.add_edge(("atom", source), node)
            result.add_edge(node, ("atom", int(target)))
    if not nx.is_weakly_connected(result):
        raise RuntimeError("augmented action graph is disconnected")
    return result


def _torsion_signature(sample, graph, values):
    atoms = []
    for atom in values:
        attributes = graph.nodes[int(atom)]
        atoms.append(
            (
                int(attributes["atom_type"]),
                int(attributes["charge"]),
                int(attributes["radical"]),
                int(graph.degree[int(atom)]),
                int(sample.orbit_size[int(atom)]),
            )
        )
    bonds = tuple(
        int(graph.edges[int(left), int(right)]["bond_type"])
        for left, right in zip(values[:-1], values[1:], strict=True)
    )
    forward = (tuple(atoms), bonds)
    reverse = (tuple(reversed(atoms)), tuple(reversed(bonds)))
    return min(forward, reverse)


@contextmanager
def _main_thread_time_limit(seconds: float):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("bounded graph witness search requires the main thread")

    def expired(_signum, _frame):
        raise TimeoutError("anchored graph witness search timed out")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _anchored_action_witnesses(
    augmented,
    source,
    target,
    *,
    maximum_witnesses: int = 1,
    timeout_seconds: float = 5.0,
):
    import networkx as nx

    left = augmented.copy()
    right = augmented.copy()
    for position, atom in enumerate(source):
        left.nodes[("atom", int(atom))]["anchor"] = position
    for position, atom in enumerate(target):
        right.nodes[("atom", int(atom))]["anchor"] = position
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        ("kind", "label", "anchor"), (None, None, -1)
    )
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        left, right, node_match=node_match
    )
    atom_count = sum(node[0] == "atom" for node in augmented.nodes)
    witnesses = {}
    try:
        with _main_thread_time_limit(timeout_seconds):
            for mapping in matcher.isomorphisms_iter():
                permutation = np.asarray(
                    [mapping[("atom", atom)][1] for atom in range(atom_count)],
                    dtype=np.int64,
                )
                if len(np.unique(permutation)) != atom_count:
                    raise RuntimeError("anchored action witness is not bijective")
                witnesses.setdefault(tuple(permutation.tolist()), permutation)
                if len(witnesses) >= maximum_witnesses:
                    break
    except TimeoutError as error:
        raise RuntimeError(
            f"anchored graph witness exceeded {timeout_seconds:g}s frozen limit"
        ) from error
    return list(witnesses.values())


def _induced_torsion_permutation(geometry, representatives, atom_permutation):
    lookup = {}
    for raw, values in enumerate(np.asarray(geometry.torsion_index).T):
        orbit = int(geometry.torsion_orbit_id[raw])
        lookup[tuple(values.tolist())] = orbit
        lookup[tuple(values[::-1].tolist())] = orbit
    induced = []
    for values in representatives:
        mapped = tuple(atom_permutation[values].tolist())
        if mapped not in lookup:
            raise RuntimeError("branch swap does not preserve torsion support")
        induced.append(lookup[mapped])
    if sorted(induced) != list(range(len(representatives))):
        raise RuntimeError("branch swap does not induce a torsion permutation")
    return tuple(induced)


def _permutation_orbits(permutations, count: int):
    remaining = set(range(count))
    groups = []
    while remaining:
        seed = min(remaining)
        reached = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            for permutation in permutations:
                target = int(permutation[current])
                if target not in reached:
                    reached.add(target)
                    frontier.append(target)
        remaining.difference_update(reached)
        if len(reached) > 1:
            groups.append(tuple(sorted(reached)))
    return tuple(groups)


def _permutation_closure(generators, *, maximum_size: int):
    if not generators:
        raise ValueError("permutation closure requires the identity")
    size = len(generators[0])
    identity = tuple(range(size))
    closure = {identity, *generators}
    frontier = list(closure)
    generator_rows = list(closure)
    while frontier:
        current = frontier.pop()
        for generator in generator_rows:
            composed = tuple(generator[current[index]] for index in range(size))
            if composed not in closure:
                closure.add(composed)
                frontier.append(composed)
                if len(closure) > maximum_size:
                    raise RuntimeError(
                        "factorized torsion-permutation closure exceeded frozen cap"
                    )
    return np.asarray(sorted(closure), dtype=np.int64)


def _factorized_components(permutations, count: int, *, maximum_slots: int):
    supports = [
        {index for index in range(count) if permutation[index] != index}
        for permutation in permutations
    ]
    supports = [support for support in supports if support]
    merged = []
    for support in supports:
        touching = [index for index, current in enumerate(merged) if current & support]
        if not touching:
            merged.append(set(support))
            continue
        combined = set(support)
        for index in reversed(touching):
            combined.update(merged.pop(index))
        merged.append(combined)
    groups = tuple(tuple(sorted(group)) for group in sorted(merged, key=min))
    if any(len(group) > maximum_slots for group in groups):
        raise RuntimeError(
            f"factorized automorphism group exceeds {maximum_slots} slots"
        )
    closures = []
    for group in groups:
        position = {orbit: local for local, orbit in enumerate(group)}
        generators = {
            tuple(position[int(permutation[orbit])] for orbit in group)
            for permutation in permutations
            if any(permutation[orbit] != orbit for orbit in group)
        }
        identity = tuple(range(len(group)))
        closure = {identity}
        frontier = [identity]
        while frontier:
            current = frontier.pop()
            for generator in generators:
                composed = tuple(generator[current[index]] for index in range(len(group)))
                if composed not in closure:
                    closure.add(composed)
                    frontier.append(composed)
        closures.append(np.asarray(sorted(closure), dtype=np.int64))
    return groups, tuple(closures)


def build_factorized_automorphism_contract(
    sample,
    geometry,
    *,
    maximum_slots: int = 6,
    maximum_permutations: int = 10_000,
) -> FactorizedAutomorphismContract:
    graph = _canonical_graph(sample)
    representatives = _representatives(geometry)
    branch_witnesses = _branch_swap_permutations(graph)
    centralized_witnesses = _centralized_branch_witnesses(
        sample, graph, branch_witnesses
    )
    augmented = _augmented_action_graph(sample, graph)
    signatures = {}
    for index, values in enumerate(representatives):
        signatures.setdefault(_torsion_signature(sample, graph, values), []).append(index)
    identity = tuple(range(len(representatives)))
    induced = {identity}
    for witness in centralized_witnesses:
        induced.add(
            _induced_torsion_permutation(geometry, representatives, witness)
        )
    induced_arrays = _permutation_closure(
        sorted(induced), maximum_size=maximum_permutations
    )
    groups = _permutation_orbits(induced_arrays, len(representatives))
    if any(len(group) > maximum_slots for group in groups):
        raise RuntimeError(
            f"factorized automorphism group exceeds {maximum_slots} slots"
        )

    anchored_witnesses = []
    anchored_attempts = 0
    for candidates in signatures.values():
        for left_position, left_index in enumerate(candidates):
            for right_index in candidates[left_position + 1 :]:
                if any(
                    int(permutation[left_index]) == right_index
                    for permutation in induced_arrays
                ):
                    continue
                anchored_attempts += 1
                if anchored_attempts > 256:
                    raise RuntimeError(
                        "factorized anchored witness attempts exceeded frozen cap"
                    )
                source = representatives[left_index]
                target = representatives[right_index]
                current_witnesses = _anchored_action_witnesses(
                    augmented, source, target
                )
                if not current_witnesses:
                    current_witnesses = _anchored_action_witnesses(
                        augmented, source, target[::-1]
                    )
                for witness in current_witnesses:
                    key = _induced_torsion_permutation(
                        geometry, representatives, witness
                    )
                    if key not in induced:
                        induced.add(key)
                        anchored_witnesses.append(witness)
                if current_witnesses:
                    induced_arrays = _permutation_closure(
                        sorted(induced), maximum_size=maximum_permutations
                    )
                    groups = _permutation_orbits(
                        induced_arrays, len(representatives)
                    )
                    if any(len(group) > maximum_slots for group in groups):
                        raise RuntimeError(
                            f"factorized automorphism group exceeds {maximum_slots} slots"
                        )
    return FactorizedAutomorphismContract(
        groups=groups,
        representative_tuples=representatives,
        allowed_torsion_permutations=induced_arrays,
        induced_generator_count=len(induced),
        allowed_torsion_permutation_count=len(induced_arrays),
        branch_swap_witness_count=len(branch_witnesses),
        anchored_witness_count=len(anchored_witnesses),
    )


def optimal_factorized_circular_assignment(prediction, target, contract):
    import torch

    permutations = torch.as_tensor(
        contract.allowed_torsion_permutations,
        dtype=torch.long,
        device=prediction.device,
    )
    candidates = target[permutations]
    losses = torch.mean(
        1.0
        - torch.sum(prediction.unsqueeze(0) * candidates, dim=-1).clamp(-1.0, 1.0),
        dim=-1,
    )
    choice = int(torch.argmin(losses).detach().cpu())
    return candidates[choice]


def factorized_circular_errors_degrees(prediction, target, contract):
    import torch

    predicted = torch.as_tensor(prediction, dtype=torch.float64)
    targets = torch.as_tensor(target, dtype=torch.float64)
    aligned = optimal_factorized_circular_assignment(
        predicted, targets, contract
    ).numpy()
    cosine = np.sum(prediction * aligned, axis=-1).clip(-1.0, 1.0)
    sine = prediction[:, 0] * aligned[:, 1] - prediction[:, 1] * aligned[:, 0]
    return np.abs(np.rad2deg(np.arctan2(sine, cosine)))
