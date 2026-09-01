"""Graph/action-only centralizer automorphisms for torsion-set supervision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraphAutomorphismContract:
    groups: tuple[tuple[int, ...], ...]
    representative_tuples: np.ndarray
    allowed_atom_permutations: np.ndarray
    allowed_torsion_permutations: np.ndarray


def _representatives(contract) -> np.ndarray:
    identifiers = np.asarray(contract.torsion_orbit_id, dtype=np.int64)
    count = int(np.max(identifiers)) + 1
    return np.asarray(
        [contract.torsion_index[:, np.flatnonzero(identifiers == index)[0]] for index in range(count)],
        dtype=np.int64,
    )


def _canonical_graph(sample):
    import networkx as nx

    graph = nx.Graph()
    for index, (atom_type, charge, radical) in enumerate(
        zip(sample.atom_types, sample.formal_charges, sample.radical_electrons, strict=True)
    ):
        graph.add_node(index, atom_type=int(atom_type), charge=int(charge), radical=int(radical))
    for edge, bond_type in zip(sample.bond_index.T, sample.bond_types, strict=True):
        graph.add_edge(int(edge[0]), int(edge[1]), bond_type=int(bond_type))
    if not nx.is_connected(graph):
        raise RuntimeError("centralizer automorphism requires a connected canonical graph")
    return graph


def _centralizer_atom_permutations(sample, *, maximum_enumerated: int = 100_000) -> np.ndarray:
    import networkx as nx

    graph = _canonical_graph(sample)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph,
        graph,
        node_match=nx.algorithms.isomorphism.categorical_node_match(
            ("atom_type", "charge", "radical"), (None, None, None)
        ),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("bond_type", None),
    )
    actions = np.asarray(sample.permutation_index, dtype=np.int64)
    accepted = set()
    enumerated = 0
    for mapping in matcher.isomorphisms_iter():
        enumerated += 1
        if enumerated > maximum_enumerated:
            raise RuntimeError("canonical graph automorphism enumeration exceeded frozen safety cap")
        permutation = np.asarray([mapping[index] for index in range(len(graph))], dtype=np.int64)
        if any(not np.array_equal(permutation[action], action[permutation]) for action in actions):
            continue
        accepted.add(tuple(permutation.tolist()))
    if not accepted:
        raise RuntimeError("centralizer automorphism contract lost the identity")
    result = np.asarray(sorted(accepted), dtype=np.int64)
    identity = np.arange(len(graph), dtype=np.int64)
    if not any(np.array_equal(row, identity) for row in result):
        raise RuntimeError("centralizer automorphism identity missing")
    return result


def _induced_torsion_permutations(contract, representatives, atom_permutations) -> np.ndarray:
    lookup = {}
    for raw_index, values in enumerate(np.asarray(contract.torsion_index, dtype=np.int64).T):
        orbit = int(contract.torsion_orbit_id[raw_index])
        for key in (tuple(values.tolist()), tuple(values[::-1].tolist())):
            if key in lookup and lookup[key] != orbit:
                raise RuntimeError("torsion tuple maps to multiple orbit identifiers")
            lookup[key] = orbit
    induced = set()
    for permutation in atom_permutations:
        current = []
        for values in representatives:
            mapped = tuple(permutation[values].tolist())
            if mapped not in lookup:
                raise RuntimeError("graph automorphism does not preserve torsion orbit support")
            current.append(lookup[mapped])
        if sorted(current) != list(range(len(representatives))):
            raise RuntimeError("induced torsion automorphism is not bijective")
        induced.add(tuple(current))
    return np.asarray(sorted(induced), dtype=np.int64)


def _permutation_orbits(permutations: np.ndarray) -> tuple[tuple[int, ...], ...]:
    count = permutations.shape[1]
    remaining = set(range(count))
    groups = []
    while remaining:
        seed = min(remaining)
        reached = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            for target in permutations[:, current].tolist():
                if target not in reached:
                    reached.add(int(target)); frontier.append(int(target))
        remaining.difference_update(reached)
        if len(reached) > 1:
            groups.append(tuple(sorted(reached)))
    return tuple(groups)


def build_graph_automorphism_contract(sample, contract) -> GraphAutomorphismContract:
    representatives = _representatives(contract)
    atom_permutations = _centralizer_atom_permutations(sample)
    torsion_permutations = _induced_torsion_permutations(
        contract, representatives, atom_permutations
    )
    return GraphAutomorphismContract(
        groups=_permutation_orbits(torsion_permutations),
        representative_tuples=representatives,
        allowed_atom_permutations=atom_permutations,
        allowed_torsion_permutations=torsion_permutations,
    )


def optimal_automorphism_circular_assignment(prediction, target, contract):
    import torch

    permutations = torch.as_tensor(
        contract.allowed_torsion_permutations, dtype=torch.long, device=prediction.device
    )
    candidates = target[permutations]
    losses = torch.mean(
        1.0 - torch.sum(prediction.unsqueeze(0) * candidates, dim=-1).clamp(-1.0, 1.0),
        dim=-1,
    )
    choice = int(torch.argmin(losses).detach().cpu())
    return candidates[choice]


def automorphism_circular_errors_degrees(prediction, target, contract) -> np.ndarray:
    import torch

    predicted = torch.as_tensor(prediction, dtype=torch.float64)
    targets = torch.as_tensor(target, dtype=torch.float64)
    aligned = optimal_automorphism_circular_assignment(predicted, targets, contract).numpy()
    cosine = np.sum(prediction * aligned, axis=-1).clip(-1.0, 1.0)
    sine = prediction[:, 0] * aligned[:, 1] - prediction[:, 1] * aligned[:, 0]
    return np.abs(np.rad2deg(np.arctan2(sine, cosine)))


def best_centralizer_automorphism_relabeling(sample, contract, coordinates: np.ndarray) -> dict:
    from .metrics import kabsch_rmsd

    candidate = np.asarray(coordinates, dtype=np.float64)
    target = np.asarray(sample.symmetric_target_angstrom, dtype=np.float64)
    best = None
    for permutation in contract.allowed_atom_permutations:
        relabelled = candidate[permutation]
        rmsd = kabsch_rmsd(target, relabelled)
        key = (rmsd, tuple(permutation.tolist()))
        if best is None or key < best[0]:
            best = (key, permutation.copy(), relabelled.copy())
    return {
        "coordinates": best[2],
        "permutation": best[1],
        "kabsch_rmsd_angstrom": float(best[0][0]),
        "assignment_count": len(contract.allowed_atom_permutations),
        "nonidentity_assignment": bool(np.any(best[1] != np.arange(len(candidate)))),
    }
