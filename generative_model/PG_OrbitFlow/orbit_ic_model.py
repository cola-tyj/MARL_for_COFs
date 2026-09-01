"""Quotient-multigraph encoder and orbit-level internal-coordinate heads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ORBIT_IC_MODEL_SCHEMA_VERSION = "pg-orbitflow-quotient-ic-model-v1"


@dataclass(frozen=True)
class OrbitICGraph:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    bond_orbit_endpoints: np.ndarray
    bond_orbit_features: np.ndarray
    bond_phase_features: np.ndarray
    angle_orbit_nodes: np.ndarray
    angle_orbit_features: np.ndarray
    angle_phase_features: np.ndarray
    torsion_orbit_nodes: np.ndarray
    torsion_orbit_features: np.ndarray
    torsion_phase_features: np.ndarray


@dataclass(frozen=True)
class OrbitICTargets:
    bond_target_lengths: np.ndarray
    angle_target_cosines: np.ndarray
    torsion_target_sincos: np.ndarray


def _orbit_means(values: np.ndarray, orbit_id: np.ndarray) -> np.ndarray:
    identifiers = np.asarray(orbit_id, dtype=np.int64)
    if not len(identifiers):
        return np.empty((0,) + values.shape[1:], dtype=np.float64)
    expected = np.arange(int(identifiers.max()) + 1, dtype=np.int64)
    if not np.array_equal(np.unique(identifiers), expected):
        raise ValueError("geometry orbit IDs must be contiguous")
    return np.stack([values[identifiers == current].mean(axis=0) for current in expected])


def _circular_orbit_means(values: np.ndarray, orbit_id: np.ndarray) -> np.ndarray:
    means = _orbit_means(values, orbit_id)
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("torsion orbit has an undefined circular mean")
    return means / norms


def _edge_type_lookup(sample) -> dict[tuple[int, int], int]:
    result = {}
    for (left, right), kind in zip(
        np.asarray(sample.bond_index).T,
        np.asarray(sample.bond_types),
        strict=True,
    ):
        key = (min(int(left), int(right)), max(int(left), int(right)))
        if key in result:
            raise ValueError("duplicate canonical edge in IC graph")
        result[key] = int(kind)
    return result


def _bond_histogram_for_tuples(tuples, orbit_id, lookup) -> np.ndarray:
    count = int(np.max(orbit_id)) + 1
    result = np.zeros((count, 4), dtype=np.float64)
    for values, current in zip(tuples.T, orbit_id, strict=True):
        left, right = (int(value) for value in values)
        kind = lookup[(min(left, right), max(left, right))]
        result[int(current), kind - 1] += 1.0
    result /= result.sum(axis=1, keepdims=True)
    return result


def _group_phase_features(sample, tuple_representatives) -> np.ndarray:
    """Encode relative group/coset phase without using Cartesian coordinates.

    For each atom, average all operation matrices whose permutation maps the
    canonical representative of its atom orbit onto that atom.  Pairwise
    relative products distinguish quotient paths that share atom-orbit IDs but
    occupy different group phases.  Canonicalizing forward/reverse paths keeps
    the feature consistent with the undirected torsion convention.
    """

    matrices = np.asarray(sample.operation_matrices, dtype=np.float64)
    permutations = np.asarray(sample.permutation_index, dtype=np.int64)
    atom_orbits = np.asarray(sample.orbit_id, dtype=np.int64)
    representatives = {
        int(current): int(np.flatnonzero(atom_orbits == current)[0])
        for current in np.unique(atom_orbits)
    }
    atom_codes = []
    for atom, current in enumerate(atom_orbits):
        representative = representatives[int(current)]
        mask = permutations[:, representative] == atom
        if not bool(np.any(mask)):
            raise ValueError("atom is unreachable from its orbit representative")
        atom_codes.append(matrices[mask].mean(axis=0))
    atom_codes = np.asarray(atom_codes, dtype=np.float64)
    tuples = np.asarray(tuple_representatives, dtype=np.int64)
    if tuples.ndim != 2 or tuples.shape[0] < 2:
        raise ValueError("geometry tuple representatives must be [K,M], K>=2")
    rows = []
    for atoms in tuples.T:
        codes = [atom_codes[int(atom)] for atom in atoms]
        relative = [
            codes[index] @ codes[index + 1].T
            for index in range(len(codes) - 1)
        ]
        forward = np.concatenate([matrix.reshape(-1) for matrix in relative])
        reverse = np.concatenate(
            [matrix.T.reshape(-1) for matrix in reversed(relative)]
        )
        if tuple(np.round(reverse, 12)) < tuple(np.round(forward, 12)):
            forward = reverse
        rows.append(forward)
    dimension = 9 * (tuples.shape[0] - 1)
    result = np.asarray(rows, dtype=np.float64).reshape(-1, dimension)
    if result.shape != (len(rows), dimension) or not np.isfinite(result).all():
        raise RuntimeError("invalid geometry group-phase feature")
    return result


def build_orbit_ic_example(sample, contract) -> tuple[OrbitICGraph, OrbitICTargets]:
    """Create strict graph-only features and orbit-level bond/angle targets."""

    atom_orbits = np.asarray(sample.orbit_id, dtype=np.int64)
    representatives = np.asarray(sample.quotient_graph.representative_indices)
    orbit_count = sample.quotient_graph.orbit_count
    if len(representatives) != orbit_count:
        raise ValueError("quotient representative count mismatch")
    atom_types = np.asarray(sample.atom_types, dtype=np.int64)[representatives]
    type_onehot = np.eye(13, dtype=np.float64)[atom_types]
    charges = np.asarray(sample.formal_charges, dtype=np.float64)[representatives, None] / 4.0
    radicals = np.asarray(sample.radical_electrons, dtype=np.float64)[representatives, None] / 4.0
    atom_group = np.asarray(sample.atom_group_features, dtype=np.float64)[representatives]
    global_group = np.repeat(
        np.asarray(sample.group_features, dtype=np.float64)[None, :], orbit_count, axis=0
    )
    pg_onehot = np.zeros((orbit_count, 2), dtype=np.float64)
    if sample.target_pg not in {"C2", "C3"}:
        raise ValueError("M1 only supports the frozen C2/C3 panel")
    pg_onehot[:, 0 if sample.target_pg == "C2" else 1] = 1.0
    node_features = np.concatenate(
        (type_onehot, charges, radicals, atom_group, global_group, pg_onehot), axis=1
    )

    quotient = sample.quotient_graph
    q_edges = np.asarray(quotient.edge_orbit_index, dtype=np.int64)
    multiplicity = np.log1p(np.asarray(quotient.edge_multiplicity, dtype=np.float64))[:, None]
    histogram = np.asarray(quotient.bond_type_histogram, dtype=np.float64)
    histogram /= histogram.sum(axis=1, keepdims=True)
    edge_features = np.concatenate((multiplicity, histogram), axis=1)

    lookup = _edge_type_lookup(sample)
    bond_count = int(np.max(contract.bond_orbit_id)) + 1
    bond_representatives = np.asarray(
        [
            contract.bond_index[:, np.flatnonzero(contract.bond_orbit_id == current)[0]]
            for current in range(bond_count)
        ],
        dtype=np.int64,
    ).T
    bond_endpoints = atom_orbits[bond_representatives]
    bond_histogram = _bond_histogram_for_tuples(
        contract.bond_index, contract.bond_orbit_id, lookup
    )
    bond_phase_features = _group_phase_features(sample, bond_representatives)

    angle_count = int(np.max(contract.angle_orbit_id)) + 1
    angle_representatives = np.asarray(
        [
            contract.angle_index[:, np.flatnonzero(contract.angle_orbit_id == current)[0]]
            for current in range(angle_count)
        ],
        dtype=np.int64,
    ).T
    angle_nodes = atom_orbits[angle_representatives]
    angle_bond_features = np.zeros((angle_count, 4), dtype=np.float64)
    for current, (left, center, right) in enumerate(angle_representatives.T):
        for endpoint in (left, right):
            kind = lookup[(min(int(endpoint), int(center)), max(int(endpoint), int(center)))]
            angle_bond_features[current, kind - 1] += 0.5
    angle_phase_features = _group_phase_features(sample, angle_representatives)

    torsion_count = int(np.max(contract.torsion_orbit_id)) + 1
    torsion_representatives = np.asarray(
        [
            contract.torsion_index[
                :, np.flatnonzero(contract.torsion_orbit_id == current)[0]
            ]
            for current in range(torsion_count)
        ],
        dtype=np.int64,
    ).T
    torsion_nodes = atom_orbits[torsion_representatives]
    torsion_bond_features = np.zeros((torsion_count, 8), dtype=np.float64)
    for current, (first, center_left, center_right, last) in enumerate(
        torsion_representatives.T
    ):
        terminal_kinds = [
            lookup[(min(int(first), int(center_left)), max(int(first), int(center_left)))],
            lookup[(min(int(center_right), int(last)), max(int(center_right), int(last)))],
        ]
        for kind in terminal_kinds:
            torsion_bond_features[current, kind - 1] += 0.5
        center_kind = lookup[
            (
                min(int(center_left), int(center_right)),
                max(int(center_left), int(center_right)),
            )
        ]
        torsion_bond_features[current, 4 + center_kind - 1] = 1.0
    torsion_phase_features = _group_phase_features(
        sample, torsion_representatives
    )

    target = np.asarray(sample.symmetric_target_angstrom, dtype=np.float64)
    bond_vectors = target[contract.bond_index[0]] - target[contract.bond_index[1]]
    bond_lengths = np.linalg.norm(bond_vectors, axis=1)
    left = target[contract.angle_index[0]] - target[contract.angle_index[1]]
    right = target[contract.angle_index[2]] - target[contract.angle_index[1]]
    angle_cosines = np.sum(left * right, axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    ).clip(1e-12)
    first = target[contract.torsion_index[0]] - target[contract.torsion_index[1]]
    axis = target[contract.torsion_index[2]] - target[contract.torsion_index[1]]
    last = target[contract.torsion_index[3]] - target[contract.torsion_index[2]]
    normal_left = np.cross(first, axis)
    normal_right = np.cross(axis, last)
    normal_left /= np.linalg.norm(normal_left, axis=1, keepdims=True).clip(1e-12)
    normal_right /= np.linalg.norm(normal_right, axis=1, keepdims=True).clip(1e-12)
    axis_unit = axis / np.linalg.norm(axis, axis=1, keepdims=True).clip(1e-12)
    torsion_cosine = np.sum(normal_left * normal_right, axis=1)
    torsion_sine = np.sum(np.cross(normal_left, normal_right) * axis_unit, axis=1)
    torsion_sincos = np.stack((torsion_sine, torsion_cosine), axis=1)
    return (
        OrbitICGraph(
            node_features=node_features.astype(np.float32),
            edge_index=q_edges,
            edge_features=edge_features.astype(np.float32),
            bond_orbit_endpoints=bond_endpoints,
            bond_orbit_features=bond_histogram.astype(np.float32),
            bond_phase_features=bond_phase_features.astype(np.float32),
            angle_orbit_nodes=angle_nodes,
            angle_orbit_features=angle_bond_features.astype(np.float32),
            angle_phase_features=angle_phase_features.astype(np.float32),
            torsion_orbit_nodes=torsion_nodes,
            torsion_orbit_features=torsion_bond_features.astype(np.float32),
            torsion_phase_features=torsion_phase_features.astype(np.float32),
        ),
        OrbitICTargets(
            bond_target_lengths=_orbit_means(
                bond_lengths, contract.bond_orbit_id
            ).astype(np.float32),
            angle_target_cosines=_orbit_means(
                angle_cosines, contract.angle_orbit_id
            ).astype(np.float32),
            torsion_target_sincos=_circular_orbit_means(
                torsion_sincos, contract.torsion_orbit_id
            ).astype(np.float32),
        ),
    )


class QuotientICPredictor:
    """Factory wrapper so importing data utilities does not require PyTorch."""

    def __new__(
        cls,
        *,
        node_feature_dim: int = 29,
        edge_feature_dim: int = 5,
        hidden_dim: int = 96,
        layers: int = 4,
        predict_torsion: bool = False,
        geometry_group_phase: bool = False,
    ):
        import torch

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.node_projection = torch.nn.Sequential(
                    torch.nn.Linear(node_feature_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                )
                self.messages = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Linear(hidden_dim + edge_feature_dim, hidden_dim),
                            torch.nn.SiLU(),
                            torch.nn.Linear(hidden_dim, hidden_dim),
                        )
                        for _ in range(layers)
                    ]
                )
                self.updates = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Linear(2 * hidden_dim, hidden_dim),
                            torch.nn.SiLU(),
                            torch.nn.Linear(hidden_dim, hidden_dim),
                        )
                        for _ in range(layers)
                    ]
                )
                self.norms = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(hidden_dim) for _ in range(layers)]
                )
                pair_dim = 3 * hidden_dim + 4 + (9 if geometry_group_phase else 0)
                angle_dim = 4 * hidden_dim + 4 + (18 if geometry_group_phase else 0)
                self.bond_head = torch.nn.Sequential(
                    torch.nn.Linear(pair_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
                self.angle_head = torch.nn.Sequential(
                    torch.nn.Linear(angle_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
                if predict_torsion:
                    torsion_dim = 5 * hidden_dim + 8 + (27 if geometry_group_phase else 0)
                    self.torsion_head = torch.nn.Sequential(
                        torch.nn.Linear(torsion_dim, hidden_dim),
                        torch.nn.SiLU(),
                        torch.nn.Linear(hidden_dim, 2),
                    )

            def forward(self, graph: OrbitICGraph, *, device: str):
                def tensor(values, dtype=torch.float32):
                    return torch.as_tensor(values, dtype=dtype, device=device)

                nodes = self.node_projection(tensor(graph.node_features))
                edge_index = tensor(graph.edge_index, torch.long)
                edge_features = tensor(graph.edge_features)
                for message_layer, update_layer, norm in zip(
                    self.messages, self.updates, self.norms, strict=True
                ):
                    aggregate = torch.zeros_like(nodes)
                    for edge in range(edge_index.shape[1]):
                        left = int(edge_index[0, edge])
                        right = int(edge_index[1, edge])
                        feature = edge_features[edge]
                        aggregate[left] += message_layer(
                            torch.cat((nodes[right], feature), dim=0)
                        )
                        if right != left:
                            aggregate[right] += message_layer(
                                torch.cat((nodes[left], feature), dim=0)
                            )
                    nodes = norm(nodes + update_layer(torch.cat((nodes, aggregate), dim=1)))
                graph_embedding = torch.cat(
                    (nodes.mean(dim=0), nodes.max(dim=0).values), dim=0
                )

                bond_endpoints = tensor(graph.bond_orbit_endpoints, torch.long)
                bond_rows = []
                for current in range(bond_endpoints.shape[1]):
                    left, right = bond_endpoints[:, current]
                    pair = torch.cat(
                        (
                            nodes[left] + nodes[right],
                            torch.abs(nodes[left] - nodes[right]),
                            graph_embedding[:hidden_dim],
                            tensor(graph.bond_orbit_features[current]),
                            *(
                                (tensor(graph.bond_phase_features[current]),)
                                if geometry_group_phase
                                else ()
                            ),
                        )
                    )
                    bond_rows.append(pair)
                raw_bond = self.bond_head(torch.stack(bond_rows)).squeeze(-1)
                bond_lengths = 0.5 + torch.nn.functional.softplus(raw_bond)

                angle_nodes = tensor(graph.angle_orbit_nodes, torch.long)
                angle_rows = []
                for current in range(angle_nodes.shape[1]):
                    left, center, right = angle_nodes[:, current]
                    row = torch.cat(
                        (
                            nodes[center],
                            nodes[left] + nodes[right],
                            torch.abs(nodes[left] - nodes[right]),
                            graph_embedding[:hidden_dim],
                            tensor(graph.angle_orbit_features[current]),
                            *(
                                (tensor(graph.angle_phase_features[current]),)
                                if geometry_group_phase
                                else ()
                            ),
                        )
                    )
                    angle_rows.append(row)
                angle_cosines = torch.tanh(
                    self.angle_head(torch.stack(angle_rows)).squeeze(-1)
                )
                result = {
                    "bond_lengths": bond_lengths,
                    "angle_cosines": angle_cosines,
                }
                if predict_torsion:
                    torsion_nodes = tensor(graph.torsion_orbit_nodes, torch.long)
                    torsion_rows = []
                    for current in range(torsion_nodes.shape[1]):
                        first, center_left, center_right, last = torsion_nodes[:, current]
                        row = torch.cat(
                            (
                                nodes[first] + nodes[last],
                                torch.abs(nodes[first] - nodes[last]),
                                nodes[center_left] + nodes[center_right],
                                torch.abs(nodes[center_left] - nodes[center_right]),
                                graph_embedding[:hidden_dim],
                                tensor(graph.torsion_orbit_features[current]),
                                *(
                                    (tensor(graph.torsion_phase_features[current]),)
                                    if geometry_group_phase
                                    else ()
                                ),
                            )
                        )
                        torsion_rows.append(row)
                    raw_torsion = self.torsion_head(torch.stack(torsion_rows))
                    result["torsion_sincos"] = raw_torsion / raw_torsion.norm(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-8)
                return result

        return _Model()
