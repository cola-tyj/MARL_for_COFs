"""Graph-only long-range pair-orbit contract and augmented M3 predictor."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .geometry import _normal_bond, _tuple_orbits
from .m3p1_model import AutomorphismSetQuotientICPredictor
from .orbit_ic_model import _group_phase_features


@dataclass(frozen=True)
class GlobalPairGraph:
    pair_index: np.ndarray
    pair_orbit_id: np.ndarray
    pair_orbit_endpoints: np.ndarray
    pair_orbit_features: np.ndarray
    pair_phase_features: np.ndarray


def _graph_distances(sample) -> np.ndarray:
    count = len(sample.atomic_numbers)
    neighbors = [[] for _ in range(count)]
    for left, right in np.asarray(sample.bond_index, dtype=np.int64).T:
        neighbors[int(left)].append(int(right))
        neighbors[int(right)].append(int(left))
    distances = np.full((count, count), count + 1, dtype=np.int64)
    for source in range(count):
        distances[source, source] = 0
        queue = [source]
        for node in queue:
            for neighbor in neighbors[node]:
                if distances[source, neighbor] > distances[source, node] + 1:
                    distances[source, neighbor] = distances[source, node] + 1
                    queue.append(neighbor)
    if np.any(distances > count):
        raise RuntimeError("global-pair contract requires a connected graph")
    return distances


def build_global_pair_graph(sample, *, minimum_graph_distance: int = 4) -> GlobalPairGraph:
    """Build long-range pair orbits without reading Cartesian coordinates."""

    distances = _graph_distances(sample)
    pairs = [
        pair
        for pair in combinations(range(len(sample.atomic_numbers)), 2)
        if distances[pair] >= minimum_graph_distance
    ]
    if not pairs:
        raise ValueError("molecule has no long-range pair at the frozen graph distance")
    orbit_id = _tuple_orbits(
        pairs, np.asarray(sample.permutation_index, dtype=np.int64), _normal_bond
    )
    orbit_count = int(np.max(orbit_id)) + 1
    representatives = np.asarray(
        [pairs[int(np.flatnonzero(orbit_id == current)[0])] for current in range(orbit_count)],
        dtype=np.int64,
    )
    atom_orbits = np.asarray(sample.orbit_id, dtype=np.int64)
    endpoints = atom_orbits[representatives].T
    diameter = int(np.max(distances[distances < len(sample.atomic_numbers) + 1]))
    features = []
    for left, right in representatives:
        distance = int(distances[left, right])
        features.append(
            (
                distance / max(diameter, 1),
                1.0 / distance,
                float(atom_orbits[left] == atom_orbits[right]),
                1.0,
            )
        )
    phase = _group_phase_features(sample, representatives.T)
    return GlobalPairGraph(
        pair_index=np.asarray(pairs, dtype=np.int64).T,
        pair_orbit_id=np.asarray(orbit_id, dtype=np.int64),
        pair_orbit_endpoints=endpoints.astype(np.int64),
        pair_orbit_features=np.asarray(features, dtype=np.float32),
        pair_phase_features=phase.astype(np.float32),
    )


def build_global_pair_targets(sample, graph: GlobalPairGraph) -> np.ndarray:
    coordinates = np.asarray(sample.symmetric_target_angstrom, dtype=np.float64)
    pairs = np.asarray(graph.pair_index, dtype=np.int64)
    lengths = np.linalg.norm(coordinates[pairs[0]] - coordinates[pairs[1]], axis=1)
    count = int(np.max(graph.pair_orbit_id)) + 1
    targets = np.asarray(
        [lengths[graph.pair_orbit_id == current].mean() for current in range(count)],
        dtype=np.float32,
    )
    if not np.isfinite(targets).all():
        raise RuntimeError("non-finite global-pair target")
    return targets


class GlobalPairAugmentedPredictor:
    def __new__(cls, **model_settings):
        import torch

        parent = AutomorphismSetQuotientICPredictor(**model_settings)
        hidden_dim = int(model_settings.get("hidden_dim", 96))
        input_dim = 3 * hidden_dim + 4 + 9

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parent = parent
                self.global_pair_head = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
                self._captured_global_pair_input = None

            def initialize_from_m3p3(self, parent_state: dict):
                self.parent.load_state_dict(parent_state, strict=True)
                bond_head = self.parent.parent.base.bond_head
                if bond_head[0].weight.shape != self.global_pair_head[0].weight.shape:
                    raise RuntimeError("bond and global-pair head inputs are incompatible")
                self.global_pair_head.load_state_dict(bond_head.state_dict(), strict=True)

            def configure_pair_head_only(self):
                for parameter in self.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.global_pair_head.parameters():
                    parameter.requires_grad_(True)
                return [
                    name
                    for name, parameter in self.named_parameters()
                    if parameter.requires_grad
                ]

            def forward(self, graph, automorphism, global_pairs, *, device: str):
                captured = []

                def capture_nodes(_module, _inputs, output):
                    captured.append(output)

                handle = self.parent.parent.base.norms[-1].register_forward_hook(
                    capture_nodes
                )
                try:
                    result = self.parent(graph, automorphism, device=device)
                finally:
                    handle.remove()
                if len(captured) != 1:
                    raise RuntimeError("failed to capture quotient node states")
                nodes = captured[0]
                graph_embedding = torch.cat(
                    (nodes.mean(dim=0), nodes.max(dim=0).values), dim=0
                )
                endpoints = torch.as_tensor(
                    global_pairs.pair_orbit_endpoints,
                    dtype=torch.long,
                    device=device,
                )
                rows = []
                for current in range(endpoints.shape[1]):
                    left, right = endpoints[:, current]
                    rows.append(
                        torch.cat(
                            (
                                nodes[left] + nodes[right],
                                torch.abs(nodes[left] - nodes[right]),
                                graph_embedding[:hidden_dim],
                                torch.as_tensor(
                                    global_pairs.pair_orbit_features[current],
                                    dtype=nodes.dtype,
                                    device=device,
                                ),
                                torch.as_tensor(
                                    global_pairs.pair_phase_features[current],
                                    dtype=nodes.dtype,
                                    device=device,
                                ),
                            )
                        )
                    )
                pair_input = torch.stack(rows)
                self._captured_global_pair_input = pair_input
                raw = self.global_pair_head(pair_input).squeeze(-1)
                result["global_pair_lengths"] = 0.5 + torch.nn.functional.softplus(raw)
                return result

            def predict_cached(self, pair_input):
                raw = self.global_pair_head(pair_input).squeeze(-1)
                return 0.5 + torch.nn.functional.softplus(raw)

        return _Model()
