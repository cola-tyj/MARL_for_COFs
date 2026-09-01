"""Permutation-invariant long-range distance-quantile head for M3.5."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .global_pair_model import _graph_distances
from .m3p1_model import AutomorphismSetQuotientICPredictor


@dataclass(frozen=True)
class GlobalShapeContract:
    pair_index: np.ndarray
    quantile_levels: np.ndarray


def build_global_shape_contract(
    sample, *, minimum_graph_distance: int = 4, quantile_count: int = 16
) -> GlobalShapeContract:
    distances = _graph_distances(sample)
    pairs = [
        pair
        for pair in combinations(range(len(sample.atomic_numbers)), 2)
        if distances[pair] >= minimum_graph_distance
    ]
    if not pairs:
        raise ValueError("molecule has no pair in the global-shape support")
    if quantile_count < 4:
        raise ValueError("global-shape quantile count must be at least four")
    return GlobalShapeContract(
        pair_index=np.asarray(pairs, dtype=np.int64).T,
        quantile_levels=np.linspace(0.0, 1.0, quantile_count, dtype=np.float64),
    )


def global_shape_target(sample, contract: GlobalShapeContract) -> np.ndarray:
    coordinates = np.asarray(sample.symmetric_target_angstrom, dtype=np.float64)
    pairs = contract.pair_index
    distances = np.linalg.norm(coordinates[pairs[0]] - coordinates[pairs[1]], axis=1)
    return np.quantile(distances, contract.quantile_levels).astype(np.float32)


def coordinate_shape_quantiles(coordinates, contract: GlobalShapeContract):
    """Differentiable quantiles for a coordinate tensor."""

    import torch

    pairs = torch.as_tensor(
        contract.pair_index, dtype=torch.long, device=coordinates.device
    )
    distances = torch.linalg.vector_norm(
        coordinates[pairs[0]] - coordinates[pairs[1]], dim=-1
    )
    levels = torch.as_tensor(
        contract.quantile_levels, dtype=coordinates.dtype, device=coordinates.device
    )
    return torch.quantile(distances, levels)


class GlobalShapeAugmentedPredictor:
    def __new__(cls, *, quantile_count: int = 16, **model_settings):
        import torch

        parent = AutomorphismSetQuotientICPredictor(**model_settings)
        hidden_dim = int(model_settings.get("hidden_dim", 96))

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parent = parent
                self.global_shape_head = torch.nn.Sequential(
                    torch.nn.Linear(2 * hidden_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, quantile_count),
                )
                self._captured_graph_embedding = None

            def initialize_from_m3p3(self, parent_state: dict):
                self.parent.load_state_dict(parent_state, strict=True)

            def configure_shape_head_only(self):
                for parameter in self.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.global_shape_head.parameters():
                    parameter.requires_grad_(True)
                return [
                    name
                    for name, parameter in self.named_parameters()
                    if parameter.requires_grad
                ]

            def forward(self, graph, automorphism, *, device: str):
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
                embedding = torch.cat(
                    (nodes.mean(dim=0), nodes.max(dim=0).values), dim=0
                )
                self._captured_graph_embedding = embedding
                raw = self.global_shape_head(embedding)
                result["global_shape_quantiles"] = torch.sort(
                    0.5 + torch.nn.functional.softplus(raw)
                ).values
                return result

            def predict_cached(self, embedding):
                raw = self.global_shape_head(embedding)
                return torch.sort(0.5 + torch.nn.functional.softplus(raw)).values

        return _Model()
