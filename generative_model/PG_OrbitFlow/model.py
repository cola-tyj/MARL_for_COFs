"""Native E(3)-equivariant, point-group-aware molecular vector field."""

from __future__ import annotations

import math

import torch
from torch import nn


MODEL_SCHEMA_VERSION = "pg-orbitflow-equivariant-vector-field-v1"


class MLP(nn.Module):
    def __init__(self, dimensions: list[int], *, final_activation: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(dimensions) - 2 or final_activation:
                layers.extend((nn.SiLU(), nn.LayerNorm(right)))
        self.net = nn.Sequential(*layers)

    def forward(self, values):
        return self.net(values)


class EquivariantPairBlock(nn.Module):
    def __init__(self, hidden_dim: int, radial_dim: int) -> None:
        super().__init__()
        edge_input = hidden_dim * 2 + radial_dim + 2
        self.message = MLP([edge_input, hidden_dim, hidden_dim])
        self.node_update = MLP([hidden_dim * 2, hidden_dim, hidden_dim])
        self.coordinate_weight = MLP([hidden_dim, hidden_dim, 1])
        self.node_norm = nn.LayerNorm(hidden_dim)
        nn.init.zeros_(self.coordinate_weight.net[-1].weight)
        nn.init.zeros_(self.coordinate_weight.net[-1].bias)

    def forward(self, h, x, radial, bond_order, bonded, source, target):
        edge_input = torch.cat(
            (h[source], h[target], radial, bond_order[:, None], bonded[:, None]), dim=-1
        )
        messages = self.message(edge_input)
        aggregate = torch.zeros_like(h)
        aggregate.index_add_(0, source, messages)
        degree = torch.bincount(source, minlength=len(h)).to(h.dtype).clamp_min(1.0)
        aggregate = aggregate / torch.sqrt(degree[:, None])
        h = self.node_norm(h + self.node_update(torch.cat((h, aggregate), dim=-1)))

        relative = x[target] - x[source]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True).clamp_min(1e-6)
        weights = self.coordinate_weight(messages)
        vector_messages = weights * relative / (1.0 + distance)
        delta = torch.zeros_like(x)
        delta.index_add_(0, source, vector_messages)
        delta = delta / torch.sqrt(degree[:, None])
        return h, delta


class PGOrbitFlow(nn.Module):
    """Coordinate vector field that preserves a supplied molecular group action.

    Symmetry is preserved without a post-hoc hard-projection layer: complete
    directed pairs, group-invariant scalar features and relative coordinate
    messages form an E(3)- and atom-permutation-equivariant vector field.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        layers: int = 6,
        radial_dim: int = 16,
        radial_max: float = 6.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or layers <= 0 or radial_dim <= 1 or radial_max <= 0:
            raise ValueError("invalid PGOrbitFlow dimensions")
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.radial_dim = radial_dim
        self.radial_max = float(radial_max)
        self.atom_embedding = nn.Embedding(13, 32)
        self.charge_embedding = nn.Embedding(9, 8)
        self.radical_embedding = nn.Embedding(5, 4)
        self.pg_embedding = nn.Embedding(4, 16)
        input_dim = 32 + 8 + 4 + 16 + 8 + 4 + 8
        self.input_projection = MLP([input_dim, hidden_dim, hidden_dim])
        self.blocks = nn.ModuleList(
            [EquivariantPairBlock(hidden_dim, radial_dim) for _ in range(layers)]
        )
        self.layer_weights = nn.Parameter(torch.zeros(layers))
        centers = torch.linspace(0.0, radial_max, radial_dim)
        self.register_buffer("radial_centers", centers)
        self.radial_gamma = float((radial_dim - 1) ** 2 / (radial_max**2))

    @staticmethod
    def _time_features(time, *, dtype, device):
        value = torch.as_tensor(time, dtype=dtype, device=device).reshape(1)
        frequencies = torch.arange(1, 5, dtype=dtype, device=device) * math.pi
        return torch.cat((torch.sin(value[:, None] * frequencies), torch.cos(value[:, None] * frequencies)), dim=-1).reshape(-1)

    @staticmethod
    def _complete_pairs(atom_count: int, device):
        indices = torch.arange(atom_count, device=device)
        source = indices.repeat_interleave(atom_count)
        target = indices.repeat(atom_count)
        keep = source != target
        return source[keep], target[keep]

    def forward(
        self,
        *,
        positions,
        time,
        atom_types,
        formal_charges,
        radical_electrons,
        target_pg_index,
        group_features,
        atom_group_features,
        invariant_bond_order,
    ):
        if positions.ndim != 2 or positions.shape[-1] != 3 or len(positions) < 2:
            raise ValueError("positions must be [N,3] with N>=2")
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("positions contains NaN/Inf")
        atom_count = len(positions)
        for name, field in (
            ("atom_types", atom_types),
            ("formal_charges", formal_charges),
            ("radical_electrons", radical_electrons),
        ):
            if field.shape != (atom_count,):
                raise ValueError(f"{name} shape mismatch")
        if atom_group_features.shape != (atom_count, 4) or group_features.shape != (8,):
            raise ValueError("group feature shape mismatch")
        if invariant_bond_order.shape != (atom_count, atom_count):
            raise ValueError("invariant_bond_order shape mismatch")
        if int(atom_types.min()) < 0 or int(atom_types.max()) >= 13:
            raise ValueError("atom_types outside frozen vocabulary")
        if int(formal_charges.min()) < -4 or int(formal_charges.max()) > 4:
            raise ValueError("formal charge outside [-4,4]")
        if int(radical_electrons.min()) < 0 or int(radical_electrons.max()) > 4:
            raise ValueError("radical count outside [0,4]")
        pg_index = int(torch.as_tensor(target_pg_index).item())
        if not 0 <= pg_index < 4:
            raise ValueError("target_pg_index outside 0..3")

        centered = positions - positions.mean(dim=0, keepdim=True)
        time_features = self._time_features(time, dtype=centered.dtype, device=centered.device)
        pg = self.pg_embedding(torch.as_tensor(pg_index, device=centered.device)).expand(atom_count, -1)
        global_values = group_features.to(dtype=centered.dtype).expand(atom_count, -1)
        time_values = time_features.expand(atom_count, -1)
        h = self.input_projection(
            torch.cat(
                (
                    self.atom_embedding(atom_types),
                    self.charge_embedding(formal_charges + 4),
                    self.radical_embedding(radical_electrons),
                    pg,
                    global_values,
                    atom_group_features.to(dtype=centered.dtype),
                    time_values,
                ),
                dim=-1,
            )
        )
        source, target = self._complete_pairs(atom_count, centered.device)
        distance = torch.linalg.vector_norm(centered[target] - centered[source], dim=-1)
        radial = torch.exp(
            -self.radial_gamma * torch.square(distance[:, None] - self.radial_centers[None, :])
        )
        pair_order = invariant_bond_order[source, target].to(dtype=centered.dtype)
        bonded = (pair_order > 0).to(dtype=centered.dtype)
        deltas = []
        current = centered
        for block in self.blocks:
            h, delta = block(h, current, radial, pair_order, bonded, source, target)
            deltas.append(delta)
            current = current + 0.1 * delta
        weights = torch.softmax(self.layer_weights, dim=0)
        velocity = sum(weight * delta for weight, delta in zip(weights, deltas))
        velocity = velocity - velocity.mean(dim=0, keepdim=True)
        if not bool(torch.isfinite(velocity).all()):
            raise RuntimeError("PGOrbitFlow produced NaN/Inf")
        return velocity
