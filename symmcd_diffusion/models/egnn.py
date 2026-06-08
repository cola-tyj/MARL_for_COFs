"""
Equivariant Graph Neural Network (EGNN) for molecular diffusion.

Adapted from MiDi (Vignac et al., 2023).
EGNN layers are E(n)-equivariant: they preserve equivariance to rotations,
translations, and reflections of 3D coordinates.

Key properties:
- Node features h_i: invariant to E(n) transformations
- Coordinates x_i: equivariant to E(n) transformations
- Messages depend on invariant distances ||x_i - x_j||^2
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """t: (batch,) integer timesteps in [0, T-1]"""
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    Applies conditioning: h' = gamma(condition) * h + beta(condition)
    """

    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: Tensor, condition: Tensor) -> Tensor:
        """
        h: (N, hidden_dim) node features
        condition: (batch_condition_dim) or (N, condition_dim)
        """
        gamma = self.gamma_net(condition)
        beta = self.beta_net(condition)
        return gamma * h + beta


class EGNNLayer(nn.Module):
    """
    Single EGNN layer with optional attention and FiLM conditioning.

    Message passing:
    1. Compute messages m_ij from node features h_i, h_j and distance d_ij^2
    2. Update coordinates: x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
    3. Aggregate messages: m_i = sum_j phi_m(m_ij)
    4. Update node features: h_i' = h_i + phi_h(h_i, m_i)

    All phi_* are MLPs.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int = 0,
        attention: bool = False,
        num_heads: int = 8,
        use_film: bool = False,
        condition_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = attention
        self.use_film = use_film

        # Edge message MLP
        # Input: h_i (hidden_dim) + h_j (hidden_dim) + d_ij^2 (1) + edge_attr (edge_feat_dim)
        edge_input_dim = 2 * hidden_dim + 1 + edge_feat_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate update MLP (phi_x)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        # Initialize last layer to near-zero for stable training
        nn.init.constant_(self.coord_mlp[-1].weight, 0.0)

        # Node update MLP (phi_h)
        # Input: h_i (hidden_dim) + aggregated_messages (hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Attention (optional)
        if attention:
            self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_dim, num_heads),
                nn.SiLU(),
                nn.Linear(num_heads, num_heads),
            )
            self.num_heads = num_heads

        # FiLM conditioning (optional)
        if use_film:
            self.film = FiLMLayer(hidden_dim, condition_dim)

        # Layer normalization
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        condition: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            h: (N, hidden_dim) node features
            x: (N, 3) 3D coordinates
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_feat_dim) optional edge features
            condition: (batch_condition_dim) optional FiLM condition
            node_mask: (N,) optional node mask

        Returns:
            h_new: (N, hidden_dim) updated node features
            x_new: (N, 3) updated coordinates
        """
        # Get source and target node indices
        row, col = edge_index[0], edge_index[1]

        # Compute pairwise distances
        coord_diff = x[row] - x[col]  # (E, 3)
        dist_sq = (coord_diff ** 2).sum(dim=-1, keepdim=True)  # (E, 1)

        # Build edge messages
        edge_input = torch.cat([h[row], h[col], dist_sq], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_input)  # (E, hidden_dim)

        # Attention over edges (optional)
        if self.attention:
            attn_scores = self.attn_mlp(m_ij)  # (E, num_heads)
            attn_weights = F.softmax(attn_scores, dim=0).mean(dim=-1, keepdim=True)
            m_ij = m_ij * attn_weights

        # Coordinate update
        coord_weight = self.coord_mlp(m_ij)  # (E, 1)
        # Normalize by degree for stability
        degree = torch.zeros(x.size(0), 1, device=x.device)
        degree = degree.index_add(0, row, torch.ones_like(coord_weight))
        degree = degree.clamp(min=1)
        coord_update = torch.zeros_like(x)
        coord_update = coord_update.index_add(
            0, row, coord_diff * coord_weight
        ) / degree[row]
        x_new = x + coord_update

        # Aggregate messages to nodes
        m_i = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        m_i = m_i.index_add(0, row, m_ij) / degree[row]

        # Node feature update
        h_new = self.node_mlp(torch.cat([h, m_i], dim=-1))
        h_new = h + h_new  # Residual connection

        # FiLM conditioning
        if self.use_film and condition is not None:
            # Expand condition to match node dimensions
            if condition.dim() == 1 or condition.size(0) != h.size(0):
                cond_expanded = condition.unsqueeze(0).expand(h.size(0), -1)
            else:
                cond_expanded = condition
            h_new = self.film(h_new, cond_expanded)

        # Layer normalization
        h_new = self.node_norm(h_new)

        # Apply node mask
        if node_mask is not None:
            h_new = h_new * node_mask.unsqueeze(-1)
            x_new = x_new * node_mask.unsqueeze(-1)

        return h_new, x_new


class EGNN(nn.Module):
    """
    Stack of EGNN layers forming the denoising backbone.

    Used in the diffusion denoiser to process noisy molecular graphs.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 9,
        edge_feat_dim: int = 0,
        attention: bool = True,
        num_heads: int = 8,
        use_film: bool = False,
        condition_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                edge_feat_dim=edge_feat_dim,
                attention=attention,
                num_heads=num_heads,
                use_film=use_film,
                condition_dim=condition_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        condition: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through all EGNN layers.

        Args:
            h: (N, hidden_dim) initial node features
            x: (N, 3) initial coordinates
            edge_index: (2, E) fully-connected edge indices
            edge_attr: (E, edge_feat_dim) optional edge attributes
            condition: (cond_dim,) optional global condition
            node_mask: (N,) optional mask for padding

        Returns:
            h: (N, hidden_dim) final node features
            x: (N, 3) final coordinates
        """
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr, condition, node_mask)
        return h, x


class EGNNEmbedding(nn.Module):
    """
    Initial embedding layer for EGNN input.
    Converts atom type one-hot and timestep to initial node features.
    """

    def __init__(
        self,
        num_atom_types: int,
        hidden_dim: int,
        time_embed_dim: Optional[int] = None,
        condition_dim: int = 0,
    ):
        super().__init__()
        self.atom_embed = nn.Linear(num_atom_types, hidden_dim)
        self.time_embed = SinusoidalTimeEmbedding(
            time_embed_dim or hidden_dim
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim or hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.condition_proj = None
        if condition_dim > 0:
            self.condition_proj = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(
        self,
        atom_types: Tensor,
        t: Tensor,
        condition: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            atom_types: (N, num_atom_types) one-hot or soft atom types
            t: (batch,) integer timesteps
            condition: (cond_dim,) optional global condition

        Returns:
            h: (N, hidden_dim) initial node features
        """
        h = self.atom_embed(atom_types.float())
        t_emb = self.time_proj(self.time_embed(t))

        # Add time embedding to node features
        # If t is per-atom (N,), t_emb shape matches h directly
        if t_emb.size(0) == h.size(0):
            h = h + t_emb
        else:
            # Per-molecule t: expand to all nodes
            h = h + t_emb.unsqueeze(0).expand(h.size(0), -1)

        if self.condition_proj is not None and condition is not None:
            cond_emb = self.condition_proj(condition)
            if cond_emb.size(0) == h.size(0):
                h = h + cond_emb
            else:
                h = h + cond_emb.unsqueeze(0).expand(h.size(0), -1)

        return h
