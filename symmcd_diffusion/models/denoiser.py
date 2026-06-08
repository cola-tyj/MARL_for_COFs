"""
Denoiser network for mixed continuous-discrete molecular diffusion.

The denoiser takes noisy molecular data (coordinates X, atom types A, bond types E)
and the diffusion timestep t, and predicts the clean components.

Architecture:
1. EGNNEmbedding: projects atom types + timestep → initial node features
2. EGNN: stacked equivariant layers process the graph
3. Output heads: predict clean coordinates, atom types, and bond types
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .egnn import EGNN, EGNNEmbedding


class Denoiser(nn.Module):
    """
    Mixed diffusion denoiser for molecular graphs.

    Predicts:
    - Noise for continuous coordinates (epsilon prediction)
    - Clean atom type logits (x0 prediction)
    - Clean bond type logits (x0 prediction)
    """

    def __init__(
        self,
        num_atom_types: int = 5,
        num_bond_types: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 9,
        edge_feat_dim: int = 0,
        attention: bool = True,
        num_heads: int = 8,
        dropout: float = 0.1,
        condition_dim: int = 0,
        use_film: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.condition_dim = condition_dim

        # Initial embedding
        self.embedding = EGNNEmbedding(
            num_atom_types=num_atom_types,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim if not use_film else 0,
        )

        # EGNN backbone
        self.egnn = EGNN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            edge_feat_dim=edge_feat_dim,
            attention=attention,
            num_heads=num_heads,
            use_film=use_film,
            condition_dim=condition_dim,
            dropout=dropout,
        )

        # Output heads
        # Coordinate head: predicts noise epsilon
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3, bias=False),
        )
        # Initialize to near-zero for stable training
        nn.init.xavier_uniform_(self.coord_head[-1].weight, gain=1e-3)

        # Atom type head: predicts clean logits
        self.atom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_atom_types),
        )

        # Bond type head: predicts clean logits from edge features
        self.bond_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_bond_types),
        )

    def forward(
        self,
        atom_types: Tensor,
        positions: Tensor,
        t: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        condition: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass of the denoiser.

        Args:
            atom_types: (N, num_atom_types) noisy atom type probs/one-hot
            positions: (N, 3) noisy coordinates
            t: (batch,) timestep indices
            edge_index: (2, E) fully-connected edge indices
            edge_attr: (E, num_bond_types) noisy bond type probs
            condition: (cond_dim,) optional conditioning vector
            node_mask: (N,) optional mask for padding

        Returns:
            dict with:
                'coord_noise': (N, 3) predicted coordinate noise
                'atom_logits': (N, num_atom_types) predicted clean atom logits
                'bond_logits': (E, num_bond_types) predicted clean bond logits
        """
        # Get initial node features
        h = self.embedding(atom_types, t, condition=None if self.egnn.layers[0].use_film else condition)

        # Pass through EGNN layers
        if self.egnn.layers[0].use_film and condition is not None:
            h, positions_out = self.egnn(
                h, positions, edge_index, edge_attr, condition, node_mask
            )
        else:
            h, positions_out = self.egnn(
                h, positions, edge_index, edge_attr, None, node_mask
            )

        # Predict coordinate noise (keep float32 for coordinate precision)
        coord_noise = self.coord_head(h).float()

        # Predict clean atom types
        atom_logits = self.atom_head(h)

        # Predict clean bond types
        row, col = edge_index[0], edge_index[1]
        edge_features = torch.cat([h[row], h[col]], dim=-1)
        bond_logits = self.bond_head(edge_features)

        return {
            "coord_noise": coord_noise,
            "atom_logits": atom_logits,
            "bond_logits": bond_logits,
            "h": h,
            "positions": positions_out,
        }
