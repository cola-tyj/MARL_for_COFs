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
        batch: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass of the denoiser.

        Args:
            atom_types: (N, num_atom_types) noisy atom type probs/one-hot
            positions: (N, 3) noisy coordinates
            t: (N,) per-atom OR (batch_size,) per-molecule timestep indices
            edge_index: (2, E) fully-connected edge indices
            edge_attr: (E, num_bond_types) noisy bond type probs
            condition: (batch_size, cond_dim) optional conditioning vector
            node_mask: (N,) optional mask for padding
            batch: (N,) batch assignment (needed if t is per-molecule)

        Returns:
            dict with:
                'coord_noise': (N, 3) predicted coordinate noise
                'atom_logits': (N, num_atom_types) predicted clean atom logits
                'bond_logits': (E, num_bond_types) predicted clean bond logits
        """
        from torch.cuda.amp import autocast

        # Expand per-molecule timesteps to per-atom if needed
        if t.size(0) != atom_types.size(0) and batch is not None:
            t = t[batch]  # (batch_size,) → (N,)

        # ── Entire forward pass in float32 ────────────────────────────────────
        # AMP (float16) causes NaN in epoch 2+ because model weight drift makes
        # intermediate activations exceed float16 max (65504). Root cause is in
        # the edge_mlp and node_mlp matmuls inside EGNN layers.
        #
        # We disable autocast for the WHOLE denoiser forward. Loss computation
        # in diffusion_process.py is independently float32-safe.
        # Cost: ~30% slower training, but guarantees numerical stability.
        with autocast(enabled=False):
            # Embedding (one-hot → Linear, time embed, optional condition)
            condition_input = condition.float() if condition is not None else None
            if self.egnn.layers[0].use_film and condition is not None:
                h = self.embedding(atom_types.float(), t, condition=None)
                c_cond = condition_input
            else:
                h = self.embedding(atom_types.float(), t, condition=condition_input)
                c_cond = None

            # EGNN backbone
            x = positions.float()
            e = edge_attr.float() if edge_attr is not None else None
            m = node_mask.float() if node_mask is not None else None

            if self.egnn.layers[0].use_film and c_cond is not None:
                h, positions_out = self.egnn(h, x, edge_index, e, c_cond, m)
            else:
                h, positions_out = self.egnn(h, x, edge_index, e, None, m)

            # Output heads
            coord_noise = self.coord_head(h)
            atom_logits = self.atom_head(h)
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
