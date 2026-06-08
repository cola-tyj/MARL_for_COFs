"""
Symmetry-conditioned denoiser for COF building block generation.

Extends the base Denoiser with:
1. SymmetryEncoder for point group conditioning
2. Embedding layers for connectivity and functional group conditions
3. FiLM conditioning in EGNN layers
4. Auxiliary losses for symmetry and connectivity prediction
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .denoiser import Denoiser
from .egnn import EGNNEmbedding
from ..symmetry.symmetry_encoder import SymmetryEncoder


class SymmetryConditionedDenoiser(nn.Module):
    """
    Denoiser conditioned on molecular symmetry + connectivity specifications.

    Condition sources:
    1. point_group: encoded via SymmetryEncoder (binary + learned fusion)
    2. num_connectors: embedded count of connection points (1-6)
    3. func_group: embedded functional group type

    Conditioning is injected via FiLM layers in each EGNN layer.
    """

    def __init__(
        self,
        num_atom_types: int = 10,
        num_bond_types: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 9,
        attention: bool = True,
        num_heads: int = 8,
        dropout: float = 0.1,
        # Symmetry
        num_point_groups: int = 14,
        symmetry_encoding_dim: int = 128,
        # Connectivity
        num_connector_types: int = 7,
        connector_embedding_dim: int = 128,
        # Functional groups
        num_func_group_types: int = 10,
        func_group_embedding_dim: int = 128,
        # Overall condition
        condition_dim: int = 384,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        # Symmetry encoder
        self.symmetry_encoder = SymmetryEncoder(
            num_groups=num_point_groups,
            encoding_dim=symmetry_encoding_dim,
        )

        # Connectivity embedding
        self.connector_embed = nn.Embedding(
            num_connector_types, connector_embedding_dim
        )

        # Functional group embedding
        self.func_group_embed = nn.Embedding(
            num_func_group_types, func_group_embedding_dim
        )

        # Condition fusion: project concatenated conditions to condition_dim
        total_cond_raw = (
            symmetry_encoding_dim +
            connector_embedding_dim +
            func_group_embedding_dim
        )
        self.condition_fusion = nn.Sequential(
            nn.Linear(total_cond_raw, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        # Base denoiser with FiLM conditioning
        self.denoiser = Denoiser(
            num_atom_types=num_atom_types,
            num_bond_types=num_bond_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            attention=attention,
            num_heads=num_heads,
            dropout=dropout,
            condition_dim=condition_dim,
            use_film=True,
        )

        # Auxiliary heads (for additional supervision)
        # Symmetry classifier: predict point group from node features
        self.symmetry_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_point_groups),
        )

        # Connectivity predictor: predict number of connectors
        self.connectivity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_connector_types),
        )

    def encode_condition(
        self,
        point_group_idx: Tensor,
        num_connectors: Tensor,
        func_group_type: Tensor,
    ) -> Tensor:
        """
        Encode all conditioning information into a single vector.

        Args:
            point_group_idx: (batch,) point group indices
            num_connectors: (batch,) number of connectors (1-6)
            func_group_type: (batch,) functional group type indices

        Returns:
            condition: (batch, condition_dim) fused condition vector
        """
        symm_emb = self.symmetry_encoder(point_group_idx)
        conn_emb = self.connector_embed(num_connectors)
        func_emb = self.func_group_embed(func_group_type)

        combined = torch.cat([symm_emb, conn_emb, func_emb], dim=-1)
        condition = self.condition_fusion(combined)

        return condition

    def forward(
        self,
        atom_types: Tensor,
        positions: Tensor,
        t: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        point_group_idx: Optional[Tensor] = None,
        num_connectors: Optional[Tensor] = None,
        func_group_type: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass with all conditions.

        Args:
            atom_types: (N, num_atom_types) noisy atom type probs
            positions: (N, 3) noisy coordinates
            t: (batch,) timestep indices
            edge_index: (2, E) edge indices
            edge_attr: (E, num_bond_types) noisy bond types
            point_group_idx: (batch,) target point group
            num_connectors: (batch,) target connector count
            func_group_type: (batch,) target functional group
            node_mask: (N,) optional mask

        Returns:
            dict with denoising outputs + auxiliary predictions
        """
        # Encode conditions
        if point_group_idx is not None:
            condition = self.encode_condition(
                point_group_idx, num_connectors, func_group_type
            )
        else:
            condition = None

        # Forward through conditional denoiser
        outputs = self.denoiser(
            atom_types=atom_types,
            positions=positions,
            t=t,
            edge_index=edge_index,
            edge_attr=edge_attr,
            condition=condition,
            node_mask=node_mask,
        )

        # Auxiliary predictions
        h = outputs["h"]

        # Symmetry classification (global: mean pool node features)
        if node_mask is not None:
            h_masked = h * node_mask.unsqueeze(-1)
            h_pooled = h_masked.sum(dim=0) / node_mask.sum().clamp(min=1)
        else:
            h_pooled = h.mean(dim=0)

        outputs["symmetry_logits"] = self.symmetry_classifier(h_pooled)

        # Connectivity prediction (per-node, only on Q/X positions)
        outputs["connectivity_logits"] = self.connectivity_predictor(h)

        return outputs
