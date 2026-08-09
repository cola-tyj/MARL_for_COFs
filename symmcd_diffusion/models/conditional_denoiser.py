"""
Symmetry-conditioned denoiser for COF Core generation.

Two modes:
  - Core-only (default): condition on symmetry only. Diffusion generates the
    rigid molecular skeleton with Q/R attachment points.
  - Full BB: additionally conditions on connector count + functional group type.

Pipeline:
  Diffusion → Core (Q+R points) → pycofbuilder assembly → Complete Building Block
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
    Denoiser conditioned on molecular symmetry.

    Core-only mode (default):
      Condition = symmetry embedding (128-dim) via FiLM injection.
      Auxiliary head: symmetry classifier.

    Full BB mode (core_only=False):
      Condition = symmetry + connector + FG (384-dim fused).
      Auxiliary heads: symmetry classifier + connectivity predictor.
    """

    def __init__(
        self,
        num_atom_types: int = 12,   # Core vocabulary (H,C,N,O,F,S,B,Cu,Zn,Co,Q,R)
        num_bond_types: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 9,
        edge_feat_dim: int = 5,
        attention: bool = False,
        num_heads: int = 8,
        dropout: float = 0.1,
        # Symmetry
        num_point_groups: int = 16,
        symmetry_encoding_dim: int = 128,
        # Mode: core-only (symmetry only) or full BB (sym+conn+FG)
        core_only: bool = True,
        # Full BB extras (ignored in core_only mode)
        num_connector_types: int = 7,
        connector_embedding_dim: int = 128,
        num_func_group_types: int = 10,
        func_group_embedding_dim: int = 128,
        condition_dim: int = 384,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.core_only = core_only

        # Symmetry encoder (always used)
        self.symmetry_encoder = SymmetryEncoder(
            num_groups=num_point_groups,
            encoding_dim=symmetry_encoding_dim,
        )

        if core_only:
            # Simple: condition = symmetry embedding only
            self.condition_dim = symmetry_encoding_dim
            self.condition_fusion = None
            self.connector_embed = None
            self.func_group_embed = None
        else:
            # Full BB: symmetry + connector + FG → fused condition
            self.condition_dim = condition_dim
            self.connector_embed = nn.Embedding(num_connector_types, connector_embedding_dim)
            self.func_group_embed = nn.Embedding(num_func_group_types, func_group_embedding_dim)
            total_cond_raw = symmetry_encoding_dim + connector_embedding_dim + func_group_embedding_dim
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
            edge_feat_dim=edge_feat_dim,
            attention=attention,
            num_heads=num_heads,
            dropout=dropout,
            condition_dim=self.condition_dim,
            use_film=True,
        )

        # Auxiliary head: symmetry classifier (always present)
        self.symmetry_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_point_groups),
        )

        # Connectivity predictor (full BB mode only)
        self.connectivity_predictor = None
        if not core_only:
            self.connectivity_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_connector_types),
            )

    def encode_condition(
        self,
        point_group_idx: Tensor,
        num_connectors: Optional[Tensor] = None,
        func_group_type: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode conditioning information.

        Core-only mode: condition = symmetry embedding (128-dim).
        Full BB mode: condition = fused(symmetry + connector + FG) (384-dim).
        """
        symm_emb = self.symmetry_encoder(point_group_idx)

        if self.core_only:
            return symm_emb  # (batch, 128)

        # Full BB: fuse all conditions
        conn_emb = self.connector_embed(num_connectors)
        func_emb = self.func_group_embed(func_group_type)
        combined = torch.cat([symm_emb, conn_emb, func_emb], dim=-1)
        return self.condition_fusion(combined)

    def forward(
        self,
        atom_types: Tensor,
        positions: Tensor,
        t: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        point_group_idx: Optional[Tensor] = None,
        num_connectors: Optional[Tensor] = None,
        func_group_type: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass. In core-only mode, num_connectors and func_group_type are ignored.

        Returns dict with: coord_noise, atom_logits, bond_logits, symmetry_logits,
        and optionally connectivity_logits (full BB mode only).
        """
        # Encode condition
        if point_group_idx is not None:
            condition = self.encode_condition(point_group_idx, num_connectors, func_group_type)
        else:
            condition = None

        # Forward through base denoiser
        outputs = self.denoiser(
            atom_types=atom_types, positions=positions, t=t,
            edge_index=edge_index, edge_attr=edge_attr,
            condition=condition, node_mask=node_mask, batch=batch,
        )

        h = outputs["h"]  # (N, hidden_dim)

        # ── Symmetry classification (mean-pool per molecule) ──
        if batch is not None:
            num_mols = int(batch.max().item()) + 1
            h_pooled = torch.zeros(num_mols, h.size(-1), device=h.device, dtype=h.dtype)
            if node_mask is not None:
                h_pooled = h_pooled.index_add(0, batch, h * node_mask.unsqueeze(-1).to(h.dtype))
                mol_counts = torch.zeros(num_mols, device=h.device).index_add(0, batch, node_mask.float())
            else:
                h_pooled = h_pooled.index_add(0, batch, h)
                mol_counts = torch.zeros(num_mols, device=h.device).index_add(
                    0, batch, torch.ones(h.size(0), device=h.device))
            h_pooled = h_pooled / mol_counts.clamp(min=1).unsqueeze(-1)
        else:
            h_pooled = h.mean(dim=0, keepdim=True)  # (1, hidden_dim)

        outputs["symmetry_logits"] = self.symmetry_classifier(h_pooled)

        # Connectivity prediction (full BB mode only)
        if self.connectivity_predictor is not None:
            outputs["connectivity_logits"] = self.connectivity_predictor(h)

        return outputs
