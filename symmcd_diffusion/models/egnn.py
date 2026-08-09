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

        # Layer normalization (eps=1e-5 is PyTorch default; slightly higher for
        # float16 safety to prevent divide-by-zero when variance approaches float16's
        # limited precision floor)
        self.node_norm = nn.LayerNorm(hidden_dim, eps=1e-4)

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        condition: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
        debug_nan: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            h: (N, hidden_dim) node features
            x: (N, 3) 3D coordinates
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_feat_dim) optional edge features
            condition: (batch_condition_dim) optional FiLM condition
            node_mask: (N,) optional node mask
            debug_nan: if True, checks for NaN after each operation and logs

        Returns:
            h_new: (N, hidden_dim) updated node features
            x_new: (N, 3) updated coordinates

        Note: This layer is AMP-safe. Under autocast, node feature computations
        run in float16 while coordinate updates stay in float32 for precision.
        """
        import math

        def _check(tensor, name):
            if not debug_nan:
                return
            if isinstance(tensor, torch.Tensor):
                n_nan = torch.isnan(tensor).sum().item()
                n_inf = torch.isinf(tensor).sum().item()
                if n_nan > 0 or n_inf > 0:
                    print(f"    [EGNN_DEBUG] NaN={n_nan} Inf={n_inf} after {name} "
                          f"(shape={tensor.shape}, min={tensor.min().item():.4g}, "
                          f"max={tensor.max().item():.4g})", flush=True)

        row, col = edge_index[0], edge_index[1]

        # Working dtype: node features may be float16 under AMP
        work_dtype = h.dtype
        _check(h, "h_input")

        # ---- Pairwise distances (coordinates stay float32) ----
        x_f32 = x.float()
        coord_diff = x_f32[row] - x_f32[col]  # (E, 3)
        _check(coord_diff, "coord_diff")
        dist_sq = (coord_diff ** 2).sum(dim=-1, keepdim=True)  # (E, 1)
        _check(dist_sq, "dist_sq")

        # ---- Edge messages ----
        # Build edge input with consistent dtype
        edge_feats = [h[row].to(work_dtype), h[col].to(work_dtype),
                      dist_sq.to(work_dtype)]
        if edge_attr is not None:
            # Ensure edge_attr is 2D: (E, feat_dim). During sampling
            # gumbel_softmax may produce (E, 1, feat_dim).
            ea = edge_attr.to(work_dtype)
            if ea.dim() == 3:
                ea = ea.squeeze(1)
            edge_feats.append(ea)
        edge_input = torch.cat(edge_feats, dim=-1)
        edge_input = edge_input.clamp(-100.0, 100.0)  # prevent SiLU overflow
        _check(edge_input, "edge_input")
        m_ij = self.edge_mlp(edge_input).to(work_dtype)  # enforce dtype
        # NaN safety on edge messages
        if torch.isnan(m_ij).any():
            m_ij = torch.nan_to_num(m_ij, nan=0.0)
        m_ij = m_ij.clamp(-100.0, 100.0)
        _check(m_ij, "edge_mlp")

        # Attention over edges (optional)
        # Force float32 for softmax to prevent overflow when scores are large.
        # Under AMP, edge features are float16; exp(score) can overflow float16
        # (max ~65504) when the model produces confident edge scores.
        if self.attention:
            attn_scores = self.attn_mlp(m_ij)  # (E, num_heads)
            # Clamp attention scores to prevent softmax overflow in float32.
            # With 20K edges, softmax(dim=0) normalizes over all edges; unbounded
            # SiLU activations can push scores to values where exp() overflows
            # even float32, producing Inf → NaN in coord_update and downstream.
            attn_scores = attn_scores.clamp(-50.0, 50.0)
            _check(attn_scores, "attn_scores")
            attn_weights = F.softmax(
                attn_scores.float(), dim=0
            ).to(work_dtype).mean(dim=-1, keepdim=True)
            _check(attn_weights, "attn_weights")
            m_ij = m_ij * attn_weights
            _check(m_ij, "m_ij_after_attn")

        # ---- Coordinate update (float32 for precision) ----
        coord_weight = self.coord_mlp(m_ij).float()  # (E, 1), force float32
        # Clamp coordinate weights to prevent explosion
        coord_weight = coord_weight.clamp(-10.0, 10.0)
        _check(coord_weight, "coord_weight")
        # Check the product BEFORE index_add
        contrib = (coord_diff * coord_weight).to(torch.float32)
        # Clamp per-edge contributions
        contrib = contrib.clamp(-50.0, 50.0)
        _check(contrib, "coord_diff*weight")
        # Compute node degree (number of incoming edges per atom)
        degree = torch.zeros(x_f32.size(0), 1, device=x.device, dtype=torch.float32)
        degree = degree.index_add(0, row, torch.ones_like(coord_weight, dtype=torch.float32))
        degree = degree.clamp(min=1)
        _check(degree, "degree")
        # Accumulate coordinate updates, then normalize by degree
        coord_update = torch.zeros_like(x_f32, dtype=torch.float32)
        coord_update = coord_update.index_add(
            0, row, contrib
        )
        _check(coord_update, "coord_update")
        coord_update = coord_update / degree  # (N, 3) / (N, 1) → (N, 3)
        # Clamp coordinate update to prevent explosion across layers
        coord_update = coord_update.clamp(-10.0, 10.0)
        x_new = x_f32 + coord_update
        # Clamp final coordinates to reasonable range
        x_new = x_new.clamp(-50.0, 50.0)
        _check(x_new, "x_new")

        # ---- Aggregate messages to nodes ----
        m_i = torch.zeros(
            h.size(0), self.hidden_dim, device=h.device, dtype=work_dtype
        )
        m_i = m_i.index_add(0, row, m_ij.to(work_dtype))
        _check(m_i, "m_i")
        m_i = m_i / degree.to(work_dtype)  # (N, H) / (N, 1) → (N, H)
        _check(m_i, "m_i_div")

        # ---- Node feature update ----
        node_mlp_in = torch.cat([h, m_i], dim=-1)
        _check(node_mlp_in, "node_mlp_in")
        # Clamp input to prevent SiLU overflow in node_mlp
        node_mlp_in = node_mlp_in.clamp(-100.0, 100.0)
        h_new = self.node_mlp(node_mlp_in)
        _check(h_new, "node_mlp_out")
        # NaN safety: replace NaN in node features before residual
        if torch.isnan(h_new).any():
            h_new = torch.nan_to_num(h_new, nan=0.0)
        h_new = h + h_new  # Residual connection
        h_new = h_new.clamp(-100.0, 100.0)
        _check(h_new, "residual")

        # FiLM conditioning
        if self.use_film and condition is not None:
            cond = condition
            # Ensure condition is (N, cond_dim) to match node features
            if cond.dim() == 1:
                # (cond_dim,) → (N, cond_dim)
                cond = cond.unsqueeze(0).expand(h.size(0), -1)
            elif cond.size(0) == 1 and cond.dim() == 2:
                # (1, cond_dim) → (N, cond_dim)
                cond = cond.expand(h.size(0), -1)
            elif cond.size(0) != h.size(0):
                # (batch, cond_dim) where batch > 1 but doesn't match N
                cond = cond.mean(dim=0, keepdim=True).expand(h.size(0), -1)
            # Now cond should be (N, cond_dim)
            cond_expanded = cond.to(work_dtype)
            h_new = self.film(h_new, cond_expanded)
            _check(h_new, "film")

        # Layer normalization
        h_new = self.node_norm(h_new)
        _check(h_new, "layer_norm")

        # Apply node mask (ensure dtype matches)
        if node_mask is not None:
            h_new = h_new * node_mask.unsqueeze(-1).to(work_dtype)
            x_new = x_new * node_mask.unsqueeze(-1).to(torch.float32)

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
        debug_nan: bool = False,
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
            debug_nan: if True, passes debug flag to each layer

        Returns:
            h: (N, hidden_dim) final node features
            x: (N, 3) final coordinates
        """
        for i, layer in enumerate(self.layers):
            if debug_nan:
                import math
                h_nan = torch.isnan(h).sum().item()
                x_nan = torch.isnan(x).sum().item()
                if h_nan > 0 or x_nan > 0:
                    print(f"    [EGNN_DEBUG] NaN BEFORE layer {i}: h_nan={h_nan}, x_nan={x_nan}", flush=True)
            h, x = layer(h, x, edge_index, edge_attr, condition, node_mask, debug_nan=debug_nan)
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
        # t_emb may be (N,) per-atom (training) or (1,) per-molecule (inference)
        if t_emb.size(0) == h.size(0):
            h = h + t_emb
        else:
            # Single-molecule batch: expand from (1, H) to (N, H)
            h = h + t_emb.expand(h.size(0), -1)

        if self.condition_proj is not None and condition is not None:
            cond_emb = self.condition_proj(condition)
            if cond_emb.size(0) == h.size(0):
                h = h + cond_emb
            else:
                h = h + cond_emb.expand(h.size(0), -1)

        return h
