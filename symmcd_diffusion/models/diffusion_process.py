"""
Diffusion process: forward noising and reverse sampling.

Handles the full diffusion lifecycle:
1. Forward process: q(x_t | x_0) = N(sqrt(alpha_cumprod)*x_0, (1-alpha_cumprod)*I)
2. Reverse process: p(x_{t-1} | x_t) learned by the denoiser

For mixed diffusion:
- Continuous coordinates: DDPM-style Gaussian diffusion
- Discrete atom/bond types: D3PM-style categorical diffusion
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import radius_graph

from .denoiser import Denoiser
from .noise_schedule import MixedNoiseScheduler


def build_fully_connected_edges(
    positions: Tensor,
    batch: Optional[Tensor] = None,
    cutoff: Optional[float] = None,
) -> Tensor:
    """
    Build edge index for molecular graph.

    For small molecules (< 50 atoms), use fully-connected edges.
    For larger molecules, use radius graph for efficiency.

    Args:
        positions: (N, 3) atom coordinates
        batch: (N,) batch indices
        cutoff: radius cutoff for sparse edges (None = fully connected)

    Returns:
        edge_index: (2, E) edge indices
    """
    if batch is None:
        batch = torch.zeros(positions.size(0), dtype=torch.long, device=positions.device)

    if cutoff is not None:
        # Use radius graph for efficiency
        edge_index = radius_graph(positions, r=cutoff, batch=batch, max_num_neighbors=32)
    else:
        # Fully connected within each molecule
        edge_indices = []
        for b in batch.unique():
            mask = batch == b
            idx = mask.nonzero(as_tuple=True)[0]
            # Create all pairs
            n = idx.size(0)
            if n <= 1:
                continue
            row = idx.unsqueeze(1).expand(-1, n).reshape(-1)
            col = idx.unsqueeze(0).expand(n, -1).reshape(-1)
            # Remove self-loops
            keep = row != col
            edge_indices.append(torch.stack([row[keep], col[keep]], dim=0))

        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=positions.device)

    return edge_index


class DiffusionProcess(nn.Module):
    """
    Mixed continuous-discrete diffusion process for molecules.

    Usage:
        # Training
        loss = diffusion.training_step(batch, denoiser)

        # Sampling
        molecules = diffusion.sample(denoiser, num_atoms=20, condition=...)
    """

    def __init__(
        self,
        num_atom_types: int = 5,
        num_bond_types: int = 5,
        timesteps: int = 1000,
        noise_schedule: str = "cosine",
        coord_loss_weight: float = 1.0,
        atom_loss_weight: float = 1.0,
        bond_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.coord_loss_weight = coord_loss_weight
        self.atom_loss_weight = atom_loss_weight
        self.bond_loss_weight = bond_loss_weight

        # Noise scheduler
        self.scheduler = MixedNoiseScheduler(
            timesteps=timesteps,
            num_atom_types=num_atom_types,
            num_bond_types=num_bond_types,
            schedule=noise_schedule,
        )

    def training_step(
        self,
        data: Data,
        denoiser: Denoiser,
        atom_marginals: Tensor,
        bond_marginals: Tensor,
        condition: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Single training step: sample t, apply noise, predict, compute loss.

        Args:
            data: PyG Data with x (atom one-hot), positions, edge_attr, edge_index
            denoiser: the denoising network
            atom_marginals: (num_atom_types,) empirical distribution
            bond_marginals: (num_bond_types,) empirical distribution
            condition: (cond_dim,) optional conditioning

        Returns:
            loss: total loss
            loss_dict: individual loss components for logging
        """
        batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') and data.batch is not None else 1
        device = data.x.device

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)

        # ---- Forward diffusion ----
        # Continuous: coordinates
        xt, noise = self.scheduler.forward_continuous(data.positions, t)

        # Discrete: atom types
        at, a0 = self.scheduler.forward_discrete(
            data.x, t, atom_marginals, "atom"
        )

        # Discrete: bond types (if present)
        if data.edge_attr is not None:
            et, e0 = self.scheduler.forward_discrete(
                data.edge_attr, t, bond_marginals, "bond"
            )
        else:
            et, e0 = None, None

        # ---- Denoiser prediction ----
        pred = denoiser(
            atom_types=at,
            positions=xt,
            t=t,
            edge_index=data.edge_index,
            edge_attr=et,
            condition=condition,
            node_mask=data.mask if hasattr(data, 'mask') else None,
        )

        # ---- Loss computation ----
        # Coordinate loss: MSE between predicted and true noise
        coord_loss = F.mse_loss(pred["coord_noise"], noise, reduction="mean")

        # Atom type loss: cross-entropy
        atom_loss = F.cross_entropy(
            pred["atom_logits"],
            data.x.argmax(dim=-1),
            reduction="mean",
        )

        # Bond type loss: cross-entropy
        if data.edge_attr is not None and et is not None:
            bond_loss = F.cross_entropy(
                pred["bond_logits"],
                data.edge_attr.argmax(dim=-1),
                reduction="mean",
            )
        else:
            bond_loss = torch.tensor(0.0, device=device)

        # Total loss
        loss = (
            self.coord_loss_weight * coord_loss +
            self.atom_loss_weight * atom_loss +
            self.bond_loss_weight * bond_loss
        )

        loss_dict = {
            "coord_loss": coord_loss.item(),
            "atom_loss": atom_loss.item(),
            "bond_loss": bond_loss.item(),
            "total_loss": loss.item(),
        }

        return loss, loss_dict

    @torch.no_grad()
    def sample(
        self,
        denoiser: Denoiser,
        num_atoms: int,
        atom_marginals: Tensor,
        bond_marginals: Tensor,
        condition: Optional[Tensor] = None,
        device: str = "cuda",
        return_trajectory: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Sample a new molecule via ancestral sampling.

        Args:
            denoiser: trained denoising network
            num_atoms: number of atoms to generate
            atom_marginals: (num_atom_types,) empirical distribution
            bond_marginals: (num_bond_types,) empirical distribution
            condition: (cond_dim,) optional conditioning
            device: device to sample on
            return_trajectory: if True, return all intermediate states

        Returns:
            dict with 'atom_types', 'positions', 'bond_types', and optionally 'trajectory'
        """
        denoiser.eval()

        # Initialize from prior
        # Coordinates: N(0, 1) centered
        x = torch.randn(num_atoms, 3, device=device)

        # Atom types: sample from marginal
        a_logits = atom_marginals.log().unsqueeze(0).expand(num_atoms, -1)
        a = F.gumbel_softmax(a_logits, tau=1.0, hard=True)

        # Build edges (fully connected for small molecules)
        edge_index = build_fully_connected_edges(x)

        # Bond types: sample from marginal
        num_edges = edge_index.size(1)
        e_logits = bond_marginals.log().unsqueeze(0).expand(num_edges, -1)
        e = F.gumbel_softmax(e_logits, tau=1.0, hard=True) if num_edges > 0 else None

        trajectory = [] if return_trajectory else None

        # Ancestral sampling: T → 0
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.tensor([t], device=device)

            # Predict denoised components
            pred = denoiser(
                atom_types=a,
                positions=x,
                t=t_tensor,
                edge_index=edge_index,
                edge_attr=e,
                condition=condition,
            )

            # Reverse continuous diffusion (coordinates)
            x = self.scheduler.reverse_continuous(
                x, pred["coord_noise"], t_tensor
            )

            # Reverse discrete diffusion (atom types)
            a = self.scheduler.reverse_discrete(
                a, pred["atom_logits"], t_tensor, "atom"
            )

            # Reverse discrete diffusion (bond types)
            if e is not None and num_edges > 0:
                e = self.scheduler.reverse_discrete(
                    e, pred["bond_logits"], t_tensor, "bond"
                )

            # Rebuild edges periodically (every 100 steps) as coordinates change
            if t % 100 == 0:
                edge_index = build_fully_connected_edges(x)
                num_edges = edge_index.size(1)
                if num_edges > 0 and e is not None and e.size(0) != num_edges:
                    e_logits = bond_marginals.log().unsqueeze(0).expand(num_edges, -1)
                    e = F.gumbel_softmax(e_logits, tau=1.0, hard=True)

            if return_trajectory:
                trajectory.append({
                    "t": t,
                    "x": x.clone(),
                    "a": a.clone(),
                    "e": e.clone() if e is not None else None,
                })

        result = {
            "atom_types": a,
            "positions": x,
            "bond_types": e,
            "edge_index": edge_index,
        }
        if return_trajectory:
            result["trajectory"] = trajectory

        denoiser.train()
        return result
