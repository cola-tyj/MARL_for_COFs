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
            # Ensure indices are on the same device as positions
            dev = positions.device
            row, col = row.to(dev), col.to(dev)
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
        use_uniform_prior: bool = False,
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
            use_uniform_prior=use_uniform_prior,
        )

    def _expand_to_atoms(self, t_per_mol: Tensor, batch: Tensor) -> Tensor:
        """Expand per-molecule timestep to per-atom indexing.

        Args:
            t_per_mol: (batch_size,) timestep per molecule
            batch: (N,) batch assignment for each atom

        Returns:
            t_per_atom: (N,) timestep for each atom
        """
        return t_per_mol[batch]

    def expand_t_to_edges(self, t_per_mol: Tensor, batch: Tensor, edge_index: Tensor) -> Tensor:
        """Expand per-molecule timestep to per-edge indexing using source atom's batch."""
        row = edge_index[0]
        return t_per_mol[batch[row]]

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
        device = data.x.device

        # Get per-molecule timesteps
        if hasattr(data, 'batch') and data.batch is not None:
            batch_size = int(data.batch.max().item()) + 1
            batch = data.batch
        else:
            batch_size = 1
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        # Sample random timesteps (one per molecule)
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)

        # Expand to per-atom for continuous noise
        t_atom = self._expand_to_atoms(t, batch)

        # ---- Forward diffusion ----
        # Continuous: coordinates (per-atom timestep)
        xt, noise = self.scheduler.forward_continuous(data.positions, t_atom)

        # Discrete: atom types (need per-molecule transition; forward_discrete
        # handles this by iterating molecules)
        at, a0 = self.scheduler.forward_discrete_batched(
            data.x, t, batch, atom_marginals, "atom"
        )

        # Discrete: bond types (if present)
        if data.edge_attr is not None:
            et, e0 = self.scheduler.forward_discrete_batched(
                data.edge_attr, t, batch, bond_marginals, "bond",
                edge_index=data.edge_index,
            )
        else:
            et, e0 = None, None

        # ---- Denoiser prediction ----
        # Pass per-atom timestep so time embedding broadcasts correctly
        pred = denoiser(
            atom_types=at,
            positions=xt,
            t=t_atom,
            edge_index=data.edge_index,
            edge_attr=et,
            condition=condition,
            node_mask=data.mask if hasattr(data, 'mask') else None,
        )

        # ---- Loss computation ----
        # IMPORTANT: All loss computations are forced to float32 to prevent
        # float16 overflow in exp() inside cross_entropy / softmax under AMP.
        # When the model becomes confident (logit > 11), exp(logit) exceeds
        # float16 max (65504), producing inf → NaN.
        #
        # Coordinate loss: MSE between predicted and true noise
        coord_loss = F.mse_loss(
            pred["coord_noise"].float(), noise.float(), reduction="mean"
        )

        # Atom type loss: cross-entropy (float32-safe)
        atom_loss = F.cross_entropy(
            pred["atom_logits"].float(),
            data.x.argmax(dim=-1),
            reduction="mean",
        )

        # Bond type loss: cross-entropy (float32-safe)
        # Guard against empty edge_attr (e.g., single-atom molecules with no bonds).
        # F.cross_entropy with reduction='mean' on empty tensors returns NaN.
        if data.edge_attr is not None and et is not None and data.edge_attr.size(0) > 0:
            bond_loss = F.cross_entropy(
                pred["bond_logits"].float(),
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

        # Force all tensors to the same device
        if isinstance(device, str):
            device = torch.device(device)
        atom_marginals = atom_marginals.to(device)
        bond_marginals = bond_marginals.to(device)
        if condition is not None:
            condition = condition.to(device)

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
                a, pred["atom_logits"], t_tensor, atom_marginals, "atom"
            )

            # Reverse discrete diffusion (bond types)
            if e is not None and num_edges > 0:
                e = self.scheduler.reverse_discrete(
                    e, pred["bond_logits"], t_tensor, bond_marginals, "bond"
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

    @torch.no_grad()
    def sample_ddim(
        self,
        denoiser,
        num_atoms: int,
        atom_marginals: Tensor,
        bond_marginals: Tensor,
        condition: Optional[Tensor] = None,
        device: str = "cuda",
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> Dict[str, Tensor]:
        """
        Accelerated sampling via DDIM (Song et al., 2021).

        Coordinates: DDIM non-Markovian jumps (ddim_steps forward passes).
        Discrete (atom/bond): Full ancestral sampling (fast matrix ops).

        With ddim_steps=50, this is ~20x faster than ancestral sampling (1000 steps).

        Args:
            denoiser: trained denoising network
            num_atoms: number of atoms to generate
            atom_marginals: (num_atom_types,) empirical distribution
            bond_marginals: (num_bond_types,) empirical distribution
            condition: (cond_dim,) optional FiLM condition
            device: device to sample on
            ddim_steps: number of DDIM steps (default 50, ~0.3s for small molecules)
            eta: DDIM stochasticity (0=deterministic, 1=DDPM-like)

        Returns:
            dict with 'atom_types', 'positions', 'bond_types', 'edge_index'
        """
        denoiser.eval()
        T = self.timesteps

        # Keep scheduler on CPU for discrete operations (avoids device issues)
        # Continuous schedule uses scalar alpha values, no GPU needed

        # Choose DDIM timesteps (uniformly spaced, descending)
        step_indices = torch.linspace(T - 1, 0, ddim_steps, device=device).long()
        # Include the final t=0 step
        if step_indices[-1] != 0:
            step_indices = torch.cat([step_indices, torch.tensor([0], device=device)])

        # Pre-compute alpha_bar for each DDIM step
        alpha_schedule = self.scheduler.continuous_schedule  # PredefinedNoiseSchedule
        alpha_bar_all = alpha_schedule.alphas_cumprod  # (timesteps,)

        # Initialize from prior
        x = torch.randn(num_atoms, 3, device=device)
        a_logits = atom_marginals.to(device).log().unsqueeze(0).expand(num_atoms, -1)
        a = F.gumbel_softmax(a_logits, tau=1.0, hard=True)

        edge_index = build_fully_connected_edges(x)
        num_edges = edge_index.size(1)
        e_logits = bond_marginals.to(device).log().unsqueeze(0).expand(num_edges, -1)
        e = F.gumbel_softmax(e_logits, tau=1.0, hard=True) if num_edges > 0 else None

        # DDIM sampling loop
        for i in range(len(step_indices) - 1):
            t = step_indices[i].item()
            t_next = step_indices[i + 1].item()

            t_tensor = torch.tensor([t], device=device)

            # Denoiser forward (this is the expensive step — only ddim_steps calls)
            pred = denoiser(
                atom_types=a,
                positions=x,
                t=t_tensor,
                edge_index=edge_index,
                edge_attr=e,
                condition=condition,
            )

            # ---- DDIM reverse for coordinates ----
            # Predict x_0 from x_t:  x_0 = (x_t - √(1-ᾱ_t)·ε) / √(ᾱ_t)
            # Clamp sqrt(alpha_bar) to avoid division by ~0 at high t (cosine schedule
            # alpha_bar[999] ≈ 0 → 1/sqrt ≈ 20000 → amplifies noise pred error 20000×)
            alpha_bar_t = alpha_bar_all[t]
            sqrt_alpha_t = alpha_bar_t.sqrt().clamp(min=1e-3)
            sqrt_one_minus_alpha_t = (1 - alpha_bar_t).sqrt()

            x0_pred = (x - sqrt_one_minus_alpha_t * pred["coord_noise"]) / sqrt_alpha_t

            # DDIM step to t_next
            alpha_bar_next = alpha_bar_all[t_next]
            sqrt_alpha_next = alpha_bar_next.sqrt()
            sqrt_one_minus_alpha_next = (1 - alpha_bar_next).sqrt()

            if eta > 0:
                # Stochastic DDIM
                sigma = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t) *
                                         (1 - alpha_bar_t / alpha_bar_next))
                noise = torch.randn_like(x)
                x = sqrt_alpha_next * x0_pred + \
                    torch.sqrt(1 - alpha_bar_next - sigma**2) * pred["coord_noise"] + \
                    sigma * noise
            else:
                # Deterministic DDIM (eta=0)
                x = sqrt_alpha_next * x0_pred + \
                    sqrt_one_minus_alpha_next * pred["coord_noise"]

            # ---- Discrete types: denoise on same device as model ----
            # Run full discrete reverse steps between t and t_next
            with torch.no_grad():
                for s in reversed(range(t_next + 1, t + 1)):
                    s_tensor = torch.tensor([s], device=device)
                    a = self.scheduler.reverse_discrete(
                        a, pred["atom_logits"], s_tensor,
                        atom_marginals, "atom"
                    )
                    if e is not None and num_edges > 0:
                        e = self.scheduler.reverse_discrete(
                            e, pred["bond_logits"], s_tensor,
                            bond_marginals, "bond"
                        )

            # Rebuild edges when coordinates have changed significantly
            edge_index = build_fully_connected_edges(x)
            num_edges = edge_index.size(1)
            if num_edges > 0 and (e is None or e.size(0) != num_edges):
                e_logits_new = bond_marginals.to(device).log().unsqueeze(0).expand(num_edges, -1)
                e = F.gumbel_softmax(e_logits_new, tau=1.0, hard=True)

        denoiser.train()
        return {
            "atom_types": a,
            "positions": x,
            "bond_types": e,
            "edge_index": edge_index,
        }

    @torch.no_grad()
    def sample_coord_only(
        self,
        denoiser,
        atom_types: Tensor,        # (N, 12) fixed atom types (one-hot)
        init_positions: Tensor,    # (N, 3) initial coordinates
        condition: Optional[Tensor] = None,
        noise_level: float = 0.5,  # [0, 1]: 0=original, 1=fully random
        num_denoise_steps: int = 10,
        device: str = "cuda",
    ) -> Dict[str, Tensor]:
        """
        Coordinate-only diffusion: perturb coordinates, denoise, return new
        conformation. Atom types remain fixed throughout.

        This is much easier than full generation because:
        1. Atom types come from a valid Core template (guaranteed correct)
        2. Only 3D geometry is varied (continuous, well-behaved)
        3. Each denoising step improves the structure

        Args:
            denoiser: trained denoising network
            atom_types: (N, 12) fixed atom type one-hot
            init_positions: (N, 3) starting coordinates (from template Core)
            condition: FiLM condition tensor
            noise_level: 0.0-1.0, how much noise to add (0=none, 1=full random)
            num_denoise_steps: number of denoising iterations
            device: target device

        Returns:
            dict with 'atom_types', 'positions', 'edge_index'
        """
        denoiser.eval()
        T = self.timesteps
        alpha_sched = self.scheduler.continuous_schedule
        n_atoms = atom_types.size(0)

        # Move to device
        x0 = init_positions.to(device).float()
        atoms = atom_types.to(device).float()
        if condition is not None:
            condition = condition.to(device)

        # Map noise_level to timestep
        t_start = int(noise_level * (T - 1))
        t_start = max(1, min(t_start, T - 1))

        # Forward: add noise at level t_start
        ab_t = alpha_sched.alphas_cumprod[t_start].to(device)
        noise = torch.randn_like(x0)
        xt = ab_t.sqrt() * x0 + (1 - ab_t).sqrt() * noise

        # Build edges + dummy edge_attr (required by EGNN even if unused)
        edge_index = build_fully_connected_edges(xt)
        num_edges = edge_index.size(1)
        edge_attr_dummy = torch.zeros(num_edges, 5, device=device)  # 5 bond types

        # Denoising steps: t_start → 0
        step_size = max(1, t_start // num_denoise_steps)
        for t_val in reversed(range(0, t_start, step_size)):
            if t_val == 0:
                t_val = 1  # avoid t=0
            t_tensor = torch.tensor([t_val], device=device)

            # Predict noise
            pred = denoiser(
                atom_types=atoms,
                positions=xt,
                t=t_tensor,
                edge_index=edge_index,
                edge_attr=edge_attr_dummy,
                condition=condition,
            )

            # DDIM-like single-step recovery
            ab = alpha_sched.alphas_cumprod[t_val].to(device)
            x0_pred = (xt - (1 - ab).sqrt() * pred["coord_noise"]) / ab.sqrt()

            # Move toward x0_pred (partial update)
            next_t = max(0, t_val - step_size)
            ab_next = alpha_sched.alphas_cumprod[next_t].to(device)
            xt = ab_next.sqrt() * x0_pred + (1 - ab_next).sqrt() * pred["coord_noise"]

        # Final denoising at t≈0
        pred_final = denoiser(
            atom_types=atoms, positions=xt,
            t=torch.tensor([1], device=device),
            edge_index=edge_index, edge_attr=edge_attr_dummy,
            condition=condition,
        )
        ab_1 = alpha_sched.alphas_cumprod[1].to(device)
        x_final = (xt - (1 - ab_1).sqrt() * pred_final["coord_noise"]) / ab_1.sqrt()

        # Clamp to reasonable range
        x_final = x_final.clamp(-20.0, 20.0)

        denoiser.train()
        return {
            "atom_types": atoms,
            "positions": x_final,
            "edge_index": edge_index,
        }
