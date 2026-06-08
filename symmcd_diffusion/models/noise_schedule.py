"""
Noise schedules for the diffusion process.

Supports:
- Continuous diffusion (Gaussian noise) for 3D coordinates
- Discrete diffusion (categorical noise) for atom types and bond types
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PredefinedNoiseSchedule(nn.Module):
    """
    Predefined noise schedule for diffusion models.

    For continuous variables (coordinates): cosine schedule (Nichol & Dhariwal, 2021)
    For discrete variables (atom types, bonds): uniform transition matrix
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        precision: float = 1e-5,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.schedule = schedule

        if schedule == "cosine":
            # Cosine schedule as in iDDPM
            betas = self._cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (not parameters, but part of module state)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # For continuous diffusion
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1),
        )

        # Posterior variance
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
        """Cosine beta schedule from iDDPM."""
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos((t / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(betas, max=0.999)

    def get_continuous_noise_params(self, t: Tensor) -> dict:
        """Get noise parameters for continuous (Gaussian) diffusion at timestep t."""
        return {
            "sqrt_alphas_cumprod": self.sqrt_alphas_cumprod[t],
            "sqrt_one_minus_alphas_cumprod": self.sqrt_one_minus_alphas_cumprod[t],
            "sqrt_recip_alphas_cumprod": self.sqrt_recip_alphas_cumprod[t],
            "sqrt_recipm1_alphas_cumprod": self.sqrt_recipm1_alphas_cumprod[t],
            "posterior_variance": self.posterior_variance[t],
        }


class DiscreteTransitionMatrix(nn.Module):
    """
    Transition matrix for discrete diffusion (atom types, bond types).

    Q_t = alpha_t * I + beta_t * m  where m is the marginal distribution.
    The stationary distribution converges to the data marginals.

    Reference: Vignac et al. (2023), DiGress
    """

    def __init__(
        self,
        num_classes: int,
        timesteps: int = 1000,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.timesteps = timesteps

        # Schedule for the mixing parameter
        if schedule == "cosine":
            self.register_buffer(
                "alpha_bar",
                torch.cos(
                    torch.linspace(0, timesteps - 1, timesteps) / timesteps * math.pi * 0.5
                ) ** 2,
            )
        else:
            self.register_buffer(
                "alpha_bar",
                1.0 - torch.linspace(0, timesteps - 1, timesteps) / timesteps * 0.999,
            )

    def get_transition_matrix(self, t: Tensor, marginals: Tensor) -> Tensor:
        """
        Compute the cumulative transition matrix Q_bar_t.

        Args:
            t: (batch,) timestep indices
            marginals: (num_classes,) empirical marginal distribution

        Returns:
            Q_bar: (batch, num_classes, num_classes) transition matrices
        """
        alpha_t = self.alpha_bar[t]  # (batch,)
        batch_size = t.size(0)

        # Q_bar_t = alpha_t * I + (1 - alpha_t) * 1 * m^T
        I = torch.eye(self.num_classes, device=t.device).unsqueeze(0)
        ones = torch.ones(self.num_classes, 1, device=t.device)

        Q_bar = alpha_t.view(-1, 1, 1) * I + \
                (1 - alpha_t).view(-1, 1, 1) * \
                (ones @ marginals.unsqueeze(0))

        return Q_bar


class MixedNoiseScheduler(nn.Module):
    """
    Combined noise scheduler for mixed continuous-discrete diffusion.

    Handles:
    - Continuous noise for 3D coordinates (Gaussian)
    - Discrete noise for atom types (categorical)
    - Discrete noise for bond types (categorical)
    """

    def __init__(
        self,
        timesteps: int = 1000,
        num_atom_types: int = 5,
        num_bond_types: int = 5,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.timesteps = timesteps

        # Continuous schedule for coordinates
        self.continuous_schedule = PredefinedNoiseSchedule(timesteps, schedule)

        # Discrete schedules for atom and bond types
        self.atom_transition = DiscreteTransitionMatrix(num_atom_types, timesteps, schedule)
        self.bond_transition = DiscreteTransitionMatrix(num_bond_types, timesteps, schedule)

    def forward_continuous(
        self, x0: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply continuous (Gaussian) forward diffusion to coordinates.

        Args:
            x0: (N, 3) clean coordinates
            t: (batch,) timestep indices

        Returns:
            xt: (N, 3) noised coordinates
            noise: (N, 3) the noise added
        """
        params = self.continuous_schedule.get_continuous_noise_params(t)
        sqrt_ac = params["sqrt_alphas_cumprod"]
        sqrt_1m_ac = params["sqrt_one_minus_alphas_cumprod"]

        # Expand to N nodes
        if sqrt_ac.dim() == 1 and x0.dim() == 2:
            sqrt_ac = sqrt_ac.unsqueeze(1)
            sqrt_1m_ac = sqrt_1m_ac.unsqueeze(1)

        if noise is None:
            noise = torch.randn_like(x0)
        xt = sqrt_ac * x0 + sqrt_1m_ac * noise
        return xt, noise

    def forward_discrete(
        self, a0: Tensor, t: Tensor, marginals: Tensor, transition_type: str = "atom"
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply discrete (categorical) forward diffusion.

        Args:
            a0: (N, num_classes) one-hot encoding of clean categories
            t: (batch,) timestep indices
            marginals: (num_classes,) empirical marginal distribution
            transition_type: "atom" or "bond"

        Returns:
            at: (N, num_classes) noised category probabilities
            a0_onehot: (N, num_classes) original one-hot (for loss computation)
        """
        if transition_type == "atom":
            Q_bar = self.atom_transition.get_transition_matrix(t, marginals)
        else:
            Q_bar = self.bond_transition.get_transition_matrix(t, marginals)

        # at = a0 @ Q_bar_t  (but Q_bar is batch x K x K)
        # Expand for batched matrix multiply
        a0_prob = a0.float()
        at = torch.bmm(
            a0_prob.unsqueeze(1),  # (N, 1, K)
            Q_bar,                   # (batch, K, K) — needs to match N
        ).squeeze(1)

        # Sample from categorical
        at_sampled = F.gumbel_softmax(at.log(), tau=1.0, hard=True)

        return at_sampled, a0_prob

    def reverse_continuous(
        self, xt: Tensor, predicted_noise: Tensor, t: Tensor
    ) -> Tensor:
        """
        Single reverse step for continuous diffusion (DDPM).

        Args:
            xt: (N, 3) noised coordinates at time t
            predicted_noise: (N, 3) predicted noise from denoiser
            t: (batch,) current timestep

        Returns:
            x_{t-1}: (N, 3) denoised coordinates
        """
        params = self.continuous_schedule.get_continuous_noise_params(t)

        sqrt_recip_ac = params["sqrt_recip_alphas_cumprod"]
        sqrt_recipm1_ac = params["sqrt_recipm1_alphas_cumprod"]
        posterior_var = params["posterior_variance"]

        # Expand dimensions
        if sqrt_recip_ac.dim() == 1 and xt.dim() == 2:
            sqrt_recip_ac = sqrt_recip_ac.unsqueeze(1)
            sqrt_recipm1_ac = sqrt_recipm1_ac.unsqueeze(1)
            posterior_var = posterior_var.unsqueeze(1)

        # Predict x0 from xt and predicted noise
        x0_pred = sqrt_recip_ac * xt - sqrt_recipm1_ac * predicted_noise

        # Compute mean for posterior q(x_{t-1} | xt, x0_pred)
        posterior_mean = x0_pred

        # Add noise for t > 0
        if t.min() > 0:
            noise = torch.randn_like(xt)
            posterior_mean = posterior_mean + torch.sqrt(posterior_var) * noise

        return posterior_mean

    def reverse_discrete(
        self, at: Tensor, denoised_logits: Tensor, t: Tensor,
        transition_type: str = "atom"
    ) -> Tensor:
        """
        Single reverse step for discrete diffusion.

        Uses the predicted denoised distribution to compute
        the posterior q(a_{t-1} | at, a0_pred).

        Args:
            at: (N, num_classes) current categorical (one-hot)
            denoised_logits: (N, num_classes) predicted denoised logits
            t: (batch,) current timestep
            transition_type: "atom" or "bond"

        Returns:
            a_{t-1}: (N, num_classes) one-hot for previous timestep
        """
        # Get the posterior distribution
        a0_probs = F.softmax(denoised_logits, dim=-1)

        if transition_type == "atom":
            num_classes = self.atom_transition.num_classes
        else:
            num_classes = self.bond_transition.num_classes

        if t.min() > 0:
            # Compute posterior: p(a_{t-1} | a_t, a_0)
            # This requires the transition matrices
            # Simplified: use the denoised probabilities directly with temperature
            tau = 0.1
            posterior = a0_probs
            a_prev = F.gumbel_softmax(posterior.log(), tau=tau, hard=True)
        else:
            # At t=0, just argmax the denoised prediction
            a_prev = F.one_hot(
                denoised_logits.argmax(dim=-1),
                num_classes=num_classes,
            ).float()

        return a_prev
