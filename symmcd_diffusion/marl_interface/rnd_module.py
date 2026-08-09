"""
Random Network Distillation (RND) for COF Design Exploration.

Parallel intrinsic reward module:
  r_intrinsic = ||f_target(s) - f_predictor(s)||²

  f_target: fixed randomly-initialized network
  f_predictor: trained to predict f_target(s)

  Novel states → large prediction error → high intrinsic reward
  Familiar states → small prediction error → low intrinsic reward

Total reward = r_external + beta * r_intrinsic

Reference: Burda et al. (2019) "Exploration by Random Network Distillation"
"""

import torch
import torch.nn as nn
import numpy as np


class RNDModule:
    """RND exploration bonus module. Trains in parallel with PPO."""

    def __init__(self, state_dim, hidden_dim=128, lr=1e-4, beta=0.5):
        # Target network: fixed random, never trained
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Freeze target
        for p in self.target.parameters():
            p.requires_grad = False

        # Predictor network: trained to match target
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.beta = beta  # intrinsic reward weight

        # Running stats for normalization
        self.running_mean = 0.0
        self.running_std = 1.0
        self.alpha = 0.01  # EMA coefficient

    def compute_intrinsic_reward(self, state: np.ndarray) -> float:
        """Compute novelty-based intrinsic reward for a state."""
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            target_feat = self.target(s)
            pred_feat = self.predictor(s)
            error = ((target_feat - pred_feat) ** 2).mean().item()

        # Normalize by running stats
        self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * error
        self.running_std = (1 - self.alpha) * self.running_std + self.alpha * abs(
            error - self.running_mean
        )

        normalized = (error - self.running_mean) / max(self.running_std, 1e-8)
        return max(0.0, self.beta * normalized)  # clip to non-negative

    def update(self, states: list):
        """Train predictor on visited states (MSE towards target)."""
        if len(states) < 4:
            return 0.0

        s = torch.FloatTensor(np.array(states))
        with torch.no_grad():
            target_feat = self.target(s)
        pred_feat = self.predictor(s)

        loss = ((target_feat - pred_feat) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def state_dict(self):
        return {
            "predictor": self.predictor.state_dict(),
            "target": self.target.state_dict(),
            "running_mean": self.running_mean,
            "running_std": self.running_std,
        }

    def load_state_dict(self, d):
        self.predictor.load_state_dict(d["predictor"])
        self.target.load_state_dict(d["target"])
        self.running_mean = d.get("running_mean", 0.0)
        self.running_std = d.get("running_std", 1.0)
