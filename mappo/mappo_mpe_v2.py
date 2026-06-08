"""
MAPPO v2 — Multi-Agent PPO with MultiDiscrete action spaces.

Extended from mappo_mpe.py to support:
- Heterogeneous action spaces per agent (MultiDiscrete)
- Multi-headed actor networks
- Compatible with the 3-agent COFDesignEnvV2
"""

import copy
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam


class Actor_MultiDiscrete(nn.Module):
    """
    Multi-headed actor for MultiDiscrete action spaces.

    Each action dimension has its own output head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dims: List[int],
        use_orthogonal: bool = True,
        use_relu: bool = False,
    ):
        super().__init__()
        self.action_dims = action_dims

        if use_orthogonal:
            init_method = nn.init.orthogonal_
        else:
            init_method = nn.init.xavier_uniform_

        gain = nn.init.calculate_gain("relu" if use_relu else "tanh")

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        init_method(self.fc1.weight, gain=gain)
        init_method(self.fc2.weight, gain=gain)

        # Multi-head output: one head per action dimension
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in action_dims
        ])
        for head in self.heads:
            init_method(head.weight, gain=0.01)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        # Return logits for each action dimension
        return [head(x) for head in self.heads]

    def get_distribution(self, x, available_actions=None):
        """Get MultiCategorical distribution."""
        logits_list = self.forward(x)

        if available_actions is not None:
            # Apply action masking: set unavailable actions to -inf
            offset = 0
            masked_logits = []
            for i, logits in enumerate(logits_list):
                dim = self.action_dims[i]
                mask = available_actions[:, offset:offset+dim]
                logits_masked = logits - (1 - mask) * 1e9
                masked_logits.append(logits_masked)
                offset += dim
            logits_list = masked_logits

        # Create independent Categorical distributions
        dists = [Categorical(logits=logits) for logits in logits_list]
        return dists

    def evaluate_actions(self, x, actions, available_actions=None):
        """
        Evaluate log probabilities and entropy of given actions.

        Args:
            x: observations
            actions: list of tensors, one per action dimension
            available_actions: action masks

        Returns:
            (action_log_probs, dist_entropy)
        """
        dists = self.get_distribution(x, available_actions)

        log_probs = []
        for i, dist in enumerate(dists):
            log_probs.append(dist.log_prob(actions[:, i]))

        action_log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)
        dist_entropy = torch.stack(
            [d.entropy() for d in dists], dim=-1
        ).sum(dim=-1)

        return action_log_probs, dist_entropy

    def sample(self, x, available_actions=None):
        """Sample actions from the policy."""
        dists = self.get_distribution(x, available_actions)
        actions = torch.stack([d.sample() for d in dists], dim=-1)
        return actions


class Critic_MLP(nn.Module):
    """Centralized critic (value function)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_orthogonal: bool = True,
        use_relu: bool = False,
    ):
        super().__init__()

        gain = nn.init.calculate_gain("relu" if use_relu else "tanh")
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        init_method(self.fc1.weight, gain=gain)
        init_method(self.fc2.weight, gain=gain)
        init_method(self.fc3.weight, gain=1.0)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.fc3(x)


class MAPPO_MPE_V2:
    """
    MAPPO for MultiDiscrete action spaces.

    Simplified version of mappo_mpe.MAPPO_MPE adapted for:
    - Heterogeneous per-agent action dimensions
    - Multi-headed actor outputs
    - 3-agent COF design environment
    """

    def __init__(self, args, action_dims_list, device="cuda"):
        self.n_agents = args.n_agents if hasattr(args, 'n_agents') else 3
        self.action_dims_list = action_dims_list  # List of [dims] per agent

        obs_dim = 128           # Observation dimension
        state_dim = 128 * self.n_agents
        hidden_dim = getattr(args, 'hidden_dim', 128)
        lr = getattr(args, 'lr', 5e-4)
        self.gamma = getattr(args, 'gamma', 0.99)
        self.gae_lambda = getattr(args, 'gae_lambda', 0.95)
        self.clip_eps = getattr(args, 'clip_eps', 0.2)
        self.k_epochs = getattr(args, 'k_epochs', 15)

        self.device = device

        # Create actor and critic for each agent
        self.actors = nn.ModuleList([
            Actor_MultiDiscrete(obs_dim, hidden_dim, dims)
            for dims in action_dims_list
        ])

        # Shared critic (centralized)
        self.critic = Critic_MLP(state_dim, hidden_dim)

        # Optimizers
        self.actor_optimizers = [
            Adam(actor.parameters(), lr=lr, eps=1e-5)
            for actor in self.actors
        ]
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr, eps=1e-5)

        self.to(device)

    def to(self, device):
        self.actors.to(device)
        self.critic.to(device)

    def choose_action(self, obs, mask=None):
        """
        Choose actions for all agents.

        Args:
            obs: (n_agents, obs_dim) observations
            mask: optional action masks

        Returns:
            actions: list of action arrays per agent
        """
        actions = []
        obs_tensor = torch.FloatTensor(obs).to(self.device)

        for i, actor in enumerate(self.actors):
            agent_obs = obs_tensor[i].unsqueeze(0)  # (1, obs_dim)

            if mask is not None and mask[i] is not None:
                mask_tensor = torch.FloatTensor(mask[i]).unsqueeze(0).to(self.device)
            else:
                mask_tensor = None

            action = actor.sample(agent_obs, mask_tensor)
            actions.append(action.squeeze(0).cpu().numpy())

        return actions

    def get_values(self, state):
        """Get state values from critic."""
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        return self.critic(state_tensor).squeeze(-1)

    def train(self, replay_buffer):
        """
        PPO training step.

        Args:
            replay_buffer: buffer with collected trajectories
        """
        # Get training data
        data = replay_buffer.get_training_data()
        if data is None:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        obs, states, values, actions, action_log_probs, rewards, dones, masks = data

        # Convert to tensors
        obs = torch.FloatTensor(obs).to(self.device)
        states = torch.FloatTensor(states).to(self.device)
        old_values = torch.FloatTensor(values).to(self.device)
        old_log_probs = torch.FloatTensor(action_log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute advantages (GAE)
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(rewards.size(1))):
            if t == rewards.size(1) - 1:
                next_value = 0.0
            else:
                next_value = old_values[0, t + 1]
            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - old_values[:, t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[:, t]) * last_gae
            advantages[:, t] = last_gae

        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(self.k_epochs):
            # Critic update
            current_values = self.get_values(states.squeeze(0))
            critic_loss = F.mse_loss(current_values, returns.squeeze(0))

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_optimizer.step()
            total_critic_loss += critic_loss.item()

            # Actor update (per agent)
            for i, (actor, optimizer) in enumerate(
                zip(self.actors, self.actor_optimizers)
            ):
                agent_obs = obs[0, :, i, :]
                agent_actions = actions[0, :, i, :]

                new_log_probs, entropy = actor.evaluate_actions(
                    agent_obs, agent_actions
                )

                ratio = torch.exp(new_log_probs - old_log_probs[0, :, i])
                advantage = advantages[0, :, i]

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                optimizer.step()
                total_actor_loss += actor_loss.item()

        return {
            "actor_loss": total_actor_loss / self.k_epochs / self.n_agents,
            "critic_loss": total_critic_loss / self.k_epochs,
        }

    def save(self, path, episode):
        """Save model checkpoints."""
        os.makedirs(path, exist_ok=True)
        for i, actor in enumerate(self.actors):
            torch.save(
                actor.state_dict(),
                os.path.join(path, f"actor_{i}_ep{episode}.pth"),
            )
        torch.save(
            self.critic.state_dict(),
            os.path.join(path, f"critic_ep{episode}.pth"),
        )

    def load(self, path, episode):
        """Load model checkpoints."""
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(
                torch.load(os.path.join(path, f"actor_{i}_ep{episode}.pth"))
            )
        self.critic.load_state_dict(
            torch.load(os.path.join(path, f"critic_ep{episode}.pth"))
        )
