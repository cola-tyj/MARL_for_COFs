#!/usr/bin/env python
"""
Phase 4: PPO Training for COF Design.

2-Agent design: Agent selects (sym_a, template_a) and (sym_b, template_b).
Topology is AUTO-DETERMINED from the symmetry pair's connector counts,
eliminating invalid topology-symmetry mismatches.

For training efficiency, validated (sym_a, core_a, sym_b, core_b, topology)
combos are pre-built as discrete actions. This is equivalent to the 2-Agent
design but simpler for PPO implementation.

Usage:
    python train_mappo_cof.py --episodes 1000 --lr 1e-4
"""

import os, sys, json, time, argparse, io
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.distributions import Categorical

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from env_v2 import COFDesignEnv, SYMMETRY_TYPES


# ═══════════════════════════════════════════════════════════════════════════════
# PPO Agent
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyNetwork(nn.Module):
    """Simple MLP policy for discrete action space."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.net(state)


class PPOAgent:
    """Single-agent PPO."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        hidden_dim: int = 128,
    ):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = AdamW(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.action_dim = action_dim

        # Training buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Sample action from policy."""
        s = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy(s)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def store(self, state, action, reward, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def update(self):
        """PPO update on stored transitions."""
        if len(self.rewards) < 2:
            self.clear_buffer()
            return {}

        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()

        # Compute returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # PPO loss
        logits = self.policy(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        ratios = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratios * returns
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * returns
        loss = -torch.min(surr1, surr2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        stats = {"loss": loss.item(), "mean_return": returns.mean().item()}
        self.clear_buffer()
        return stats

    def clear_buffer(self):
        self.states, self.actions, self.rewards, self.log_probs = [], [], [], []


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def _build_action_map(env: COFDesignEnv) -> List[Tuple[int, int]]:
    """Build flat action map for 2-Agent design from validated combos."""
    actions = []
    for sym_a in SYMMETRY_TYPES:
        for sym_b in SYMMETRY_TYPES:
            # Skip incompatible pairs
            from env_v2 import SYMMETRY_TO_CONNECTORS, SYMMETRY_PAIR_TO_TOPOLOGY
            ca = SYMMETRY_TO_CONNECTORS.get(sym_a, 2)
            cb = SYMMETRY_TO_CONNECTORS.get(sym_b, 2)
            if (ca, cb) not in SYMMETRY_PAIR_TO_TOPOLOGY:
                continue
            na = min(5, len(env.cores.get(sym_a, [])))
            nb = min(8, len(env.cores.get(sym_b, [])))
            for ta in range(na):
                for tb in range(nb):
                    actions.append(((SYMMETRY_TYPES.index(sym_a), ta),
                                    (SYMMETRY_TYPES.index(sym_b), tb)))
    return actions


def train(args):
    import logging
    logging.getLogger("pycofbuilder").setLevel(logging.ERROR)

    n2_path = "symmcd_diffusion/generated/n2_lookup.json" if args.real_reward else None
    env = COFDesignEnv(n2_lookup_path=n2_path)

    # Build validated action map (topology auto-determined)
    action_map = _build_action_map(env)
    n_actions = len(action_map)

    agent = PPOAgent(
        state_dim=8,
        action_dim=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        hidden_dim=args.hidden_dim,
    )

    print(f"=== Phase 4: PPO COF Design (2-Agent, topology auto) ===")
    print(f"Action space: {n_actions} validated designs")
    print(f"PPO: lr={args.lr}, gamma={args.gamma}, eps_clip={args.eps_clip}")
    print(f"Episodes: {args.episodes}, update every: {args.update_every}")
    print()

    episode_rewards = []
    best_reward = -float("inf")
    best_design = None
    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        s1, s2 = env.reset()
        # Single-agent policy selects from flat action map
        action, log_prob = agent.select_action(s1)
        a1, a2 = action_map[action]
        (ns1, ns2), reward, done, info = env.step(a1, a2)
        agent.store(s1, action, reward, log_prob)

        episode_rewards.append(reward)

        if ep % args.update_every == 0:
            stats = agent.update()
            if stats:
                avg_r = np.mean(episode_rewards[-args.update_every:])
                success_rate = env.assembly_success / max(env.assembly_total, 1)
                elapsed = time.time() - start_time
                print(
                    f"Ep {ep:5d}/{args.episodes} | "
                    f"AvgR: {avg_r:6.1f} | "
                    f"Loss: {stats['loss']:6.3f} | "
                    f"Success: {success_rate:.0%} | "
                    f"Time: {elapsed:.0f}s"
                )

        if reward > best_reward:
            best_reward = reward
            best_design = info.get("sym_a", "?") + "+" + info.get("sym_b", "?") + "→" + info.get("topology", "?")

        if ep % 100 == 0:
            recent = episode_rewards[-100:]
            print(f"  [Ep {ep}] mean={np.mean(recent):.1f} max={np.max(recent):.1f} "
                  f"success={env.assembly_success}/{env.assembly_total}")

    elapsed = time.time() - start_time
    success_rate = env.assembly_success / max(env.assembly_total, 1)
    print(f"\n=== Training Complete ===")
    print(f"Episodes: {args.episodes} | Time: {elapsed:.0f}s")
    print(f"Success: {env.assembly_success}/{env.assembly_total} ({success_rate:.0%})")
    print(f"Best: {best_reward:.1f} ({best_design})")
    print(f"Mean reward: {np.mean(episode_rewards):.1f}")

    out = Path(__file__).parent / "checkpoints" / "mappo_cof_policy.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({
        "policy_state_dict": agent.policy.state_dict(),
        "action_dim": n_actions,
        "best_reward": best_reward,
        "best_design": best_design,
        "success_rate": success_rate,
    }, str(out))
    print(f"Policy saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAPPO COF Design Training")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--update_every", type=int, default=10)
    parser.add_argument("--real_reward", action="store_true", default=False,
                        help="Use real N₂ adsorption values (requires n2_lookup.json)")
    args = parser.parse_args()
    train(args)
