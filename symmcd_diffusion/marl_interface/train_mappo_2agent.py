#!/usr/bin/env python
"""
Phase 4: True 2-Agent MAPPO for COF Design.

Agent 1: Core-A (symmetry + template) → action = (sym_idx, template_idx)
Agent 2: Core-B (symmetry + template) → action = (sym_idx, template_idx)
Topology: auto-determined from (conn_a, conn_b)

Features:
- Independent policy networks for each agent
- Centralized critic (value network takes joint state)
- Covers ALL valid symmetry pairs: (2,3), (3,2), (2,4), (4,2), (2,2), (2,6), (6,2)
"""

import os, sys, json, time, argparse, io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent))
from env_v2 import (
    COFDesignEnv, SYMMETRY_TYPES, TOPOLOGIES,
    SYMMETRY_TO_CONNECTORS, SYMMETRY_PAIR_TO_TOPOLOGY,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 2-Agent MAPPO
# ═══════════════════════════════════════════════════════════════════════════════

class ActorNetwork(nn.Module):
    """Policy network for a single agent: sym + template + topology."""
    def __init__(self, state_dim, n_sym, n_templates, max_topos=3, hidden=128):
        super().__init__()
        self.n_sym = n_sym
        self.n_templates = n_templates
        self.max_topos = max_topos
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.sym_head = nn.Linear(hidden, n_sym)
        self.template_head = nn.Linear(hidden, n_templates)
        self.topo_head = nn.Linear(hidden, max_topos)

    def forward(self, state):
        h = self.shared(state)
        return self.sym_head(h), self.template_head(h), self.topo_head(h)

    def get_action(self, state):
        sym_logits, tmpl_logits, topo_logits = self.forward(state)
        sym_dist = Categorical(logits=sym_logits)
        tmpl_dist = Categorical(logits=tmpl_logits)
        topo_dist = Categorical(logits=topo_logits)
        sym_a = sym_dist.sample(); tmpl_a = tmpl_dist.sample(); topo_a = topo_dist.sample()
        log_prob = sym_dist.log_prob(sym_a) + tmpl_dist.log_prob(tmpl_a) + topo_dist.log_prob(topo_a)
        return (sym_a.item(), tmpl_a.item()), topo_a.item(), log_prob

    def evaluate(self, state, action_sym_tmpl, action_topo):
        sym_logits, tmpl_logits, topo_logits = self.forward(state)
        sym_dist = Categorical(logits=sym_logits)
        tmpl_dist = Categorical(logits=tmpl_logits)
        topo_dist = Categorical(logits=topo_logits)
        sym_a, tmpl_a = action_sym_tmpl[:, 0], action_sym_tmpl[:, 1]
        log_prob = (sym_dist.log_prob(sym_a) + tmpl_dist.log_prob(tmpl_a) +
                    topo_dist.log_prob(action_topo))
        entropy = (sym_dist.entropy().mean() + tmpl_dist.entropy().mean() +
                   topo_dist.entropy().mean())
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """Centralized value network (takes joint state)."""
    def __init__(self, state_dim_a, state_dim_b, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim_a + state_dim_b, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s_a, s_b):
        return self.net(torch.cat([s_a, s_b], dim=-1))


class MAPPOAgent:
    """2-Agent MAPPO with centralized critic."""

    def __init__(self, state_dim, n_sym, n_tmpl_a, n_tmpl_b,
                 lr=1e-4, gamma=0.99, eps_clip=0.2):
        self.actor1 = ActorNetwork(state_dim, n_sym, n_tmpl_a)
        self.actor2 = ActorNetwork(state_dim, n_sym, n_tmpl_b)
        self.critic = CriticNetwork(state_dim, state_dim)

        self.optimizer = AdamW(
            list(self.actor1.parameters()) +
            list(self.actor2.parameters()) +
            list(self.critic.parameters()),
            lr=lr,
        )
        self.gamma = gamma
        self.eps_clip = eps_clip

        # Buffers
        self.states_a, self.states_b = [], []
        self.actions_a, self.actions_b = [], []
        self.topos_a, self.topos_b = [], []  # topology selections
        self.rewards = []
        self.log_probs_a, self.log_probs_b = [], []
        self.values = []

    def select_actions(self, s_a, s_b):
        s_a_t = torch.FloatTensor(s_a).unsqueeze(0)
        s_b_t = torch.FloatTensor(s_b).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(s_a_t, s_b_t)
        action_a, topo_a, lp_a = self.actor1.get_action(s_a_t)
        action_b, topo_b, lp_b = self.actor2.get_action(s_b_t)
        return action_a, action_b, topo_a, topo_b, lp_a, lp_b, value.item()

    def store(self, s_a, s_b, a_a, a_b, t_a, t_b, r, lp_a, lp_b, v):
        self.states_a.append(s_a); self.states_b.append(s_b)
        self.actions_a.append(a_a); self.actions_b.append(a_b)
        self.topos_a.append(t_a); self.topos_b.append(t_b)
        self.rewards.append(r)
        self.log_probs_a.append(lp_a); self.log_probs_b.append(lp_b)
        self.values.append(v)

    def update(self):
        n = len(self.rewards)
        if n < 2:
            self.clear_buffer()
            return {}

        # Compute returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        values = torch.FloatTensor(self.values)

        # Actor loss for both agents
        s_a = torch.FloatTensor(np.array(self.states_a))
        s_b = torch.FloatTensor(np.array(self.states_b))
        a_a = torch.LongTensor([[x[0], x[1]] for x in self.actions_a])
        a_b = torch.LongTensor([[x[0], x[1]] for x in self.actions_b])
        t_a = torch.LongTensor(self.topos_a)
        t_b = torch.LongTensor(self.topos_b)
        old_lp_a = torch.stack(self.log_probs_a).detach()
        old_lp_b = torch.stack(self.log_probs_b).detach()

        advantages = returns - values.detach()

        # Agent 1 loss (sym + template + topology)
        new_lp_a, ent_a = self.actor1.evaluate(s_a, a_a, t_a)
        ratio_a = torch.exp(new_lp_a - old_lp_a)
        surr1_a = ratio_a * advantages
        surr2_a = torch.clamp(ratio_a, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss_a = -torch.min(surr1_a, surr2_a).mean()

        # Agent 2 loss
        new_lp_b, ent_b = self.actor2.evaluate(s_b, a_b, t_b)
        ratio_b = torch.exp(new_lp_b - old_lp_b)
        surr1_b = ratio_b * advantages
        surr2_b = torch.clamp(ratio_b, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss_b = -torch.min(surr1_b, surr2_b).mean()

        # Critic loss
        critic_loss = F.mse_loss(values, returns)

        # Total
        loss = actor_loss_a + actor_loss_b + 0.5 * critic_loss - 0.01 * (ent_a + ent_b)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor1.parameters()) +
            list(self.actor2.parameters()) +
            list(self.critic.parameters()), 1.0)
        self.optimizer.step()

        stats = {"loss": loss.item(), "actor1": actor_loss_a.item(),
                 "actor2": actor_loss_b.item(), "critic": critic_loss.item()}
        self.clear_buffer()
        return stats

    def clear_buffer(self):
        self.states_a.clear(); self.states_b.clear()
        self.actions_a.clear(); self.actions_b.clear()
        self.topos_a.clear(); self.topos_b.clear()
        self.rewards.clear(); self.log_probs_a.clear()
        self.log_probs_b.clear(); self.values.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    import logging
    logging.getLogger("pycofbuilder").setLevel(logging.ERROR)

    n2_path = "symmcd_diffusion/generated/n2_lookup.json" if args.real_reward else None
    env = COFDesignEnv(n2_lookup_path=n2_path)

    n_tmpl_a = max(len(env.cores.get("T3", [])), len(env.cores.get("S4", [])), len(env.cores.get("H6", [])))
    n_tmpl_b = max(len(env.cores.get("L2", [])), 1)

    agent = MAPPOAgent(
        state_dim=8,
        n_sym=env.n_symmetries,
        n_tmpl_a=min(n_tmpl_a, 10),
        n_tmpl_b=min(n_tmpl_b, 10),
        lr=args.lr, gamma=args.gamma, eps_clip=args.eps_clip,
    )

    # All valid symmetry pairs with topology coverage
    valid_pairs = list(SYMMETRY_PAIR_TO_TOPOLOGY.keys())
    print(f"=== Phase 4: True 2-Agent MAPPO (with topology selection) ===")
    print(f"Agent 1: {agent.actor1.n_sym} sym × {agent.actor1.n_templates} templates × {agent.actor1.max_topos} topologies")
    print(f"Agent 2: {agent.actor2.n_sym} sym × {agent.actor2.n_templates} templates × {agent.actor2.max_topos} topologies")
    print(f"Valid symmetry pairs: {len(valid_pairs)}")
    print(f"Topology coverage ({len(TOPOLOGIES)}): {TOPOLOGIES}")
    print(f"PPO: lr={args.lr}, gamma={args.gamma}, eps={args.eps_clip}")
    print()

    episode_rewards = []
    best_reward = -float("inf")
    best_design = None
    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        s_a, s_b = env.reset()

        # Select actions (sym + template + topology)
        a_a, a_b, t_a, t_b, lp_a, lp_b, value = agent.select_actions(s_a, s_b)

        # Ensure action pairs are valid
        sym_a = SYMMETRY_TYPES[a_a[0]]
        sym_b = SYMMETRY_TYPES[a_b[0]]
        ca = SYMMETRY_TO_CONNECTORS.get(sym_a, 2)
        cb = SYMMETRY_TO_CONNECTORS.get(sym_b, 2)
        if (ca, cb) not in SYMMETRY_PAIR_TO_TOPOLOGY:
            reward = -1.0
            info = {"success": False, "error": "invalid_pair"}
        else:
            # Use Agent 1's topology choice (both agents agree on topology)
            (ns_a, ns_b), reward, done, info = env.step(a_a, a_b, topology_idx=t_a)

        agent.store(s_a, s_b, a_a, a_b, t_a, t_b, reward, lp_a, lp_b, value)
        episode_rewards.append(reward)

        if ep % args.update_every == 0:
            stats = agent.update()
            if stats:
                avg_r = np.mean(episode_rewards[-args.update_every:])
                success_rate = env.assembly_success / max(env.assembly_total, 1)
                elapsed = time.time() - start_time
                print(f"Ep {ep:5d}/{args.episodes} | AvgR: {avg_r:6.1f} | "
                      f"Success: {success_rate:.0%} | "
                      f"A1: {stats['actor1']:5.2f} A2: {stats['actor2']:5.2f} "
                      f"C: {stats['critic']:5.2f} | {elapsed:.0f}s")

        if reward > best_reward:
            best_reward = reward
            best_design = f"{info.get('sym_a','?')}+{info.get('sym_b','?')}→{info.get('topology','?')} (topo_idx={t_a})"

        if ep % 100 == 0:
            recent = episode_rewards[-100:]
            print(f"  [Ep {ep}] mean={np.mean(recent):.1f} "
                  f"success={env.assembly_success}/{env.assembly_total}")

    # Summary
    elapsed = time.time() - start_time
    success_rate = env.assembly_success / max(env.assembly_total, 1)
    print(f"\n=== Training Complete ===")
    print(f"Episodes: {args.episodes} | Time: {elapsed:.0f}s")
    print(f"Success: {env.assembly_success}/{env.assembly_total} ({success_rate:.0%})")
    print(f"Best: {best_reward:.1f} ({best_design})")
    print(f"Mean reward: {np.mean(episode_rewards):.1f}")

    # Save
    out = Path(__file__).parent / "checkpoints"
    out.mkdir(exist_ok=True)
    torch.save({
        "actor1": agent.actor1.state_dict(),
        "actor2": agent.actor2.state_dict(),
        "critic": agent.critic.state_dict(),
        "best_reward": best_reward,
        "best_design": best_design,
        "success_rate": success_rate,
        "valid_pairs": list(SYMMETRY_PAIR_TO_TOPOLOGY.keys()),
    }, str(out / "mappo_2agent.pt"))
    print(f"Saved: {out}/mappo_2agent.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--update_every", type=int, default=20)
    parser.add_argument("--real_reward", action="store_true", default=False)
    args = parser.parse_args()
    train(args)
