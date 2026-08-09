#!/usr/bin/env python
"""
Phase 4 v6: MAPPO with rich state + sequential coordination.

Key design:
- 3-step sequential episode (Agent 1 → Agent 2 → Agent 3)
- Agent 1/2: action = (sym_idx, core_idx) — 2 discrete heads
- Agent 3: action = topology_idx (masked by symmetry pair)
- Rich 42-dim state with Core features + partner info + topology mask
- RND exploration bonus
- Diversity bonus for under-explored topologies
"""

import os, sys, time, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent))

# Suppress pycofbuilder import noise
import io as _io
_old_stdout = sys.stdout; sys.stdout = _io.StringIO()
from env_v6 import (COFDesignEnvV6, TOPOLOGIES, SYMMETRY_TYPES, N_TOPOS, N_SYMS,
                     SYMMETRY_TO_CONNECTORS, SYMMETRY_PAIR_TO_TOPOLOGY)
from rnd_module import RNDModule
sys.stdout = _old_stdout


# ── Networks ──────────────────────────────────────────────────────────────

class MultiHeadActor(nn.Module):
    """Two-headed actor for Agent 1/2: sym (6) + core (variable templates)."""

    def __init__(self, s_dim=42, n_sym=6, max_cores=20, h=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(s_dim, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(),
        )
        self.sym_head = nn.Linear(h, n_sym)
        self.core_head = nn.Linear(h, max_cores)
        self.n_sym = n_sym
        self.max_cores = max_cores

    def forward(self, s):
        feat = self.shared(s)
        return self.sym_head(feat), self.core_head(feat)

    def get_action(self, s, core_mask=None):
        """
        Args:
            s: state tensor (1, state_dim)
            core_mask: binary mask for valid core indices (1, max_cores)
        Returns: (sym_idx, core_idx), (log_prob_sym, log_prob_core)
        """
        sym_logits, core_logits = self.forward(s)

        # Mask invalid cores
        if core_mask is not None:
            core_logits = core_logits + (1 - core_mask) * -1e9

        sym_dist = Categorical(logits=sym_logits)
        core_dist = Categorical(logits=core_logits)

        sym_a = sym_dist.sample()
        core_a = core_dist.sample()

        return (sym_a.item(), core_a.item()), (sym_dist.log_prob(sym_a),
                                                core_dist.log_prob(core_a))

    def evaluate(self, s, sym_a, core_a, core_mask=None):
        sym_logits, core_logits = self.forward(s)
        if core_mask is not None:
            core_logits = core_logits + (1 - core_mask) * -1e9
        sym_dist = Categorical(logits=sym_logits)
        core_dist = Categorical(logits=core_logits)
        lp_sym = sym_dist.log_prob(sym_a)
        lp_core = core_dist.log_prob(core_a)
        ent = (sym_dist.entropy().mean() + core_dist.entropy().mean()) / 2.0
        return lp_sym + lp_core, ent


class TopoActor(nn.Module):
    """Single-headed actor for Agent 3: topology selection (masked)."""

    def __init__(self, s_dim=42, n_topo=N_TOPOS, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, n_topo),
        )
        self.n_topo = n_topo

    def forward(self, s):
        return self.net(s)

    def get_action(self, s, topo_mask=None):
        logits = self.forward(s)
        if topo_mask is not None:
            logits = logits + (1 - topo_mask) * -1e9
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a.item(), dist.log_prob(a)

    def evaluate(self, s, a, topo_mask=None):
        logits = self.forward(s)
        if topo_mask is not None:
            logits = logits + (1 - topo_mask) * -1e9
        dist = Categorical(logits=logits)
        return dist.log_prob(a), dist.entropy().mean()


class CentralizedCritic(nn.Module):
    """Critic sees all 3 agent states concatenated."""

    def __init__(self, s_dim=42, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim * 3, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, s1, s2, s3):
        return self.net(torch.cat([s1, s2, s3], dim=-1))


# ── Core mask builder ─────────────────────────────────────────────────────

def build_core_masks(env, device='cpu'):
    """Build fixed core validity masks per symmetry (max template count)."""
    max_cores = max(len(env.core_names.get(s, [])) for s in SYMMETRY_TYPES)
    masks = {}
    for si, sym in enumerate(SYMMETRY_TYPES):
        n_cores = len(env.core_names.get(sym, []))
        mask = torch.zeros(max_cores, device=device)
        mask[:n_cores] = 1.0
        masks[si] = mask
    return masks, max_cores


# ── Topology mask from state ──────────────────────────────────────────────

def extract_topo_mask(state, topo_start=19):
    """Extract topology validity mask from state vector."""
    return state[..., topo_start:topo_start + N_TOPOS]


# ── PPO update ────────────────────────────────────────────────────────────

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Compute GAE returns for a 3-step episode."""
    advantages = []
    gae = 0.0
    values_np = values.detach().cpu().numpy()
    rewards_np = np.array(rewards)

    for t in reversed(range(len(rewards))):
        next_val = values_np[t + 1] if t + 1 < len(values_np) else 0.0
        delta = rewards_np[t] + gamma * next_val - values_np[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values_np)]
    return torch.FloatTensor(returns), torch.FloatTensor(advantages)


# ── Main training loop ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--steps_per_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.02)
    parser.add_argument('--vf_coef', type=float, default=1.0)
    parser.add_argument('--rnd_beta', type=float, default=0.3)
    parser.add_argument('--ppo_epochs', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── Env ────────────────────────────────────────────────────────────
    env = COFDesignEnvV6()
    core_masks, max_cores = build_core_masks(env, device)
    s_dim = env.state_dim

    print(f"=== MAPPO v6: Rich State + Sequential ===", flush=True)
    print(f"State dim: {s_dim}, Max cores: {max_cores}", flush=True)
    print(f"Symmetries: {N_SYMS}, Topologies: {N_TOPOS}", flush=True)
    print(f"Device: {device}", flush=True)

    # ── Networks ───────────────────────────────────────────────────────
    actor1 = MultiHeadActor(s_dim, N_SYMS, max_cores, args.hidden_dim).to(device)
    actor2 = MultiHeadActor(s_dim, N_SYMS, max_cores, args.hidden_dim).to(device)
    actor3 = TopoActor(s_dim, N_TOPOS, args.hidden_dim).to(device)
    critic = CentralizedCritic(s_dim, args.hidden_dim).to(device)

    # RND exploration
    rnd = RNDModule(s_dim, hidden_dim=128, lr=1e-4, beta=args.rnd_beta)

    # Optimizers (separate per actor for independent backward passes)
    opt_actor1 = AdamW(actor1.parameters(), lr=args.lr)
    opt_actor2 = AdamW(actor2.parameters(), lr=args.lr)
    opt_actor3 = AdamW(actor3.parameters(), lr=args.lr)
    opt_critic = AdamW(critic.parameters(), lr=args.lr * 2)

    # ── Training ───────────────────────────────────────────────────────
    best_reward = -999
    topo_history = {t: [] for t in TOPOLOGIES}

    for epoch in range(args.epochs):
        # Collect episodes
        buffer = []
        ep_rewards = []
        ep_success = []

        for _ in range(args.steps_per_epoch):
            s1, s2, s3 = env.reset()
            s1_t = torch.FloatTensor(s1).unsqueeze(0).to(device)
            s2_t = torch.FloatTensor(s2).unsqueeze(0).to(device)
            s3_t = torch.FloatTensor(s3).unsqueeze(0).to(device)

            # Step 1: Agent 1
            a1, (lp1_sym, lp1_core) = actor1.get_action(s1_t)
            sym1 = a1[0]
            core_mask_1 = core_masks[sym1].unsqueeze(0)

            # Step 1 → env
            (s1_n, s2_n, s3_n), r1, done, info1 = env.step(a1, (0, 0), 0)

            s1_nt = torch.FloatTensor(s1_n).unsqueeze(0).to(device)
            s2_nt = torch.FloatTensor(s2_n).unsqueeze(0).to(device)
            s3_nt = torch.FloatTensor(s3_n).unsqueeze(0).to(device)

            # Step 2: Agent 2
            a2, (lp2_sym, lp2_core) = actor2.get_action(s2_nt)
            sym2 = a2[0]
            core_mask_2 = core_masks[sym2].unsqueeze(0)

            (s1_n2, s2_n2, s3_n2), r2, done, info2 = env.step(a1, a2, 0)

            s1_nt2 = torch.FloatTensor(s1_n2).unsqueeze(0).to(device)
            s2_nt2 = torch.FloatTensor(s2_n2).unsqueeze(0).to(device)
            s3_nt2 = torch.FloatTensor(s3_n2).unsqueeze(0).to(device)

            if done:
                # Invalid pair at step 2 → episode ends
                reward = r2
            else:
                # Step 3: Agent 3
                topo_mask = extract_topo_mask(s3_nt2).squeeze(0)
                a3, lp3 = actor3.get_action(s3_nt2, topo_mask.unsqueeze(0))

                (s1_n3, s2_n3, s3_n3), r3, done, info3 = env.step(a1, a2, a3)
                reward = r3
                ep_success.append(info3.get('success', False))

            ep_rewards.append(reward)

            # RND intrinsic reward (on key states)
            rnd_r1 = rnd.compute_intrinsic_reward(s1_n)
            rnd_r2 = rnd.compute_intrinsic_reward(s2_n)
            rnd_r3 = rnd.compute_intrinsic_reward(s3_n)
            intrinsic = (rnd_r1 + rnd_r2 + rnd_r3) / 3.0

            # Store step data (simplified: store final reward for all steps)
            # Step 1 data
            buffer.append({
                's1': s1_t, 's2': s2_t, 's3': s3_t,
                'sym1': torch.tensor([a1[0]]), 'core1': torch.tensor([a1[1]]),
                'lp1': lp1_sym + lp1_core,
                'sym2': torch.tensor([0]), 'core2': torch.tensor([0]),
                'lp2': torch.tensor([0.0]),
                'topo': torch.tensor([0]), 'lp3': torch.tensor([0.0]),
                'r': reward + intrinsic, 'done': done,
                'core_mask_1': core_mask_1,
                'core_mask_2': core_masks[0].unsqueeze(0),
                'topo_mask': torch.zeros(N_TOPOS),
                'step': 0,
            })

            # Step 2 data
            if not done:
                sym2_used = a2[0] if not done else 0
                buffer.append({
                    's1': s1_nt, 's2': s2_nt, 's3': s3_nt,
                    'sym1': torch.tensor([0]), 'core1': torch.tensor([0]),
                    'lp1': torch.tensor([0.0]),
                    'sym2': torch.tensor([a2[0]]), 'core2': torch.tensor([a2[1]]),
                    'lp2': lp2_sym + lp2_core,
                    'topo': torch.tensor([0]), 'lp3': torch.tensor([0.0]),
                    'r': reward + intrinsic, 'done': done,
                    'core_mask_1': core_masks[0].unsqueeze(0),
                    'core_mask_2': core_mask_2,
                    'topo_mask': torch.zeros(N_TOPOS),
                    'step': 1,
                })

            # Step 3 data
            if not done and 'info3' in dir():
                topo_m = extract_topo_mask(s3_nt2).squeeze(0)
                buffer.append({
                    's1': s1_nt2, 's2': s2_nt2, 's3': s3_nt2,
                    'sym1': torch.tensor([0]), 'core1': torch.tensor([0]),
                    'lp1': torch.tensor([0.0]),
                    'sym2': torch.tensor([0]), 'core2': torch.tensor([0]),
                    'lp2': torch.tensor([0.0]),
                    'topo': torch.tensor([a3]), 'lp3': lp3,
                    'r': reward + intrinsic, 'done': True,
                    'core_mask_1': core_masks[0].unsqueeze(0),
                    'core_mask_2': core_masks[0].unsqueeze(0),
                    'topo_mask': topo_m,
                    'step': 2,
                })

        # ── PPO Update ──────────────────────────────────────────────
        buffer_a1 = [b for b in buffer if b['step'] == 0]
        buffer_a2 = [b for b in buffer if b['step'] == 1]
        buffer_a3 = [b for b in buffer if b['step'] == 2]

        def ppo_update_actor(actor, buf, actor_id, mask_key, opt):
            if len(buf) < args.batch_size // 4:
                return 0.0

            old_lp = torch.cat([b['lp1'] if actor_id == 1 else
                                (b['lp2'] if actor_id == 2 else b['lp3'])
                                for b in buf]).detach()

            # Compute returns and advantages once
            s1_all = torch.cat([b['s1'] for b in buf])
            s2_all = torch.cat([b['s2'] for b in buf])
            s3_all = torch.cat([b['s3'] for b in buf])
            with torch.no_grad():
                vals = critic(s1_all, s2_all, s3_all).squeeze()

            returns, advantages = compute_gae(
                [b['r'] for b in buf],
                torch.cat([vals, torch.zeros(1, device=device)])
            )
            returns = returns.to(device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.to(device)

            total_loss = 0.0
            for _ in range(args.ppo_epochs):
                indices = torch.randperm(len(buf))
                for start in range(0, len(buf), args.batch_size):
                    end = min(start + args.batch_size, len(buf))
                    idx = indices[start:end]

                    batch = [buf[i] for i in idx]
                    s1_b = torch.cat([b['s1'] for b in batch])
                    s2_b = torch.cat([b['s2'] for b in batch])
                    s3_b = torch.cat([b['s3'] for b in batch])

                    if actor_id == 1:
                        sym_a = torch.cat([b['sym1'] for b in batch]).to(device)
                        core_a = torch.cat([b['core1'] for b in batch]).to(device)
                        masks = torch.cat([b['core_mask_1'] for b in batch])
                        lp, ent = actor.evaluate(s1_b, sym_a, core_a, masks)
                    elif actor_id == 2:
                        sym_a = torch.cat([b['sym2'] for b in batch]).to(device)
                        core_a = torch.cat([b['core2'] for b in batch]).to(device)
                        masks = torch.cat([b['core_mask_2'] for b in batch])
                        lp, ent = actor.evaluate(s2_b, sym_a, core_a, masks)
                    else:
                        topo_a = torch.cat([b['topo'] for b in batch]).to(device)
                        masks = torch.cat([b['topo_mask'] for b in batch])
                        lp, ent = actor.evaluate(s3_b, topo_a, masks)

                    ratio = (lp - old_lp[idx].to(device)).exp()
                    adv = advantages[idx].to(device)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Critic loss
                    vals_b = critic(s1_b, s2_b, s3_b).squeeze()
                    ret = returns[idx].to(device)
                    critic_loss = F.mse_loss(vals_b, ret)

                    loss = actor_loss + args.vf_coef * critic_loss - args.ent_coef * ent
                    total_loss += loss.item()

                    opt.zero_grad()
                    opt_critic.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 2.0)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 2.0)
                    opt.step()
                    opt_critic.step()

            return total_loss / max(len(buf) // args.batch_size, 1)

        loss1 = ppo_update_actor(actor1, buffer_a1, 1, 'core_mask_1', opt_actor1)
        loss2 = ppo_update_actor(actor2, buffer_a2, 2, 'core_mask_2', opt_actor2)
        loss3 = ppo_update_actor(actor3, buffer_a3, 3, 'topo_mask', opt_actor3)

        # RND update
        all_states = [b['s1'].squeeze(0).cpu().numpy() for b in buffer]
        rnd_loss = rnd.update(all_states)

        # Stats
        avg_reward = np.mean(ep_rewards) if ep_rewards else 0
        succ_rate = np.mean(ep_success) * 100 if ep_success else 0

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:4d}/{args.epochs} | "
                  f"Reward: {avg_reward:6.2f} | Success: {succ_rate:5.1f}% | "
                  f"Assembly: {env.assembly_success}/{env.assembly_total} | "
                  f"Loss: {loss1:.3f}/{loss2:.3f}/{loss3:.3f} | "
                  f"Topo dist: {dict(env.topo_counts)}", flush=True)

        if avg_reward > best_reward:
            best_reward = avg_reward
            ckpt_dir = Path(__file__).parent / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({
                'actor1': actor1.state_dict(),
                'actor2': actor2.state_dict(),
                'actor3': actor3.state_dict(),
                'critic': critic.state_dict(),
                'rnd': rnd.state_dict(),
                'epoch': epoch,
                'best_reward': best_reward,
            }, ckpt_dir / "mappo_v6_best.pt")

    print(f"\n=== Training complete ===")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final topology distribution: {dict(env.topo_counts)}")


if __name__ == '__main__':
    main()
