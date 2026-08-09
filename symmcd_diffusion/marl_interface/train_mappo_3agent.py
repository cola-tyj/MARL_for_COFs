#!/usr/bin/env python
"""Phase 4: 3-Agent MAPPO with action masking."""

import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent))
from env_v3 import (COFDesignEnvV3, TOPOLOGIES, SYMMETRY_TYPES,
                    SYMMETRY_TO_CONNECTORS, TOPOLOGY_TO_SYM_PAIRS)
from rnd_module import RNDModule


# ── Networks ────────────────────────────────────────────────────────────────

class DiscreteActor(nn.Module):
    """Policy for a single discrete action space."""
    def __init__(self, s_dim=16, a_dim=9, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s_dim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, a_dim))

    def forward(self, s): return self.net(s)

    def get_action(self, s, mask=None):
        logits = self.forward(s)
        if mask is not None:
            logits = logits + (1 - mask) * -1e9  # mask out invalid actions
        d = Categorical(logits=logits)
        a = d.sample()
        return a.item(), d.log_prob(a)

    def evaluate(self, s, a, mask=None):
        logits = self.forward(s)
        if mask is not None:
            logits = logits + (1 - mask) * -1e9
        d = Categorical(logits=logits)
        return d.log_prob(a), d.entropy().mean()


class CentralizedCritic(nn.Module):
    def __init__(self, d0, d1, d2, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d0+d1+d2, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, s0, s1, s2):
        return self.net(torch.cat([s0, s1, s2], dim=-1))


# ── Pre-compute valid symmetries per topology ─────────────────────────────────

def build_constraints():
    """For each topology, list valid (sym_idx_a, sym_idx_b) pairs."""
    constraints = {}
    for ti, topo in enumerate(TOPOLOGIES):
        pairs = TOPOLOGY_TO_SYM_PAIRS[topo]
        valid = []
        for ca, cb in pairs:
            a_idxs = [i for i, s in enumerate(SYMMETRY_TYPES)
                      if SYMMETRY_TO_CONNECTORS[s] == ca]
            b_idxs = [i for i, s in enumerate(SYMMETRY_TYPES)
                      if SYMMETRY_TO_CONNECTORS[s] == cb]
            for ia in a_idxs:
                for ib in b_idxs:
                    valid.append((ia, ib))
        constraints[ti] = valid
    return constraints


# ── MAPPO Agent ──────────────────────────────────────────────────────────────

class MAPPO3Agent:
    def __init__(self, n_topo, n_sym, lr=1e-4, gamma=0.99, eps=0.2):
        self.actor0 = DiscreteActor(16, n_topo)
        self.actor1 = DiscreteActor(16, n_sym)
        self.actor2 = DiscreteActor(16, n_sym)
        self.critic = CentralizedCritic(16, 16, 16)
        params = (list(self.actor0.parameters()) + list(self.actor1.parameters()) +
                  list(self.actor2.parameters()) + list(self.critic.parameters()))
        self.opt = AdamW(params, lr=lr)
        self.gamma = gamma
        self.eps_clip = eps
        self.clear_buffer()

    def select(self, s0, s1, s2, mask1=None, mask2=None):
        s0t = torch.FloatTensor(s0).unsqueeze(0)
        s1t = torch.FloatTensor(s1).unsqueeze(0)
        s2t = torch.FloatTensor(s2).unsqueeze(0)
        with torch.no_grad():
            v = self.critic(s0t, s1t, s2t)
        if mask1 is not None: mask1 = torch.FloatTensor(mask1).unsqueeze(0)
        if mask2 is not None: mask2 = torch.FloatTensor(mask2).unsqueeze(0)
        a0, lp0 = self.actor0.get_action(s0t)
        a1, lp1 = self.actor1.get_action(s1t, mask1)
        a2, lp2 = self.actor2.get_action(s2t, mask2)
        return a0, a1, a2, lp0, lp1, lp2, v.item()

    def store(self, s0, s1, s2, a0, a1, a2, r, lp0, lp1, lp2, v):
        self.buf['s0'].append(s0); self.buf['s1'].append(s1); self.buf['s2'].append(s2)
        self.buf['a0'].append(a0); self.buf['a1'].append(a1); self.buf['a2'].append(a2)
        self.buf['r'].append(r); self.buf['lp0'].append(lp0)
        self.buf['lp1'].append(lp1); self.buf['lp2'].append(lp2); self.buf['v'].append(v)

    def update(self):
        b = self.buf; n = len(b['r'])
        if n < 2: self.clear_buffer(); return {}
        R = 0; ret = []
        for r in reversed(b['r']): R = r + self.gamma * R; ret.insert(0, R)
        ret = torch.FloatTensor(ret); ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        s0 = torch.FloatTensor(np.array(b['s0'])); s1 = torch.FloatTensor(np.array(b['s1']))
        s2 = torch.FloatTensor(np.array(b['s2']))
        a0 = torch.LongTensor(b['a0']); a1 = torch.LongTensor(b['a1']); a2 = torch.LongTensor(b['a2'])
        olp0 = torch.stack(b['lp0']).detach(); olp1 = torch.stack(b['lp1']).detach()
        olp2 = torch.stack(b['lp2']).detach()
        v = torch.FloatTensor(b['v']); adv = ret - v.detach()

        nlp0, e0 = self.actor0.evaluate(s0, a0)
        nlp1, e1 = self.actor1.evaluate(s1, a1)
        nlp2, e2 = self.actor2.evaluate(s2, a2)

        def ppo(new_lp, old_lp):
            r = torch.exp(new_lp - old_lp)
            return -torch.min(r * adv, torch.clamp(r, 1 - self.eps_clip, 1 + self.eps_clip) * adv).mean()

        loss = (ppo(nlp0, olp0) + ppo(nlp1, olp1) + ppo(nlp2, olp2) +
                0.5 * F.mse_loss(v, ret) - 0.01 * (e0 + e1 + e2))
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.actor0.parameters()) + list(self.actor1.parameters()) +
                                       list(self.actor2.parameters()) + list(self.critic.parameters()), 1.0)
        self.opt.step()
        self.clear_buffer()
        return {"loss": loss.item()}

    def clear_buffer(self):
        self.buf = {k: [] for k in ['s0','s1','s2','a0','a1','a2','r','lp0','lp1','lp2','v']}


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    import logging; logging.getLogger("pycofbuilder").setLevel(logging.ERROR)
    n2p = "symmcd_diffusion/generated/n2_lookup.json" if args.real_reward else None
    env = COFDesignEnvV3(n2_lookup_path=n2p)
    constraints = build_constraints()
    agent = MAPPO3Agent(env.n_topologies, env.n_symmetries, lr=args.lr,
                        gamma=args.gamma, eps=args.eps_clip)

    print(f"=== 3-Agent MAPPO (action-masked) ===")
    print(f"Agent 0: {env.n_topologies} topologies")
    print(f"Agent 1/2: {env.n_symmetries} symmetries (masked by topology)")
    print(f"Topologies: {TOPOLOGIES}")
    for ti, topo in enumerate(TOPOLOGIES):
        valid = constraints[ti]
        a_syms = sorted(set(v[0] for v in valid))
        b_syms = sorted(set(v[1] for v in valid))
        print(f"  {topo}: A∈{[SYMMETRY_TYPES[i] for i in a_syms]}, B∈{[SYMMETRY_TYPES[i] for i in b_syms]}")

    # RND exploration module (optional)
    rnd = RNDModule(state_dim=21, beta=0.3) if not args.no_rnd else None  # 9 topo + 6 sym_a + 6 sym_b
    ep_r = []; best_r = -float('inf'); best_d = None; t0 = time.time()
    rnd_states = []  # batch states for periodic RND update

    for ep in range(1, args.episodes + 1):
        s0, s1, s2 = env.reset()
        s0t = torch.FloatTensor(s0).unsqueeze(0)
        s1t = torch.FloatTensor(s1).unsqueeze(0)
        s2t = torch.FloatTensor(s2).unsqueeze(0)

        with torch.no_grad():
            v = agent.critic(s0t, s1t, s2t)

        # Agent 0: select topology (no mask)
        a0, lp0 = agent.actor0.get_action(s0t)

        # Build masks based on Agent 0's choice
        valid_pairs = constraints[a0]
        mask1 = torch.zeros(env.n_symmetries)
        mask2 = torch.zeros(env.n_symmetries)
        for ia, ib in valid_pairs:
            mask1[ia] = 1.0
            mask2[ib] = 1.0

        # Agent 1/2: select from masked action space
        a1, lp1 = agent.actor1.get_action(s1t, mask1.unsqueeze(0))
        a2, lp2 = agent.actor2.get_action(s2t, mask2.unsqueeze(0))

        (ns0, ns1, ns2), r_ext, done, info = env.step(a0, (a1, 0), (a2, 0))

        # RND intrinsic reward: encourage novel topo+sym combinations
        if rnd is not None:
            joint = np.concatenate([ns0[:9], ns1[:6], ns2[:6]])
            r_int = rnd.compute_intrinsic_reward(joint)
            r_total = r_ext + r_int
            rnd_states.append(joint)
        else:
            r_total = r_ext

        agent.store(s0, s1, s2, a0, a1, a2, r_total, lp0, lp1, lp2, v.item())
        ep_r.append(r_total)

        if ep % args.update_every == 0:
            st = agent.update()
            if rnd is not None:
                rnd_loss = rnd.update(rnd_states[-100:]); rnd_states = []
            if st:
                sr = env.assembly_success / max(env.assembly_total, 1)
                rnd_str = f"RND: {rnd_loss:.3f} | " if rnd is not None else ""
                print(f"Ep {ep:5d}/{args.episodes} | AvgR: {np.mean(ep_r[-args.update_every:]):6.1f} | "
                      f"Success: {sr:.0%} | Loss: {st['loss']:5.2f} | {rnd_str}"
                      f"{time.time()-t0:.0f}s")

        if r_total > best_r:
            best_r = r_total
            best_d = f"{info.get('topology','?')} {info.get('sym_a','?')}+{info.get('sym_b','?')}"

        if ep % 100 == 0:
            print(f"  [Ep {ep}] mean={np.mean(ep_r[-100:]):.1f} "
                  f"success={env.assembly_success}/{env.assembly_total}")

    sr = env.assembly_success / max(env.assembly_total, 1)
    print(f"\n=== Done: {sr:.0%} | Best={best_r:.1f} ({best_d}) | Mean={np.mean(ep_r):.1f} ===")
    out = Path(__file__).parent / "checkpoints"; out.mkdir(exist_ok=True)
    save_dict = {"actor0": agent.actor0.state_dict(), "actor1": agent.actor1.state_dict(),
                 "actor2": agent.actor2.state_dict(), "critic": agent.critic.state_dict(),
                 "best": best_r, "design": best_d}
    if rnd is not None: save_dict["rnd"] = rnd.state_dict()
    torch.save(save_dict, str(out / "mappo_3agent.pt"))
    print(f"Saved: {out}/mappo_3agent.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps_clip", type=float, default=0.2)
    p.add_argument("--update_every", type=int, default=20)
    p.add_argument("--real_reward", action="store_true", default=False)
    p.add_argument("--no_rnd", action="store_true", default=False,
                   help="Disable RND intrinsic reward")
    train(p.parse_args())
