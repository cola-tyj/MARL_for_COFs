#!/usr/bin/env python
"""Phase 4 v5: Full action space with connector + FG selection."""

import sys, os, time, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent))
from env_v5 import (COFDesignEnvV5, SYMMETRY_TYPES, TOPOLOGIES,
                    N_CONNECTORS, N_FUNC_GROUPS,
                    SYMMETRY_TO_CONNECTORS, SYMMETRY_PAIR_TO_TOPOLOGY)


class MHActor(nn.Module):
    def __init__(self, d=16, n_sym=6, n_core=10, n_conn=10, n_fg=10, h=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(h, n_sym), nn.Linear(h, n_core),
                                    nn.Linear(h, n_conn), nn.Linear(h, n_fg)])
    def forward(self, x):
        h = self.shared(x); return [head(h) for head in self.heads]
    def act(self, x):
        logits = self.forward(x); actions, lp = [], torch.tensor(0.0)
        for l in logits:
            d = Categorical(logits=l); a = d.sample()
            actions.append(a.item()); lp = lp + d.log_prob(a)
        return tuple(actions), lp
    def evaluate(self, x, actions):
        logits = self.forward(x); lp, ent = torch.tensor(0.0), torch.tensor(0.0)
        for i, l in enumerate(logits):
            d = Categorical(logits=l); lp = lp + d.log_prob(actions[:, i])
            ent = ent + d.entropy().mean()
        return lp, ent


class TopoActor(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 9))
    def forward(self, x): return self.net(x)
    def act(self, x, mask=None):
        l = self.forward(x)
        if mask is not None: l = l + (1 - mask) * -1e9
        d = Categorical(logits=l); a = d.sample(); return a.item(), d.log_prob(a)
    def evaluate(self, x, a):
        l = self.forward(x); d = Categorical(logits=l); return d.log_prob(a), d.entropy().mean()


class Critic(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(48, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x): return self.net(x)


def train(args):
    import logging; logging.getLogger("pycofbuilder").setLevel(logging.ERROR)
    n2p = "symmcd_diffusion/generated/n2_lookup.json" if args.real_reward else None
    env = COFDesignEnvV5(n2_lookup_path=n2p)
    a1 = MHActor(); a2 = MHActor(); a3 = TopoActor(); critic = Critic()
    opt = AdamW(list(a1.parameters()) + list(a2.parameters()) +
                list(a3.parameters()) + list(critic.parameters()), lr=args.lr)

    print(f"=== v5: Full Action Space ===")
    print(f"Connectors: {N_CONNECTORS} | Func Groups: {N_FUNC_GROUPS}")
    print(f"Agent 1/2: sym×core×conn×fg | Agent 3: topo (masked)")

    ep_r = []; best_r = -float("inf"); best_d = None; t0 = time.time()
    buf = {"s1": [], "s2": [], "s3": [], "a1": [], "a2": [], "a3": [],
           "r": [], "lp1": [], "lp2": [], "lp3": [], "v": []}

    for ep in range(1, args.episodes + 1):
        s1, s2, s3 = env.reset()
        s1t = torch.FloatTensor(s1).unsqueeze(0)
        s2t = torch.FloatTensor(s2).unsqueeze(0)
        s3t = torch.FloatTensor(s3).unsqueeze(0)

        with torch.no_grad():
            v = critic(torch.cat([s1t, s2t, s3t], -1))
        a1v, lp1 = a1.act(s1t); a2v, lp2 = a2.act(s2t)

        # Topology mask from symmetry pair
        ca = SYMMETRY_TO_CONNECTORS.get(SYMMETRY_TYPES[a1v[0]], 2)
        cb = SYMMETRY_TO_CONNECTORS.get(SYMMETRY_TYPES[a2v[0]], 2)
        valid = SYMMETRY_PAIR_TO_TOPOLOGY.get((ca, cb), ["HCB_A"])
        tm = torch.zeros(9)
        for t in valid:
            if t in TOPOLOGIES: tm[TOPOLOGIES.index(t)] = 1.0
        a3v, lp3 = a3.act(s3t, tm.unsqueeze(0))

        (_, _, _), r, _, info = env.step(a1v, a2v, a3v)

        for k, val in [("s1", s1), ("s2", s2), ("s3", s3), ("a1", a1v), ("a2", a2v),
                       ("a3", a3v), ("r", r), ("lp1", lp1), ("lp2", lp2), ("lp3", lp3),
                       ("v", v.item())]:
            buf[k].append(val)
        ep_r.append(r)

        if ep % args.update_every == 0 and len(buf["r"]) >= 2:
            R = 0; ret = []
            for rr in reversed(buf["r"]): R = rr + args.gamma * R; ret.insert(0, R)
            ret = torch.FloatTensor(ret)
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)

            s1b = torch.FloatTensor(np.array(buf["s1"]))
            s2b = torch.FloatTensor(np.array(buf["s2"]))
            s3b = torch.FloatTensor(np.array(buf["s3"]))
            a1b = torch.LongTensor([list(x) for x in buf["a1"]])
            a2b = torch.LongTensor([list(x) for x in buf["a2"]])
            a3b = torch.LongTensor(buf["a3"])
            olp1 = torch.stack(buf["lp1"]).detach()
            olp2 = torch.stack(buf["lp2"]).detach()
            olp3 = torch.stack(buf["lp3"]).detach()
            vb = torch.FloatTensor(buf["v"])
            adv = ret - vb.detach()

            nlp1, e1 = a1.evaluate(s1b, a1b)
            nlp2, e2 = a2.evaluate(s2b, a2b)
            nlp3, e3 = a3.evaluate(s3b, a3b)

            def ppo(n, o):
                r = torch.exp(n - o)
                return -torch.min(r * adv, torch.clamp(r, 1 - args.eps_clip, 1 + args.eps_clip) * adv).mean()

            loss = (ppo(nlp1, olp1) + ppo(nlp2, olp2) + ppo(nlp3, olp3) +
                    0.5 * F.mse_loss(vb, ret) - 0.01 * (e1 + e2 + e3))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(a1.parameters()) + list(a2.parameters()) +
                                           list(a3.parameters()) + list(critic.parameters()), 1.0)
            opt.step()
            for k in buf: buf[k] = []

            sr = env.assembly_success / max(env.assembly_total, 1)
            print(f"Ep {ep:5d}/{args.episodes} | AvgR: {np.mean(ep_r[-args.update_every:]):6.1f} | "
                  f"Success: {sr:.0%} | Loss: {loss.item():5.2f} | {time.time()-t0:.0f}s")

        if r > best_r:
            best_r = r
            best_d = "{} {}_{}+{}_{}".format(info.get("topology", "?"),
                info.get("sym_a", "?"), info.get("conn_a", "?"),
                info.get("sym_b", "?"), info.get("conn_b", "?"))

        if ep % 100 == 0:
            td = {t: env.topo_counts[t] for t in sorted(env.topo_counts) if env.topo_counts[t] > 0}
            print(f"  [Ep {ep}] topo_dist={td}")

    sr = env.assembly_success / max(env.assembly_total, 1)
    print(f"\nDone: {sr:.0%} | Best={best_r:.1f} ({best_d}) | Mean={np.mean(ep_r):.1f}")
    out = Path(__file__).parent / "checkpoints"; out.mkdir(exist_ok=True)
    torch.save({"a1": a1.state_dict(), "a2": a2.state_dict(), "a3": a3.state_dict(),
                "critic": critic.state_dict(), "best": best_r, "design": best_d,
                "topo_counts": env.topo_counts}, str(out / "mappo_v5.pt"))
    print(f"Saved: {out}/mappo_v5.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps_clip", type=float, default=0.2)
    p.add_argument("--update_every", type=int, default=20)
    p.add_argument("--real_reward", action="store_true", default=False)
    train(p.parse_args())
