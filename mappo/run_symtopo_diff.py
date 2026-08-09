"""
Diffusion-integrated MAPPO training for continuous Core generation.

Agent 1/2: sym(6, discrete) + noise_level(1, continuous Gaussian) + conn(15, discrete)
Agent 3:   topo(N, discrete, masked)

Scaffold diffusion with noise_level ∈ [0,1] controls Core geometry variation.
"""
import torch
import torch.nn as nn
import numpy as np
import argparse
import os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from env_symtopo import SYMMETRY_TYPES, N_SYMS, ALL_CONNECTORS, N_CONNECTORS
from env_symtopo_diff import SymTopoEnvDiff
from replay_buffer import ReplayBuffer
from mappo_mpe import orthogonal_init


# ── RND ──────────────────────────────────────────────────────────────

class _RNDNet(nn.Module):
    def __init__(self, in_dim, out_dim, n_hid):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, out_dim)
    def forward(self, x): return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

class RND:
    def __init__(self, in_dim, out_dim=32, n_hid=256):
        self.target = _RNDNet(in_dim, out_dim, n_hid)
        for p in self.target.parameters(): p.requires_grad = False
        self.model = _RNDNet(in_dim, out_dim, n_hid)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
    def get_reward(self, obs_n):
        if not isinstance(obs_n, torch.Tensor): obs_n = torch.FloatTensor(obs_n)
        if obs_n.dim() == 1: obs_n = obs_n.unsqueeze(0)
        with torch.no_grad(): y_true = self.target(obs_n).detach()
        return torch.pow(self.model(obs_n) - y_true, 2).sum(dim=1)
    def update(self, replay_buffer):
        batch = replay_buffer.get_training_data()
        S0 = batch['obs_n'].clone().detach().reshape(-1, self.target.fc1.in_features)
        Ri = self.get_reward(S0); Ri.sum().backward()
        self.optimizer.step(); self.optimizer.zero_grad()


# ── Multi-head actor with continuous noise_level ─────────────────────

class MultiHeadActor12Diff(nn.Module):
    """
    3-headed actor: sym(discrete), noise_level(continuous Gaussian), conn(discrete).
    noise_level controls scaffold diffusion strength [0, 1].
    """

    def __init__(self, obs_dim, n_sym, noise_dim, n_conn, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())

        self.sym_head = nn.Linear(hidden, n_sym)
        self.noise_mu_head = nn.Linear(hidden, noise_dim)
        self.conn_head = nn.Linear(hidden, n_conn)

        # Learnable log-std for noise_level
        self.noise_logstd = nn.Parameter(torch.tensor(-1.0))  # init σ≈0.37

        self.n_sym = n_sym
        self.noise_dim = noise_dim
        self.n_conn = n_conn

    def forward(self, obs):
        feat = self.shared(obs)
        return self.sym_head(feat), self.noise_mu_head(feat), self.conn_head(feat)

    def get_action(self, obs, sym_mask, conn_mask):
        sym_logits, noise_mu, conn_logits = self.forward(obs)
        sym_logits = sym_logits + (1 - sym_mask) * -1e9
        conn_logits = conn_logits + (1 - conn_mask) * -1e9

        sym_a = torch.distributions.Categorical(logits=sym_logits).sample()
        conn_a = torch.distributions.Categorical(logits=conn_logits).sample()

        # Continuous noise_level: Gaussian policy, clamp to [0, 1]
        std = torch.exp(self.noise_logstd)
        noise_a = noise_mu + torch.randn_like(noise_mu) * std
        noise_a = noise_a.clamp(0.0, 1.0)

        return (sym_a, noise_a, conn_a)

    def evaluate(self, obs, sym_a, noise_a, conn_a, sym_mask, conn_mask):
        sym_logits, noise_mu, conn_logits = self.forward(obs)
        sym_logits = sym_logits + (1 - sym_mask) * -1e9
        conn_logits = conn_logits + (1 - conn_mask) * -1e9

        # Discrete log-probs
        d_sym = torch.distributions.Categorical(logits=sym_logits)
        d_conn = torch.distributions.Categorical(logits=conn_logits)

        # Continuous log-prob (Gaussian)
        std = torch.exp(self.noise_logstd)
        d_noise = torch.distributions.Normal(noise_mu, std)
        noise_lp = d_noise.log_prob(noise_a).sum(dim=-1)

        lp = d_sym.log_prob(sym_a) + noise_lp + d_conn.log_prob(conn_a)
        ent = (d_sym.entropy() + d_noise.entropy().mean() +
               d_conn.entropy()).mean() / 3.0
        return lp, ent


class TopoActor(nn.Module):
    """Single-head actor for topology selection."""
    def __init__(self, obs_dim, n_topo, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_topo))
    def forward(self, obs): return self.net(obs)
    def get_action(self, obs, mask):
        logits = self.forward(obs) + (1 - mask) * -1e9
        return torch.distributions.Categorical(logits=logits).sample()
    def evaluate(self, obs, a, mask):
        logits = self.forward(obs) + (1 - mask) * -1e9
        d = torch.distributions.Categorical(logits=logits)
        return d.log_prob(a), d.entropy().mean()


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x)


# ── Runner ───────────────────────────────────────────────────────────

class RunnerSymTopoDiff:
    def __init__(self, args):
        self.args = args
        np.random.seed(args.seed); torch.manual_seed(args.seed)

        # Load diffusion model
        diffusion_model = None
        diffusion_process = None
        if args.diff_ckpt and os.path.exists(args.diff_ckpt):
            print(f"Loading diffusion model from {args.diff_ckpt}...", flush=True)
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from symmcd_diffusion.models.conditional_denoiser import (
                SymmetryConditionedDenoiser)
            from symmcd_diffusion.models.diffusion_process import DiffusionProcess
            diffusion_model = SymmetryConditionedDenoiser(core_only=True)
            ckpt = torch.load(args.diff_ckpt, map_location=args.device)
            diffusion_model.load_state_dict(ckpt['model_state_dict'])
            diffusion_model = diffusion_model.to(args.device)
            diffusion_model.eval()
            diffusion_process = DiffusionProcess(
                num_atom_types=12, timesteps=1000)
            print("  Diffusion model loaded.", flush=True)

        topo_mode = getattr(args, 'topo_mode', 'all')
        self.env = SymTopoEnvDiff(
            diffusion_model=diffusion_model,
            diffusion_process=diffusion_process,
            topo_mode=topo_mode,
            device=args.device,
        )

        args.N = self.env.n
        args.obs_dim = self.env.observation_space
        args.state_dim = args.obs_dim * args.N
        args.episode_limit = self.env.episode_limit
        self.total_steps = 0

        n_topos = self.env._n_topos
        noise_dim = self.env.noise_dim

        print(f"=== SymTopo MAPPO + Diffusion ({topo_mode.upper()}) ===", flush=True)
        print(f"N={args.N} obs={args.obs_dim} state={args.state_dim}", flush=True)
        print(f"Agent1/2: sym({N_SYMS}) + noise_level({noise_dim}) + conn({N_CONNECTORS})", flush=True)
        print(f"Agent3: topo({n_topos}) = {self.env._topo_list}", flush=True)
        print(f"Core: scaffold diffusion (noise_level ∈ [0,1])", flush=True)

        self.actor_12 = MultiHeadActor12Diff(
            args.obs_dim, N_SYMS, noise_dim, N_CONNECTORS, args.mlp_hidden_dim)
        self.actor_3 = TopoActor(args.obs_dim, n_topos, args.mlp_hidden_dim)
        self.critic = Critic(args.state_dim, args.mlp_hidden_dim)

        self.opt_12 = torch.optim.Adam(self.actor_12.parameters(), lr=args.lr)
        self.opt_3 = torch.optim.Adam(self.actor_3.parameters(), lr=args.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=args.lr * 2)
        self.rnd = RND(args.obs_dim, 32, 256)
        self.replay_buffer = ReplayBuffer(args)
        from run_symtopo import N2Predictor
        self.predictor = N2Predictor()

    def choose_action(self, obs_n, masks, agent_id, evaluate=False):
        obs_t = torch.FloatTensor(obs_n[agent_id]).unsqueeze(0)
        if agent_id in [0, 1]:
            with torch.no_grad():
                sym_mask = torch.FloatTensor(masks[agent_id]['sym']).unsqueeze(0)
                conn_mask = torch.FloatTensor(masks[agent_id]['conn']).unsqueeze(0)
                sym_a, noise_a, conn_a = self.actor_12.get_action(
                    obs_t, sym_mask, conn_mask)
            return (sym_a.item(), noise_a.item(), conn_a.item())
        else:
            with torch.no_grad():
                topo_mask = torch.FloatTensor(masks[agent_id]['topo']).unsqueeze(0)
                topo_a = self.actor_3.get_action(obs_t, topo_mask)
            return topo_a.item()

    def _encode_action_12(self, sym, noise, conn):
        """Encode (sym, noise, conn) as single float: sym*1000 + noise*100 + conn."""
        return float(sym) * 1000.0 + float(noise) * 100.0 + float(conn)

    def _decode_action_12(self, code):
        """Decode back to (sym, noise, conn)."""
        c = float(code)
        sym = int(c // 1000)
        noise = (c % 1000) / 100.0
        conn = int(c % 100)
        return sym, noise, conn

    def run_episode(self, evaluate=False):
        obs_n, info, masks = self.env.reset()
        episode_reward = 0.0

        for step in range(self.args.episode_limit):
            agent_id = step
            a = self.choose_action(obs_n, masks, agent_id, evaluate)

            if agent_id == 0:
                actions = [a, (0, 0.0, 0), 0]
                a_store = np.array([self._encode_action_12(*a), 0.0, 0.0])
            elif agent_id == 1:
                actions = [(0, 0.0, 0), a, 0]
                a_store = np.array([0.0, self._encode_action_12(*a), 0.0])
            else:
                actions = [(0, 0.0, 0), (0, 0.0, 0), a]
                a_store = np.array([0.0, 0.0, float(a)])

            s = obs_n.flatten()
            s_t = torch.FloatTensor(s).unsqueeze(0)
            with torch.no_grad(): v_full = self.critic(s_t).item()

            obs_next_n, r_n, done_n, info, masks = self.env.step(actions)

            rnd_vals = self.rnd.get_reward(obs_next_n).detach().numpy()
            r_n_total = np.array(r_n) + self.args.rnd_beta * rnd_vals
            episode_reward += r_n[agent_id]

            if not evaluate:
                self.replay_buffer.store_transition(
                    step, obs_n.copy(), obs_n.flatten(),
                    np.array([v_full] * self.args.N),
                    a_store,
                    np.zeros(self.args.N),
                    r_n_total,
                    np.array(done_n, dtype=float))

            obs_n = obs_next_n
            if all(done_n):
                if not evaluate:
                    s_last = obs_n.flatten()
                    s_t_last = torch.FloatTensor(s_last).unsqueeze(0)
                    with torch.no_grad(): v_last = self.critic(s_t_last).item()
                    for fs in range(step + 1, self.args.episode_limit):
                        self.replay_buffer.store_transition(
                            fs, obs_n.copy(), obs_n.flatten(),
                            np.array([v_last] * self.args.N),
                            np.zeros(self.args.N),
                            np.zeros(self.args.N),
                            np.zeros(self.args.N), np.ones(self.args.N))
                    self.replay_buffer.store_last_value(
                        step + 1, np.array([v_last] * self.args.N))
                break
        else:
            if not evaluate:
                s_last = obs_n.flatten()
                s_t_last = torch.FloatTensor(s_last).unsqueeze(0)
                with torch.no_grad(): v_last = self.critic(s_t_last).item()
                self.replay_buffer.store_last_value(
                    self.args.episode_limit, np.array([v_last] * self.args.N))

        return episode_reward, step + 1

    def train(self):
        if self.replay_buffer.episode_num < self.args.batch_size:
            return 0.0

        batch = self.replay_buffer.get_training_data()

        adv = []; gae = 0.0
        with torch.no_grad():
            deltas = (batch['r_n'] + self.args.gamma * batch['v_n'][:, 1:] *
                      (1 - batch['done_n']) - batch['v_n'][:, :-1])
            for t in reversed(range(self.args.episode_limit)):
                gae = deltas[:, t] + self.args.gamma * self.args.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)
            v_target = adv + batch['v_n'][:, :-1]
            if self.args.use_adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        obs = batch['obs_n']; a_n = batch['a_n']
        done_n = batch['done_n']; valid_mask = (done_n[:, :, 0] == 0)

        total_loss = 0.0
        for _ in range(self.args.K_epochs):
            for st in range(0, self.args.batch_size, self.args.mini_batch_size):
                ed = min(st + self.args.mini_batch_size, self.args.batch_size)
                mb_obs = obs[st:ed]; mb_adv = adv[st:ed, :, 0]
                mb_target = v_target[st:ed, :, 0]
                mb_a = a_n[st:ed]; mb_valid = valid_mask[st:ed]

                loss_12 = torch.tensor(0.0); loss_3 = torch.tensor(0.0)

                for t in [0, 1]:
                    sv = mb_valid[:, t]
                    if sv.sum() == 0: continue
                    for ag in [0, 1]:
                        o = mb_obs[:, t, ag, :][sv]
                        adv_t = mb_adv[:, t][sv]
                        if len(o) == 0: continue

                        a_codes = mb_a[:, t, ag][sv]  # scalar floats
                        sym_a = torch.zeros(len(o), dtype=torch.long)
                        noise_a = torch.zeros(len(o), 1)
                        conn_a = torch.zeros(len(o), dtype=torch.long)

                        for i in range(len(o)):
                            s, n, c = self._decode_action_12(float(a_codes[i]))
                            sym_a[i] = s
                            noise_a[i] = n
                            conn_a[i] = c

                        sym_mask = torch.ones(len(o), N_SYMS)
                        conn_mask = torch.ones(len(o), N_CONNECTORS)

                        try:
                            lp, ent = self.actor_12.evaluate(
                                o, sym_a, noise_a, conn_a,
                                sym_mask, conn_mask)
                        except Exception:
                            continue
                        ratio = torch.exp(lp - lp.detach())
                        s1 = ratio * adv_t
                        s2 = torch.clamp(ratio, 1 - self.args.epsilon,
                                         1 + self.args.epsilon) * adv_t
                        loss_12 = loss_12 - torch.min(s1, s2).mean() - \
                                  self.args.entropy_coef * ent

                # Agent 3
                if mb_obs.size(1) > 2:
                    sv2 = mb_valid[:, 2]
                    if sv2.sum() > 0:
                        o3 = mb_obs[:, 2, 2, :][sv2]
                        adv3 = mb_adv[:, 2][sv2]
                        if len(o3) > 0:
                            a3_codes = mb_a[:, 2, 2][sv2]
                            a3 = a3_codes.long()
                            topo_mask = torch.ones(len(o3), self.env._n_topos)
                            lp3, ent3 = self.actor_3.evaluate(o3, a3, topo_mask)
                            ratio3 = torch.exp(lp3 - lp3.detach())
                            s13 = ratio3 * adv3
                            s23 = torch.clamp(ratio3, 1 - self.args.epsilon,
                                             1 + self.args.epsilon) * adv3
                            loss_3 = loss_3 - torch.min(s13, s23).mean() - \
                                     self.args.entropy_coef_3 * ent3

                s_mb = mb_obs.reshape(mb_obs.size(0), mb_obs.size(1), -1)
                s_val = s_mb[mb_valid]; t_val = mb_target[mb_valid]
                critic_loss = ((self.critic(s_val).squeeze() - t_val)**2).mean() \
                    if len(s_val) > 0 else torch.tensor(0.0)

                loss = loss_12 + loss_3 + critic_loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                total_loss += loss.item()
                self.opt_12.zero_grad(); self.opt_3.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_12.parameters()) +
                    list(self.actor_3.parameters()) +
                    list(self.critic.parameters()), 10.0)
                self.opt_12.step(); self.opt_3.step(); self.opt_critic.step()

        self.replay_buffer.reset_buffer()
        return total_loss

    def run(self):
        print("Starting training...", flush=True)
        batch_idx = 0
        while self.total_steps < self.args.max_train_steps:
            ep_rewards = []
            for _ in range(self.args.batch_size):
                ep_r, ep_steps = self.run_episode(evaluate=False)
                ep_rewards.append(ep_r)
                self.total_steps += ep_steps

            if self.replay_buffer.episode_num < self.args.batch_size:
                continue

            self.rnd.update(self.replay_buffer)
            loss = self.train()
            batch_idx += 1

            avg_r = np.mean(ep_rewards)
            succ = self.env.assembly_success / max(self.env.assembly_total, 1) * 100
            td = {k: v for k, v in sorted(
                self.env.topo_counts.items(), key=lambda x: -x[1])[:5]}
            cd = dict(self.env.conn_pair_counts.most_common(3))

            if batch_idx % 10 == 0:
                zs = {t: f"{s['n']}/{s['mean']:.2f}" for t, s in
                      sorted(self.env.topo_n2_stats.items(), key=lambda x: -x[1]['n'])[:5]
                      if s['n'] > 0}
                print(f"B{batch_idx:4d} S{self.total_steps:6d} | "
                      f"R:{avg_r:5.1f} Succ:{succ:4.0f}% | "
                      f"Loss:{loss if isinstance(loss, float) else 0:.2f} | "
                      f"DiffCalls:{self.env.diffusion_calls} | "
                      f"Topo:{td} | Conn:{cd} | Z(n/μ):{zs}", flush=True)
            else:
                print(f"B{batch_idx:4d} S{self.total_steps:6d} | "
                      f"R:{avg_r:5.1f} Succ:{succ:4.0f}% | "
                      f"Loss:{loss if isinstance(loss, float) else 0:.2f} | "
                      f"Diff:{self.env.diffusion_calls} | "
                      f"Topo:{td}", flush=True)

            if batch_idx % self.args.predictor_freq == 0 and \
               len(self.env.success_combos) > 0:
                self.predictor.run(self.env.success_combos, self.env.n2_cache)
                self.env.success_combos = []

            if batch_idx % 50 == 0:
                d = '/home/tianyajun/MARL_for_COFs/mappo/model'
                os.makedirs(d, exist_ok=True)
                torch.save(self.actor_12.state_dict(),
                          f'{d}/symtopo_diff_a12_s{self.total_steps}.pth')
                torch.save(self.actor_3.state_dict(),
                          f'{d}/symtopo_diff_a3_s{self.total_steps}.pth')

        print(f"Done. Topo: {dict(self.env.topo_counts)}")
        print(f"Conn pairs: {dict(self.env.conn_pair_counts)}")
        print(f"Diffusion calls: {self.env.diffusion_calls}")
        if self.env.diffusion_rmsds:
            print(f"Avg RMSD: {np.mean(self.env.diffusion_rmsds):.3f}Å")
        self.env.restore_cores()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--max_train_steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--mini_batch_size", type=int, default=4)
    p.add_argument("--mlp_hidden_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lamda", type=float, default=0.95)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--K_epochs", type=int, default=4)
    p.add_argument("--entropy_coef", type=float, default=0.05)
    p.add_argument("--entropy_coef_3", type=float, default=0.20)
    p.add_argument("--predictor_freq", type=int, default=20)
    p.add_argument("--rnd_beta", type=float, default=0.3)
    p.add_argument("--use_adv_norm", type=bool, default=True)
    p.add_argument("--use_orthogonal_init", type=bool, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topo_mode", type=str, default="all",
                   choices=["all", "2d", "3d"])
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--diff_ckpt", type=str,
                   default="/home/tianyajun/MARL_for_COFs/symmcd_diffusion/"
                           "checkpoints/symmcd_epoch100.pt",
                   help="Phase 2 diffusion checkpoint")
    args = p.parse_args()
    RunnerSymTopoDiff(args).run()
