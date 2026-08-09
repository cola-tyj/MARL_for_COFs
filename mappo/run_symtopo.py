"""
run_symtopo.py — MAPPO training for SymTopo env with connector selection.

Multi-head actor design:
  Agent 1/2: 3 heads — sym(6), core(~41), connector(7)
  Agent 3:   1 head  — topology(9, masked by sym pair)

Connector pairs validated by env: invalid pair → -1 reward.
"""
import torch
import torch.nn as nn
import numpy as np
import argparse
import os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from env_symtopo import (SymTopoEnv, SYMMETRY_TYPES, N_SYMS,
                          ALL_CONNECTORS, N_CONNECTORS)
from replay_buffer import ReplayBuffer
from mappo_mpe import orthogonal_init


# ── RND ──────────────────────────────────────────────────────────────────

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


# ── Multi-head actor ─────────────────────────────────────────────────────

class MultiHeadActor12(nn.Module):
    """3-headed actor: sym, core, connector. Shared trunk, per-head outputs."""
    def __init__(self, obs_dim, n_sym, max_cores, n_conn, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.sym_head = nn.Linear(hidden, n_sym)
        self.core_head = nn.Linear(hidden, max_cores)
        self.conn_head = nn.Linear(hidden, n_conn)
        self.n_sym, self.max_cores, self.n_conn = n_sym, max_cores, n_conn

    def forward(self, obs):
        feat = self.shared(obs)
        return self.sym_head(feat), self.core_head(feat), self.conn_head(feat)

    def get_action(self, obs, sym_mask, core_mask, conn_mask):
        sym_logits, core_logits, conn_logits = self.forward(obs)
        sym_logits = sym_logits + (1 - sym_mask) * -1e9
        core_logits = core_logits + (1 - core_mask) * -1e9
        conn_logits = conn_logits + (1 - conn_mask) * -1e9

        sym_a = torch.distributions.Categorical(logits=sym_logits).sample()
        core_a = torch.distributions.Categorical(logits=core_logits).sample()
        conn_a = torch.distributions.Categorical(logits=conn_logits).sample()
        return (sym_a, core_a, conn_a)

    def evaluate(self, obs, sym_a, core_a, conn_a, sym_mask, core_mask, conn_mask):
        sym_logits, core_logits, conn_logits = self.forward(obs)
        sym_logits = sym_logits + (1 - sym_mask) * -1e9
        core_logits = core_logits + (1 - core_mask) * -1e9
        conn_logits = conn_logits + (1 - conn_mask) * -1e9

        d_sym = torch.distributions.Categorical(logits=sym_logits)
        d_core = torch.distributions.Categorical(logits=core_logits)
        d_conn = torch.distributions.Categorical(logits=conn_logits)
        lp = d_sym.log_prob(sym_a) + d_core.log_prob(core_a) + d_conn.log_prob(conn_a)
        ent = (d_sym.entropy() + d_core.entropy() + d_conn.entropy()).mean() / 3.0
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


# ── Runner ───────────────────────────────────────────────────────────────

# ── Dynamic N2 Predictor ─────────────────────────────────────────────────

class N2Predictor:
    """Batch PMTransformer predictor: saves CIFs, runs doPredict, updates cache."""

    def __init__(self, cif_dir='/home/tianyajun/MARL_for_COFs/cofs',
                 log_dir='/home/tianyajun/MARL_for_COFs/logs',
                 device='cuda:3'):
        self.cif_dir = os.path.join(cif_dir, 'cif')
        self.log_dir = log_dir
        self.device = device
        self.call_count = 0
        self.predicted_names = set()  # avoid re-predicting same COF
        os.makedirs(self.cif_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    def run(self, success_combos, n2_cache):
        """
        Save successful COFs as CIF, run PMTransformer, update cache.

        Args:
            success_combos: list of dicts with COF info from env
            n2_cache: dict {cof_name: n2_value} to update in-place

        Returns:
            Number of new predictions added
        """
        # Find unpredicted COFs
        new_combos = [c for c in success_combos
                      if c['name'] not in self.predicted_names]

        if len(new_combos) < 10:
            return 0  # don't bother for <10 new COFs

        self.call_count += 1
        ts = f"symtopo_{self.call_count:04d}"

        # Step 1: Copy pre-saved CIFs from cache (env saved them during assembly)
        # Skip problematic CIFs gracefully
        saved_names = []
        import shutil
        for combo in new_combos:
            cif_path = combo.get('cif_path')
            if not cif_path or not os.path.exists(cif_path):
                continue
            # Skip CIFs with names that might break PMTransformer
            name = combo['name']
            if len(name) > 100 or '///' in name:
                continue
            try:
                dst = os.path.join(self.cif_dir, os.path.basename(cif_path))
                shutil.copy(cif_path, dst)
                saved_names.append(name)
                # NOTE: do NOT add to predicted_names yet —
                # only after successful prediction below
            except Exception:
                continue

        if len(saved_names) < 5:
            return 0

        # Step 2: Write topology.json (required by PredictDataset)
        import json
        topo_map = {}
        for combo in new_combos:
            if combo['name'] in saved_names:
                topo = combo['topo']
                # Map COF topologies to PMTransformer topology2D.json keys
                topo_lower = topo.lower()
                if topo in ('HCB', 'HCB_A'): topo_map[combo['name']] = 'hcb'
                elif topo in ('SQL', 'SQL_A'): topo_map[combo['name']] = 'sql'
                elif topo == 'KGD': topo_map[combo['name']] = 'kgd-a'
                elif topo == 'HXL_A': topo_map[combo['name']] = 'hxl'
                elif 'dia' in topo_lower: topo_map[combo['name']] = 'dia'
                elif 'lon' in topo_lower: topo_map[combo['name']] = 'lon'
                elif 'fxt' in topo_lower: topo_map[combo['name']] = 'fxt-a'
                else: topo_map[combo['name']] = 'unknown'  # PMTransformer fallback

        topo_path = os.path.join(os.path.dirname(self.cif_dir), 'topology.json')
        with open(topo_path, 'w') as f:
            json.dump(topo_map, f, indent=4)

        # Step 3: Run PMTransformer
        print(f"  [Predictor] Running PMTransformer on {len(saved_names)} COFs...", flush=True)
        result_path = None
        try:
            sys.path.insert(0, '/home/tianyajun/MARL_for_COFs')
            from cof_predictor.main import doPredict
            result_path = doPredict(
                cifPath=self.cif_dir,
                logPath=self.log_dir,
                ts=ts,
            )
        except Exception as e:
            print(f"  [Predictor] PMTransformer error: {e}", flush=True)
            # Even if doPredict fails, try to find the CSV it might have written
            import glob as _glob
            candidates = sorted(_glob.glob(
                os.path.join(os.path.dirname(self.cif_dir), f"n2a_{ts}.csv")))
            if candidates:
                result_path = candidates[0]
                print(f"  [Predictor] Found existing CSV: {result_path}", flush=True)
            else:
                return 0

        # Step 4: Read results and update cache
        import csv
        new_count = 0
        try:
            with open(result_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get('cifName', '').strip()
                    if not name:
                        continue
                    pred_str = row.get('pred', '0').strip()
                    if pred_str == '':
                        continue
                    try:
                        pred = float(pred_str)
                    except ValueError:
                        continue
                    if name not in n2_cache:
                        n2_cache[name] = pred
                        new_count += 1
                    # Mark as predicted only on success
                    self.predicted_names.add(name)
        except Exception as e:
            print(f"  [Predictor] CSV read error: {e}", flush=True)
            return 0

        # Cleanup CIFs to save disk space
        import shutil
        for f in os.listdir(self.cif_dir):
            if f.endswith('.cif'):
                os.remove(os.path.join(self.cif_dir, f))
        if os.path.exists(topo_path):
            os.remove(topo_path)

        print(f"  [Predictor] Done: {new_count} new N2 predictions (total: {len(self.predicted_names)})",
              flush=True)
        return new_count


class RunnerSymTopo:
    def __init__(self, args):
        self.args = args
        np.random.seed(args.seed); torch.manual_seed(args.seed)

        topo_mode = getattr(args, 'topo_mode', 'all')
        self.env = SymTopoEnv(topo_mode=topo_mode)
        args.N = self.env.n
        args.obs_dim = self.env.observation_space  # 80
        args.state_dim = args.obs_dim * args.N     # 300
        if not hasattr(args, 'predictor_freq'):
            args.predictor_freq = 20
        args.episode_limit = self.env.episode_limit
        self.total_steps = 0

        n_topos = self.env._n_topos
        topo_list = self.env._topo_list

        print(f"=== SymTopo MAPPO ({topo_mode.upper()} topologies) ===", flush=True)
        print(f"N={args.N} obs={args.obs_dim} state={args.state_dim}", flush=True)
        print(f"Connectors: {ALL_CONNECTORS} ({N_CONNECTORS})", flush=True)
        print(f"Agent1/2 heads: sym({N_SYMS}) core({self.env.max_cores}) conn({N_CONNECTORS})", flush=True)
        print(f"Agent3 head: topo({n_topos}) = {topo_list}", flush=True)

        self.actor_12 = MultiHeadActor12(
            args.obs_dim, N_SYMS, self.env.max_cores, N_CONNECTORS, args.mlp_hidden_dim)
        self.actor_3 = TopoActor(args.obs_dim, n_topos, args.mlp_hidden_dim)
        self.critic = Critic(args.state_dim, args.mlp_hidden_dim)

        if args.use_orthogonal_init:
            for m in [self.actor_12, self.actor_3, self.critic]:
                if hasattr(m, 'net'):
                    layers = list(m.net)
                else:
                    layers = list(m.shared) + [m.sym_head, m.core_head, m.conn_head]
                # Use gain=1.0 for hidden layers, 0.01 for output heads
                for i, layer in enumerate(layers):
                    if hasattr(layer, 'weight'):
                        is_final = (i == len(layers) - 1)
                        orthogonal_init(layer, gain=0.01 if is_final else 1.0)

        self.opt_12 = torch.optim.Adam(self.actor_12.parameters(), lr=args.lr)
        self.opt_3 = torch.optim.Adam(self.actor_3.parameters(), lr=args.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=args.lr * 2)
        self.rnd = RND(args.obs_dim, 32, 256)
        self.replay_buffer = ReplayBuffer(args)
        self.predictor = N2Predictor()

    def choose_action(self, obs_n, masks, agent_id, evaluate=False):
        obs_t = torch.FloatTensor(obs_n[agent_id]).unsqueeze(0)
        if agent_id in [0, 1]:
            with torch.no_grad():
                sym_mask = torch.FloatTensor(masks[agent_id]['sym']).unsqueeze(0)
                core_mask = torch.FloatTensor(masks[agent_id]['core']).unsqueeze(0)
                conn_mask = torch.FloatTensor(masks[agent_id]['conn']).unsqueeze(0)
                sym_a, core_a, conn_a = self.actor_12.get_action(
                    obs_t, sym_mask, core_mask, conn_mask)
            return (sym_a.item(), core_a.item(), conn_a.item())
        else:
            with torch.no_grad():
                topo_mask = torch.FloatTensor(masks[agent_id]['topo']).unsqueeze(0)
                topo_a = self.actor_3.get_action(obs_t, topo_mask)
            return topo_a.item()

    def _encode_action_12(self, sym, core, conn):
        """Encode (sym, core, conn) into a single integer for ReplayBuffer storage."""
        C = self.env.max_cores
        return sym * C * N_CONNECTORS + core * N_CONNECTORS + conn

    def _decode_action_12(self, code):
        """Decode back to (sym, core, conn)."""
        C = self.env.max_cores
        sym = code // (C * N_CONNECTORS)
        rest = code % (C * N_CONNECTORS)
        core = rest // N_CONNECTORS
        conn = rest % N_CONNECTORS
        return sym, core, conn

    def run_episode(self, evaluate=False):
        obs_n, info, masks = self.env.reset()
        episode_reward = 0.0

        for step in range(self.args.episode_limit):
            agent_id = step
            a = self.choose_action(obs_n, masks, agent_id, evaluate)

            # Build actions list for env
            if agent_id == 0:
                actions = [a, (0, 0, 0), 0]
                # Encode for replay buffer: (sym, core, conn) → flat int
                a_store = np.array([
                    self._encode_action_12(*a),  # Agent 1's action
                    0,                            # Agent 2 (not yet)
                    0                             # Agent 3 (not yet)
                ])
            elif agent_id == 1:
                actions = [(0, 0, 0), a, 0]
                a_store = np.array([0, self._encode_action_12(*a), 0])
            else:
                actions = [(0, 0, 0), (0, 0, 0), a]
                a_store = np.array([0, 0, a])  # Agent 3: flat topo index

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
                            np.zeros(self.args.N), np.zeros(self.args.N),
                            np.zeros(self.args.N), np.ones(self.args.N))
                    self.replay_buffer.store_last_value(step + 1, np.array([v_last] * self.args.N))
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

                # Agent 1/2 loss (steps 0,1)
                for t in [0, 1]:
                    sv = mb_valid[:, t]
                    if sv.sum() == 0: continue
                    for ag in [0, 1]:
                        o = mb_obs[:, t, ag, :][sv]
                        adv_t = mb_adv[:, t][sv]
                        if len(o) == 0: continue

                        # Decode multi-part action
                        a_codes = mb_a[:, t, ag][sv].long()
                        sym_a = torch.zeros(len(o), dtype=torch.long)
                        core_a = torch.zeros(len(o), dtype=torch.long)
                        conn_a = torch.zeros(len(o), dtype=torch.long)
                        for i, code in enumerate(a_codes):
                            s, c, cn = self._decode_action_12(int(code.item()))
                            sym_a[i] = s; core_a[i] = c; conn_a[i] = cn

                        sym_mask = torch.ones(len(o), N_SYMS)
                        core_mask = torch.ones(len(o), self.env.max_cores)
                        conn_mask = torch.ones(len(o), N_CONNECTORS)

                        try:
                            lp, ent = self.actor_12.evaluate(
                                o, sym_a, core_a, conn_a,
                                sym_mask, core_mask, conn_mask)
                        except:
                            continue
                        ratio = torch.exp(lp - lp.detach())
                        s1 = ratio * adv_t; s2 = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon) * adv_t
                        loss_12 = loss_12 - torch.min(s1, s2).mean() - self.args.entropy_coef * ent

                # Agent 3 loss (higher entropy coef for topology exploration)
                if mb_obs.size(1) > 2:
                    sv2 = mb_valid[:, 2]
                    if sv2.sum() > 0:
                        o3 = mb_obs[:, 2, 2, :][sv2]
                        adv3 = mb_adv[:, 2][sv2]
                        if len(o3) > 0:
                            a3 = mb_a[:, 2, 2][sv2].long()
                            topo_mask = torch.ones(len(o3), self.env._n_topos)
                            lp3, ent3 = self.actor_3.evaluate(o3, a3, topo_mask)
                            ratio3 = torch.exp(lp3 - lp3.detach())
                            s13 = ratio3 * adv3; s23 = torch.clamp(ratio3, 1 - self.args.epsilon, 1 + self.args.epsilon) * adv3
                            # Decoupled Ag3: higher entropy coefficient → natural topology exploration
                            loss_3 = loss_3 - torch.min(s13, s23).mean() - self.args.entropy_coef_3 * ent3

                # Critic
                s_mb = mb_obs.reshape(mb_obs.size(0), mb_obs.size(1), -1)
                s_val = s_mb[mb_valid]; t_val = mb_target[mb_valid]
                critic_loss = ((self.critic(s_val).squeeze() - t_val)**2).mean() if len(s_val) > 0 else torch.tensor(0.0)

                loss = loss_12 + loss_3 + critic_loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue  # skip this mini-batch if NaN/Inf
                total_loss += loss.item()
                self.opt_12.zero_grad(); self.opt_3.zero_grad(); self.opt_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_12.parameters()) + list(self.actor_3.parameters()) +
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

            # Print every batch; detailed z-stats every 10 batches
            avg_r = np.mean(ep_rewards)
            succ = self.env.assembly_success / max(self.env.assembly_total, 1) * 100
            td = {k: v for k, v in sorted(self.env.topo_counts.items(), key=lambda x: -x[1])[:5]}
            cd = dict(self.env.conn_pair_counts.most_common(3))
            if batch_idx % 10 == 0:
                zs = {t: f"{s['n']}/{s['mean']:.2f}" for t, s in
                      sorted(self.env.topo_n2_stats.items(), key=lambda x: -x[1]['n'])[:5]
                      if s['n'] > 0}
                print(f"B{batch_idx:4d} S{self.total_steps:6d} | R:{avg_r:5.1f} Succ:{succ:4.0f}% | "
                      f"Loss:{loss if isinstance(loss, float) else 0:.2f} | "
                      f"N2cache:{len(self.env.n2_cache)} | Topo:{td} | Conn:{cd} | "
                      f"Z(n/μ):{zs}", flush=True)
            else:
                print(f"B{batch_idx:4d} S{self.total_steps:6d} | R:{avg_r:5.1f} Succ:{succ:4.0f}% | "
                      f"Loss:{loss if isinstance(loss, float) else 0:.2f} | "
                      f"Topo:{td} | Conn:{cd}", flush=True)

            # Dynamic predictor: every K batches, predict N2 for new successful COFs
            if batch_idx % self.args.predictor_freq == 0 and len(self.env.success_combos) > 0:
                self.predictor.run(self.env.success_combos, self.env.n2_cache)
                self.env.success_combos = []  # clear after prediction

            if batch_idx % 50 == 0:
                d = '/home/tianyajun/MARL_for_COFs/mappo/model'
                torch.save(self.actor_12.state_dict(), f'{d}/symtopo_a12_s{self.total_steps}.pth')
                torch.save(self.actor_3.state_dict(), f'{d}/symtopo_a3_s{self.total_steps}.pth')

        print(f"Done. Topo: {dict(self.env.topo_counts)}")
        print(f"Conn pairs: {dict(self.env.conn_pair_counts)}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--max_train_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--mini_batch_size", type=int, default=4)
    p.add_argument("--mlp_hidden_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lamda", type=float, default=0.95)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--K_epochs", type=int, default=4)
    p.add_argument("--entropy_coef", type=float, default=0.05,
                   help="Entropy coef for Ag1/2")
    p.add_argument("--entropy_coef_3", type=float, default=0.20,
                   help="Entropy coef for Ag3 (higher = more topology exploration)")
    p.add_argument("--predictor_freq", type=int, default=20,
                   help="Run PMTransformer every N batches (0=disable)")
    p.add_argument("--rnd_beta", type=float, default=0.3)
    p.add_argument("--use_adv_norm", type=bool, default=True)
    p.add_argument("--use_orthogonal_init", type=bool, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topo_mode", type=str, default="all",
                   choices=["all", "2d", "3d"],
                   help="Topology class: all, 2d (HCB_A/KGD/FXT_A/SQL_A/HXL_A), 3d (DIA/DIA_A/LON/LON_A)")
    args = p.parse_args()
    RunnerSymTopo(args).run()
