"""
Symmetry+Topology+Connector COF design environment — MAPPO interface.

3 agents, 3-step sequential decision:
  Step 1: Agent 1 selects (symmetry, core_template, connector)
  Step 2: Agent 2 sees Agent 1's connector → compatible connectors masked
           → selects (symmetry, core_template, connector)
           → invalid connector pair → -1, episode ends
  Step 3: Agent 3 sees both symmetries → topology (masked)
          → assembly with chosen connectors → reward

Valid connector reaction pairs (pycofbuilder):
  NH2↔CHO, NHOH↔CHO, CH2CN↔CHO, COOH↔NH2, OHc↔NH2, Cl↔Cl
"""
import os, sys, json, io, re
import numpy as np
from collections import Counter

# ── Constants ──
SYMMETRY_TYPES = ["L2", "T3", "S4", "H6", "D4", "R4"]
N_SYMS = len(SYMMETRY_TYPES)
SYMMETRY_TO_CONNECTORS_COUNT = {"L2": 2, "T3": 3, "S4": 4, "H6": 6, "D4": 4, "R4": 4}

SYMMETRY_PAIR_TO_TOPOLOGY = {
    # (2,3) and (3,2): trigonal + linear → 2D honeycomb/kagome
    (2, 3): ["HCB_A", "KGD", "FXT_A"], (3, 2): ["HCB_A", "KGD", "FXT_A"],
    # (2,4) and (4,2): tetrahedral + linear → 3D (DIA_A/LON_A) or 2D (SQL_A)
    (2, 4): ["SQL_A", "DIA_A", "LON_A"], (4, 2): ["SQL_A", "DIA_A", "LON_A"],
    # (2,6): hexagonal + linear → HXL_A
    (2, 6): ["HXL_A"], (6, 2): ["HXL_A"],
    # (3,3): trigonal + trigonal → KGD
    (3, 3): ["KGD"],
    # (4,4): tetrahedral + tetrahedral → 3D DIA/LON (stacking "1")
    (4, 4): ["DIA", "LON"],
    # NOTE: (2,2) removed — pycofbuilder has no topology for 2+2 connectivity
    # NOTE: BOR removed — pycofbuilder raises NotImplementedError
    # NOTE: KGM_A removed — does not exist in pycofbuilder
}
TOPOLOGIES = sorted(set(t for tops in SYMMETRY_PAIR_TO_TOPOLOGY.values() for t in tops))
N_TOPOS = len(TOPOLOGIES)

# 2D vs 3D classification (by stacking: "AA"=2D, "1"=3D)
TOPO_2D = {"HCB_A", "KGD", "FXT_A", "SQL_A", "HXL_A"}
TOPO_3D = {"DIA", "DIA_A", "LON", "LON_A"}

def _filter_topologies(mode):
    """Return filtered topology table and list for 2D/3D mode."""
    if mode == 'all':
        return SYMMETRY_PAIR_TO_TOPOLOGY, TOPOLOGIES, N_TOPOS
    allowed = TOPO_2D if mode == '2d' else TOPO_3D
    filtered_table = {}
    for pair, topos in SYMMETRY_PAIR_TO_TOPOLOGY.items():
        valid = [t for t in topos if t in allowed]
        if valid:
            filtered_table[pair] = valid
    filtered_list = sorted(set(t for tops in filtered_table.values() for t in tops))
    return filtered_table, filtered_list, len(filtered_list)

# All 15 pycofbuilder connectors
ALL_CONNECTORS = [
    "BOH2", "Br", "CCH3O", "CH2CN", "CH3", "CHNNH2", "CHO", "Cl",
    "COCHCHOH", "CONHNH2", "COOH", "NH2", "NHOH", "O", "OH2",
]
N_CONNECTORS = len(ALL_CONNECTORS)

# Chemistry-informed valid reaction pairs (COF literature):
#   Imine: CHO ↔ NH2/NHOH/CHNNH2/CONHNH2/CH2CN
#   Amide: COOH ↔ NH2/NHOH/CONHNH2
#   Knoevenagel: CH2CN ↔ CHO
#   Boronate ester: BOH2 ↔ OH2
#   Halogen coupling: Cl ↔ Cl/Br,  Br ↔ Br/Cl
#   Non-reactive: CH3, O, CCH3O, COCHCHOH (rare/special conditions)
COMPATIBLE_CONNECTORS = {
    "CHO":      {"NH2", "NHOH", "CHNNH2", "CONHNH2", "CH2CN"},
    "COOH":     {"NH2", "NHOH", "CONHNH2"},
    "COCHCHOH": {"NH2", "NHOH"},
    "CH2CN":    {"CHO"},
    "BOH2":     {"OH2"},
    "OH2":      {"BOH2"},
    "Cl":       {"Cl", "Br"},
    "Br":       {"Br", "Cl"},
    # NH2, NHOH, CHNNH2, CONHNH2: compatible with CHO, COOH (symmetric in dict above)
    # CH3, O, CCH3O: non-reactive — no known COF linkage
}

def _get_compatible(conn):
    """Return set of connectors compatible with the given connector."""
    # Direct match (conn as first partner)
    compat = COMPATIBLE_CONNECTORS.get(conn, set()).copy()
    # Reverse match (conn as second partner)
    for c1, c2s in COMPATIBLE_CONNECTORS.items():
        if conn in c2s:
            compat.add(c1)
    return compat


class SymTopoEnv:
    """3-agent symmetry+core+connector+topology COF design env for MAPPO."""

    def __init__(self, core_dir=None, n2_lookup_path=None, topo_mode='all'):
        if core_dir is None:
            core_dir = '/home/tianyajun/MARL_for_COFs/symmcd_diffusion/data/core'

        # Topology mode: filter 2D/3D topologies
        self.topo_mode = topo_mode
        self._topo_table, self._topo_list, self._n_topos = _filter_topologies(topo_mode)

        # Load Core templates per symmetry
        self.core_names = {}
        self.core_features = {}
        for sym in SYMMETRY_TYPES:
            d = os.path.join(core_dir, sym)
            if os.path.exists(d):
                names = sorted([f.replace('.cjson', '') for f in os.listdir(d) if f.endswith('.cjson')])
                feats = []
                for fn in sorted(os.listdir(d)):
                    if fn.endswith('.cjson'):
                        with open(os.path.join(d, fn)) as f:
                            data = json.load(f)
                        elements = data['atoms']['elements']['type']
                        coords = np.array(data['atoms']['coords']['3d']).reshape(-1, 3)
                        ec = Counter(elements)
                        extent = (coords.max(axis=0) - coords.min(axis=0)) if len(elements) > 1 else np.zeros(3)
                        feats.append({
                            'n_atoms': len(elements),
                            'n_Q': ec.get('Q', 0),
                            'n_R': sum(1 for e in elements if e.startswith('R')),
                            'elem_C': ec.get('C', 0),
                            'elem_N': ec.get('N', 0),
                            'max_extent': float(max(extent)),
                        })
                self.core_names[sym] = names
                self.core_features[sym] = feats
            else:
                self.core_names[sym] = []
                self.core_features[sym] = []

        self.max_cores = max(len(v) for v in self.core_names.values())

        # Valid cores filter
        vc_path = '/home/tianyajun/MARL_for_COFs/symmcd_diffusion/generated/valid_cores.json'
        self.valid_cores = json.load(open(vc_path)) if os.path.exists(vc_path) else self.core_names

        # N2 prediction cache (populated dynamically by predictor, NOT static lookup)
        self.n2_cache = {}

        # CIF cache: save during assembly, skip re-assembly in predictor (10x speedup)
        self.cif_cache_dir = '/tmp/symtopo_cif_cache'
        os.makedirs(self.cif_cache_dir, exist_ok=True)

        # Per-topology N2 running stats for z-score normalization (Welford)
        self.topo_n2_stats = {t: {'n': 0, 'mean': 0.0, 'M2': 0.0} for t in self._topo_list}

        # MAPPO interface
        self.n = 3
        self.observation_space = 100  # 3+3+12+6+15+15+5+15+9+3+2+pad
        # Action dims per agent type (used by training script to build actor heads)
        self.action_dim_sym = N_SYMS           # 6
        self.action_dim_core = self.max_cores   # ~41
        self.action_dim_conn = N_CONNECTORS     # 7
        self.action_dim_topo = self._n_topos    # 9(all), 5(2D), 4(3D)
        self.action_space = max(N_SYMS, self.max_cores, N_CONNECTORS, N_TOPOS)
        self.episode_limit = 3

        # Stats
        self.assembly_success = 0
        self.assembly_total = 0
        self.topo_counts = {t: 0 for t in self._topo_list}
        self.conn_pair_counts = Counter()
        self.combo_counts = {}       # (sym_a, sym_b, conn_a, conn_b, topo) → count
        self.success_combos = []     # successful COF records for dynamic predictor

        # Episode state
        self._step_idx = 0
        self._sym_a = None
        self._sym_b = None
        self._core_a = None
        self._core_b = None
        self._conn_a = None
        self._conn_b = None

    # ── State builder ──────────────────────────────────────────────────────

    def _build_obs(self, agent_id):
        obs = np.zeros(self.observation_space, dtype=np.float32)
        offset = 0

        # Step indicator (3)
        obs[offset + self._step_idx] = 1.0; offset += 3
        # Agent ID (3)
        obs[offset + agent_id] = 1.0; offset += 3
        # Partner symmetry (12 = 6+6 for two partner slots)
        if agent_id == 1 and self._sym_a is not None:
            obs[offset + self._sym_a] = 1.0
        elif agent_id == 0 and self._sym_b is not None:
            obs[offset + self._sym_b] = 1.0
        elif agent_id == 2:
            if self._sym_a is not None: obs[offset + self._sym_a] = 1.0
            if self._sym_b is not None: obs[offset + 6 + self._sym_b] = 1.0
        offset += 12

        # Self symmetry (6)
        if agent_id == 0 and self._sym_a is not None:
            obs[offset + self._sym_a] = 1.0
        elif agent_id == 1 and self._sym_b is not None:
            obs[offset + self._sym_b] = 1.0
        offset += 6

        # Partner connector one-hot (7)
        if agent_id == 1 and self._conn_a is not None:
            if self._conn_a in ALL_CONNECTORS:
                obs[offset + ALL_CONNECTORS.index(self._conn_a)] = 1.0
        elif agent_id == 0 and self._conn_b is not None:
            if self._conn_b in ALL_CONNECTORS:
                obs[offset + ALL_CONNECTORS.index(self._conn_b)] = 1.0
        elif agent_id == 2:
            if self._conn_a is not None and self._conn_a in ALL_CONNECTORS:
                obs[offset + ALL_CONNECTORS.index(self._conn_a)] = 1.0
            if self._conn_b is not None and self._conn_b in ALL_CONNECTORS:
                obs[offset + 7 + ALL_CONNECTORS.index(self._conn_b)] = 1.0
        offset += 14  # 7+7 partner connectors

        # Core features (5)
        MAX_A, MAX_E, MAX_X = 200.0, 150.0, 40.0
        feats = None
        if agent_id == 0 and self._sym_a is not None and self._core_a is not None:
            feats = self.core_features.get(SYMMETRY_TYPES[self._sym_a], [])
            if feats and self._core_a < len(feats): feats = feats[self._core_a]
        elif agent_id == 1 and self._sym_b is not None and self._core_b is not None:
            feats = self.core_features.get(SYMMETRY_TYPES[self._sym_b], [])
            if feats and self._core_b < len(feats): feats = feats[self._core_b]
        if feats:
            obs[offset] = min(feats['n_atoms'] / MAX_A, 1.0)
            obs[offset+1] = min(feats['n_Q'] / 20.0, 1.0)
            obs[offset+2] = min(feats['elem_C'] / MAX_E, 1.0)
            obs[offset+3] = min(feats['elem_N'] / MAX_E, 1.0)
            obs[offset+4] = min(feats['max_extent'] / MAX_X, 1.0)
        offset += 5

        # Connector compatibility mask for Agent 2 (chemistry-informed)
        if agent_id == 1 and self._conn_a is not None:
            compat = _get_compatible(self._conn_a)
            for c in compat:
                if c in ALL_CONNECTORS:
                    obs[offset + ALL_CONNECTORS.index(c)] = 1.0
        offset += N_CONNECTORS

        # Topology mask for Agent 3 (9)
        if agent_id == 2 and self._sym_a is not None and self._sym_b is not None:
            ca = SYMMETRY_TO_CONNECTORS_COUNT.get(SYMMETRY_TYPES[self._sym_a], 2)
            cb = SYMMETRY_TO_CONNECTORS_COUNT.get(SYMMETRY_TYPES[self._sym_b], 2)
            valid = SYMMETRY_PAIR_TO_TOPOLOGY.get((ca, cb), [])
            for t in valid:
                if t in TOPOLOGIES:
                    obs[offset + TOPOLOGIES.index(t)] = 1.0
        offset += self._n_topos

        # Diversity stats (3)
        total = sum(self.topo_counts.values()) + 1
        obs[offset] = min(total / 200.0, 1.0)
        obs[offset+1] = self.assembly_success / max(self.assembly_total, 1)
        offset += 3

        # Connection counts (2)
        if self._sym_a is not None:
            obs[offset] = SYMMETRY_TO_CONNECTORS_COUNT.get(SYMMETRY_TYPES[self._sym_a], 2) / 6.0
        if self._sym_b is not None:
            obs[offset+1] = SYMMETRY_TO_CONNECTORS_COUNT.get(SYMMETRY_TYPES[self._sym_b], 2) / 6.0
        # offset += 2  -- fills to 80

        return obs

    # ── Action masks ───────────────────────────────────────────────────────

    def build_sym_mask(self, partner_sym=None):
        """Sym mask: Agent1 sees all, Agent2 sees only compatible with partner's connectivity."""
        mask = np.zeros(N_SYMS, dtype=np.float32)
        if partner_sym is None:
            mask[:] = 1.0  # Agent 1: all symmetries available
        else:
            # Agent 2: only symmetries whose connector count + partner's form valid topology
            # Use instance-level filtered table (respects 2D/3D mode)
            p_conn = SYMMETRY_TO_CONNECTORS_COUNT.get(SYMMETRY_TYPES[partner_sym], 2)
            for si, sym in enumerate(SYMMETRY_TYPES):
                s_conn = SYMMETRY_TO_CONNECTORS_COUNT.get(sym, 2)
                if (p_conn, s_conn) in self._topo_table:
                    mask[si] = 1.0
        return mask

    def build_core_mask(self, sym_idx):
        mask = np.zeros(self.max_cores, dtype=np.float32)
        sym = SYMMETRY_TYPES[sym_idx]
        mask[:len(self.core_names.get(sym, []))] = 1.0
        return mask

    def build_conn_mask(self, partner_conn=None):
        """Connector mask: Agent1 sees reactive connectors, Agent2 sees compatible ones."""
        mask = np.zeros(N_CONNECTORS, dtype=np.float32)
        if partner_conn is None:
            # Agent 1: only connectors that have at least 1 compatible partner
            for ci, conn in enumerate(ALL_CONNECTORS):
                if len(_get_compatible(conn)) > 0:
                    mask[ci] = 1.0
        else:
            # Agent 2: only connectors compatible with Agent 1's choice
            compat = _get_compatible(partner_conn)
            if len(compat) == 0:
                return mask  # all zeros → episode should not reach here (Agent 1 can't pick this)
            for c in compat:
                if c in ALL_CONNECTORS:
                    mask[ALL_CONNECTORS.index(c)] = 1.0
        return mask

    def build_topo_mask(self, sym_a, sym_b):
        mask = np.zeros(self._n_topos, dtype=np.float32)
        if sym_a is not None and sym_b is not None:
            ca = SYMMETRY_TO_CONNECTORS_COUNT.get(SYMMETRY_TYPES[sym_a], 2)
            cb = SYMMETRY_TO_CONNECTORS_COUNT.get(SYMMETRY_TYPES[sym_b], 2)
            valid = self._topo_table.get((ca, cb), [])
            for t in valid:
                if t in self._topo_list:
                    mask[self._topo_list.index(t)] = 1.0
        else:
            mask[:] = 1.0
        return mask

    def get_valid_topologies(self, sym_a, sym_b):
        ca = SYMMETRY_TO_CONNECTORS_COUNT.get(sym_a, 2)
        cb = SYMMETRY_TO_CONNECTORS_COUNT.get(sym_b, 2)
        return self._topo_table.get((ca, cb), [self._topo_list[0]] if self._topo_list else ["HCB_A"])

    # ── Gym-style interface ────────────────────────────────────────────────

    def reset(self):
        self._step_idx = 0
        self._sym_a = self._sym_b = None
        self._core_a = self._core_b = None
        self._conn_a = self._conn_b = None

        obs_n = np.array([self._build_obs(i) for i in range(self.n)])
        info = ['init'] * self.n
        # Return per-head masks for each agent
        action_mask = self._get_masks()
        return obs_n, info, action_mask

    def _get_masks(self):
        """Return action masks per agent. Each is a dict of per-head masks."""
        masks = [{}, {}, {}]

        # Agent 1 (step 0)
        masks[0]['sym'] = self.build_sym_mask()
        masks[0]['core'] = self.build_core_mask(0)  # placeholder, any sym OK
        masks[0]['conn'] = self.build_conn_mask(None)

        # Agent 2 (step 1) — sym masked by partner connectivity
        masks[1]['sym'] = self.build_sym_mask(partner_sym=self._sym_a)
        masks[1]['core'] = self.build_core_mask(0)
        masks[1]['conn'] = self.build_conn_mask(partner_conn=self._conn_a)

        # Agent 3 (step 2)
        masks[2]['topo'] = self.build_topo_mask(self._sym_a, self._sym_b)

        return masks

    def step(self, actions):
        """
        Execute one step. actions = [(sym, core, conn), (sym, core, conn), topo]
        Each entry is the raw index for that agent's active head(s).
        """
        self._step_idx += 1

        if self._step_idx == 1:
            # Agent 1: (sym, core, conn)
            sym_idx, core_idx, conn_idx = actions[0]
            sym_idx = sym_idx % N_SYMS
            core_idx = core_idx % max(len(self.core_names.get(SYMMETRY_TYPES[sym_idx], ['default'])), 1)
            conn_idx = conn_idx % N_CONNECTORS

            self._sym_a = sym_idx
            self._core_a = core_idx
            self._conn_a = ALL_CONNECTORS[conn_idx]

            obs_n = np.array([self._build_obs(i) for i in range(self.n)])
            masks = self._get_masks()
            return obs_n, [0.0]*self.n, [False]*self.n, \
                   [f'sym={SYMMETRY_TYPES[sym_idx]},conn={self._conn_a}']*self.n, masks

        elif self._step_idx == 2:
            # Agent 2: (sym, core, conn)
            sym_idx, core_idx, conn_idx = actions[1]
            sym_idx = sym_idx % N_SYMS
            core_idx = core_idx % max(len(self.core_names.get(SYMMETRY_TYPES[sym_idx], ['default'])), 1)
            conn_idx = conn_idx % N_CONNECTORS
            conn_b = ALL_CONNECTORS[conn_idx]

            # Validate sym pair (connector pair validated implicitly by assembly)
            ca = SYMMETRY_TO_CONNECTORS_COUNT.get(SYMMETRY_TYPES[self._sym_a], 2)
            cb = SYMMETRY_TO_CONNECTORS_COUNT.get(SYMMETRY_TYPES[sym_idx], 2)
            if (ca, cb) not in self._topo_table:
                self.assembly_total += 1
                obs_n = np.zeros((self.n, self.observation_space), dtype=np.float32)
                masks = [{}, {}, {}]
                return obs_n, [-1.0, -1.0, -1.0], [True]*self.n, \
                       [f'invalid_sym_pair ({ca},{cb})']*self.n, masks

            self._sym_b = sym_idx
            self._core_b = core_idx
            self._conn_b = conn_b

            obs_n = np.array([self._build_obs(i) for i in range(self.n)])
            masks = self._get_masks()
            return obs_n, [0.0]*self.n, [False]*self.n, \
                   [f'sym={SYMMETRY_TYPES[sym_idx]},conn={conn_b}']*self.n, masks

        else:  # step == 3
            # Agent 3: topology
            topo_idx = actions[2] if isinstance(actions[2], int) else actions[2][0]
            sym_a_name = SYMMETRY_TYPES[self._sym_a]
            sym_b_name = SYMMETRY_TYPES[self._sym_b]
            valid_topos = self.get_valid_topologies(sym_a_name, sym_b_name)
            topo = valid_topos[topo_idx % len(valid_topos)]

            cores_a = self.valid_cores.get(sym_a_name, self.core_names.get(sym_a_name, ['default']))
            cores_b = self.valid_cores.get(sym_b_name, self.core_names.get(sym_b_name, ['default']))
            core_a = cores_a[self._core_a % len(cores_a)] if cores_a else 'default'
            core_b = cores_b[self._core_b % len(cores_b)] if cores_b else 'default'

            # ── Assembly ──
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = io.StringIO(); sys.stderr = io.StringIO()
            try:
                from pycofbuilder.building_block import BuildingBlock
                from pycofbuilder.framework import Framework
                bb1 = BuildingBlock()
                bb1.from_name(f"{sym_a_name}_{core_a}_{self._conn_a}_H")
                bb2 = BuildingBlock()
                bb2.from_name(f"{sym_b_name}_{core_b}_{self._conn_b}_H")
                fw = Framework(dist_threshold=0.4)
                stacking = "1" if topo in ("DIA", "DIA_A", "LON", "LON_A") else "AA"
                if bb1.connectivity >= bb2.connectivity:
                    fw.from_building_blocks(bb1, bb2, topo, stacking)
                else:
                    fw.from_building_blocks(bb2, bb1, topo, stacking)

                # ── Tier 2 reward (z-score normalized per topology) ──
                # N2: use dynamic prediction cache if available, else heuristic
                if fw.name in self.n2_cache:
                    n2_val = self.n2_cache[fw.name]
                else:
                    cell_vol = (fw.cellParameters[0] * fw.cellParameters[1] *
                                fw.cellParameters[2])
                    density = len(fw.atom_types) / max(cell_vol, 1.0)
                    # Heuristic: lower density (larger pores) → higher N2 capacity
                    n2_val = 1.5 + 0.5 * max(0, (0.015 - density) * 1000)

                # Update per-topology running stats (Welford's online algorithm)
                stats = self.topo_n2_stats[topo]
                stats['n'] += 1
                delta = n2_val - stats['mean']
                stats['mean'] += delta / stats['n']
                delta2 = n2_val - stats['mean']
                stats['M2'] += delta * delta2
                # Numerical safety: M2 can go slightly negative from fp rounding
                stats['M2'] = max(stats['M2'], 0.0)

                # Z-score: normalize within topology so DIA ~ HCB_A ~ LON
                # Min 5 samples warmup; capped at [-3, 3] to prevent outliers
                if stats['n'] >= 5:
                    var = stats['M2'] / max(stats['n'] - 1, 1)
                    std = np.sqrt(max(var, 0.0))
                    z_n2 = (n2_val - stats['mean']) / max(std, 0.01)
                    z_n2 = float(np.clip(z_n2, -3.0, 3.0))
                else:
                    z_n2 = 0.0  # warmup: z=0, exploration bonuses drive diversity

                # UCB diversity bonus: sqrt(log(total)/count) — decays with count
                total = sum(self.topo_counts.values()) + 1
                combo_key = (sym_a_name, sym_b_name, self._conn_a, self._conn_b, topo)
                combo_cnt = self.combo_counts.get(combo_key, 0) + 1
                ucb = np.sqrt(5.0 * np.log(max(total, 1)) / max(combo_cnt, 1))

                # Topology bonus: extra reward for under-sampled topologies
                topo_cnt = self.topo_counts.get(topo, 0) + 1
                topo_bonus = 3.0 * np.log(max(total, 1)) / max(topo_cnt, 1)

                # Shared reward: z-scored N2 (performance) + UCB (diversity) + topo bonus
                reward = z_n2 * 5.0 + ucb * 5.0 + topo_bonus * 5.0
                # Reward range: z_n2∈[-3,3]→[-15,15], ucb∈[0,1.5]→[0,7.5],
                #               topo_bonus∈[0,1.5]→[0,7.5], total≈[-15,30]
                self.assembly_success += 1
                self.topo_counts[topo] += 1
                self.conn_pair_counts[(self._conn_a, self._conn_b)] += 1
                self.combo_counts[combo_key] = combo_cnt

                # Save CIF immediately (avoids re-assembly in predictor → 10x speedup)
                cif_path = os.path.join(self.cif_cache_dir, f"{fw.name}.cif")
                try:
                    fw.save(fmt='cif', supercell=[1, 1, 1], save_dir=self.cif_cache_dir)
                except Exception:
                    cif_path = None  # save failed, predictor will skip

                # Record for dynamic predictor
                self.success_combos.append({
                    'name': fw.name, 'topo': topo,
                    'sym_a': sym_a_name, 'sym_b': sym_b_name,
                    'core_a': core_a, 'core_b': core_b,
                    'conn_a': self._conn_a, 'conn_b': self._conn_b,
                    'n_atoms': len(fw.atom_types), 'sg': fw.space_group,
                    'cif_path': cif_path,
                })
                info_str = (f'success:{topo}:{sym_a_name}+{sym_b_name}:'
                           f'{self._conn_a}+{self._conn_b}:'
                           f'N2={n2_val:.2f},z={z_n2:+.2f},tot={stats["n"]}')
            except Exception as e:
                # ── Tier 1: shaped by failure proximity ──
                err = str(e)
                if 'connectivity' in err.lower():
                    reward = 1.0   # hard fail
                elif 'from_building_blocks' in err.lower():
                    reward = 3.0   # BBs OK, assembly structure issue
                else:
                    reward = 2.0   # other (atom clash, etc.)
                info_str = (f'fail:{topo}:{sym_a_name}+{sym_b_name}:'
                           f'{self._conn_a}+{self._conn_b}:{err[:40]}')
                self.topo_counts[topo] = self.topo_counts.get(topo, 0) + 1
            finally:
                sys.stdout = old_out; sys.stderr = old_err

            self.assembly_total += 1
            rewards = [reward, reward, reward]
            obs_n = np.zeros((self.n, self.observation_space), dtype=np.float32)
            # Normalize reward to [0, 1] for obs; negative→0, max clipped at 40
            obs_n[:, -1] = min(max(reward, 0.0) / 40.0, 1.0)
            masks = [{}, {}, {}]
            return obs_n, rewards, [True]*self.n, [info_str]*self.n, masks

    def close(self):
        pass


# ── Test ──
if __name__ == '__main__':
    env = SymTopoEnv()
    print(f"SymTopoEnv: N={env.n}, obs={env.observation_space}, "
          f"sym={env.action_dim_sym}, core={env.action_dim_core}, "
          f"conn={env.action_dim_conn}, topo={env.action_dim_topo}")
    print(f"Connectors ({N_CONNECTORS}): {ALL_CONNECTORS}")
    print(f"Note: all connector pairs allowed — assembly determines validity")

    obs_n, info, masks = env.reset()
    print(f"\nStep 0 masks: sym={masks[0]['sym'].sum()}, core={masks[0]['core'].sum()}, "
          f"conn={masks[0]['conn'].sum()}")

    # Step 1: Agent 1 picks T3(idx=1), core0, NH2(idx=11)
    obs_n, r, done, info, masks = env.step([(1, 0, 11), (0,0,0), 0])
    print(f"Step 1 (Ag1: T3+NH2): {info[0]}, conn_mask_2={masks[1]['conn'].sum()} (compatible)")

    # Step 2: Agent 2 picks L2(idx=0), core0, CHO(idx=6) — pycofbuilder handles any pair
    obs_n, r, done, info, masks = env.step([(0,0,0), (0, 0, 6), 0])
    print(f"Step 2 (Ag2: L2+CHO): {info[0]}, topo_mask={masks[2]['topo'].sum()}")

    # Step 3: Agent 3 picks topology 0
    obs_n, r, done, info, masks = env.step([(0,0,0), (0,0,0), 0])
    print(f"Step 3: reward={r[0]:.1f}, {info[0]}")

    # Test: any connector pair is allowed → assembly determines success
    env.reset()
    obs_n, r, done, info, _ = env.step([(1, 0, 0), (0,0,0), 0])   # BOH2
    obs_n, r, done, info, _ = env.step([(0,0,0), (0, 0, 0), 0])   # also BOH2
    print(f"\nTest BOH2+BOH2: r={r[0]}, {info[0]} (assembly decides validity)")
    print("env_symtopo ready ✓")
