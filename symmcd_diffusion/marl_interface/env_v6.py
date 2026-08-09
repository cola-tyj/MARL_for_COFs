"""
Phase 4 v6: Rich state + sequential coordination.

Key improvements over v4/v5:
1. State expanded from 16 to 42 dims:
   - Symmetry one-hot (6)
   - Core features: n_atoms, n_Q, n_R, element composition (C/N/O/H), spatial extent (8)
   - Partner info: partner symmetry (6), partner connections (1)
   - Topology mask: valid topologies bitmap (9)
   - Step indicator one-hot (3)
   - Diversity stats: topo count percentile (1)
   - Reward signal (1)

2. Sequential 3-step decision:
   Step 1: Agent 1 selects (sym, core)
   Step 2: Agent 2 sees Agent 1's choice → selects (sym, core)
   Step 3: Agent 3 sees both → selects topology
   → Assembly → Reward
   Done flag only True at step 3.

3. Core features pre-loaded from cjson files.
"""
import os, json, sys, io
from pathlib import Path
import numpy as np
from collections import Counter

SYMMETRY_TYPES = ["L2", "T3", "S4", "H6", "D4", "R4"]
SYMMETRY_TO_CONNECTORS = {"L2": 2, "T3": 3, "S4": 4, "H6": 6, "D4": 4, "R4": 4}

SYMMETRY_PAIR_TO_TOPOLOGY = {
    (2, 3): ["HCB_A", "KGD", "FXT_A"], (3, 2): ["HCB_A", "KGD", "FXT_A"],
    (2, 4): ["SQL_A", "DIA_A", "BOR"], (4, 2): ["SQL_A", "DIA_A", "BOR"],
    (2, 2): ["KGM_A"], (2, 6): ["HXL_A", "LON_A"], (6, 2): ["HXL_A", "LON_A"],
    (3, 3): ["KGD"], (4, 4): ["DIA_A", "BOR"],
}
TOPOLOGIES = sorted(set(t for tops in SYMMETRY_PAIR_TO_TOPOLOGY.values() for t in tops))
N_TOPOS = len(TOPOLOGIES)
N_SYMS = len(SYMMETRY_TYPES)


def _load_core_features(core_dir):
    """Pre-load compact feature vectors for all Cores.

    Returns: dict[symmetry] -> list of feature dicts
    Each feature dict: {name, n_atoms, n_Q, n_R, elem_C, elem_N, elem_O, elem_H,
                         extent_x, extent_y, extent_z}
    """
    features = {}
    for sym in SYMMETRY_TYPES:
        d = os.path.join(core_dir, sym)
        if not os.path.exists(d):
            features[sym] = []
            continue
        sym_feats = []
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.cjson'):
                continue
            with open(os.path.join(d, fn)) as f:
                data = json.load(f)
            elements = data['atoms']['elements']['type']
            coords = np.array(data['atoms']['coords']['3d']).reshape(-1, 3)

            # Count elements
            elem_counts = Counter(elements)
            n_atoms = len(elements)
            n_Q = elem_counts.get('Q', 0)
            n_R = sum(1 for e in elements if e.startswith('R'))

            # Spatial extent
            if n_atoms > 1:
                extent = coords.max(axis=0) - coords.min(axis=0)
            else:
                extent = np.zeros(3)

            sym_feats.append({
                'name': fn.replace('.cjson', ''),
                'n_atoms': n_atoms,
                'n_Q': n_Q,
                'n_R': n_R,
                'elem_C': elem_counts.get('C', 0),
                'elem_N': elem_counts.get('N', 0),
                'elem_O': elem_counts.get('O', 0),
                'elem_H': elem_counts.get('H', 0),
                'extent_x': float(extent[0]),
                'extent_y': float(extent[1]),
                'extent_z': float(extent[2]),
            })
        features[sym] = sym_feats
    return features


# ── Normalization constants (estimated from existing + ReDD Cores) ──
MAX_ATOMS = 200.0
MAX_Q = 20.0
MAX_R = 20.0
MAX_ELEM = 150.0
MAX_EXTENT = 40.0  # Angstroms


class COFDesignEnvV6:
    """Rich-state sequential 3-agent COF design environment."""

    def __init__(self, core_dir=None, n2_lookup_path=None):
        if core_dir is None:
            core_dir = str(Path(__file__).parent.parent / "data" / "core")
        self.core_dir = core_dir

        # Suppress pycofbuilder output during init
        old = sys.stdout; sys.stdout = io.StringIO()
        self.core_features = _load_core_features(core_dir)
        sys.stdout = old
        # Build core name lists (keeping old format for compatibility)
        self.cores = {}
        self.core_names = {}
        for sym in SYMMETRY_TYPES:
            feats = self.core_features.get(sym, [])
            self.cores[sym] = [f['name'] for f in feats]
            self.core_names[sym] = [f['name'] for f in feats]

        self.n2_lookup = {}
        if n2_lookup_path and os.path.exists(n2_lookup_path):
            with open(n2_lookup_path) as f:
                self.n2_lookup = json.load(f)

        # Valid cores filter
        vc_path = str(Path(__file__).parent.parent / "generated" / "valid_cores.json")
        if os.path.exists(vc_path):
            with open(vc_path) as f:
                self.valid_cores = json.load(f)
        else:
            self.valid_cores = self.cores

        self.n_sym = N_SYMS
        self.n_topo = N_TOPOS
        self.state_dim = 42
        self.assembly_success = 0
        self.assembly_total = 0
        self.episode_rewards = []
        self.topo_counts = {t: 0 for t in TOPOLOGIES}
        self.topo_successes = {t: 0 for t in TOPOLOGIES}

        # Episode internal state
        self._step_idx = 0
        self._sym_a = None
        self._sym_b = None
        self._core_a = None
        self._core_b = None

    def get_valid_topologies(self, sym_a, sym_b):
        ca = SYMMETRY_TO_CONNECTORS.get(sym_a, 2)
        cb = SYMMETRY_TO_CONNECTORS.get(sym_b, 2)
        return SYMMETRY_PAIR_TO_TOPOLOGY.get((ca, cb), ["HCB_A"])

    def _build_topology_mask(self, valid_topos):
        """Build a binary mask over all topologies (1=valid)."""
        mask = np.zeros(N_TOPOS, dtype=np.float32)
        for t in valid_topos:
            if t in TOPOLOGIES:
                mask[TOPOLOGIES.index(t)] = 1.0
        return mask

    def _build_state(self, agent_id, sym_self=None, core_idx_self=None,
                     sym_partner=None):
        """Build rich state vector for an agent.

        Args:
            agent_id: 0, 1, or 2
            sym_self: symmetry index selected by this agent (None if not yet)
            core_idx_self: core index selected (None if not yet)
            sym_partner: symmetry of the other agent (None if unknown)
        """
        s = np.zeros(self.state_dim, dtype=np.float32)
        offset = 0

        # 1. Self symmetry one-hot (6 dims)
        if sym_self is not None:
            s[offset + sym_self] = 1.0
        offset += N_SYMS

        # 2. Self Core features (8 dims, normalized)
        if sym_self is not None and core_idx_self is not None:
            feats = self.core_features.get(SYMMETRY_TYPES[sym_self], [])
            if feats and core_idx_self < len(feats):
                f = feats[core_idx_self]
                s[offset + 0] = min(f['n_atoms'] / MAX_ATOMS, 1.0)
                s[offset + 1] = min(f['n_Q'] / MAX_Q, 1.0)
                s[offset + 2] = min(f['n_R'] / MAX_R, 1.0)
                s[offset + 3] = min(f['elem_C'] / MAX_ELEM, 1.0)
                s[offset + 4] = min(f['elem_N'] / MAX_ELEM, 1.0)
                s[offset + 5] = min(f['elem_O'] / MAX_ELEM, 1.0)
                s[offset + 6] = min(f['elem_H'] / MAX_ELEM, 1.0)
                s[offset + 7] = min(max(f['extent_x'], f['extent_y'], f['extent_z']) / MAX_EXTENT, 1.0)
        offset += 8

        # 3. Partner symmetry one-hot (6 dims)
        if sym_partner is not None:
            s[offset + sym_partner] = 1.0
        offset += N_SYMS

        # 4. Partner connections (1 dim, normalized)
        if sym_partner is not None:
            s[offset] = SYMMETRY_TO_CONNECTORS.get(SYMMETRY_TYPES[sym_partner], 2) / 6.0
        offset += 1

        # 5. Topology validity mask (9 dims)
        if sym_self is not None and sym_partner is not None:
            valid = self.get_valid_topologies(
                SYMMETRY_TYPES[sym_self], SYMMETRY_TYPES[sym_partner]
            )
            mask = self._build_topology_mask(valid)
            s[offset:offset + N_TOPOS] = mask
        offset += N_TOPOS

        # 6. Step indicator (3 dims one-hot)
        s[offset + min(agent_id, 2)] = 1.0
        offset += 3

        # 7. Topology diversity stats (1 dim)
        total = sum(self.topo_counts.values()) + 1
        # Average count per topology
        avg_count = total / max(N_TOPOS, 1)
        s[offset] = min(avg_count / 50.0, 1.0)
        offset += 1

        # Total: 6 + 8 + 6 + 1 + 9 + 3 + 1 = 34... missing 8
        # Pad remaining with zeros (reserved for future use)
        # Actually let me recount: 6+8+6+1+9+3+1 = 34, so 42-34 = 8 reserved

        return s

    def reset(self):
        """Reset episode. Returns initial states for all 3 agents."""
        self._step_idx = 0
        self._sym_a = None
        self._sym_b = None
        self._core_a = None
        self._core_b = None

        s1 = self._build_state(0)
        s2 = self._build_state(1)
        s3 = self._build_state(2)
        return s1, s2, s3

    def step(self, action_1, action_2, action_3):
        """Execute one step of the sequential decision process.

        Step 1 (_step_idx=0): Process Agent 1's action, return states for Step 2
        Step 2 (_step_idx=1): Process Agent 2's action, return states for Step 3
        Step 3 (_step_idx=2): Process Agent 3's action, assemble COF, return reward
        """
        if self._step_idx == 0:
            # Agent 1 selects symmetry + core
            sym_idx = action_1[0] % N_SYMS
            core_idx = action_1[1] % max(len(self.core_names.get(SYMMETRY_TYPES[sym_idx], [])), 1)

            self._sym_a = sym_idx
            self._core_a = core_idx
            self._step_idx = 1

            # Build states for step 2: Agent 2 sees Agent 1's choice
            s2 = self._build_state(1, sym_self=None, core_idx_self=None,
                                   sym_partner=sym_idx)
            # Agent 1 & 3 wait (zero reward, not done)
            s1 = self._build_state(0, sym_self=sym_idx, core_idx_self=core_idx)
            s3 = self._build_state(2, sym_partner=sym_idx)
            return (s1, s2, s3), 0.0, False, {'step': 1, 'sym_a': SYMMETRY_TYPES[sym_idx]}

        elif self._step_idx == 1:
            # Agent 2 selects symmetry + core (can see Agent 1's choice)
            sym_idx = action_2[0] % N_SYMS
            core_idx = action_2[1] % max(len(self.core_names.get(SYMMETRY_TYPES[sym_idx], [])), 1)

            # Validate compatibility
            ca = SYMMETRY_TO_CONNECTORS.get(SYMMETRY_TYPES[self._sym_a], 2)
            cb = SYMMETRY_TO_CONNECTORS.get(SYMMETRY_TYPES[sym_idx], 2)
            valid_topos = SYMMETRY_PAIR_TO_TOPOLOGY.get((ca, cb), None)

            if valid_topos is None:
                # Invalid pair → negative reward, episode ends
                self._step_idx = 2
                reward = -1.0
                info = {'success': False, 'tier': 0, 'error': 'invalid_pair',
                        'sym_a': SYMMETRY_TYPES[self._sym_a],
                        'sym_b': SYMMETRY_TYPES[sym_idx]}
                self.assembly_total += 1
                s = np.zeros(self.state_dim, dtype=np.float32)
                return (s.copy(), s.copy(), s.copy()), reward, True, info

            self._sym_b = sym_idx
            self._core_b = core_idx
            self._step_idx = 2

            # Build states for step 3: Agent 3 sees both choices
            s3 = self._build_state(2, sym_self=None, core_idx_self=None,
                                   sym_partner=self._sym_a)
            # Add second partner info
            s3[12 + sym_idx] = 1.0  # Partner 2 symmetry
            s3[18] = SYMMETRY_TO_CONNECTORS.get(SYMMETRY_TYPES[sym_idx], 2) / 6.0
            # Update topology mask
            mask = self._build_topology_mask(valid_topos)
            s3[19:19 + N_TOPOS] = mask

            s1 = self._build_state(0, sym_self=self._sym_a, core_idx_self=self._core_a,
                                   sym_partner=sym_idx)
            s2 = self._build_state(1, sym_self=sym_idx, core_idx_self=core_idx,
                                   sym_partner=self._sym_a)
            return (s1, s2, s3), 0.0, False, {'step': 2, 'sym_a': SYMMETRY_TYPES[self._sym_a],
                                               'sym_b': SYMMETRY_TYPES[sym_idx]}

        else:  # self._step_idx == 2
            # Agent 3 selects topology → assemble COF
            sym_a_name = SYMMETRY_TYPES[self._sym_a]
            sym_b_name = SYMMETRY_TYPES[self._sym_b]
            ca = SYMMETRY_TO_CONNECTORS.get(sym_a_name, 2)
            cb = SYMMETRY_TO_CONNECTORS.get(sym_b_name, 2)
            valid_topos = SYMMETRY_PAIR_TO_TOPOLOGY.get((ca, cb), ["HCB_A"])

            topo = valid_topos[action_3 % len(valid_topos)]
            core_a_name = self.valid_cores.get(sym_a_name, self.core_names.get(sym_a_name, ["default"]))
            core_a = core_a_name[self._core_a % len(core_a_name)] if core_a_name else "default"
            core_b_name = self.valid_cores.get(sym_b_name, self.core_names.get(sym_b_name, ["default"]))
            core_b = core_b_name[self._core_b % len(core_b_name)] if core_b_name else "default"

            # Assemble COF
            import contextlib
            old_out = sys.stdout; old_err = sys.stderr
            sys.stdout = io.StringIO(); sys.stderr = io.StringIO()
            try:
                from pycofbuilder.building_block import BuildingBlock
                from pycofbuilder.framework import Framework
                bb1 = BuildingBlock()
                bb1.from_name(f"{sym_a_name}_{core_a}_COOH_H")
                bb2 = BuildingBlock()
                bb2.from_name(f"{sym_b_name}_{core_b}_COOH_H")
                fw = Framework(dist_threshold=0.4)
                if bb1.connectivity >= bb2.connectivity:
                    fw.from_building_blocks(bb1, bb2, topo, "AA")
                else:
                    fw.from_building_blocks(bb2, bb1, topo, "AA")

                if self.n2_lookup and fw.name in self.n2_lookup:
                    n2_reward = self.n2_lookup[fw.name] * 10.0 + 8.0
                else:
                    n2_reward = 18.0 + min(len(fw.atom_types) / 500.0, 5.0)

                # Diversity bonus
                total = sum(self.topo_counts.values()) + 1
                count = self.topo_counts.get(topo, 0)
                div_bonus = 3.0 / (count / max(total / N_TOPOS, 1) + 1)

                reward = n2_reward + div_bonus
                self.assembly_success += 1
                self.topo_counts[topo] += 1
                self.topo_successes[topo] += 1
                info = {'success': True, 'tier': 2, 'topology': topo,
                        'sym_a': sym_a_name, 'sym_b': sym_b_name,
                        'core_a': core_a, 'core_b': core_b,
                        'atoms': len(fw.atom_types), 'sg': fw.space_group,
                        'n2_reward': n2_reward, 'div_bonus': div_bonus}
            except Exception as e:
                reward = 1.0
                info = {'success': False, 'tier': 1, 'error': str(e)[:60],
                        'topology': topo, 'sym_a': sym_a_name, 'sym_b': sym_b_name}
            finally:
                sys.stdout = old_out; sys.stderr = old_err

            self.assembly_total += 1
            self.episode_rewards.append(reward)
            self.topo_counts[topo] = self.topo_counts.get(topo, 0) + 1

            # Terminal states (all zeros + reward signal)
            s1 = np.zeros(self.state_dim, dtype=np.float32)
            s1[-1] = reward / 20.0
            s2 = np.zeros(self.state_dim, dtype=np.float32)
            s2[-1] = reward / 20.0
            s3 = np.zeros(self.state_dim, dtype=np.float32)
            s3[-1] = reward / 20.0

            return (s1, s2, s3), reward, True, info


if __name__ == "__main__":
    env = COFDesignEnvV6()
    print(f"=== v6: Rich State + Sequential Coordination ===")
    print(f"State dim: {env.state_dim}")
    print(f"Symmetries: {env.n_sym}, Topologies: {env.n_topo}")
    print(f"Core features pre-loaded for {sum(len(v) for v in env.core_features.values())} templates")

    # Test sequential steps
    s1, s2, s3 = env.reset()
    print(f"\nStep 0 (reset): s1={s1[:12]}, s2={s2[:12]}")

    # Step 1: Agent 1 picks T3 (idx=1), core 0
    a1 = (1, 0)
    a2 = (0, 0)  # ignored in step 1
    a3 = 0       # ignored in step 1
    (s1, s2, s3), r, done, info = env.step(a1, a2, a3)
    print(f"Step 1: sym_a={info['sym_a']}, done={done}, r={r}")
    print(f"  s2 (Agent 2 sees partner): sym_partner={np.where(s2[14:20])[0].tolist()}")

    # Step 2: Agent 2 picks L2 (idx=0), core 0
    a2 = (0, 0)
    (s1, s2, s3), r, done, info = env.step(a1, a2, a3)
    print(f"Step 2: sym_b={info.get('sym_b')}, done={done}, r={r}")
    print(f"  s3 (Agent 3 sees topology mask): valid_topos={np.where(s3[19:19+9])[0].tolist()}")

    # Step 3: Agent 3 picks topology
    a3 = 0
    (s1, s2, s3), r, done, info = env.step(a1, a2, a3)
    print(f"Step 3: topo={info.get('topology')}, success={info.get('success')}, r={r:.1f}, done={done}")
    print("v6 ready ✓")
