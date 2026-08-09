"""
Phase 4 v5: Full MARL action space.

Agent 1: Core-A (symmetry + template + connector + functional group)
Agent 2: Core-B (symmetry + template + connector + functional group)
Agent 3: Topology (masked by symmetry pair)

Connector (15) and Functional Group (31) are selectable — NOT hardcoded.
Total action space: 6 sym × ~10 templates × 15 connectors × 31 FGs × 2 agents
"""

import os, json, sys, io
from pathlib import Path
import numpy as np

SYMMETRY_TYPES = ["L2", "T3", "S4", "H6", "D4", "R4"]
SYMMETRY_TO_CONNECTORS = {"L2": 2, "T3": 3, "S4": 4, "H6": 6, "D4": 4, "R4": 4}

SYMMETRY_PAIR_TO_TOPOLOGY = {
    (2, 3): ["HCB_A", "KGD", "FXT_A"], (3, 2): ["HCB_A", "KGD", "FXT_A"],
    (2, 4): ["SQL_A", "DIA_A", "BOR"], (4, 2): ["SQL_A", "DIA_A", "BOR"],
    (2, 2): ["KGM_A"], (2, 6): ["HXL_A", "LON_A"], (6, 2): ["HXL_A", "LON_A"],
    (3, 3): ["KGD"], (4, 4): ["DIA_A", "BOR"],
}

TOPOLOGIES = sorted(set(t for tops in SYMMETRY_PAIR_TO_TOPOLOGY.values() for t in tops))


def _get_connectors_and_groups():
    """Get available connectors and functional groups from pycofbuilder."""
    try:
        from pycofbuilder.building_block import BuildingBlock
        bb = BuildingBlock()
        connectors = bb.get_available_conector()
        func_groups = bb.get_available_R()
        return connectors, func_groups
    except:
        return (["COOH", "CHO", "NH2", "Br", "Cl", "OH2", "CH3", "O", "CH2CN", "NHOH", "BOH2", "CCH3O", "COCHCHOH", "CONHNH2", "CHNNH2"],
                ["H", "NH2", "CHO", "COOH", "Br", "Cl", "CH3", "OH", "CN", "NO2"])


CONNECTORS, FUNC_GROUPS = _get_connectors_and_groups()
N_CONNECTORS = len(CONNECTORS)
N_FUNC_GROUPS = len(FUNC_GROUPS)


class COFDesignEnvV5:
    """Full-action-space COF design environment."""

    def __init__(self, core_dir=None, n2_lookup_path=None):
        if core_dir is None:
            core_dir = str(Path(__file__).parent.parent / "data" / "core")
        self.cores = {}
        for sym in SYMMETRY_TYPES:
            d = os.path.join(core_dir, sym)
            if os.path.exists(d):
                self.cores[sym] = sorted([f.replace(".cjson", "") for f in os.listdir(d) if f.endswith(".cjson")])
        self.n2_lookup = {}
        if n2_lookup_path and os.path.exists(n2_lookup_path):
            with open(n2_lookup_path) as f: self.n2_lookup = json.load(f)
        self.n_sym = len(SYMMETRY_TYPES)
        self.n_topo = len(TOPOLOGIES)
        self.n_conn = N_CONNECTORS
        self.n_fg = N_FUNC_GROUPS
        self.assembly_success = 0; self.assembly_total = 0
        self.episode_rewards = []
        self.topo_counts = {t: 0 for t in TOPOLOGIES}
        self.sym_pair_counts = {}
        self._builtin_connectors = set()

    def auto_detect_connectors(self, sym, core):
        """Detect which connectors work with a specific Core template."""
        key = (sym, core)
        if key in self._builtin_connectors:
            return None  # already known
        # Try to find working connectors
        working = []
        for conn in CONNECTORS[:5]:  # Test top 5 for speed
            try:
                from pycofbuilder.building_block import BuildingBlock
                bb = BuildingBlock()
                bb.from_name(f"{sym}_{core}_{conn}_H")
                if bb.connectivity == SYMMETRY_TO_CONNECTORS.get(sym, 2):
                    working.append(conn)
            except: pass
        return working or ["COOH"]  # fallback

    def get_valid_topologies(self, sym_a, sym_b):
        ca = SYMMETRY_TO_CONNECTORS.get(sym_a, 2)
        cb = SYMMETRY_TO_CONNECTORS.get(sym_b, 2)
        return SYMMETRY_PAIR_TO_TOPOLOGY.get((ca, cb), ["HCB_A"])

    def reset(self):
        return (np.zeros(16, dtype=np.float32), np.zeros(16, dtype=np.float32),
                np.zeros(16, dtype=np.float32))

    def step(self, action_1, action_2, action_3):
        """action_1/2: (sym_idx, core_idx, conn_idx, fg_idx)"""
        sym_a = SYMMETRY_TYPES[action_1[0]]
        sym_b = SYMMETRY_TYPES[action_2[0]]
        valid_topos = self.get_valid_topologies(sym_a, sym_b)
        topo = valid_topos[action_3 % len(valid_topos)]

        core_a = self.cores.get(sym_a, ["default"])[action_1[1] % max(len(self.cores.get(sym_a, [])), 1)]
        core_b = self.cores.get(sym_b, ["default"])[action_2[1] % max(len(self.cores.get(sym_b, [])), 1)]
        conn_a = CONNECTORS[action_1[2] % N_CONNECTORS]
        conn_b = CONNECTORS[action_2[2] % N_CONNECTORS]
        fg_a = FUNC_GROUPS[action_1[3] % N_FUNC_GROUPS]
        fg_b = FUNC_GROUPS[action_2[3] % N_FUNC_GROUPS]

        ca = SYMMETRY_TO_CONNECTORS.get(sym_a, 2)
        cb = SYMMETRY_TO_CONNECTORS.get(sym_b, 2)
        if (ca, cb) not in SYMMETRY_PAIR_TO_TOPOLOGY:
            s = np.zeros(16, dtype=np.float32); s[action_1[0]] = 1.0
            return (s, s.copy(), s.copy()), -1.0, True, {"success": False, "error": "invalid_pair"}

        old = sys.stdout; sys.stdout = io.StringIO()
        try:
            from pycofbuilder.building_block import BuildingBlock
            from pycofbuilder.framework import Framework
            bb1 = BuildingBlock(); bb1.from_name(f"{sym_a}_{core_a}_{conn_a}_{fg_a}")
            bb2 = BuildingBlock(); bb2.from_name(f"{sym_b}_{core_b}_{conn_b}_{fg_b}")
            fw = Framework(dist_threshold=0.4)
            if bb1.connectivity >= bb2.connectivity:
                fw.from_building_blocks(bb1, bb2, topo, "AA")
            else:
                fw.from_building_blocks(bb2, bb1, topo, "AA")

            if self.n2_lookup and fw.name in self.n2_lookup:
                n2_reward = self.n2_lookup[fw.name] * 10.0 + 8.0
            else:
                n2_reward = 18.0 + min(len(fw.atom_types) / 500.0, 5.0)

            total = sum(self.topo_counts.values()) + 1
            count = self.topo_counts.get(topo, 0)
            div_bonus = 3.0 / (count / max(total / self.n_topo, 1) + 1)
            reward = n2_reward + div_bonus
            self.assembly_success += 1
            self.topo_counts[topo] += 1
            info = {"success": True, "tier": 2, "topology": topo, "sym_a": sym_a, "sym_b": sym_b,
                    "core_a": core_a, "core_b": core_b, "conn_a": conn_a, "conn_b": conn_b,
                    "fg_a": fg_a, "fg_b": fg_b,
                    "atoms": len(fw.atom_types), "sg": fw.space_group}
        except Exception as e:
            reward = 1.0
            info = {"success": False, "tier": 1, "error": str(e)[:60],
                    "topology": topo, "sym_a": sym_a, "sym_b": sym_b}
        finally:
            sys.stdout = old

        self.assembly_total += 1; self.episode_rewards.append(reward)
        self.topo_counts[topo] = self.topo_counts.get(topo, 0) + 1
        key = (sym_a, sym_b)
        self.sym_pair_counts[key] = self.sym_pair_counts.get(key, 0) + 1

        s1 = np.zeros(16, dtype=np.float32); s1[action_1[0]] = 1.0; s1[9] = reward / 20.0
        s2 = np.zeros(16, dtype=np.float32); s2[action_2[0]] = 1.0; s2[9] = reward / 20.0
        s3 = np.zeros(16, dtype=np.float32); s3[action_3 % self.n_topo] = 1.0; s3[9] = reward / 20.0
        return (s1, s2, s3), reward, True, info


if __name__ == "__main__":
    env = COFDesignEnvV5()
    print(f"=== v5: Full Action Space ===")
    print(f"Symmetries: {env.n_sym} | Topologies: {env.n_topo}")
    print(f"Connectors: {env.n_conn} | Func Groups: {env.n_fg}")
    print(f"Total action combos: {env.n_sym} × templates × {env.n_conn} × {env.n_fg}")
    print(f"Topologies: {TOPOLOGIES}")

    # Quick test
    a1 = (1, 0, 0, 1)  # T3, first core, COOH, NH2
    a2 = (0, 0, 0, 0)  # L2, first core, COOH, H
    a3 = 0
    (_, _, _), r, _, info = env.step(a1, a2, a3)
    print(f"\nTest T3(COOH,NH2) + L2(COOH,H) → {info.get('topology', '?')}: "
          f"reward={r:.1f} ok={info['success']}")
    if info['success']:
        print(f"  Connectors: {info.get('conn_a')} + {info.get('conn_b')}")
    print("v5 ready ✓")
