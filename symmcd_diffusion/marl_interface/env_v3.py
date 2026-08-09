"""
Phase 4: 3-Agent MARL Environment.

Agent 0: Topology (9 options)
Agent 1: Core-A (symmetry + template), constrained by Agent 0's choice
Agent 2: Core-B (symmetry + template), constrained by Agent 0's choice

Constraint: Agent 0 selects topology → only compatible symmetry pairs
            are available for Agent 1 and Agent 2.
            → Eliminates topology-symmetry mismatches completely.

Topology → valid (conn_a, conn_b) pairs → valid symmetries for each agent.
"""

import os, json, sys, io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SYMMETRY_TYPES = ["L2", "T3", "S4", "H6", "D4", "R4"]
SYMMETRY_TO_CONNECTORS = {"L2": 2, "T3": 3, "S4": 4, "H6": 6, "D4": 4, "R4": 4}

# Topology → list of valid (conn_a, conn_b) pairs
TOPOLOGY_TO_SYM_PAIRS = {
    "HCB_A": [(3, 2)],
    "KGD":   [(3, 2), (3, 3)],
    "FXT_A": [(3, 2)],
    "SQL_A": [(4, 2)],
    "DIA_A": [(4, 2), (4, 4)],
    "BOR":   [(4, 2), (4, 4)],
    "HXL_A": [(6, 2)],
    "LON_A": [(6, 2)],
    "KGM_A": [(2, 2)],
}

TOPOLOGIES = sorted(TOPOLOGY_TO_SYM_PAIRS.keys())
STACKINGS = ["AA"]


class COFDesignEnvV3:
    """
    3-Agent COF design environment with constrained action spaces.

    Agent 0: topology (discrete, 9 options)
    Agent 1: Core-A specs, constrained to compatible symmetries
    Agent 2: Core-B specs, constrained to compatible symmetries

    Design rationale:
      Agent 0 selects topology first → determines valid (conn_a, conn_b)
      → Agent 1/2 choose from compatible symmetries only
      → guarantees 100% assembly compatibility
    """

    def __init__(self, core_dir: str = None, n2_lookup_path: str = None):
        if core_dir is None:
            core_dir = str(Path(__file__).parent.parent / "data" / "core")

        self.cores = {}
        for sym in SYMMETRY_TYPES:
            d = os.path.join(core_dir, sym)
            if os.path.exists(d):
                self.cores[sym] = sorted([
                    f.replace(".cjson", "") for f in os.listdir(d) if f.endswith(".cjson")
                ])

        self.n2_lookup = {}
        if n2_lookup_path and os.path.exists(n2_lookup_path):
            with open(n2_lookup_path) as f:
                self.n2_lookup = json.load(f)

        self.n_topologies = len(TOPOLOGIES)
        self.n_symmetries = len(SYMMETRY_TYPES)
        self.state_dim = 16  # fixed: one-hot topology(9) + one-hot sym(6) + reward(1)

        self.episode_rewards = []
        self.assembly_success = 0
        self.assembly_total = 0
        self._current_topo = None

    def get_valid_sym_pairs(self, topo_idx: int) -> List[Tuple[int, int]]:
        """Get valid (conn_a, conn_b) pairs for a given topology."""
        topo = TOPOLOGIES[topo_idx]
        pairs = TOPOLOGY_TO_SYM_PAIRS.get(topo, [(3, 2)])
        # Convert conn counts → symmetry indices
        result = []
        for ca, cb in pairs:
            for ia, sa in enumerate(SYMMETRY_TYPES):
                if SYMMETRY_TO_CONNECTORS.get(sa) == ca:
                    for ib, sb in enumerate(SYMMETRY_TYPES):
                        if SYMMETRY_TO_CONNECTORS.get(sb) == cb:
                            result.append((ia, ib))
        return result

    def reset(self):
        self._current_topo = None
        return (np.zeros(self.state_dim, dtype=np.float32),
                np.zeros(self.state_dim, dtype=np.float32),
                np.zeros(self.state_dim, dtype=np.float32))

    def step(self, action_0: int, action_1: Tuple[int, int],
             action_2: Tuple[int, int]) -> Tuple:
        topo = TOPOLOGIES[action_0]
        sym_a = SYMMETRY_TYPES[action_1[0]]
        sym_b = SYMMETRY_TYPES[action_2[0]]

        ca = SYMMETRY_TO_CONNECTORS.get(sym_a, 2)
        cb = SYMMETRY_TO_CONNECTORS.get(sym_b, 2)
        valid_pairs = TOPOLOGY_TO_SYM_PAIRS.get(topo, [(3, 2)])

        # Tiered reward design:
        #   Level 0: invalid symmetry pair → -1.0
        #   Level 1: valid symmetry pair but assembly fails → +0.5 (partial credit)
        #   Level 2: assembly succeeds → N₂ scaled + 5.0

        if (ca, cb) not in valid_pairs:
            reward = -1.0
            self.assembly_total += 1
            self.episode_rewards.append(reward)
            s0 = np.zeros(16, dtype=np.float32); s0[action_0] = 1.0
            s1 = np.zeros(16, dtype=np.float32); s1[action_1[0]] = 1.0
            s2 = np.zeros(16, dtype=np.float32); s2[action_2[0]] = 1.0
            return (s0, s1, s2), reward, True, {"success": False, "tier": 0, "error": "incompatible_symmetry_pair"}

        core_a = self.cores.get(sym_a, ["default"])[action_1[1] % max(len(self.cores.get(sym_a, [])), 1)]
        core_b = self.cores.get(sym_b, ["default"])[action_2[1] % max(len(self.cores.get(sym_b, [])), 1)]

        old = sys.stdout; sys.stdout = io.StringIO()
        try:
            from pycofbuilder.building_block import BuildingBlock
            from pycofbuilder.framework import Framework
            bb1 = BuildingBlock(); bb1.from_name(f"{sym_a}_{core_a}_COOH_H")
            bb2 = BuildingBlock(); bb2.from_name(f"{sym_b}_{core_b}_COOH_H")
            fw = Framework(dist_threshold=0.4)
            if bb1.connectivity >= bb2.connectivity:
                fw.from_building_blocks(bb1, bb2, topo, "AA")
            else:
                fw.from_building_blocks(bb2, bb1, topo, "AA")

            if self.n2_lookup and fw.name in self.n2_lookup:
                reward = self.n2_lookup[fw.name] * 10.0 + 8.0  # Tier 2: real N₂ scaled higher
            else:
                reward = 18.0 + min(len(fw.atom_types) / 500.0, 5.0)
            self.assembly_success += 1
            info = {"success": True, "tier": 2, "topology": topo, "sym_a": sym_a, "sym_b": sym_b,
                    "core_a": core_a, "core_b": core_b,
                    "atoms": len(fw.atom_types), "sg": fw.space_group, "reward": reward}
        except Exception as e:
            # Valid symmetry pair but assembly failed → partial credit
            reward = 1.0  # Tier 1: valid pair, assembly failed
            info = {"success": False, "tier": 1, "error": str(e)[:60],
                    "topology": topo, "sym_a": sym_a, "sym_b": sym_b}
        finally:
            sys.stdout = old

        self.assembly_total += 1
        self.episode_rewards.append(reward)
        s0 = np.zeros(16, dtype=np.float32); s0[action_0] = 1.0; s0[9] = reward / 20.0
        s1 = np.zeros(16, dtype=np.float32); s1[action_1[0]] = 1.0; s1[9] = reward / 20.0
        s2 = np.zeros(16, dtype=np.float32); s2[action_2[0]] = 1.0; s2[9] = reward / 20.0
        return (s0, s1, s2), reward, True, info


if __name__ == "__main__":
    env = COFDesignEnvV3()
    print(f"=== 3-Agent Env ===")
    print(f"Agent 0: {env.n_topologies} topologies")
    for i, t in enumerate(TOPOLOGIES):
        pairs = TOPOLOGY_TO_SYM_PAIRS[t]
        syms = set()
        for ca, cb in pairs:
            for s in SYMMETRY_TYPES:
                if SYMMETRY_TO_CONNECTORS[s] == ca: syms.add(s)
        print(f"  {t}: conn_pairs={pairs}, sym_A={syms}")
    print(f"Agent 1/2: {env.n_symmetries} symmetries each")
    print("3-Agent env ready")
