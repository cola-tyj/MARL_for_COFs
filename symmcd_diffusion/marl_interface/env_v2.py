"""
Phase 4: 2-Agent MARL Environment with Diffusion Core Generation.

Agent 1: Core-A specification (symmetry + template)
Agent 2: Core-B specification (symmetry + template)

Topology is AUTO-DETERMINED from the symmetry pair's connector counts,
NOT selected by an agent. This eliminates invalid topology-symmetry mismatches.

Design rationale:
  (conn_a, conn_b) → compatible topologies
  L2(2) + T3(3)   → HCB_A, KGD, FXT_A
  L2(2) + S4(4)   → SQL_A, DIA_A, BOR
  L2(2) + L2(2)   → KGM_A
  L2(2) + H6(6)   → HXL_A, LON_A

Action → Diffusion refines Core → pycofbuilder assembles BB → COF → Reward
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Specifications ──────────────────────────────────────────────────────────

SYMMETRY_TYPES = ["L2", "T3", "S4", "H6", "D4", "R4"]
SYMMETRY_TO_CONNECTORS = {"L2": 2, "T3": 3, "S4": 4, "H6": 6, "D4": 4, "R4": 4}

# Topology → expected symmetry pair (conn_a, conn_b)
SYMMETRY_PAIR_TO_TOPOLOGY = {
    (2, 3): ["HCB_A", "KGD", "FXT_A"],
    (3, 2): ["HCB_A", "KGD", "FXT_A"],
    (2, 4): ["SQL_A", "DIA_A", "BOR"],
    (4, 2): ["SQL_A", "DIA_A", "BOR"],
    (2, 2): ["KGM_A"],
    (2, 6): ["HXL_A", "LON_A"],
    (6, 2): ["HXL_A", "LON_A"],
    (3, 3): ["KGD"],
    (4, 4): ["DIA_A", "BOR"],
}

TOPOLOGIES = sorted(set(t for tops in SYMMETRY_PAIR_TO_TOPOLOGY.values() for t in tops))
STACKINGS = ["AA", "AB"]


class COFDesignEnv:
    """
    2-Agent COF design environment.

    Agent 1: Core-A symmetry (6 options) + template selection
    Agent 2: Core-B symmetry (6 options) + template selection
    Topology: auto-determined from (conn_a, conn_b)

    Action space:
      Agent 1: (sym_idx_a, template_idx_a) — MultiDiscrete(6, N_cores_a)
      Agent 2: (sym_idx_b, template_idx_b) — MultiDiscrete(6, N_cores_b)

    State space:
      Agent 1: [sym_a_onehot(6), prev_reward, conn_count]
      Agent 2: [sym_b_onehot(6), prev_reward, conn_count]

    Reward:
      Assembly success → N₂ adsorption value (real or mock)
      Assembly failure → -1.0
    """

    def __init__(self, core_dir: str = None, n2_lookup_path: str = None):
        if core_dir is None:
            core_dir = str(Path(__file__).parent.parent / "data" / "core")

        # Load Cores
        self.cores = {}
        for sym in SYMMETRY_TYPES:
            d = os.path.join(core_dir, sym)
            if os.path.exists(d):
                self.cores[sym] = sorted([
                    f.replace(".cjson", "")
                    for f in os.listdir(d)
                    if f.endswith(".cjson")
                ])

        # N₂ lookup
        self.n2_lookup = {}
        if n2_lookup_path and os.path.exists(n2_lookup_path):
            with open(n2_lookup_path) as f:
                self.n2_lookup = json.load(f)

        # Action space sizes
        self.n_symmetries = len(SYMMETRY_TYPES)
        self.n_cores_a = max(len(self.cores.get("T3", [])), len(self.cores.get("S4", [])))
        self.n_cores_b = max(len(self.cores.get("L2", [])), 1)

        # State dimensions
        self.state_dim = 8  # one-hot sym(6) + prev_reward(1) + conn_count(1)

        # Stats
        self.episode_rewards = []
        self.assembly_success = 0
        self.assembly_total = 0

    def get_topologies(self, sym_a: str, sym_b: str) -> List[str]:
        """Get all compatible topologies for a symmetry pair."""
        conn_a = SYMMETRY_TO_CONNECTORS.get(sym_a, 2)
        conn_b = SYMMETRY_TO_CONNECTORS.get(sym_b, 2)
        return SYMMETRY_PAIR_TO_TOPOLOGY.get((conn_a, conn_b), ["HCB_A"])

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return initial states for 2 agents."""
        s1 = np.zeros(self.state_dim, dtype=np.float32)
        s2 = np.zeros(self.state_dim, dtype=np.float32)
        return s1, s2

    def step(
        self,
        action_a: Tuple[int, int],
        action_b: Tuple[int, int],
        topology_idx: int = 0,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict]:
        """
        Execute one design step.

        Args:
            action_a: (sym_idx_a, template_idx_a)
            action_b: (sym_idx_b, template_idx_b)
            topology_idx: index into compatible topology list (0=default)

        Returns:
            (state_a, state_b), reward, done, info
        """
        sym_a = SYMMETRY_TYPES[action_a[0]]
        sym_b = SYMMETRY_TYPES[action_b[0]]

        core_a = self.cores.get(sym_a, ["default"])[
            action_a[1] % len(self.cores.get(sym_a, ["default"]))
        ]
        core_b = self.cores.get(sym_b, ["default"])[
            action_b[1] % len(self.cores.get(sym_b, ["default"]))
        ]

        # Agent-selectable topology (from compatible list)
        topos = self.get_topologies(sym_a, sym_b)
        topology = topos[topology_idx % len(topos)]

        # Try COF assembly
        try:
            import sys, io as _io
            _old = sys.stdout
            sys.stdout = _io.StringIO()

            from pycofbuilder.building_block import BuildingBlock
            from pycofbuilder.framework import Framework

            bb1 = BuildingBlock()
            bb1.from_name(f"{sym_a}_{core_a}_COOH_H")
            bb2 = BuildingBlock()
            bb2.from_name(f"{sym_b}_{core_b}_COOH_H")

            fw = Framework(dist_threshold=0.4)
            if bb1.connectivity >= bb2.connectivity:
                fw.from_building_blocks(bb1, bb2, topology, "AA")
            else:
                fw.from_building_blocks(bb2, bb1, topology, "AA")

            sys.stdout = _old

            # Reward: real N₂ or mock
            if self.n2_lookup and fw.name in self.n2_lookup:
                reward = self.n2_lookup[fw.name] * 10.0 + 5.0
            else:
                reward = 15.0 + min(len(fw.atom_types) / 500.0, 5.0) + np.random.normal(0, 0.5)

            self.assembly_success += 1
            info = {
                "success": True,
                "topology": topology,
                "sym_a": sym_a, "sym_b": sym_b,
                "core_a": core_a, "core_b": core_b,
                "atoms": len(fw.atom_types),
                "sg": fw.space_group,
                "reward": reward,
            }
        except Exception as e:
            reward = -1.0
            info = {"success": False, "error": str(e)[:60]}

        self.assembly_total += 1
        self.episode_rewards.append(reward)

        # Next state (simple encoding)
        s1 = np.zeros(self.state_dim, dtype=np.float32)
        s1[action_a[0]] = 1.0
        s1[6] = reward / 20.0
        s1[7] = SYMMETRY_TO_CONNECTORS.get(sym_a, 0) / 6.0

        s2 = np.zeros(self.state_dim, dtype=np.float32)
        s2[action_b[0]] = 1.0
        s2[6] = reward / 20.0
        s2[7] = SYMMETRY_TO_CONNECTORS.get(sym_b, 0) / 6.0

        return (s1, s2), reward, True, info

    def render(self):
        """Print latest stats."""
        if self.assembly_total > 0:
            rate = self.assembly_success / self.assembly_total
            avg_r = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            print(f"Success: {rate:.0%} | AvgR: {avg_r:.1f} | "
                  f"Episodes: {self.assembly_total}")


if __name__ == "__main__":
    env = COFDesignEnv()
    s1, s2 = env.reset()
    print(f"=== 2-Agent COF Design Environment ===")
    print(f"Agent 1: symmetry ({env.n_symmetries} types) × "
          f"template ({env.n_cores_a} cores)")
    print(f"Agent 2: symmetry ({env.n_symmetries} types) × "
          f"template ({env.n_cores_b} cores)")
    print(f"Topology: auto-determined from (conn_a, conn_b)")
    print(f"Cores: { {k: len(v) for k, v in env.cores.items()} }")
    print()

    for i in range(3):
        a1 = (np.random.randint(0, env.n_symmetries), np.random.randint(0, 5))
        a2 = (np.random.randint(0, env.n_symmetries), np.random.randint(0, 5))
        _, reward, _, info = env.step(a1, a2)
        status = "✓" if info["success"] else "✗"
        topo = info.get("topology", "FAIL")
        sym_a = SYMMETRY_TYPES[a1[0]] if info["success"] else "?"
        sym_b = SYMMETRY_TYPES[a2[0]] if info["success"] else "?"
        print(f"  {status} {sym_a}+{sym_b} → {topo}: reward={reward:.1f}")

    print(f"\n2-Agent environment ready ✓")
