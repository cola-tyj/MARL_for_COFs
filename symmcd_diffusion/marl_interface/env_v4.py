"""
Phase 4 v4: 3-Agent MARL — symmetries drive topology.

Agent 1: Core-A symmetry (6 options) + template
Agent 2: Core-B symmetry (6 options) + template
Agent 3: Topology — masked to only show options compatible
         with the connector pair (conn_a, conn_b)

This is the correct causal order: symmetries determine connector counts,
connector counts determine compatible topologies.
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

class COFDesignEnvV4:
    def __init__(self, core_dir=None, n2_lookup_path=None):
        if core_dir is None:
            core_dir = str(Path(__file__).parent.parent / "data" / "core")
        self.cores = {}
        for sym in SYMMETRY_TYPES:
            d = os.path.join(core_dir, sym)
            if os.path.exists(d):
                self.cores[sym] = sorted([f.replace(".cjson","") for f in os.listdir(d) if f.endswith(".cjson")])
        self.n2_lookup = {}
        if n2_lookup_path and os.path.exists(n2_lookup_path):
            with open(n2_lookup_path) as f: self.n2_lookup = json.load(f)
        # Filter valid cores
        vc_path = str(Path(__file__).parent.parent / "generated" / "valid_cores.json")
        if os.path.exists(vc_path):
            with open(vc_path) as f: self.valid_cores = json.load(f)
        else:
            self.valid_cores = self.cores
        self.n_sym = len(SYMMETRY_TYPES)
        self.n_topo = len(TOPOLOGIES)
        self.state_dim = 16
        self.assembly_success = 0; self.assembly_total = 0
        self.episode_rewards = []
        # Topology diversity tracking
        self.topo_counts = {t: 0 for t in TOPOLOGIES}
        self.topo_successes = {t: 0 for t in TOPOLOGIES}

    def get_valid_topologies(self, sym_a, sym_b):
        ca = SYMMETRY_TO_CONNECTORS.get(sym_a, 2)
        cb = SYMMETRY_TO_CONNECTORS.get(sym_b, 2)
        return SYMMETRY_PAIR_TO_TOPOLOGY.get((ca, cb), ["HCB_A"])

    def reset(self):
        return (np.zeros(16, dtype=np.float32), np.zeros(16, dtype=np.float32),
                np.zeros(16, dtype=np.float32))

    def step(self, action_1, action_2, action_3):
        sym_a = SYMMETRY_TYPES[action_1[0]]
        sym_b = SYMMETRY_TYPES[action_2[0]]
        valid_topos = self.get_valid_topologies(sym_a, sym_b)
        topo = valid_topos[action_3 % len(valid_topos)]

        core_a = self.valid_cores.get(sym_a, self.cores.get(sym_a, ["default"]))[
            action_1[1] % max(len(self.valid_cores.get(sym_a, ["default"])), 1)]
        core_b = self.valid_cores.get(sym_b, self.cores.get(sym_b, ["default"]))[
            action_2[1] % max(len(self.valid_cores.get(sym_b, ["default"])), 1)]

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
                n2_reward = self.n2_lookup[fw.name] * 10.0 + 8.0
            else:
                n2_reward = 18.0 + min(len(fw.atom_types) / 500.0, 5.0)

            # Topology diversity bonus: under-explored topologies get extra reward
            total = sum(self.topo_counts.values()) + 1
            count = self.topo_counts.get(topo, 0)
            diversity_bonus = 3.0 / (count / max(total / self.n_topo, 1) + 1)

            reward = n2_reward + diversity_bonus
            self.assembly_success += 1
            self.topo_counts[topo] += 1
            self.topo_successes[topo] += 1
            info = {"success": True, "tier": 2, "topology": topo, "sym_a": sym_a,
                    "sym_b": sym_b, "core_a": core_a, "core_b": core_b,
                    "atoms": len(fw.atom_types), "sg": fw.space_group,
                    "n2_reward": n2_reward, "diversity_bonus": diversity_bonus}
        except Exception as e:
            reward = 1.0  # Tier 1: valid pair, assembly failed
            info = {"success": False, "tier": 1, "error": str(e)[:60],
                    "topology": topo, "sym_a": sym_a, "sym_b": sym_b}
        finally:
            sys.stdout = old

        self.assembly_total += 1; self.episode_rewards.append(reward)
        self.topo_counts[topo] = self.topo_counts.get(topo, 0) + 1
        s1 = np.zeros(16, dtype=np.float32); s1[action_1[0]] = 1.0; s1[9] = reward / 20.0
        s2 = np.zeros(16, dtype=np.float32); s2[action_2[0]] = 1.0; s2[9] = reward / 20.0
        s3 = np.zeros(16, dtype=np.float32); s3[action_3 % self.n_topo] = 1.0; s3[9] = reward / 20.0
        return (s1, s2, s3), reward, True, info


if __name__ == "__main__":
    env = COFDesignEnvV4()
    print(f"=== v4: Symmetries → Topology ===")
    print(f"Agent 1/2: {env.n_sym} symmetries each")
    print(f"Agent 3: topology (masked, {env.n_topo} total)")
    print("Topology mask examples:")
    for (ca,cb),tops in SYMMETRY_PAIR_TO_TOPOLOGY.items():
        print(f"  ({ca},{cb}): {tops}")
    s1,s2,s3 = env.reset()
    a1=(1,0); a2=(0,0); a3=1  # T3+L2 → HCB_A
    (_,_,_),r,_,i = env.step(a1,a2,a3)
    print(f"\nTest T3+L2→{i.get('topology','?')}: reward={r:.1f} success={i['success']}")
    print("v4 ready ✓")
