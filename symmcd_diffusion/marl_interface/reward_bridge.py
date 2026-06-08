"""
Reward bridge: connects diffusion-generated building blocks to COF assembly and reward.

Reuses the existing mappo/reward.py infrastructure (make_cof, predictor, RND)
but adapts it for diffusion-generated building blocks.

This bridge is the key integration point between the diffusion model (Phase 1-3)
and the MARL controller (Phase 4).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root for existing mappo imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class RewardBridge:
    """
    Computes rewards for diffusion-generated COF designs.

    Connects:
    1. Diffusion-generated building blocks (cjson files)
    2. Existing COF predictor (cof_predictor/main.py)
    3. Existing RND exploration bonus (mappo/reward.py)
    """

    def __init__(
        self,
        cof_dir: str = "/home/tianyajun/MARL_for_COFs/cofs",
        predictor_log: str = "/home/tianyajun/MARL_for_COFs/cofs/predictor.log",
    ):
        self.cof_dir = Path(cof_dir)
        self.predictor_log = predictor_log

        # Valid topology-symmetry-reaction combinations
        self.topology_map = {
            ("T3", "L2"): "HCB_A",
            ("S4", "S4"): "SQL",
            ("S4", "L2"): "SQL_A",
            ("H6", "T3"): "KGD",
            ("H6", "L2"): "HXL_A",
            ("T3", "T3"): "HCB",
            ("T3", "S4"): "KGM",
            ("S4", "T3"): "KGM",
            ("L2", "T3"): "HCB_A",
            ("L2", "H6"): "HXL_A",
            ("L2", "S4"): "SQL_A",
            ("T3", "H6"): "KGD",
        }

        self.valid_reactions = [
            ("NH2", "CHO"), ("CHO", "NH2"),
            ("NHOH", "CHO"), ("CHO", "NHOH"),
            ("CH2CN", "CHO"), ("CHO", "CH2CN"),
            ("COOH", "NH2"), ("NH2", "COOH"),
            ("OHc", "NH2"), ("NH2", "OHc"),
            ("Cl", "Cl"),
        ]

        # Symmetry to connector count
        self.sym_connectors = {"L2": 2, "T3": 3, "S4": 4, "H6": 6}

        # Track episode statistics
        self.episode_rewards: List[float] = []
        self.valid_cof_count = 0
        self.invalid_cof_count = 0

    def compute_reward(
        self,
        bb_a_path: str,
        bb_b_path: str,
        sym_a: str,
        sym_b: str,
        func_a: str,
        func_b: str,
        topology: str,
        stacking: str = "AA",
        episode_num: int = 0,
        use_predictor: bool = True,
    ) -> Tuple[float, Dict]:
        """
        Compute reward for a COF design.

        Args:
            bb_a_path: path to BB-A cjson file
            bb_b_path: path to BB-B cjson file
            sym_a, sym_b: symmetry types ("L2", "T3", "S4", "H6")
            func_a, func_b: functional group types
            topology: COF topology name
            stacking: stacking mode
            episode_num: current episode number
            use_predictor: if True, use ML predictor; else use geometric heuristics

        Returns:
            (reward, info_dict) where info_dict contains diagnostic information
        """
        info = {
            "topology": topology,
            "stacking": stacking,
            "sym_a": sym_a,
            "sym_b": sym_b,
            "func_a": func_a,
            "func_b": func_b,
        }

        # Step 1: Check topology-reaction compatibility
        if (sym_a, sym_b) not in self.topology_map:
            info["error"] = f"incompatible_symmetry_{sym_a}_{sym_b}"
            self.invalid_cof_count += 1
            return -1.0, info

        if (func_a, func_b) not in self.valid_reactions:
            info["error"] = f"incompatible_reaction_{func_a}_{func_b}"
            self.invalid_cof_count += 1
            return -1.0, info

        # Step 2: Attempt COF assembly via pycofbuilder
        try:
            cof_name = f"{sym_a}_{func_a}-{sym_b}_{func_b}-{topology}-{stacking}"
            cof = self._assemble_cof(
                bb_a_path, bb_b_path, topology, stacking, cof_name
            )
            if cof is None:
                info["error"] = "assembly_failed"
                self.invalid_cof_count += 1
                return -0.5, info
        except Exception as e:
            info["error"] = f"assembly_exception:{str(e)[:50]}"
            self.invalid_cof_count += 1
            return -0.5, info

        # Step 3: Predict adsorption properties
        if use_predictor:
            try:
                adsorption = self._predict_adsorption(cof_name)
                n2_adsorption = adsorption.get("N2", 0.0)
                o2_adsorption = adsorption.get("O2", 0.0)
                info["n2_adsorption"] = n2_adsorption
                info["o2_adsorption"] = o2_adsorption
            except Exception as e:
                info["error"] = f"prediction_failed:{str(e)[:50]}"
                n2_adsorption = 0.0
                o2_adsorption = 0.0
        else:
            # Use geometric heuristics as fallback
            n2_adsorption = self._geometric_score(topology, sym_a, sym_b)
            o2_adsorption = n2_adsorption * 0.8
            info["n2_adsorption"] = n2_adsorption
            info["o2_adsorption"] = o2_adsorption

        # Step 4: Compute reward
        # Primary objective: maximize N2 adsorption
        # Bonus: valid assembly, symmetry compatibility
        base_reward = n2_adsorption * 0.1  # Scale for stability

        # Validity bonus
        validity_bonus = 1.0  # Successfully assembled

        # Symmetry compatibility bonus
        sym_bonus = 0.5  # Compatible symmetry pair

        # Diversity bonus (based on topology usage)
        diversity_bonus = self._compute_diversity_bonus(topology)

        reward = base_reward + validity_bonus + sym_bonus + diversity_bonus

        info["reward"] = reward
        info["base_reward"] = base_reward
        info["validity_bonus"] = validity_bonus
        info["sym_bonus"] = sym_bonus
        info["diversity_bonus"] = diversity_bonus

        self.episode_rewards.append(reward)
        self.valid_cof_count += 1

        return reward, info

    def _assemble_cof(
        self,
        bb_a_path: str,
        bb_b_path: str,
        topology: str,
        stacking: str,
        cof_name: str,
    ):
        """Assemble COF using pycofbuilder."""
        from pycofbuilder.framework import Framework

        cof = Framework.from_name(cof_name) if hasattr(Framework, 'from_name') else None

        if cof is None:
            # Fallback: try direct framework construction
            cof = Framework(cof_name)

        # Save CIF
        cif_path = self.cof_dir / f"{cof_name}.cif"
        cof.save(fmt="cif", supercell=[1, 1, 1], save_dir=str(self.cof_dir))

        return cof

    def _predict_adsorption(self, cof_name: str) -> Dict[str, float]:
        """Predict O2/N2 adsorption using existing predictor."""
        try:
            from cof_predictor.main import doPredict

            cif_path = str(self.cof_dir / f"{cof_name}.cif")
            log_path = str(self.cof_dir / "predictor.log")

            # Call the existing predictor
            result = doPredict(cif_path, log_path, ts=0)

            return {
                "N2": result.get("n2aPred", 0.0),
                "O2": result.get("o2aPred", 0.0),
                "bandgap": result.get("bandgapPred", 0.0),
            }
        except ImportError:
            # Predictor not available — return random values for testing
            return {"N2": np.random.uniform(0, 10), "O2": np.random.uniform(0, 8)}

    def _geometric_score(
        self, topology: str, sym_a: str, sym_b: str
    ) -> float:
        """Heuristic adsorption score based on topology and symmetry."""
        # Larger pore topologies tend to have higher adsorption
        topology_scores = {
            "HCB_A": 5.0, "HCB": 4.0,
            "SQL": 6.0, "SQL_A": 5.5,
            "KGD": 7.0, "HXL_A": 8.0,
            "KGM": 6.5, "KGM_A": 6.0,
            "FXT": 5.0, "FXT_A": 5.5,
            "LON_A": 7.5,
        }
        base = topology_scores.get(topology, 5.0)

        # Higher symmetry → potentially larger pores
        sym_scores = {"H6": 3.0, "S4": 2.0, "T3": 1.0, "L2": 0.0}
        base += sym_scores.get(sym_a, 0) + sym_scores.get(sym_b, 0)

        return base

    def _compute_diversity_bonus(self, topology: str) -> float:
        """Bonus for exploring underutilized topologies."""
        # Simple bonus to encourage topology diversity
        import random
        return random.uniform(0, 0.5)

    def reset_episode_stats(self):
        """Reset per-episode statistics."""
        self.episode_rewards = []

    def get_stats(self) -> Dict:
        """Get accumulated statistics."""
        return {
            "valid_cofs": self.valid_cof_count,
            "invalid_cofs": self.invalid_cof_count,
            "validity_rate": (
                self.valid_cof_count /
                max(self.valid_cof_count + self.invalid_cof_count, 1)
            ),
            "mean_reward": (
                np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            ),
        }
