"""
COF Design Environment v2 — 3-agent diffusion-integrated environment.

Redesign from 6-agent token-selection to 3-agent design-specification:
- Agent 0: selects topology (14 types) + stacking (8 modes)
- Agent 1: specifies Building Block A (symmetry, connectors, functional group)
- Agent 2: specifies Building Block B (symmetry, connectors, functional group)

The diffusion model generates actual molecular building blocks from agent specs.
"""

import gym
import numpy as np
import torch
from gym import spaces


# Available topologies and stacking modes
TOPOLOGIES = [
    "HCB", "HCB_A", "SQL", "SQL_A", "KGD", "HXL_A",
    "KGM", "KGM_A", "FXT", "FXT_A", "LON_A",
]

STACKING_MODES = ["AA", "AB1", "AB2", "ABC1", "ABC2", "AAl", "AAt"]

# Symmetry types and their connector counts
SYMMETRY_TYPES = ["L2", "T3", "S4", "H6"]
SYMMETRY_CONNECTORS = {"L2": 2, "T3": 3, "S4": 4, "H6": 6}

# Functional group types
FUNC_GROUPS = ["CHO", "NH2", "COOH", "CN", "OH", "Cl", "Br", "CH3"]


class COFDesignEnvV2(gym.Env):
    """
    3-Agent COF design environment with diffusion model integration.

    State space: 128-dim embedding (from existing TransformerEncoder)
    Action spaces:
    - Agent 0: MultiDiscrete([11, 7])  → topology + stacking
    - Agent 1: MultiDiscrete([4, 6, 8]) → symmetry + connector + func_group
    - Agent 2: MultiDiscrete([4, 6, 8]) → symmetry + connector + func_group
    """

    def __init__(self, args, diffusion_wrapper=None):
        super().__init__()
        self.n = 3  # 3 agents
        self.max_step = getattr(args, 'episode_limit', 4)
        self.current_step = 0

        # Observation space: 128-dim per agent
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32
        )
        self.share_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(128 * self.n,), dtype=np.float32
        )

        # Per-agent action spaces
        self.action_space = [
            spaces.MultiDiscrete([len(TOPOLOGIES), len(STACKING_MODES)]),
            spaces.MultiDiscrete([len(SYMMETRY_TYPES), 6, len(FUNC_GROUPS)]),
            spaces.MultiDiscrete([len(SYMMETRY_TYPES), 6, len(FUNC_GROUPS)]),
        ]

        # Diffusion wrapper (injected after initialization)
        self.diffusion_wrapper = diffusion_wrapper

        # Transformer for state encoding (reuse existing)
        self.embedding_layer = None
        self.transformer_model = None
        self.vocab = None

        # Episode state
        self.current_design = {
            "topology": None,
            "stacking": None,
            "bb_a": None,
            "bb_b": None,
        }
        self.last_reward = 0.0

    def set_diffusion_wrapper(self, wrapper):
        """Inject the diffusion generator wrapper."""
        self.diffusion_wrapper = wrapper

    def set_embedding(self, embedding_layer, transformer_model, vocab):
        """Inject pre-trained embedding components from existing env.py."""
        self.embedding_layer = embedding_layer
        self.transformer_model = transformer_model
        self.vocab = vocab

    def reset(self):
        """Reset environment state."""
        self.current_step = 0
        self.current_design = {
            "topology": None, "stacking": None,
            "bb_a": None, "bb_b": None,
        }
        self.last_reward = 0.0

        # Initial observations (128-dim zero vectors)
        observations = np.zeros((self.n, 128), dtype=np.float32)

        # Action masks: all actions available initially
        action_masks = [
            np.ones(sum(self.action_space[i].nvec), dtype=np.float32)
            for i in range(self.n)
        ]

        return observations, self.current_design, action_masks

    def step(self, actions, episode_num):
        """
        Execute one step of the environment.

        Args:
            actions: list of 3 action tuples/arrays
            episode_num: current episode number

        Returns:
            (observations, rewards, dones, info)
        """
        # Parse actions
        topo_idx, stack_idx = actions[0][0], actions[0][1]
        sym_a_idx, conn_a, fg_a_idx = actions[1][0], actions[1][1], actions[1][2]
        sym_b_idx, conn_b, fg_b_idx = actions[2][0], actions[2][1], actions[2][2]

        topology = TOPOLOGIES[topo_idx % len(TOPOLOGIES)]
        stacking = STACKING_MODES[stack_idx % len(STACKING_MODES)]
        sym_a = SYMMETRY_TYPES[sym_a_idx % len(SYMMETRY_TYPES)]
        sym_b = SYMMETRY_TYPES[sym_b_idx % len(SYMMETRY_TYPES)]
        fg_a = FUNC_GROUPS[fg_a_idx % len(FUNC_GROUPS)]
        fg_b = FUNC_GROUPS[fg_b_idx % len(FUNC_GROUPS)]

        # Override connectors based on symmetry
        conn_a = SYMMETRY_CONNECTORS[sym_a]
        conn_b = SYMMETRY_CONNECTORS[sym_b]

        self.current_design = {
            "topology": topology,
            "stacking": stacking,
            "sym_a": sym_a, "conn_a": conn_a, "fg_a": fg_a,
            "sym_b": sym_b, "conn_b": conn_b, "fg_b": fg_b,
        }

        # Generate building blocks via diffusion model
        bb_a_paths = []
        bb_b_paths = []
        reward = -1.0

        if self.diffusion_wrapper is not None:
            bb_a_paths = self.diffusion_wrapper.get_or_generate(
                sym_a, conn_a, fg_a, num_samples=3
            )
            bb_b_paths = self.diffusion_wrapper.get_or_generate(
                sym_b, conn_b, fg_b, num_samples=3
            )

            # Try to assemble COF and compute reward
            if bb_a_paths and bb_b_paths:
                from symmcd_diffusion.marl_interface.reward_bridge import RewardBridge
                bridge = RewardBridge()
                reward, info = bridge.compute_reward(
                    bb_a_paths[0], bb_b_paths[0],
                    sym_a, sym_b, fg_a, fg_b,
                    topology, stacking, episode_num,
                )
            else:
                reward = -1.0
        else:
            # No diffusion wrapper — use random reward for testing
            reward = np.random.uniform(-1, 5)

        self.last_reward = reward
        self.current_step += 1

        # Generate next observations (use topology embedding as signal)
        # In practice, this would use the TransformerEncoder on the design state
        observations = np.zeros((self.n, 128), dtype=np.float32)
        # Encode topology index in first 11 dims, stacking in next 7
        observations[0, topo_idx % 11] = 1.0
        observations[0, 11 + stack_idx % 7] = 1.0

        # All agents share the same reward
        rewards = np.array([reward, reward, reward])

        # Episode ends after one design cycle
        dones = np.array([True, True, True])

        action_masks = [np.ones(s, dtype=np.float32) for s in [
            sum(self.action_space[i].nvec) for i in range(self.n)
        ]]

        info = {
            "design": self.current_design,
            "reward": reward,
            "bb_a_paths": bb_a_paths,
            "bb_b_paths": bb_b_paths,
        }

        return observations, rewards, dones, info

    def get_state(self):
        """Get global state for centralized critic."""
        return np.zeros(128 * self.n, dtype=np.float32)
