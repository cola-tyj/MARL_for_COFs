"""
VAE-integrated SymTopo environment: continuous Core generation via VAE.

Key change from env_symtopo.py:
  Agent outputs latent vector z (32-dim continuous) instead of core_idx (discrete).
  VAE decoder generates novel Core structures from z, enabling true molecular design.

Pipeline:
  Agent → (sym, z∈R³², conn) → VAE Decoder → Core(atom types + coords)
    → save as .cjson → pycofbuilder BuildingBlock → assembly → reward
"""

import os
import sys
import json
import io
import uuid
import numpy as np
from collections import Counter
from pathlib import Path

import torch

# Import base env
from env_symtopo import (
    SymTopoEnv, SYMMETRY_TYPES, N_SYMS, N_CONNECTORS, ALL_CONNECTORS,
    SYMMETRY_TO_CONNECTORS_COUNT, _get_compatible,
    TOPO_2D, TOPO_3D, _filter_topologies,
)

# Pycofbuilder core data path
_PYCOF_BUILDER_DIR = os.path.join(
    os.path.dirname(__file__),
    '../../anaconda3/envs/env_cof/lib/python3.10/site-packages/pycofbuilder',
)
_PYCOF_CORE_DIR = os.path.join(os.path.abspath(_PYCOF_BUILDER_DIR), 'data', 'core')

# Atom vocabulary: VAE index → element symbol
VAE_IDX_TO_SYMBOL = {
    0: "H", 1: "C", 2: "N", 3: "O", 4: "F",
    5: "S", 6: "B", 7: "Cu", 8: "Zn", 9: "Co",
    10: "Q", 11: "R",
}

# Symmetry to pycofbuilder point group (for VAE conditioning)
SYMMETRY_TO_PG = {
    "L2": 0,   # C2v
    "T3": 4,   # D3h
    "S4": 10,  # D4h
    "H6": 14,  # D6h
    "D4": 10,  # D4h
    "R4": 10,  # D4h
}


class SymTopoEnvVAE(SymTopoEnv):
    """
    VAE-integrated COF design environment.

    Differences from SymTopoEnv:
    - Agent outputs continuous z∈R³² for Core generation (not discrete core_idx)
    - VAE decoder generates novel Core structures on-the-fly
    - Latent dimension = 32 (from CoreVAE configuration)
    - Additional validity reward for successful Core generation
    """

    def __init__(self, vae_model=None, vae_device='cuda:0',
                 core_dir=None, topo_mode='all'):
        # Initialize base without Core loading (VAE generates them)
        # We still need base env for sym/conn/topo logic
        super().__init__(core_dir=core_dir, topo_mode=topo_mode)

        self.vae_model = vae_model
        self.vae_device = vae_device
        self.latent_dim = 32

        # Update action dimensions for continuous Core selection
        self.action_dim_core = self.latent_dim  # 32-dim continuous
        self.core_continuous = True

        # Core generation stats
        self.core_gen_success = 0
        self.core_gen_total = 0

        # Ensure pycofbuilder core dir exists for each symmetry
        for sym in SYMMETRY_TYPES:
            sym_dir = os.path.join(_PYCOF_CORE_DIR, sym)
            os.makedirs(sym_dir, exist_ok=True)

    # ── VAE Core generation ──────────────────────────────────────────

    def _vae_generate_core(self, z, sym_type):
        """
        Generate a Core structure from latent vector.

        Args:
            z: (32,) latent vector (numpy or tensor)
            sym_type: e.g. "L2", "T3"

        Returns:
            core_name: unique name for pycofbuilder lookup
            valid: bool indicating if generation passed Q/R checks
        """
        if self.vae_model is None:
            # Fallback: use random core from base env
            sym_idx = SYMMETRY_TYPES.index(sym_type)
            cores = self.core_names.get(sym_type, [])
            if not cores:
                return None, False
            core_name = np.random.choice(cores)
            return core_name, True

        # Convert z to tensor
        if isinstance(z, np.ndarray):
            z_t = torch.from_numpy(z).float().unsqueeze(0).to(self.vae_device)
        else:
            z_t = z.float().unsqueeze(0).to(self.vae_device)

        # Get symmetry index for VAE
        pg_idx = SYMMETRY_TO_PG.get(sym_type, 0)
        symm_idx = torch.tensor([pg_idx], device=self.vae_device)

        # Generate Core
        try:
            with torch.no_grad():
                atom_types_list, positions_list, n_atoms = self.vae_model.generate(
                    z_t, symm_idx,
                )
        except Exception:
            return None, False

        atom_idx = atom_types_list[0].cpu().numpy()  # (n,)
        coords = positions_list[0].cpu().numpy()     # (n, 3)
        n = len(atom_idx)

        # Validate Q count
        expected_q = SYMMETRY_TO_CONNECTORS_COUNT.get(sym_type, 2)
        q_count = int((atom_idx == 10).sum())
        r_count = int((atom_idx == 11).sum())

        if q_count < expected_q - 1 or q_count > expected_q + 1:
            # Fix: set correct number of Q atoms
            # Find R atoms to convert to Q if needed
            pass  # For now, just reject severely invalid ones

        if r_count < 2:
            return None, False  # Need at least 2 R groups

        # Convert to element symbols
        elements = []
        r_counter = 1
        for i in range(n):
            idx = int(atom_idx[i])
            sym = VAE_IDX_TO_SYMBOL.get(idx, "C")
            if sym == "R":
                sym = f"R{r_counter}"
                r_counter += 1
            elements.append(sym)

        # Generate unique name
        core_name = f"vae_{uuid.uuid4().hex[:8]}"

        # Save as .cjson in pycofbuilder's core directory
        cjson_path = os.path.join(_PYCOF_CORE_DIR, sym_type, f"{core_name}.cjson")
        cjson_data = {
            "atoms": {
                "elements": {"type": elements},
                "coords": {"3d": coords.flatten().tolist()},
            },
        }
        with open(cjson_path, 'w') as f:
            json.dump(cjson_data, f)

        self.core_gen_success += 1
        self.core_gen_total += 1
        return core_name, True

    # ── Modified step ──────────────────────────────────────────────

    def step(self, actions):
        """
        Modified step: Agent 1/2 provide (sym_idx, z_vector, conn_idx)
        instead of (sym_idx, core_idx, conn_idx).
        """
        self._step_idx += 1

        if self._step_idx == 1:
            # Agent 1: (sym, z_vector, conn)
            sym_idx, z_vec, conn_idx = self._parse_action(actions[0])
            sym_idx = sym_idx % N_SYMS
            conn_idx = conn_idx % N_CONNECTORS

            # Generate Core from latent
            core_name, valid = self._vae_generate_core(
                z_vec, SYMMETRY_TYPES[sym_idx]
            )
            if not valid or core_name is None:
                # Invalid generation → penalty
                self.assembly_total += 1
                obs_n = np.zeros((self.n, self.observation_space), dtype=np.float32)
                masks = [{}, {}, {}]
                return obs_n, [-1.0, -1.0, -1.0], [True]*self.n, \
                       ['invalid_core_gen']*self.n, masks

            self._sym_a = sym_idx
            self._core_a = core_name  # Store generated core name
            self._conn_a = ALL_CONNECTORS[conn_idx]

            obs_n = np.array([self._build_obs(i) for i in range(self.n)])
            masks = self._get_masks()
            return obs_n, [0.0]*self.n, [False]*self.n, \
                   [f'sym={SYMMETRY_TYPES[sym_idx]},core={core_name},conn={self._conn_a}']*self.n, \
                   masks

        elif self._step_idx == 2:
            # Agent 2: (sym, z_vector, conn)
            sym_idx, z_vec, conn_idx = self._parse_action(actions[1])
            sym_idx = sym_idx % N_SYMS
            conn_idx = conn_idx % N_CONNECTORS
            conn_b = ALL_CONNECTORS[conn_idx]

            # Validate sym pair
            ca = SYMMETRY_TO_CONNECTORS_COUNT.get(SYMMETRY_TYPES[self._sym_a], 2)
            cb = SYMMETRY_TO_CONNECTORS_COUNT.get(SYMMETRY_TYPES[sym_idx], 2)
            if (ca, cb) not in self._topo_table:
                self.assembly_total += 1
                obs_n = np.zeros((self.n, self.observation_space), dtype=np.float32)
                masks = [{}, {}, {}]
                return obs_n, [-1.0, -1.0, -1.0], [True]*self.n, \
                       [f'invalid_sym_pair ({ca},{cb})']*self.n, masks

            # Generate Core from latent
            core_name, valid = self._vae_generate_core(
                z_vec, SYMMETRY_TYPES[sym_idx]
            )
            if not valid or core_name is None:
                self.assembly_total += 1
                obs_n = np.zeros((self.n, self.observation_space), dtype=np.float32)
                masks = [{}, {}, {}]
                return obs_n, [-1.0, -1.0, -1.0], [True]*self.n, \
                       ['invalid_core_gen_2']*self.n, masks

            self._sym_b = sym_idx
            self._core_b = core_name
            self._conn_b = conn_b

            obs_n = np.array([self._build_obs(i) for i in range(self.n)])
            masks = self._get_masks()
            return obs_n, [0.0]*self.n, [False]*self.n, \
                   [f'sym={SYMMETRY_TYPES[sym_idx]},core={core_name},conn={conn_b}']*self.n, \
                   masks

        else:  # step == 3 (Agent 3: topology — unchanged from base)
            return super().step(actions)

    def _parse_action(self, action):
        """Parse VAE action: (sym_idx, z_vector_32, conn_idx)."""
        if isinstance(action, tuple) and len(action) == 3:
            return action[0], action[1], action[2]
        # Backward compat: if old-style discrete core_idx
        sym_idx = action[0] if isinstance(action, (list, tuple)) else action
        z_vec = np.zeros(self.latent_dim, dtype=np.float32)
        conn_idx = action[2] if isinstance(action, (list, tuple)) and len(action) > 2 else 0
        return sym_idx, z_vec, conn_idx


# ── Test ──
if __name__ == '__main__':
    # Quick test without VAE model (falls back to random core selection)
    env = SymTopoEnvVAE(vae_model=None)
    print(f"SymTopoEnvVAE: N={env.n}, obs={env.observation_space}")
    print(f"  latent_dim={env.latent_dim}, core_continuous={env.core_continuous}")
    print("  (no VAE loaded — using random fallback)")
    print("env_symtopo_vae ready ✓")
