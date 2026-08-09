"""
Diffusion-integrated SymTopo environment.

Inherits from SymTopoEnv, overrides Core selection:
  Agent outputs noise_level ∈ [0,1] (continuous) instead of core_idx (discrete).
  Scaffold diffusion generates new Core geometry while keeping Q/R fixed.

Pipeline:
  Agent → (sym, noise_level, conn)
       → pick best Core template for sym
       → scaffold diffusion with noise_level
       → overwrite coords in pycofbuilder
       → BuildingBlock.from_name() → assembly
"""

import os, sys, json, io, shutil
import numpy as np
import torch

from env_symtopo import (
    SymTopoEnv, SYMMETRY_TYPES, N_SYMS, N_CONNECTORS, ALL_CONNECTORS,
    SYMMETRY_TO_CONNECTORS_COUNT, _get_compatible, _filter_topologies,
)


# Best Core templates per symmetry (exist in both valid_cores and pycofbuilder)
BEST_TEMPLATES = {
    "L2": "DPDA", "T3": "BRZN", "S4": "OTPR",
    "D4": "CUBA", "H6": "HECO", "R4": "ETKB",
}

# Symmetry → point group for diffusion conditioning
SYM_TO_PG = {"L2": 0, "T3": 4, "S4": 10, "H6": 14, "D4": 10, "R4": 10}

# Atom type index mapping (same as CoreDataset)
ATOM_MAP = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'B': 6, 'Q': 10}


class SymTopoEnvDiff(SymTopoEnv):
    """
    Diffusion-integrated COF design environment.

    Differences from SymTopoEnv:
    - Agent outputs continuous noise_level ∈ [0,1] for Core generation
    - Scaffold diffusion modifies scaffold coords, Q/R positions fixed
    - Uses Phase 2 denoiser for coordinate refinement
    """

    def __init__(self, diffusion_model=None, diffusion_process=None,
                 core_dir=None, topo_mode='all', device='cuda:2'):
        super().__init__(core_dir=core_dir, topo_mode=topo_mode)

        self.diffusion_model = diffusion_model
        self.diffusion_process = diffusion_process
        self.device = device

        # Continuous Core selection
        self.core_continuous = True
        self.noise_dim = 1  # 1D continuous noise_level
        self.action_dim_core = self.noise_dim

        # Map core names → indices for _build_obs compatibility
        self._core_name_to_idx = {}
        for sym in SYMMETRY_TYPES:
            names = self.core_names.get(sym, [])
            self._core_name_to_idx[sym] = {n: i for i, n in enumerate(names)}

        # Core generation stats
        self.diffusion_calls = 0
        self.diffusion_rmsds = []

        # Pycofbuilder core path (where we write diffused coords)
        self.pb_core_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'pycofbuilder', 'data', 'core'
        )

        # Backup original Cores
        self._core_backups = {}

    # ── Diffusion Core generation ────────────────────────────────────

    def _diffuse_core(self, sym_type, noise_level):
        """
        Generate scaffold-diffused Core for given symmetry.

        Args:
            sym_type: e.g. "L2", "T3"
            noise_level: float in [0, 1]

        Returns:
            core_name: name of the Core (same template, coords overwritten)
            rmsd: scaffold RMSD from original (for logging)
        """
        core_name = BEST_TEMPLATES.get(sym_type)
        if not core_name:
            # Fallback: pick first valid core for this sym
            cores = self.valid_cores.get(sym_type, self.core_names.get(sym_type, []))
            core_name = cores[0] if cores else 'DPDA'

        pb_path = os.path.join(self.pb_core_dir, sym_type, f'{core_name}.cjson')

        # Backup on first call
        if core_name not in self._core_backups:
            if os.path.exists(pb_path):
                with open(pb_path) as f:
                    self._core_backups[core_name] = f.read()

        # Load cjson
        with open(pb_path) as f:
            cjson = json.load(f)

        elements = cjson['atoms']['elements']['type']
        coords = np.array(cjson['atoms']['coords']['3d']).reshape(-1, 3)
        n = len(elements)

        # Map atoms
        atoms = np.array([ATOM_MAP.get(e[0] if e.startswith('R') else e, 1)
                         for e in elements])
        is_qr = (atoms == 10) | (atoms == 11)

        if noise_level > 0 and self.diffusion_model is not None:
            # Scaffold diffusion
            x0 = torch.from_numpy(coords).float().to(self.device)
            x_oh = torch.zeros(n, 12, device=self.device)
            for i, a in enumerate(atoms):
                x_oh[i, a] = 1.0

            from symmcd_diffusion.models.diffusion_process import build_fully_connected_edges
            ei = build_fully_connected_edges(x0)
            ea = torch.zeros(ei.size(1), 5, device=self.device)
            ea[:, 1] = 1.0

            from symmcd_diffusion.symmetry.symmetry_encoder import POINT_GROUP_TO_IDX
            pg_mapping = {'L2': 'C2v', 'T3': 'D3h', 'S4': 'D4h',
                         'H6': 'D6h', 'D4': 'D4h', 'R4': 'D4h'}
            pg_idx = POINT_GROUP_TO_IDX.get(pg_mapping.get(sym_type, 'C2v'), 0)
            cond = self.diffusion_model.encode_condition(
                torch.tensor([pg_idx], device=self.device))

            qr_idx = torch.from_numpy(np.where(is_qr)[0]).to(self.device)

            t = int(noise_level * 999)
            alpha = self.diffusion_process.scheduler.continuous_schedule
            ab_t = alpha.alphas_cumprod[t].to(self.device)

            noise = torch.randn_like(x0)
            xt = ab_t.sqrt() * x0 + (1 - ab_t).sqrt() * noise

            with torch.no_grad():
                pred = self.diffusion_model.denoiser(
                    x_oh, xt, torch.tensor([t], device=self.device),
                    ei, ea, condition=cond)

            x0_pred = (xt - (1 - ab_t).sqrt() *
                      pred['coord_noise'].detach()) / ab_t.sqrt()
            x0_pred[qr_idx] = x0[qr_idx]
            coord_new = x0_pred.detach().cpu().numpy()

            # Track stats
            scaffold_idx = np.where(~is_qr)[0]
            if len(scaffold_idx) > 0:
                rmsd = np.sqrt(
                    ((coord_new[scaffold_idx] - coords[scaffold_idx])**2).mean())
            else:
                rmsd = 0.0
            self.diffusion_rmsds.append(rmsd)
        else:
            coord_new = coords.copy()
            rmsd = 0.0

        # Overwrite pycofbuilder's copy
        cjson['atoms']['coords']['3d'] = coord_new.flatten().tolist()
        with open(pb_path, 'w') as f:
            json.dump(cjson, f)

        self.diffusion_calls += 1
        return core_name, rmsd

    def restore_cores(self):
        """Restore all modified Cores to original state."""
        for core_name, backup in self._core_backups.items():
            # Find which sym dir contains this core
            for sym in SYMMETRY_TYPES:
                pb_path = os.path.join(self.pb_core_dir, sym, f'{core_name}.cjson')
                if os.path.exists(pb_path):
                    with open(pb_path, 'w') as f:
                        f.write(backup)
                    break

    # ── Modified step ──────────────────────────────────────────────

    def step(self, actions):
        """
        Modified step: Agent 1/2 provide (sym_idx, noise_level, conn_idx).
        noise_level is a float ∈ [0, 1] controlling scaffold diffusion strength.
        """
        self._step_idx += 1

        if self._step_idx == 1:
            sym_idx, noise_level, conn_idx = self._parse_action(actions[0])
            sym_idx = sym_idx % N_SYMS
            conn_idx = conn_idx % N_CONNECTORS
            noise_level = float(np.clip(noise_level, 0.0, 1.0))

            # Generate Core via diffusion
            core_name, rmsd = self._diffuse_core(
                SYMMETRY_TYPES[sym_idx], noise_level)

            self._sym_a = sym_idx
            self._core_a = core_name
            # Also store idx for _build_obs compatibility
            sym_name = SYMMETRY_TYPES[sym_idx]
            self._core_a_idx = self._core_name_to_idx.get(sym_name, {}).get(core_name, 0)
            self._conn_a = ALL_CONNECTORS[conn_idx]

            obs_n = np.array([self._build_obs(i) for i in range(self.n)])
            masks = self._get_masks()
            return obs_n, [0.0]*self.n, [False]*self.n, \
                   [f'sym={sym_name},diff_nl={noise_level:.2f},rmsd={rmsd:.2f}']*self.n, \
                   masks

        elif self._step_idx == 2:
            sym_idx, noise_level, conn_idx = self._parse_action(actions[1])
            sym_idx = sym_idx % N_SYMS
            conn_idx = conn_idx % N_CONNECTORS
            conn_b = ALL_CONNECTORS[conn_idx]
            noise_level = float(np.clip(noise_level, 0.0, 1.0))

            # Validate sym pair
            ca = SYMMETRY_TO_CONNECTORS_COUNT.get(SYMMETRY_TYPES[self._sym_a], 2)
            cb = SYMMETRY_TO_CONNECTORS_COUNT.get(SYMMETRY_TYPES[sym_idx], 2)
            if (ca, cb) not in self._topo_table:
                self.assembly_total += 1
                obs_n = np.zeros((self.n, self.observation_space), dtype=np.float32)
                masks = [{}, {}, {}]
                return obs_n, [-1.0, -1.0, -1.0], [True]*self.n, \
                       [f'invalid_sym_pair ({ca},{cb})']*self.n, masks

            # Generate Core via diffusion
            core_name, rmsd = self._diffuse_core(
                SYMMETRY_TYPES[sym_idx], noise_level)

            self._sym_b = sym_idx
            self._core_b = core_name
            sym_name_b = SYMMETRY_TYPES[sym_idx]
            self._core_b_idx = self._core_name_to_idx.get(sym_name_b, {}).get(core_name, 0)
            self._conn_b = conn_b

            obs_n = np.array([self._build_obs(i) for i in range(self.n)])
            masks = self._get_masks()
            return obs_n, [0.0]*self.n, [False]*self.n, \
                   [f'sym={SYMMETRY_TYPES[sym_idx]},diff_nl={noise_level:.2f},rmsd={rmsd:.2f}']*self.n, \
                   masks

        else:  # step == 3 — same as base but use template core names
            # The diffusion env always uses BEST_TEMPLATES cores.
            # Override step 3 to use the correct core names.
            topo_idx = actions[2] if isinstance(actions[2], int) else actions[2][0]
            sym_a_name = SYMMETRY_TYPES[self._sym_a]
            sym_b_name = SYMMETRY_TYPES[self._sym_b]
            valid_topos = self.get_valid_topologies(sym_a_name, sym_b_name)
            topo = valid_topos[topo_idx % len(valid_topos)]

            # Use the template names that _diffuse_core wrote to
            core_a = self._core_a if isinstance(self._core_a, str) else \
                     self.valid_cores.get(sym_a_name, [self._core_a])[self._core_a % 1]
            core_b = self._core_b if isinstance(self._core_b, str) else \
                     self.valid_cores.get(sym_b_name, [self._core_b])[self._core_b % 1]

            # Assembly
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

                # Reward (same as base)
                if fw.name in self.n2_cache:
                    n2_val = self.n2_cache[fw.name]
                else:
                    cell_vol = (fw.cellParameters[0] * fw.cellParameters[1] *
                                fw.cellParameters[2])
                    density = len(fw.atom_types) / max(cell_vol, 1.0)
                    n2_val = 1.5 + 0.5 * max(0, (0.015 - density) * 1000)

                stats = self.topo_n2_stats[topo]
                stats['n'] += 1
                delta = n2_val - stats['mean']
                stats['mean'] += delta / stats['n']
                delta2 = n2_val - stats['mean']
                stats['M2'] += delta * delta2
                stats['M2'] = max(stats['M2'], 0.0)

                if stats['n'] >= 5:
                    var = stats['M2'] / max(stats['n'] - 1, 1)
                    std = np.sqrt(max(var, 0.0))
                    z_n2 = (n2_val - stats['mean']) / max(std, 0.01)
                    z_n2 = float(np.clip(z_n2, -3.0, 3.0))
                else:
                    z_n2 = 0.0

                total = sum(self.topo_counts.values()) + 1
                combo_key = (sym_a_name, sym_b_name, self._conn_a, self._conn_b, topo)
                combo_cnt = self.combo_counts.get(combo_key, 0) + 1
                ucb = np.sqrt(5.0 * np.log(max(total, 1)) / max(combo_cnt, 1))
                topo_cnt = self.topo_counts.get(topo, 0) + 1
                topo_bonus = 3.0 * np.log(max(total, 1)) / max(topo_cnt, 1)
                reward = z_n2 * 5.0 + ucb * 5.0 + topo_bonus * 5.0

                self.assembly_success += 1
                self.topo_counts[topo] += 1
                self.conn_pair_counts[(self._conn_a, self._conn_b)] += 1
                self.combo_counts[combo_key] = combo_cnt

                cif_path = os.path.join(self.cif_cache_dir, f"{fw.name}.cif")
                try:
                    fw.save(fmt='cif', supercell=[1, 1, 1], save_dir=self.cif_cache_dir)
                except Exception:
                    cif_path = None

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
                           f'N2={n2_val:.2f},z={z_n2:+.2f}')
            except Exception as e:
                err = str(e)
                if 'connectivity' in err.lower():
                    reward = 1.0
                elif 'from_building_blocks' in err.lower():
                    reward = 3.0
                else:
                    reward = 2.0
                info_str = (f'fail:{topo}:{sym_a_name}+{sym_b_name}:'
                           f'{self._conn_a}+{self._conn_b}:{err[:40]}')
                self.topo_counts[topo] = self.topo_counts.get(topo, 0) + 1
            finally:
                sys.stdout = old_out; sys.stderr = old_err

            self.assembly_total += 1
            rewards = [reward, reward, reward]
            obs_n = np.zeros((self.n, self.observation_space), dtype=np.float32)
            obs_n[:, -1] = min(max(reward, 0.0) / 40.0, 1.0)
            masks = [{}, {}, {}]
            return obs_n, rewards, [True]*self.n, [info_str]*self.n, masks

    def _build_obs(self, agent_id):
        """Override: map core_name → core_idx for feature lookup."""
        # Temporarily map string core names to indices
        orig_a, orig_b = self._core_a, self._core_b
        if isinstance(self._core_a, str) and hasattr(self, '_core_a_idx'):
            self._core_a = self._core_a_idx
        if isinstance(self._core_b, str) and hasattr(self, '_core_b_idx'):
            self._core_b = self._core_b_idx
        obs = super()._build_obs(agent_id)
        self._core_a, self._core_b = orig_a, orig_b
        return obs

    def _parse_action(self, action):
        """Parse diffusion action: (sym_idx, noise_level, conn_idx)."""
        if isinstance(action, (list, tuple)) and len(action) == 3:
            return action[0], action[1], action[2]
        # Backward compat: discrete core_idx
        sym_idx = action[0] if isinstance(action, (list, tuple)) else action
        noise_level = 0.0
        conn_idx = action[2] if isinstance(action, (list, tuple)) and len(action) > 2 else 0
        return sym_idx, noise_level, conn_idx

    def close(self):
        """Restore original Cores and cleanup."""
        self.restore_cores()
        super().close()
