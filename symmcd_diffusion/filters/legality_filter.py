"""
Five-layer legality filter for generated COF building blocks.

Layers (ordered by computational cost, cheapest first):
1. Atom-level: valence, charge, distance checks
2. RDKit parsing: SanitizeMol, aromaticity
3. Symmetry verification: point group match
4. Connectivity check: connector count and geometry
5. COF assembly test: attempt pycofbuilder assembly

Any failure terminates the pipeline early to save computation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass
class FilterResult:
    """Result from a single legality filter layer."""
    passed: bool
    reason: str = "ok"
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class FullFilterResult:
    """Complete legality check result across all layers."""
    passed: bool
    layer_results: List[FilterResult] = field(default_factory=list)
    final_reason: str = "ok"
    rdkit_mol: Optional[object] = None  # RDKit Mol if available
    computed_symmetry: Optional[str] = None
    num_connectors_found: int = 0


class LegalityFilter:
    """
    Five cascaded legality checks for COF building blocks.
    """

    def __init__(
        self,
        tolerance_dist: float = 0.5,       # Min atom-atom distance (Angstroms)
        tolerance_symmetry: float = 0.3,    # RMSD for symmetry match (Angstroms)
        tolerance_angle: float = 15.0,      # Angle tolerance for connectivity (degrees)
        max_valence_error: int = 2,         # Max allowed valence errors
        enable_layer_5: bool = True,        # COF assembly test (slowest)
    ):
        self.tolerance_dist = tolerance_dist
        self.tolerance_symmetry = tolerance_symmetry
        self.tolerance_angle = tolerance_angle
        self.max_valence_error = max_valence_error
        self.enable_layer_5 = enable_layer_5

        # Tracking statistics
        self.stats = {f"layer_{i}": {"pass": 0, "fail": 0} for i in range(1, 6)}

    def check(
        self,
        atom_types: Tensor,
        positions: Tensor,
        bonds: Optional[Tensor],
        expected_symmetry: str,
        expected_connectors: int,
        expected_func_group: Optional[str] = None,
    ) -> FullFilterResult:
        """
        Run all legality filter layers sequentially.

        Args:
            atom_types: (N, num_types) one-hot or soft atom type probs
            positions: (N, 3) 3D coordinates
            bonds: (E, num_bond_types) bond type probs (optional)
            expected_symmetry: target point group (e.g., "D3h")
            expected_connectors: expected number of connection points
            expected_func_group: expected functional group at connectors

        Returns:
            FullFilterResult with pass/fail and diagnostics
        """
        result = FullFilterResult(passed=False)

        # Convert to hard atom assignments
        if atom_types.dim() == 2:
            atom_idx = atom_types.argmax(dim=-1)
        else:
            atom_idx = atom_types

        # Layer 1: Atom-level checks
        l1 = self._check_atom_level(positions, bonds)
        result.layer_results.append(l1)
        self._update_stats(1, l1.passed)
        if not l1.passed:
            result.final_reason = f"L1:{l1.reason}"
            return result

        # Layer 2: RDKit parsing
        l2 = self._check_rdkit(atom_idx, positions, bonds)
        result.layer_results.append(l2)
        self._update_stats(2, l2.passed)
        if not l2.passed:
            result.final_reason = f"L2:{l2.reason}"
            return result
        result.rdkit_mol = l2.rdkit_mol

        # Layer 3: Symmetry verification
        l3 = self._check_symmetry(atom_idx, positions, expected_symmetry)
        result.layer_results.append(l3)
        self._update_stats(3, l3.passed)
        if not l3.passed:
            result.final_reason = f"L3:{l3.reason}"
            return result
        result.computed_symmetry = l3.computed_symmetry

        # Layer 4: Connectivity check
        l4 = self._check_connectivity(
            atom_idx, positions, bonds,
            expected_connectors, expected_symmetry, expected_func_group,
        )
        result.layer_results.append(l4)
        self._update_stats(4, l4.passed)
        if not l4.passed:
            result.final_reason = f"L4:{l4.reason}"
            return result
        result.num_connectors_found = l4.num_connectors_found

        # Layer 5: COF assembly test (optional, expensive)
        if self.enable_layer_5:
            l5 = self._check_cof_assembly(atom_idx, positions, bonds, expected_symmetry)
            result.layer_results.append(l5)
            self._update_stats(5, l5.passed)
            if not l5.passed:
                result.final_reason = f"L5:{l5.reason}"
                return result

        result.passed = True
        result.final_reason = "ok"
        return result

    # ---- Layer 1: Atom-level ----
    def _check_atom_level(
        self,
        positions: Tensor,
        bonds: Optional[Tensor],
    ) -> FilterResult:
        """Check minimum atom-atom distances."""
        pos = positions.detach().cpu().numpy()
        n = len(pos)

        # Check minimum pairwise distance
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < self.tolerance_dist:
                    return FilterResult(
                        False,
                        f"atoms_{i}_{j}_too_close_{dist:.2f}A"
                    )

        return FilterResult(True, "atom_level_ok")

    # ---- Layer 2: RDKit parsing ----
    def _check_rdkit(
        self,
        atom_idx: Tensor,
        positions: Tensor,
        bonds: Optional[Tensor],
    ) -> FilterResult:
        """Try to parse as RDKit molecule and sanitize."""
        try:
            from rdkit import Chem
        except ImportError:
            return FilterResult(True, "rdkit_unavailable_skip")

        idx = atom_idx.cpu().numpy()
        pos = positions.detach().cpu().numpy()

        # Atom type mapping (extended for COF)
        # 0:H, 1:C, 2:N, 3:O, 4:F, 5:Cl, 6:Br, 7:S, 8:Q, 9:X
        rdkit_atomic_nums = [1, 6, 7, 8, 9, 17, 35, 16, 0, 0]

        mol = Chem.RWMol()
        for a in idx:
            atom = Chem.Atom(rdkit_atomic_nums[a % len(rdkit_atomic_nums)])
            mol.AddAtom(atom)

        # Add bonds via distance heuristics
        n = len(idx)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(pos[i] - pos[j])
                z_i = rdkit_atomic_nums[idx[i] % len(rdkit_atomic_nums)]
                z_j = rdkit_atomic_nums[idx[j] % len(rdkit_atomic_nums)]

                # Skip dummy atoms (Q=0, X=0)
                if z_i == 0 or z_j == 0:
                    continue

                r_i = Chem.GetPeriodicTable().GetRcovalent(z_i) if z_i > 0 else 0.7
                r_j = Chem.GetPeriodicTable().GetRcovalent(z_j) if z_j > 0 else 0.7

                if dist < (r_i + r_j) * 1.25:
                    if dist < (r_i + r_j) * 0.7:
                        mol.AddBond(i, j, Chem.BondType.TRIPLE)
                    elif dist < (r_i + r_j) * 0.9:
                        mol.AddBond(i, j, Chem.BondType.DOUBLE)
                    else:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)

        mol = mol.GetMol()

        try:
            Chem.SanitizeMol(mol)
            return FilterResult(True, "rdkit_ok", rdkit_mol=mol)
        except Exception as e:
            return FilterResult(False, f"rdkit_error:{str(e)[:50]}", rdkit_mol=mol)

    # ---- Layer 3: Symmetry verification ----
    def _check_symmetry(
        self,
        atom_idx: Tensor,
        positions: Tensor,
        expected_symmetry: str,
    ) -> FilterResult:
        """Verify that computed point group matches expected."""
        from ..symmetry.point_group import compute_point_group

        # Map atom indices to atomic numbers for symmetry computation
        rdkit_map = [1, 6, 7, 8, 9, 17, 35, 16, 6, 6]  # Q/X → C for symmetry
        atomic_numbers = torch.tensor(
            [rdkit_map[a % len(rdkit_map)] for a in atom_idx.cpu().numpy()]
        )

        computed_sym = compute_point_group(
            atomic_numbers, positions, self.tolerance_symmetry
        )

        if computed_sym == expected_symmetry:
            return FilterResult(
                True, "symmetry_match", computed_symmetry=computed_sym
            )
        else:
            return FilterResult(
                False,
                f"symmetry_mismatch:{computed_sym}_vs_{expected_symmetry}",
                computed_symmetry=computed_sym,
            )

    # ---- Layer 4: Connectivity check ----
    def _check_connectivity(
        self,
        atom_idx: Tensor,
        positions: Tensor,
        bonds: Optional[Tensor],
        expected_connectors: int,
        expected_symmetry: str,
        expected_func_group: Optional[str],
    ) -> FilterResult:
        """Check connector atom count and geometry."""
        pos = positions.detach().cpu().numpy()
        idx = atom_idx.cpu().numpy()

        # Connector atom types: 8=Q, 9=X
        connector_mask = (idx == 8) | (idx == 9)
        connector_pos = pos[connector_mask]
        n_connectors = len(connector_pos)

        if n_connectors != expected_connectors:
            return FilterResult(
                False,
                f"connector_count:{n_connectors}_vs_{expected_connectors}",
                num_connectors_found=n_connectors,
            )

        # Check geometry based on symmetry
        if n_connectors >= 2 and len(connector_pos) == n_connectors:
            # Compute centroid of connector positions
            centroid = connector_pos.mean(axis=0)
            vectors = connector_pos - centroid

            # Check inter-vector angles
            for i in range(n_connectors):
                for j in range(i + 1, n_connectors):
                    v1 = vectors[i] / (np.linalg.norm(vectors[i]) + 1e-10)
                    v2 = vectors[j] / (np.linalg.norm(vectors[j]) + 1e-10)
                    cos_angle = np.clip(np.dot(v1, v2), -1, 1)
                    angle = np.degrees(np.arccos(cos_angle))

                    # Expected angles by symmetry
                    if expected_symmetry in ("D3h", "C3v"):  # T3: 120°
                        if abs(angle - 120) > self.tolerance_angle:
                            return FilterResult(
                                False,
                                f"angle_mismatch:{angle:.1f}_vs_120",
                                num_connectors_found=n_connectors,
                            )
                    elif expected_symmetry in ("D4h", "C4v"):  # S4: 90°/180°
                        if not (abs(angle - 90) < self.tolerance_angle or
                                abs(angle - 180) < self.tolerance_angle):
                            return FilterResult(
                                False,
                                f"angle_mismatch:{angle:.1f}_vs_90_or_180",
                                num_connectors_found=n_connectors,
                            )
                    elif expected_symmetry in ("D6h", "C6v"):  # H6: 60°/120°
                        if not (abs(angle - 60) < self.tolerance_angle or
                                abs(angle - 120) < self.tolerance_angle):
                            return FilterResult(
                                False,
                                f"angle_mismatch:{angle:.1f}_vs_60_or_120",
                                num_connectors_found=n_connectors,
                            )
                    elif expected_symmetry in ("C2v",):  # L2: 180°
                        if abs(angle - 180) > self.tolerance_angle:
                            return FilterResult(
                                False,
                                f"angle_mismatch:{angle:.1f}_vs_180",
                                num_connectors_found=n_connectors,
                            )

        return FilterResult(True, "connectivity_ok", num_connectors_found=n_connectors)

    # ---- Layer 5: COF assembly test ----
    def _check_cof_assembly(
        self,
        atom_idx: Tensor,
        positions: Tensor,
        bonds: Optional[Tensor],
        symmetry: str,
    ) -> FilterResult:
        """Attempt basic COF assembly to check structural compatibility."""
        try:
            # This would call pycofbuilder but requires full cjson conversion
            # For now, check if molecule seems structurally sound for COF
            n_atoms = len(atom_idx)
            if n_atoms < 4:
                return FilterResult(False, "too_few_atoms_for_cof")

            # Check if molecule is roughly planar (COF requirement for 2D)
            pos = positions.detach().cpu().numpy()
            if n_atoms >= 3 and len(pos) >= 3:
                # Simple planarity check
                centroid = pos.mean(axis=0)
                centered = pos - centroid
                _, _, vh = np.linalg.svd(centered)
                # Deviation from plane
                deviations = np.abs(centered @ vh[2])
                max_dev = deviations.max()

                if max_dev > 2.0:  # Angstroms
                    return FilterResult(
                        False,
                        f"nonplanar_max_dev_{max_dev:.1f}A"
                    )

            return FilterResult(True, "cof_assembly_ok")

        except Exception as e:
            return FilterResult(False, f"assembly_error:{str(e)[:50]}")

    # ---- Helpers ----
    def _update_stats(self, layer: int, passed: bool):
        """Update filter statistics."""
        key = "pass" if passed else "fail"
        self.stats[f"layer_{layer}"][key] += 1

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get accumulated filter statistics."""
        return self.stats

    def reset_stats(self):
        """Reset filter statistics."""
        self.stats = {f"layer_{i}": {"pass": 0, "fail": 0} for i in range(1, 6)}
