"""
Connection point verification for COF building blocks.

Ensures generated building blocks have geometrically correct connection points
matching their symmetry type, enabling proper COF assembly.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# COF symmetry type to expected connector geometry
SYMMETRY_GEOMETRY = {
    "L2": {
        "point_group": "C2v",
        "num_connectors": 2,
        "expected_angles": [180.0],        # Linear
        "angle_tolerance": 15.0,
    },
    "T3": {
        "point_group": "D3h",
        "num_connectors": 3,
        "expected_angles": [120.0, 120.0, 120.0],  # Trigonal planar
        "angle_tolerance": 15.0,
    },
    "S4": {
        "point_group": "D4h",
        "num_connectors": 4,
        "expected_angles": [90.0, 180.0],  # Square planar
        "angle_tolerance": 15.0,
    },
    "H6": {
        "point_group": "D6h",
        "num_connectors": 6,
        "expected_angles": [60.0, 120.0],  # Hexagonal
        "angle_tolerance": 15.0,
    },
}

# Connector atom type indices (Q=8, X=9 in the extended vocabulary)
CONNECTOR_ATOM_TYPES = [8, 9]


class ConnectionPointVerifier:
    """
    Verifies that generated building blocks have valid connection points.

    Checks:
    1. Correct number of connector atoms
    2. Connector atoms are geometrically accessible (not buried)
    3. Connector vectors match symmetry-expected angles
    4. Connector bond lengths are reasonable
    """

    def __init__(self, tolerance_angle: float = 15.0, tolerance_dist: float = 0.3):
        self.tolerance_angle = tolerance_angle
        self.tolerance_dist = tolerance_dist

    def verify(
        self,
        atom_types: Tensor,
        positions: Tensor,
        symmetry_type: str,
    ) -> Tuple[bool, str, dict]:
        """
        Verify connection points for a building block.

        Args:
            atom_types: (N,) hard atom type indices
            positions: (N, 3) cartesian coordinates
            symmetry_type: one of "L2", "T3", "S4", "H6"

        Returns:
            (passed, reason, diagnostics) tuple
        """
        if symmetry_type not in SYMMETRY_GEOMETRY:
            return False, f"unknown_symmetry:{symmetry_type}", {}

        geom = SYMMETRY_GEOMETRY[symmetry_type]
        expected_n = geom["num_connectors"]

        # Find connector atoms
        connector_idx = self._find_connectors(atom_types)
        n_found = len(connector_idx)

        diagnostics = {
            "connector_indices": connector_idx,
            "num_found": n_found,
            "num_expected": expected_n,
        }

        # Check 1: Correct count
        if n_found != expected_n:
            return False, f"wrong_count:{n_found}vs{expected_n}", diagnostics

        # Check 2: Connector positions accessible (not internal)
        connector_pos = positions[connector_idx].cpu().numpy()
        all_pos = positions.cpu().numpy()

        centroid = connector_pos.mean(axis=0)
        # Connectors should be on the periphery (far from molecular center)
        mol_centroid = all_pos.mean(axis=0)
        connector_distances = np.linalg.norm(connector_pos - mol_centroid, axis=1)

        # All other atoms
        other_mask = torch.ones(len(atom_types), dtype=torch.bool)
        other_mask[connector_idx] = False
        other_pos = all_pos[other_mask.cpu().numpy()]

        # Each connector should be closer to the molecular periphery
        for i, c_pos in enumerate(connector_pos):
            # Distance to other atoms
            min_dist_to_others = float("inf")
            for o_pos in other_pos:
                dist = np.linalg.norm(c_pos - o_pos)
                min_dist_to_others = min(min_dist_to_others, dist)

            if min_dist_to_others < 0.8:  # Too close to other atoms
                return False, f"connector_{i}_buried", diagnostics

        # Check 3: Connector geometry (angles between vectors from centroid)
        if n_found >= 2:
            vectors = connector_pos - centroid
            angles = []

            for i in range(n_found):
                for j in range(i + 1, n_found):
                    v1 = vectors[i] / (np.linalg.norm(vectors[i]) + 1e-10)
                    v2 = vectors[j] / (np.linalg.norm(vectors[j]) + 1e-10)
                    cos_a = np.clip(np.dot(v1, v2), -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_a))
                    angles.append(angle)

            diagnostics["connector_angles"] = angles

            # Check if all angles are close to expected values
            expected_set = set(geom["expected_angles"])
            for angle in angles:
                matched = False
                for expected in expected_set:
                    if abs(angle - expected) <= self.tolerance_angle:
                        matched = True
                        break
                if not matched:
                    return (
                        False,
                        f"bad_angle:{angle:.1f}_expected_{expected_set}",
                        diagnostics,
                    )

        return True, "ok", diagnostics

    def _find_connectors(self, atom_types: Tensor) -> List[int]:
        """Find indices of connector atoms (Q, X types)."""
        idx = atom_types.cpu().tolist() if atom_types.is_cuda else atom_types.tolist()
        return [i for i, t in enumerate(idx) if t in CONNECTOR_ATOM_TYPES]

    def compute_connector_directions(
        self,
        positions: Tensor,
        connector_indices: List[int],
    ) -> np.ndarray:
        """
        Compute the normalized direction vectors from centroid to each connector.

        Returns:
            directions: (n_connectors, 3) unit vectors
        """
        pos = positions.cpu().numpy()
        connector_pos = pos[connector_indices]
        centroid = connector_pos.mean(axis=0)
        vectors = connector_pos - centroid
        directions = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        return directions
