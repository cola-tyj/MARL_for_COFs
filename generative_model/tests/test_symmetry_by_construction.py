"""Tests for orbit-representative hard-symmetry coordinate expansion."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from generative_model.models.graph_pg_3d_projection import operation_errors
from generative_model.models.symmetry_by_construction import (
    compress_orbit_representatives,
    expand_orbit_representatives,
    expand_orbit_representatives_torch,
    validate_orbit_action,
)


class TestSymmetryByConstruction(unittest.TestCase):
    def setUp(self) -> None:
        self.c2_operations = np.stack((np.eye(3), np.diag([-1.0, -1.0, 1.0])))
        self.c2_permutations = np.asarray([[0, 1, 2], [1, 0, 2]], dtype=np.int64)
        self.c2_orbits = np.asarray([0, 0, 1], dtype=np.int64)

    def test_c2_expansion_and_axis_stabilizer(self) -> None:
        representatives = np.asarray([[1.0, 0.2, 0.3], [0.8, -0.4, 2.0]])
        expanded = expand_orbit_representatives(
            representatives, self.c2_orbits, self.c2_operations, self.c2_permutations
        )
        errors = operation_errors(expanded, self.c2_operations, self.c2_permutations)
        self.assertLessEqual(errors["max_atom_error_angstrom"], 1e-12)
        self.assertAlmostEqual(expanded[2, 0], 0.0, places=12)
        self.assertAlmostEqual(expanded[2, 1], 0.0, places=12)
        np.testing.assert_allclose(expanded.mean(axis=0), 0.0, atol=1e-12)

    def test_c3_single_orbit_expansion(self) -> None:
        angles = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
        operations = np.asarray([
            [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
            for a in angles
        ])
        permutations = np.asarray([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int64)
        expanded = expand_orbit_representatives(
            np.asarray([[1.0, 0.1, 0.4]]), np.zeros(3, dtype=np.int64), operations, permutations
        )
        errors = operation_errors(expanded, operations, permutations)
        self.assertLessEqual(errors["max_atom_error_angstrom"], 1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(expanded[:, :2], axis=1),
            np.linalg.norm(expanded[0, :2]),
            atol=1e-12,
        )

    def test_rotation_covariance_with_conjugated_operations(self) -> None:
        representatives = np.asarray([[1.0, 0.2, 0.3], [0.0, 0.0, 2.0]])
        reference = expand_orbit_representatives(
            representatives, self.c2_orbits, self.c2_operations, self.c2_permutations
        )
        angle = 0.41
        rotation = np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ])
        conjugated = np.asarray([rotation @ matrix @ rotation.T for matrix in self.c2_operations])
        actual = expand_orbit_representatives(
            representatives @ rotation.T,
            self.c2_orbits,
            conjugated,
            self.c2_permutations,
        )
        np.testing.assert_allclose(actual, reference @ rotation.T, atol=1e-12)

    def test_torch_expansion_matches_numpy_and_has_gradient(self) -> None:
        representatives = torch.tensor(
            [[1.0, 0.2, 0.3], [0.0, 0.0, 2.0]], dtype=torch.double, requires_grad=True
        )
        actual = expand_orbit_representatives_torch(
            representatives,
            torch.as_tensor(self.c2_orbits),
            torch.as_tensor(self.c2_operations),
            torch.as_tensor(self.c2_permutations),
        )
        expected = expand_orbit_representatives(
            representatives.detach().numpy(),
            self.c2_orbits,
            self.c2_operations,
            self.c2_permutations,
        )
        np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-12)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(representatives.grad).all())
        self.assertGreater(float(representatives.grad.abs().sum()), 0.0)

    def test_compress_uses_deterministic_smallest_atom_representative(self) -> None:
        positions = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        compressed = compress_orbit_representatives(
            positions, self.c2_orbits, self.c2_operations, self.c2_permutations
        )
        np.testing.assert_array_equal(compressed["representative_indices"], [0, 2])
        self.assertEqual(compressed["representative_positions"].shape, (2, 3))

    def test_invalid_cross_orbit_action_fails_strictly(self) -> None:
        invalid_orbits = np.asarray([0, 1, 1], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "跨 orbit"):
            validate_orbit_action(
                invalid_orbits, self.c2_operations, self.c2_permutations
            )


if __name__ == "__main__":
    unittest.main()
