from __future__ import annotations

import unittest

import numpy as np

from generative_model.models.graph_pg_3d_projection import (
    add_centered_gaussian_noise,
    coordinate_rmsd,
    operation_errors,
    project_reynolds,
)


class TestGraphPG3DProjection(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        self.operations = np.stack((np.eye(3), np.diag([-1.0, -1.0, 1.0])))
        self.permutations = np.asarray([[0, 1], [1, 0]], dtype=np.int64)

    def test_noise_is_deterministic_and_centered(self) -> None:
        first = add_centered_gaussian_noise(
            self.reference, sigma_angstrom=0.2, seed=20260821
        )
        second = add_centered_gaussian_noise(
            self.reference, sigma_angstrom=0.2, seed=20260821
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first.mean(axis=0), 0.0, atol=1e-15)

    def test_reynolds_projection_enforces_c2_action(self) -> None:
        noisy = self.reference + np.asarray([[0.2, 0.3, -0.1], [-0.1, 0.1, 0.4]])
        projected = project_reynolds(
            noisy, self.operations, self.permutations, iterations=1
        )
        errors = operation_errors(projected, self.operations, self.permutations)
        self.assertLessEqual(errors["max_atom_error_angstrom"], 1e-12)
        self.assertLess(coordinate_rmsd(self.reference, projected), coordinate_rmsd(self.reference, noisy))


if __name__ == "__main__":
    unittest.main()
