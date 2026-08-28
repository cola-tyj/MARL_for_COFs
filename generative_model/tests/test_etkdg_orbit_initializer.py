"""Engineering tests for coordinate-free ETKDG and orbit projection."""

from __future__ import annotations

import unittest

import numpy as np

from generative_model.data.graph_pg_3d_dataset import graph_pg_3d_sample
from generative_model.data.v2_dataset import COFSymmetryDataset
from generative_model.optimization.etkdg_orbit_initializer import (
    build_coordinate_free_rdkit_graph,
    embed_etkdg,
    orient_and_project_to_orbits,
    proper_axis_orientation_matrices,
    validate_typed_graph_automorphisms,
)
from generative_model.optimization.etkdg_orbit_resonance import (
    validate_resonance_compatible_graph_automorphisms,
)


PACKAGE = "generative_model/data/processed/v2"
class TestETKDGOrbitInitializer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw = COFSymmetryDataset(PACKAGE)[1215]
        cls.sample = graph_pg_3d_sample(raw)
        cls.sample["aromaticity"] = raw["aromaticity"]

    def test_proper_axis_orientations_are_24_unique_so3_matrices(self) -> None:
        matrices = proper_axis_orientation_matrices()
        self.assertEqual(matrices.shape, (24, 3, 3))
        self.assertEqual(len({tuple(matrix.reshape(-1)) for matrix in matrices}), 24)
        np.testing.assert_allclose(
            np.swapaxes(matrices, 1, 2) @ matrices,
            np.broadcast_to(np.eye(3), (24, 3, 3)),
            atol=0,
        )
        np.testing.assert_allclose(np.linalg.det(matrices), np.ones(24), atol=0)

    def test_canonical_atom_order_and_typed_graph_action_are_strict(self) -> None:
        validate_typed_graph_automorphisms(self.sample)
        molecule = build_coordinate_free_rdkit_graph(self.sample)
        self.assertEqual(molecule.GetNumConformers(), 0)
        self.assertEqual(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
            self.sample["atomic_numbers"].tolist(),
        )

    def test_etkdg_is_deterministic_and_projection_is_exact(self) -> None:
        first = embed_etkdg(
            self.sample, seed=2026082401, maximum_iterations=1000,
            use_random_coordinates=False,
        )
        second = embed_etkdg(
            self.sample, seed=2026082401, maximum_iterations=1000,
            use_random_coordinates=False,
        )
        np.testing.assert_array_equal(first, second)
        result = orient_and_project_to_orbits(
            first, self.sample, match_radius_of_gyration=True
        )
        self.assertEqual(result["orientation_candidates"], 24)
        self.assertLessEqual(result["maximum_operation_error_angstrom"], 1e-5)
        self.assertTrue(np.isfinite(result["positions"]).all())

    def test_non_automorphism_fails_strictly(self) -> None:
        broken = dict(self.sample)
        permutations = self.sample["target_permutation_index"].copy()
        permutations[0, 0], permutations[0, 1] = permutations[0, 1], permutations[0, 0]
        broken["target_permutation_index"] = permutations
        with self.assertRaises(ValueError):
            validate_typed_graph_automorphisms(broken)

    def test_explicit_terminal_resonance_contract_accepts_only_known_lewis_swaps(self) -> None:
        dataset = COFSymmetryDataset(PACKAGE)
        normalized = []
        for package_index in (781, 2436, 2516):
            raw = dataset[package_index]
            sample = graph_pg_3d_sample(raw); sample["aromaticity"] = raw["aromaticity"]
            audit = validate_resonance_compatible_graph_automorphisms(sample)
            normalized.append(audit["resonance_normalized_operation_count"])
        self.assertTrue(all(value > 0 for value in normalized))

        broken = dict(self.sample)
        permutations = self.sample["target_permutation_index"].copy()
        permutations[0, 0], permutations[0, 3] = permutations[0, 3], permutations[0, 0]
        broken["target_permutation_index"] = permutations
        with self.assertRaises(ValueError):
            validate_resonance_compatible_graph_automorphisms(broken)

if __name__ == "__main__":
    unittest.main()
