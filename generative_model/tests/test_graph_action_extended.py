"""Engineering tests for graph-only S4/D6h action recovery."""

from __future__ import annotations

import unittest

import numpy as np

from generative_model.data.graph_only_dataset import CanonicalGraphOnlyDataset
from generative_model.symmetry.graph_action_extended import (
    recover_full_target_graph_action,
)


PACKAGE = "generative_model/data/processed/v2"


class TestGraphActionExtended(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = CanonicalGraphOnlyDataset(PACKAGE)

    def _recover(self, package_index: int) -> dict:
        sample = self.dataset[package_index]
        return recover_full_target_graph_action(
            sample["atomic_numbers"],
            sample["formal_charges"],
            sample["radical_electrons"],
            sample["bond_index"],
            sample["bond_types"],
            str(sample["metadata"]["Target_PG"]),
        )

    def test_s4_is_deterministic_improper_cyclic_action(self) -> None:
        first = self._recover(48)
        second = self._recover(48)
        np.testing.assert_array_equal(
            first["permutation_index"], second["permutation_index"]
        )
        self.assertEqual(first["validation"]["operation_count"], 4)
        self.assertEqual(first["validation"]["unique_permutation_count"], 4)
        determinants = np.linalg.det(first["operation_matrices"])
        np.testing.assert_allclose(determinants, [1.0, -1.0, 1.0, -1.0])
        self.assertFalse(first["recovery_audit"]["coordinates_read"])
        self.assertFalse(first["recovery_audit"]["stored_symmetry_action_read"])

    def test_d6h_is_complete_planar_dihedral_action(self) -> None:
        action = self._recover(858)
        self.assertEqual(action["operation_matrices"].shape, (24, 3, 3))
        self.assertEqual(action["validation"]["operation_count"], 24)
        self.assertEqual(action["validation"]["unique_permutation_count"], 12)
        self.assertEqual(
            action["recovery_audit"]["horizontal_reflection_permutation"],
            "identity (graph-only planar D6h contract)",
        )
        self.assertLessEqual(
            action["validation"]["projection_probe_max_operation_error_angstrom"],
            1e-10,
        )
        determinants = np.rint(np.linalg.det(action["operation_matrices"])).astype(int)
        self.assertEqual(int(np.sum(determinants == 1)), 12)
        self.assertEqual(int(np.sum(determinants == -1)), 12)

    def test_extended_actions_never_cross_elements(self) -> None:
        for package_index in (48, 858):
            sample = self.dataset[package_index]
            action = self._recover(package_index)
            numbers = sample["atomic_numbers"]
            for permutation in action["permutation_index"]:
                np.testing.assert_array_equal(numbers[permutation], numbers)

    def test_asymmetric_graph_fails_s4_strictly(self) -> None:
        with self.assertRaises(ValueError):
            recover_full_target_graph_action(
                np.asarray([6, 7, 8]),
                np.zeros(3, dtype=np.int64),
                np.zeros(3, dtype=np.int64),
                np.asarray([[0, 1], [1, 2]], dtype=np.int64),
                np.asarray([1, 2], dtype=np.int64),
                "S4",
            )


if __name__ == "__main__":
    unittest.main()
