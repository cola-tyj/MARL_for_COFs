"""Engineering tests for graph-only C2/C3 symmetry-action recovery."""

from __future__ import annotations

import unittest

import numpy as np

from generative_model.data.graph_only_dataset import (
    GRAPH_ARRAY_NAMES,
    CanonicalGraphOnlyDataset,
)
from generative_model.symmetry.graph_action import (
    recover_cyclic_graph_action,
    validate_recovered_graph_action,
)


PACKAGE = "generative_model/data/processed/v2"
def _recover(sample: dict) -> dict:
    return recover_cyclic_graph_action(
        sample["atomic_numbers"], sample["formal_charges"],
        sample["radical_electrons"], sample["bond_index"], sample["bond_types"],
        str(sample["metadata"]["Target_PG"]),
    )


class TestGraphActionE3(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = CanonicalGraphOnlyDataset(PACKAGE)

    def test_graph_only_loader_cannot_expose_coordinates(self) -> None:
        self.assertEqual(set(self.dataset.loaded_array_names), set(GRAPH_ARRAY_NAMES))
        self.assertNotIn("positions", self.dataset.arrays)
        self.assertNotIn("centroids", self.dataset.arrays)
        self.assertNotIn("positions", self.dataset[45])

    def test_c2_and_c3_recovery_is_deterministic_without_coordinates(self) -> None:
        for package_index in (45, 81):
            sample = self.dataset[package_index]
            first = _recover(sample); second = _recover(sample)
            np.testing.assert_array_equal(
                first["operation_matrices"], second["operation_matrices"]
            )
            np.testing.assert_array_equal(
                first["permutation_index"], second["permutation_index"]
            )
            np.testing.assert_array_equal(first["orbit_id"], second["orbit_id"])
            self.assertFalse(first["recovery_audit"]["coordinates_read"])
            self.assertFalse(first["recovery_audit"]["stored_symmetry_action_read"])

    def test_recovered_actions_preserve_full_explicit_h_graph(self) -> None:
        for package_index in (45, 81, 92):
            sample = self.dataset[package_index]
            action = _recover(sample)
            audit = validate_recovered_graph_action(
                action, sample["atomic_numbers"], sample["formal_charges"],
                sample["radical_electrons"], sample["bond_index"], sample["bond_types"],
            )
            order = int(str(sample["metadata"]["Target_PG"])[1])
            self.assertEqual(audit["operation_count"], order)
            self.assertGreater(audit["moved_atom_count"], 0)
            self.assertEqual(len(action["orbit_id"]), len(sample["atomic_numbers"]))

    def test_sn_is_never_mapped_to_another_element(self) -> None:
        sample = self.dataset[1684]
        action = _recover(sample)
        sn = np.flatnonzero(sample["atomic_numbers"] == 50)
        self.assertGreater(len(sn), 0)
        for permutation in action["permutation_index"]:
            self.assertTrue(np.all(sample["atomic_numbers"][permutation[sn]] == 50))

    def test_atom_reindexing_still_returns_a_valid_conjugate_action(self) -> None:
        sample = self.dataset[81]
        atom_count = len(sample["atomic_numbers"])
        order = np.arange(atom_count - 1, -1, -1, dtype=np.int64)
        inverse = np.empty(atom_count, dtype=np.int64); inverse[order] = np.arange(atom_count)
        reindexed = {
            "atomic_numbers": sample["atomic_numbers"][order],
            "formal_charges": sample["formal_charges"][order],
            "radical_electrons": sample["radical_electrons"][order],
            "bond_index": inverse[sample["bond_index"]],
            "bond_types": sample["bond_types"],
            "metadata": sample["metadata"],
        }
        action = _recover(reindexed)
        validate_recovered_graph_action(
            action, reindexed["atomic_numbers"], reindexed["formal_charges"],
            reindexed["radical_electrons"], reindexed["bond_index"],
            reindexed["bond_types"],
        )

    def test_unsupported_target_and_asymmetric_graph_fail_strictly(self) -> None:
        sample = self.dataset[45]
        with self.assertRaises(ValueError):
            recover_cyclic_graph_action(
                sample["atomic_numbers"], sample["formal_charges"],
                sample["radical_electrons"], sample["bond_index"], sample["bond_types"],
                "S4",
            )
        with self.assertRaises(ValueError):
            recover_cyclic_graph_action(
                np.asarray([6, 7, 8]), np.zeros(3, dtype=np.int64),
                np.zeros(3, dtype=np.int64), np.asarray([[0, 1], [1, 2]]),
                np.asarray([1, 2]), "C2",
            )

if __name__ == "__main__":
    unittest.main()
