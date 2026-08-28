"""Tests for reference-free S4/D6h graph-action candidates."""

from __future__ import annotations

import unittest

import numpy as np

from generative_model.data.graph_only_dataset import CanonicalGraphOnlyDataset
from generative_model.symmetry.graph_action_candidates_v2 import (
    recover_s4_graph_action_candidates,
)


class TestS4ActionCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = CanonicalGraphOnlyDataset(
            "generative_model/data/processed/v2"
        )

    def _recover(self, package_index: int) -> list[dict]:
        sample = self.dataset[package_index]
        return recover_s4_graph_action_candidates(
            sample["atomic_numbers"], sample["formal_charges"],
            sample["radical_electrons"], sample["bond_index"],
            sample["bond_types"], maximum_candidates=64,
        )

    def test_s4_candidates_are_deterministic_unique_and_graph_only(self) -> None:
        first = self._recover(53)
        second = self._recover(53)
        self.assertGreater(len(first), 1)
        self.assertEqual(len(first), len(second))
        keys = []
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(
                left["permutation_index"], right["permutation_index"]
            )
            self.assertEqual(left["validation"]["operation_count"], 4)
            self.assertFalse(left["recovery_audit"]["coordinates_read"])
            self.assertFalse(
                left["recovery_audit"]["stored_symmetry_action_read"]
            )
            keys.append(tuple(left["permutation_index"].ravel().tolist()))
        self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
