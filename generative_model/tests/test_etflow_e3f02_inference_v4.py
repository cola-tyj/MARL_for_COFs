"""Tests for S4/D6h multi-action v4 inference."""

from __future__ import annotations

import unittest

from generative_model.data.graph_only_dataset import CanonicalGraphOnlyDataset
from generative_model.inference.generate_etflow_symmetric_xyz_v4 import action_samples_v4


class TestETFlowE3F02InferenceV4(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = CanonicalGraphOnlyDataset("generative_model/data/processed/v2")
        cls.protocol = {
            "rare_action_selection": {
                target: {"maximum_actions": 64, "maximum_search_matches": 100_000}
                for target in ("S4", "D6h")
            }
        }

    def test_s4_and_d6h_both_return_unique_graph_only_candidates(self) -> None:
        for package_index, target in ((53, "S4"), (858, "D6h")):
            samples = action_samples_v4(
                self.dataset[package_index], target, self.protocol
            )
            self.assertEqual(len(samples), 64)
            self.assertEqual(len({
                tuple(value["target_permutation_index"].ravel().tolist())
                for value in samples
            }), 64)
            self.assertTrue(all(
                not value["graph_action_audit"]["recovery"]["coordinates_read"]
                and not value["graph_action_audit"]["recovery"]["stored_symmetry_action_read"]
                for value in samples
            ))

    def test_c2_c3_remain_single_action(self) -> None:
        for package_index, target in ((902, "C2"), (1160, "C3")):
            self.assertEqual(len(action_samples_v4(
                self.dataset[package_index], target, self.protocol
            )), 1)


if __name__ == "__main__":
    unittest.main()
