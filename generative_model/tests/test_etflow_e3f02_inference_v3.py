"""Tests for D6h multi-action selection in v3 inference."""

from __future__ import annotations

import unittest

from generative_model.data.graph_only_dataset import CanonicalGraphOnlyDataset
from generative_model.inference.generate_etflow_symmetric_xyz_v3 import (
    action_samples,
    projection_rank_key,
)


class TestETFlowE3F02InferenceV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = CanonicalGraphOnlyDataset(
            "generative_model/data/processed/v2"
        )
        cls.protocol = {
            "d6h_action_selection": {
                "maximum_actions": 64,
                "maximum_search_matches": 100_000,
            }
        }

    def test_projection_ranking_prioritizes_strict_collision_counts(self) -> None:
        clean = {
            "pair_count_below_0p6_angstrom": 0,
            "pair_count_below_0p8_angstrom": 10,
            "minimum_pair_distance_angstrom": 0.61,
            "projection_rmsd_angstrom": 3.0,
            "raw_candidate_id": 1,
            "action_candidate_id": 2,
        }
        colliding = {
            **clean,
            "pair_count_below_0p6_angstrom": 1,
            "pair_count_below_0p8_angstrom": 1,
            "minimum_pair_distance_angstrom": 0.59,
            "projection_rmsd_angstrom": 0.1,
        }
        self.assertLess(projection_rank_key(clean), projection_rank_key(colliding))

    def test_d6h_returns_64_graph_only_actions_and_s4_returns_one(self) -> None:
        d6h = action_samples(self.dataset[858], "D6h", self.protocol)
        s4 = action_samples(self.dataset[48], "S4", self.protocol)
        self.assertEqual(len(d6h), 64)
        self.assertEqual(len(s4), 1)
        self.assertEqual(
            len({
                tuple(sample["target_permutation_index"].ravel().tolist())
                for sample in d6h
            }),
            64,
        )
        for sample in d6h:
            self.assertNotIn("target_positions", sample)
            self.assertFalse(
                sample["graph_action_audit"]["recovery"]["coordinates_read"]
            )
            self.assertFalse(
                sample["graph_action_audit"]["recovery"][
                    "stored_symmetry_action_read"
                ]
            )


if __name__ == "__main__":
    unittest.main()
