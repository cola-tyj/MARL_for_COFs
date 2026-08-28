from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from generative_model.data.cof_graph_dataset import COFGraphDataset
from generative_model.evaluation.our_etflow_ablation import (
    ROUTES,
    candidate_diversity,
    derived_seed,
    run_full_f02_route,
)
from generative_model.inference.generate_etflow_symmetric_xyz import _fingerprint
from generative_model.inference.generate_etflow_symmetric_xyz_v4 import (
    action_samples_v4,
)


ROOT = Path(__file__).resolve().parents[2]


class TestOurETFlowAblation(unittest.TestCase):
    def test_derived_seed_is_stable_and_signed_32_bit(self) -> None:
        self.assertEqual(derived_seed(7, "route", 0), 7)
        first = derived_seed(7, "route", 1)
        self.assertEqual(first, derived_seed(7, "route", 1))
        self.assertNotEqual(first, derived_seed(7, "route", 2))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**31)

    def test_candidate_diversity_is_translation_invariant(self) -> None:
        first = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        second = first + np.asarray([4.0, -2.0, 9.0])
        result = candidate_diversity([first, second])
        self.assertAlmostEqual(result["mean_pairwise_kabsch_rmsd_angstrom"], 0.0)
        self.assertAlmostEqual(result["mean_pairwise_distance_mae_angstrom"], 0.0)

    def test_protocol_is_balanced_and_reference_free(self) -> None:
        path = ROOT / "generative_model/evaluation/our_etflow_ablation_protocol_v1.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["protocol_fingerprint"], _fingerprint(protocol))
        self.assertEqual(
            protocol["panel"]["target_pg_counts"],
            {"C2": 8, "C3": 8, "D6h": 8, "S4": 8},
        )
        self.assertEqual(set(protocol["routes"]), set(ROUTES))
        self.assertFalse(protocol["paired_contract"]["reference_xyz_used_for_generation"])
        self.assertFalse(protocol["paired_contract"]["reference_xyz_used_for_selection"])

    def test_full_route_accepts_shared_raw_coordinates(self) -> None:
        dataset = COFGraphDataset(ROOT / "generative_model/data/processed/v2")
        canonical = dataset[2092]
        protocol = json.loads(
            (ROOT / "generative_model/inference/etflow_e3f02_protocol_v5.json")
            .read_text(encoding="utf-8")
        )
        samples = action_samples_v4(canonical, "C2", protocol)
        result = run_full_f02_route(
            samples,
            [canonical["positions"]],
            "C2",
            123,
            protocol,
            prior_seconds=0.0,
        )
        self.assertTrue(result["selected_f02"]["passed_geometry_gate"])
        self.assertTrue(result["hard_projection_used"])
        self.assertTrue(result["f02_used"])


if __name__ == "__main__":
    unittest.main()
