from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from generative_model.PG_OrbitFlow.data import PGOrbitFlowDataset


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "generative_model" / "data" / "processed" / "v2"


class TestV2Adapter(unittest.TestCase):
    def test_real_c2_c3_samples_are_strict_and_read_only(self):
        dataset = PGOrbitFlowDataset(
            PACKAGE,
            split="val",
            point_groups=("C2", "C3"),
            max_samples=4,
        )
        self.assertEqual(len(dataset), 4)
        for sample in (dataset[0], dataset[-1]):
            self.assertIn(sample.target_pg, {"C2", "C3"})
            self.assertEqual(sample.invariant_bond_order.shape, (len(sample.atom_types),) * 2)
            self.assertTrue(np.isfinite(sample.symmetric_target_angstrom).all())
            self.assertEqual(sample.quotient_graph.orbit_count, len(np.unique(sample.orbit_id)))
            self.assertTrue(np.any(sample.atomic_numbers > 1))


if __name__ == "__main__":
    unittest.main()
