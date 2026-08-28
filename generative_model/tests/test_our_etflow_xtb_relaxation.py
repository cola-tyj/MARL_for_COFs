from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from generative_model.evaluation.our_etflow_xtb_relaxation import (
    center_positions,
    electronic_state,
    relaxation_geometry_metrics,
)
from generative_model.inference.generate_etflow_symmetric_xyz import _fingerprint


ROOT = Path(__file__).resolve().parents[2]


class TestOurETFlowXTBRelaxation(unittest.TestCase):
    def test_center_positions_is_translation_invariant(self) -> None:
        positions = np.asarray([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]])
        self.assertTrue(np.allclose(
            center_positions(positions),
            center_positions(positions + np.asarray([7.0, -4.0, 2.0])),
        ))

    def test_electronic_state_is_strict(self) -> None:
        result = electronic_state(np.asarray([0, -1]), np.asarray([1, 0]))
        self.assertEqual(result, {"charge": -1, "unpaired_electrons": 1, "multiplicity": 2})
        with self.assertRaises(ValueError):
            electronic_state(np.asarray([0]), np.asarray([-1]))

    def test_geometry_metrics_are_rigid_motion_invariant(self) -> None:
        before = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        after = before + np.asarray([4., 7., -2.])
        metrics = relaxation_geometry_metrics(
            before, after, before, np.asarray([[0, 1], [1, 2]]).T
        )
        self.assertAlmostEqual(metrics["kabsch_rmsd_pre_to_post_angstrom"], 0.0)
        self.assertAlmostEqual(metrics["bond_length_mae_pre_to_post_angstrom"], 0.0)

    def test_frozen_protocol_panel_and_no_projection(self) -> None:
        path = ROOT / "generative_model/evaluation/our_etflow_xtb_protocol_v1.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["protocol_fingerprint"], _fingerprint(protocol))
        self.assertEqual(protocol["panel"]["molecule_count"], 32)
        self.assertEqual(
            protocol["panel"]["target_pg_counts"],
            {"C2": 8, "C3": 8, "D6h": 8, "S4": 8},
        )
        self.assertFalse(protocol["relaxation"]["hard_projection_during_relaxation"])
        self.assertIsNone(protocol["relaxation"]["constraints"])
        self.assertEqual(
            protocol["relaxation"]["thread_environment"],
            {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )


if __name__ == "__main__":
    unittest.main()
