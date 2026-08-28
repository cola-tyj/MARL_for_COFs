"""Tests for the deployment-style ET-Flow + E3/F0.2 inference interface."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from generative_model.data.cof_graph_dataset import COFGraphDataset
from generative_model.inference.generate_etflow_symmetric_xyz import (
    _sample_from_canonical,
    _select_candidate,
    _write_xyz,
)


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "generative_model/data/processed/v2"


class TestETFlowE3F02Inference(unittest.TestCase):
    def test_known_graph_action_uses_requested_pg_without_reference_xyz(self) -> None:
        canonical = COFGraphDataset(PACKAGE)[0]
        sample = _sample_from_canonical(canonical, "C2")
        self.assertEqual(sample["target_pg"], "C2")
        self.assertNotIn("target_positions", sample)
        self.assertFalse(sample["graph_action_audit"]["recovery"]["coordinates_read"])
        self.assertFalse(
            sample["graph_action_audit"]["recovery"]["stored_symmetry_action_read"]
        )

    def test_candidate_selection_is_strict_and_reference_free(self) -> None:
        records = [
            {"candidate_id": 0, "passed_geometry_gate": True,
             "final_total_objective_kcal_mol": 5.0},
            {"candidate_id": 1, "passed_geometry_gate": False,
             "final_total_objective_kcal_mol": 1.0},
            {"candidate_id": 2, "passed_geometry_gate": True,
             "final_total_objective_kcal_mol": 3.0},
        ]
        self.assertEqual(_select_candidate(records), 2)
        with self.assertRaises(RuntimeError):
            _select_candidate([{**records[0], "passed_geometry_gate": False}])

    def test_xyz_writer_preserves_symbols_and_atom_count(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "molecule.xyz"
            _write_xyz(
                path,
                ["Sn", "H"],
                np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                "strict test",
            )
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "2")
            self.assertTrue(lines[2].startswith("Sn"))
            self.assertTrue(lines[3].startswith("H"))


if __name__ == "__main__":
    unittest.main()
