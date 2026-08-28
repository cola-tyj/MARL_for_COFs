"""Accounting tests for the all-canonical S4/D6h inference Gate."""

from __future__ import annotations

import unittest

from generative_model.inference.audit_etflow_e3f02_rare_targets import summarize_point_groups
from generative_model.inference.run_etflow_e3f02_rare_targets import summarize_geometry


def _protocol() -> dict:
    return {
        "panel": {"package_indices": [0, 1, 2, 3], "molecule_count": 4},
        "geometry_gate_thresholds": {
            "graph_action_recovery_fraction_min": 1.0,
            "end_to_end_success_fraction_min": 0.90,
            "s4_success_fraction_min": 0.90,
            "d6h_success_fraction_min": 0.85,
        },
        "point_group_gate_thresholds": {
            "analyzer_success_fraction_min": 0.90,
            "pg_compatible_fraction_min": 0.90,
            "s4_pg_compatible_fraction_min": 0.90,
            "d6h_pg_compatible_fraction_min": 0.85,
        },
    }


class TestRareTargetAccounting(unittest.TestCase):
    def test_generation_failures_use_full_denominator(self) -> None:
        records = []
        for index, pg in enumerate(("S4", "S4", "D6h", "D6h")):
            success = index != 3
            records.append({
                "package_index": index, "requested_target_pg": pg,
                "graph_action_recovered": success,
                "status": "success" if success else "failure",
                "failure_stage": None if success else "graph_action",
                "scope": {"reference_xyz_used": False, "stored_symmetry_action_used": False, "etkdg_used": False},
            })
        metrics, checks = summarize_geometry(records, _protocol())
        self.assertEqual(metrics["end_to_end_success_fraction"], 0.75)
        self.assertEqual(metrics["d6h_success_fraction"], 0.5)
        self.assertFalse(checks["end_to_end_success_fraction_min"])
        self.assertFalse(checks["d6h_success_fraction_min"])

    def test_point_group_failures_use_full_denominator(self) -> None:
        records = []
        for index, pg in enumerate(("S4", "S4", "D6h", "D6h")):
            success = index != 3
            records.append({
                "requested_target_pg": pg, "analyzer_success": success,
                "pg_compatible": success, "pg_exact_match": success,
                "actual_pg": pg if success else None,
            })
        metrics, checks = summarize_point_groups(records, _protocol(), True)
        self.assertEqual(metrics["pg_compatible_fraction"], 0.75)
        self.assertEqual(metrics["d6h_pg_compatible_fraction"], 0.5)
        self.assertFalse(checks["pg_compatible_fraction_min"])
        self.assertFalse(checks["d6h_pg_compatible_fraction_min"])


if __name__ == "__main__":
    unittest.main()
