"""Tests for frozen dataset-level E3/F0.2 inference accounting."""

from __future__ import annotations

import unittest

from generative_model.inference.audit_etflow_e3f02_iidtest import (
    summarize_point_groups,
)
from generative_model.inference.build_etflow_e3f02_iidtest_protocol import (
    select_size_stratified,
)
from generative_model.inference.run_etflow_e3f02_iidtest import (
    summarize_geometry,
)


class TestETFlowE3F02IIDTest(unittest.TestCase):
    def test_size_stratification_has_fixed_quota_and_is_deterministic(self) -> None:
        candidates = list(range(40))
        atom_counts = {index: 10 + index for index in candidates}
        first, audit = select_size_stratified(candidates, atom_counts, 16)
        second, _ = select_size_stratified(candidates, atom_counts, 16)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(len(set(first)), 16)
        self.assertEqual([item["selected_count"] for item in audit], [4] * 4)
        for item in audit:
            self.assertLessEqual(
                item["minimum_atom_count"], item["maximum_atom_count"]
            )

    def test_geometry_gate_counts_failures_in_full_panel_denominator(self) -> None:
        records = [
            {
                "package_index": index,
                "requested_target_pg": "C2" if index < 2 else "C3",
                "graph_action_recovered": index != 3,
                "status": "success" if index in {0, 2} else "failure",
                "failure_stage": None if index in {0, 2} else "strict_candidate_gate",
                "scope": {
                    "reference_xyz_used": False,
                    "stored_symmetry_action_used": False,
                    "etkdg_used": False,
                },
            }
            for index in range(4)
        ]
        protocol = {
            "panel": {"package_indices": [0, 1, 2, 3], "molecule_count": 4},
            "geometry_gate_thresholds": {
                "graph_action_recovery_fraction_min": 1.0,
                "end_to_end_success_fraction_min": 0.9,
                "c2_success_fraction_min": 0.9,
                "c3_success_fraction_min": 0.875,
            },
        }
        metrics, checks = summarize_geometry(records, protocol)
        self.assertEqual(metrics["end_to_end_success_fraction"], 0.5)
        self.assertEqual(metrics["graph_action_recovery_fraction"], 0.75)
        self.assertFalse(checks["end_to_end_success_fraction_min"])
        self.assertFalse(checks["graph_action_recovery_fraction_min"])

    def test_pg_gate_counts_generation_failure_as_incompatible(self) -> None:
        records = [
            {
                "requested_target_pg": "C2",
                "analyzer_success": True,
                "pg_compatible": True,
                "pg_exact_match": True,
                "actual_pg": "C2",
            },
            {
                "requested_target_pg": "C2",
                "analyzer_success": False,
                "pg_compatible": False,
                "pg_exact_match": False,
                "actual_pg": None,
            },
            {
                "requested_target_pg": "C3",
                "analyzer_success": True,
                "pg_compatible": True,
                "pg_exact_match": False,
                "actual_pg": "D3h",
            },
            {
                "requested_target_pg": "C3",
                "analyzer_success": False,
                "pg_compatible": False,
                "pg_exact_match": False,
                "actual_pg": None,
            },
        ]
        protocol = {
            "panel": {"molecule_count": 4},
            "point_group_gate_thresholds": {
                "analyzer_success_fraction_min": 0.9,
                "pg_compatible_fraction_min": 0.9,
                "c2_pg_compatible_fraction_min": 0.9,
                "c3_pg_compatible_fraction_min": 0.875,
            },
        }
        metrics, checks = summarize_point_groups(records, protocol, True)
        self.assertEqual(metrics["pg_compatible_fraction"], 0.5)
        self.assertEqual(metrics["c2_pg_compatible_fraction"], 0.5)
        self.assertEqual(metrics["c3_pg_compatible_fraction"], 0.5)
        self.assertFalse(checks["pg_compatible_fraction_min"])


if __name__ == "__main__":
    unittest.main()
