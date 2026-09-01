from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "generative_model" / "PG_OrbitFlow" / "configs" / "overfit_protocol_v1.json"
H_AUDIT = (
    ROOT
    / "generative_model"
    / "PG_OrbitFlow"
    / "reports"
    / "h1_h3_protocol_audit.json"
)


class TestOverfitProtocol(unittest.TestCase):
    def test_nested_balanced_panels_and_no_evaluation_leakage(self):
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(protocol["schema_version"], "pg-orbitflow-overfit-protocol-v1")
        previous: set[int] = set()
        for tier in ("4", "16", "32"):
            block = protocol["tiers"][tier]
            records = block["records"]
            indices = {record["package_index"] for record in records}
            self.assertEqual(len(indices), int(tier))
            self.assertTrue(previous.issubset(indices))
            self.assertEqual(
                sum(record["target_pg"] == "C2" for record in records), int(tier) // 2
            )
            self.assertEqual(
                sum(record["target_pg"] == "C3" for record in records), int(tier) // 2
            )
            previous = indices
        self.assertFalse(protocol["data"]["test_used"])
        self.assertFalse(protocol["data"]["validation_used"])
        self.assertFalse(protocol["data"]["core_ood_used"])

    def test_h1_h3_protocols_are_frozen_single_variable_ablations(self):
        audit = json.loads(H_AUDIT.read_text(encoding="utf-8"))
        self.assertTrue(audit["passed"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertEqual(
            audit["adjacent_diffs"]["h2_to_h3"],
            ["objective.endpoint_weight"],
        )


if __name__ == "__main__":
    unittest.main()
