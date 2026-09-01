import json
import unittest
from pathlib import Path

from generative_model.PG_OrbitFlow.factorized_automorphism import (
    build_factorized_automorphism_contract,
)
from generative_model.PG_OrbitFlow.geometry import build_geometry_contract
from generative_model.PG_OrbitFlow.data import PGOrbitFlowDataset
from generative_model.PG_OrbitFlow.overfit import _panel_samples


ROOT = Path(__file__).resolve().parents[1]


class TestFactorizedAutomorphism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "configs/m3p6_shape_decoder_v2.json").read_text(
                encoding="utf-8"
            )
        )
        cls.samples = _panel_samples(
            cls.protocol,
            {
                "records": cls.protocol["panel"]["records"],
                "evaluation": {},
                "gate": {},
            },
        )

    def test_boh2_branch_exchange_has_explicit_factorized_witness(self):
        sample = next(row for row in self.samples if row.package_index == 39)
        contract = build_factorized_automorphism_contract(
            sample, build_geometry_contract(sample)
        )
        self.assertIn((4, 5), contract.groups)
        self.assertIn((9, 10), contract.groups)
        self.assertGreater(contract.branch_swap_witness_count, 0)

    def test_all_m3_panel_factorized_groups_fit_generalized_head(self):
        for sample in self.samples:
            contract = build_factorized_automorphism_contract(
                sample, build_geometry_contract(sample)
            )
            self.assertTrue(all(2 <= len(group) <= 6 for group in contract.groups))

    def test_large_synchronized_branch_group_fails_before_anchored_search(self):
        dataset = PGOrbitFlowDataset(
            ROOT.parent / "data/processed/v2",
            split="train",
            split_scheme="iid",
            point_groups=("C2", "C3"),
        )
        sample = next(
            dataset[position]
            for position in range(len(dataset))
            if dataset[position].package_index == 28
        )
        with self.assertRaisesRegex(RuntimeError, "exceeds 6 slots"):
            build_factorized_automorphism_contract(
                sample, build_geometry_contract(sample)
            )


if __name__ == "__main__":
    unittest.main()
