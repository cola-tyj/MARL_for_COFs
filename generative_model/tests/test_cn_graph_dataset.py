"""Contract tests for the uploaded Cn graph adapter."""

from __future__ import annotations

import unittest

import numpy as np

from generative_model.data.cn_graph_dataset import CnGraphDataset
from generative_model.symmetry.graph_action_extended import (
    recover_full_target_graph_action,
)


class TestCnGraphDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = CnGraphDataset("generative_model/data/Cn")

    def test_exact_group_counts_and_explicit_h_contract(self) -> None:
        self.assertEqual(len(self.dataset), 98)
        groups = [value["metadata"]["Source_Group"] for value in self.dataset.records]
        self.assertEqual(groups.count("C3k2"), 24)
        self.assertEqual(groups.count("C3k3"), 74)
        self.assertGreater(sum(
            int(np.sum(sample["atomic_numbers"] == 1))
            for sample in self.dataset.records
        ), 0)
        for sample in self.dataset.records:
            self.assertGreaterEqual(np.sum(sample["atomic_numbers"] == 1), 0)
            self.assertFalse(sample["adapter_audit"]["element_fallback_used"])
            self.assertFalse(sample["adapter_audit"]["graph_repair_used"])

    def test_all_records_recover_strict_c3_action(self) -> None:
        for sample in self.dataset.records:
            action = recover_full_target_graph_action(
                sample["atomic_numbers"], sample["formal_charges"],
                sample["radical_electrons"], sample["bond_index"],
                sample["bond_types"], "C3",
            )
            self.assertEqual(action["permutation_index"].shape[0], 3)
            self.assertEqual(
                action["validation"]["atom_count"], len(sample["atomic_numbers"])
            )
            self.assertGreater(action["validation"]["moved_atom_count"], 0)


if __name__ == "__main__":
    unittest.main()
