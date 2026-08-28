"""v1 图 + 3D 数据包的完整性测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from generative_model.data import ATOM_SYMBOLS, COFGraphDataset, collate_graphs

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "data/processed/v1"


class TestCOFGraphDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = COFGraphDataset(PACKAGE_DIR)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dataset.close()

    def test_global_counts_and_vocab(self) -> None:
        self.assertEqual(len(self.dataset), 2532)
        self.assertEqual(len(self.dataset.arrays["atom_types"]), 91113)
        self.assertEqual(tuple(self.dataset.vocab["atom_symbols"]), ATOM_SYMBOLS)
        self.assertEqual(len(ATOM_SYMBOLS), 13)

    def test_every_sample_is_finite_centered_and_indexed(self) -> None:
        for sample in self.dataset:
            atom_count = len(sample["atom_types"])
            self.assertTrue(np.isfinite(sample["positions"]).all())
            np.testing.assert_allclose(sample["positions"].mean(axis=0), 0.0, atol=2e-5)
            self.assertTrue(np.all(sample["atom_types"] < len(ATOM_SYMBOLS)))
            self.assertTrue(np.all((sample["bond_types"] >= 1) & (sample["bond_types"] <= 4)))
            if sample["bond_index"].size:
                self.assertGreaterEqual(int(sample["bond_index"].min()), 0)
                self.assertLess(int(sample["bond_index"].max()), atom_count)
                self.assertTrue(np.all(sample["bond_index"][0] < sample["bond_index"][1]))

    def test_corrected_exception(self) -> None:
        sample = self.dataset[1327]
        self.assertEqual(
            sample["metadata"]["XYZ_RelPath"],
            "cof_symmetry_pipeline/output/hierarchical/xyz/mol_000820.xyz",
        )
        self.assertEqual(len(sample["atom_types"]), 75)
        self.assertEqual(sample["bond_index"].shape[1], 81)

    def test_collate_builds_batch_indices_and_offsets_bonds(self) -> None:
        samples = [self.dataset[index] for index in (0, 1, 2)]
        batch = collate_graphs(samples)
        self.assertEqual(len(batch["batch_index"]), int(batch["atom_counts"].sum()))
        self.assertEqual(batch["bond_index"].shape[1], len(batch["bond_types"]))
        for index, count in enumerate(batch["atom_counts"]):
            self.assertEqual(int(np.count_nonzero(batch["batch_index"] == index)), int(count))
        self.assertLess(int(batch["bond_index"].max()), len(batch["batch_index"]))

    def test_splits_cover_without_overlap(self) -> None:
        expected = set(range(len(self.dataset)))
        for name in ("split_iid.json", "split_core_ood.json"):
            data = json.loads((PACKAGE_DIR / name).read_text(encoding="utf-8"))
            parts = [set(data["indices"][key]) for key in ("train", "val", "test")]
            self.assertEqual(set.union(*parts), expected)
            self.assertFalse(parts[0] & parts[1])
            self.assertFalse(parts[0] & parts[2])
            self.assertFalse(parts[1] & parts[2])

    def test_core_ood_has_no_core_leakage(self) -> None:
        split_data = json.loads((PACKAGE_DIR / "split_core_ood.json").read_text(encoding="utf-8"))
        core_sets = {
            split: set(self.dataset.metadata.iloc[split_data["indices"][split]]["Core"])
            for split in ("train", "val", "test")
        }
        self.assertFalse(core_sets["train"] & core_sets["val"])
        self.assertFalse(core_sets["train"] & core_sets["test"])
        self.assertFalse(core_sets["val"] & core_sets["test"])


if __name__ == "__main__":
    unittest.main()
