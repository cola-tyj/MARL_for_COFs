"""v2 symmetry layers 与模型 adapters 测试。"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from generative_model.data import (
    COFSymmetryDataset,
    MiDiAdapter,
    SemlaFlowAdapter,
    SparseGraphAdapter,
    UAE3DAdapter,
)
from generative_model.data.build_cof_package import sha256_file
from generative_model.data.validate_v2 import validate_v2

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "data/processed/v2"
V1_DIR = Path(__file__).resolve().parents[1] / "data/processed/v1"


class TestV2Package(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = COFSymmetryDataset(PACKAGE_DIR)

    def test_canonical_layer_is_unchanged(self) -> None:
        self.assertEqual(sha256_file(PACKAGE_DIR / "cof_graphs.npz"), sha256_file(V1_DIR / "cof_graphs.npz"))
        self.assertEqual(len(self.dataset), 2532)
        self.assertEqual(len(self.dataset.arrays["atom_types"]), 91113)

    def test_symmetry_annotation_matches_every_molecule(self) -> None:
        self.assertEqual(len(self.dataset.symmetry["target_pg"]), len(self.dataset))
        self.assertTrue(self.dataset.symmetry["pg_compatible"].all())
        for index in (0, 50, 1327, 1684, 2378):
            sample = self.dataset[index]
            atom_count = len(sample["atom_types"])
            self.assertEqual(sample["actual_symmetry"]["permutation_index"].shape[1], atom_count)
            self.assertEqual(sample["target_symmetry"]["permutation_index"].shape[1], atom_count)
            self.assertEqual(len(sample["orbit_id"]), atom_count)
            self.assertEqual(sample["target_pg"], sample["metadata"]["Target_PG"])

    def test_full_pair_adapter_preserves_sparse_bonds(self) -> None:
        sparse = SparseGraphAdapter(self.dataset)[0]
        dense = MiDiAdapter(self.dataset)[0]["full_pair_bond_types"]
        self.assertTrue(np.array_equal(dense, dense.T))
        self.assertTrue(np.all(np.diag(dense) == 0))
        begin, end = sparse["bond_index"]
        np.testing.assert_array_equal(dense[begin, end], sparse["bond_types"])
        self.assertEqual(int(np.count_nonzero(np.triu(dense))), len(sparse["bond_types"]))

    def test_midi_adapter_is_strict_and_never_remaps_sn(self) -> None:
        adapter = MiDiAdapter(self.dataset)
        sample = adapter[0]
        np.testing.assert_array_equal(sample["midi_positions"], sample["positions"])
        np.testing.assert_array_equal(
            sample["midi_charge_types"], sample["formal_charges"] + 2
        )
        with self.assertRaisesRegex(ValueError, "Sn"):
            adapter[1684]
        extended = MiDiAdapter(self.dataset, mode="extended_vocabulary")[1684]
        sn_mask = extended["atomic_numbers"] == 50
        self.assertTrue(np.all(extended["midi_atom_types"][sn_mask] == 16))
        self.assertFalse(np.any(extended["midi_atom_types"][sn_mask] == 2))
        self.assertFalse(np.any(extended["midi_atom_types"][sn_mask] == 7))
        with self.assertRaisesRegex(ValueError, "radical-electron"):
            adapter[1374]

    def test_remove_h_preserves_symmetry_permutations(self) -> None:
        sample = SparseGraphAdapter(self.dataset, include_h=False)[0]
        self.assertTrue(np.all(sample["atomic_numbers"] != 1))
        atom_count = len(sample["atomic_numbers"])
        for key in ("actual_symmetry", "target_symmetry"):
            permutations = sample[key]["permutation_index"]
            self.assertTrue(np.all(permutations >= 0))
            self.assertTrue(np.all(permutations < atom_count))

    def test_sn_is_never_remapped(self) -> None:
        with self.assertRaisesRegex(ValueError, "Sn"):
            UAE3DAdapter(self.dataset, mode="compatible_subset")[1684]
        sample = UAE3DAdapter(self.dataset, mode="extended_vocabulary")[1684]
        sn_mask = sample["atomic_numbers"] == 50
        self.assertEqual(int(sn_mask.sum()), 2)
        self.assertTrue(np.all(sample["uae_atom_types"][sn_mask] == 16))
        self.assertFalse(np.any(sample["uae_atom_types"][sn_mask] == 2))  # official C index
        self.assertFalse(np.any(sample["uae_atom_types"][sn_mask] == 7))  # official Si index

    def test_semlaflow_adapter_is_strict_and_preserves_ground_truth(self) -> None:
        adapter = SemlaFlowAdapter(self.dataset)
        sample = adapter[0]
        np.testing.assert_allclose(
            sample["semlaflow_positions"] * sample["semlaflow_coordinate_std"],
            sample["positions"],
            atol=1e-6,
        )
        self.assertEqual(sample["full_pair_bond_types"].shape, (68, 68))
        self.assertEqual(len(sample["semlaflow_charge_types"]), 68)
        with self.assertRaisesRegex(ValueError, "Sn"):
            adapter[1684]
        with self.assertRaisesRegex(ValueError, "radical-electron"):
            adapter[1374]

    def test_manifest_and_full_validator(self) -> None:
        manifest = json.loads((PACKAGE_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["one_conformer_per_smiles"])
        self.assertEqual(manifest["environment"]["pymatgen"], "2025.3.10")
        report = validate_v2(PACKAGE_DIR, write_report=False)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["sn"]["molecules"], 2)
        self.assertEqual(report["sn"]["atoms"], 4)


if __name__ == "__main__":
    unittest.main()
