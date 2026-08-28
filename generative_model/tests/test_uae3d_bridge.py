"""UAE-3D 严格 bridge、兼容子集与冻结 split 回归测试。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from generative_model.data import COFSymmetryDataset
from generative_model.models.uae3d_bridge import (
    OFFICIAL_EDGE_DIM,
    OFFICIAL_NODE_DIM,
    canonical_to_uae3d_data,
    uae3d_model_args,
    uae3d_data_to_canonical,
)
from generative_model.models.uae3d_reconstruction import (
    RECONSTRUCTION_METRICS_VERSION,
    _strict_rdkit_status,
    _bond_representation_status,
    evaluate_mean_reconstruction,
    full_pair_targets,
    reconstruction_gate,
)
from generative_model.smoke.build_uae3d_split import build_uae3d_split


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "generative_model/data/processed/v2"
SPLIT = ROOT / "generative_model/smoke/splits/uae3d_v1.json"
HAS_PYG = importlib.util.find_spec("torch_geometric") is not None


@unittest.skipUnless(HAS_PYG, "torch_geometric not installed")
class TestUAE3DBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = COFSymmetryDataset(PACKAGE)

    def test_frozen_split_rebuilds_identically(self) -> None:
        frozen = json.loads(SPLIT.read_text(encoding="utf-8"))
        rebuilt = build_uae3d_split(PACKAGE)
        self.assertEqual(rebuilt, frozen)
        self.assertEqual(frozen["compatible_counts"]["full_dataset"], 2242)
        self.assertEqual(frozen["compatible_counts"]["iid_train"], 1784)
        self.assertEqual(frozen["tier_indices"]["1"], [864])
        self.assertTrue(set(frozen["tier_indices"]["1"]).issubset(frozen["tier_indices"]["4"]))
        self.assertTrue(set(frozen["tier_indices"]["4"]).issubset(frozen["tier_indices"]["32"]))
        self.assertEqual(frozen["stress_indices"]["max_iid_train_atoms"], 101)
        self.assertEqual(frozen["stress_indices"]["max_full_dataset_atoms"], 105)

    def test_bridge_roundtrip_is_lossless_for_discrete_fields(self) -> None:
        for index in (0, 2485):
            sample = self.dataset[index]
            data = canonical_to_uae3d_data(sample)
            restored = uae3d_data_to_canonical(data)
            for key in (
                "atom_types", "atomic_numbers", "formal_charges", "radical_electrons",
                "bond_index", "bond_types",
            ):
                np.testing.assert_array_equal(restored[key], sample[key])
            np.testing.assert_allclose(restored["positions"], sample["positions"], atol=1e-6, rtol=0)

    def test_official_feature_and_full_pair_shapes(self) -> None:
        sample = self.dataset[864]
        data = canonical_to_uae3d_data(sample)
        atom_count = len(sample["atom_types"])
        self.assertEqual(tuple(data.x.shape), (atom_count, OFFICIAL_NODE_DIM))
        self.assertEqual(tuple(data.edge_index.shape), (2, atom_count * atom_count))
        self.assertEqual(tuple(data.edge_attr.shape), (atom_count * atom_count, OFFICIAL_EDGE_DIM + 1))
        dense = data.edge_attr.numpy().reshape(atom_count, atom_count, OFFICIAL_EDGE_DIM + 1)
        np.testing.assert_array_equal(dense[:, :, -1], np.eye(atom_count, dtype=np.float32))
        np.testing.assert_array_equal(dense[:, :, :4], dense[:, :, :4].transpose(1, 0, 2))

    def test_sn_charge_and_radical_fail_strictly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Sn"):
            canonical_to_uae3d_data(self.dataset[1684])
        with self.assertRaisesRegex(ValueError, "formal-charge head"):
            canonical_to_uae3d_data(self.dataset[2])
        with self.assertRaisesRegex(ValueError, "radical-electron head"):
            canonical_to_uae3d_data(self.dataset[1374])

    def test_official_model_args_and_full_pair_targets(self) -> None:
        args = uae3d_model_args("official")
        self.assertEqual(args.node_dim, 58)
        self.assertEqual(args.encoder_blocks, 6)
        self.assertEqual(args.decoder_blocks, 4)
        data = canonical_to_uae3d_data(self.dataset[864])
        atom_count = data.x.shape[0]
        targets = full_pair_targets(data.edge_attr).numpy().reshape(atom_count, atom_count)
        np.testing.assert_array_equal(np.diag(targets), np.full(atom_count, 5))
        np.testing.assert_array_equal(targets, targets.T)

        atom_classes = data.x[:, :16].argmax(dim=-1).numpy()
        chemistry = _strict_rdkit_status(atom_classes, targets)
        self.assertTrue(chemistry["sanitize_valid"])
        self.assertTrue(chemistry["connected"])

        bad_diagonal = targets.copy()
        np.fill_diagonal(bad_diagonal, 0)
        chemistry = _strict_rdkit_status(atom_classes, bad_diagonal)
        representation = _bond_representation_status(bad_diagonal)
        self.assertTrue(chemistry["sanitize_valid"])
        self.assertTrue(chemistry["connected"])
        self.assertFalse(representation["bond_representation_valid"])
        self.assertEqual(representation["bond_representation_status"], "BAD_SELF_LOOP")

    def test_reconstruction_gate_requires_all_checks(self) -> None:
        metrics = {
            "molecule_count": 1,
            "atom_exact_molecules": 1,
            "bond_exact_molecules": 1,
            "sanitize_valid_molecules": 1,
            "connected_molecules": 1,
            "coordinate_rmsd_max_angstrom": 0.049,
        }
        self.assertTrue(reconstruction_gate(metrics)["passed"])
        metrics["coordinate_rmsd_max_angstrom"] = 0.051
        self.assertFalse(reconstruction_gate(metrics)["passed"])

    def test_perfect_mean_reconstruction_metrics_pass_gate(self) -> None:
        import torch
        import torch.nn.functional as functional

        data = canonical_to_uae3d_data(self.dataset[864])
        bond_targets = full_pair_targets(data.edge_attr)

        class PerfectModel(torch.nn.Module):
            def encoder(self, x, edge_index, edge_attr, pos):
                mean = x[:, :16] * 20.0
                return mean, torch.zeros_like(mean)

            def decode(self, mean, batch=None):
                bonds = functional.one_hot(bond_targets, num_classes=6).float() * 20.0
                return mean, bonds, data.pos

        metrics = evaluate_mean_reconstruction(PerfectModel(), [data], torch.device("cpu"))
        self.assertEqual(metrics["schema_version"], RECONSTRUCTION_METRICS_VERSION)
        self.assertEqual(metrics["atom_exact_molecules"], 1)
        self.assertEqual(metrics["bond_exact_molecules"], 1)
        self.assertEqual(metrics["self_loop_exact_molecules"], 1)
        self.assertEqual(metrics["bond_representation_valid_molecules"], 1)
        self.assertEqual(metrics["sanitize_valid_molecules"], 1)
        self.assertEqual(metrics["connected_molecules"], 1)
        self.assertEqual(metrics["coordinate_rmsd_max_angstrom"], 0.0)
        self.assertTrue(reconstruction_gate(metrics)["passed"])


if __name__ == "__main__":
    unittest.main()
