"""Offline engineering tests for the ET-Flow canonical bridge and symmetry layer."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from rdkit import Chem

from generative_model.conformer.etflow_bridge import (
    audit_model_molecule,
    audit_official_node_features,
    build_mapped_explicit_h_smiles,
    reorder_positions_to_canonical,
)
from generative_model.conformer.etflow_symmetry import project_etflow_conformer
from generative_model.inference.generate_etflow_symmetric_xyz import _atomic_json


def _water_sample() -> dict:
    return {
        "atomic_numbers": np.asarray([8, 1, 1], dtype=np.int64),
        "formal_charges": np.zeros(3, dtype=np.int64),
        "radical_electrons": np.zeros(3, dtype=np.int64),
        "bond_index": np.asarray([[0, 0], [1, 2]], dtype=np.int64),
        "bond_types": np.ones(2, dtype=np.int64),
        "target_operation_matrices": np.asarray(
            [np.eye(3), np.diag([-1.0, -1.0, 1.0])], dtype=np.float64
        ),
        "target_permutation_index": np.asarray([[0, 1, 2], [0, 2, 1]], dtype=np.int64),
    }


class TestETFlowBridge(unittest.TestCase):
    def test_explicit_h_graph_roundtrip_is_exact(self) -> None:
        sample = _water_sample()
        smiles = build_mapped_explicit_h_smiles(sample)
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        audit = audit_model_molecule(sample, molecule, maximum_atomic_number=100)
        self.assertEqual(audit["atom_count"], 3)
        self.assertEqual(audit["explicit_h_count"], 2)
        self.assertFalse(audit["element_fallback_used"])
        self.assertFalse(audit["graph_mutation_used"])

    def test_atom_maps_losslessly_restore_canonical_coordinates(self) -> None:
        canonical = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 2.0, 4.0]])
        model_to_canonical = np.asarray([2, 0, 1], dtype=np.int64)
        model_order = canonical[model_to_canonical]
        restored = reorder_positions_to_canonical(
            model_order, model_to_canonical, atom_count=3
        )
        np.testing.assert_array_equal(restored, canonical)
        stacked = reorder_positions_to_canonical(
            np.stack([model_order, model_order + 1.0]), model_to_canonical, atom_count=3
        )
        np.testing.assert_array_equal(stacked[0], canonical)

    def test_sn_is_preserved_and_embedding_bound_is_strict(self) -> None:
        sample = {
            "atomic_numbers": np.asarray([50], dtype=np.int64),
            "formal_charges": np.asarray([0], dtype=np.int64),
            "radical_electrons": np.asarray([0], dtype=np.int64),
            "bond_index": np.empty((2, 0), dtype=np.int64),
            "bond_types": np.empty(0, dtype=np.int64),
        }
        atom = Chem.Atom(50); atom.SetNoImplicit(True); atom.SetAtomMapNum(1)
        editable = Chem.RWMol(); editable.AddAtom(atom); molecule = editable.GetMol()
        audit = audit_model_molecule(sample, molecule, maximum_atomic_number=100)
        self.assertEqual(audit["sn_count"], 1)
        self.assertTrue(audit["sn_preserved"])
        with self.assertRaises(ValueError):
            audit_model_molecule(sample, molecule, maximum_atomic_number=50)

    def test_missing_atom_map_and_typed_bond_change_fail(self) -> None:
        sample = _water_sample()
        molecule = Chem.MolFromSmiles("[OH2]")
        with self.assertRaises(ValueError):
            audit_model_molecule(sample, molecule, maximum_atomic_number=100)
        smiles = build_mapped_explicit_h_smiles(sample)
        molecule = Chem.RWMol(Chem.AddHs(Chem.MolFromSmiles(smiles)))
        molecule.RemoveBond(0, 1)
        with self.assertRaises(ValueError):
            audit_model_molecule(sample, molecule.GetMol(), maximum_atomic_number=100)

    def test_official_misc_buckets_are_explicitly_reported(self) -> None:
        node_attr = np.zeros((3, 10), dtype=np.float32)
        node_attr[0, 5] = 6
        audit = audit_official_node_features(node_attr, atom_count=3)
        self.assertEqual(audit["misc_counts"]["hybridization"], 1)
        self.assertTrue(audit["misc_is_reported_not_silenced"])

    def test_etflow_projection_is_exact_without_reference_coordinates(self) -> None:
        sample = _water_sample()
        raw = np.asarray([[0.1, 0.0, 0.2], [1.2, 0.7, -0.1], [-0.8, 0.4, 0.3]])
        first = project_etflow_conformer(raw, sample)
        second = project_etflow_conformer(raw, sample)
        np.testing.assert_array_equal(first["positions"], second["positions"])
        self.assertLess(first["operation_errors"]["max_atom_error_angstrom"], 1e-8)
        self.assertFalse(first["reference_coordinates_used"])
        self.assertFalse(first["etkdg_coordinates_used"])

    def test_ef1_v2_serializes_numpy_audit_fields_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "record.json"
            _atomic_json(output, {
                "model_to_canonical": np.asarray([2, 0, 1], dtype=np.int64),
                "count": np.int64(3), "score": np.float64(0.5),
            })
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["model_to_canonical"], [2, 0, 1])
            self.assertEqual(value["count"], 3)
            self.assertEqual(value["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
