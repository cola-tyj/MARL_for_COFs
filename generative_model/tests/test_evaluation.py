"""统一生成分子评测契约与参考报告测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from generative_model.data import COFSymmetryDataset
from generative_model.evaluation import EvaluationConfig, evaluate, validate_report, write_report
from generative_model.evaluation.evaluate_generated_npz import load_samples
from generative_model.evaluation.reference import build_reference_report


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "data/processed/v2"


def _sample() -> dict[str, object]:
    return {
        "molecule_id": "test_molecule",
        "atomic_numbers": np.asarray([6, 1], dtype=np.int16),
        "formal_charges": np.asarray([0, 0], dtype=np.int8),
        "positions": np.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32),
        "bond_index": np.asarray([[0], [1]], dtype=np.int64),
        "bond_types": np.asarray([1], dtype=np.uint8),
        "target_pg": "C2",
        "actual_pg": "C2",
        "pg_compatible": True,
    }


def _provenance() -> dict[str, str]:
    return {
        "source_type": "unit_test",
        "source_fingerprint": "0" * 64,
        "coordinate_unit": "angstrom",
    }


class TestEvaluationContract(unittest.TestCase):
    def test_valid_sample_produces_deterministic_report(self) -> None:
        config = EvaluationConfig(run_mmff=False, run_symmetry=False)
        first = evaluate([_sample()], provenance=_provenance(), report_type="unit_test", config=config)
        second = evaluate([_sample()], provenance=_provenance(), report_type="unit_test", config=config)
        self.assertEqual(first, second)
        self.assertEqual(first["summary"]["status"], "PASS")
        validate_report(first)
        self.assertEqual(first["metrics"]["graph"]["connected_rate"], 1.0)
        self.assertEqual(first["metrics"]["provided_conditions"]["target_pg_counts"], {"C2": 1})
        self.assertEqual(first["metrics"]["chemistry"]["rdkit_sanitize_valid_rate"], 1.0)

    def test_schema_violation_is_recorded_with_stable_reason(self) -> None:
        invalid = _sample()
        invalid["bond_index"] = np.asarray([[1], [0]], dtype=np.int64)
        report = evaluate(
            [invalid],
            provenance=_provenance(),
            report_type="unit_test",
            config=EvaluationConfig(run_mmff=False, run_symmetry=False),
        )
        self.assertEqual(report["summary"]["status"], "FAIL")
        self.assertEqual(report["failures"]["counts"], {"non_canonical_bond": 1})
        self.assertEqual(report["summary"]["evaluated_molecules"], 0)

    def test_non_finite_coordinates_are_measured_not_silently_repaired(self) -> None:
        invalid = _sample()
        invalid["positions"] = np.asarray([[np.nan, 0.0, 0.0], [0.5, 0.0, 0.0]])
        report = evaluate(
            [invalid],
            provenance=_provenance(),
            report_type="unit_test",
            config=EvaluationConfig(run_mmff=False, run_symmetry=False),
        )
        self.assertEqual(report["summary"]["status"], "PASS")
        self.assertEqual(report["metrics"]["numerical"]["finite_coordinate_rate"], 0.0)
        self.assertEqual(report["metrics"]["numerical"]["max_radius_angstrom"]["count"], 0)

    def test_writer_is_byte_deterministic(self) -> None:
        report = evaluate(
            [_sample()],
            provenance=_provenance(),
            report_type="unit_test",
            config=EvaluationConfig(run_mmff=False, run_symmetry=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_report(report, first)
            write_report(report, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(json.loads(first.read_text(encoding="utf-8")), report)

    def test_generated_compact_npz_roundtrip(self) -> None:
        sample = _sample()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.npz"
            np.savez(
                path,
                atomic_numbers=sample["atomic_numbers"],
                formal_charges=sample["formal_charges"],
                positions=sample["positions"],
                bond_index=sample["bond_index"],
                bond_types=sample["bond_types"],
                atom_offsets=np.asarray([0, 2], dtype=np.int64),
                bond_offsets=np.asarray([0, 1], dtype=np.int64),
            )
            loaded = load_samples(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["molecule_id"], "generated_000000")
        for field in (
            "atomic_numbers",
            "formal_charges",
            "positions",
            "bond_index",
            "bond_types",
        ):
            np.testing.assert_array_equal(loaded[0][field], sample[field])

    def test_real_test_splits_pass_and_are_distinct(self) -> None:
        iid = build_reference_report(PACKAGE_DIR, split_scheme="iid")
        core_ood = build_reference_report(PACKAGE_DIR, split_scheme="core_ood")
        self.assertEqual(iid["summary"]["status"], "PASS")
        self.assertEqual(core_ood["summary"]["status"], "PASS")
        self.assertEqual(iid["summary"]["evaluated_molecules"], 253)
        self.assertEqual(core_ood["summary"]["evaluated_molecules"], 258)
        self.assertEqual(iid["metrics"]["chemistry"]["rdkit_sanitize_valid_rate"], 1.0)
        self.assertEqual(core_ood["metrics"]["symmetry"]["actual_label_match_rate"], 1.0)
        self.assertEqual(core_ood["metrics"]["two_dimensional"]["novelty_rate"], 1.0)
        self.assertNotEqual(iid["provenance"]["source_fingerprint"], core_ood["provenance"]["source_fingerprint"])
        self.assertEqual(len(bytes.fromhex(iid["report_fingerprint"])), hashlib.sha256().digest_size)

    def test_sn_remains_sn_and_mmff_reports_unsupported(self) -> None:
        dataset = COFSymmetryDataset(PACKAGE_DIR)
        report = evaluate(
            [dataset[1684]],
            provenance=_provenance(),
            report_type="sn_audit",
        )
        self.assertEqual(report["metrics"]["two_dimensional"]["element_counts"]["Sn"], 2)
        self.assertEqual(report["metrics"]["chemistry"]["rdkit_sanitize_valid_rate"], 1.0)
        self.assertEqual(
            report["metrics"]["three_dimensional"]["mmff"]["status_counts"],
            {"unsupported_parameters": 1},
        )
        self.assertEqual(report["metrics"]["symmetry"]["actual_label_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
