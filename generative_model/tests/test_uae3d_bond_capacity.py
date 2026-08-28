"""Tests for deterministic UAE-3D frozen-feature capacity probes."""

from __future__ import annotations

import unittest

import numpy as np

from generative_model.models.uae3d_bond_capacity import (
    balanced_fit_rows,
    binary_scores,
    feature_table_fingerprint,
    fit_binary_logistic,
    fit_type_ridge,
    make_feature_table,
    roc_auc,
    type_metrics,
    type_predictions,
    validate_feature_table,
)


class TestUAE3DBondCapacity(unittest.TestCase):
    @staticmethod
    def _table():
        targets = np.asarray([0, 1, 0, 2, 0, 0, 3, 4], dtype=np.int16)
        base = np.asarray(targets > 0, dtype=np.float32)[:, None]
        pair = np.repeat(base, 64, axis=1)
        pair[:, 1] = np.arange(len(pair), dtype=np.float32)
        hidden = pair * np.float32(0.5)
        logits = np.zeros((len(pair), 6), dtype=np.float32)
        return make_feature_table(
            pair, hidden, logits, targets,
            np.asarray([0, 4, 8]), np.asarray([10, 20]),
        )

    def test_balanced_rows_are_deterministic_per_molecule(self) -> None:
        table = self._table()
        first = balanced_fit_rows(table)
        second = balanced_fit_rows(table)
        np.testing.assert_array_equal(first, second)
        labels = table["targets"][first] > 0
        self.assertEqual(int(labels.sum()) * 2, len(labels))

    def test_logistic_and_auc_detect_separable_signal(self) -> None:
        features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        labels = np.asarray([0, 0, 1, 1])
        model = fit_binary_logistic(features, labels)
        scores = binary_scores(model, features)
        self.assertTrue(model["converged"])
        self.assertAlmostEqual(roc_auc(scores, labels), 1.0)
        self.assertAlmostEqual(roc_auc(np.asarray([0.0, 0.0]), np.asarray([0, 1])), 0.5)

    def test_type_ridge_roundtrip(self) -> None:
        features = np.eye(4, dtype=np.float64).repeat(2, axis=0)
        targets = np.arange(1, 5, dtype=np.int64).repeat(2)
        # repeat() orders rows [e1,e1,e2,e2,...], matching repeated labels.
        model = fit_type_ridge(features, targets)
        predicted = type_predictions(model, features)
        metrics = type_metrics(predicted, targets)
        self.assertEqual(metrics["micro_accuracy"], 1.0)
        self.assertEqual(metrics["macro_supported_class_accuracy"], 1.0)

    def test_fingerprint_rejects_tampering(self) -> None:
        table = self._table()
        self.assertEqual(table["fingerprint"], feature_table_fingerprint(table))
        table["pair_features"] = table["pair_features"].copy()
        table["pair_features"][0, 0] += 1.0
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            validate_feature_table(table)


if __name__ == "__main__":
    unittest.main()
