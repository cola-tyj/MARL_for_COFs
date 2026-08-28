from __future__ import annotations

import unittest

import numpy as np

from generative_model.models.uae3d_gate_a import (
    fit_random_fourier_map,
    gate_a_checks,
    random_fourier_features,
)


class TestUAE3DGateA(unittest.TestCase):
    def test_random_fourier_map_is_deterministic(self) -> None:
        features = np.arange(8 * 64, dtype=np.float64).reshape(8, 64) / 100.0
        first = fit_random_fourier_map(features, output_dim=16, seed=20260821)
        second = fit_random_fourier_map(features, output_dim=16, seed=20260821)
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        np.testing.assert_array_equal(
            random_fourier_features(first, features),
            random_fourier_features(second, features),
        )
        self.assertEqual(random_fourier_features(first, features).shape, (8, 80))

    def test_gate_requires_every_predeclared_check(self) -> None:
        metric = {
            "macro_molecule_balanced_existence_accuracy": 0.83,
            "minimum_molecule_balanced_existence_accuracy": 0.70,
            "none_accuracy": 0.80,
            "present_existence_recall": 0.80,
            "existence_exact_molecules": 4,
        }
        result = {
            "source_model_unchanged": True,
            "all_probe_outputs_finite": True,
            "tier4_oracle": {
                "nonlinear": {"evaluation_at_unchanged_threshold": dict(metric)}
            },
            "iid_transfer": {
                "linear": {"evaluation_at_unchanged_threshold": {
                    "macro_molecule_balanced_existence_accuracy": 0.80
                }},
                "nonlinear": {
                    "evaluation_roc_auc_all_pairs": 0.91,
                    "evaluation_at_unchanged_threshold": dict(metric),
                },
            },
        }
        thresholds = {
            "tier4_existence_exact_molecules_min": 4,
            "iid_transfer_roc_auc_min": 0.90,
            "iid_transfer_macro_balanced_min": 0.82,
            "iid_transfer_minimum_balanced_min": 0.65,
            "iid_transfer_none_accuracy_min": 0.75,
            "iid_transfer_present_recall_min": 0.75,
            "nonlinear_macro_gain_over_linear_min": 0.01,
        }
        self.assertTrue(all(gate_a_checks(result, thresholds).values()))
        result["iid_transfer"]["nonlinear"]["evaluation_roc_auc_all_pairs"] = 0.89
        checks = gate_a_checks(result, thresholds)
        self.assertFalse(checks["iid_transfer_roc_auc_min"])
        self.assertFalse(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
