import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from generative_model.PG_OrbitFlow.global_shape_fingerprint_model import (
    FingerprintGlobalShapePredictor,
    graph_wl_fingerprint,
)
from generative_model.PG_OrbitFlow.m3_tier32_training import _model_setting
from generative_model.PG_OrbitFlow.overfit import _panel_samples


ROOT = Path(__file__).resolve().parents[1]


class TestM3P6FingerprintShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "configs/m3p6_fingerprint_shape_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_wl_fingerprint_is_atom_order_invariant(self):
        sample = _panel_samples(
            self.protocol,
            {
                "records": self.protocol["panel"]["records"],
                "evaluation": {},
                "gate": {},
            },
        )[0]
        atom_count = len(sample.atom_types)
        permutation = np.random.default_rng(20260831).permutation(atom_count)
        old_to_new = np.argsort(permutation)
        reordered = replace(
            sample,
            atom_types=sample.atom_types[permutation],
            atomic_numbers=sample.atomic_numbers[permutation],
            formal_charges=sample.formal_charges[permutation],
            radical_electrons=sample.radical_electrons[permutation],
            bond_index=old_to_new[sample.bond_index],
        )
        settings = self.protocol["model"]
        original = graph_wl_fingerprint(
            sample,
            bits=int(settings["wl_fingerprint_bits"]),
            radius=int(settings["wl_fingerprint_radius"]),
        )
        candidate = graph_wl_fingerprint(
            reordered,
            bits=int(settings["wl_fingerprint_bits"]),
            radius=int(settings["wl_fingerprint_radius"]),
        )
        np.testing.assert_array_equal(candidate, original)

    def test_wl_fingerprint_settings_fail_strictly(self):
        sample = _panel_samples(
            self.protocol,
            {
                "records": self.protocol["panel"]["records"],
                "evaluation": {},
                "gate": {},
            },
        )[0]
        with self.assertRaises(ValueError):
            graph_wl_fingerprint(sample, bits=0, radius=4)
        with self.assertRaises(ValueError):
            graph_wl_fingerprint(sample, bits=512, radius=-1)

    def test_shape_head_only_freezes_parent_and_outputs_sorted_quantiles(self):
        settings = self.protocol["model"]
        model = FingerprintGlobalShapePredictor(
            quantile_count=int(settings["global_shape_quantile_count"]),
            fingerprint_bits=int(settings["wl_fingerprint_bits"]),
            **_model_setting(self.protocol),
        )
        trainable_names = model.configure_shape_head_only()
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(name.startswith("global_shape_head.") for name in trainable_names)
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.parent.parameters())
        )
        cached_dim = 2 * int(settings["hidden_dim"]) + int(
            settings["wl_fingerprint_bits"]
        )
        prediction = model.predict_cached(torch.zeros(cached_dim))
        self.assertEqual(
            tuple(prediction.shape),
            (int(settings["global_shape_quantile_count"]),),
        )
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.all(prediction[1:] >= prediction[:-1]))


if __name__ == "__main__":
    unittest.main()
