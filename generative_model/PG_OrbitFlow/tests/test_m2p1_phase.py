from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from generative_model.PG_OrbitFlow.geometry import build_geometry_contract
from generative_model.PG_OrbitFlow.m0_oracle_reconstruction import (
    build_oracle_targets,
    reconstruction_metrics,
)
from generative_model.PG_OrbitFlow.m2p1_phase_training import (
    _balanced_schedule,
    _inherit_backbone,
    _load_protocol,
    _model_setting,
)
from generative_model.PG_OrbitFlow.orbit_ic_model import (
    QuotientICPredictor,
    build_orbit_ic_example,
)
from generative_model.PG_OrbitFlow.overfit import _panel_samples


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = (
    ROOT
    / "generative_model"
    / "PG_OrbitFlow"
    / "configs"
    / "m2p1_phase_tier16_v1.json"
)


class TestM2P1PhaseAwareTier16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = _load_protocol(PROTOCOL)
        tier = {"records": cls.protocol["panel"]["records"], "evaluation": {}, "gate": {}}
        cls.samples = _panel_samples(cls.protocol, tier)
        cls.examples = []
        for sample in cls.samples:
            contract = build_geometry_contract(sample)
            graph, targets = build_orbit_ic_example(sample, contract)
            cls.examples.append((sample, contract, graph, targets))

    def test_panel_is_nested_balanced_nonplanar_and_chiral(self):
        self.assertEqual(len(self.samples), 16)
        self.assertEqual(sum(row.target_pg == "C2" for row in self.samples), 8)
        self.assertEqual(sum(row.target_pg == "C3" for row in self.samples), 8)
        tier4 = {864, 1045, 1135, 2391}
        self.assertTrue(tier4.issubset({row.package_index for row in self.samples}))
        for sample, contract, _, targets in self.examples:
            if sample.package_index in tier4:
                continue
            angles = np.abs(
                np.rad2deg(
                    np.arctan2(
                        targets.torsion_target_sincos[:, 0],
                        targets.torsion_target_sincos[:, 1],
                    )
                )
            )
            self.assertTrue(np.any(np.minimum(angles, np.abs(180.0 - angles)) > 5.0))
            oracle = build_oracle_targets(sample, contract)
            self.assertTrue(np.any(np.abs(oracle.chirality) >= 0.05))

    def test_balanced_schedule_has_exact_exposures(self):
        steps = int(self.protocol["training"]["optimizer_steps"])
        schedule = _balanced_schedule(self.examples, steps=steps, seed=991)
        self.assertEqual(len(schedule), steps)
        counts = np.bincount(np.asarray(schedule).reshape(-1), minlength=16)
        self.assertTrue(np.all(counts == 512))
        for batch in schedule:
            groups = [self.examples[index][0].target_pg for index in batch]
            self.assertEqual(groups.count("C2"), 2)
            self.assertEqual(groups.count("C3"), 2)

    def test_only_backbone_is_inherited_and_heads_are_reinitialized(self):
        import torch

        torch.manual_seed(123)
        model = QuotientICPredictor(**_model_setting(self.protocol))
        before = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if "head" in key
        }
        parent = torch.load(
            self.protocol["initialization"]["m2_checkpoint"]["path"],
            map_location="cpu",
            weights_only=False,
        )
        audit = _inherit_backbone(
            model,
            parent["model"],
            tuple(self.protocol["initialization"]["inherited_parameter_prefixes"]),
        )
        self.assertGreater(audit["inherited_parameter_count"], 0)
        self.assertTrue(
            all(torch.equal(before[key], model.state_dict()[key]) for key in before)
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            self.protocol["model"]["parameter_count_expected"],
        )

    def test_target_reconstruction_metrics_expose_nonplanar_chirality(self):
        sample, contract, _, _ = next(
            row for row in self.examples if row[0].package_index == 1124
        )
        oracle = build_oracle_targets(sample, contract)
        metrics = reconstruction_metrics(
            sample, contract, oracle, sample.symmetric_target_angstrom
        )
        self.assertGreater(metrics["nonplanar_torsion_count"], 0)
        self.assertGreater(metrics["active_chirality_count"], 0)
        self.assertAlmostEqual(metrics["chirality_preserved_fraction"], 1.0)
        self.assertLess(metrics["nonplanar_torsion_circular_mae_degrees"], 1e-4)


if __name__ == "__main__":
    unittest.main()
