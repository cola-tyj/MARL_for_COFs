from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from generative_model.PG_OrbitFlow.geometry import build_geometry_contract
from generative_model.PG_OrbitFlow.m0_oracle_reconstruction import build_oracle_targets
from generative_model.PG_OrbitFlow.m2_torsion_training import (
    _learned_decoder_targets,
    _load_protocol,
)
from generative_model.PG_OrbitFlow.audit_m2_tier16_panel import (
    _head_feature_conflicts,
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
    / "m2_torsion_tier4_v1.json"
)


class TestM2Torsion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = _load_protocol(PROTOCOL)
        tier = {"records": cls.protocol["panel"]["records"], "evaluation": {}, "gate": {}}
        cls.samples = _panel_samples(cls.protocol, tier)

    def test_torsion_targets_are_orbit_consistent_and_planar(self):
        for sample in self.samples:
            contract = build_geometry_contract(sample)
            _, targets = build_orbit_ic_example(sample, contract)
            oracle = build_oracle_targets(sample, contract)
            expanded = targets.torsion_target_sincos[contract.torsion_orbit_id]
            self.assertTrue(np.allclose(expanded, oracle.torsion_sincos, atol=1e-6))
            self.assertTrue(
                np.allclose(
                    np.linalg.norm(targets.torsion_target_sincos, axis=1),
                    1.0,
                    atol=1e-6,
                )
            )
            angles = np.abs(
                np.rad2deg(
                    np.arctan2(
                        targets.torsion_target_sincos[:, 0],
                        targets.torsion_target_sincos[:, 1],
                    )
                )
            )
            distance_to_planar = np.minimum(angles, np.abs(180.0 - angles))
            self.assertTrue(np.all(distance_to_planar <= 1e-3))

    def test_model_torsion_output_is_normalized_and_has_finite_backward(self):
        import torch

        torch.manual_seed(29)
        sample = self.samples[0]
        contract = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, contract)
        setting = {
            key: self.protocol["model"][key]
            for key in (
                "node_feature_dim",
                "edge_feature_dim",
                "hidden_dim",
                "layers",
                "predict_torsion",
            )
        }
        model = QuotientICPredictor(**setting)
        prediction = model(graph, device="cpu")
        self.assertEqual(
            prediction["torsion_sincos"].shape,
            targets.torsion_target_sincos.shape,
        )
        self.assertTrue(
            torch.allclose(
                prediction["torsion_sincos"].norm(dim=-1),
                torch.ones(len(targets.torsion_target_sincos)),
                atol=1e-5,
            )
        )
        loss = 1.0 - torch.mean(
            torch.sum(
                prediction["torsion_sincos"]
                * torch.as_tensor(targets.torsion_target_sincos),
                dim=-1,
            )
        )
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_decoder_is_all_learned_and_oracle_free(self):
        sample = self.samples[0]
        contract = build_geometry_contract(sample)
        _, targets = build_orbit_ic_example(sample, contract)
        prediction = {
            "bond_lengths": targets.bond_target_lengths + 0.01,
            "angle_cosines": targets.angle_target_cosines + 0.001,
            "torsion_sincos": targets.torsion_target_sincos.copy(),
        }
        decoder = _learned_decoder_targets(prediction, contract)
        self.assertTrue(np.all(decoder.local_pair_lengths == 0.0))
        self.assertTrue(np.all(decoder.chirality == 0.0))
        self.assertTrue(
            np.array_equal(
                decoder.torsion_sincos,
                prediction["torsion_sincos"][contract.torsion_orbit_id],
            )
        )
        self.assertFalse(
            self.protocol["isolation"]["oracle_internal_coordinates_used_by_decoder"]
        )

    def test_phase_features_resolve_nonplanar_quotient_aliasing(self):
        import json
        import torch

        panel_path = (
            ROOT
            / "generative_model"
            / "PG_OrbitFlow"
            / "configs"
            / "m2_tier16_panel_v2_phase_aware.json"
        )
        panel = json.loads(panel_path.read_text(encoding="utf-8"))
        record = next(
            row for row in panel["records"] if int(row["package_index"]) == 1124
        )
        sample = _panel_samples(
            {
                "package_dir": panel["package_dir"],
                "canonical_manifest_sha256": panel["canonical_manifest_sha256"],
            },
            {"records": [record], "evaluation": {}, "gate": {}},
        )[0]
        contract = build_geometry_contract(sample)
        conflicts = _head_feature_conflicts(sample, contract)
        self.assertGreater(
            conflicts["bond_head_exact_feature_conflicts"]
            + conflicts["angle_head_exact_feature_conflicts"]
            + conflicts["torsion_head_exact_feature_conflicts"],
            0,
        )
        self.assertEqual(
            conflicts["phase_aware_bond_head_exact_feature_conflicts"]
            + conflicts["phase_aware_angle_head_exact_feature_conflicts"]
            + conflicts["phase_aware_torsion_head_exact_feature_conflicts"],
            0,
        )
        graph, targets = build_orbit_ic_example(sample, contract)
        model = QuotientICPredictor(
            node_feature_dim=29,
            edge_feature_dim=5,
            hidden_dim=96,
            layers=4,
            predict_torsion=True,
            geometry_group_phase=True,
        )
        prediction = model(graph, device="cpu")
        loss = (
            torch.mean(
                torch.square(
                    prediction["bond_lengths"]
                    - torch.as_tensor(targets.bond_target_lengths)
                )
            )
            + torch.mean(
                torch.square(
                    prediction["angle_cosines"]
                    - torch.as_tensor(targets.angle_target_cosines)
                )
            )
            + torch.mean(
                1.0
                - torch.sum(
                    prediction["torsion_sincos"]
                    * torch.as_tensor(targets.torsion_target_sincos),
                    dim=-1,
                )
            )
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
