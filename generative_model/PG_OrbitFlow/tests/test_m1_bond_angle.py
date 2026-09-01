from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from generative_model.PG_OrbitFlow.geometry import build_geometry_contract
from generative_model.PG_OrbitFlow.m0_oracle_reconstruction import build_oracle_targets
from generative_model.PG_OrbitFlow.m1_bond_angle_training import (
    _load_protocol,
    _predicted_decoder_targets,
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
    / "m1_bond_angle_tier4_v1.json"
)


class TestM1BondAngle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = _load_protocol(PROTOCOL)
        tier = {"records": cls.protocol["panel"]["records"], "evaluation": {}, "gate": {}}
        cls.samples = _panel_samples(cls.protocol, tier)

    def test_graph_input_is_separate_from_coordinate_targets(self):
        for sample in self.samples:
            contract = build_geometry_contract(sample)
            graph, targets = build_orbit_ic_example(sample, contract)
            self.assertFalse(hasattr(graph, "bond_target_lengths"))
            self.assertFalse(hasattr(graph, "angle_target_cosines"))
            self.assertEqual(graph.node_features.shape[0], sample.quotient_graph.orbit_count)
            self.assertEqual(
                len(targets.bond_target_lengths),
                len(np.unique(contract.bond_orbit_id)),
            )
            self.assertEqual(
                len(targets.angle_target_cosines),
                len(np.unique(contract.angle_orbit_id)),
            )

    def test_model_outputs_valid_orbit_shapes_and_finite_backward(self):
        import torch

        torch.manual_seed(19)
        sample = self.samples[0]
        contract = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, contract)
        setting = {
            key: self.protocol["model"][key]
            for key in ("node_feature_dim", "edge_feature_dim", "hidden_dim", "layers")
        }
        model = QuotientICPredictor(**setting)
        prediction = model(graph, device="cpu")
        self.assertEqual(prediction["bond_lengths"].shape, targets.bond_target_lengths.shape)
        self.assertEqual(prediction["angle_cosines"].shape, targets.angle_target_cosines.shape)
        self.assertTrue(torch.all(prediction["bond_lengths"] > 0.5))
        self.assertTrue(torch.all(torch.abs(prediction["angle_cosines"]) < 1.0))
        loss = torch.mean(
            torch.square(
                prediction["bond_lengths"]
                - torch.as_tensor(targets.bond_target_lengths)
            )
        ) + torch.mean(
            torch.square(
                prediction["angle_cosines"]
                - torch.as_tensor(targets.angle_target_cosines)
            )
        )
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_decoder_uses_predictions_and_disables_oracle_local_pairs(self):
        sample = self.samples[0]
        contract = build_geometry_contract(sample)
        _, targets = build_orbit_ic_example(sample, contract)
        oracle = build_oracle_targets(sample, contract)
        prediction = {
            "bond_lengths": targets.bond_target_lengths + 0.01,
            "angle_cosines": targets.angle_target_cosines + 0.001,
        }
        decoder = _predicted_decoder_targets(prediction, contract, oracle)
        self.assertTrue(np.all(decoder.local_pair_lengths == 0.0))
        self.assertTrue(
            np.allclose(
                decoder.bond_lengths,
                prediction["bond_lengths"][contract.bond_orbit_id],
            )
        )
        self.assertTrue(
            np.allclose(
                decoder.angle_cosines,
                prediction["angle_cosines"][contract.angle_orbit_id],
            )
        )
        self.assertTrue(np.array_equal(decoder.torsion_sincos, oracle.torsion_sincos))
        self.assertEqual(self.protocol["decoder"]["energy_weights"]["local_pair"], 0.0)


if __name__ == "__main__":
    unittest.main()
