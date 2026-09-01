import json
import unittest
from pathlib import Path

import torch

from generative_model.PG_OrbitFlow.geometry import build_geometry_contract
from generative_model.PG_OrbitFlow.graph_automorphism import build_graph_automorphism_contract
from generative_model.PG_OrbitFlow.m3_tier32_training import _model_setting
from generative_model.PG_OrbitFlow.m3p1_model import AutomorphismSetQuotientICPredictor
from generative_model.PG_OrbitFlow.m3p2_torsion_refinement import _predict_cached
from generative_model.PG_OrbitFlow.orbit_ic_model import build_orbit_ic_example
from generative_model.PG_OrbitFlow.overfit import _panel_samples


ROOT = Path(__file__).resolve().parents[1]


class TestM3P2Refinement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "configs/m3p1_automorphism_tier32_v2.json").read_text(
                encoding="utf-8"
            )
        )
        cls.model = AutomorphismSetQuotientICPredictor(
            **_model_setting(cls.protocol)
        )
        state = torch.load(
            ROOT / "runs/m3p1_automorphism_tier32_v2/last.pt",
            map_location="cpu",
            weights_only=False,
        )["model"]
        cls.model.load_state_dict(state, strict=True)

    def test_only_torsion_heads_are_trainable(self):
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for module in (
            self.model.parent.base.torsion_head,
            self.model.automorphism_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        trainable = {
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        expected = {
            name for name, _ in self.model.parent.base.torsion_head.named_parameters(
                prefix="parent.base.torsion_head"
            )
        } | {
            name for name, _ in self.model.automorphism_head.named_parameters(
                prefix="automorphism_head"
            )
        }
        self.assertEqual(trainable, expected)

    def test_cached_features_reproduce_full_forward_torsions(self):
        samples = _panel_samples(
            self.protocol,
            {
                "records": self.protocol["panel"]["records"],
                "evaluation": {},
                "gate": {},
            },
        )
        sample = next(row for row in samples if row.package_index == 39)
        geometry = build_geometry_contract(sample)
        graph, _ = build_orbit_ic_example(sample, geometry)
        automorphism = build_graph_automorphism_contract(sample, geometry)
        self.model.eval()
        with torch.no_grad():
            full = self.model(graph, automorphism, device="cpu")["torsion_sincos"]
            features = self.model.parent._captured_torsion_input.detach().clone()
            cached = _predict_cached(
                self.model, features, automorphism, device="cpu"
            )
        torch.testing.assert_close(cached, full, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
