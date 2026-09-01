import json
import unittest
from pathlib import Path

import numpy as np
import torch

from generative_model.PG_OrbitFlow.geometry import build_geometry_contract
from generative_model.PG_OrbitFlow.graph_automorphism import (
    automorphism_circular_errors_degrees,
    build_graph_automorphism_contract,
    optimal_automorphism_circular_assignment,
)
from generative_model.PG_OrbitFlow.m3_tier32_training import _model_setting
from generative_model.PG_OrbitFlow.m3p1_model import AutomorphismSetQuotientICPredictor
from generative_model.PG_OrbitFlow.orbit_ic_model import build_orbit_ic_example
from generative_model.PG_OrbitFlow.overfit import _panel_samples


ROOT = Path(__file__).resolve().parents[1]


class TestM3P1Automorphism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT / "configs/m3_tier32_v2.json").read_text(encoding="utf-8"))
        cls.samples = _panel_samples(cls.protocol, {"records": cls.protocol["panel"]["records"], "evaluation": {}, "gate": {}})

    def test_boh2_branch_exchange_is_detected_without_coordinates(self):
        sample = next(row for row in self.samples if row.package_index == 39)
        contract = build_graph_automorphism_contract(sample, build_geometry_contract(sample))
        self.assertIn((4, 5), contract.groups)
        self.assertIn((9, 10), contract.groups)
        for permutation in contract.allowed_atom_permutations:
            for action in sample.permutation_index:
                np.testing.assert_array_equal(permutation[action], action[permutation])

    def test_global_assignment_uses_one_allowed_automorphism(self):
        sample = next(row for row in self.samples if row.package_index == 39)
        geometry = build_geometry_contract(sample)
        contract = build_graph_automorphism_contract(sample, geometry)
        _, targets = build_orbit_ic_example(sample, geometry)
        permutation = contract.allowed_torsion_permutations[-1]
        prediction = targets.torsion_target_sincos[permutation]
        errors = automorphism_circular_errors_degrees(prediction, targets.torsion_target_sincos, contract)
        np.testing.assert_allclose(errors, 0.0, atol=1e-5)

    def test_generalized_head_supports_six_slots_and_inherits_parent(self):
        sample = next(row for row in self.samples if row.package_index == 1786)
        geometry = build_geometry_contract(sample)
        contract = build_graph_automorphism_contract(sample, geometry)
        self.assertEqual(max(map(len, contract.groups)), 6)
        graph, targets = build_orbit_ic_example(sample, geometry)
        parent_state = torch.load(ROOT / "runs/m3_tier32_v2/last.pt", map_location="cpu", weights_only=False)["model"]
        model = AutomorphismSetQuotientICPredictor(**_model_setting(self.protocol))
        inheritance = model.initialize_from_m3_parent(parent_state)
        isolation = model.configure_trainable_parameters()
        self.assertTrue(inheritance["automorphism_head_compatible_initialization"])
        self.assertGreater(isolation["trainable_parameter_count"], 0)
        prediction = model(graph, contract, device="cpu")
        self.assertEqual(prediction["torsion_sincos"].shape, targets.torsion_target_sincos.shape)
        target = torch.as_tensor(targets.torsion_target_sincos)
        aligned = optimal_automorphism_circular_assignment(prediction["torsion_sincos"], target, contract)
        loss = (1.0 - (prediction["torsion_sincos"] * aligned).sum(dim=-1).clamp(-1, 1)).mean()
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.parent.local_rotor_head.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.automorphism_head.parameters()))


if __name__ == "__main__":
    unittest.main()
