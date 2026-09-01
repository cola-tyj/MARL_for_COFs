from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from generative_model.PG_OrbitFlow.c0_local_recovery import (
    _load_protocol,
    c0_loss,
    differentiable_short_rollout,
    make_local_recovery_example,
)
from generative_model.PG_OrbitFlow.geometry import (
    build_geometry_contract,
    geometry_loss_bundle,
)
from generative_model.PG_OrbitFlow.group import operation_error_numpy
from generative_model.PG_OrbitFlow.model import PGOrbitFlow
from generative_model.PG_OrbitFlow.overfit import _panel_samples


ROOT = Path(__file__).resolve().parents[3]
C0_PROTOCOL = (
    ROOT
    / "generative_model"
    / "PG_OrbitFlow"
    / "configs"
    / "c0_local_recovery_v2_pg_balanced.json"
)
H1_PROTOCOL = (
    ROOT
    / "generative_model"
    / "PG_OrbitFlow"
    / "configs"
    / "overfit_protocol_h1_harmonic.json"
)


class TestC0LocalRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = _load_protocol(C0_PROTOCOL)
        tier = {
            "records": cls.protocol["panel"]["records"],
            "evaluation": cls.protocol["original_tier4_evaluation"],
            "gate": cls.protocol["original_tier4_gate"],
        }
        cls.samples = _panel_samples(cls.protocol, tier)

    def test_protocol_preserves_panel_model_budget_and_original_gate(self):
        h1 = json.loads(H1_PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(self.protocol["model"], h1["model"])
        self.assertEqual(self.protocol["panel"]["records"], h1["tiers"]["4"]["records"])
        self.assertEqual(self.protocol["training"]["optimizer_steps"], 1024)
        self.assertEqual(self.protocol["training"]["batch_size_molecules"], 4)
        self.assertEqual(self.protocol["training"]["short_rollout_steps"], 4)
        self.assertTrue(
            self.protocol["curriculum"]["point_group_curriculum_independent"]
        )
        self.assertEqual(
            self.protocol["original_tier4_gate"], h1["tiers"]["4"]["gate"]
        )

    def test_geometry_contract_is_group_closed_and_exact_on_target(self):
        import torch

        for sample in self.samples:
            contract = build_geometry_contract(sample)
            self.assertGreater(contract.bond_index.shape[1], 0)
            self.assertGreater(contract.angle_index.shape[1], 0)
            self.assertGreater(contract.local_pair_index.shape[1], 0)
            target = torch.as_tensor(
                sample.symmetric_target_angstrom / 3.0, dtype=torch.float32
            )
            values = geometry_loss_bundle(
                target, target, contract, coordinate_scale_angstrom=3.0
            )
            self.assertLess(float(values["bond_loss"].item()), 1e-10)
            self.assertLess(float(values["angle_loss"].item()), 1e-10)
            self.assertLess(float(values["torsion_loss"].item()), 1e-6)
            self.assertEqual(float(values["chirality_preserved_fraction"].item()), 1.0)

    def test_local_noise_is_deterministic_scaled_and_group_invariant(self):
        sample = self.samples[0]
        kwargs = {
            "seed": 771,
            "sigma_range_angstrom": (0.10, 0.10),
            "device": "cpu",
            "coordinate_scale_angstrom": 3.0,
        }
        first = make_local_recovery_example(sample, **kwargs)
        second = make_local_recovery_example(sample, **kwargs)
        self.assertTrue(np.array_equal(first["positions_t"].numpy(), second["positions_t"].numpy()))
        self.assertAlmostEqual(float(first["time"].item()), 0.90, places=6)
        coordinates = first["positions_t"].numpy() * 3.0
        self.assertLess(
            operation_error_numpy(
                coordinates, sample.operation_matrices, sample.permutation_index
            )["max_atom_error_angstrom"],
            1e-5,
        )

    def test_c0_loss_and_four_step_rollout_have_finite_backward(self):
        import torch

        torch.manual_seed(41)
        sample = self.samples[0]
        contract = build_geometry_contract(sample)
        example = make_local_recovery_example(
            sample,
            seed=41,
            sigma_range_angstrom=(0.10, 0.10),
            device="cpu",
            coordinate_scale_angstrom=3.0,
        )
        model = PGOrbitFlow(hidden_dim=24, layers=2, radial_dim=6, radial_max=5.0)
        values = c0_loss(model, example, contract, self.protocol)
        self.assertTrue(torch.isfinite(values["loss"]))
        values["loss"].backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )
        rollout = differentiable_short_rollout(model, example, steps=4)
        self.assertLess(
            operation_error_numpy(
                rollout.detach().numpy() * 3.0,
                sample.operation_matrices,
                sample.permutation_index,
            )["max_atom_error_angstrom"],
            1e-5,
        )


if __name__ == "__main__":
    unittest.main()
