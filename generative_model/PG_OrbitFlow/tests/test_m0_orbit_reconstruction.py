from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from generative_model.PG_OrbitFlow.c0_local_recovery import _load_protocol as load_c0
from generative_model.PG_OrbitFlow.geometry import build_geometry_contract
from generative_model.PG_OrbitFlow.group import operation_error_numpy
from generative_model.PG_OrbitFlow.m0_oracle_reconstruction import (
    _load_protocol,
    build_oracle_targets,
    oracle_energy,
)
from generative_model.PG_OrbitFlow.orbit_kinematics import (
    build_orbit_parameterization,
    encode_orbit_parameters,
    lift_orbit_parameters,
)
from generative_model.PG_OrbitFlow.overfit import _panel_samples


ROOT = Path(__file__).resolve().parents[3]
M0_PROTOCOL = (
    ROOT
    / "generative_model"
    / "PG_OrbitFlow"
    / "configs"
    / "m0_oracle_reconstruction_v1.json"
)
C0_PROTOCOL = (
    ROOT
    / "generative_model"
    / "PG_OrbitFlow"
    / "configs"
    / "c0_local_recovery_v2_pg_balanced.json"
)


class TestM0OrbitReconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = _load_protocol(M0_PROTOCOL)
        c0 = load_c0(C0_PROTOCOL)
        tier = {
            "records": c0["panel"]["records"],
            "evaluation": c0["original_tier4_evaluation"],
            "gate": c0["original_tier4_gate"],
        }
        cls.samples = _panel_samples(c0, tier)

    def test_protocol_is_oracle_only_and_preserves_formal_panel(self):
        c0 = json.loads(C0_PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(self.protocol["panel"], c0["panel"])
        self.assertFalse(
            self.protocol["oracle_targets"]["target_cartesian_coordinate_loss_used"]
        )
        self.assertFalse(
            self.protocol["oracle_targets"]["target_cartesian_initialization_used"]
        )
        self.assertFalse(self.protocol["isolation"]["neural_network_used"])
        self.assertFalse(self.protocol["isolation"]["mds_used"])

    def test_real_panel_target_roundtrip_is_exact(self):
        import torch

        for sample in self.samples:
            specification = build_orbit_parameterization(sample)
            parameters = encode_orbit_parameters(
                sample.symmetric_target_angstrom, specification
            )
            reconstructed = lift_orbit_parameters(
                torch.as_tensor(parameters, dtype=torch.float64), specification
            ).numpy()
            self.assertLess(
                float(
                    np.max(
                        np.abs(reconstructed - sample.symmetric_target_angstrom)
                    )
                ),
                1e-5,
            )
            self.assertLess(
                operation_error_numpy(
                    reconstructed,
                    sample.operation_matrices,
                    sample.permutation_index,
                )["max_atom_error_angstrom"],
                1e-6,
            )

    def test_stabilizer_fixed_atom_has_one_dimensional_basis(self):
        import torch

        synthetic = SimpleNamespace(
            operation_matrices=np.asarray(
                [np.eye(3), np.diag([-1.0, -1.0, 1.0])], dtype=np.float64
            ),
            permutation_index=np.asarray([[0, 1, 2], [0, 2, 1]], dtype=np.int64),
            orbit_id=np.asarray([0, 1, 1], dtype=np.int64),
            atomic_numbers=np.asarray([6, 6, 6], dtype=np.int64),
        )
        specification = build_orbit_parameterization(synthetic)
        self.assertEqual(specification.fixed_dimensions.tolist(), [1, 3])
        parameters = torch.tensor([1.2, 0.5, -0.7, 0.3], dtype=torch.float64)
        coordinates = lift_orbit_parameters(parameters, specification)
        self.assertLess(float(torch.abs(coordinates[0, :2]).max()), 1e-12)
        self.assertLess(
            operation_error_numpy(
                coordinates.numpy(),
                synthetic.operation_matrices,
                synthetic.permutation_index,
            )["max_atom_error_angstrom"],
            1e-12,
        )

    def test_oracle_energy_is_zero_on_target_and_has_finite_gradient(self):
        import torch

        sample = self.samples[0]
        contract = build_geometry_contract(sample)
        targets = build_oracle_targets(sample, contract)
        specification = build_orbit_parameterization(sample)
        encoded = encode_orbit_parameters(sample.symmetric_target_angstrom, specification)
        parameters = torch.tensor(encoded, dtype=torch.float64, requires_grad=True)
        coordinates = lift_orbit_parameters(parameters, specification)
        values = oracle_energy(
            coordinates,
            sample,
            contract,
            targets,
            self.protocol["energy_weights"],
        )
        self.assertLess(float(values["total"].item()), 1e-10)
        values["total"].backward()
        self.assertTrue(torch.isfinite(parameters.grad).all())


if __name__ == "__main__":
    unittest.main()
