from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from generative_model.PG_OrbitFlow.data import (
    PGOrbitFlowDataset,
    graph_laplacian,
    make_symmetric_noise,
)
from generative_model.PG_OrbitFlow.flow import (
    centralizer_averaged_transport,
    centralizer_rotations,
    sample_raw_ode,
)
from generative_model.PG_OrbitFlow.group import operation_error_numpy
from generative_model.PG_OrbitFlow.losses import compute_loss
from generative_model.PG_OrbitFlow.metrics import raw_geometry_metrics
from generative_model.PG_OrbitFlow.model import PGOrbitFlow


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "generative_model" / "data" / "processed" / "v2"


def _c3_matrices() -> np.ndarray:
    values = []
    for index in range(3):
        angle = 2.0 * np.pi * index / 3.0
        values.append(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    return np.asarray(values, dtype=np.float64)


class TestHarmonicPriorAndTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = PGOrbitFlowDataset(
            PACKAGE,
            split="train",
            point_groups=("C2", "C3"),
        )
        cls.samples = {
            point_group: next(
                cls.dataset[index]
                for index in range(len(cls.dataset))
                if cls.dataset.canonical[
                    int(cls.dataset.local_indices[index])
                ]["target_pg"]
                == point_group
            )
            for point_group in ("C2", "C3")
        }

    def test_harmonic_prior_is_deterministic_centered_and_symmetric(self):
        for sample in self.samples.values():
            first = make_symmetric_noise(
                sample,
                seed=12345,
                coordinate_scale_angstrom=3.0,
                prior_type="graph_harmonic",
            )
            second = make_symmetric_noise(
                sample,
                seed=12345,
                coordinate_scale_angstrom=3.0,
                prior_type="graph_harmonic",
            )
            self.assertTrue(np.array_equal(first, second))
            self.assertTrue(np.isfinite(first).all())
            self.assertLess(float(np.max(np.abs(first.mean(axis=0)))), 1e-6)
            self.assertLess(
                operation_error_numpy(
                    first, sample.operation_matrices, sample.permutation_index
                )["max_atom_error_angstrom"],
                1e-5,
            )

    def test_laplacian_has_one_zero_mode_and_commutes_with_every_action(self):
        for sample in self.samples.values():
            laplacian = graph_laplacian(sample)
            eigenvalues = np.linalg.eigvalsh(laplacian)
            self.assertEqual(int(np.count_nonzero(eigenvalues <= 1e-8)), 1)
            for permutation in sample.permutation_index:
                self.assertTrue(
                    np.allclose(
                        laplacian[np.ix_(permutation, permutation)],
                        laplacian,
                        atol=1e-12,
                        rtol=0.0,
                    )
                )

    def test_disconnected_graph_strictly_fails_on_extra_zero_mode(self):
        sample = self.samples["C2"]
        bonds = []
        for orbit in np.unique(sample.orbit_id):
            atoms = np.flatnonzero(sample.orbit_id == orbit)
            bonds.extend(
                (int(atoms[left]), int(atoms[right]))
                for left in range(len(atoms))
                for right in range(left + 1, len(atoms))
            )
        self.assertTrue(bonds)
        disconnected = replace(
            sample,
            bond_index=np.asarray(bonds, dtype=np.int64).T,
        )
        with self.assertRaisesRegex(ValueError, "one connected component"):
            make_symmetric_noise(
                disconnected,
                seed=7,
                coordinate_scale_angstrom=3.0,
                prior_type="graph_harmonic",
            )

    def test_sn_is_explicit_and_never_falls_back(self):
        sn_sample = None
        for index in range(len(self.dataset)):
            raw = self.dataset.canonical[int(self.dataset.local_indices[index])]
            if 50 in np.asarray(raw["atomic_numbers"], dtype=np.int64):
                sn_sample = self.dataset[index]
                break
        self.assertIsNotNone(sn_sample, "frozen C2/C3 train split must contain Sn")
        sn_mask = sn_sample.atomic_numbers == 50
        self.assertTrue(np.all(sn_sample.atom_types[sn_mask] == 12))
        values = make_symmetric_noise(
            sn_sample,
            seed=50,
            coordinate_scale_angstrom=3.0,
            prior_type="graph_harmonic",
        )
        self.assertEqual(values.shape[0], len(sn_sample.atomic_numbers))

    def test_c2_c3_centralizer_quadrature_counts_and_commutes(self):
        for sample, expected in (
            (self.samples["C2"], 48),
            (self.samples["C3"], 24),
        ):
            rotations = centralizer_rotations(
                sample.operation_matrices,
                target_pg=sample.target_pg,
                phase_count=24,
            )
            self.assertEqual(len(rotations), expected)
            for rotation in rotations:
                self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=5)
                for operation in sample.operation_matrices:
                    self.assertLess(
                        float(np.max(np.abs(rotation @ operation - operation @ rotation))),
                        2e-5,
                    )

    def test_c3_quadrature_is_closed_under_generator(self):
        matrices = _c3_matrices()
        rotations = centralizer_rotations(
            matrices, target_pg="C3", phase_count=24
        ).astype(np.float64)
        generator = matrices[1]
        for rotation in rotations:
            transformed = generator @ rotation
            self.assertLess(
                min(float(np.max(np.abs(transformed - candidate))) for candidate in rotations),
                2e-6,
            )

    def test_centralizer_posterior_is_finite_normalized_and_equivariant(self):
        for sample in self.samples.values():
            prior = make_symmetric_noise(
                sample,
                seed=91,
                coordinate_scale_angstrom=3.0,
                prior_type="graph_harmonic",
            )
            target = sample.symmetric_target_angstrom / 3.0
            for time_value in (0.02, 0.98):
                result = centralizer_averaged_transport(
                    sample,
                    prior=prior,
                    target=target,
                    time_value=time_value,
                    seed=92,
                )
                weights = result["posterior_weights"]
                self.assertTrue(np.isfinite(weights).all())
                self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
                self.assertTrue(np.isfinite(result["target_velocity"]).all())
                for key in (
                    "positions_t",
                    "transport_target_positions",
                    "geometry_target_positions",
                ):
                    self.assertLess(
                        operation_error_numpy(
                            result[key],
                            sample.operation_matrices,
                            sample.permutation_index,
                        )["max_atom_error_angstrom"],
                        1e-5,
                    )

    def test_endpoint_auxiliary_equals_scaled_velocity_error(self):
        import torch

        positions_t = torch.tensor(
            [[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=torch.float64
        )
        transport_target = torch.tensor(
            [[-1.0, 0.1, 0.0], [1.0, -0.1, 0.0]], dtype=torch.float64
        )
        time = torch.tensor(0.37, dtype=torch.float64)
        target_velocity = (transport_target - positions_t) / (1.0 - time)
        predicted_velocity = target_velocity + torch.tensor(
            [[0.2, -0.3, 0.1], [-0.1, 0.4, -0.2]], dtype=torch.float64
        )
        values = compute_loss(
            predicted_velocity=predicted_velocity,
            target_velocity=target_velocity,
            positions_t=positions_t,
            time=time,
            geometry_target_positions=transport_target,
            transport_target_positions=transport_target,
            atomic_numbers=torch.tensor([6, 6]),
            bond_index=torch.tensor([[0], [1]]),
            matrices=torch.eye(3, dtype=torch.float64)[None, :, :],
            permutations=torch.tensor([[0, 1]]),
            coordinate_scale_angstrom=3.0,
            bond_weight=0.0,
            overlap_weight=0.0,
            symmetry_weight=0.0,
            endpoint_weight=1.0,
        )
        expected = torch.square(1.0 - time) * values["flow_mse"]
        self.assertTrue(
            torch.allclose(values["endpoint_auxiliary_mse"], expected, atol=1e-14)
        )

    def test_same_seed_raw_generation_and_metrics_are_identical(self):
        import torch

        torch.manual_seed(20260832)
        model = PGOrbitFlow(hidden_dim=24, layers=2, radial_dim=6, radial_max=5.0)
        sample = self.samples["C2"]
        kwargs = {
            "seed": 909,
            "steps": 2,
            "device": "cpu",
            "coordinate_scale_angstrom": 3.0,
            "method": "heun",
            "prior_type": "graph_harmonic",
            "harmonic_alpha": 1.0,
        }
        first = sample_raw_ode(model, sample, **kwargs)
        second = sample_raw_ode(model, sample, **kwargs)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(
            raw_geometry_metrics(sample, first),
            raw_geometry_metrics(sample, second),
        )


if __name__ == "__main__":
    unittest.main()
