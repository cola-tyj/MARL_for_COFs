from __future__ import annotations

import unittest

import numpy as np

from generative_model.PG_OrbitFlow.group import (
    group_features,
    operation_error_numpy,
    project_vectors_numpy,
    validate_group_action,
)
from generative_model.PG_OrbitFlow.core_arm import validate_core_arm_decomposition
from generative_model.PG_OrbitFlow.model import PGOrbitFlow
from generative_model.PG_OrbitFlow.flow import align_cyclic_target_phase
from generative_model.PG_OrbitFlow.quotient import build_quotient_graph


def c2_action():
    matrices = np.asarray(
        [np.eye(3), np.diag([-1.0, -1.0, 1.0])], dtype=np.float32
    )
    permutations = np.asarray([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int64)
    orbit_id = np.asarray([0, 0, 1, 1], dtype=np.int64)
    return matrices, permutations, orbit_id


class TestGroupAndQuotient(unittest.TestCase):
    def test_c2_action_closure_projection_and_quotient(self):
        matrices, permutations, orbit_id = c2_action()
        audit = validate_group_action(
            matrices,
            permutations,
            atomic_numbers=np.asarray([6, 6, 7, 7]),
            orbit_id=orbit_id,
        )
        self.assertEqual(audit.order, 2)
        values = np.asarray(
            [[1.1, 0.2, 0.3], [-1.0, -0.4, 0.2], [0.3, 1.2, -0.1], [-0.2, -1.1, 0.1]]
        )
        projected = project_vectors_numpy(values, matrices, permutations)
        self.assertLess(
            operation_error_numpy(projected, matrices, permutations)["max_atom_error_angstrom"],
            1e-10,
        )
        quotient = build_quotient_graph(
            atom_count=4,
            bond_index=np.asarray([[0, 1, 0, 2], [2, 3, 1, 3]], dtype=np.int64),
            bond_types=np.asarray([1, 1, 1, 1], dtype=np.int64),
            orbit_id=orbit_id,
            operation_matrices=matrices,
            permutation_index=permutations,
        )
        self.assertEqual(quotient.orbit_count, 2)

    def test_invalid_cross_element_permutation_fails(self):
        matrices, permutations, orbit_id = c2_action()
        with self.assertRaisesRegex(ValueError, "across elements"):
            validate_group_action(
                matrices,
                permutations,
                atomic_numbers=np.asarray([6, 7, 7, 7]),
                orbit_id=orbit_id,
            )

    def test_explicit_core_arm_contract(self):
        matrices, permutations, _ = c2_action()
        decomposition = validate_core_arm_decomposition(
            atomic_numbers=np.asarray([6, 6, 7, 7]),
            bond_index=np.asarray([[0, 1, 0], [1, 3, 2]], dtype=np.int64),
            operation_matrices=matrices,
            permutation_index=permutations,
            atom_role=np.asarray([0, 0, 1, 1], dtype=np.int64),
            arm_copy_id=np.asarray([-1, -1, 0, 1], dtype=np.int64),
        )
        self.assertEqual(len(decomposition.arm_atom_indices), 2)
        self.assertTrue(np.array_equal(decomposition.copy_permutation[1], [1, 0]))

    def test_cyclic_phase_alignment_preserves_action_and_improves_pairing(self):
        matrices, permutations, _ = c2_action()
        target = np.asarray(
            [[1.0, 0.0, 0.2], [-1.0, 0.0, 0.2], [0.0, 1.5, -0.2], [0.0, -1.5, -0.2]],
            dtype=np.float64,
        )
        angle = 0.73
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        prior = target @ rotation.T
        aligned, _ = align_cyclic_target_phase(target, prior, matrices)
        before = np.sqrt(np.mean(np.sum(np.square(target - prior), axis=1)))
        after = np.sqrt(np.mean(np.sum(np.square(aligned - prior), axis=1)))
        self.assertLess(after, before * 1e-6)
        self.assertLess(
            operation_error_numpy(aligned, matrices, permutations)["max_atom_error_angstrom"],
            1e-8,
        )


class TestPGOrbitFlowModel(unittest.TestCase):
    def test_vector_field_preserves_action_and_has_finite_backward(self):
        import torch

        torch.manual_seed(7)
        matrices_np, permutations_np, orbit_id = c2_action()
        global_features, atom_features = group_features(
            matrices_np, permutations_np, orbit_id
        )
        model = PGOrbitFlow(hidden_dim=24, layers=2, radial_dim=6, radial_max=5.0)
        for block in model.blocks:
            torch.nn.init.normal_(block.coordinate_weight.net[-1].weight, std=0.02)
        positions = torch.tensor(
            [[1.0, 0.2, 0.0], [-1.0, -0.2, 0.0], [0.3, 1.0, 0.1], [-0.3, -1.0, 0.1]],
            dtype=torch.float32,
            requires_grad=True,
        )
        bond_order = torch.tensor(
            [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=torch.float32,
        )
        kwargs = {
            "positions": positions,
            "time": torch.tensor(0.4),
            "atom_types": torch.tensor([1, 1, 2, 2]),
            "formal_charges": torch.zeros(4, dtype=torch.long),
            "radical_electrons": torch.zeros(4, dtype=torch.long),
            "target_pg_index": 0,
            "group_features": torch.tensor(global_features),
            "atom_group_features": torch.tensor(atom_features),
            "invariant_bond_order": bond_order,
        }
        velocity = model(**kwargs)
        rotation = torch.tensor(matrices_np[1])
        permutation = torch.tensor(permutations_np[1])
        self.assertTrue(
            torch.allclose(velocity @ rotation.T, velocity[permutation], atol=2e-6, rtol=2e-6)
        )
        loss = torch.square(velocity - positions.detach()).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(positions.grad).all())
        self.assertTrue(
            all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        )


if __name__ == "__main__":
    unittest.main()
