"""Tests for all-target E3/F0.2 v2 inference inputs."""

from __future__ import annotations

import unittest

import numpy as np

from generative_model.data.graph_only_dataset import CanonicalGraphOnlyDataset
from generative_model.inference.generate_etflow_symmetric_xyz_v2 import (
    sample_from_canonical_v2,
)
from generative_model.models.graph_pg_3d_projection import (
    operation_errors,
    project_reynolds,
)


PACKAGE = "generative_model/data/processed/v2"


class TestETFlowE3F02InferenceV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = CanonicalGraphOnlyDataset(PACKAGE)

    def test_s4_and_d6h_samples_are_graph_only_and_complete(self) -> None:
        for package_index, target_pg, operation_count in (
            (48, "S4", 4),
            (858, "D6h", 24),
        ):
            canonical = self.dataset[package_index]
            sample = sample_from_canonical_v2(canonical, target_pg)
            self.assertNotIn("positions", canonical)
            self.assertNotIn("target_positions", sample)
            self.assertNotIn("reference_positions", sample)
            self.assertEqual(
                sample["target_operation_matrices"].shape,
                (operation_count, 3, 3),
            )
            self.assertEqual(
                sample["target_permutation_index"].shape,
                (operation_count, len(sample["atomic_numbers"])),
            )
            self.assertFalse(
                sample["graph_action_audit"]["recovery"]["coordinates_read"]
            )
            self.assertFalse(
                sample["graph_action_audit"]["recovery"][
                    "stored_symmetry_action_read"
                ]
            )

    def test_s4_and_d6h_reynolds_projection_is_exact(self) -> None:
        for package_index, target_pg in ((48, "S4"), (858, "D6h")):
            sample = sample_from_canonical_v2(self.dataset[package_index], target_pg)
            atom_count = len(sample["atomic_numbers"])
            values = np.sin(
                np.arange(atom_count * 3, dtype=np.float64).reshape(atom_count, 3)
                * 0.17
            )
            projected = project_reynolds(
                values,
                sample["target_operation_matrices"],
                sample["target_permutation_index"],
            )
            error = operation_errors(
                projected,
                sample["target_operation_matrices"],
                sample["target_permutation_index"],
            )
            self.assertLessEqual(error["max_atom_error_angstrom"], 1e-10)

    def test_c2_c3_delegate_without_changing_contract(self) -> None:
        for package_index, target_pg, count in ((45, "C2", 2), (81, "C3", 3)):
            sample = sample_from_canonical_v2(self.dataset[package_index], target_pg)
            self.assertEqual(len(sample["target_operation_matrices"]), count)
            self.assertNotIn("target_positions", sample)


if __name__ == "__main__":
    unittest.main()
