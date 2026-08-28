from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from generative_model.data import (
    GRAPH_PG_3D_SCHEMA_VERSION,
    TARGET_POINT_GROUPS,
    GraphPG3DDataset,
    collate_graph_pg_3d,
)


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "generative_model/data/processed/v2"


class TestGraphPG3DDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = GraphPG3DDataset(PACKAGE)

    def test_sample_preserves_known_graph_and_target_condition(self) -> None:
        sample = self.dataset[0]
        self.assertEqual(sample["schema_version"], GRAPH_PG_3D_SCHEMA_VERSION)
        self.assertIn(sample["target_pg"], TARGET_POINT_GROUPS)
        self.assertEqual(sample["target_positions"].shape, (len(sample["atom_types"]), 3))
        self.assertLessEqual(float(np.abs(sample["target_positions"].mean(0)).max()), 1e-4)
        self.assertEqual(sample["bond_index"].shape[0], 2)
        self.assertEqual(sample["bond_index"].shape[1], len(sample["bond_types"]))
        self.assertEqual(sample["target_pg_one_hot"].sum(), 1.0)

    def test_four_point_group_batch_offsets(self) -> None:
        selected = {}
        for index in range(len(self.dataset)):
            sample = self.dataset[index]
            selected.setdefault(sample["target_pg"], sample)
            if len(selected) == len(TARGET_POINT_GROUPS):
                break
        batch = collate_graph_pg_3d([selected[name] for name in TARGET_POINT_GROUPS])
        self.assertEqual(batch["target_pg_indices"].tolist(), [0, 1, 2, 3])
        self.assertEqual(int(batch["atom_offsets"][-1]), len(batch["target_positions"]))
        self.assertEqual(
            int(batch["operation_offsets"][-1]), len(batch["target_operation_matrices"])
        )
        self.assertEqual(
            int(batch["permutation_offsets"][-1]), len(batch["target_permutation_index"])
        )


if __name__ == "__main__":
    unittest.main()
