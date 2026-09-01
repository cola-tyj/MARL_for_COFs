"""Numerical helper tests for the Cn layered validity audit."""

from __future__ import annotations

import unittest

import numpy as np

from generative_model.evaluation.audit_our_etflow_cn_validity import _kabsch_rmsd


class TestOurETFlowCnValidity(unittest.TestCase):
    def test_kabsch_rmsd_is_rigid_transform_invariant(self) -> None:
        reference = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0], [0.0, 0.0, 3.0],
        ])
        angle = 0.73
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        moving = reference @ rotation.T + np.asarray([4.0, -2.0, 7.0])
        self.assertLess(_kabsch_rmsd(reference, moving), 1e-12)

    def test_kabsch_rmsd_detects_nonrigid_change(self) -> None:
        reference = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
        ])
        moving = reference.copy(); moving[2, 1] = 2.0
        self.assertGreater(_kabsch_rmsd(reference, moving), 0.1)


if __name__ == "__main__":
    unittest.main()

