"""Tests for the reference-free S4 inertia selection guard."""

from __future__ import annotations

import unittest

import numpy as np

from generative_model.inference.s4_inertia_selection import (
    mass_weighted_inertia_separation, select_s4_inertia_stable_candidate,
)


class TestS4InertiaSelection(unittest.TestCase):
    def test_separation_is_rigid_and_scale_invariant(self) -> None:
        numbers = np.asarray([6, 6, 7, 7])
        positions = np.asarray([
            [2.0, 0.0, 0.4], [-2.0, 0.0, 0.4],
            [0.0, 1.0, -0.4], [0.0, -1.0, -0.4],
        ])
        angle = 0.73
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0],
        ])
        baseline = mass_weighted_inertia_separation(numbers, positions)
        transformed = positions @ rotation.T * 3.7 + np.asarray([4.0, -2.0, 8.0])
        self.assertAlmostEqual(
            baseline, mass_weighted_inertia_separation(numbers, transformed), places=12
        )

    def test_lower_objective_ambiguous_candidate_is_not_selected(self) -> None:
        numbers = np.asarray([6, 6, 6, 6])
        spherical = np.asarray([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ], dtype=float)
        stable = np.asarray([
            [2, 0, 0.3], [-2, 0, 0.3], [0, 1, -0.3], [0, -1, -0.3]
        ], dtype=float)
        records = [
            {"final_total_objective_kcal_mol": 1.0, "candidate_id": 0},
            {"final_total_objective_kcal_mol": 2.0, "candidate_id": 1},
        ]
        selected, separations = select_s4_inertia_stable_candidate(
            records, [{"final": spherical}, {"final": stable}], numbers,
            minimum_separation=0.012,
        )
        self.assertEqual(selected, 1)
        self.assertLess(separations[0], 0.012)
        self.assertGreater(separations[1], 0.012)


if __name__ == "__main__":
    unittest.main()
