import unittest

import numpy as np

from generative_model.models.probability_audit import (
    align_target_nodes,
    categorical_item_records,
    pair_upper_triangle,
    summarize_categorical,
    target_rank_route_decision,
)


class TestProbabilityAudit(unittest.TestCase):
    def test_target_alignment_recovers_permuted_graph(self):
        target_atoms = np.asarray([1, 2, 1])
        target_charges = np.asarray([0, 0, 1])
        target_adjacency = np.asarray([[0, 1, 0], [1, 0, 2], [0, 2, 0]])
        permutation = np.asarray([2, 0, 1])  # predicted node -> target node
        atom_probs = np.full((3, 3), 0.01)
        charge_probs = np.full((3, 2), 0.01)
        bonds = np.full((3, 3, 3), 0.01)
        for predicted, target in enumerate(permutation):
            atom_probs[predicted, target_atoms[target]] = 0.98
            charge_probs[predicted, target_charges[target]] = 0.98
        for begin in range(3):
            for end in range(3):
                bonds[begin, end, target_adjacency[permutation[begin], permutation[end]]] = 0.98
        atom_probs /= atom_probs.sum(axis=1, keepdims=True)
        charge_probs /= charge_probs.sum(axis=1, keepdims=True)
        bonds /= bonds.sum(axis=2, keepdims=True)
        target_coords = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        predicted_coords = target_coords[permutation]
        assignment, audit = align_target_nodes(
            atom_probs, charge_probs, bonds, predicted_coords,
            target_atoms, target_charges, target_adjacency, target_coords,
            seed=7, random_starts=1,
        )
        self.assertEqual(assignment.tolist(), permutation.tolist())
        self.assertGreaterEqual(audit["joint_nll_improvement"], 0.0)

    def test_categorical_ranks_and_confusion(self):
        probabilities = np.asarray([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
        targets = np.asarray([2, 0])
        records = categorical_item_records(probabilities, targets)
        self.assertEqual(records["target_rank"].tolist(), [2, 1])
        summary = summarize_categorical(probabilities, targets)
        self.assertEqual(summary["top1_accuracy"], 0.5)
        self.assertEqual(summary["target_top2_rate"], 1.0)
        self.assertEqual(summary["confusion_target_rows_prediction_columns"][2][1], 1)

    def test_pair_upper_triangle_symmetrizes(self):
        probabilities = np.zeros((2, 2, 2), dtype=float)
        probabilities[0, 1] = [0.2, 0.8]
        probabilities[1, 0] = [0.4, 0.6]
        probabilities[0, 0] = probabilities[1, 1] = [1.0, 0.0]
        pairs, indices = pair_upper_triangle(probabilities)
        np.testing.assert_allclose(pairs, [[0.3, 0.7]])
        self.assertEqual(indices.tolist(), [[0, 1]])

    def test_route_decision(self):
        self.assertEqual(
            target_rank_route_decision(
                np.asarray([1, 2, 2, 1, 3]),
                wrong_present_target_probabilities=np.asarray([0.7, 0.2, 0.3, 0.8, 0.1]),
                wrong_present_margins=np.asarray([0.1, 0.2, 0.3, 0.1, 0.7]),
            )["recommended_route"],
            "structured_decoder_first",
        )
        self.assertEqual(
            target_rank_route_decision(
                np.asarray([1, 2, 2, 1, 2]),
                wrong_present_target_probabilities=np.asarray([1e-6] * 5),
                wrong_present_margins=np.asarray([0.999] * 5),
            )["recommended_route"],
            "training_objective_first",
        )
        self.assertEqual(
            target_rank_route_decision(np.asarray([1]))["recommended_route"],
            "insufficient_evidence",
        )


if __name__ == "__main__":
    unittest.main()
