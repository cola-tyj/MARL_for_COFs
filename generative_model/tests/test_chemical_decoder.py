"""化学约束 decoder 的确定性与严格失败回归测试。"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from generative_model.models.chemical_decoder import valence_greedy_decode, valence_prune_decode


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch not available in this environment")
class TestChemicalDecoder(unittest.TestCase):
    @staticmethod
    def _output(atom_indices: list[int], pair_probabilities: dict[tuple[int, int], list[float]]):
        import torch

        atom_count = len(atom_indices)
        atomics = torch.zeros((1, atom_count, 4), dtype=torch.float32)
        for index, atom_type in enumerate(atom_indices):
            atomics[0, index, atom_type] = 1.0
        bonds = torch.zeros((1, atom_count, atom_count, 5), dtype=torch.float32)
        bonds[..., 0] = 1.0
        for (begin, end), probabilities in pair_probabilities.items():
            bonds[0, begin, end] = torch.tensor(probabilities)
            bonds[0, end, begin] = torch.tensor(probabilities)
        charges = torch.ones((1, atom_count, 1), dtype=torch.float32)
        return {
            "coords": torch.zeros((1, atom_count, 3), dtype=torch.float32),
            "atomics": atomics,
            "bonds": bonds,
            "charges": charges,
            "mask": torch.ones((1, atom_count), dtype=torch.bool),
        }

    def test_overvalent_carbon_loses_low_priority_fifth_hydrogen(self) -> None:
        probabilities = {
            (0, hydrogen): [0.1, 0.9 - hydrogen * 0.01, 0.0, 0.0, 0.0]
            for hydrogen in range(1, 6)
        }
        output = self._output([3, 2, 2, 2, 2, 2], probabilities)
        decoded, audit = valence_greedy_decode(
            output,
            symbols=("<PAD>", "<MASK>", "H", "C"),
            charge_values=(0,),
        )
        bond_types = decoded["bonds"].argmax(dim=-1)[0].numpy()
        self.assertEqual(int(np.count_nonzero(np.triu(bond_types, k=1))), 4)
        np.testing.assert_array_equal(bond_types, bond_types.T)
        np.testing.assert_array_equal(np.diag(bond_types), np.zeros(6, dtype=np.int64))
        self.assertGreaterEqual(audit.rejected_overvalence, 1)
        self.assertEqual(audit.atom_types_changed, 0)
        self.assertEqual(audit.formal_charges_changed, 0)

    def test_aromatic_bridge_is_explicitly_replaced_by_single_bond(self) -> None:
        output = self._output([3, 3], {(0, 1): [0.05, 0.40, 0.05, 0.05, 0.45]})
        decoded, audit = valence_greedy_decode(
            output,
            symbols=("<PAD>", "<MASK>", "H", "C"),
            charge_values=(0,),
        )
        bond_types = decoded["bonds"].argmax(dim=-1)[0]
        self.assertEqual(int(bond_types[0, 1]), 1)
        self.assertEqual(audit.aromatic_bridge_edges_replaced, 1)
        self.assertEqual(audit.aromatic_bridge_edges_removed, 0)

    def test_aromatic_edge_in_mixed_bond_cycle_is_preserved(self) -> None:
        output = self._output(
            [3, 3, 3],
            {
                (0, 1): [0.05, 0.40, 0.05, 0.05, 0.45],
                (1, 2): [0.05, 0.90, 0.02, 0.02, 0.01],
                (0, 2): [0.05, 0.90, 0.02, 0.02, 0.01],
            },
        )
        decoded, audit = valence_greedy_decode(
            output,
            symbols=("<PAD>", "<MASK>", "H", "C"),
            charge_values=(0,),
        )
        bond_types = decoded["bonds"].argmax(dim=-1)[0]
        self.assertEqual(int(bond_types[0, 1]), 4)
        self.assertEqual(audit.aromatic_bridge_edges_replaced, 0)
        self.assertEqual(audit.aromatic_bridge_edges_removed, 0)

    def test_special_atom_token_fails_instead_of_falling_back(self) -> None:
        output = self._output([0], {})
        with self.assertRaisesRegex(ValueError, "特殊原子 token"):
            valence_greedy_decode(
                output,
                symbols=("<PAD>", "<MASK>", "H", "C"),
                charge_values=(0,),
            )

    def test_prune_decoder_minimally_removes_lowest_confidence_excess_bond(self) -> None:
        probabilities = {
            (0, hydrogen): [0.1, 0.9 - hydrogen * 0.01, 0.0, 0.0, 0.0]
            for hydrogen in range(1, 6)
        }
        output = self._output([3, 2, 2, 2, 2, 2], probabilities)
        decoded, audit = valence_prune_decode(
            output,
            symbols=("<PAD>", "<MASK>", "H", "C"),
            charge_values=(0,),
        )
        bond_types = decoded["bonds"].argmax(dim=-1)[0]
        self.assertEqual(int(np.count_nonzero(np.triu(bond_types.numpy(), k=1))), 4)
        self.assertEqual(int(bond_types[0, 5]), 0)
        self.assertEqual(audit.initial_overvalent_atoms, 1)
        self.assertEqual(audit.removed_bonds, 1)
        self.assertEqual(audit.bridge_removals, 1)

    def test_prune_decoder_downgrades_aromatic_bridge_without_disconnect(self) -> None:
        output = self._output([3, 3], {(0, 1): [0.05, 0.40, 0.05, 0.05, 0.45]})
        decoded, audit = valence_prune_decode(
            output,
            symbols=("<PAD>", "<MASK>", "H", "C"),
            charge_values=(0,),
        )
        self.assertEqual(int(decoded["bonds"].argmax(dim=-1)[0, 0, 1]), 1)
        self.assertEqual(audit.aromatic_bridge_edges_replaced, 1)
        self.assertEqual(audit.removed_bonds, 0)


if __name__ == "__main__":
    unittest.main()
