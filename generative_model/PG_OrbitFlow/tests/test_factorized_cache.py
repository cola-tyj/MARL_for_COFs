import tempfile
import unittest
from pathlib import Path

import numpy as np

from generative_model.PG_OrbitFlow.factorized_automorphism import (
    FactorizedAutomorphismContract,
)
from generative_model.PG_OrbitFlow.factorized_cache import (
    load_contract,
    save_contract,
    selection_fingerprint,
    sha256,
)


class TestFactorizedCache(unittest.TestCase):
    def test_pickle_free_roundtrip_is_lossless(self):
        contract = FactorizedAutomorphismContract(
            groups=((0, 2),),
            representative_tuples=np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]]),
            allowed_torsion_permutations=np.asarray([[0, 1], [1, 0]]),
            induced_generator_count=2,
            allowed_torsion_permutation_count=2,
            branch_swap_witness_count=3,
            anchored_witness_count=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.npz"
            repeated_path = Path(directory) / "contract-repeated.npz"
            save_contract(path, contract)
            save_contract(repeated_path, contract)
            self.assertEqual(sha256(path), sha256(repeated_path))
            loaded = load_contract(path)
        self.assertEqual(loaded.groups, contract.groups)
        np.testing.assert_array_equal(
            loaded.representative_tuples, contract.representative_tuples
        )
        np.testing.assert_array_equal(
            loaded.allowed_torsion_permutations,
            contract.allowed_torsion_permutations,
        )
        self.assertEqual(loaded.anchored_witness_count, 4)

    def test_selection_fingerprint_is_order_sensitive_and_deterministic(self):
        records = [
            {"package_index": 1, "molecule_id": "a", "target_pg": "C2", "num_atoms": 7},
            {"package_index": 2, "molecule_id": "b", "target_pg": "C3", "num_atoms": 9},
        ]
        self.assertEqual(selection_fingerprint(records), selection_fingerprint(records))
        self.assertNotEqual(selection_fingerprint(records), selection_fingerprint(records[::-1]))


if __name__ == "__main__":
    unittest.main()
