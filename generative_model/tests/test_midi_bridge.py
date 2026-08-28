from pathlib import Path
import unittest

import numpy as np

from generative_model.data import COFSymmetryDataset
from generative_model.models.midi_bridge import (
    canonical_to_midi_data,
    midi_data_to_canonical,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "data/processed/v2"


class TestMiDiBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch_geometric  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest("torch_geometric not installed") from error
        cls.dataset = COFSymmetryDataset(PACKAGE_DIR)

    def test_pyg_roundtrip_is_lossless(self):
        sample = self.dataset[0]
        data = canonical_to_midi_data(sample)
        restored = midi_data_to_canonical(data)
        for key in (
            "positions", "atom_types", "atomic_numbers", "formal_charges",
            "bond_index", "bond_types",
        ):
            np.testing.assert_array_equal(restored[key], sample[key])
        self.assertEqual(data.edge_index.shape[1], 2 * len(sample["bond_types"]))

    def test_sn_and_radical_samples_fail_strictly(self):
        with self.assertRaisesRegex(ValueError, "Sn"):
            canonical_to_midi_data(self.dataset[1684])
        with self.assertRaisesRegex(ValueError, "radical-electron"):
            canonical_to_midi_data(self.dataset[1374])


if __name__ == "__main__":
    unittest.main()
