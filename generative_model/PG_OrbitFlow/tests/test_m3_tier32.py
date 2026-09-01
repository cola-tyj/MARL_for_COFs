import unittest
from types import SimpleNamespace

import numpy as np

from generative_model.PG_OrbitFlow.m3_tier32_training import _balanced_schedule


class TestM3Tier32(unittest.TestCase):
    def test_schedule_is_deterministic_pg_balanced_and_equal_exposure(self):
        examples = [
            (SimpleNamespace(target_pg=point_group),)
            for point_group in ("C2", "C3")
            for _ in range(16)
        ]
        first = _balanced_schedule(examples, steps=4096, seed=20261845)
        second = _balanced_schedule(examples, steps=4096, seed=20261845)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4096)
        for batch in first:
            self.assertEqual(len(batch), 4)
            self.assertEqual(sum(examples[index][0].target_pg == "C2" for index in batch), 2)
            self.assertEqual(sum(examples[index][0].target_pg == "C3" for index in batch), 2)
        exposures = np.bincount(np.asarray(first).reshape(-1), minlength=32)
        np.testing.assert_array_equal(exposures, np.full(32, 512))


if __name__ == "__main__":
    unittest.main()
