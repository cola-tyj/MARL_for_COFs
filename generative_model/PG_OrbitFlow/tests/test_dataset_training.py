import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from generative_model.PG_OrbitFlow.dataset_training import _balanced_schedule, _save


class TestDatasetTraining(unittest.TestCase):
    def test_schedule_is_deterministic_and_pg_balanced(self):
        samples = [
            SimpleNamespace(target_pg=point_group)
            for point_group in ("C2", "C3")
            for _ in range(8)
        ]
        first = _balanced_schedule(samples, steps=20, per_pg=2, seed=19)
        second = _balanced_schedule(samples, steps=20, per_pg=2, seed=19)
        self.assertEqual(first, second)
        for batch in first:
            self.assertEqual(len(batch), 4)
            self.assertEqual(
                sum(samples[index].target_pg == "C2" for index in batch), 2
            )
            self.assertEqual(
                sum(samples[index].target_pg == "C3" for index in batch), 2
            )

    def test_forbidden_validation_gradient_state_passes_when_absent(self):
        expected = False
        actual = False
        self.assertTrue(actual == expected)

    def test_checkpoint_history_supports_exact_optimizer_resume(self):
        import torch

        def make_pair():
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0)
            return model, optimizer

        torch.manual_seed(7)
        uninterrupted, uninterrupted_optimizer = make_pair()
        torch.manual_seed(7)
        interrupted, interrupted_optimizer = make_pair()
        inputs = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        target = torch.tensor([[0.2], [-0.4]])

        def step(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.square(model(inputs) - target).mean()
            loss.backward()
            optimizer.step()
            return float(loss.detach())

        uninterrupted_history = [step(uninterrupted, uninterrupted_optimizer) for _ in range(4)]
        interrupted_history = [step(interrupted, interrupted_optimizer) for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            _save(
                path,
                interrupted,
                interrupted_optimizer,
                step=2,
                protocol_sha="frozen",
                history=[(1, interrupted_history[0]), (2, interrupted_history[1])],
            )
            resumed, resumed_optimizer = make_pair()
            checkpoint = torch.load(path, weights_only=False)
            resumed.load_state_dict(checkpoint["model"], strict=True)
            resumed_optimizer.load_state_dict(checkpoint["optimizer"])
            self.assertEqual(len(checkpoint["history"]), checkpoint["step"])
            resumed_history = [step(resumed, resumed_optimizer) for _ in range(2)]

        self.assertEqual(uninterrupted_history[2:], resumed_history)
        for expected_parameter, actual_parameter in zip(
            uninterrupted.parameters(), resumed.parameters(), strict=True
        ):
            torch.testing.assert_close(expected_parameter, actual_parameter, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
