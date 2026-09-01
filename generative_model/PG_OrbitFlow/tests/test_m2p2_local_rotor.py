import json
import unittest
from pathlib import Path

import numpy as np
import torch

from generative_model.PG_OrbitFlow.geometry import build_geometry_contract
from generative_model.PG_OrbitFlow.local_rotor import (
    build_local_rotor_set_contract,
    coupled_permutation_invariant_circular_loss,
    permutation_invariant_circular_errors_degrees,
    permutation_invariant_circular_loss,
)
from generative_model.PG_OrbitFlow.local_atom_matching import (
    best_terminal_sibling_relabeling,
    build_terminal_sibling_contract,
)
from generative_model.PG_OrbitFlow.m2p2_model import (
    SetValuedQuotientICPredictor,
    freeze_m2p1_parent,
    load_m2p1_parent,
)
from generative_model.PG_OrbitFlow.orbit_ic_model import build_orbit_ic_example
from generative_model.PG_OrbitFlow.overfit import _panel_samples


ROOT = Path(__file__).resolve().parents[1]


class TestM2P2LocalRotorSets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "configs/m2p1_phase_tier16_v1.json").read_text(encoding="utf-8")
        )
        tier = {
            "records": cls.protocol["panel"]["records"],
            "evaluation": {},
            "gate": {},
        }
        cls.samples = _panel_samples(cls.protocol, tier)

    def test_contract_is_graph_only_disjoint_and_covers_terminal_rotors(self):
        groups = {}
        for sample in self.samples:
            local = build_local_rotor_set_contract(
                sample, build_geometry_contract(sample)
            )
            flat = [orbit for group in local.groups for orbit in group]
            self.assertEqual(len(flat), len(set(flat)))
            self.assertTrue(all(len(group) in {2, 3} for group in local.groups))
            groups[sample.package_index] = local.groups
        self.assertIn((4, 5, 6), groups[875])
        self.assertIn((6, 7, 8), groups[2403])
        sample = next(row for row in self.samples if row.package_index == 875)
        local = build_local_rotor_set_contract(sample, build_geometry_contract(sample))
        self.assertEqual(local.coupled_group_components, ((0, 1),))
        self.assertEqual(local.coupled_component_slot_maps, (((0, 1, 2), (0, 1, 2)),))

    def test_operation_linked_terminal_groups_share_a_conjugated_component(self):
        sample = next(row for row in self.samples if row.package_index == 1075)
        local = build_local_rotor_set_contract(sample, build_geometry_contract(sample))
        self.assertEqual(local.terminal_atom_groups, ((12, 13), (14, 15)))
        self.assertEqual(local.coupled_group_components, ((0, 1),))
        self.assertEqual(local.coupled_component_slot_maps, (((0, 1), (0, 1)),))
        sample = next(row for row in self.samples if row.package_index == 2403)
        local = build_local_rotor_set_contract(sample, build_geometry_contract(sample))
        self.assertEqual(local.coupled_group_components, ((0, 1),))
        self.assertEqual(local.coupled_component_slot_maps, (((0, 1, 2), (0, 2, 1)),))

    def test_circular_set_loss_is_permutation_invariant_and_finite(self):
        angles = torch.deg2rad(torch.tensor([0.0, 120.0, -120.0]))
        target = torch.stack((torch.sin(angles), torch.cos(angles)), dim=1)
        prediction = target[[1, 2, 0]].clone().requires_grad_(True)
        label_loss = (1.0 - (prediction * target).sum(dim=-1)).mean()
        set_loss = permutation_invariant_circular_loss(
            prediction, target, ((0, 1, 2),)
        )
        self.assertGreater(float(label_loss), 1.0)
        self.assertLess(float(set_loss), 1e-12)
        set_loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_numpy_metric_uses_the_same_assignment(self):
        angles = np.deg2rad(np.asarray([10.0, 130.0, -110.0]))
        target = np.stack((np.sin(angles), np.cos(angles)), axis=1)
        prediction = target[[2, 0, 1]]
        error = permutation_invariant_circular_errors_degrees(
            prediction, target, ((0, 1, 2),)
        )
        np.testing.assert_allclose(error, 0.0, atol=1e-10)

    def test_set_head_starts_at_parent_and_only_it_is_trainable(self):
        settings = {
            key: self.protocol["model"][key]
            for key in (
                "node_feature_dim",
                "edge_feature_dim",
                "hidden_dim",
                "layers",
                "predict_torsion",
                "geometry_group_phase",
            )
        }
        parent = torch.load(
            ROOT / "runs/m2p1_phase_tier16_v1/last.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = SetValuedQuotientICPredictor(**settings)
        load_m2p1_parent(model, parent["model"])
        isolation = freeze_m2p1_parent(model)
        self.assertGreater(isolation["trainable_parameter_count"], 0)
        self.assertTrue(
            all(name.startswith("local_rotor_head.") for name in isolation["trainable_keys"])
        )
        sample = next(row for row in self.samples if row.package_index == 875)
        contract = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, contract)
        local = build_local_rotor_set_contract(sample, contract)
        model.eval()
        prediction = model(graph, local, device="cpu")
        self.assertEqual(prediction["torsion_sincos"].shape, targets.torsion_target_sincos.shape)
        loss = permutation_invariant_circular_loss(
            prediction["torsion_sincos"],
            torch.as_tensor(targets.torsion_target_sincos),
            local.groups,
        )
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.local_rotor_head.parameters()
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.base.parameters()))

    def test_empty_set_batch_has_a_finite_zero_gradient_anchor(self):
        layer = torch.nn.Linear(4, 2)
        loss = sum(parameter.sum() * 0.0 for parameter in layer.parameters())
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(
            all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in layer.parameters())
        )

    def test_terminal_sibling_matching_is_graph_constrained_and_recovers_swap(self):
        sample = next(row for row in self.samples if row.package_index == 875)
        sibling = build_terminal_sibling_contract(sample)
        self.assertTrue(any(len(group) == 3 for group in sibling.groups))
        group = next(group for group in sibling.groups if len(group) == 3)
        candidate = np.asarray(sample.symmetric_target_angstrom).copy()
        candidate[list(group)] = candidate[list((group[1], group[2], group[0]))]
        matched = best_terminal_sibling_relabeling(sample, candidate)
        self.assertLess(matched["kabsch_rmsd_angstrom"], 1e-10)
        self.assertTrue(matched["nonidentity_assignment"])

    def test_coupled_loss_requires_one_permutation_across_torsion_views(self):
        from types import SimpleNamespace
        angles = torch.deg2rad(torch.tensor([0.0, 120.0, -120.0]))
        target_one = torch.stack((torch.sin(angles), torch.cos(angles)), dim=1)
        target = torch.cat((target_one, target_one))
        prediction = torch.cat((target_one[[1, 2, 0]], target_one[[1, 2, 0]])).requires_grad_(True)
        contract = SimpleNamespace(groups=((0, 1, 2), (3, 4, 5)), coupled_group_components=((0, 1),), coupled_component_slot_maps=(((0, 1, 2), (0, 1, 2)),))
        loss = coupled_permutation_invariant_circular_loss(prediction, target, contract)
        self.assertLess(float(loss), 1e-12)
        inconsistent = torch.cat((target_one[[1, 2, 0]], target_one[[2, 0, 1]])).requires_grad_(True)
        inconsistent_loss = coupled_permutation_invariant_circular_loss(inconsistent, target, contract)
        self.assertGreater(float(inconsistent_loss), 0.5)


if __name__ == "__main__":
    unittest.main()
