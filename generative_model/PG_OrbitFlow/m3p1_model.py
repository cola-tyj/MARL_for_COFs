"""Generalized 2--6 slot automorphism-set torsion head."""

from __future__ import annotations

from .m2p2_model import SetValuedQuotientICPredictor


class AutomorphismSetQuotientICPredictor:
    def __new__(cls, **model_settings):
        import torch

        parent = SetValuedQuotientICPredictor(**model_settings)
        hidden_dim = int(model_settings.get("hidden_dim", 96))
        torsion_input_dim = int(parent.base.torsion_head[0].in_features)
        maximum_slots = 6

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parent = parent
                self.maximum_slots = maximum_slots
                self.torsion_input_dim = torsion_input_dim
                self.automorphism_head = torch.nn.Sequential(
                    torch.nn.Linear(torsion_input_dim + maximum_slots + (maximum_slots - 1), hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, 2),
                )

            def initialize_from_m3_parent(self, parent_state: dict):
                target = self.state_dict()
                inherited = []
                for key, value in parent_state.items():
                    destination = f"parent.{key}"
                    if destination not in target or target[destination].shape != value.shape:
                        raise ValueError(f"incompatible M3 parent tensor: {key}")
                    target[destination] = value.detach().clone(); inherited.append(destination)
                old_first = parent_state["local_rotor_head.0.weight"]
                new_first = target["automorphism_head.0.weight"]
                new_first.zero_()
                new_first[:, :torsion_input_dim].copy_(old_first[:, :torsion_input_dim])
                new_first[:, torsion_input_dim : torsion_input_dim + 3].copy_(
                    old_first[:, torsion_input_dim : torsion_input_dim + 3]
                )
                new_size_start = torsion_input_dim + maximum_slots
                new_first[:, new_size_start : new_size_start + 2].copy_(
                    old_first[:, torsion_input_dim + 3 : torsion_input_dim + 5]
                )
                target["automorphism_head.0.weight"] = new_first
                target["automorphism_head.0.bias"] = parent_state["local_rotor_head.0.bias"].detach().clone()
                target["automorphism_head.2.weight"] = parent_state["local_rotor_head.2.weight"].detach().clone()
                target["automorphism_head.2.bias"] = parent_state["local_rotor_head.2.bias"].detach().clone()
                self.load_state_dict(target, strict=True)
                return {
                    "inherited_tensor_count": len(inherited),
                    "inherited_parameter_count": int(sum(target[key].numel() for key in inherited)),
                    "inherited_keys": sorted(inherited),
                    "automorphism_head_compatible_initialization": True,
                }

            def configure_trainable_parameters(self):
                for parameter in self.parameters():
                    parameter.requires_grad_(True)
                # The legacy terminal-only head is retained for exact parent
                # identity but is superseded by the generalized head.
                for parameter in self.parent.local_rotor_head.parameters():
                    parameter.requires_grad_(False)
                trainable = [name for name, parameter in self.named_parameters() if parameter.requires_grad]
                if any(name.startswith("parent.local_rotor_head.") for name in trainable):
                    raise RuntimeError("superseded local rotor head remained trainable")
                return {
                    "trainable_keys": trainable,
                    "trainable_parameter_count": int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)),
                    "frozen_legacy_local_rotor_parameter_count": int(sum(parameter.numel() for parameter in self.parent.local_rotor_head.parameters())),
                }

            def forward(self, graph, contract, *, device: str):
                self.parent._captured_torsion_input = None
                result = self.parent.base(graph, device=device)
                torsion_input = self.parent._captured_torsion_input
                if torsion_input is None:
                    raise RuntimeError("failed to capture automorphism torsion input")
                rows = [row for row in result["torsion_sincos"]]
                for group in contract.groups:
                    values = self.predict_group(torsion_input, group, device=device)
                    for orbit, value in zip(group, values, strict=True):
                        rows[orbit] = value
                result["torsion_sincos"] = torch.stack(rows)
                return result

            def predict_group(self, torsion_input, group, *, device: str):
                size = len(group)
                if not 2 <= size <= self.maximum_slots:
                    raise ValueError("generalized automorphism head supports 2--6 slots")
                rows = []
                for slot, orbit in enumerate(group):
                    slot_onehot = torch.zeros(self.maximum_slots, dtype=torsion_input.dtype, device=device)
                    slot_onehot[slot] = 1.0
                    size_onehot = torch.zeros(self.maximum_slots - 1, dtype=torsion_input.dtype, device=device)
                    size_onehot[size - 2] = 1.0
                    raw = self.automorphism_head(
                        torch.cat((torsion_input[orbit], slot_onehot, size_onehot))
                    )
                    rows.append(raw / raw.norm().clamp_min(1e-8))
                return torch.stack(rows)

        return _Model()
