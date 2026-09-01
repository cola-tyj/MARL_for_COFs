"""Set-valued local-rotor torsion head layered on frozen M2.1 features."""

from __future__ import annotations

from .orbit_ic_model import QuotientICPredictor


class SetValuedQuotientICPredictor:
    """Factory wrapper that preserves M2.1 and replaces only grouped slots."""

    def __new__(cls, **model_settings):
        import torch

        base = QuotientICPredictor(**model_settings)
        hidden_dim = int(model_settings.get("hidden_dim", 96))
        torsion_input_dim = int(base.torsion_head[0].in_features)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = base
                self.local_rotor_head = torch.nn.Sequential(
                    torch.nn.Linear(torsion_input_dim + 5, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, 2),
                )
                self._captured_torsion_input = None
                self._hook = self.base.torsion_head[0].register_forward_pre_hook(
                    self._capture_torsion_input
                )

            def _capture_torsion_input(self, _module, arguments):
                self._captured_torsion_input = arguments[0]

            def initialize_local_rotor_head_from_base(self):
                """Start all slots at the exact M2.1 prediction before learning."""

                with torch.no_grad():
                    first = self.local_rotor_head[0]
                    source_first = self.base.torsion_head[0]
                    first.weight.zero_()
                    first.weight[:, :torsion_input_dim].copy_(source_first.weight)
                    first.bias.copy_(source_first.bias)
                    self.local_rotor_head[2].weight.copy_(
                        self.base.torsion_head[2].weight
                    )
                    self.local_rotor_head[2].bias.copy_(self.base.torsion_head[2].bias)

            def forward(self, graph, local_rotor_contract, *, device: str):
                self._captured_torsion_input = None
                result = self.base(graph, device=device)
                torsion_input = self._captured_torsion_input
                if torsion_input is None:
                    raise RuntimeError("M2.2 failed to capture M2.1 torsion features")
                rows = [row for row in result["torsion_sincos"]]
                for group in local_rotor_contract.groups:
                    values = self.predict_local_rotor_group(torsion_input, group, device=device)
                    for orbit, value in zip(group, values, strict=True):
                        rows[orbit] = value
                result["torsion_sincos"] = torch.stack(rows)
                return result

            def predict_local_rotor_group(self, torsion_input, group, *, device: str):
                size = len(group)
                rows = []
                for slot, orbit in enumerate(group):
                    slot_onehot = torch.zeros(3, dtype=torsion_input.dtype, device=device)
                    slot_onehot[slot] = 1.0
                    size_onehot = torch.zeros(2, dtype=torsion_input.dtype, device=device)
                    size_onehot[size - 2] = 1.0
                    raw = self.local_rotor_head(
                        torch.cat((torsion_input[orbit], slot_onehot, size_onehot))
                    )
                    rows.append(raw / raw.norm().clamp_min(1e-8))
                return torch.stack(rows)

        return _Model()


def load_m2p1_parent(model, parent_state: dict) -> dict:
    """Load the complete frozen M2.1 model under the M2.2 ``base`` prefix."""

    target = model.state_dict()
    inherited = []
    for key, value in parent_state.items():
        destination = f"base.{key}"
        if destination not in target or target[destination].shape != value.shape:
            raise ValueError(f"incompatible M2.1 parent tensor: {key}")
        target[destination] = value.detach().clone()
        inherited.append(destination)
    model.load_state_dict(target, strict=True)
    model.initialize_local_rotor_head_from_base()
    return {
        "inherited_tensor_count": len(inherited),
        "inherited_parameter_count": int(sum(target[key].numel() for key in inherited)),
        "inherited_keys": sorted(inherited),
    }


def freeze_m2p1_parent(model) -> dict:
    """Freeze every M2.1 tensor so Gate B changes only the set-valued head."""

    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith("local_rotor_head.") for name in trainable):
        raise RuntimeError("M2.2 trainable-parameter isolation failed")
    return {
        "trainable_keys": trainable,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
