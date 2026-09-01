"""WL-fingerprint-conditioned global-shape head for M3.6."""

from __future__ import annotations

import hashlib

import numpy as np

from .m3p1_model import AutomorphismSetQuotientICPredictor


def graph_wl_fingerprint(sample, *, bits: int = 512, radius: int = 4) -> np.ndarray:
    """Return a deterministic atom-order-invariant counted WL fingerprint."""

    if bits <= 0 or radius < 0:
        raise ValueError("invalid WL fingerprint settings")
    atom_count = len(sample.atomic_numbers)
    neighbors = [[] for _ in range(atom_count)]
    for (left, right), bond_type in zip(
        np.asarray(sample.bond_index, dtype=np.int64).T,
        np.asarray(sample.bond_types, dtype=np.int64),
        strict=True,
    ):
        neighbors[int(left)].append((int(right), int(bond_type)))
        neighbors[int(right)].append((int(left), int(bond_type)))
    labels = [
        f"{int(atom_type)}:{int(charge)}:{int(radical)}"
        for atom_type, charge, radical in zip(
            sample.atom_types,
            sample.formal_charges,
            sample.radical_electrons,
            strict=True,
        )
    ]
    fingerprint = np.zeros(bits, dtype=np.float64)
    for current_radius in range(radius + 1):
        for label in labels:
            digest = hashlib.sha256(
                f"r{current_radius}|{label}".encode("utf-8")
            ).digest()
            fingerprint[int.from_bytes(digest[:8], "big") % bits] += 1.0
        if current_radius < radius:
            labels = [
                hashlib.sha256(
                    (
                        labels[node]
                        + "|"
                        + "|".join(
                            sorted(
                                f"{bond}:{labels[neighbor]}"
                                for neighbor, bond in neighbors[node]
                            )
                        )
                    ).encode("utf-8")
                ).hexdigest()
                for node in range(atom_count)
            ]
    norm = float(np.linalg.norm(fingerprint))
    if norm == 0.0:
        raise RuntimeError("empty WL fingerprint")
    return (fingerprint / norm).astype(np.float32)


class FingerprintGlobalShapePredictor:
    def __new__(
        cls,
        *,
        quantile_count: int = 16,
        fingerprint_bits: int = 512,
        **model_settings,
    ):
        import torch

        parent = AutomorphismSetQuotientICPredictor(**model_settings)
        hidden_dim = int(model_settings.get("hidden_dim", 96))
        shape_hidden = 256

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parent = parent
                self.global_shape_head = torch.nn.Sequential(
                    torch.nn.Linear(2 * hidden_dim + fingerprint_bits, shape_hidden),
                    torch.nn.SiLU(),
                    torch.nn.Linear(shape_hidden, shape_hidden),
                    torch.nn.SiLU(),
                    torch.nn.Linear(shape_hidden, quantile_count),
                )
                self._captured_shape_input = None

            def initialize_from_m3p3(self, parent_state: dict):
                self.parent.load_state_dict(parent_state, strict=True)

            def configure_shape_head_only(self):
                for parameter in self.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.global_shape_head.parameters():
                    parameter.requires_grad_(True)
                return [
                    name
                    for name, parameter in self.named_parameters()
                    if parameter.requires_grad
                ]

            def forward(self, graph, automorphism, fingerprint, *, device: str):
                captured = []

                def capture_nodes(_module, _inputs, output):
                    captured.append(output)

                handle = self.parent.parent.base.norms[-1].register_forward_hook(
                    capture_nodes
                )
                try:
                    result = self.parent(graph, automorphism, device=device)
                finally:
                    handle.remove()
                if len(captured) != 1:
                    raise RuntimeError("failed to capture quotient node states")
                nodes = captured[0]
                embedding = torch.cat(
                    (nodes.mean(dim=0), nodes.max(dim=0).values), dim=0
                )
                fp = torch.as_tensor(
                    fingerprint, dtype=nodes.dtype, device=device
                )
                shape_input = torch.cat((embedding, fp), dim=0)
                self._captured_shape_input = shape_input
                result["global_shape_quantiles"] = self.predict_cached(shape_input)
                return result

            def predict_cached(self, shape_input):
                raw = self.global_shape_head(shape_input)
                return torch.sort(0.5 + torch.nn.functional.softplus(raw)).values

        return _Model()
