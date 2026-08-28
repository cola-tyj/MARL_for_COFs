"""锁定 MiDi 的 canonical→PyG→dense→GraphTransformer 单 batch Gate。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

from generative_model.data import COFSymmetryDataset
from generative_model.data.build_cof_package import sha256_file
from generative_model.models.midi_bridge import (
    DEFAULT_SOURCE,
    assert_official_source,
    canonical_to_midi_data,
    midi_data_to_canonical,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT / "generative_model/data/processed/v2"
DEFAULT_OUTPUT = ROOT / "generative_model/smoke/reports/midi_random_forward_cuda.json"


def _all_finite(tensor) -> bool:
    import torch

    return bool(torch.isfinite(tensor).all())


def run(args: argparse.Namespace) -> dict[str, Any]:
    commit = assert_official_source(args.source)
    source_text = str(args.source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Batch
    import torch_geometric
    import midi
    import midi.utils as midi_utils
    from midi.models.transformer_model import GraphTransformer

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但 torch.cuda.is_available() 为 False")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    dataset = COFSymmetryDataset(args.package_dir)
    sample = dataset[args.package_index]
    pyg_data = canonical_to_midi_data(sample, source_root=args.source)
    restored = midi_data_to_canonical(pyg_data)
    roundtrip_fields = (
        "positions", "atom_types", "atomic_numbers", "formal_charges",
        "bond_index", "bond_types",
    )
    roundtrip_exact = all(
        np.array_equal(restored[field], sample[field]) for field in roundtrip_fields
    )
    if not roundtrip_exact:
        raise RuntimeError("canonical→PyG→canonical 往返不一致")

    class DatasetInfo:
        num_atom_types = 16

        @staticmethod
        def to_one_hot(X, charges, E, node_mask):
            placeholder = midi_utils.PlaceHolder(
                X=F.one_hot(X, num_classes=16).float(),
                charges=F.one_hot(charges + 2, num_classes=6).float(),
                E=F.one_hot(E, num_classes=5).float(),
                y=None,
                pos=None,
                node_mask=node_mask,
            ).mask()
            return placeholder.X, placeholder.charges, placeholder.E

    batch = Batch.from_data_list([pyg_data]).to(device)
    dense = midi_utils.to_dense(batch, DatasetInfo(), device=device)
    dense.y = torch.zeros((1, 1), dtype=dense.X.dtype, device=device)
    input_dims = midi_utils.PlaceHolder(X=16, charges=6, E=5, y=1, pos=3)
    output_dims = midi_utils.PlaceHolder(X=16, charges=6, E=5, y=0, pos=3)
    model = GraphTransformer(
        input_dims=input_dims,
        n_layers=args.layers,
        hidden_mlp_dims={"X": 64, "E": 32, "y": 32, "pos": 32},
        hidden_dims={
            "dx": 64, "de": 32, "dy": 32, "n_head": 8,
            "dim_ffX": 64, "dim_ffE": 32, "dim_ffy": 64,
        },
        output_dims=output_dims,
    ).to(device)
    model.train()
    start = time.perf_counter()
    output = model(dense)
    loss = (
        output.pos.square().mean()
        + output.X.square().mean()
        + output.charges.square().mean()
        + output.E.square().mean()
    )
    if not _all_finite(loss):
        raise RuntimeError("MiDi forward loss 含 NaN/Inf")
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    named_gradients = [
        (name, parameter.grad) for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    gradients = [gradient for _, gradient in named_gradients if gradient is not None]
    missing_gradient_names = [name for name, gradient in named_gradients if gradient is None]
    if not gradients:
        raise RuntimeError("MiDi 没有任何参数获得 gradient")
    required_heads = ("mlp_out_X", "mlp_out_E", "mlp_out_pos")
    for head in required_heads:
        if not any(name.startswith(head) and gradient is not None
                   for name, gradient in named_gradients):
            raise RuntimeError(f"MiDi 输出 head 没有 gradient: {head}")
    gradients_finite = all(_all_finite(gradient) for gradient in gradients)
    if not gradients_finite:
        raise RuntimeError("MiDi gradient 含 NaN/Inf")
    atom_count = len(sample["atom_types"])
    expected_shapes = {
        "pos": [1, atom_count, 3],
        "X": [1, atom_count, 16],
        "charges": [1, atom_count, 6],
        "E": [1, atom_count, atom_count, 5],
        "y": [1, 0],
    }
    actual_shapes = {
        key: list(getattr(output, key).shape) for key in expected_shapes
    }
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"MiDi 输出形状错误: {actual_shapes}")
    dense_target_exact = bool(
        torch.equal(dense.X.argmax(dim=-1)[0], pyg_data.x.to(device))
        and torch.equal(dense.charges.argmax(dim=-1)[0] - 2, pyg_data.charges.to(device))
    )
    if not dense_target_exact:
        raise RuntimeError("官方 to_dense 改变了 atom/charge target")

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "PASS",
        "purpose": "MiDi random mini GraphTransformer interface forward/backward gate",
        "official_source_commit": commit,
        "input": {
            "package_index": args.package_index,
            "molecule_id": sample["molecule_id"],
            "atoms": atom_count,
            "undirected_bonds": len(sample["bond_types"]),
            "pyg_directed_edges": int(pyg_data.edge_index.shape[1]),
            "explicit_h": True,
            "canonical_pyg_roundtrip_exact": roundtrip_exact,
            "official_to_dense_atom_charge_exact": dense_target_exact,
            "package_manifest_sha256": sha256_file(args.package_dir / "manifest.json"),
        },
        "model": {
            "class": "midi.models.transformer_model.GraphTransformer",
            "random_initialization": True,
            "layers": args.layers,
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "output_shapes": actual_shapes,
        },
        "numerics": {
            "loss": float(loss.detach().cpu()),
            "outputs_finite": all(_all_finite(getattr(output, key)) for key in expected_shapes),
            "parameter_tensors_with_gradient": len(gradients),
            "parameter_tensors_without_gradient": len(missing_gradient_names),
            "parameters_without_gradient": missing_gradient_names,
            "required_output_heads_have_gradients": True,
            "all_gradients_finite": gradients_finite,
            "coordinate_com_max_abs": float(output.pos.mean(dim=1).abs().max().detach().cpu()),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_geometric": torch_geometric.__version__,
            "midi_path": str(Path(midi.__file__).resolve()),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "timing_seconds": {"forward_backward": elapsed},
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--package-index", type=int, default=2408)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
