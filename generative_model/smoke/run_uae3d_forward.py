"""锁定 UAE-3D ``UnifiedAutoEncoder`` 的随机权重前后向与排列一致性 Gate。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

from generative_model.data import COFSymmetryDataset
from generative_model.models.uae3d_bridge import (
    DEFAULT_SOURCE,
    OFFICIAL_GEOM_POSITION_STD,
    assert_uae3d_official_source,
    canonical_to_uae3d_data,
    uae3d_model_args,
    uae3d_data_to_canonical,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT / "generative_model/data/processed/v2"
DEFAULT_SPLIT = ROOT / "generative_model/smoke/splits/uae3d_v1.json"
DEFAULT_OUTPUT = ROOT / "generative_model/smoke/reports/uae3d_random_forward_cpu.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_finite(tensor) -> bool:
    import torch

    return bool(torch.isfinite(tensor).all())


def _permuted_data(data, permutation):
    """按新节点顺序 ``old[permutation]`` 重排完整 PyG graph。"""

    import torch

    atom_count = int(data.x.shape[0])
    old_to_new = torch.empty(atom_count, dtype=torch.long)
    old_to_new[permutation] = torch.arange(atom_count, dtype=torch.long)
    edge_index = old_to_new[data.edge_index]
    order = (edge_index[0] * atom_count + edge_index[1]).argsort()
    output = data.clone()
    output.x = data.x[permutation]
    output.z = data.z[permutation]
    output.pos = data.pos[permutation]
    output.edge_index = edge_index[:, order]
    output.edge_attr = data.edge_attr[order]
    return output


def _permutation_audit(model, pyg_data, device, seed: int) -> dict[str, float]:
    import torch
    from torch_geometric.data import Batch

    atom_count = int(pyg_data.x.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    permutation = torch.randperm(atom_count, generator=generator)
    original = Batch.from_data_list([pyg_data]).to(device)
    permuted = Batch.from_data_list([_permuted_data(pyg_data, permutation)]).to(device)
    permutation_device = permutation.to(device)
    model.eval()
    with torch.no_grad():
        mean, log_var = model.encoder(
            original.x, original.edge_index, original.edge_attr, original.pos
        )
        p_mean, p_log_var = model.encoder(
            permuted.x, permuted.edge_index, permuted.edge_attr, permuted.pos
        )
        atom, bond, coordinates = model.decode(mean, batch=original.batch)
        p_atom, p_bond, p_coordinates = model.decode(p_mean, batch=permuted.batch)
    bond = bond.reshape(atom_count, atom_count, -1)
    p_bond = p_bond.reshape(atom_count, atom_count, -1)
    expected_bond = bond[permutation_device][:, permutation_device]
    return {
        "encoder_mean_max_abs_error": float(
            (p_mean - mean[permutation_device]).abs().max().cpu()
        ),
        "encoder_log_var_max_abs_error": float(
            (p_log_var - log_var[permutation_device]).abs().max().cpu()
        ),
        "atom_logits_max_abs_error": float(
            (p_atom - atom[permutation_device]).abs().max().cpu()
        ),
        "bond_logits_max_abs_error": float((p_bond - expected_bond).abs().max().cpu()),
        "coordinates_max_abs_error": float(
            (p_coordinates - coordinates[permutation_device]).abs().max().cpu()
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    commit = assert_uae3d_official_source(args.source)
    source_text = str(args.source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    import torch
    import torch_geometric
    from rdkit import rdBase
    from torch_geometric.data import Batch
    from model.autoencoder.unified_autoencoder import UnifiedAutoEncoder

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

    split = json.loads(args.split.read_text(encoding="utf-8"))
    package_index = (
        int(args.package_index)
        if args.package_index is not None
        else int(split["tier_indices"]["1"][0])
    )
    dataset = COFSymmetryDataset(args.package_dir)
    sample = dataset[package_index]
    pyg_data = canonical_to_uae3d_data(
        sample, source_root=args.source, position_std=args.position_std
    )
    restored = uae3d_data_to_canonical(pyg_data, position_std=args.position_std)
    exact_fields = (
        "atom_types", "atomic_numbers", "formal_charges", "radical_electrons",
        "bond_index", "bond_types",
    )
    roundtrip_exact = all(
        np.array_equal(restored[field], sample[field]) for field in exact_fields
    )
    position_roundtrip_max_abs = float(
        np.max(np.abs(restored["positions"] - sample["positions"]))
    )
    if not roundtrip_exact or position_roundtrip_max_abs > 1e-6:
        raise RuntimeError("canonical→UAE PyG→canonical 往返不一致")

    batch = Batch.from_data_list([pyg_data]).to(device)
    model_args = uae3d_model_args(args.profile)
    model = UnifiedAutoEncoder(model_args).to(device)
    model.train()
    start = time.perf_counter()
    losses = model(batch)
    loss = losses["loss"]
    if not all(_all_finite(value) for value in losses.values()):
        raise RuntimeError("UAE-3D forward loss 含 NaN/Inf")
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
    if not gradients or not all(_all_finite(gradient) for gradient in gradients):
        raise RuntimeError("UAE-3D gradient 缺失或含 NaN/Inf")
    required_prefixes = (
        "encoder.node_embedding", "encoder.encoder_blocks", "decoder.atom_head",
        "decoder.bond_head", "decoder.coordinate_head",
    )
    missing_required = [
        prefix for prefix in required_prefixes
        if not any(name.startswith(prefix) and gradient is not None
                   for name, gradient in named_gradients)
    ]
    if missing_required:
        raise RuntimeError(f"UAE-3D 必需模块没有 gradient: {missing_required}")

    model.eval()
    with torch.no_grad():
        z_mean, z_log_var = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.pos)
        atom_logits, bond_logits, coordinates = model.decode(z_mean, batch=batch.batch)
    atom_count = len(sample["atom_types"])
    expected_shapes = {
        "z_mean": [atom_count, 16],
        "z_log_var": [atom_count, 16],
        "atom_logits": [atom_count, 16],
        "bond_logits": [atom_count * atom_count, 6],
        "coordinates": [atom_count, 3],
    }
    tensors = {
        "z_mean": z_mean,
        "z_log_var": z_log_var,
        "atom_logits": atom_logits,
        "bond_logits": bond_logits,
        "coordinates": coordinates,
    }
    actual_shapes = {name: list(tensor.shape) for name, tensor in tensors.items()}
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"UAE-3D 输出形状错误: {actual_shapes}")
    if not all(_all_finite(tensor) for tensor in tensors.values()):
        raise RuntimeError("UAE-3D encoder/decoder 输出含 NaN/Inf")

    permutation = _permutation_audit(model, pyg_data, device, args.seed)
    permutation_max_error = max(permutation.values())
    if permutation_max_error > args.permutation_tolerance:
        raise RuntimeError(
            f"UAE-3D 排列一致性误差 {permutation_max_error} > "
            f"{args.permutation_tolerance}"
        )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "PASS",
        "purpose": "UAE-3D random UnifiedAutoEncoder interface forward/backward gate",
        "official_source_commit": commit,
        "input": {
            "package_index": package_index,
            "molecule_id": sample["molecule_id"],
            "target_pg": sample["target_pg"],
            "atoms": atom_count,
            "undirected_bonds": len(sample["bond_types"]),
            "full_pair_edges": int(pyg_data.edge_index.shape[1]),
            "explicit_h": True,
            "canonical_pyg_roundtrip_exact": roundtrip_exact,
            "position_roundtrip_max_abs_angstrom": position_roundtrip_max_abs,
            "position_std": args.position_std,
            "package_manifest_sha256": _sha256_file(args.package_dir / "manifest.json"),
            "split_sha256": _sha256_file(args.split),
            "split_fingerprint": split["split_fingerprint"],
        },
        "model": {
            "class": "model.autoencoder.unified_autoencoder.UnifiedAutoEncoder",
            "profile": args.profile,
            "random_initialization": True,
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "model_args": vars(model_args),
            "output_shapes": actual_shapes,
        },
        "numerics": {
            "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
            "outputs_finite": True,
            "parameter_tensors_with_gradient": len(gradients),
            "parameter_tensors_without_gradient": len(missing_gradient_names),
            "parameters_without_gradient": missing_gradient_names,
            "required_modules_have_gradients": True,
            "all_gradients_finite": True,
            "permutation_tolerance": args.permutation_tolerance,
            "permutation": permutation,
            "permutation_max_abs_error": permutation_max_error,
            "se3_note": (
                "UAE-3D learns SE(3) behavior through augmentation; random weights are not "
                "required to be rotation/translation equivariant. Audit after reconstruction training."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_geometric": torch_geometric.__version__,
            "rdkit": rdBase.rdkitVersion,
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
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--package-index", type=int)
    parser.add_argument("--position-std", type=float, default=OFFICIAL_GEOM_POSITION_STD)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", choices=("mini", "official"), default="official")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--permutation-tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
