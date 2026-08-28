"""锁定 SemlaFlow core 的单样本前向/反向及等变性 smoke test。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from generative_model.data import COFSymmetryDataset, SemlaFlowAdapter
from generative_model.models.semlaflow_bridge import DEFAULT_SOURCE, assert_official_source


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT / "generative_model/data/processed/v2"
DEFAULT_REPORT = ROOT / "generative_model/smoke/reports/semlaflow_random_forward.json"
DEFAULT_SPLIT = ROOT / "generative_model/smoke/splits/semlaflow_v1.json"


def _max_abs(left, right) -> float:
    import torch

    return float(torch.max(torch.abs(left - right)).detach().cpu())


def run_smoke(
    package_dir: Path,
    source_root: Path,
    package_index: int,
    device_name: str,
) -> dict[str, Any]:
    source_head = assert_official_source(source_root)
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    import torch
    import torch.nn.functional as functional
    from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator

    torch.manual_seed(20260811)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但 torch.cuda.is_available() 为 False")
    device = torch.device(device_name)

    dataset = COFSymmetryDataset(package_dir)
    split = json.loads(DEFAULT_SPLIT.read_text(encoding="utf-8"))
    if split["dataset_fingerprint"] != dataset.manifest["dataset_fingerprint"]:
        raise RuntimeError("smoke split 与加载的数据包 fingerprint 不一致")
    if package_index not in split["ordered_indices"]:
        raise ValueError(f"package index {package_index} 不在冻结 SemlaFlow smoke split")
    sample = SemlaFlowAdapter(dataset)[package_index]
    coords = torch.as_tensor(sample["semlaflow_positions"], dtype=torch.float32, device=device)[None]
    atom_indices = torch.as_tensor(sample["semlaflow_atom_types"], dtype=torch.long, device=device)[None]
    bond_indices = torch.as_tensor(sample["full_pair_bond_types"], dtype=torch.long, device=device)[None]
    atom_mask = torch.ones(atom_indices.shape, dtype=torch.long, device=device)

    vocab_size = len(sample["semlaflow_vocabulary"])
    n_bond_types = 5
    time = torch.full((*atom_indices.shape, 1), 0.5, dtype=torch.float32, device=device)
    invariant_features = torch.cat(
        (functional.one_hot(atom_indices, num_classes=vocab_size).float(), time), dim=-1
    )
    edge_features = functional.one_hot(bond_indices, num_classes=n_bond_types).float()

    dynamics = EquiInvDynamics(
        d_model=32,
        d_message=16,
        n_coord_sets=4,
        n_layers=3,
        n_attn_heads=4,
        d_message_hidden=32,
        d_edge=16,
        bond_refine=True,
        self_cond=False,
        coord_norm="length",
    )
    model = SemlaGenerator(
        d_model=32,
        dynamics=dynamics,
        vocab_size=vocab_size,
        n_atom_feats=vocab_size + 1,
        d_edge=16,
        n_edge_types=n_bond_types,
        self_cond=False,
        size_emb=16,
        max_atoms=193,
    ).to(device)
    model.eval()

    outputs = model(coords, invariant_features, edge_features, atom_mask=atom_mask)
    if len(outputs) != 4:
        raise RuntimeError(f"SemlaGenerator 输出数量应为 4，实际 {len(outputs)}")
    out_coords, out_atoms, out_bonds, out_charges = outputs
    output_names = ("coords", "atom_logits", "bond_logits", "charge_logits")
    if not all(torch.isfinite(tensor).all() for tensor in outputs):
        raise RuntimeError("随机权重前向产生 NaN/Inf")

    loss = sum(tensor.square().mean() for tensor in outputs)
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    grad_norm = float(
        torch.sqrt(
            sum(
                parameter.grad.detach().square().sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        ).cpu()
    )

    with torch.no_grad():
        matrix = torch.randn((3, 3), device=device)
        rotation, _ = torch.linalg.qr(matrix)
        if torch.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        rotated_outputs = model(
            coords @ rotation.T,
            invariant_features,
            edge_features,
            atom_mask=atom_mask,
        )
        rotation_errors = {
            "coords_max_abs": _max_abs(rotated_outputs[0], out_coords.detach() @ rotation.T),
            "atom_logits_max_abs": _max_abs(rotated_outputs[1], out_atoms.detach()),
            "bond_logits_max_abs": _max_abs(rotated_outputs[2], out_bonds.detach()),
            "charge_logits_max_abs": _max_abs(rotated_outputs[3], out_charges.detach()),
        }

        permutation = torch.randperm(coords.size(1), device=device)
        inverse = torch.argsort(permutation)
        permuted_outputs = model(
            coords[:, permutation],
            invariant_features[:, permutation],
            edge_features[:, permutation][:, :, permutation],
            atom_mask=atom_mask[:, permutation],
        )
        permutation_errors = {
            "coords_max_abs": _max_abs(permuted_outputs[0][:, inverse], out_coords.detach()),
            "atom_logits_max_abs": _max_abs(permuted_outputs[1][:, inverse], out_atoms.detach()),
            "bond_logits_max_abs": _max_abs(
                permuted_outputs[2][:, inverse][:, :, inverse], out_bonds.detach()
            ),
            "charge_logits_max_abs": _max_abs(
                permuted_outputs[3][:, inverse], out_charges.detach()
            ),
        }

    max_equivariance_error = max((*rotation_errors.values(), *permutation_errors.values()))
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "PASS" if finite_gradients and max_equivariance_error < 1e-4 else "FAIL",
        "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
        "smoke_split_fingerprint": split["split_fingerprint"],
        "source_commit": source_head,
        "package_index": package_index,
        "molecule_id": str(sample["metadata"]["Molecule_ID"]),
        "atoms": int(coords.size(1)),
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "model": {
            "architecture": "SemlaGenerator/EquiInvDynamics-mini",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "random_seed": 20260811,
        },
        "outputs": {
            name: list(tensor.shape) for name, tensor in zip(output_names, outputs)
        },
        "loss": float(loss.detach().cpu()),
        "gradient_norm": grad_norm,
        "finite_gradients": bool(finite_gradients),
        "rotation_errors": rotation_errors,
        "permutation_errors": permutation_errors,
        "acceptance": {"max_equivariance_error": 1e-4},
    }
    if report["status"] != "PASS":
        raise RuntimeError(f"SemlaFlow forward smoke 未通过: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    # 冻结 semlaflow_v1 split 中的最大分子（87 atoms, C3），覆盖较重的 full-pair 路径。
    parser.add_argument("--package-index", type=int, default=2408)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_smoke(args.package, args.source, args.package_index, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
