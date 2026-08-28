"""Audit frozen UAE-3D bond-feature separability with deterministic probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from generative_model.data import COFSymmetryDataset
from generative_model.models.uae3d_bond_capacity import (
    CAPACITY_SCHEMA_VERSION,
    audit_feature_probe,
    audit_legacy_head,
    make_feature_table,
    merge_feature_tables,
)
from generative_model.models.uae3d_bridge import (
    DEFAULT_SOURCE,
    OFFICIAL_GEOM_POSITION_STD,
    assert_uae3d_official_source,
    canonical_to_uae3d_data,
    uae3d_model_args,
)
from generative_model.models.uae3d_reconstruction import full_pair_targets
from generative_model.smoke.train_uae3d_reconstruction import (
    DEFAULT_PACKAGE,
    _sha256_file,
    _state_fingerprint,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPACITY_SPLIT = (
    ROOT / "generative_model/smoke/splits/uae3d_bond_capacity_v1.json"
)
DEFAULT_SOURCE_CHECKPOINT = (
    ROOT / "generative_model/runs/uae3d_overfit4/"
    "from_tier1_w10_lr3em5_0512/checkpoints/step-003008.pt"
)
DEFAULT_CANDIDATE_CHECKPOINT = (
    ROOT / "generative_model/runs/uae3d_overfit4/"
    "hierarchical_v4_balanced_0016/last.pt"
)
DEFAULT_OUTPUT = (
    ROOT / "generative_model/smoke/reports/uae3d_bond_capacity_v1.json"
)


def _load_panel(
    dataset: COFSymmetryDataset,
    indices: list[int],
    *,
    source: Path,
    position_std: float,
) -> list[Any]:
    result = [
        canonical_to_uae3d_data(
            dataset[index],
            source_root=source,
            position_std=position_std,
            verify_source=False,
        )
        for index in indices
    ]
    if [int(item.idx) for item in result] != indices:
        raise RuntimeError("capacity panel package index 顺序发生变化")
    return result


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    *,
    split: dict[str, Any],
    tier4_indices: list[int],
    source_commit: str,
) -> None:
    if checkpoint.get("schema_version") != "uae3d-reconstruction-checkpoint-v1":
        raise RuntimeError("capacity audit checkpoint schema 错误")
    identity = checkpoint.get("identity", {})
    expected = {
        "tier": 4,
        "package_indices": tier4_indices,
        "official_source_commit": source_commit,
        "dataset_fingerprint": split["dataset_fingerprint"],
        "package_manifest_sha256": split["package_manifest_sha256"],
        "split_fingerprint": split["source_uae_split_fingerprint"],
    }
    mismatch = [key for key, value in expected.items() if identity.get(key) != value]
    if mismatch:
        raise RuntimeError(f"capacity checkpoint identity 不一致: {mismatch}")


def _extract_feature_table(model, data_list: list[Any], *, device, batch_size: int):
    import torch
    from torch_geometric.data import Batch

    tables = []
    prior_mode = model.training
    model.eval()
    for begin in range(0, len(data_list), batch_size):
        items = data_list[begin : begin + batch_size]
        batch = Batch.from_data_list(items).to(device)
        pair_capture: list[Any] = []
        hidden_capture: list[Any] = []

        def capture_pair(_module, inputs):
            if len(inputs) != 1:
                raise RuntimeError("bond_head first linear hook input 非法")
            pair_capture.append(inputs[0].detach())

        def capture_hidden(_module, inputs):
            if len(inputs) != 1:
                raise RuntimeError("bond_head final linear hook input 非法")
            hidden_capture.append(inputs[0].detach())

        pair_hook = model.decoder.bond_head[0].register_forward_pre_hook(capture_pair)
        hidden_hook = model.decoder.bond_head[2].register_forward_pre_hook(capture_hidden)
        try:
            with torch.no_grad():
                z_mean, z_log_var = model.encoder(
                    batch.x, batch.edge_index, batch.edge_attr, batch.pos
                )
                _, bond_logits, _ = model.decode(z_mean, batch=batch.batch)
        finally:
            pair_hook.remove()
            hidden_hook.remove()
        if len(pair_capture) != 1 or len(hidden_capture) != 1:
            raise RuntimeError("bond-head hook 不是恰好触发一次")
        pair_full = pair_capture[0]
        hidden_full = hidden_capture[0]
        tensors = (z_mean, z_log_var, pair_full, hidden_full, bond_logits)
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise RuntimeError("capacity feature extraction 含 NaN/Inf")
        if pair_full.shape != hidden_full.shape or pair_full.shape[1] != 64:
            raise RuntimeError("bond-head pair/hidden feature shape 非法")
        if bond_logits.shape != (len(pair_full), 6):
            raise RuntimeError("bond logits shape 与 captured feature 不一致")

        targets_full = full_pair_targets(batch.edge_attr)
        pair_parts = []
        hidden_parts = []
        logit_parts = []
        target_parts = []
        offsets = [0]
        full_begin = 0
        for molecule_index, item in enumerate(items):
            atom_count = int(item.x.shape[0])
            full_end = full_begin + atom_count * atom_count
            local_pair = pair_full[full_begin:full_end].reshape(atom_count, atom_count, 64)
            local_hidden = hidden_full[full_begin:full_end].reshape(atom_count, atom_count, 64)
            local_logits = bond_logits[full_begin:full_end].reshape(atom_count, atom_count, 6)
            local_targets = targets_full[full_begin:full_end].reshape(atom_count, atom_count)
            for name, tensor in (
                ("pair", local_pair), ("hidden", local_hidden), ("logits", local_logits)
            ):
                error = float((tensor - tensor.transpose(0, 1)).abs().max().cpu())
                if error > 1e-6:
                    raise RuntimeError(
                        f"molecule {molecule_index} {name} feature 非对称: {error}"
                    )
            if not bool(torch.equal(local_targets, local_targets.T)):
                raise RuntimeError("capacity target 非对称")
            if not bool((torch.diag(local_targets) == 5).all()):
                raise RuntimeError("capacity target diagonal 不是 self")
            upper = torch.triu(
                torch.ones((atom_count, atom_count), dtype=torch.bool, device=device),
                diagonal=1,
            )
            upper_targets = local_targets[upper]
            if not bool((upper_targets == 0).any()) or not bool((upper_targets > 0).any()):
                raise RuntimeError("capacity molecule 缺少 none/present stratum")
            pair_parts.append(local_pair[upper].cpu().to(torch.float32).numpy())
            hidden_parts.append(local_hidden[upper].cpu().to(torch.float32).numpy())
            logit_parts.append(local_logits[upper].cpu().to(torch.float32).numpy())
            target_parts.append(upper_targets.cpu().to(torch.int16).numpy())
            offsets.append(offsets[-1] + int(upper_targets.numel()))
            full_begin = full_end
        if full_begin != len(pair_full):
            raise RuntimeError("capacity full-pair offsets 不一致")
        tables.append(make_feature_table(
            np.concatenate(pair_parts),
            np.concatenate(hidden_parts),
            np.concatenate(logit_parts),
            np.concatenate(target_parts),
            np.asarray(offsets, dtype=np.int64),
            np.asarray([int(item.idx) for item in items], dtype=np.int64),
        ))
    model.train(prior_mode)
    return tables[0] if len(tables) == 1 else merge_feature_tables(tables)


def _checkpoint_audit(
    checkpoint_path: Path,
    *,
    model,
    fit_data: list[Any],
    transfer_data: list[Any],
    device,
    batch_size: int,
    split: dict[str, Any],
    tier4_indices: list[int],
    source_commit: str,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_checkpoint(
        checkpoint,
        split=split,
        tier4_indices=tier4_indices,
        source_commit=source_commit,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    state_before = _state_fingerprint(model.state_dict())
    fit_table = _extract_feature_table(
        model, fit_data, device=device, batch_size=batch_size
    )
    transfer_table = _extract_feature_table(
        model, transfer_data, device=device, batch_size=batch_size
    )
    state_after = _state_fingerprint(model.state_dict())
    if state_before != state_after:
        raise RuntimeError("只读 capacity audit 修改了 UAE-3D model state")
    probes = {
        feature_name: audit_feature_probe(fit_table, transfer_table, feature_name)
        for feature_name in ("pair_features", "hidden_features")
    }
    classes = {
        name: result["linear_signal_classification"]
        for name, result in probes.items()
    }
    if classes["hidden_features"] == "FEATURE_SEPARABLE":
        diagnosis = "POST_GELU_FEATURES_SUPPORT_EXPLICIT_EXISTENCE_HEAD"
    elif classes["pair_features"] == "FEATURE_SEPARABLE":
        diagnosis = "PAIR_FEATURES_SEPARABLE_BUT_CURRENT_HIDDEN_TRANSFORM_LOSES_SIGNAL"
    elif "WEAK_LINEAR_SIGNAL" in classes.values():
        diagnosis = "FROZEN_FEATURES_HAVE_WEAK_SIGNAL_ONLY"
    else:
        diagnosis = "FROZEN_FEATURES_NOT_LINEARLY_SEPARABLE"
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "model_state_fingerprint": state_before,
        "model_state_unchanged": state_before == state_after,
        "fit_feature_table_fingerprint": fit_table["fingerprint"],
        "transfer_feature_table_fingerprint": transfer_table["fingerprint"],
        "legacy_head": audit_legacy_head(fit_table, transfer_table),
        "auxiliary_probes": probes,
        "diagnosis": diagnosis,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_commit = assert_uae3d_official_source(args.source)
    source_text = str(args.source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    import torch
    import torch_geometric
    from rdkit import rdBase
    from model.autoencoder.unified_autoencoder import UnifiedAutoEncoder

    if args.batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境不可用")
    if device.type != "cpu" and args.output.resolve() == DEFAULT_OUTPUT.resolve():
        raise RuntimeError(
            "正式 capacity report 固定使用 CPU 单线程；CUDA 预检必须指定独立 --output"
        )
    split = json.loads(args.capacity_split.read_text(encoding="utf-8"))
    if split.get("schema_version") != "uae3d-bond-capacity-split-v1":
        raise ValueError("capacity split schema 错误")
    leakage = split.get("leakage_policy", {})
    required_false = ("test_used", "core_ood_used", "probe_is_a_training_gate")
    if any(leakage.get(name) is not False for name in required_false):
        raise RuntimeError("capacity split 泄漏或被误标为 training gate")
    if leakage.get("checkpoint_training_molecules_excluded_from_probe_fit") is not True:
        raise RuntimeError("capacity fit 未明确排除 checkpoint training molecules")
    fit_indices = list(map(int, split["fit"]["package_indices"]))
    transfer_indices = list(map(int, split["transfer"]["package_indices"]))
    if set(fit_indices) & set(transfer_indices):
        raise RuntimeError("capacity fit/transfer 泄漏")

    dataset = COFSymmetryDataset(args.package_dir)
    if dataset.manifest["dataset_fingerprint"] != split["dataset_fingerprint"]:
        raise RuntimeError("capacity dataset fingerprint 不一致")
    if _sha256_file(args.package_dir / "manifest.json") != split["package_manifest_sha256"]:
        raise RuntimeError("capacity package manifest SHA-256 不一致")
    fit_data = _load_panel(
        dataset, fit_indices, source=args.source, position_std=args.position_std
    )
    transfer_data = _load_panel(
        dataset, transfer_indices, source=args.source, position_std=args.position_std
    )
    tier4_indices = list(map(int, json.loads(
        (ROOT / "generative_model/smoke/splits/uae3d_v1.json").read_text(
            encoding="utf-8"
        )
    )["tier_indices"]["4"]))
    if set(tier4_indices) & set(fit_indices):
        raise RuntimeError("capacity fit panel 含 tier-4 checkpoint training molecule")

    model = UnifiedAutoEncoder(uae3d_model_args("official"))
    source_result = _checkpoint_audit(
        args.source_checkpoint,
        model=model,
        fit_data=fit_data,
        transfer_data=transfer_data,
        device=device,
        batch_size=args.batch_size,
        split=split,
        tier4_indices=tier4_indices,
        source_commit=source_commit,
    )
    candidate_result = _checkpoint_audit(
        args.candidate_checkpoint,
        model=model,
        fit_data=fit_data,
        transfer_data=transfer_data,
        device=device,
        batch_size=args.batch_size,
        split=split,
        tier4_indices=tier4_indices,
        source_commit=source_commit,
    )
    diagnoses = [source_result["diagnosis"], candidate_result["diagnosis"]]
    if any(value == "POST_GELU_FEATURES_SUPPORT_EXPLICIT_EXISTENCE_HEAD" for value in diagnoses):
        status = "EVIDENCE_SUPPORTS_EXPLICIT_EXISTENCE_HEAD_PROTOTYPE"
    elif any("PAIR_FEATURES_SEPARABLE" in value for value in diagnoses):
        status = "EVIDENCE_SUPPORTS_REPLACING_CURRENT_BOND_HIDDEN_TRANSFORM"
    elif any("WEAK_SIGNAL" in value for value in diagnoses):
        status = "WEAK_SIGNAL_REQUIRES_BOUNDED_ARCHITECTURE_AUDIT"
    else:
        status = "NO_LINEAR_SEPARABILITY_STOP_LOCAL_HEAD_REFINEMENT"
    report = {
        "schema_version": CAPACITY_SCHEMA_VERSION,
        "status": status,
        "purpose": (
            "Read-only diagnosis of whether frozen UAE-3D pair/post-GELU features "
            "contain transferable bond-existence and conditional-type signal."
        ),
        "protocol": {
            "capacity_split": str(args.capacity_split.resolve()),
            "capacity_split_sha256": _sha256_file(args.capacity_split),
            "capacity_split_fingerprint": split["split_fingerprint"],
            "fit_role": split["leakage_policy"]["probe_fit_role"],
            "transfer_role": split["leakage_policy"]["probe_transfer_role"],
            "fit_molecule_count": len(fit_indices),
            "transfer_molecule_count": len(transfer_indices),
            "test_used": False,
            "core_ood_used": False,
            "uae3d_weights_updated": False,
            "probe_is_a_tier_gate": False,
            "canonical_report_execution": "CPU single-thread only",
            "features": {
                "pair_features": "decoder pair=h_i+h_j before bond_head",
                "hidden_features": "64-d post-GELU input to final 6-class linear",
            },
            "existence_fit_sampling": (
                "per molecule all present unique pairs plus equal evenly-spaced none pairs"
            ),
            "predeclared_signal_classification": {
                "FEATURE_SEPARABLE": (
                    "transfer ROC-AUC>=0.85, calibrated macro-balanced>=0.75, "
                    "none accuracy>=0.70, present recall>=0.70"
                ),
                "WEAK_LINEAR_SIGNAL": "transfer ROC-AUC>=0.70 otherwise",
                "NOT_LINEARLY_SEPARABLE": "transfer ROC-AUC<0.70",
            },
        },
        "identity": {
            "dataset_fingerprint": split["dataset_fingerprint"],
            "package_manifest_sha256": split["package_manifest_sha256"],
            "official_source_commit": source_commit,
            "position_std": args.position_std,
            "fit_package_indices": fit_indices,
            "transfer_package_indices": transfer_indices,
            "tier4_checkpoint_training_indices": tier4_indices,
        },
        "checkpoints": {
            "source_step3008": source_result,
            "candidate_step16": candidate_result,
        },
        "decision_scope": {
            "resume_current_tier4_training": False,
            "enter_tier32": False,
            "reason": (
                "This report selects only the next bounded architecture diagnostic; "
                "strict tier gates remain unchanged."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "rdkit": rdBase.rdkitVersion,
            "device": str(device),
            "seed": args.seed,
            "torch_threads": torch.get_num_threads(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--capacity-split", type=Path, default=DEFAULT_CAPACITY_SPLIT)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--candidate-checkpoint", type=Path, default=DEFAULT_CANDIDATE_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--position-std", type=float, default=OFFICIAL_GEOM_POSITION_STD)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260820)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    compact = {
        "status": report["status"],
        "source_diagnosis": report["checkpoints"]["source_step3008"]["diagnosis"],
        "candidate_diagnosis": report["checkpoints"]["candidate_step16"]["diagnosis"],
        "output": str(args.output),
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
