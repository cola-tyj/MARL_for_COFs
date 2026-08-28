"""Run the frozen final UAE-3D Gate A without updating UAE-3D weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from generative_model.data import COFSymmetryDataset
from generative_model.models.uae3d_bridge import (
    DEFAULT_SOURCE,
    OFFICIAL_GEOM_POSITION_STD,
    assert_uae3d_official_source,
    uae3d_model_args,
)
from generative_model.models.uae3d_gate_a import fit_existence_probe, gate_a_checks
from generative_model.smoke.audit_uae3d_bond_capacity import (
    _extract_feature_table,
    _load_panel,
    _validate_checkpoint,
)
from generative_model.smoke.train_uae3d_reconstruction import (
    DEFAULT_PACKAGE,
    _sha256_file,
    _state_fingerprint,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "generative_model/smoke/reports/uae3d_gate_a_protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "generative_model/smoke/reports/uae3d_gate_a_v1.json"


def _protocol_fingerprint(protocol: dict[str, Any]) -> str:
    payload = dict(protocol)
    payload.pop("protocol_fingerprint", None)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "uae3d-gate-a-protocol-v1":
        raise RuntimeError("Gate A protocol schema 错误")
    if protocol.get("status") != "FROZEN_BEFORE_GATE_A_EXECUTION":
        raise RuntimeError("Gate A protocol 未在运行前冻结")
    if protocol.get("protocol_fingerprint") != _protocol_fingerprint(protocol):
        raise RuntimeError("Gate A protocol fingerprint 不匹配")
    policy = protocol["leakage_and_mutation_policy"]
    if any(policy[name] is not False for name in (
        "test_used", "core_ood_used", "uae3d_weights_updated",
        "tier4_oracle_used_for_transfer_threshold",
        "iid_validation_used_for_probe_or_threshold_fit",
    )):
        raise RuntimeError("Gate A protocol 泄漏/更新策略非法")

    source_checkpoint = Path(protocol["source"]["checkpoint"])
    if _sha256_file(source_checkpoint) != protocol["source"]["checkpoint_sha256"]:
        raise RuntimeError("Gate A source checkpoint SHA-256 不一致")
    capacity_split_path = Path(protocol["identity"]["capacity_split"])
    uae_split_path = Path(protocol["identity"]["uae_split"])
    if _sha256_file(capacity_split_path) != protocol["identity"]["capacity_split_sha256"]:
        raise RuntimeError("Gate A capacity split SHA-256 不一致")
    if _sha256_file(uae_split_path) != protocol["identity"]["uae_split_sha256"]:
        raise RuntimeError("Gate A UAE split SHA-256 不一致")
    split = json.loads(capacity_split_path.read_text(encoding="utf-8"))
    uae_split = json.loads(uae_split_path.read_text(encoding="utf-8"))
    fit_indices = list(map(int, protocol["panels"]["iid_probe_fit"]["package_indices"]))
    transfer_indices = list(map(int, protocol["panels"]["iid_validation_transfer"]["package_indices"]))
    tier4_indices = list(map(int, protocol["panels"]["tier4_oracle"]["package_indices"]))
    if fit_indices != list(map(int, split["fit"]["package_indices"])):
        raise RuntimeError("Gate A fit panel 与 frozen split 不一致")
    if transfer_indices != list(map(int, split["transfer"]["package_indices"])):
        raise RuntimeError("Gate A transfer panel 与 frozen split 不一致")
    if tier4_indices != list(map(int, uae_split["tier_indices"]["4"])):
        raise RuntimeError("Gate A tier4 oracle 与 frozen split 不一致")
    if set(fit_indices) & set(transfer_indices):
        raise RuntimeError("Gate A fit/transfer panel 泄漏")

    source_commit = assert_uae3d_official_source(args.source)
    source_text = str(args.source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    import torch
    import torch_geometric
    from rdkit import rdBase
    from model.autoencoder.unified_autoencoder import UnifiedAutoEncoder

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境不可用")
    if args.batch_size <= 0:
        raise ValueError("batch_size 必须为正")

    dataset = COFSymmetryDataset(args.package_dir)
    if dataset.manifest["dataset_fingerprint"] != protocol["identity"]["dataset_fingerprint"]:
        raise RuntimeError("Gate A dataset fingerprint 不一致")
    if _sha256_file(args.package_dir / "manifest.json") != protocol["identity"]["package_manifest_sha256"]:
        raise RuntimeError("Gate A package manifest SHA-256 不一致")
    fit_data = _load_panel(dataset, fit_indices, source=args.source, position_std=args.position_std)
    transfer_data = _load_panel(dataset, transfer_indices, source=args.source, position_std=args.position_std)
    tier4_data = _load_panel(dataset, tier4_indices, source=args.source, position_std=args.position_std)

    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    _validate_checkpoint(
        checkpoint,
        split=split,
        tier4_indices=tier4_indices,
        source_commit=source_commit,
    )
    model = UnifiedAutoEncoder(uae3d_model_args("official"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    state_before = _state_fingerprint(model.state_dict())
    fit_table = _extract_feature_table(model, fit_data, device=device, batch_size=args.batch_size)
    transfer_table = _extract_feature_table(model, transfer_data, device=device, batch_size=args.batch_size)
    tier4_table = _extract_feature_table(model, tier4_data, device=device, batch_size=args.batch_size)
    state_after = _state_fingerprint(model.state_dict())
    if state_before != state_after:
        raise RuntimeError("只读 Gate A 修改了 UAE-3D model state")

    probe_args = {
        "random_feature_dim": int(protocol["probe"]["nonlinear"]["random_feature_dim"]),
        "random_feature_seed": int(protocol["probe"]["nonlinear"]["seed"]),
        "l2": float(protocol["probe"]["l2"]),
    }
    result = {
        "source_model_unchanged": state_before == state_after,
        "all_probe_outputs_finite": True,
        "iid_transfer": {
            "linear": fit_existence_probe(fit_table, transfer_table, nonlinear=False, **probe_args),
            "nonlinear": fit_existence_probe(fit_table, transfer_table, nonlinear=True, **probe_args),
        },
        "tier4_oracle": {
            "linear": fit_existence_probe(tier4_table, tier4_table, nonlinear=False, **probe_args),
            "nonlinear": fit_existence_probe(tier4_table, tier4_table, nonlinear=True, **probe_args),
        },
    }
    result["all_probe_outputs_finite"] = _all_finite(result)
    checks = gate_a_checks(result, protocol["gate_thresholds"])
    passed = all(checks.values())
    status = (
        protocol["decision"]["pass_status"] if passed
        else protocol["decision"]["fail_status"]
    )
    report = {
        "schema_version": "uae3d-gate-a-report-v1",
        "status": status,
        "passed": passed,
        "purpose": protocol["purpose"],
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": _sha256_file(args.protocol),
            "fingerprint": protocol["protocol_fingerprint"],
            "gate_thresholds": protocol["gate_thresholds"],
            "probe": protocol["probe"],
            "panels": {
                name: {
                    "role": panel["role"],
                    "molecule_count": panel["molecule_count"],
                }
                for name, panel in protocol["panels"].items()
            },
            "test_used": False,
            "core_ood_used": False,
            "uae3d_weights_updated": False,
        },
        "identity": {
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_checkpoint_sha256": _sha256_file(source_checkpoint),
            "source_checkpoint_step": int(checkpoint["step"]),
            "source_model_state_fingerprint": state_before,
            "fit_feature_table_fingerprint": fit_table["fingerprint"],
            "transfer_feature_table_fingerprint": transfer_table["fingerprint"],
            "tier4_feature_table_fingerprint": tier4_table["fingerprint"],
            "dataset_fingerprint": protocol["identity"]["dataset_fingerprint"],
            "official_source_commit": source_commit,
        },
        "results": result,
        "checks": checks,
        "decision": {
            "run_optional_gate_b": passed,
            "gate_b_is_mainline_prerequisite": False,
            "start_graph_plus_target_pg_to_3d_regardless": True,
            "resume_old_uae_local_head_branch": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "rdkit": rdBase.rdkitVersion,
            "device": str(device),
            "seed": int(args.seed),
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
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--position-std", type=float, default=OFFICIAL_GEOM_POSITION_STD)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260821)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    nonlinear = report["results"]["iid_transfer"]["nonlinear"]
    compact = {
        "status": report["status"],
        "passed": report["passed"],
        "iid_transfer_auc": nonlinear["evaluation_roc_auc_all_pairs"],
        "iid_transfer": nonlinear["evaluation_at_unchanged_threshold"],
        "checks": report["checks"],
        "output": str(args.output),
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
