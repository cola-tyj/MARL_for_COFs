"""Freeze M3.4 long-range pair-orbit head training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..c0_local_recovery import _sha256
from ..global_pair_model import GlobalPairAugmentedPredictor
from ..m3_tier32_training import _model_setting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--failure-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path("generative_model/PG_OrbitFlow").resolve()
    parent_protocol_path = args.parent_protocol.resolve()
    parent_run = args.parent_run_dir.resolve()
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    report_path = parent_run / "report.json"
    checkpoint_path = parent_run / "last.pt"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audit_path = args.failure_audit.resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_M3P3_PREDICTION_GATE_ADVANCE_TO_DECODER":
        raise RuntimeError("M3.4 requires the passed M3.3 predictor")
    if audit.get("status") != "DIAGNOSED_M3P3_DECODER_ENERGY_RANKING_FAILURE":
        raise RuntimeError("M3.4 requires the frozen decoder failure diagnosis")

    model = GlobalPairAugmentedPredictor(**_model_setting(parent_protocol))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.initialize_from_m3p3(state["model"])
    trainable_keys = model.configure_pair_head_only()
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    settings = _model_setting(parent_protocol)
    settings["parameter_count_expected"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    settings["trainable_parameter_count_expected"] = trainable_count

    sources = {
        "runner": root / "m3p4_global_pair_training.py",
        "model": root / "global_pair_model.py",
        "parent_model": root / "m3p1_model.py",
        "graph_automorphism": root / "graph_automorphism.py",
        "orbit_ic_model": root / "orbit_ic_model.py",
    }
    protocol = {
        "schema_version": "pg-orbitflow-m3p4-global-pair-protocol-v1",
        "seed": int(parent_protocol["seed"]) + 401,
        "package_dir": parent_protocol["package_dir"],
        "canonical_manifest_sha256": parent_protocol[
            "canonical_manifest_sha256"
        ],
        "panel": parent_protocol["panel"],
        "model": settings,
        "parent": {
            "protocol": {
                "path": str(parent_protocol_path),
                "sha256": _sha256(parent_protocol_path),
            },
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
            },
        },
        "failure_audit": {
            "path": str(audit_path),
            "sha256": _sha256(audit_path),
        },
        "implementation": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "initialization": {
            "complete_m3p3_parent_inherited_and_frozen": True,
            "global_pair_head_initialized_from_bond_head": True,
            "trainable_keys": trainable_keys,
            "trainable_parameter_count": trainable_count,
        },
        "global_pair_contract": {
            "minimum_graph_distance": 4,
            "target": "Target-PG orbit mean Euclidean distance",
            "graph_only_input": True,
            "coordinate_target_used_only_for_training": True,
        },
        "training": {
            "optimizer_steps": 1024,
            "batch_size_molecules": 32,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "checkpoint_every": 256,
            "mean_molecule_mse_weight": 1.0,
            "worst_molecule_mse_weight": 1.0,
        },
        "pair_prediction_gate": {
            "global_pair_orbit_mae_max_angstrom": 0.03,
            "c2_and_c3_each_pass": True,
            "parent_prediction_gate_remains_passed": True,
        },
        "prediction_gate": parent_protocol["prediction_gate"],
        "decoder": parent_protocol["decoder"],
        "reconstruction_gate": parent_protocol["reconstruction_gate"],
        "evaluation_semantics": parent_protocol["evaluation_semantics"],
        "progression": {
            "if_passes": "run symmetric decoder with learned global-pair energy",
            "if_fails": "stop before decoder and audit pair-head representation",
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_M3P4_GLOBAL_PAIR_PROTOCOL {output}")


if __name__ == "__main__":
    main()
