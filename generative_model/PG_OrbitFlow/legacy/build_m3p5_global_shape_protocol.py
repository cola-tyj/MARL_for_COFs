"""Freeze M3.5 permutation-invariant global-shape head training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..c0_local_recovery import _sha256
from ..global_shape_model import GlobalShapeAugmentedPredictor
from ..m3_tier32_training import _model_setting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--pair-run-dir", type=Path, required=True)
    parser.add_argument("--failure-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path("generative_model/PG_OrbitFlow").resolve()
    parent_protocol_path = args.parent_protocol.resolve()
    parent_run = args.parent_run_dir.resolve()
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    parent_report_path = parent_run / "report.json"
    parent_checkpoint_path = parent_run / "last.pt"
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    pair_report_path = args.pair_run_dir.resolve() / "report.json"
    pair_report = json.loads(pair_report_path.read_text(encoding="utf-8"))
    failure_path = args.failure_audit.resolve()
    if parent_report.get("status") != "PASS_M3P3_PREDICTION_GATE_ADVANCE_TO_DECODER":
        raise RuntimeError("M3.5 requires the passed M3.3 predictor")
    if pair_report.get("status") != "FAIL_M3P4_GLOBAL_PAIR_GATE_STOP_BEFORE_DECODER":
        raise RuntimeError("M3.5 requires the frozen M3.4 representation failure")

    model = GlobalShapeAugmentedPredictor(
        quantile_count=16, **_model_setting(parent_protocol)
    )
    state = torch.load(parent_checkpoint_path, map_location="cpu", weights_only=False)
    model.initialize_from_m3p3(state["model"])
    trainable_keys = model.configure_shape_head_only()
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    settings = _model_setting(parent_protocol)
    settings["parameter_count_expected"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    settings["trainable_parameter_count_expected"] = trainable_count
    settings["global_shape_quantile_count"] = 16

    sources = {
        "runner": root / "m3p5_global_shape_training.py",
        "model": root / "global_shape_model.py",
        "parent_model": root / "m3p1_model.py",
        "orbit_ic_model": root / "orbit_ic_model.py",
    }
    protocol = {
        "schema_version": "pg-orbitflow-m3p5-global-shape-protocol-v1",
        "seed": int(parent_protocol["seed"]) + 501,
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
            "report": {
                "path": str(parent_report_path),
                "sha256": _sha256(parent_report_path),
            },
            "checkpoint": {
                "path": str(parent_checkpoint_path),
                "sha256": _sha256(parent_checkpoint_path),
            },
        },
        "m3p4_failure": {
            "report": {
                "path": str(pair_report_path),
                "sha256": _sha256(pair_report_path),
            },
            "cause": "310 exact pair-row feature aliases among 1070 targets",
        },
        "decoder_failure_audit": {
            "path": str(failure_path),
            "sha256": _sha256(failure_path),
        },
        "implementation": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "initialization": {
            "complete_m3p3_parent_inherited_and_frozen": True,
            "global_shape_head_seed_initialized": True,
            "trainable_keys": trainable_keys,
            "trainable_parameter_count": trainable_count,
        },
        "global_shape_contract": {
            "minimum_graph_distance": 4,
            "quantile_count": 16,
            "quantile_levels": "uniform inclusive [0,1]",
            "permutation_invariant": True,
        },
        "training": {
            "optimizer_steps": 2048,
            "batch_size_molecules": 32,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "checkpoint_every": 512,
            "mean_molecule_mse_weight": 1.0,
            "worst_molecule_mse_weight": 1.0,
        },
        "shape_prediction_gate": {
            "global_shape_quantile_mae_max_angstrom": 0.03,
            "c2_and_c3_each_pass": True,
            "parent_prediction_gate_remains_passed": True,
            "exact_embedding_target_conflicts": 0,
        },
        "prediction_gate": parent_protocol["prediction_gate"],
        "decoder": parent_protocol["decoder"],
        "reconstruction_gate": parent_protocol["reconstruction_gate"],
        "evaluation_semantics": parent_protocol["evaluation_semantics"],
        "progression": {
            "if_passes": "audit shape-aware selection on package 1054, then run Tier-32",
            "if_fails": "stop and audit graph-embedding capacity",
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_M3P5_GLOBAL_SHAPE_PROTOCOL {output}")


if __name__ == "__main__":
    main()
