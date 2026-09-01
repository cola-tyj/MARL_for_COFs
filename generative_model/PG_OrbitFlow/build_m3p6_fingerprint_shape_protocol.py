"""Freeze M3.6 WL-fingerprint-conditioned global-shape training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .c0_local_recovery import _sha256
from .global_shape_fingerprint_model import FingerprintGlobalShapePredictor
from .m3_tier32_training import _model_setting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--shape-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("generative_model/PG_OrbitFlow").resolve()
    parent_protocol_path = args.parent_protocol.resolve()
    parent_run = args.parent_run_dir.resolve()
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    parent_report_path = parent_run / "report.json"
    parent_checkpoint_path = parent_run / "last.pt"
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    shape_report_path = args.shape_run_dir.resolve() / "report.json"
    shape_report = json.loads(shape_report_path.read_text(encoding="utf-8"))
    if parent_report.get("status") != "PASS_M3P3_PREDICTION_GATE_ADVANCE_TO_DECODER":
        raise RuntimeError("M3.6 requires the passed M3.3 predictor")
    if shape_report.get("status") != "FAIL_M3P5B_GLOBAL_SHAPE_GATE_STOP_BRANCH":
        raise RuntimeError("M3.6 requires the frozen pooled-shape capacity failure")
    model = FingerprintGlobalShapePredictor(
        quantile_count=16,
        fingerprint_bits=512,
        **_model_setting(parent_protocol),
    )
    state = torch.load(parent_checkpoint_path, map_location="cpu", weights_only=False)
    model.initialize_from_m3p3(state["model"])
    trainable_keys = model.configure_shape_head_only()
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    settings = _model_setting(parent_protocol)
    settings.update(
        {
            "parameter_count_expected": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "trainable_parameter_count_expected": trainable_count,
            "global_shape_quantile_count": 16,
            "wl_fingerprint_bits": 512,
            "wl_fingerprint_radius": 4,
        }
    )
    sources = {
        "runner": root / "m3p6_fingerprint_shape_training.py",
        "model": root / "global_shape_fingerprint_model.py",
        "shape_contract": root / "global_shape_model.py",
        "parent_model": root / "m3p1_model.py",
    }
    protocol = {
        "schema_version": "pg-orbitflow-m3p6-fingerprint-shape-protocol-v1",
        "seed": int(parent_protocol["seed"]) + 601,
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
        "m3p5b_failure": {
            "path": str(shape_report_path),
            "sha256": _sha256(shape_report_path),
            "cause": "frozen local-IC pooled embedding lacks global topology capacity",
        },
        "implementation": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "initialization": {
            "complete_m3p3_parent_inherited_and_frozen": True,
            "shape_head_seed_initialized": True,
            "trainable_keys": trainable_keys,
            "trainable_parameter_count": trainable_count,
        },
        "global_shape_contract": {
            "minimum_graph_distance": 4,
            "quantile_count": 16,
            "fingerprint": "normalized counted WL radius 0..4, SHA-256 folded to 512 bits",
            "atom_order_invariant": True,
            "target_coordinates_used_only_for_training": True,
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
            "exact_input_target_conflicts": 0,
        },
        "prediction_gate": parent_protocol["prediction_gate"],
        "decoder": parent_protocol["decoder"],
        "reconstruction_gate": parent_protocol["reconstruction_gate"],
        "evaluation_semantics": parent_protocol["evaluation_semantics"],
        "progression": {
            "if_passes": "run package-1054 selection audit then formal Tier-32 decoder",
            "if_fails": "stop and reconsider global-shape target",
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_M3P6_FINGERPRINT_SHAPE_PROTOCOL {output}")


if __name__ == "__main__":
    main()
