"""Freeze M3.3 Gate-aligned worst-molecule torsion refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .c0_local_recovery import _sha256
from .m3_tier32_training import _model_setting
from .m3p1_model import AutomorphismSetQuotientICPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--readiness-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path("generative_model/PG_OrbitFlow").resolve()
    parent_protocol_path = args.parent_protocol.resolve()
    parent_run = args.parent_run_dir.resolve()
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    parent_report_path = parent_run / "report.json"
    parent_checkpoint_path = parent_run / "last.pt"
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    readiness_path = args.readiness_audit.resolve()
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if parent_report.get("status") != "FAIL_M3P2_PREDICTION_GATE_STOP_BEFORE_DECODER":
        raise RuntimeError("M3.3 requires the frozen M3.2 prediction failure")
    if readiness.get("status") != "PASS_M3P3_WORST_MOLECULE_REFINEMENT_GATE":
        raise RuntimeError("M3.3 readiness audit did not pass")

    model = AutomorphismSetQuotientICPredictor(**_model_setting(parent_protocol))
    state = torch.load(parent_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.parent.base.torsion_head, model.automorphism_head):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    trainable_keys = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    settings = _model_setting(parent_protocol)
    settings["parameter_count_expected"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    settings["trainable_parameter_count_expected"] = trainable_count

    sources = {
        "runner": root / "m3p3_worst_molecule_refinement.py",
        "model": root / "m3p1_model.py",
        "graph_automorphism": root / "graph_automorphism.py",
        "orbit_ic_model": root / "orbit_ic_model.py",
        "metrics": root / "m3p1_automorphism_training.py",
        "cached_forward": root / "m3p2_torsion_refinement.py",
    }
    protocol = {
        "schema_version": "pg-orbitflow-m3p3-refinement-protocol-v1",
        "seed": int(parent_protocol["seed"]) + 101,
        "package_dir": parent_protocol["package_dir"],
        "canonical_manifest_sha256": parent_protocol["canonical_manifest_sha256"],
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
        "readiness_audit": {
            "path": str(readiness_path),
            "sha256": _sha256(readiness_path),
        },
        "implementation": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "initialization": {
            "complete_m3p2_final_state_inherited": True,
            "optimizer_reused": False,
            "frozen_feature_cache": True,
            "trainable_keys": trainable_keys,
            "trainable_parameter_count": trainable_count,
        },
        "training": {
            "optimizer_steps": 512,
            "batch_size_molecules": 32,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "checkpoint_every": 128,
            "mean_molecule_torsion_weight": 1.0,
            "worst_molecule_torsion_weight": 1.0,
            "worst_molecule_nonplanar_weight": 1.0,
            "loss": (
                "global-automorphism-aligned mean molecule torsion loss plus "
                "worst molecule torsion and worst molecule nonplanar losses"
            ),
            "trainable_modules": [
                "parent.base.torsion_head",
                "automorphism_head",
            ],
        },
        "prediction_gate": parent_protocol["prediction_gate"],
        "reconstruction_gate": parent_protocol["reconstruction_gate"],
        "decoder": parent_protocol["decoder"],
        "evaluation_semantics": parent_protocol["evaluation_semantics"],
        "single_scientific_change": (
            "add losses aligned to the maximum-over-molecules torsion and "
            "nonplanar Gate statistics; retain the full-panel mean anchor"
        ),
        "isolation": parent_protocol["isolation"],
        "progression": {
            "if_prediction_passes": "freeze checkpoint and run unchanged M3 decoder Gate",
            "if_prediction_fails": "stop and audit shared-head capacity",
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_M3P3_PROTOCOL {output}")


if __name__ == "__main__":
    main()
