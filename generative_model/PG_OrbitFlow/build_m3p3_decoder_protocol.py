"""Freeze the unchanged Tier-32 decoder Gate for the passed M3.3 predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .c0_local_recovery import _sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--training-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path("generative_model/PG_OrbitFlow").resolve()
    training_protocol_path = args.training_protocol.resolve()
    training_run = args.training_run_dir.resolve()
    training_protocol = json.loads(
        training_protocol_path.read_text(encoding="utf-8")
    )
    report_path = training_run / "report.json"
    checkpoint_path = training_run / "last.pt"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_M3P3_PREDICTION_GATE_ADVANCE_TO_DECODER":
        raise RuntimeError("M3.3 prediction Gate did not pass")

    sources = {
        "runner": root / "m3p3_decoder_gate.py",
        "model": root / "m3p1_model.py",
        "graph_automorphism": root / "graph_automorphism.py",
        "solver": root / "m0_oracle_reconstruction.py",
        "kinematics": root / "orbit_kinematics.py",
        "decoder_worker": root / "m3_tier32_training.py",
        "matching_metrics": root / "m3p1_automorphism_training.py",
    }
    protocol = {
        "schema_version": "pg-orbitflow-m3p3-decoder-protocol-v1",
        "seed": int(training_protocol["seed"]),
        "package_dir": training_protocol["package_dir"],
        "canonical_manifest_sha256": training_protocol[
            "canonical_manifest_sha256"
        ],
        "panel": training_protocol["panel"],
        "model": training_protocol["model"],
        "predictor": {
            "training_protocol": {
                "path": str(training_protocol_path),
                "sha256": _sha256(training_protocol_path),
            },
            "prediction_report": {
                "path": str(report_path),
                "sha256": _sha256(report_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
            },
        },
        "implementation": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "prediction_gate": training_protocol["prediction_gate"],
        "decoder": training_protocol["decoder"],
        "reconstruction_gate": training_protocol["reconstruction_gate"],
        "evaluation_semantics": training_protocol["evaluation_semantics"],
        "scope": {
            "training_performed": False,
            "oracle_internal_coordinates_used": False,
            "target_coordinates_used_by_decoder": False,
            "hard_projection_used": False,
            "f0p2_used": False,
            "centralizer_automorphism_used_only_for_evaluation": True,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_M3P3_DECODER_PROTOCOL {output}")


if __name__ == "__main__":
    main()
