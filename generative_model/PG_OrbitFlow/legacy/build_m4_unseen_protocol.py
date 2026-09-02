"""Freeze the M4 read-only unseen IID prediction protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..c0_local_recovery import _sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-protocol", type=Path, required=True)
    parser.add_argument("--m3-report", type=Path, required=True)
    parser.add_argument("--m3-reproducibility", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path("generative_model/PG_OrbitFlow").resolve()
    m3_protocol_path = args.m3_protocol.resolve()
    m3_report_path = args.m3_report.resolve()
    m3_reproduction_path = args.m3_reproducibility.resolve()
    panel_path = args.panel.resolve()
    audit_path = args.panel_audit.resolve()
    m3 = json.loads(m3_protocol_path.read_text(encoding="utf-8"))
    m3_report = json.loads(m3_report_path.read_text(encoding="utf-8"))
    reproduction = json.loads(m3_reproduction_path.read_text(encoding="utf-8"))
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if m3_report.get("status") != "PASS_M3_TIER32" or not m3_report.get("passed"):
        raise RuntimeError("M4 requires passed M3")
    if reproduction.get("status") != "PASS_M3_REPRODUCIBILITY" or not reproduction.get("passed"):
        raise RuntimeError("M4 requires reproducible M3")
    if audit.get("status") != "PASS_M4_UNSEEN_IID_PANEL_AUDIT" or not audit.get("passed"):
        raise RuntimeError("M4 panel audit did not pass")

    sources = {
        "runner": root / "evaluate_m4_unseen_prediction.py",
        "split_loader": root / "unseen_iid.py",
        "model": root / "global_shape_fingerprint_model.py",
        "shape_contract": root / "global_shape_model.py",
        "graph_automorphism": root / "graph_automorphism.py",
        "prediction_metrics": root / "m3p1_automorphism_training.py",
    }
    protocol = {
        "schema_version": "pg-orbitflow-m4-unseen-prediction-protocol-v1",
        "seed": int(m3["seed"]),
        "package_dir": panel["package_dir"],
        "canonical_manifest_sha256": panel["canonical_manifest_sha256"],
        "panel": panel,
        "model": m3["model"],
        "predictor": m3["predictor"],
        "global_shape_contract": m3["global_shape_contract"],
        "prediction_gate": m3["prediction_gate"],
        "shape_prediction_gate": {
            "global_shape_quantile_mae_max_angstrom": 0.03,
            "c2_and_c3_each_pass": True,
        },
        "m3_parent": {
            "protocol": {"path": str(m3_protocol_path), "sha256": _sha256(m3_protocol_path)},
            "report": {"path": str(m3_report_path), "sha256": _sha256(m3_report_path)},
            "reproducibility": {"path": str(m3_reproduction_path), "sha256": _sha256(m3_reproduction_path)},
        },
        "panel_audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
        "implementation": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "scope": {
            "training_performed": False,
            "checkpoint_frozen": True,
            "iid_train_used": False,
            "iid_validation_used_read_only": True,
            "iid_test_used": False,
            "core_ood_used": False,
            "target_coordinates_used_as_model_input": False,
            "target_coordinates_used_for_metrics_only": True,
            "hard_projection_used": False,
            "f0p2_used": False,
        },
        "decision_rule": {
            "if_prediction_and_shape_pass": "advance to frozen unseen decoder Gate, then dataset-level training",
            "if_either_fails": "record closed-panel generalization failure and proceed to dataset-level training without changing M3",
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_M4_UNSEEN_PROTOCOL {output}")


if __name__ == "__main__":
    main()
