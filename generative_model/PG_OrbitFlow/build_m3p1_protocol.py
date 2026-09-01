"""Freeze the M3.1 generalized centralizer-automorphism Tier-32 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .c0_local_recovery import _sha256
from .m3_tier32_training import _model_setting
from .m3p1_model import AutomorphismSetQuotientICPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-protocol", type=Path, required=True)
    parser.add_argument("--m3-run-dir", type=Path, required=True)
    parser.add_argument("--representation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("generative_model/PG_OrbitFlow").resolve()
    m3_protocol_path = args.m3_protocol.resolve(); m3_run = args.m3_run_dir.resolve()
    m3_protocol = json.loads(m3_protocol_path.read_text(encoding="utf-8"))
    m3_report_path = m3_run / "report.json"; m3_checkpoint_path = m3_run / "last.pt"
    m3_report = json.loads(m3_report_path.read_text(encoding="utf-8"))
    audit_path = args.representation_audit.resolve(); audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if m3_report.get("status") != "FAIL_M3_PREDICTION_GATE_STOP_BEFORE_DECODER":
        raise RuntimeError("M3.1 requires frozen M3 v2 prediction failure")
    if audit.get("status") != "PASS_M3P1_GENERALIZED_AUTOMORPHISM_CONTRACT_GATE":
        raise RuntimeError("M3.1 representation Gate did not pass")
    settings = _model_setting(m3_protocol)
    model = AutomorphismSetQuotientICPredictor(**settings)
    import torch
    parent_state = torch.load(m3_checkpoint_path, map_location="cpu", weights_only=False)["model"]
    inheritance = model.initialize_from_m3_parent(parent_state)
    isolation = model.configure_trainable_parameters()
    settings["parameter_count_expected"] = sum(parameter.numel() for parameter in model.parameters())
    settings["trainable_parameter_count_expected"] = isolation["trainable_parameter_count"]
    sources = {
        name: root / filename
        for name, filename in {
            "runner": "m3p1_automorphism_training.py",
            "model": "m3p1_model.py",
            "graph_automorphism": "graph_automorphism.py",
            "orbit_ic_model": "orbit_ic_model.py",
            "solver": "m0_oracle_reconstruction.py",
            "kinematics": "orbit_kinematics.py",
            "parallel_decoder": "m3_tier32_training.py",
        }.items()
    }
    training = dict(m3_protocol["training"])
    training["torsion_assignment"] = (
        "one whole-molecule graph automorphism commuting with every Target_PG action"
    )
    protocol = {
        "schema_version": "pg-orbitflow-m3p1-automorphism-protocol-v1",
        "seed": int(m3_protocol["seed"]) + 100,
        "package_dir": m3_protocol["package_dir"],
        "canonical_manifest_sha256": m3_protocol["canonical_manifest_sha256"],
        "panel": m3_protocol["panel"],
        "model": settings,
        "parent": {
            "protocol": {"path": str(m3_protocol_path), "sha256": _sha256(m3_protocol_path)},
            "report": {"path": str(m3_report_path), "sha256": _sha256(m3_report_path)},
            "checkpoint": {"path": str(m3_checkpoint_path), "sha256": _sha256(m3_checkpoint_path)},
        },
        "representation_audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
        "implementation": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in sources.items()},
        "initialization": {
            "m3_v2_state_inherited": True,
            "automorphism_head_compatible_initialization": True,
            "legacy_terminal_head_frozen_and_superseded": True,
            "optimizer_reused": False,
            "inheritance": inheritance,
            "isolation": isolation,
        },
        "training": training,
        "decoder": m3_protocol["decoder"],
        "prediction_gate": m3_protocol["prediction_gate"],
        "reconstruction_gate": m3_protocol["reconstruction_gate"],
        "evaluation_semantics": {
            "torsion_assignment": "one whole-molecule graph automorphism commuting with every Target_PG action",
            "chemical_geometry_matching": "same frozen centralizer graph automorphism set",
            "raw_group_action_and_collision_metrics_preserved": True,
            "reference_coordinates_used_only_for_evaluation_matching": True,
        },
        "single_scientific_change": (
            "replace terminal-only 2/3-slot torsion sets with graph/action-only centralizer automorphism "
            "sets of size 2--6 and one globally coupled assignment"
        ),
        "isolation": m3_protocol["isolation"],
        "progression": {
            "if_passes": "freeze M3 Tier-32 and run selected-start reproducibility audit",
            "if_prediction_fails": "diagnose residual non-centralizer aliases without relaxing Gate",
            "if_reconstruction_fails": "diagnose decoder records without relaxing Gate",
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE_M3P1_PROTOCOL {args.output.resolve()}")


if __name__ == "__main__": main()
