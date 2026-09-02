"""Freeze exposure-equivalent M3.2 full-panel torsion-head refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..c0_local_recovery import _sha256
from ..m3_tier32_training import _model_setting
from ..m3p1_model import AutomorphismSetQuotientICPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--parent-protocol", type=Path, required=True); parser.add_argument("--parent-run-dir", type=Path, required=True); parser.add_argument("--readiness-audit", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    root = Path("generative_model/PG_OrbitFlow").resolve(); parent_protocol_path = args.parent_protocol.resolve(); parent_run = args.parent_run_dir.resolve(); parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8")); parent_report_path = parent_run / "report.json"; parent_checkpoint_path = parent_run / "last.pt"; parent_report = json.loads(parent_report_path.read_text(encoding="utf-8")); readiness_path = args.readiness_audit.resolve(); readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if parent_report.get("status") != "FAIL_M3P1_PREDICTION_GATE_STOP_BEFORE_DECODER" or readiness.get("status") != "PASS_M3P2_FULL_BATCH_REFINEMENT_GATE": raise RuntimeError("M3.2 prerequisites did not pass")
    model = AutomorphismSetQuotientICPredictor(**_model_setting(parent_protocol)); import torch; model.load_state_dict(torch.load(parent_checkpoint_path, map_location="cpu", weights_only=False)["model"], strict=True)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for module in (model.parent.base.torsion_head, model.automorphism_head):
        for parameter in module.parameters(): parameter.requires_grad_(True)
    trainable_keys = [name for name, parameter in model.named_parameters() if parameter.requires_grad]; trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    settings = _model_setting(parent_protocol); settings["parameter_count_expected"] = sum(parameter.numel() for parameter in model.parameters()); settings["trainable_parameter_count_expected"] = trainable_count
    sources = {name: root / filename for name, filename in {"runner": "m3p2_torsion_refinement.py", "model": "m3p1_model.py", "graph_automorphism": "graph_automorphism.py", "orbit_ic_model": "orbit_ic_model.py", "solver": "m0_oracle_reconstruction.py", "kinematics": "orbit_kinematics.py", "parallel_decoder": "m3_tier32_training.py", "m3p1_metrics": "m3p1_automorphism_training.py"}.items()}
    protocol = {"schema_version": "pg-orbitflow-m3p2-refinement-protocol-v1", "seed": int(parent_protocol["seed"]) + 100, "package_dir": parent_protocol["package_dir"], "canonical_manifest_sha256": parent_protocol["canonical_manifest_sha256"], "panel": parent_protocol["panel"], "model": settings, "parent": {"protocol": {"path": str(parent_protocol_path), "sha256": _sha256(parent_protocol_path)}, "report": {"path": str(parent_report_path), "sha256": _sha256(parent_report_path)}, "checkpoint": {"path": str(parent_checkpoint_path), "sha256": _sha256(parent_checkpoint_path)}}, "readiness_audit": {"path": str(readiness_path), "sha256": _sha256(readiness_path)}, "implementation": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in sources.items()}, "initialization": {"complete_m3p1_final_state_inherited": True, "optimizer_reused": False, "frozen_feature_cache": True, "trainable_keys": trainable_keys, "trainable_parameter_count": trainable_count}, "training": {"optimizer_steps": 512, "batch_size_molecules": 32, "molecule_exposures_each": 512, "molecule_exposures_total": 16384, "learning_rate": 1e-3, "weight_decay": 0.0, "gradient_clip_norm": 5.0, "checkpoint_every": 128, "loss": "full-panel mean globally automorphism-aligned circular torsion loss", "trainable_modules": ["parent.base.torsion_head", "automorphism_head"]}, "decoder": parent_protocol["decoder"], "prediction_gate": parent_protocol["prediction_gate"], "reconstruction_gate": parent_protocol["reconstruction_gate"], "evaluation_semantics": parent_protocol["evaluation_semantics"], "single_scientific_change": "replace stochastic 2+2 minibatches with one deterministic 16+16 full-panel torsion-head batch while preserving exactly 512 exposures per molecule and freezing already-passed features/bond/angle", "isolation": parent_protocol["isolation"], "progression": {"if_passes": "run decoder and M3 selected-start reproducibility", "if_prediction_fails": "diagnose head capacity without changing Gate", "if_reconstruction_fails": "diagnose decoder without changing Gate"}}
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True); args.output.resolve().write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(f"WROTE_M3P2_PROTOCOL {args.output.resolve()}")


if __name__ == "__main__": main()
