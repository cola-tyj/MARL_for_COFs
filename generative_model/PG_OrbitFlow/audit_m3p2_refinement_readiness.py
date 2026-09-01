"""Audit whether M3.1 is eligible for exposure-equivalent full-batch torsion refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .graph_automorphism import build_graph_automorphism_contract
from .m3_tier32_training import _model_setting
from .m3p1_model import AutomorphismSetQuotientICPredictor
from .orbit_ic_model import build_orbit_ic_example
from .overfit import _panel_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); import torch
    protocol_path = args.protocol.resolve(); run_dir = args.run_dir.resolve(); protocol = json.loads(protocol_path.read_text(encoding="utf-8")); report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "FAIL_M3P1_PREDICTION_GATE_STOP_BEFORE_DECODER": raise RuntimeError("M3.2 readiness requires frozen M3.1 prediction failure")
    model = AutomorphismSetQuotientICPredictor(**_model_setting(protocol)); checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False); model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}; signatures = {}; conflicts = []
    for sample in _panel_samples(protocol, tier):
        geometry = build_geometry_contract(sample); graph, targets = build_orbit_ic_example(sample, geometry); automorphism = build_graph_automorphism_contract(sample, geometry)
        with torch.no_grad(): model(graph, automorphism, device="cpu")
        features = model.parent._captured_torsion_input.detach().numpy()
        grouped = {orbit for group in automorphism.groups for orbit in group}
        for orbit, (feature, target) in enumerate(zip(features, targets.torsion_target_sincos, strict=True)):
            if orbit in grouped: continue
            key = feature.tobytes()
            if key in signatures:
                previous = signatures[key]; cosine = float(np.dot(previous["target"], target)); sine = float(previous["target"][0] * target[1] - previous["target"][1] * target[0]); error = abs(float(np.rad2deg(np.arctan2(sine, cosine))))
                if error > 1.0: conflicts.append({"left": [previous["package_index"], previous["orbit"]], "right": [sample.package_index, orbit], "target_error_degrees": error})
            else: signatures[key] = {"package_index": int(sample.package_index), "orbit": int(orbit), "target": np.asarray(target)}
    gate = report["prediction_gate"]
    checks = {
        "m3p1_bond_gate_passed": bool(gate["checks"]["bond_orbit_mae_max_angstrom"]),
        "m3p1_angle_gate_passed": bool(gate["checks"]["angle_orbit_mae_max_degrees"]),
        "m3p1_torsion_gate_failed": not bool(gate["checks"]["torsion_orbit_circular_mae_max_degrees"]),
        "no_exact_ungrouped_torsion_input_conflicts": len(conflicts) == 0,
        "exposure_budget_preserved": 4096 * 4 == 512 * 32 == 16384,
        "panel_size_exact": len(protocol["panel"]["records"]) == 32,
    }
    passed = all(checks.values()); output = {"schema_version": "pg-orbitflow-m3p2-refinement-readiness-v1", "status": "PASS_M3P2_FULL_BATCH_REFINEMENT_GATE" if passed else "FAIL_M3P2_FULL_BATCH_REFINEMENT_GATE", "passed": passed, "checks": checks, "exact_ungrouped_conflicts": conflicts, "refinement": {"parent_optimizer_steps": 4096, "parent_batch_size": 4, "refinement_optimizer_steps": 512, "refinement_batch_size": 32, "molecule_exposures_each": 512, "total_molecule_exposures": 16384, "trainable_modules": ["parent.base.torsion_head", "automorphism_head"]}, "identity": {"protocol_sha256": _sha256(protocol_path), "report_sha256": _sha256(run_dir / "report.json"), "checkpoint_sha256": _sha256(run_dir / "last.pt"), "auditor_sha256": _sha256(Path(__file__).resolve())}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(f"{output['status']} report={args.output.resolve()}")
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
