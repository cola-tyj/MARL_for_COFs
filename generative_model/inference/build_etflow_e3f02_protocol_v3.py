"""Freeze v3 with D6h multi-action raw-geometry selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "generative_model/inference/etflow_e3f02_protocol_v3.json"
V2_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_protocol_v2.json"
CHECKPOINT = ROOT / "generative_model/checkpoints/etflow/drugs-o3.ckpt"
MANIFEST = ROOT / "generative_model/data/processed/v2/manifest.json"
EXTENDED_REPORT = ROOT / "generative_model/inference/graph_action_extended_audit_v2/report.json"
S4_REPORT = ROOT / "generative_model/runs/etflow_e3f02_v2_examples/s4_000048/point_group_report.json"
D6H_FAILURE = ROOT / "generative_model/runs/etflow_e3f02_v2_examples/d6h_000858_diagnosis/diagnosis_v2.json"
D6H_DIRECTION = ROOT / "generative_model/runs/etflow_e3f02_v2_examples/d6h_000858_diagnosis/action_candidates_64_opt8.json"
SOURCES = (
    "generative_model/inference/generate_etflow_symmetric_xyz.py",
    "generative_model/inference/generate_etflow_symmetric_xyz_v2.py",
    "generative_model/inference/generate_etflow_symmetric_xyz_v3.py",
    "generative_model/inference/audit_etflow_e3f02_point_group.py",
    "generative_model/inference/build_etflow_e3f02_protocol_v3.py",
    "generative_model/tests/test_etflow_e3f02_inference_v3.py",
    "generative_model/symmetry/graph_action.py",
    "generative_model/symmetry/graph_action_extended.py",
    "generative_model/symmetry/graph_action_candidates.py",
    "generative_model/conformer/etflow_bridge.py",
    "generative_model/conformer/etflow_ef1_projection.py",
    "generative_model/optimization/orbit_force_field_repulsion.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: dict) -> str:
    payload = dict(value)
    payload.pop("protocol_fingerprint", None)
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    extended = json.loads(EXTENDED_REPORT.read_text(encoding="utf-8"))
    s4 = json.loads(S4_REPORT.read_text(encoding="utf-8"))
    failure = json.loads(D6H_FAILURE.read_text(encoding="utf-8"))
    direction = json.loads(D6H_DIRECTION.read_text(encoding="utf-8"))
    if not extended["passed"] or not s4["passed"]:
        raise RuntimeError("v3 requires extended-action and S4 pilot PASS")
    if failure["passed_candidate_count"] != 0:
        raise RuntimeError("v3 requires preserved single-action D6h failure")
    optimized = direction["optimized_top"]
    if len(optimized) != 8 or not all(
        item["success"] and item["record"]["passed_geometry_gate"]
        for item in optimized
    ):
        raise RuntimeError("v3 requires 8/8 multi-action D6h direction PASS")
    protocol = {
        "schema_version": "etflow-e3f02-inference-protocol-v3",
        "status": "FROZEN_D6H_MULTI_ACTION_BEFORE_FORMAL_PILOT",
        "route": "ET-Flow prior + graph-only action search + E3 projection + F0.2",
        "supported_target_pgs": ["C2", "C3", "S4", "D6h"],
        "etflow": {
            "model_name": "drugs-o3",
            "candidate_count": 4,
            "n_timesteps": 50,
            "sampler_type": "ODE",
            "role": "learned initial-conformer prior only",
        },
        "d6h_action_selection": {
            "maximum_actions": 64,
            "maximum_search_matches": 100_000,
            "combination_count": 256,
            "projection_ranking": [
                "pair_count_below_0.6 ascending",
                "pair_count_below_0.8 ascending",
                "minimum_pair_distance descending",
                "projection_rmsd ascending",
                "raw_candidate_id ascending",
                "action_candidate_id ascending",
            ],
            "optimize_top_combinations": 8,
            "reference_xyz_used_for_ranking": False,
            "stored_action_used_for_ranking": False,
        },
        "projection": {
            "method": "24-orientation search + one Reynolds projection",
            "match_radius_of_gyration": True,
            "hard_symmetry_projection": True,
        },
        "optimizer": {
            "method": "L-BFGS-B",
            "primary_force_field": "UFF",
            "repulsion_cutoff_angstrom": 0.8,
            "repulsion_force_constant_kcal_mol_angstrom2": 2000.0,
            "maximum_iterations": 500,
            "function_tolerance": 1e-9,
            "gradient_tolerance": 1e-5,
            "orbit_subspace_constrained": True,
        },
        "gate_thresholds": {
            "acceptable_gradient_norm_max": 1.0,
            "minimum_pair_distance_angstrom": 0.6,
            "maximum_operation_error_angstrom": 1e-5,
        },
        "candidate_selection": (
            "minimum final UFF+repulsion objective among strict geometry-passing "
            "optimized combinations, then preselection rank"
        ),
        "scope": {
            "known_canonical_graph": True,
            "requested_target_pg": True,
            "reference_xyz_used": False,
            "stored_symmetry_action_used": False,
            "etkdg_used": False,
            "etflow_used_as_initial_conformer_prior": True,
            "hard_symmetry_projection_used": True,
            "orbit_constrained_force_field_used": True,
            "raw_learned_target_pg_claim": False,
        },
        "decision": {
            "pass_status": "PASS_ETFLOW_E3F02_V3_D6H_ADVANCE_TO_RARE_TARGET_PANEL",
            "fail_status": "FAIL_ETFLOW_E3F02_V3_D6H_STOP_AND_DIAGNOSE",
        },
        "prerequisite_evidence": {
            "v2_protocol_sha256": _sha256(V2_PROTOCOL),
            "extended_action_report_sha256": _sha256(EXTENDED_REPORT),
            "s4_point_group_report_sha256": _sha256(S4_REPORT),
            "single_action_d6h_failure_sha256": _sha256(D6H_FAILURE),
            "multi_action_direction_report_sha256": _sha256(D6H_DIRECTION),
        },
        "identity": {
            "checkpoint_sha256": _sha256(CHECKPOINT),
            "manifest_sha256": _sha256(MANIFEST),
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "symmetry_protocol": manifest["symmetry_protocol"],
            "source_sha256": {
                relative: _sha256(ROOT / relative) for relative in SOURCES
            },
        },
    }
    protocol["protocol_fingerprint"] = _fingerprint(protocol)
    OUTPUT.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"FROZEN protocol={OUTPUT}")
    print(f"sha256={_sha256(OUTPUT)}")
    print(f"fingerprint={protocol['protocol_fingerprint']}")


if __name__ == "__main__":
    main()
