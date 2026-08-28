"""Freeze the evidence boundary for the four-target known-graph-to-XYZ route."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "generative_model/smoke/reports/etflow_e3f02_completion_v1.json"
ARTIFACTS = {
    "c2_c3_protocol": (
        "generative_model/inference/etflow_e3f02_iidtest_protocol_v1.json",
        "6dee96f0092747aaa6bf18660949652aa56718416976fa810d3212467676f86c",
    ),
    "c2_c3_geometry": (
        "generative_model/runs/etflow_e3f02_iidtest_v1/geometry_report.json",
        "5c0efdedfef123cfcf93846a15c440a9a571c19978761c8d589f28149111e587",
    ),
    "c2_c3_point_group": (
        "generative_model/runs/etflow_e3f02_iidtest_v1/point_group_report.json",
        "6dcada7369e3d3b61db249310d9385cb944f7ce6790beee3404585edb32adf0a",
    ),
    "s4_d6h_protocol": (
        "generative_model/inference/etflow_e3f02_rare_targets_protocol_v3.json",
        "290b2127b079da2a27432e0f7ab225353791a0fc6e06b2ec7c63cee1eb73e2a1",
    ),
    "s4_d6h_geometry": (
        "generative_model/runs/etflow_e3f02_rare_targets_v3/geometry_report.json",
        "04d2d6224f6a16d688a34ce15916eb6c87d87b8c334a143d99bc5bc8c3707df9",
    ),
    "s4_d6h_point_group": (
        "generative_model/runs/etflow_e3f02_rare_targets_v3/point_group_report.json",
        "76e94776fb41c0d8ae6410be695118db4319fd20d01754f9882357096af91cd3",
    ),
    "inference_protocol": (
        "generative_model/inference/etflow_e3f02_protocol_v5.json",
        "cfcb872cf4d8902f2ccdba1bd2738b6476eba27b7a7be9e5108e7e0d4a2678d2",
    ),
    "deliverable_xyz": (
        "generative_model/runs/etflow_e3f02_v5_examples/s4_001102/final.xyz",
        "3cb94ecbe4d6137338ed4549df4859496488b3b9a1695dc7aee2c485473ac207",
    ),
    "deliverable_geometry": (
        "generative_model/runs/etflow_e3f02_v5_examples/s4_001102/report.json",
        "288e11f22b5d3bcf7e525afc1db313c57858c51dc160424c9be7b0e47ba8d40c",
    ),
    "deliverable_point_group": (
        "generative_model/runs/etflow_e3f02_v5_examples/s4_001102/point_group_report.json",
        "b676c44c5747a7d04002ed71f1aa5a0f024ea4024b8ffb02a02326f6f422f7b5",
    ),
    "official_checkpoint": (
        "generative_model/checkpoints/etflow/drugs-o3.ckpt",
        "a24ae9a1fed2708696929308ed1dc10ab167fd66a2d51c44a4afb6c11badccb2",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(key: str) -> dict[str, Any]:
    return json.loads((ROOT / ARTIFACTS[key][0]).read_text(encoding="utf-8"))


def _xyz_is_valid(path: Path) -> tuple[bool, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        return False, 0
    try:
        count = int(lines[0].strip())
    except ValueError:
        return False, 0
    return len(lines) == count + 2 and all(len(line.split()) == 4 for line in lines[2:]), count


def main() -> None:
    artifact_hashes = {
        key: {"path": relative, "sha256": _sha256(ROOT / relative)}
        for key, (relative, _) in ARTIFACTS.items()
    }
    identity_ok = all(
        artifact_hashes[key]["sha256"] == expected
        for key, (_, expected) in ARTIFACTS.items()
    )
    c23_geometry = _load_json("c2_c3_geometry")
    c23_pg = _load_json("c2_c3_point_group")
    rare_geometry = _load_json("s4_d6h_geometry")
    rare_pg = _load_json("s4_d6h_point_group")
    deliverable_geometry = _load_json("deliverable_geometry")
    deliverable_pg = _load_json("deliverable_point_group")
    xyz_valid, xyz_atom_count = _xyz_is_valid(ROOT / ARTIFACTS["deliverable_xyz"][0])

    target_evidence = {
        "C2": {"panel_count": 48, "geometry_fraction": c23_geometry["metrics"]["c2_success_fraction"],
               "compatible_fraction": c23_pg["metrics"]["c2_pg_compatible_fraction"]},
        "C3": {"panel_count": 16, "geometry_fraction": c23_geometry["metrics"]["c3_success_fraction"],
               "compatible_fraction": c23_pg["metrics"]["c3_pg_compatible_fraction"]},
        "S4": {"panel_count": 38, "geometry_fraction": rare_geometry["metrics"]["s4_success_fraction"],
               "compatible_fraction": rare_pg["metrics"]["s4_pg_compatible_fraction"]},
        "D6h": {"panel_count": 14, "geometry_fraction": rare_geometry["metrics"]["d6h_success_fraction"],
                "compatible_fraction": rare_pg["metrics"]["d6h_pg_compatible_fraction"]},
    }
    forbidden_dependency_checks = {
        "reference_xyz_never_used": bool(c23_geometry["checks"]["reference_xyz_never_used"]
                                          and rare_geometry["checks"]["reference_xyz_never_used"]),
        "stored_action_never_used": bool(c23_geometry["checks"]["stored_action_never_used"]
                                          and rare_geometry["checks"]["stored_action_never_used"]),
        "etkdg_never_used": bool(c23_geometry["checks"]["etkdg_never_used"]
                                  and rare_geometry["checks"]["etkdg_never_used"]),
    }
    checks = {
        "artifact_hashes_frozen": identity_ok,
        "all_four_target_groups_covered": set(target_evidence) == {"C2", "C3", "S4", "D6h"},
        "all_target_geometry_fractions_one": all(v["geometry_fraction"] == 1.0 for v in target_evidence.values()),
        "all_target_compatible_fractions_one": all(v["compatible_fraction"] == 1.0 for v in target_evidence.values()),
        "all_independent_analyzers_succeeded": (
            c23_pg["metrics"]["analyzer_success_fraction"] == 1.0
            and rare_pg["metrics"]["analyzer_success_fraction"] == 1.0
        ),
        "forbidden_dependencies_absent": all(forbidden_dependency_checks.values()),
        "single_inference_geometry_passed": bool(deliverable_geometry["passed_geometry_gate"]),
        "single_inference_point_group_passed": bool(deliverable_pg["passed"]),
        "single_inference_xyz_is_well_formed": xyz_valid,
    }
    passed = all(checks.values())
    report = {
        "schema_version": "etflow-e3f02-four-target-completion-v1",
        "status": (
            "PASS_ETFLOW_E3F02_KNOWN_GRAPH_TARGET_PG_TO_XYZ_DELIVERABLE"
            if passed else "FAIL_ETFLOW_E3F02_COMPLETION_AUDIT"
        ),
        "passed": passed,
        "task_boundary": {
            "input": "canonical known molecular graph + requested Target_PG",
            "supported_target_pg": ["C2", "C3", "S4", "D6h"],
            "output": "explicit-H final.xyz plus deterministic NPZ and audit JSON",
            "guarantee": "strict geometry gate and independent actual-PG compatible audit",
            "not_claimed": [
                "raw ET-Flow learned Target_PG conditioning without hard projection",
                "novel 2D graph or SMILES generation",
                "success for graph/point-group pairs without a legal graph automorphism",
            ],
        },
        "evidence": {
            "target_groups": target_evidence,
            "total_fixed_panel_molecules": sum(v["panel_count"] for v in target_evidence.values()),
            "c2_c3_exact_fraction": c23_pg["metrics"]["pg_exact_match_fraction"],
            "s4_d6h_exact_fraction": rare_pg["metrics"]["pg_exact_match_fraction"],
            "s4_minimum_selected_inertia_separation": rare_geometry["metrics"]["s4_minimum_selected_inertia_separation"],
            "deliverable_xyz_atom_count": xyz_atom_count,
        },
        "forbidden_dependency_checks": forbidden_dependency_checks,
        "checks": checks,
        "identity": artifact_hashes,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2, sort_keys=True))
    print(f"report={OUTPUT}")
    print(f"sha256={_sha256(OUTPUT)}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
