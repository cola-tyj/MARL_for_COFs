"""Freeze the all-52 rerun with the v5 S4 inertia guard."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "generative_model/inference/etflow_e3f02_rare_targets_protocol_v2.json"
INFERENCE = ROOT / "generative_model/inference/etflow_e3f02_protocol_v5.json"
V2_GEOMETRY = ROOT / "generative_model/runs/etflow_e3f02_rare_targets_v2/geometry_report.json"
V2_POINT_GROUP = ROOT / "generative_model/runs/etflow_e3f02_rare_targets_v2/point_group_report.json"
DIAGNOSIS = ROOT / "generative_model/runs/etflow_e3f02_rare_targets_v2/s4_inertia_boundary_diagnosis.json"
PILOT_REPORT = ROOT / "generative_model/runs/etflow_e3f02_v5_examples/s4_001102/report.json"
PILOT_POINT_GROUP = ROOT / "generative_model/runs/etflow_e3f02_v5_examples/s4_001102/point_group_report.json"
OUTPUT = ROOT / "generative_model/inference/etflow_e3f02_rare_targets_protocol_v3.json"
PARENT_SHA256 = "02a36db68853a505d5fb74df27a4639687c02bb1aaaaf6a315cd8339c76fb7bd"
INFERENCE_SHA256 = "cfcb872cf4d8902f2ccdba1bd2738b6476eba27b7a7be9e5108e7e0d4a2678d2"
SOURCES = (
    "generative_model/inference/run_etflow_e3f02_rare_targets_v3.py",
    "generative_model/inference/build_etflow_e3f02_rare_targets_protocol_v3.py",
    "generative_model/inference/audit_etflow_e3f02_rare_targets.py",
    "generative_model/inference/run_etflow_e3f02_rare_targets.py",
    "generative_model/tests/test_etflow_e3f02_rare_targets.py",
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
    if _sha256(PARENT) != PARENT_SHA256 or _sha256(INFERENCE) != INFERENCE_SHA256:
        raise RuntimeError("rare-target v3 parent protocol SHA 改变")
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    inference = json.loads(INFERENCE.read_text(encoding="utf-8"))
    geometry = json.loads(V2_GEOMETRY.read_text(encoding="utf-8"))
    point_group = json.loads(V2_POINT_GROUP.read_text(encoding="utf-8"))
    diagnosis = json.loads(DIAGNOSIS.read_text(encoding="utf-8"))
    pilot = json.loads(PILOT_REPORT.read_text(encoding="utf-8"))
    pilot_pg = json.loads(PILOT_POINT_GROUP.read_text(encoding="utf-8"))
    if not geometry["passed_geometry_gate"] or geometry["metrics"]["end_to_end_success_fraction"] != 1.0:
        raise RuntimeError("v3 要求 v2 geometry 52/52")
    if point_group["metrics"]["d6h_pg_compatible_fraction"] != 1.0:
        raise RuntimeError("v3 不得改写 D6h")
    if point_group["metrics"]["s4_pg_compatible_fraction"] >= 0.9:
        raise RuntimeError("v3 缺少 S4 inertia-boundary failure")
    if not diagnosis["passed"] or not pilot["passed_geometry_gate"] or not pilot_pg["passed"]:
        raise RuntimeError("v3 inertia diagnosis/pilot 未通过")

    protocol = deepcopy(parent)
    protocol.update({
        "schema_version": "etflow-e3f02-rare-targets-protocol-v3",
        "status": "FROZEN_BEFORE_ALL_CANONICAL_S4_D6H_INERTIA_GUARD_RERUN",
        "route": "ET-Flow v5 + S4/D6h multi-action + S4 inertia guard + E3/F0.2",
        "purpose": (
            "Repeat the identical all-52 panel after the sole reference-free "
            "S4 candidate-selection guard demonstrated on package 1102."
        ),
        "s4_inertia_selection": inference["s4_inertia_selection"],
        "execution_revision": {
            "parent_v2_protocol_sha256": PARENT_SHA256,
            "v2_geometry_status_preserved": geometry["status"],
            "v2_point_group_status_preserved": point_group["status"],
            "scientific_change": (
                "filter passed S4 candidates at mass-weighted inertia separation "
                ">=0.012, then apply the unchanged minimum-objective selection"
            ),
            "held_fixed": [
                "all 52 package indices and per-molecule seeds",
                "official checkpoint, 4 raw samples and 50 ODE steps",
                "64-action limits, projection ranking and top-8 F0.2 optimization",
                "geometry/point-group thresholds and full-panel denominators",
                "all D6h behavior",
            ],
        },
    })
    protocol["identity"].update({
        "parent_v2_protocol_sha256": PARENT_SHA256,
        "inference_protocol_sha256": INFERENCE_SHA256,
        "rare_target_v2_geometry_sha256": _sha256(V2_GEOMETRY),
        "rare_target_v2_point_group_sha256": _sha256(V2_POINT_GROUP),
        "s4_inertia_diagnosis_sha256": _sha256(DIAGNOSIS),
        "s4_v5_pilot_report_sha256": _sha256(PILOT_REPORT),
        "s4_v5_pilot_point_group_sha256": _sha256(PILOT_POINT_GROUP),
        "source_sha256": {
            **{relative: _sha256(ROOT / relative) for relative in SOURCES},
            **inference["identity"]["source_sha256"],
        },
    })
    protocol["protocol_fingerprint"] = _fingerprint(protocol)
    OUTPUT.write_text(json.dumps(
        protocol, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n", encoding="utf-8")
    print(f"FROZEN protocol={OUTPUT}")
    print(f"sha256={_sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
