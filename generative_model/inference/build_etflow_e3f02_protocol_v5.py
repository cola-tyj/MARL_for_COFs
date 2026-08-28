"""Freeze v5 after the formal v4 rare-target inertia-boundary failure."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "generative_model/inference/etflow_e3f02_protocol_v4.json"
V2_GEOMETRY = ROOT / "generative_model/runs/etflow_e3f02_rare_targets_v2/geometry_report.json"
V2_POINT_GROUP = ROOT / "generative_model/runs/etflow_e3f02_rare_targets_v2/point_group_report.json"
DIAGNOSIS = ROOT / "generative_model/runs/etflow_e3f02_rare_targets_v2/s4_inertia_boundary_diagnosis.json"
OUTPUT = ROOT / "generative_model/inference/etflow_e3f02_protocol_v5.json"
PARENT_SHA256 = "cf078d2088e30ba3ce3aea588b0699a5a05ec9813e01095f77f1846e4b80e7cc"
SOURCES = (
    "generative_model/inference/generate_etflow_symmetric_xyz_v5.py",
    "generative_model/inference/build_etflow_e3f02_protocol_v5.py",
    "generative_model/inference/s4_inertia_selection.py",
    "generative_model/inference/diagnose_s4_inertia_boundary.py",
    "generative_model/tests/test_s4_inertia_selection.py",
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
    if _sha256(PARENT) != PARENT_SHA256:
        raise RuntimeError("v4 parent protocol SHA 改变")
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    geometry = json.loads(V2_GEOMETRY.read_text(encoding="utf-8"))
    point_group = json.loads(V2_POINT_GROUP.read_text(encoding="utf-8"))
    diagnosis = json.loads(DIAGNOSIS.read_text(encoding="utf-8"))
    if not geometry["passed_geometry_gate"] or geometry["metrics"]["s4_success_fraction"] != 1.0:
        raise RuntimeError("v5 要求 v4 batch geometry 52/52")
    if point_group["metrics"]["d6h_pg_compatible_fraction"] != 1.0:
        raise RuntimeError("v5 不得改写已通过的 D6h")
    if point_group["metrics"]["s4_pg_compatible_fraction"] >= 0.9:
        raise RuntimeError("v5 缺少 S4 analyzer failure 依据")
    if not diagnosis["passed"] or diagnosis["status"] != "PASS_S4_INERTIA_BOUNDARY_DIAGNOSIS_FREEZE_SELECTION_GUARD":
        raise RuntimeError("S4 inertia diagnosis 未通过")

    protocol = deepcopy(parent)
    protocol.update({
        "schema_version": "etflow-e3f02-inference-protocol-v5",
        "status": "FROZEN_S4_INERTIA_GUARD_BEFORE_FORMAL_PILOT",
        "route": "ET-Flow/E3/F0.2-v5-S4-inertia-stable-selection",
        "purpose": (
            "Keep v4 geometry and multi-action search, but reject S4 candidates "
            "inside pymatgen's near-spherical inertia classification boundary."
        ),
        "s4_inertia_selection": {
            "metric": "largest adjacent mass-weighted principal-moment gap / largest moment",
            "minimum_separation": 0.012,
            "pymatgen_eigen_tolerance": 0.01,
            "selection": "filter passed S4 candidates, then retain original minimum F0.2 objective rule",
            "translation_rotation_scale_invariant": True,
            "reference_xyz_used": False,
            "pymatgen_called_during_selection": False,
        },
        "decision": {
            "next_gate": "formal package-1102 S4 geometry plus independent point-group audit",
            "pass_action": "freeze all-52 v3 batch with identical panel/seeds/Gates",
            "fail_action": "preserve v5 and diagnose candidate coverage",
        },
    })
    protocol["identity"].update({
        "parent_v4_protocol_sha256": PARENT_SHA256,
        "rare_target_v2_geometry_sha256": _sha256(V2_GEOMETRY),
        "rare_target_v2_point_group_sha256": _sha256(V2_POINT_GROUP),
        "s4_inertia_diagnosis_sha256": _sha256(DIAGNOSIS),
        "source_sha256": {
            **parent["identity"]["source_sha256"],
            **{relative: _sha256(ROOT / relative) for relative in SOURCES},
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
