"""Freeze v4 after the all-rare-target v1 S4 failure and candidate diagnosis."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "generative_model/inference/etflow_e3f02_protocol_v3.json"
V1_GEOMETRY = ROOT / "generative_model/runs/etflow_e3f02_rare_targets_v1/geometry_report.json"
V1_POINT_GROUP = ROOT / "generative_model/runs/etflow_e3f02_rare_targets_v1/point_group_report.json"
DIAGNOSIS = ROOT / "generative_model/runs/etflow_e3f02_s4_diagnosis/package_000053.json"
OUTPUT = ROOT / "generative_model/inference/etflow_e3f02_protocol_v4.json"
PARENT_SHA256 = "367f336f3a11101d025f4cc177f44b9177df79af6058b644acd183fbe1692cf8"
SOURCES = (
    "generative_model/inference/generate_etflow_symmetric_xyz_v4.py",
    "generative_model/inference/build_etflow_e3f02_protocol_v4.py",
    "generative_model/symmetry/graph_action_candidates_v2.py",
    "generative_model/tests/test_etflow_e3f02_inference_v4.py",
    "generative_model/tests/test_graph_action_candidates_v2.py",
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
        raise RuntimeError("v3 parent protocol SHA 改变")
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    geometry = json.loads(V1_GEOMETRY.read_text(encoding="utf-8"))
    point_group = json.loads(V1_POINT_GROUP.read_text(encoding="utf-8"))
    diagnosis = json.loads(DIAGNOSIS.read_text(encoding="utf-8"))
    if geometry["status"] != "FAIL_ETFLOW_E3F02_RARE_TARGETS_DIAGNOSE_FAILURES":
        raise RuntimeError("v4 只能从冻结的 rare-target v1 geometry failure 建立")
    if geometry["metrics"]["d6h_success_fraction"] != 1.0:
        raise RuntimeError("v4 不得改动已通过的 D6h 策略")
    if geometry["metrics"]["s4_success_fraction"] >= 0.9:
        raise RuntimeError("v4 缺少 S4 multi-action 的失败依据")
    if point_group["metrics"]["d6h_pg_compatible_fraction"] != 1.0:
        raise RuntimeError("D6h independent audit 未通过")
    if diagnosis["successful_optimized_count"] != 8:
        raise RuntimeError("S4 multi-action diagnosis 未证明方向有效")

    protocol = deepcopy(parent)
    protocol.update({
        "schema_version": "etflow-e3f02-inference-protocol-v4",
        "status": "FROZEN_S4_D6H_MULTI_ACTION_BEFORE_FORMAL_S4_PILOT",
        "route": "ET-Flow/E3/F0.2-v4-S4-D6h-multi-action",
        "purpose": (
            "Preserve v3 D6h behavior and add reference-free S4 action/raw "
            "conformer matching after the all-canonical v1 S4 failure."
        ),
        "rare_action_selection": {
            target: {
                "maximum_actions": 64,
                "maximum_search_matches": 100_000,
                "optimize_top_combinations": 8,
            }
            for target in ("S4", "D6h")
        },
        "decision": {
            "next_gate": "formal package-53 S4 geometry plus independent point-group audit",
            "pass_action": "freeze a new all-52 rare-target batch protocol",
            "fail_action": "preserve v4 and diagnose without changing v3 evidence",
        },
    })
    protocol["identity"].update({
        "parent_v3_protocol_sha256": PARENT_SHA256,
        "rare_target_v1_geometry_sha256": _sha256(V1_GEOMETRY),
        "rare_target_v1_point_group_sha256": _sha256(V1_POINT_GROUP),
        "s4_package53_diagnosis_sha256": _sha256(DIAGNOSIS),
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
