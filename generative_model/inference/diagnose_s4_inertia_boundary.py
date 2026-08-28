"""Read-only diagnosis of pymatgen S4 misses near its inertia boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from generative_model.inference.generate_etflow_symmetric_xyz import _atomic_json
from generative_model.inference.s4_inertia_selection import (
    mass_weighted_inertia_separation,
)
from generative_model.models.graph_pg_3d_projection import operation_errors


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "generative_model/runs/etflow_e3f02_rare_targets_v2"
OUTPUT = RUN / "s4_inertia_boundary_diagnosis.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    point_path = RUN / "point_group_report.json"
    geometry_path = RUN / "geometry_report.json"
    point = json.loads(point_path.read_text(encoding="utf-8"))
    if point["status"] != "FAIL_ETFLOW_E3F02_RARE_TARGETS_DIAGNOSE_FAILURES":
        raise RuntimeError("S4 inertia diagnosis 只允许读取冻结 v2 failure")
    records = []
    for source in point["records"]:
        if source["requested_target_pg"] != "S4":
            continue
        package_index = int(source["package_index"])
        coordinate_path = RUN / "molecules" / f"{package_index:06d}.npz"
        with np.load(coordinate_path, allow_pickle=False) as archive:
            numbers = np.asarray(archive["atomic_numbers"], dtype=np.int64)
            positions = np.asarray(archive["final_positions"], dtype=np.float64)
            matrices = np.asarray(archive["operation_matrices"], dtype=np.float64)
            permutations = np.asarray(archive["permutation_index"], dtype=np.int64)
        errors = operation_errors(positions, matrices, permutations)
        records.append({
            "package_index": package_index,
            "pg_compatible": bool(source["pg_compatible"]),
            "actual_pg": source["actual_pg"],
            "inertia_separation": mass_weighted_inertia_separation(numbers, positions),
            "stored_action_max_error_angstrom": errors["max_atom_error_angstrom"],
            "coordinates_sha256": _sha256(coordinate_path),
        })
    compatible = [value for value in records if value["pg_compatible"]]
    failed = [value for value in records if not value["pg_compatible"]]
    metrics = {
        "s4_count": len(records),
        "compatible_count": len(compatible),
        "failed_count": len(failed),
        "compatible_minimum_inertia_separation": min(
            value["inertia_separation"] for value in compatible
        ),
        "failed_maximum_inertia_separation": max(
            value["inertia_separation"] for value in failed
        ),
        "maximum_stored_action_error_angstrom": max(
            value["stored_action_max_error_angstrom"] for value in records
        ),
        "proposed_reference_free_threshold": 0.012,
        "pymatgen_eigen_tolerance": 0.01,
    }
    checks = {
        "all_38_s4_present": len(records) == 38,
        "all_failed_below_proposed_threshold": all(
            value["inertia_separation"] < 0.012 for value in failed
        ),
        "all_compatible_above_proposed_threshold": all(
            value["inertia_separation"] >= 0.012 for value in compatible
        ),
        "stored_s4_action_exact": metrics["maximum_stored_action_error_angstrom"] <= 1e-10,
        "threshold_above_analyzer_boundary": 0.012 > 0.01,
    }
    report = {
        "schema_version": "s4-inertia-boundary-diagnosis-v1",
        "status": "PASS_S4_INERTIA_BOUNDARY_DIAGNOSIS_FREEZE_SELECTION_GUARD" if all(checks.values()) else "FAIL_S4_INERTIA_BOUNDARY_DIAGNOSIS",
        "passed": all(checks.values()),
        "metrics": metrics, "checks": checks, "records": records,
        "scope": {
            "read_only": True, "reference_xyz_used": False,
            "stored_action_used_for_diagnosis_only": True,
            "pymatgen_used_for_candidate_selection": False,
        },
        "identity": {
            "geometry_report_sha256": _sha256(geometry_path),
            "point_group_report_sha256": _sha256(point_path),
            "source_sha256": _sha256(Path(__file__)),
        },
    }
    _atomic_json(OUTPUT, report)
    print(json.dumps({
        "status": report["status"], "metrics": metrics, "checks": checks,
        "output": str(OUTPUT),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
