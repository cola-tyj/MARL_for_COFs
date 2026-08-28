"""Independent pymatgen audit for one ET-Flow -> E3 -> F0.2 inference run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from generative_model.data.symmetry import SymmetryProtocol
from generative_model.evaluation.symmetry_metrics import analyze_evaluation_symmetry
from generative_model.models.graph_pg_3d_projection import (
    minimum_pair_distance,
    operation_errors,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_protocol_v1.json"
SYMMETRY_PROTOCOL = SymmetryProtocol(
    analyzer="pymatgen.symmetry.analyzer.PointGroupAnalyzer",
    tolerance_angstrom=0.3,
    eigen_tolerance=0.01,
    matrix_tolerance=0.1,
    assignment="scipy.optimize.linear_sum_assignment, element-blocked",
    assignment_metric=(
        "Euclidean distance in centroid-centered Cartesian coordinates"
    ),
    operation_acceptance=(
        "operation RMS assigned distance <= tolerance_angstrom"
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit(run_dir: Path, protocol_path: Path) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    coordinates_path = run_dir / "coordinates.npz"
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    identity = source_report["identity"]
    if identity["protocol_sha256"] != _sha256(protocol_path):
        raise RuntimeError("inference report/protocol SHA 不一致")
    if identity["coordinates_sha256"] != _sha256(coordinates_path):
        raise RuntimeError("inference report/coordinates SHA 不一致")
    if not source_report["passed_geometry_gate"]:
        raise RuntimeError("不能审计未通过 geometry gate 的 inference run")

    with np.load(coordinates_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    final = np.asarray(arrays["final_positions"], dtype=np.float64)
    target_pg = str(source_report["requested_target_pg"])
    symmetry = analyze_evaluation_symmetry(
        np.asarray(arrays["atomic_numbers"], dtype=np.int64),
        final,
        target_pg,
        SYMMETRY_PROTOCOL,
    )
    errors = operation_errors(
        final,
        np.asarray(arrays["operation_matrices"], dtype=np.float64),
        np.asarray(arrays["permutation_index"], dtype=np.int64),
    )
    minimum_distance = minimum_pair_distance(final)
    thresholds = protocol["gate_thresholds"]
    checks = {
        "source_geometry_gate_passed": True,
        "analyzer_success": True,
        "requested_pg_compatible": bool(symmetry["pg_compatible"]),
        "minimum_pair_distance": (
            minimum_distance >= thresholds["minimum_pair_distance_angstrom"]
        ),
        "maximum_operation_error": (
            errors["max_atom_error_angstrom"]
            <= thresholds["maximum_operation_error_angstrom"]
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "etflow-e3f02-independent-point-group-audit-v1",
        "status": (
            "PASS_ETFLOW_E3F02_KNOWN_GRAPH_TARGET_PG_TO_XYZ"
            if passed
            else "FAIL_ETFLOW_E3F02_POINT_GROUP_AUDIT"
        ),
        "passed": passed,
        "package_index": int(source_report["package_index"]),
        "molecule_id": str(source_report["molecule_id"]),
        "requested_target_pg": target_pg,
        "analyzer_pg": str(symmetry["analyzer_pg"]),
        "actual_pg": str(symmetry["actual_pg"]),
        "pg_exact_match": bool(symmetry["pg_exact_match"]),
        "pg_compatible": bool(symmetry["pg_compatible"]),
        "minimum_pair_distance_angstrom": float(minimum_distance),
        "maximum_operation_error_angstrom": float(
            errors["max_atom_error_angstrom"]
        ),
        "mean_operation_rms_error_angstrom": float(
            errors["mean_operation_rms_angstrom"]
        ),
        "checks": checks,
        "scope": source_report["scope"],
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "source_report_sha256": _sha256(report_path),
            "coordinates_sha256": _sha256(coordinates_path),
            "auditor_source_sha256": _sha256(Path(__file__)),
            "symmetry_protocol": SYMMETRY_PROTOCOL.__dict__,
        },
    }
    _atomic_json(run_dir / "point_group_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    result = audit(args.run_dir.resolve(), args.protocol.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
