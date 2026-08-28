"""Independent pymatgen audit for the all-canonical S4/D6h panel."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from generative_model.inference.generate_etflow_symmetric_xyz import _atomic_json
from generative_model.models.graph_pg_3d_projection import minimum_pair_distance, operation_errors


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_rare_targets_protocol_v1.json"
DEFAULT_INPUT = ROOT / "generative_model/runs/etflow_e3f02_rare_targets_v1"
TARGETS = ("S4", "D6h")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("protocol_fingerprint", None)
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def summarize_point_groups(
    records: list[dict[str, Any]], protocol: dict[str, Any], geometry_passed: bool,
) -> tuple[dict[str, Any], dict[str, bool]]:
    by_pg = {point_group: [] for point_group in TARGETS}
    analyzer, compatible, exact = [], [], []
    actual_counts: Counter[str] = Counter()
    for record in records:
        point_group = str(record["requested_target_pg"])
        analyzer.append(bool(record["analyzer_success"]))
        compatible.append(bool(record["pg_compatible"]))
        exact.append(bool(record["pg_exact_match"]))
        by_pg[point_group].append(bool(record["pg_compatible"]))
        if record["actual_pg"] is not None:
            actual_counts[str(record["actual_pg"])] += 1
    metrics = {
        "molecule_count": len(records),
        "analyzer_success_fraction": float(np.mean(analyzer)),
        "pg_compatible_fraction": float(np.mean(compatible)),
        "pg_exact_match_fraction": float(np.mean(exact)),
        "s4_pg_compatible_fraction": float(np.mean(by_pg["S4"])),
        "d6h_pg_compatible_fraction": float(np.mean(by_pg["D6h"])),
        "actual_pg_counts": dict(sorted(actual_counts.items())),
    }
    threshold = protocol["point_group_gate_thresholds"]
    checks = {
        "geometry_gate_passed": bool(geometry_passed),
        "molecule_count_exact": len(records) == protocol["panel"]["molecule_count"],
        "analyzer_success_fraction_min": metrics["analyzer_success_fraction"] >= threshold["analyzer_success_fraction_min"],
        "pg_compatible_fraction_min": metrics["pg_compatible_fraction"] >= threshold["pg_compatible_fraction_min"],
        "s4_pg_compatible_fraction_min": metrics["s4_pg_compatible_fraction"] >= threshold["s4_pg_compatible_fraction_min"],
        "d6h_pg_compatible_fraction_min": metrics["d6h_pg_compatible_fraction"] >= threshold["d6h_pg_compatible_fraction_min"],
    }
    return metrics, checks


def audit(protocol_path: Path, input_dir: Path) -> dict[str, Any]:
    from generative_model.data.symmetry import SymmetryProtocol
    from generative_model.evaluation.symmetry_metrics import analyze_evaluation_symmetry

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_fingerprint"] != _fingerprint(protocol):
        raise RuntimeError("rare-target protocol fingerprint 不一致")
    geometry_path = input_dir / "geometry_report.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if geometry["identity"]["protocol_sha256"] != _sha256(protocol_path):
        raise RuntimeError("geometry report/protocol SHA 不一致")
    symmetry_protocol = SymmetryProtocol(**protocol["identity"]["symmetry_protocol"])
    molecule_dir = input_dir / "molecules"
    thresholds = protocol["geometry_gate_thresholds"]
    records = []
    for progress, package_index in enumerate(map(int, protocol["panel"]["package_indices"]), start=1):
        source = json.loads((molecule_dir / f"{package_index:06d}.json").read_text(encoding="utf-8"))
        result = {
            "package_index": package_index,
            "molecule_id": str(source["molecule_id"]),
            "requested_target_pg": str(source["requested_target_pg"]),
            "generation_success": source["status"] == "success",
            "failure_stage": source["failure_stage"],
            "analyzer_success": False, "analyzer_pg": None, "actual_pg": None,
            "pg_exact_match": False, "pg_compatible": False,
            "minimum_pair_distance_angstrom": None,
            "maximum_operation_error_angstrom": None,
            "failure": source["failure"],
        }
        if source["status"] == "success":
            coordinates_path = molecule_dir / f"{package_index:06d}.npz"
            if source["coordinates_sha256"] != _sha256(coordinates_path):
                raise RuntimeError(f"molecule coordinate SHA 不一致: {package_index}")
            try:
                with np.load(coordinates_path, allow_pickle=False) as archive:
                    arrays = {name: archive[name] for name in archive.files}
                positions = np.asarray(arrays["final_positions"], dtype=np.float64)
                symmetry = analyze_evaluation_symmetry(
                    np.asarray(arrays["atomic_numbers"], dtype=np.int64),
                    positions, result["requested_target_pg"], symmetry_protocol,
                )
                errors = operation_errors(
                    positions, np.asarray(arrays["operation_matrices"], dtype=np.float64),
                    np.asarray(arrays["permutation_index"], dtype=np.int64),
                )
                minimum_distance = minimum_pair_distance(positions)
                if minimum_distance < thresholds["minimum_pair_distance_angstrom"]:
                    raise RuntimeError("independent minimum-pair check failed")
                if errors["max_atom_error_angstrom"] > thresholds["maximum_operation_error_angstrom"]:
                    raise RuntimeError("independent operation-error check failed")
                result.update({
                    "analyzer_success": True,
                    "analyzer_pg": str(symmetry["analyzer_pg"]),
                    "actual_pg": str(symmetry["actual_pg"]),
                    "pg_exact_match": bool(symmetry["pg_exact_match"]),
                    "pg_compatible": bool(symmetry["pg_compatible"]),
                    "minimum_pair_distance_angstrom": float(minimum_distance),
                    "maximum_operation_error_angstrom": float(errors["max_atom_error_angstrom"]),
                    "failure": None,
                })
            except Exception as error:
                result["failure"] = f"{type(error).__name__}: {error}"
        records.append(result)
        print(json.dumps({
            "audited": progress, "total": protocol["panel"]["molecule_count"],
            "package_index": package_index, "generation_success": result["generation_success"],
            "actual_pg": result["actual_pg"], "compatible": result["pg_compatible"],
        }, ensure_ascii=False), flush=True)

    metrics, checks = summarize_point_groups(records, protocol, bool(geometry["passed_geometry_gate"]))
    passed = all(checks.values())
    report = {
        "schema_version": "etflow-e3f02-rare-targets-point-group-v1",
        "status": protocol["decision"]["final_pass_status"] if passed else protocol["decision"]["fail_status"],
        "passed": passed, "metrics": metrics, "checks": checks,
        "records": records, "scope": protocol["scope"],
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "geometry_report_sha256": _sha256(geometry_path),
            "auditor_source_sha256": _sha256(Path(__file__)),
            "symmetry_protocol": protocol["identity"]["symmetry_protocol"],
        },
    }
    _atomic_json(input_dir / "point_group_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    report = audit(args.protocol.resolve(), args.input_dir.resolve())
    print(json.dumps({
        key: report[key] for key in ("status", "passed", "metrics", "checks")
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
