"""Independent point-group and response audit for Target_PG controllability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from generative_model.data.symmetry import SymmetryProtocol
from generative_model.evaluation.symmetry_metrics import analyze_evaluation_symmetry
from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT,
    _atomic_json,
    _sha256,
)


DEFAULT_PROTOCOL = ROOT / "generative_model/evaluation/our_etflow_controllability_protocol_v1.json"
DEFAULT_RUN = ROOT / "generative_model/runs/our_etflow_controllability_v1"


def audit(run_dir: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_path = run_dir / "report.json"
    coordinate_path = run_dir / "coordinates.npz"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source["identity"]["protocol_sha256"] != _sha256(protocol_path):
        raise RuntimeError("controllability report/protocol mismatch")
    if source["identity"]["coordinates_sha256"] != _sha256(coordinate_path):
        raise RuntimeError("controllability report/coordinates mismatch")
    settings = protocol["symmetry_protocol"]
    analyzer_protocol = SymmetryProtocol(
        analyzer=settings["analyzer"],
        tolerance_angstrom=float(settings["tolerance_angstrom"]),
        eigen_tolerance=float(settings["eigen_tolerance"]),
        matrix_tolerance=float(settings["matrix_tolerance"]),
        assignment=settings["assignment"],
        assignment_metric=settings["assignment_metric"],
        operation_acceptance=settings["operation_acceptance"],
    )
    with np.load(coordinate_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    offsets = arrays["atom_offsets"]
    numbers = arrays["atomic_numbers"]
    targets = tuple(protocol["targets"])
    records = []
    for panel_id, source_record in enumerate(source["records"]):
        start, end = map(int, offsets[panel_id : panel_id + 2])
        record = {
            "panel_id": panel_id,
            "package_index": int(source_record["package_index"]),
            "molecule_id": str(source_record["molecule_id"]),
            "targets": {},
        }
        for target in targets:
            target_record = {
                "generation_success": bool(source_record["targets"][target]["success"]),
                "analyzer_success": False,
                "pg_compatible": False,
                "pg_exact_match": False,
            }
            if target_record["generation_success"]:
                positions = arrays[f"final_positions_{target}"][start:end]
                try:
                    actual = analyze_evaluation_symmetry(
                        numbers[start:end], positions, None, analyzer_protocol
                    )
                    target_record.update({
                        "analyzer_success": True,
                        "analyzer_pg": str(actual["analyzer_pg"]),
                        "actual_pg": str(actual["actual_pg"]),
                        "pg_exact_match": bool(actual["actual_pg"] == target),
                    })
                    try:
                        requested = analyze_evaluation_symmetry(
                            numbers[start:end], positions, target, analyzer_protocol
                        )
                        target_record["pg_compatible"] = bool(requested["pg_compatible"])
                    except Exception as error:
                        target_record["compatibility_failure"] = (
                            f"{type(error).__name__}: {error}"
                        )
                except Exception as error:
                    target_record["failure"] = f"{type(error).__name__}: {error}"
            record["targets"][target] = target_record
        records.append(record)
    target_summary = {}
    for target in targets:
        values = [record["targets"][target] for record in records]
        target_summary[target] = {
            "count": len(values),
            "generation_success_fraction": float(np.mean([
                item["generation_success"] for item in values
            ])),
            "analyzer_success_fraction": float(np.mean([
                item["analyzer_success"] for item in values
            ])),
            "pg_compatible_fraction": float(np.mean([
                item["pg_compatible"] for item in values
            ])),
            "pg_exact_match_fraction": float(np.mean([
                item["pg_exact_match"] for item in values
            ])),
        }
    minimum_fraction = float(
        protocol["response_gate"]["minimum_responsive_fraction_per_target_pair"]
    )
    pair_summary = source["summary_pre_point_group"]["pairwise_response"]
    checks = {
        "all_targets_generated": all(
            value["generation_success_fraction"] == 1.0
            for value in target_summary.values()
        ),
        "all_requested_targets_compatible": all(
            value["pg_compatible_fraction"] == 1.0
            for value in target_summary.values()
        ),
        "every_target_pair_responsive": all(
            value["responsive_fraction"] >= minimum_fraction
            for value in pair_summary.values()
        ),
        "shared_raw_exact_by_construction": bool(
            source["summary_pre_point_group"][
                "raw_coordinates_shared_exactly_by_construction"
            ]
        ),
    }
    passed = all(checks.values())
    prior_audit = run_dir / "controllability_audit.json"
    result = {
        "schema_version": "our-etflow-target-pg-controllability-audit-v2",
        "status": (
            "PASS_OUR_ETFLOW_TARGET_PG_CONTROLLABILITY"
            if passed else "FAIL_OUR_ETFLOW_TARGET_PG_CONTROLLABILITY_DIAGNOSE"
        ),
        "passed": passed,
        "checks": checks,
        "target_summary": target_summary,
        "pairwise_response_summary": pair_summary,
        "records": records,
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "source_report_sha256": _sha256(source_path),
            "coordinates_sha256": _sha256(coordinate_path),
            "auditor_source_sha256": _sha256(Path(__file__)),
        },
        "execution_revision": {
            "scientific_protocol_changed": False,
            "coordinates_changed": False,
            "reason": (
                "extend actual-point-group read-only classification to pymatgen D6; "
                "the v1 audit incorrectly counted D6 as analyzer failure"
            ),
            "prior_v1_audit_sha256": (
                _sha256(prior_audit) if prior_audit.exists() else None
            ),
        },
    }
    _atomic_json(run_dir / "controllability_audit_v2.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    result = audit(args.run_dir.resolve(), args.protocol.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
