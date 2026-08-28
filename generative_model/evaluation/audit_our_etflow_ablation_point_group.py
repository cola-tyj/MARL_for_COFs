"""Independent pymatgen point-group audit for the paired ablation package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from generative_model.data.symmetry import SymmetryProtocol
from generative_model.evaluation.our_etflow_ablation import ROUTES
from generative_model.evaluation.symmetry_metrics import analyze_evaluation_symmetry
from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT,
    _atomic_json,
    _fingerprint,
    _sha256,
)


DEFAULT_PROTOCOL = ROOT / "generative_model/evaluation/our_etflow_ablation_protocol_v1.json"
DEFAULT_RUN = ROOT / "generative_model/runs/our_etflow_ablation_v1"


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    target_pgs = sorted({record["target_pg"] for record in records})
    for route in ROUTES:
        route_records = [record["routes"][route] for record in records]
        analyzer = sum(bool(item.get("analyzer_success")) for item in route_records)
        compatible = sum(bool(item.get("pg_compatible")) for item in route_records)
        exact = sum(bool(item.get("pg_exact_match")) for item in route_records)
        result[route] = {
            "full_panel_count": len(records),
            "generation_success_count": sum(
                bool(item.get("generation_success")) for item in route_records
            ),
            "analyzer_success_count": analyzer,
            "analyzer_success_fraction_full_panel": analyzer / len(records),
            "pg_compatible_count": compatible,
            "pg_compatible_fraction_full_panel": compatible / len(records),
            "pg_exact_match_count": exact,
            "pg_exact_match_fraction_full_panel": exact / len(records),
            "by_target_pg": {},
        }
        for target_pg in target_pgs:
            selected = [
                record["routes"][route]
                for record in records
                if record["target_pg"] == target_pg
            ]
            result[route]["by_target_pg"][target_pg] = {
                "count": len(selected),
                "analyzer_success_fraction": sum(
                    bool(item.get("analyzer_success")) for item in selected
                ) / len(selected),
                "pg_compatible_fraction": sum(
                    bool(item.get("pg_compatible")) for item in selected
                ) / len(selected),
                "pg_exact_match_fraction": sum(
                    bool(item.get("pg_exact_match")) for item in selected
                ) / len(selected),
            }
    return result


def audit(run_dir: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_fingerprint") != _fingerprint(protocol):
        raise RuntimeError("ablation protocol fingerprint mismatch")
    report_path = run_dir / "report.json"
    coordinates_path = run_dir / "coordinates.npz"
    source = json.loads(report_path.read_text(encoding="utf-8"))
    if source["identity"]["protocol_sha256"] != _sha256(protocol_path):
        raise RuntimeError("report/protocol hash mismatch")
    if source["identity"]["coordinates_sha256"] != _sha256(coordinates_path):
        raise RuntimeError("report/coordinates hash mismatch")
    symmetry_settings = protocol["symmetry_protocol"]
    symmetry_protocol = SymmetryProtocol(
        analyzer=symmetry_settings["analyzer"],
        tolerance_angstrom=float(symmetry_settings["tolerance_angstrom"]),
        eigen_tolerance=float(symmetry_settings["eigen_tolerance"]),
        matrix_tolerance=float(symmetry_settings["matrix_tolerance"]),
        assignment=symmetry_settings["assignment"],
        assignment_metric=symmetry_settings["assignment_metric"],
        operation_acceptance=symmetry_settings["operation_acceptance"],
    )
    with np.load(coordinates_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    offsets = np.asarray(arrays["atom_offsets"], dtype=np.int64)
    numbers = np.asarray(arrays["atomic_numbers"], dtype=np.int64)
    records: list[dict[str, Any]] = []
    for panel_id, source_record in enumerate(source["records"]):
        start, end = map(int, offsets[panel_id : panel_id + 2])
        target_pg = str(source_record["target_pg"])
        record = {
            "panel_id": panel_id,
            "package_index": int(source_record["package_index"]),
            "molecule_id": str(source_record["molecule_id"]),
            "target_pg": target_pg,
            "routes": {},
        }
        for route in ROUTES:
            generation_success = bool(source_record["routes"][route]["success"])
            route_record: dict[str, Any] = {
                "generation_success": generation_success,
                "analyzer_success": False,
                "pg_compatible": False,
                "pg_exact_match": False,
            }
            if generation_success:
                positions = np.asarray(
                    arrays[f"positions_{route}"][start:end], dtype=np.float64
                )
                try:
                    actual = analyze_evaluation_symmetry(
                        numbers[start:end], positions, None, symmetry_protocol
                    )
                    route_record.update({
                        "analyzer_success": True,
                        "analyzer_pg": str(actual["analyzer_pg"]),
                        "actual_pg": str(actual["actual_pg"]),
                        "pg_exact_match": bool(actual["actual_pg"] == target_pg),
                        "actual_mean_rms_error_angstrom": float(
                            actual["actual"]["mean_rms_error"]
                        ),
                        "actual_max_atom_error_angstrom": float(
                            actual["actual"]["max_atom_error"]
                        ),
                    })
                    try:
                        symmetry = analyze_evaluation_symmetry(
                            numbers[start:end], positions, target_pg, symmetry_protocol
                        )
                        route_record.update({
                            "pg_compatible": bool(symmetry["pg_compatible"]),
                            "target_mean_rms_error_angstrom": float(
                                symmetry["target"]["mean_rms_error"]
                            ),
                            "target_max_atom_error_angstrom": float(
                                symmetry["target"]["max_atom_error"]
                            ),
                        })
                    except Exception as error:
                        route_record["target_compatibility_failure"] = (
                            f"{type(error).__name__}: {error}"
                        )
                except Exception as error:
                    route_record["failure"] = f"{type(error).__name__}: {error}"
            else:
                route_record["failure"] = source_record["routes"][route].get(
                    "failure", "generation failed"
                )
            record["routes"][route] = route_record
        records.append(record)
        print(json.dumps({
            "audited": panel_id + 1,
            "total": len(source["records"]),
            "package_index": record["package_index"],
        }, ensure_ascii=False), flush=True)
    result = {
        "schema_version": "our-etflow-paired-ablation-point-group-audit-v1",
        "status": "PASS_INDEPENDENT_POINT_GROUP_AUDIT_COMPLETED",
        "quality_claim": False,
        "records": records,
        "summary": _summary(records),
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "source_report_sha256": _sha256(report_path),
            "coordinates_sha256": _sha256(coordinates_path),
            "auditor_source_sha256": _sha256(Path(__file__)),
            "symmetry_protocol": symmetry_settings,
        },
    }
    _atomic_json(run_dir / "point_group_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    result = audit(args.run_dir.resolve(), args.protocol.resolve())
    print(json.dumps({
        "status": result["status"],
        "summary": result["summary"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
