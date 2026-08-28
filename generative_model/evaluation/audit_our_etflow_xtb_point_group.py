"""Independent pre/post point-group audit for GFN2-xTB relaxation."""

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
    _fingerprint,
    _sha256,
)


DEFAULT_PROTOCOL = ROOT / "generative_model/evaluation/our_etflow_xtb_protocol_v1.json"
DEFAULT_RUN = ROOT / "generative_model/runs/our_etflow_xtb_relaxation_v1"


def _analyze(
    numbers: np.ndarray,
    positions: np.ndarray,
    target_pg: str,
    protocol: SymmetryProtocol,
) -> dict[str, Any]:
    actual = analyze_evaluation_symmetry(numbers, positions, None, protocol)
    result: dict[str, Any] = {
        "analyzer_success": True,
        "analyzer_pg": str(actual["analyzer_pg"]),
        "actual_pg": str(actual["actual_pg"]),
        "pg_exact_match": str(actual["actual_pg"]) == target_pg,
        "actual_mean_rms_error_angstrom": float(actual["actual"]["mean_rms_error"]),
        "actual_max_atom_error_angstrom": float(actual["actual"]["max_atom_error"]),
        "pg_compatible": False,
    }
    try:
        target = analyze_evaluation_symmetry(numbers, positions, target_pg, protocol)
        result.update({
            "pg_compatible": bool(target["pg_compatible"]),
            "target_mean_rms_error_angstrom": float(target["target"]["mean_rms_error"]),
            "target_max_atom_error_angstrom": float(target["target"]["max_atom_error"]),
        })
    except Exception as error:
        result["target_compatibility_failure"] = f"{type(error).__name__}: {error}"
    return result


def audit(run_dir: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_fingerprint") != _fingerprint(protocol):
        raise RuntimeError("xTB protocol fingerprint mismatch")
    report_path = run_dir / "relaxation_report.json"
    coordinates_path = run_dir / "relaxed_coordinates.npz"
    source = json.loads(report_path.read_text(encoding="utf-8"))
    if source["identity"]["protocol_sha256"] != _sha256(protocol_path):
        raise RuntimeError("xTB report/protocol hash mismatch")
    if source["identity"]["relaxed_coordinates_sha256"] != _sha256(coordinates_path):
        raise RuntimeError("xTB report/coordinate hash mismatch")
    settings = protocol["symmetry_protocol"]
    symmetry_protocol = SymmetryProtocol(
        analyzer=settings["analyzer"],
        tolerance_angstrom=float(settings["tolerance_angstrom"]),
        eigen_tolerance=float(settings["eigen_tolerance"]),
        matrix_tolerance=float(settings["matrix_tolerance"]),
        assignment=settings["assignment"],
        assignment_metric=settings["assignment_metric"],
        operation_acceptance=settings["operation_acceptance"],
    )
    with np.load(coordinates_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    offsets = np.asarray(arrays["atom_offsets"], dtype=np.int64)
    numbers = np.asarray(arrays["atomic_numbers"], dtype=np.int64)
    records = []
    for local_id, source_record in enumerate(source["records"]):
        start, end = map(int, offsets[local_id : local_id + 2])
        target_pg = str(source_record["target_pg"])
        record: dict[str, Any] = {
            "panel_id": int(source_record["panel_id"]),
            "package_index": int(source_record["package_index"]),
            "molecule_id": str(source_record["molecule_id"]),
            "target_pg": target_pg,
            "calculation_success": bool(source_record["success"]),
        }
        try:
            record["before"] = _analyze(
                numbers[start:end], arrays["input_positions"][start:end],
                target_pg, symmetry_protocol,
            )
        except Exception as error:
            record["before"] = {
                "analyzer_success": False,
                "pg_compatible": False,
                "failure": f"{type(error).__name__}: {error}",
            }
        if source_record["success"]:
            try:
                record["after"] = _analyze(
                    numbers[start:end], arrays["relaxed_positions"][start:end],
                    target_pg, symmetry_protocol,
                )
            except Exception as error:
                record["after"] = {
                    "analyzer_success": False,
                    "pg_compatible": False,
                    "failure": f"{type(error).__name__}: {error}",
                }
        else:
            record["after"] = {
                "analyzer_success": False,
                "pg_compatible": False,
                "failure": "GFN2-xTB calculation failed",
            }
        records.append(record)
    count = len(records)
    summary = {
        "panel_count": count,
        "before_analyzer_success_fraction_full_panel": sum(
            bool(record["before"]["analyzer_success"]) for record in records
        ) / count,
        "before_pg_compatible_fraction_full_panel": sum(
            bool(record["before"]["pg_compatible"]) for record in records
        ) / count,
        "after_analyzer_success_fraction_full_panel": sum(
            bool(record["after"]["analyzer_success"]) for record in records
        ) / count,
        "after_pg_compatible_fraction_full_panel": sum(
            bool(record["after"]["pg_compatible"]) for record in records
        ) / count,
        "after_pg_exact_match_fraction_full_panel": sum(
            bool(record["after"].get("pg_exact_match")) for record in records
        ) / count,
        "by_target_pg": {},
    }
    for target_pg in ("C2", "C3", "S4", "D6h"):
        selected = [record for record in records if record["target_pg"] == target_pg]
        summary["by_target_pg"][target_pg] = {
            "count": len(selected),
            "before_compatible_fraction": (
                None if not selected else sum(
                    bool(record["before"]["pg_compatible"]) for record in selected
                ) / len(selected)
            ),
            "after_compatible_fraction": (
                None if not selected else sum(
                    bool(record["after"]["pg_compatible"]) for record in selected
                ) / len(selected)
            ),
        }
    result = {
        "schema_version": "our-etflow-gfn2-xtb-point-group-audit-v1",
        "status": "PASS_INDEPENDENT_POINT_GROUP_AUDIT_COMPLETED",
        "quality_claim": False,
        "records": records,
        "summary": summary,
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "relaxation_report_sha256": _sha256(report_path),
            "relaxed_coordinates_sha256": _sha256(coordinates_path),
            "auditor_source_sha256": _sha256(Path(__file__)),
            "symmetry_protocol": settings,
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
    print(json.dumps({"status": result["status"], "summary": result["summary"]},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
