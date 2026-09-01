"""Independent point-group audit for the Cn our_ET_Flow batch."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

from generative_model.data.symmetry import (
    _assign_operation, _classify_group, _largest_valid_subgroup, _safe_closure,
)
from generative_model.evaluation.symmetry_metrics import atomic_symbols
from generative_model.inference.audit_etflow_e3f02_point_group import audit
from generative_model.inference.audit_etflow_e3f02_point_group import (
    SYMMETRY_PROTOCOL,
)
from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT, _atomic_json, _sha256,
)
from generative_model.inference.generate_etflow_symmetric_xyz_v5 import DEFAULT_PROTOCOL
from generative_model.models.graph_pg_3d_projection import (
    minimum_pair_distance, operation_errors,
)


DEFAULT_RUN = ROOT / "generative_model/results/Cn"
EXTENDED_GROUP_ORDERS = {
    "T": 12, "Th": 24, "Td": 24, "O": 24, "Oh": 48,
    "I": 60, "Ih": 120,
}
C3_VERIFIED_SUPERGROUPS = {
    "C3", "C3h", "C3v", "D3", "D3d", "D3h", "D6h",
    "T", "Th", "Td", "O", "Oh", "I", "Ih",
}


def _validate_xyz_outputs(run_root: Path, record_dir: Path) -> None:
    source = json.loads((record_dir / "report.json").read_text(encoding="utf-8"))
    convenience = run_root / source["outputs"]["convenience_final_xyz"]
    local = record_dir / source["outputs"]["final_xyz"]
    if _sha256(convenience) != source["identity"]["final_xyz_sha256"]:
        raise RuntimeError(f"convenience final XYZ SHA 不一致: {convenience}")
    if convenience.read_bytes() != local.read_bytes():
        raise RuntimeError(f"convenience/local final XYZ 字节不一致: {record_dir}")
    lines = convenience.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0])
    if atom_count != source["atom_count"] or len(lines) != atom_count + 2:
        raise RuntimeError(f"final XYZ atom count/行数不一致: {convenience}")
    xyz_positions = np.asarray([
        [float(value) for value in line.split()[1:4]] for line in lines[2:]
    ], dtype=np.float64)
    with np.load(record_dir / "coordinates.npz", allow_pickle=False) as archive:
        expected = np.asarray(archive["final_positions"], dtype=np.float64)
    if xyz_positions.shape != expected.shape or not np.allclose(
        xyz_positions, expected, rtol=0.0, atol=5.1e-11
    ):
        raise RuntimeError(f"final XYZ 与 coordinates.npz 不一致: {convenience}")


def _write_manifest(run_root: Path, protocol_path: Path) -> Path:
    artifact_paths = [
        run_root / "batch_report.json", run_root / "index.csv",
        run_root / "point_group_batch_report.json",
        run_root / "point_group_index.csv",
    ]
    artifact_paths.extend(sorted(run_root.glob("C3k*/xyz/*.xyz")))
    artifact_paths.extend(sorted(run_root.glob("C3k*/records/*/coordinates.npz")))
    artifact_paths.extend(sorted(run_root.glob("C3k*/records/*/raw.xyz")))
    artifact_paths.extend(sorted(run_root.glob("C3k*/records/*/projected.xyz")))
    artifact_paths.extend(sorted(run_root.glob("C3k*/records/*/final.xyz")))
    artifact_paths.extend(sorted(run_root.glob("C3k*/records/*/report.json")))
    artifact_paths.extend(sorted(run_root.glob("C3k*/records/*/point_group_report.json")))
    input_paths = sorted((ROOT / "generative_model/data/Cn").glob("C3k*/*"))
    input_paths.append(ROOT / "generative_model/data/Cn/README_FORMAT.md")
    source_paths = [
        ROOT / "generative_model/data/cn_graph_dataset.py",
        ROOT / "generative_model/inference/run_our_etflow_cn.py",
        Path(__file__).resolve(),
        protocol_path,
    ]
    missing = [str(path) for path in artifact_paths + input_paths + source_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cn manifest 缺少文件: {missing[:10]}")
    value = {
        "schema_version": "our-etflow-cn-results-manifest-v1",
        "artifact_count": len(artifact_paths),
        "input_file_count": len(input_paths),
        "source_file_count": len(source_paths),
        "artifacts": {
            str(path.relative_to(run_root)): _sha256(path) for path in artifact_paths
        },
        "inputs": {
            str(path.relative_to(ROOT)): _sha256(path) for path in input_paths
        },
        "sources": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
        },
        "checkpoint": {
            "relative_path": "generative_model/checkpoints/etflow/drugs-o3.ckpt",
            "sha256": _sha256(
                ROOT / "generative_model/checkpoints/etflow/drugs-o3.ckpt"
            ),
        },
    }
    output = run_root / "manifest.json"
    _atomic_json(output, value)
    return output


def _audit_extended_cubic_record(run_dir: Path, protocol_path: Path) -> dict:
    """Validate a largest proven subgroup when the legacy schema rejects T/O/I.

    Pymatgen occasionally labels a Cn geometry as a cubic supergroup while its
    public generator list closes only to a smaller subgroup.  We therefore do
    not blindly trust the label: every closed operation is element-assigned,
    and ``actual_pg`` is the full analyzer label only when the expected group
    order is recovered.  Otherwise it is the validated closed subgroup.
    """

    report_path = run_dir / "report.json"
    coordinates_path = run_dir / "coordinates.npz"
    source = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if source["identity"]["protocol_sha256"] != _sha256(protocol_path):
        raise RuntimeError("Cn inference report/protocol SHA 不一致")
    if source["identity"]["coordinates_sha256"] != _sha256(coordinates_path):
        raise RuntimeError("Cn inference report/coordinates SHA 不一致")
    if not source["passed_geometry_gate"]:
        raise RuntimeError("不能审计未通过 geometry gate 的 Cn record")
    with np.load(coordinates_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    final = np.asarray(arrays["final_positions"], dtype=np.float64)
    centered = final - final.mean(axis=0)
    numbers = np.asarray(arrays["atomic_numbers"], dtype=np.int64)
    symbols = atomic_symbols(numbers)
    analyzer = PointGroupAnalyzer(
        Molecule(symbols, centered),
        tolerance=SYMMETRY_PROTOCOL.tolerance_angstrom,
        eigen_tolerance=SYMMETRY_PROTOCOL.eigen_tolerance,
        matrix_tolerance=SYMMETRY_PROTOCOL.matrix_tolerance,
    )
    candidate_pg = str(analyzer.sch_symbol)
    if candidate_pg not in EXTENDED_GROUP_ORDERS:
        raise ValueError(f"Cn extended auditor 不支持 analyzer PG: {candidate_pg}")
    expected_order = EXTENDED_GROUP_ORDERS[candidate_pg]
    operations = _safe_closure(
        [np.asarray(item.rotation_matrix, dtype=np.float64) for item in analyzer.symmops],
        SYMMETRY_PROTOCOL.matrix_tolerance,
        maximum_size=expected_order,
    )
    assignments = [_assign_operation(symbols, centered, matrix) for matrix in operations]
    rms_errors = np.asarray([value[1] for value in assignments], dtype=np.float64)
    orthogonality = np.asarray([
        np.max(np.abs(matrix.T @ matrix - np.eye(3))) for matrix in operations
    ])
    valid = (
        (rms_errors <= SYMMETRY_PROTOCOL.tolerance_angstrom + 1e-12)
        & (orthogonality <= 1e-6)
    )
    if len(operations) == expected_order and bool(valid.all()):
        actual_pg = candidate_pg
        actual_indices = tuple(range(len(operations)))
        validation_scope = "full_extended_analyzer_group"
    else:
        actual_indices = _largest_valid_subgroup(
            operations, valid, SYMMETRY_PROTOCOL.matrix_tolerance
        )
        actual_pg = _classify_group(
            [operations[index] for index in actual_indices]
        )
        validation_scope = "largest_valid_closed_subgroup_of_analyzer_generators"
    selected_rms = rms_errors[list(actual_indices)]
    target_errors = operation_errors(
        final,
        np.asarray(arrays["operation_matrices"], dtype=np.float64),
        np.asarray(arrays["permutation_index"], dtype=np.int64),
    )
    minimum_distance = minimum_pair_distance(final)
    thresholds = protocol["gate_thresholds"]
    compatible = actual_pg in C3_VERIFIED_SUPERGROUPS
    checks = {
        "source_geometry_gate_passed": True,
        "analyzer_success": True,
        "extended_group_operations_element_assigned": bool(valid.all()),
        "requested_pg_compatible": compatible,
        "minimum_pair_distance": (
            minimum_distance >= thresholds["minimum_pair_distance_angstrom"]
        ),
        "maximum_operation_error": (
            target_errors["max_atom_error_angstrom"]
            <= thresholds["maximum_operation_error_angstrom"]
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "our-etflow-cn-extended-point-group-audit-v1",
        "status": (
            "PASS_ETFLOW_E3F02_KNOWN_GRAPH_TARGET_PG_TO_XYZ"
            if passed else "FAIL_OUR_ETFLOW_CN_EXTENDED_POINT_GROUP_AUDIT"
        ),
        "passed": passed,
        "package_index": int(source["package_index"]),
        "molecule_id": str(source["molecule_id"]),
        "requested_target_pg": "C3",
        "analyzer_pg": candidate_pg,
        "actual_pg": actual_pg,
        "pg_exact_match": actual_pg == "C3",
        "pg_compatible": compatible,
        "actual_group_validation_scope": validation_scope,
        "analyzer_expected_operation_count": expected_order,
        "validated_operation_count": len(actual_indices),
        "validated_mean_operation_rms_error_angstrom": float(selected_rms.mean()),
        "validated_max_operation_rms_error_angstrom": float(selected_rms.max()),
        "minimum_pair_distance_angstrom": float(minimum_distance),
        "maximum_operation_error_angstrom": float(
            target_errors["max_atom_error_angstrom"]
        ),
        "mean_operation_rms_error_angstrom": float(
            target_errors["mean_operation_rms_angstrom"]
        ),
        "checks": checks,
        "scope": source["scope"],
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


def audit_batch(run_root: Path, protocol_path: Path) -> dict:
    source_batch_path = run_root / "batch_report.json"
    source_batch = json.loads(source_batch_path.read_text(encoding="utf-8"))
    if not source_batch["passed_generation"]:
        raise RuntimeError("Cn generation batch 未完整通过，拒绝汇总点群")
    results = []
    failures = []
    xyz_integrity_count = 0
    for report_path in sorted(run_root.glob("C3k*/records/*/report.json")):
        try:
            _validate_xyz_outputs(run_root, report_path.parent)
            xyz_integrity_count += 1
            try:
                result = audit(report_path.parent, protocol_path)
            except ValueError as error:
                if "当前 schema 不支持的点群" not in str(error):
                    raise
                result = _audit_extended_cubic_record(
                    report_path.parent, protocol_path
                )
            results.append(result)
        except Exception as error:
            failures.append({
                "record": str(report_path.parent.relative_to(run_root)),
                "failure": f"{type(error).__name__}: {error}",
            })
    if len(results) + len(failures) != source_batch["completed_count"]:
        raise RuntimeError("Cn point-group audit 记录数与 generation batch 不一致")
    rows = [{
        "package_index": value["package_index"],
        "molecule_id": value["molecule_id"],
        "requested_target_pg": value["requested_target_pg"],
        "analyzer_pg": value["analyzer_pg"],
        "actual_pg": value["actual_pg"],
        "pg_exact_match": value["pg_exact_match"],
        "pg_compatible": value["pg_compatible"],
        "minimum_pair_distance_angstrom": value["minimum_pair_distance_angstrom"],
        "maximum_operation_error_angstrom": value["maximum_operation_error_angstrom"],
        "passed": value["passed"],
    } for value in sorted(results, key=lambda item: item["package_index"])]
    index_path = run_root / "point_group_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) if rows else [
            "package_index", "molecule_id", "requested_target_pg", "analyzer_pg",
            "actual_pg", "pg_exact_match", "pg_compatible",
            "minimum_pair_distance_angstrom", "maximum_operation_error_angstrom",
            "passed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    actual = Counter(value["actual_pg"] for value in results)
    passed_count = sum(value["passed"] for value in results)
    summary = {
        "schema_version": "our-etflow-cn-independent-point-group-batch-audit-v1",
        "status": (
            "PASS_OUR_ETFLOW_CN_2D_GRAPH_TO_C3_XYZ"
            if not failures and passed_count == len(results) == 98
            else "FAIL_OUR_ETFLOW_CN_POINT_GROUP_AUDIT"
        ),
        "passed": not failures and passed_count == len(results) == 98,
        "molecule_count": len(results),
        "passed_count": passed_count,
        "failure_count": len(failures),
        "pg_compatible_count": sum(value["pg_compatible"] for value in results),
        "pg_exact_match_count": sum(value["pg_exact_match"] for value in results),
        "xyz_integrity_count": xyz_integrity_count,
        "actual_pg_distribution": dict(sorted(actual.items())),
        "minimum_pair_distance_angstrom": min(
            value["minimum_pair_distance_angstrom"] for value in results
        ),
        "maximum_operation_error_angstrom": max(
            value["maximum_operation_error_angstrom"] for value in results
        ),
        "failures": failures,
        "identity": {
            "source_batch_sha256": _sha256(source_batch_path),
            "protocol_sha256": _sha256(protocol_path),
            "point_group_index_sha256": _sha256(index_path),
        },
    }
    _atomic_json(run_root / "point_group_batch_report.json", summary)
    manifest_path = _write_manifest(run_root, protocol_path)
    print(json.dumps({
        "manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path)
    }, ensure_ascii=False, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    result = audit_batch(args.run_root.resolve(), args.protocol.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
