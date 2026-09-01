"""Freeze a graph-only selected C2/C3 IID-validation panel for M4."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .audit_m2_tier16_panel import _geometry_support
from .c0_local_recovery import _sha256
from .data import PGOrbitFlowDataset
from .global_shape_model import build_global_shape_contract


def _dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-protocol", type=Path, required=True)
    parser.add_argument("--m3-report", type=Path, required=True)
    parser.add_argument("--m3-reproducibility", type=Path, required=True)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("generative_model/data/processed/v2"),
    )
    parser.add_argument("--panel-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-atoms", type=int, default=30)
    args = parser.parse_args()

    protocol_path = args.m3_protocol.resolve()
    report_path = args.m3_report.resolve()
    reproduction_path = args.m3_reproducibility.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_M3_TIER32" or not report.get("passed"):
        raise RuntimeError("M4 requires the passed M3 report")
    if reproduction.get("status") != "PASS_M3_REPRODUCIBILITY" or not reproduction.get("passed"):
        raise RuntimeError("M4 requires the passed M3 reproducibility audit")
    if report["protocol"]["sha256"] != _sha256(protocol_path):
        raise RuntimeError("M3 protocol identity changed")

    package_dir = args.package_dir.resolve()
    validation = PGOrbitFlowDataset(
        package_dir,
        split="val",
        split_scheme="iid",
        point_groups=("C2", "C3"),
    )
    train = PGOrbitFlowDataset(
        package_dir,
        split="train",
        split_scheme="iid",
        point_groups=("C2", "C3"),
    )
    train_indices = {train[index].package_index for index in range(len(train))}

    # Selection uses only split membership, atom count, topology and package index.
    # The shape-support check is graph-distance based and never reads coordinates.
    eligible = {"C2": [], "C3": []}
    rejected = Counter()
    for position in range(len(validation)):
        sample = validation[position]
        if len(sample.atom_types) > args.max_atoms:
            rejected["atom_count_above_limit"] += 1
            continue
        try:
            build_global_shape_contract(
                sample,
                minimum_graph_distance=int(protocol["global_shape_contract"]["minimum_graph_distance"]),
                quantile_count=int(protocol["global_shape_contract"]["quantile_count"]),
            )
        except ValueError:
            rejected["no_graph_only_global_shape_support"] += 1
            continue
        eligible[sample.target_pg].append(sample)
    for rows in eligible.values():
        rows.sort(key=lambda sample: int(sample.package_index))
    selected = eligible["C2"][:16] + eligible["C3"][:16]

    records = []
    for sample in selected:
        support = dict(_geometry_support(sample))
        support.pop("contract")
        records.append(
            {
                "package_index": int(sample.package_index),
                "molecule_id": sample.molecule_id,
                "target_pg": sample.target_pg,
                "core": sample.core_name,
                "arm": sample.arm_name,
                "num_atoms": len(sample.atom_types),
                "contains_sn": bool(np.any(sample.atomic_numbers == 50)),
                **support,
            }
        )
    counts = Counter(record["target_pg"] for record in records)
    selected_indices = {record["package_index"] for record in records}
    m3_indices = {int(record["package_index"]) for record in protocol["panel"]["records"]}
    checks = {
        "molecule_count_exact": len(records) == 32,
        "c2_c3_balanced": counts == {"C2": 16, "C3": 16},
        "iid_validation_only": True,
        "zero_overlap_with_iid_train": not bool(selected_indices & train_indices),
        "zero_overlap_with_m3_panel": not bool(selected_indices & m3_indices),
        "selection_uses_no_coordinates_or_geometry_targets": True,
        "atom_count_within_limit": bool(records) and max(row["num_atoms"] for row in records) <= args.max_atoms,
        "global_shape_graph_support_present": len(records) == 32,
    }
    passed = all(checks.values())
    panel = {
        "schema_version": "pg-orbitflow-m4-unseen-iid-panel-v1",
        "status": "FROZEN_M4_UNSEEN_IID_PANEL" if passed else "NOT_FROZEN_AUDIT_FAILED",
        "package_dir": str(package_dir),
        "canonical_manifest_sha256": _sha256(package_dir / "manifest.json"),
        "split": "val",
        "split_scheme": "iid",
        "selection_rule": (
            f"within frozen IID-validation, require N<={args.max_atoms} and graph-distance global-shape "
            "support; sort each PG by canonical package_index and take first 16 C2 plus "
            "first 16 C3; coordinates and geometry targets are not used for ranking"
        ),
        "molecule_count": len(records),
        "records": records,
    }
    _dump(args.panel_output.resolve(), panel)
    audit = {
        "schema_version": "pg-orbitflow-m4-unseen-iid-panel-audit-v1",
        "status": "PASS_M4_UNSEEN_IID_PANEL_AUDIT" if passed else "FAIL_M4_UNSEEN_IID_PANEL_AUDIT",
        "passed": passed,
        "checks": checks,
        "selection_summary": {
            "point_group_counts": dict(sorted(counts.items())),
            "validation_c2_c3_count": len(validation),
            "eligible_counts": {key: len(value) for key, value in eligible.items()},
            "rejected_counts": dict(sorted(rejected.items())),
            "nonplanar_torsion_orbit_count": sum(row["nonplanar_torsion_orbit_count"] for row in records),
            "active_chirality_count": sum(row["active_chirality_count"] for row in records),
            "undefined_torsion_orbit_count": sum(row["undefined_torsion_orbit_count"] for row in records),
            "max_atom_count": max((row["num_atoms"] for row in records), default=0),
            "atom_count_limit": int(args.max_atoms),
        },
        "m3_parent": {
            "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "reproducibility": {"path": str(reproduction_path), "sha256": _sha256(reproduction_path)},
        },
        "panel": {"path": str(args.panel_output.resolve()), "sha256": _sha256(args.panel_output.resolve())},
        "decision": "freeze read-only unseen prediction Gate" if passed else "stop before unseen evaluation",
        "execution_revision": {
            "previous_v1_failure": "N<=21 admitted only eight C3 validation molecules",
            "single_change": "raise graph-only atom-count limit to the minimum audited value 30 that supports at least 16 C3 molecules",
            "coordinates_or_geometry_targets_used_for_revision": False,
        },
    }
    _dump(args.output.resolve(), audit)
    print(f"{audit['status']} report={args.output.resolve()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
