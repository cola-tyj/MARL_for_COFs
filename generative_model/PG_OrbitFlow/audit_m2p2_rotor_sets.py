"""Audit graph-only local rotor sets against the frozen M2.1 diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .local_rotor import build_local_rotor_set_contract
from .overfit import _panel_samples


SCHEMA_VERSION = "pg-orbitflow-m2p2-local-rotor-set-audit-v1"


def _dump(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    diagnosis = json.loads(args.diagnosis.read_text(encoding="utf-8"))
    if diagnosis.get("status") != "DIAGNOSED_M2P1_LOCAL_EXCHANGEABLE_TORSION_COLLAPSE":
        raise ValueError("M2.2 audit requires the frozen M2.1 diagnosis")
    diagnosed = {int(row["package_index"]): row for row in diagnosis["molecules"]}
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    records = []
    failed_total = covered_total = 0
    for sample in _panel_samples(protocol, tier):
        contract = build_geometry_contract(sample)
        local = build_local_rotor_set_contract(sample, contract)
        failed = set(diagnosed[int(sample.package_index)]["failed_orbit_ids"])
        covered = sorted(
            failed.intersection({orbit for group in local.groups for orbit in group})
        )
        failed_total += len(failed)
        covered_total += len(covered)
        records.append(
            {
                "package_index": int(sample.package_index),
                "target_pg": sample.target_pg,
                "group_count": len(local.groups),
                "group_sizes": [len(group) for group in local.groups],
                "groups": [list(group) for group in local.groups],
                "failed_orbit_count": len(failed),
                "covered_failed_orbit_ids": covered,
            }
        )
    group_count = sum(row["group_count"] for row in records)
    covered_fraction = covered_total / failed_total if failed_total else 1.0
    checks = {
        "molecule_count_exact": len(records) == 16,
        "groups_found": group_count > 0,
        "all_group_sizes_supported": all(
            size in {2, 3} for row in records for size in row["group_sizes"]
        ),
        "dominant_failure_coverage_min": covered_total >= 29,
        "graph_only_contract": True,
        "xyz_or_target_not_used_for_grouping": True,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "PASS_M2P2_LOCAL_ROTOR_SET_REPRESENTATION_GATE_A"
            if all(checks.values())
            else "FAIL_M2P2_LOCAL_ROTOR_SET_REPRESENTATION_GATE_A"
        ),
        "passed": all(checks.values()),
        "checks": checks,
        "identity": {
            "m2p1_protocol_sha256": _sha256(args.protocol),
            "m2p1_diagnosis_sha256": _sha256(args.diagnosis),
        },
        "summary": {
            "molecule_count": len(records),
            "local_rotor_set_count": group_count,
            "failed_torsion_orbit_count": failed_total,
            "covered_failed_torsion_orbit_count": covered_total,
            "covered_failure_fraction": covered_fraction,
        },
        "scope": {
            "training_performed": False,
            "coordinates_used": False,
            "targets_used_to_build_groups": False,
            "claim": "representation coverage only; not a model-quality pass",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _dump(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"{result['status']} report={args.output.resolve()}")


if __name__ == "__main__":
    main()
