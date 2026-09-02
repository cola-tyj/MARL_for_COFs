"""Audit eligibility for Gate-aligned worst-molecule torsion refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..c0_local_recovery import _sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3p1-report", type=Path, required=True)
    parser.add_argument("--m3p2-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    m3p1_path = args.m3p1_report.resolve()
    m3p2_path = args.m3p2_report.resolve()
    m3p1 = json.loads(m3p1_path.read_text(encoding="utf-8"))
    m3p2 = json.loads(m3p2_path.read_text(encoding="utf-8"))
    gate = m3p2["prediction_gate"]
    thresholds = gate["thresholds"]

    failures = []
    for record in m3p2["prediction_records"]:
        reasons = []
        if record["torsion_orbit_circular_mae_degrees"] > thresholds[
            "torsion_orbit_circular_mae_max_degrees"
        ]:
            reasons.append("torsion")
        if record["nonplanar_torsion_orbit_circular_mae_degrees"] > thresholds[
            "nonplanar_torsion_orbit_circular_mae_max_degrees"
        ]:
            reasons.append("nonplanar_torsion")
        if reasons:
            failures.append(
                {
                    "package_index": record["package_index"],
                    "target_pg": record["target_pg"],
                    "reasons": reasons,
                    "torsion_mae_degrees": record[
                        "torsion_orbit_circular_mae_degrees"
                    ],
                    "nonplanar_mae_degrees": record[
                        "nonplanar_torsion_orbit_circular_mae_degrees"
                    ],
                }
            )

    previous = m3p1["prediction_gate"]["values"]
    current = gate["values"]
    checks = {
        "m3p2_status_is_prediction_failure": m3p2.get("status")
        == "FAIL_M3P2_PREDICTION_GATE_STOP_BEFORE_DECODER",
        "bond_gate_remains_passed": bool(
            gate["checks"]["bond_orbit_mae_max_angstrom"]
        ),
        "angle_gate_remains_passed": bool(
            gate["checks"]["angle_orbit_mae_max_degrees"]
        ),
        "torsion_gate_is_only_primary_failure": not bool(
            gate["checks"]["torsion_orbit_circular_mae_max_degrees"]
        ),
        "nonplanar_gate_is_only_secondary_failure": not bool(
            gate["checks"][
                "nonplanar_torsion_orbit_circular_mae_max_degrees"
            ]
        ),
        "m3p2_improved_over_m3p1": current[
            "torsion_orbit_circular_mae_max_degrees"
        ]
        < previous["torsion_orbit_circular_mae_max_degrees"],
        "failures_are_sparse": 0 < len(failures) <= 2,
        "one_c2_and_one_c3_failure": {row["target_pg"] for row in failures}
        == {"C2", "C3"},
    }
    passed = all(checks.values())
    output = {
        "schema_version": "pg-orbitflow-m3p3-worst-molecule-readiness-v1",
        "status": (
            "PASS_M3P3_WORST_MOLECULE_REFINEMENT_GATE"
            if passed
            else "FAIL_M3P3_WORST_MOLECULE_REFINEMENT_GATE"
        ),
        "passed": passed,
        "checks": checks,
        "failing_molecules": failures,
        "metrics": {
            "m3p1_torsion_max_degrees": previous[
                "torsion_orbit_circular_mae_max_degrees"
            ],
            "m3p2_torsion_max_degrees": current[
                "torsion_orbit_circular_mae_max_degrees"
            ],
            "m3p2_nonplanar_max_degrees": current[
                "nonplanar_torsion_orbit_circular_mae_max_degrees"
            ],
        },
        "decision": {
            "change": (
                "add worst molecule-level torsion and nonplanar circular losses "
                "to the existing full-panel mean loss"
            ),
            "unchanged": [
                "panel",
                "original prediction and reconstruction Gate",
                "frozen graph features, bond head, and angle head",
                "decoder",
            ],
        },
        "identity": {
            "m3p1_report_sha256": _sha256(m3p1_path),
            "m3p2_report_sha256": _sha256(m3p2_path),
            "auditor_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{output['status']} report={args.output.resolve()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
