"""Create the frozen H1-H3 Tier-4 result and branch-decision report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("generative_model/PG_OrbitFlow")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    variants = ("h1_harmonic", "h2_centralizer", "h3_endpoint")
    records = {}
    passing = []
    for variant in variants:
        run_dir = root / "runs" / f"overfit_tier4_{variant}"
        report_path = run_dir / "report.json"
        diagnosis_path = root / "reports" / f"tier4_{variant}_diagnosis.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
        if report["passed"]:
            passing.append(variant)
        records[variant] = {
            "status": report["status"],
            "passed": report["passed"],
            "prior": report["prior"],
            "transport": report["transport"],
            "endpoint_weight": json.loads(
                Path(report["protocol"]["path"]).read_text(encoding="utf-8")
            )["objective"]["endpoint_weight"],
            "training": report["training"],
            "data": report["data"],
            "gate_values": report["gate"]["values"],
            "gate_thresholds": report["gate"]["thresholds"],
            "failed_checks": sorted(
                key for key, value in report["gate"]["checks"].items() if not value
            ),
            "diagnostic_summary": diagnosis["diagnostic_summary"],
            "artifacts": {
                str(path.relative_to(run_dir)): _sha256(path)
                for path in sorted(run_dir.iterdir())
                if path.is_file()
            },
            "diagnosis_sha256": _sha256(diagnosis_path),
        }
    selected = passing[0] if passing else None
    summary = {
        "schema_version": "pg-orbitflow-h1-h3-tier4-summary-v1",
        "status": "PASS_SELECT_SIMPLEST_TIER4_PARENT"
        if selected
        else "FAIL_ALL_H1_H3_STOP_CARTESIAN_BACKBONE",
        "all_experiments_completed": True,
        "frozen_comparison": {
            "molecules": 4,
            "point_group_panel": {"C2": 2, "C3": 2},
            "optimizer_steps": 1024,
            "batch_size_molecules": 4,
            "molecule_exposures": 4096,
            "uniform_time_range": [0.02, 0.98],
            "raw_integrator": "50-step Heun",
            "same_seed_model_panel_optimizer_and_gate": True,
            "independent_training_from_same_random_initialization": True,
            "posthoc_hard_projection_used": False,
            "f02_used": False,
        },
        "selection_policy": "first passing variant in H1 > H2 > H3 order",
        "selected_tier4_parent": selected,
        "tier16_started": False,
        "tier32_started": False,
        "decision": (
            f"advance {selected} to original Tier-16"
            if selected
            else "stop Cartesian backbone; move to fixed-graph bond/angle/torsion manifold decomposition"
        ),
        "records": records,
        "protocol_audit_sha256": _sha256(
            root / "reports" / "h1_h3_protocol_audit.json"
        ),
        "raw_reproducibility_sha256": _sha256(
            root / "reports" / "h1_h3_raw_reproducibility.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{summary['status']} report={args.output.resolve()}")


if __name__ == "__main__":
    main()
