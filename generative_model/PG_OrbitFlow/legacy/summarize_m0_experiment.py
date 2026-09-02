"""Create the tracked M0 oracle reconstruction result summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


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
    run_dir = root / "runs" / "m0_oracle_reconstruction_v1"
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    starts = json.loads((run_dir / "all_starts.json").read_text(encoding="utf-8"))
    convergence = {}
    for molecule in starts:
        records = molecule["records"]
        package_index = str(molecule["package_index"])
        convergence[package_index] = {
            "starts": len(records),
            "rmsd_le_0_1_angstrom": int(
                sum(record["kabsch_rmsd_angstrom"] <= 0.1 for record in records)
            ),
            "collision_free": int(sum(record["collision_free"] for record in records)),
            "energy_min": float(min(record["final_energy"] for record in records)),
            "energy_median": float(
                np.median([record["final_energy"] for record in records])
            ),
            "energy_max": float(max(record["final_energy"] for record in records)),
        }
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }
    summary = {
        "schema_version": "pg-orbitflow-m0-oracle-summary-v1",
        "status": report["status"],
        "passed": report["passed"],
        "claim_scope": report["claim_scope"],
        "protocol": report["protocol"],
        "gate": report["gate"],
        "records": report["records"],
        "multi_start_convergence": convergence,
        "interpretation": (
            "Exact group lifting and oracle orbit-internal-coordinate energy can "
            "reconstruct all four frozen molecules. C3 remains more non-convex "
            "than C2, so deterministic multi-start or a learned/global initializer "
            "must remain in M1/M2."
        ),
        "next_step": "implement M1 learned bond/angle orbit heads with oracle torsions",
        "reproducibility_report_sha256": _sha256(
            root / "reports" / "m0_selected_start_reproducibility_v1.json"
        ),
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{summary['status']} report={args.output.resolve()}")


if __name__ == "__main__":
    main()
