"""Create the tracked M2 training/evaluation summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .c0_local_recovery import _sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("generative_model/PG_OrbitFlow")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    run_dir = root / "runs" / "m2_torsion_tier4_v1"
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    console = root / "m2_torsion_tier4_v1_console.log"
    pattern = re.compile(
        r"step=(\d+)/1024 loss=([0-9.eE+-]+) bond=([0-9.eE+-]+)A "
        r"angle=([0-9.eE+-]+)deg torsion=([0-9.eE+-]+)deg"
    )
    milestones = [
        {
            "step": int(match.group(1)),
            "loss": float(match.group(2)),
            "bond_orbit_mae_angstrom": float(match.group(3)),
            "angle_orbit_mae_degrees": float(match.group(4)),
            "torsion_orbit_circular_mae_degrees": float(match.group(5)),
        }
        for match in pattern.finditer(console.read_text(encoding="utf-8"))
    ]
    if [row["step"] for row in milestones] != [1, 256, 512, 768, 1024]:
        raise RuntimeError("M2 frozen console milestones are incomplete")
    reproducibility = root / "reports" / "m2_reproducibility_v1.json"
    reproduction = json.loads(reproducibility.read_text(encoding="utf-8"))
    if not reproduction.get("passed"):
        raise RuntimeError("M2 reproducibility must pass before summarization")
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }
    summary = {
        "schema_version": "pg-orbitflow-m2-summary-v1",
        "status": report["status"],
        "passed": report["passed"],
        "claim_scope": report["claim_scope"],
        "claim_limit": report["claim_limit"],
        "protocol": report["protocol"],
        "training": {
            **report["training"],
            "molecule_count": 4,
            "point_group_panel": {"C2": 2, "C3": 2},
            "torsion_orbit_counts": {"864": 7, "1045": 14, "1135": 4, "2391": 8},
            "milestones": milestones,
            "final_checkpoint_sha256": _sha256(run_dir / "last.pt"),
        },
        "prediction_gate": report["prediction_gate"],
        "reconstruction_gate": report["reconstruction_gate"],
        "prediction_records": report["prediction_records"],
        "reconstruction_records": report["reconstruction_records"],
        "isolation": report["isolation"],
        "reproducibility": {
            "status": reproduction["status"],
            "sha256": _sha256(reproducibility),
            "first_cpu_attempt_explanation": (
                "CPU inference changed prediction metrics at about 1e-6 and therefore "
                "the terminal optimizer coordinates. Reproduction with the original "
                "CUDA prediction device was exact at the metric level and passed the "
                "frozen coordinate tolerances; no scientific setting changed."
            ),
        },
        "artifacts": artifacts,
        "interpretation": (
            "M2 closes the graph-to-orbit-IC-to-symmetric-3D interface on the frozen "
            "four-molecule memorization panel without oracle decoder geometry. The "
            "panel torsions are all planar, so the next permitted action is a Tier-16 "
            "panel audit containing non-planar/chiral support, not a generalization claim."
        ),
        "next_step": (
            "audit and freeze Tier-16 composition, prioritizing non-planar torsion and "
            "chiral support before training/evaluation"
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
