"""Create the tracked M1 training/evaluation summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    training_dir = root / "runs" / "m1_bond_angle_tier4_v1"
    evaluation_dir = root / "runs" / "m1_bond_angle_tier4_v1_cpu_eval"
    report = json.loads((evaluation_dir / "report.json").read_text(encoding="utf-8"))
    console_path = training_dir / "console_interrupted_cuda_decoder.log"
    pattern = re.compile(
        r"step=(\d+)/1024 loss=([0-9.eE+-]+) bond=([0-9.eE+-]+)A angle=([0-9.eE+-]+)deg"
    )
    milestones = [
        {
            "step": int(match.group(1)),
            "loss": float(match.group(2)),
            "bond_orbit_mae_angstrom": float(match.group(3)),
            "angle_orbit_mae_degrees": float(match.group(4)),
        }
        for match in pattern.finditer(console_path.read_text(encoding="utf-8"))
    ]
    if [row["step"] for row in milestones] != [1, 256, 512, 768, 1024]:
        raise RuntimeError("M1 frozen console milestones are incomplete")
    checkpoints = {
        path.name: _sha256(path)
        for path in sorted(training_dir.glob("*.pt"))
    }
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(evaluation_dir.iterdir())
        if path.is_file()
    }
    summary = {
        "schema_version": "pg-orbitflow-m1-summary-v1",
        "status": report["status"],
        "passed": report["passed"],
        "claim_scope": report["claim_scope"],
        "protocol": report["protocol"],
        "training": {
            "molecule_count": 4,
            "point_group_panel": {"C2": 2, "C3": 2},
            "optimizer_steps": 1024,
            "batch_size_molecules": 4,
            "parameter_count": 266402,
            "final_checkpoint_sha256": _sha256(training_dir / "last.pt"),
            "milestones": milestones,
            "full_loss_array_unavailable_reason": (
                "The v1 runner originally persisted losses after decoder evaluation; "
                "CUDA decoder was interrupted after training. The runner is fixed for "
                "future runs, while the five predeclared console milestones and all "
                "checkpoints are retained."
            ),
            "checkpoints": checkpoints,
            "console_sha256": _sha256(console_path),
        },
        "prediction_gate": report["prediction_gate"],
        "reconstruction_gate": report["reconstruction_gate"],
        "prediction_records": report["prediction_records"],
        "reconstruction_records": report["reconstruction_records"],
        "execution_revision": report["execution_revision"],
        "isolation": report["isolation"],
        "interpretation": (
            "The quotient encoder can memorize bond/angle orbits on Tier-4, and the "
            "M0 solver remains accurate without oracle local-pair distances. Oracle "
            "torsions still prevent a claim of fully learned 2D-to-3D generation."
        ),
        "next_step": "M2 learned torsion-orbit sin/cos head with ring/chirality gate",
        "reproducibility_report_sha256": _sha256(
            root / "reports" / "m1_reproducibility_v1.json"
        ),
        "evaluation_artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{summary['status']} report={args.output.resolve()}")


if __name__ == "__main__":
    main()
