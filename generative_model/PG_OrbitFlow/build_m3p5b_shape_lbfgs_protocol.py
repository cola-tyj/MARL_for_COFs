"""Freeze M3.5b full-batch L-BFGS refinement of the global-shape head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .c0_local_recovery import _sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("generative_model/PG_OrbitFlow").resolve()
    parent_protocol_path = args.parent_protocol.resolve()
    parent_run = args.parent_run_dir.resolve()
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    report_path = parent_run / "report.json"
    checkpoint_path = parent_run / "last.pt"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "FAIL_M3P5_GLOBAL_SHAPE_GATE_STOP_BRANCH":
        raise RuntimeError("M3.5b requires the converging M3.5 shape failure")
    if report["shape_gate"]["exact_embedding_target_conflicts"] != 0:
        raise RuntimeError("M3.5b cannot refine conflicting shape embeddings")
    protocol = {
        **parent_protocol,
        "schema_version": "pg-orbitflow-m3p5b-shape-lbfgs-protocol-v1",
        "seed": int(parent_protocol["seed"]) + 1,
        "parent_shape_run": {
            "protocol": {
                "path": str(parent_protocol_path),
                "sha256": _sha256(parent_protocol_path),
            },
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
            },
        },
        "implementation": {
            "runner": {
                "path": str(root / "m3p5b_shape_lbfgs_refinement.py"),
                "sha256": _sha256(root / "m3p5b_shape_lbfgs_refinement.py"),
            },
            "model": {
                "path": str(root / "global_shape_model.py"),
                "sha256": _sha256(root / "global_shape_model.py"),
            },
        },
        "training": {
            "optimizer": "LBFGS",
            "learning_rate": 0.8,
            "maximum_iterations": 500,
            "history_size": 100,
            "line_search": "strong_wolfe",
            "tolerance_grad": 1e-12,
            "tolerance_change": 1e-15,
            "mean_molecule_mse_weight": 1.0,
            "worst_molecule_mse_weight": 1.0,
            "trainable_modules": ["global_shape_head"],
        },
        "single_scientific_change": (
            "replace Adam continuation by bounded deterministic full-batch L-BFGS "
            "for the same frozen global-shape regression objective"
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_M3P5B_SHAPE_LBFGS_PROTOCOL {output}")


if __name__ == "__main__":
    main()
