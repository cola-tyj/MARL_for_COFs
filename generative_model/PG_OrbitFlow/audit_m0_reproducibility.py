"""Recompute the four selected M0 starts and compare their saved coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import build_geometry_contract
from .m0_oracle_reconstruction import (
    _load_protocol,
    build_oracle_targets,
    optimize_one_start,
)
from .metrics import kabsch_rmsd
from .orbit_kinematics import build_orbit_parameterization
from .overfit import _panel_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = _load_protocol(args.protocol.resolve())
    run_dir = args.run_dir.resolve()
    original = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    archive = np.load(run_dir / "coordinates.npz", allow_pickle=False)
    offsets = archive["atom_offsets"]
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    samples = _panel_samples(protocol, tier)
    original_records = {
        int(record["package_index"]): record for record in original["records"]
    }
    records = []
    for index, sample in enumerate(samples):
        reference = archive["coordinates"][offsets[index] : offsets[index + 1]]
        source = original_records[sample.package_index]
        contract = build_geometry_contract(sample)
        targets = build_oracle_targets(sample, contract)
        specification = build_orbit_parameterization(sample)
        repeated, coordinates = optimize_one_start(
            sample,
            contract,
            targets,
            specification,
            protocol,
            seed=int(source["seed"]),
            device=args.device,
        )
        max_abs = float(np.max(np.abs(reference.astype(np.float64) - coordinates)))
        aligned = kabsch_rmsd(reference, coordinates)
        records.append(
            {
                "package_index": sample.package_index,
                "seed": int(source["seed"]),
                "saved_vs_repeat_max_abs_difference_angstrom": max_abs,
                "saved_vs_repeat_kabsch_rmsd_angstrom": aligned,
                "collision_state_identical": bool(
                    source["collision_free"] == repeated["collision_free"]
                ),
                "passed": bool(
                    max_abs <= 5e-7
                    and aligned <= 1e-7
                    and source["collision_free"] == repeated["collision_free"]
                ),
            }
        )
    passed = all(record["passed"] for record in records)
    report = {
        "schema_version": "pg-orbitflow-m0-selected-start-reproducibility-v1",
        "status": "PASS_M0_SELECTED_START_REPRODUCIBILITY"
        if passed
        else "FAIL_M0_SELECTED_START_REPRODUCIBILITY",
        "passed": passed,
        "device": args.device,
        "coordinate_max_abs_tolerance_angstrom": 5e-7,
        "kabsch_tolerance_angstrom": 1e-7,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit(report["status"])
    print(f"{report['status']} report={args.output.resolve()}")


if __name__ == "__main__":
    main()
