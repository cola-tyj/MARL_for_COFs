"""Verify one M3 decoder task is identical in serial and spawn-worker execution."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .m2_torsion_training import _learned_decoder_targets
from .m3_tier32_training import _decoder_task
from .orbit_kinematics import build_orbit_parameterization
from .overfit import _panel_samples


def _numeric_leaves(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, current in value.items():
            result.update(_numeric_leaves(current, f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, (bool, int, float, np.number)):
        return {prefix: float(value)}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-protocol", type=Path, required=True)
    parser.add_argument("--m2-run-dir", type=Path, required=True)
    parser.add_argument("--package-index", type=int, default=864)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.m2_protocol.resolve()
    run_dir = args.m2_run_dir.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    sample = next(sample for sample in _panel_samples(protocol, tier) if sample.package_index == args.package_index)
    contract = build_geometry_contract(sample)
    with np.load(run_dir / "predictions.npz", allow_pickle=False) as archive:
        prediction = {
            name: np.asarray(archive[f"{sample.package_index}_{name}"])
            for name in ("bond_lengths", "angle_cosines", "torsion_sincos")
        }
    decoder_targets = _learned_decoder_targets(prediction, contract)
    specification = build_orbit_parameterization(sample)
    decoder_protocol = {key: protocol["decoder"][key] for key in ("initialization", "optimization", "energy_weights")}
    stride = int(protocol["decoder"]["initialization"]["molecule_specific_seed_stride"])
    task = (
        sample, contract, decoder_targets, specification, decoder_protocol,
        int(protocol["seed"]) + sample.package_index * stride + args.start_index,
        args.start_index,
    )
    serial = _decoder_task(task)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        worker = next(executor.map(_decoder_task, [task]))
    coordinate_max_abs = float(np.max(np.abs(serial[3] - worker[3])))
    serial_metrics = _numeric_leaves(serial[2])
    worker_metrics = _numeric_leaves(worker[2])
    if set(serial_metrics) != set(worker_metrics):
        raise RuntimeError("serial and worker candidate metric structure differs")
    candidate_max_abs = max(
        abs(serial_metrics[key] - worker_metrics[key]) for key in serial_metrics
    )
    passed = coordinate_max_abs <= 1e-12 and candidate_max_abs <= 1e-12 and serial[:2] == worker[:2]
    output = {
        "schema_version": "pg-orbitflow-m3-decoder-parallelism-audit-v1",
        "status": "PASS_M3_DECODER_PARALLELISM" if passed else "FAIL_M3_DECODER_PARALLELISM",
        "passed": passed,
        "package_index": sample.package_index,
        "start_index": args.start_index,
        "coordinate_max_abs_difference_angstrom": coordinate_max_abs,
        "candidate_metric_max_abs_difference": candidate_max_abs,
        "tolerance": 1e-12,
        "identity": {
            "m2_protocol_sha256": _sha256(protocol_path),
            "m2_predictions_sha256": _sha256(run_dir / "predictions.npz"),
            "m3_runner_sha256": _sha256(Path(__file__).resolve().parent / "m3_tier32_training.py"),
            "auditor_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{output['status']} report={args.output.resolve()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
