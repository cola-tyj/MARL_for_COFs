"""Repeat H1-H3 raw sampling in one deterministic process and compare bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .diagnose_overfit import _model
from ..overfit import _load_protocol, _panel_samples, _raw_evaluation, _sha256


def _array_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        values = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("generative_model/PG_OrbitFlow")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True, warn_only=False)
    root = args.root.resolve()
    variants = ("h1_harmonic", "h2_centralizer", "h3_endpoint")
    records = {}
    for variant in variants:
        protocol_path = root / "configs" / f"overfit_protocol_{variant}.json"
        run_dir = root / "runs" / f"overfit_tier4_{variant}"
        checkpoint_path = run_dir / "last.pt"
        protocol, tier_config = _load_protocol(protocol_path, "4")
        samples = _panel_samples(protocol, tier_config)
        model, state = _model(protocol, checkpoint_path, args.device)
        if state.get("protocol_sha256") != _sha256(protocol_path):
            raise ValueError(f"checkpoint/protocol mismatch for {variant}")
        first_metrics, first_arrays = _raw_evaluation(
            model, samples, protocol, tier_config, args.device
        )
        second_metrics, second_arrays = _raw_evaluation(
            model, samples, protocol, tier_config, args.device
        )
        array_checks = {
            key: bool(np.array_equal(first_arrays[key], second_arrays[key]))
            for key in sorted(first_arrays)
        }
        records[variant] = {
            "protocol_sha256": _sha256(protocol_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "raw_array_sha256_repeat_1": _array_sha256(first_arrays),
            "raw_array_sha256_repeat_2": _array_sha256(second_arrays),
            "array_exact_checks": array_checks,
            "metrics_exact": first_metrics == second_metrics,
            "passed": bool(all(array_checks.values()) and first_metrics == second_metrics),
        }
    passed = all(record["passed"] for record in records.values())
    report = {
        "schema_version": "pg-orbitflow-h1-h3-raw-reproducibility-v1",
        "status": "PASS_H1_H3_RAW_REPRODUCIBILITY"
        if passed
        else "FAIL_H1_H3_RAW_REPRODUCIBILITY",
        "passed": passed,
        "device": args.device,
        "same_process": True,
        "deterministic_algorithms": True,
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
