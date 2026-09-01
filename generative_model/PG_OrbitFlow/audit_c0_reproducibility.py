"""Repeat the formal C0 local/raw evaluations and require array-exact results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .c0_local_recovery import _load_protocol, _local_evaluation, _sha256
from .geometry import build_geometry_contract
from .model import PGOrbitFlow
from .overfit import _panel_samples, _raw_evaluation


def _array_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        values = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _compare(first_metrics, first_arrays, second_metrics, second_arrays) -> dict:
    array_checks = {
        key: bool(np.array_equal(first_arrays[key], second_arrays[key]))
        for key in sorted(first_arrays)
    }
    metrics_exact = first_metrics == second_metrics
    return {
        "array_exact_checks": array_checks,
        "array_sha256_repeat_1": _array_sha256(first_arrays),
        "array_sha256_repeat_2": _array_sha256(second_arrays),
        "metrics_exact": metrics_exact,
        "passed": bool(all(array_checks.values()) and metrics_exact),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True, warn_only=False)
    protocol_path = args.protocol.resolve()
    checkpoint_path = args.checkpoint.resolve()
    protocol = _load_protocol(protocol_path)
    tier_config = {
        "records": protocol["panel"]["records"],
        "evaluation": protocol["original_tier4_evaluation"],
        "gate": protocol["original_tier4_gate"],
    }
    samples = _panel_samples(protocol, tier_config)
    contracts = {
        sample.package_index: build_geometry_contract(sample) for sample in samples
    }
    model = PGOrbitFlow(
        **{
            key: protocol["model"][key]
            for key in ("hidden_dim", "layers", "radial_dim", "radial_max")
        }
    ).to(args.device)
    state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    if state.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("C0 checkpoint/protocol mismatch")
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    local_1, local_arrays_1 = _local_evaluation(
        model, samples, contracts, protocol, args.device
    )
    local_2, local_arrays_2 = _local_evaluation(
        model, samples, contracts, protocol, args.device
    )
    raw_1, raw_arrays_1 = _raw_evaluation(
        model, samples, protocol, tier_config, args.device
    )
    raw_2, raw_arrays_2 = _raw_evaluation(
        model, samples, protocol, tier_config, args.device
    )
    checks = {
        "local": _compare(local_1, local_arrays_1, local_2, local_arrays_2),
        "raw": _compare(raw_1, raw_arrays_1, raw_2, raw_arrays_2),
    }
    passed = all(check["passed"] for check in checks.values())
    report = {
        "schema_version": "pg-orbitflow-c0-reproducibility-v1",
        "status": "PASS_C0_REPRODUCIBILITY" if passed else "FAIL_C0_REPRODUCIBILITY",
        "passed": passed,
        "device": args.device,
        "same_process": True,
        "deterministic_algorithms": True,
        "protocol_sha256": _sha256(protocol_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checks": checks,
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
