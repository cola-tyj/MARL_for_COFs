"""Prove that frozen H0-H3 protocols differ only by declared variables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _leaf_diff(left, right, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_leaf_diff(left[key], right[key], path))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        paths = []
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            paths.extend(
                _leaf_diff(left_value, right_value, f"{prefix}[{index}]")
            )
        return paths
    return [] if left == right else [prefix]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("h0", "h1", "h2", "h3"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name).resolve() for name in ("h0", "h1", "h2", "h3")}
    protocols = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    expected_diffs = {
        "h0_to_h1": {
            "prior.type",
            "prior.per_sample_rms_normalization",
        },
        "h1_to_h2": {
            "transport.type",
            "transport.target_phase_aligned_to_prior",
            "transport.alignment_group",
            "transport.posterior_density",
        },
        "h2_to_h3": {"objective.endpoint_weight"},
    }
    pairs = (("h0", "h1"), ("h1", "h2"), ("h2", "h3"))
    actual = {
        f"{left}_to_{right}": _leaf_diff(protocols[left], protocols[right])
        for left, right in pairs
    }
    checks = {
        name: set(actual[name]) == allowed
        for name, allowed in expected_diffs.items()
    }
    shared_checks = {
        "same_seed": len({protocol["seed"] for protocol in protocols.values()}) == 1,
        "same_model": len(
            {json.dumps(protocol["model"], sort_keys=True) for protocol in protocols.values()}
        )
        == 1,
        "same_tiers_and_gates": len(
            {json.dumps(protocol["tiers"], sort_keys=True) for protocol in protocols.values()}
        )
        == 1,
        "same_data": len(
            {json.dumps(protocol["data"], sort_keys=True) for protocol in protocols.values()}
        )
        == 1,
    }
    passed = bool(all(checks.values()) and all(shared_checks.values()))
    report = {
        "schema_version": "pg-orbitflow-h-protocol-audit-v1",
        "status": "PASS_H1_H3_SINGLE_VARIABLE_PROTOCOL_AUDIT"
        if passed
        else "FAIL_H1_H3_PROTOCOL_DIFF",
        "passed": passed,
        "protocols": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "adjacent_diffs": actual,
        "expected_adjacent_diffs": {
            name: sorted(values) for name, values in expected_diffs.items()
        },
        "checks": {**checks, **shared_checks},
        "scientific_variables": {
            "h1": "projected Gaussian -> unnormalized graph-harmonic prior",
            "h2": "single phase -> centralizer-averaged transport target",
            "h3": "endpoint auxiliary weight 0.0 -> 1.0",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit(json.dumps(report["checks"], sort_keys=True))
    print(f"{report['status']} report={args.output.resolve()}")


if __name__ == "__main__":
    main()
