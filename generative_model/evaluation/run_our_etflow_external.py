"""Strict wrapper for the external-to-COF graph package evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generative_model.evaluation.run_our_etflow_ablation import run
from generative_model.inference.generate_etflow_symmetric_xyz import (
    DEFAULT_CACHE,
    ROOT,
    _sha256,
)


DEFAULT_PROTOCOL = ROOT / "generative_model/evaluation/our_etflow_external_protocol_v1.json"
DEFAULT_PACKAGE = ROOT / "generative_model/data/external_symmetric_v1"
DEFAULT_OUTPUT = ROOT / "generative_model/runs/our_etflow_external_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    package = args.package.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol["identity"]["external_package_sha256"]
    for filename, digest in expected.items():
        observed = _sha256(package / filename)
        if observed != digest:
            raise RuntimeError(
                f"external package hash mismatch: {filename}: {observed} != {digest}"
            )
    report = run(
        protocol_path=protocol_path,
        package_path=package,
        output_dir=args.output_dir.resolve(),
        cache=args.cache.resolve(),
        device=args.device,
    )
    print(json.dumps({
        "status": report["status"],
        "panel_count": report["panel_count"],
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
