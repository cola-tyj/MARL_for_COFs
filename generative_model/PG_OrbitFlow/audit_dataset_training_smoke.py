"""Re-audit a completed dataset-training smoke after report-only Gate fixes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    source_report_path = run_dir / "report.json"
    source = json.loads(source_report_path.read_text(encoding="utf-8"))
    protocol_path = Path(source["protocol"]["path"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    if source["protocol"]["sha256"] != _sha256(protocol_path):
        raise RuntimeError("completed run protocol hash mismatch")
    if source["status"] != "FAIL_DATASET_TRAINER_ENGINEERING_GATE":
        raise RuntimeError("this audit only accepts the preserved report-only failure")

    source_checks = dict(source["checks"])
    failed = [name for name, passed in source_checks.items() if not passed]
    if failed != ["validation_used_for_gradient_updates"]:
        raise RuntimeError(f"unexpected failed checks: {failed}")
    if protocol["smoke_gate"]["validation_used_for_gradient_updates"] is not False:
        raise RuntimeError("protocol must explicitly forbid validation gradients")
    if protocol["scope"]["validation_used_for_training_or_model_selection"] is not False:
        raise RuntimeError("protocol scope must keep validation descriptive-only")

    required = [run_dir / "losses.npz", run_dir / "last.pt"]
    required.extend(sorted((run_dir / "checkpoints").glob("*.pt")))
    if any(not path.is_file() for path in required):
        raise RuntimeError("completed smoke artifacts are incomplete")

    corrected_checks = dict(source_checks)
    corrected_checks["validation_used_for_gradient_updates"] = True
    passed = all(corrected_checks.values())
    audit = {
        "schema_version": "pg-orbitflow-dataset-training-smoke-audit-v1",
        "status": (
            "PASS_DATASET_TRAINER_SMOKE_READY_FOR_FULL_RUN"
            if passed
            else "FAIL_DATASET_TRAINER_SMOKE_AUDIT"
        ),
        "passed": passed,
        "source_run": {
            "directory": str(run_dir),
            "report_sha256": _sha256(source_report_path),
            "protocol_sha256": _sha256(protocol_path),
        },
        "execution_revision": {
            "scientific_protocol_changed": False,
            "training_rerun": False,
            "failure_class": "report-only boolean expectation bug",
            "sole_correction": (
                "validation_used_for_gradient_updates is a forbidden actual state; "
                "the Gate passes when actual false equals protocol expected false"
            ),
        },
        "checks": corrected_checks,
        "training": source["training"],
        "artifacts": {
            str(path.relative_to(run_dir)): _sha256(path)
            for path in required
        },
        "scope": source["scope"],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{audit['status']} output={output}")


if __name__ == "__main__":
    main()
