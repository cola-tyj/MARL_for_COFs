"""Freeze the C0-v1 execution diagnosis and formal C0-v2 branch decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("generative_model/PG_OrbitFlow")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    v1_dir = root / "runs" / "c0_local_recovery_v1"
    v2_dir = root / "runs" / "c0_local_recovery_v2_pg_balanced"
    v1 = _read(v1_dir / "report.json")
    v2 = _read(v2_dir / "report.json")
    v1_exposures = v1["data"]["curriculum_exposures"]
    v2_exposures = v2["data"]["curriculum_exposures"]
    v1_confound = (
        v1_exposures.get("C2_harmonic", 0) == 0
        and v1_exposures.get("C3_harmonic", 0) > 0
    )
    v2_balanced = all(
        v2_exposures[f"C2_{kind}"] == v2_exposures[f"C3_{kind}"]
        for kind in ("small", "medium", "large", "harmonic")
    )
    if not v1_confound or not v2_balanced:
        raise RuntimeError("C0 execution diagnosis no longer matches the frozen runs")
    diagnosis = {
        "schema_version": "pg-orbitflow-c0-v1-execution-diagnosis-v1",
        "status": "INVALIDATE_C0_V1_SCIENTIFIC_CONCLUSION_RETAIN_EXECUTION_AUDIT",
        "v1_report_sha256": _sha256(v1_dir / "report.json"),
        "cause": (
            "A global ten-slot curriculum cursor was correlated with deterministic "
            "C2/C3 alternation; every harmonic slot was assigned to C3."
        ),
        "v1_curriculum_exposures": v1_exposures,
        "checkpoint_written_before_diagnosis": True,
        "scientific_conclusion_valid": False,
        "artifact_policy": "retain v1 as an execution-bug audit; never use as parent",
        "formal_replacement": "c0_local_recovery_v2_pg_balanced",
        "sole_execution_fix": (
            "Use independent deterministic ten-slot curriculum cursors for C2 and C3."
        ),
    }
    diagnosis_path = root / "reports" / "c0_v1_sampling_correlation_diagnosis.json"
    _dump(diagnosis_path, diagnosis)

    local = v2["local_gate"]
    original = v2["original_tier4_gate_unchanged"]
    failed_local = sorted(key for key, value in local["checks"].items() if not value)
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(v2_dir.iterdir())
        if path.is_file()
    }
    summary = {
        "schema_version": "pg-orbitflow-c0-local-recovery-summary-v1",
        "status": v2["status"],
        "passed": v2["passed"],
        "formal_run": "c0_local_recovery_v2_pg_balanced",
        "protocol": v2["protocol"],
        "training": {
            **v2["training"],
            "molecule_count": v2["data"]["molecule_count"],
            "molecule_exposures": v2["data"]["molecule_exposures"],
            "curriculum_exposures": v2_exposures,
        },
        "local_gate": local,
        "failed_local_checks": failed_local,
        "chirality_gate_interpretation": (
            "non-informative on this panel because active_chirality_case_count=0"
        ),
        "original_tier4_gate_unchanged": original,
        "decision": v2["decision"],
        "tier16_started": False,
        "tier32_started": False,
        "next_route": "fixed-graph bond/angle/torsion internal-coordinate decoder",
        "v1_execution_diagnosis_sha256": _sha256(diagnosis_path),
        "reproducibility_report_sha256": _sha256(
            root / "reports" / "c0_reproducibility_v1.json"
        ),
        "artifacts": artifacts,
    }
    output = root / "reports" / "c0_local_recovery_v2_summary.json"
    _dump(output, summary)
    print(f"{summary['status']} report={output}")


if __name__ == "__main__":
    main()
