"""Combine the physical and point-group GFN2-xTB audit into one frozen gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT,
    _atomic_json,
    _fingerprint,
    _sha256,
)


DEFAULT_PROTOCOL = ROOT / "generative_model/evaluation/our_etflow_xtb_protocol_v1.json"
DEFAULT_RUN = ROOT / "generative_model/runs/our_etflow_xtb_relaxation_v1"


def summarize(run_dir: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_fingerprint") != _fingerprint(protocol):
        raise RuntimeError("xTB protocol fingerprint mismatch")
    relaxation_path = run_dir / "relaxation_report.json"
    pg_path = run_dir / "point_group_report.json"
    relaxation = json.loads(relaxation_path.read_text(encoding="utf-8"))
    pg = json.loads(pg_path.read_text(encoding="utf-8"))
    metrics = {
        **relaxation["summary_pre_point_group"],
        "pg_compatible_before_fraction_full_panel": pg["summary"][
            "before_pg_compatible_fraction_full_panel"
        ],
        "pg_compatible_after_fraction_full_panel": pg["summary"][
            "after_pg_compatible_fraction_full_panel"
        ],
        "pg_exact_match_after_fraction_full_panel": pg["summary"][
            "after_pg_exact_match_fraction_full_panel"
        ],
    }
    gate = protocol["gate"]
    checks = {
        "full_frozen_panel": bool(relaxation["full_frozen_panel"]),
        "calculation_success_fraction_min": (
            metrics["calculation_success_fraction_full_panel"]
            >= float(gate["calculation_success_fraction_min"])
        ),
        "energy_nonincreasing_fraction_min": (
            metrics["energy_nonincreasing_fraction_among_success"]
            >= float(gate["energy_nonincreasing_fraction_among_success_min"])
        ),
        "collision_free_after_fraction_min": (
            metrics["collision_free_after_fraction_full_panel"]
            >= float(gate["collision_free_after_fraction_full_panel_min"])
        ),
        "pg_compatible_after_fraction_min": (
            metrics["pg_compatible_after_fraction_full_panel"]
            >= float(gate["pg_compatible_after_fraction_full_panel_min"])
        ),
        "median_kabsch_rmsd_max": (
            metrics["median_kabsch_rmsd_pre_to_post_angstrom"] is not None
            and metrics["median_kabsch_rmsd_pre_to_post_angstrom"]
            <= float(gate["median_kabsch_rmsd_pre_to_post_angstrom_max"])
        ),
        "all_successful_values_finite": bool(metrics["all_successful_values_finite"]),
        "hard_projection_not_used_during_relaxation": not bool(
            protocol["relaxation"]["hard_projection_during_relaxation"]
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "our-etflow-gfn2-xtb-summary-v1",
        "status": (
            "PASS_OUR_ETFLOW_GFN2_XTB_PHYSICAL_STABILITY_PILOT"
            if passed else "FAIL_OUR_ETFLOW_GFN2_XTB_DIAGNOSE"
        ),
        "passed": passed,
        "quality_claim": False,
        "metrics": metrics,
        "by_target_pg": pg["summary"]["by_target_pg"],
        "checks": checks,
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "relaxation_report_sha256": _sha256(relaxation_path),
            "point_group_report_sha256": _sha256(pg_path),
            "relaxed_coordinates_sha256": _sha256(run_dir / "relaxed_coordinates.npz"),
            "summarizer_source_sha256": _sha256(Path(__file__)),
        },
    }
    _atomic_json(run_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    result = summarize(args.run_dir.resolve(), args.protocol.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
