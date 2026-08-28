"""Create presentation-ready paired tables from the four-route ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from generative_model.evaluation.our_etflow_ablation import ROUTES
from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT,
    _atomic_json,
    _sha256,
)


DEFAULT_RUN = ROOT / "generative_model/runs/our_etflow_ablation_v1"


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return None if not values else float(np.mean(values))


def _median(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return None if not values else float(np.median(values))


def summarize(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    pg_path = run_dir / "point_group_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    point_group = json.loads(pg_path.read_text(encoding="utf-8"))
    if len(report["records"]) != len(point_group["records"]):
        raise RuntimeError("geometry/point-group record count mismatch")
    rows, route_summary = [], {}
    for route in ROUTES:
        geometry_records, pg_records = [], []
        for geometry, pg in zip(
            report["records"], point_group["records"], strict=True
        ):
            if geometry["package_index"] != pg["package_index"]:
                raise RuntimeError("geometry/point-group identity mismatch")
            geo_route = geometry["routes"][route]
            pg_route = pg["routes"][route]
            if geo_route["success"]:
                geometry_records.append(geo_route)
            pg_records.append(pg_route)
        full_count = len(report["records"])
        summary = {
            "route": route,
            "full_panel_count": full_count,
            "generation_success_fraction": len(geometry_records) / full_count,
            "analyzer_success_fraction": sum(
                bool(value["analyzer_success"]) for value in pg_records
            ) / full_count,
            "pg_compatible_fraction": sum(
                bool(value["pg_compatible"]) for value in pg_records
            ) / full_count,
            "pg_exact_match_fraction": sum(
                bool(value["pg_exact_match"]) for value in pg_records
            ) / full_count,
            "collision_free_fraction": sum(
                bool(value.get("collision_free_at_0p6")) for value in geometry_records
            ) / full_count,
            "mean_bond_length_mae_angstrom": _mean(
                geometry_records, "bond_length_mae_to_canonical_angstrom"
            ),
            "median_bond_length_mae_angstrom": _median(
                geometry_records, "bond_length_mae_to_canonical_angstrom"
            ),
            "mean_uff_energy_per_atom_kcal_mol": _mean(
                geometry_records, "uff_single_point_energy_per_atom_kcal_mol"
            ),
            "median_uff_energy_per_atom_kcal_mol": _median(
                geometry_records, "uff_single_point_energy_per_atom_kcal_mol"
            ),
            "mean_runtime_seconds": _mean(geometry_records, "runtime_seconds"),
            "median_runtime_seconds": _median(geometry_records, "runtime_seconds"),
            "mean_candidate_kabsch_diversity_angstrom": _mean(
                [value["diversity"] for value in geometry_records],
                "mean_pairwise_kabsch_rmsd_angstrom",
            ),
            "mean_candidate_pair_distance_diversity_angstrom": _mean(
                [value["diversity"] for value in geometry_records],
                "mean_pairwise_distance_mae_angstrom",
            ),
            "by_target_pg": point_group["summary"][route]["by_target_pg"],
        }
        route_summary[route] = summary
        rows.append(summary)

    pairs = {
        "hard_projection_minus_raw": (
            "etflow_hard_projection", "etflow_raw"
        ),
        "f02_minus_hard_projection": (
            "etflow_hard_projection_f02", "etflow_hard_projection"
        ),
        "full_f02_minus_etkdg": (
            "etflow_hard_projection_f02", "etkdg_v3_best_of_n"
        ),
    }
    paired_effects = {}
    for name, (candidate, parent) in pairs.items():
        paired_effects[name] = {
            key: float(route_summary[candidate][key] - route_summary[parent][key])
            for key in (
                "generation_success_fraction",
                "pg_compatible_fraction",
                "collision_free_fraction",
                "mean_bond_length_mae_angstrom",
                "median_uff_energy_per_atom_kcal_mol",
                "median_runtime_seconds",
                "mean_candidate_kabsch_diversity_angstrom",
            )
        }
    output = {
        "schema_version": "our-etflow-paired-ablation-summary-v1",
        "status": "PASS_PAIRED_ABLATION_COMPLETED",
        "quality_claim": True,
        "route_summary": route_summary,
        "paired_effects": paired_effects,
        "interpretation_contract": {
            "pg_compatible_primary": True,
            "pg_exact_match_descriptive": True,
            "failed_generation_uses_full_panel_denominator": True,
            "reference_xyz_used_only_for_posthoc_error_metrics": True,
            "runtime_machine_dependent": True,
            "uff_energy_compared_paired_within_graph": True,
        },
        "identity": {
            "source_report_sha256": _sha256(report_path),
            "point_group_report_sha256": _sha256(pg_path),
            "coordinates_sha256": _sha256(run_dir / "coordinates.npz"),
            "summarizer_source_sha256": _sha256(Path(__file__)),
        },
    }
    json_path = run_dir / "paired_ablation_summary.json"
    _atomic_json(json_path, output)
    csv_path = run_dir / "paired_ablation_table.csv"
    scalar_keys = [
        key for key, value in rows[0].items() if not isinstance(value, dict)
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in scalar_keys})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    output = summarize(args.run_dir.resolve())
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
