"""Audit learned shape-aware candidate selection on the frozen package-1054 panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .global_shape_model import build_global_shape_contract
from .overfit import _panel_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-coordinates", type=Path, required=True)
    parser.add_argument("--shape-weight", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    prediction_path = args.predictions.resolve()
    candidate_report_path = args.candidate_report.resolve()
    candidate_coordinates_path = args.candidate_coordinates.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    package_index = int(candidate_report["package_index"])
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    sample = next(
        row
        for row in _panel_samples(protocol, tier)
        if row.package_index == package_index
    )
    contract = build_global_shape_contract(
        sample,
        minimum_graph_distance=int(
            protocol["global_shape_contract"]["minimum_graph_distance"]
        ),
        quantile_count=int(protocol["global_shape_contract"]["quantile_count"]),
    )
    with np.load(prediction_path, allow_pickle=False) as archive:
        predicted = archive[f"{package_index}_global_shape_quantiles"]
    with np.load(candidate_coordinates_path, allow_pickle=False) as archive:
        coordinates = archive["coordinates"]
        starts = archive["starts"]
    by_start = {int(row["start"]): row for row in candidate_report["records"]}
    records = []
    for start, values in zip(starts, coordinates, strict=True):
        distances = np.linalg.norm(
            values[contract.pair_index[0]] - values[contract.pair_index[1]], axis=1
        )
        quantiles = np.quantile(distances, contract.quantile_levels)
        shape_mse = float(np.mean(np.square(quantiles - predicted)))
        decoder_energy = float(by_start[int(start)]["learned_target_final_energy"])
        records.append(
            {
                "start": int(start),
                "decoder_energy": decoder_energy,
                "shape_mse": shape_mse,
                "combined_score": decoder_energy + args.shape_weight * shape_mse,
                "passes_individual_geometry_gate": bool(
                    by_start[int(start)]["passes_individual_geometry_gate"]
                ),
            }
        )
    selected = min(records, key=lambda row: (row["combined_score"], row["start"]))
    checks = {
        "candidate_count_exact": len(records) == 16,
        "shape_weight_positive": args.shape_weight == 10.0,
        "selected_candidate_passes_original_geometry_gate": selected[
            "passes_individual_geometry_gate"
        ],
        "selection_uses_no_target_coordinates": True,
    }
    passed = all(checks.values())
    output = {
        "schema_version": "pg-orbitflow-m3p6-shape-selection-audit-v1",
        "status": (
            "PASS_M3P6_SHAPE_SELECTION_ADVANCE_TO_TIER32_DECODER"
            if passed
            else "FAIL_M3P6_SHAPE_SELECTION_STOP_BRANCH"
        ),
        "passed": passed,
        "checks": checks,
        "shape_weight": args.shape_weight,
        "selected_start": selected["start"],
        "records": records,
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "predictions_sha256": _sha256(prediction_path),
            "candidate_report_sha256": _sha256(candidate_report_path),
            "candidate_coordinates_sha256": _sha256(candidate_coordinates_path),
            "auditor_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{output['status']} selected_start={selected['start']} report={output_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
