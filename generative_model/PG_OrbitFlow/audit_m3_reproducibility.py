"""Independently recompute every selected M3 Tier-32 decoder start."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .global_shape_model import (
    build_global_shape_contract,
    coordinate_shape_quantiles,
)
from .graph_automorphism import build_graph_automorphism_contract
from .m0_oracle_reconstruction import build_oracle_targets, reconstruction_metrics
from .m2_torsion_training import _learned_decoder_targets
from .m2p1_phase_training import _reconstruction_checks
from .m3_tier32_training import _decoder_task
from .m3p1_automorphism_training import _matched_metrics
from .orbit_kinematics import build_orbit_parameterization
from .overfit import _panel_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    protocol_path = args.protocol.resolve()
    run_dir = args.run_dir.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report_path = run_dir / "report.json"
    coordinates_path = run_dir / "coordinates.npz"
    predictions_path = run_dir / "predictions.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_M3_TIER32":
        raise RuntimeError("M3 reproducibility requires the passed formal report")
    selection = {
        int(row["package_index"]): row
        for row in report["candidate_selection_records"]
    }
    with np.load(predictions_path, allow_pickle=False) as archive:
        prediction_archive = {name: archive[name] for name in archive.files}
    with np.load(coordinates_path, allow_pickle=False) as archive:
        stored_coordinates = archive["coordinates"]
        stored_offsets = archive["atom_offsets"]
        stored_indices = archive["package_indices"]
    stored = {
        int(package_index): stored_coordinates[stored_offsets[index] : stored_offsets[index + 1]]
        for index, package_index in enumerate(stored_indices)
    }
    shape_setting = protocol["global_shape_contract"]
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    examples = []
    tasks = []
    stride = int(
        protocol["decoder"]["initialization"]["molecule_specific_seed_stride"]
    )
    decoder_protocol = {
        key: protocol["decoder"][key]
        for key in ("initialization", "optimization", "energy_weights")
    }
    for sample in _panel_samples(protocol, tier):
        geometry = build_geometry_contract(sample)
        automorphism = build_graph_automorphism_contract(sample, geometry)
        shape = build_global_shape_contract(
            sample,
            minimum_graph_distance=int(shape_setting["minimum_graph_distance"]),
            quantile_count=int(shape_setting["quantile_count"]),
        )
        prediction = {
            name: prediction_archive[f"{sample.package_index}_{name}"]
            for name in (
                "bond_lengths",
                "angle_cosines",
                "torsion_sincos",
                "global_shape_quantiles",
            )
        }
        start = int(selection[sample.package_index]["selected_start_index"])
        task = (
            sample,
            geometry,
            _learned_decoder_targets(prediction, geometry),
            build_orbit_parameterization(sample),
            decoder_protocol,
            int(protocol["seed"]) + sample.package_index * stride + start,
            start,
        )
        examples.append((sample, geometry, automorphism, shape, prediction, start))
        tasks.append(task)
    context = multiprocessing.get_context("spawn")
    results = {}
    with ProcessPoolExecutor(
        max_workers=int(protocol["decoder"]["parallel_workers"]),
        mp_context=context,
    ) as executor:
        for package_index, start, candidate, coordinates in executor.map(
            _decoder_task, tasks, chunksize=1
        ):
            results[int(package_index)] = (int(start), candidate, coordinates)
    records = []
    matched_records = []
    coordinate_max = 0.0
    energy_max = 0.0
    shape_mse_max = 0.0
    score_max = 0.0
    weight = float(protocol["candidate_selection"]["shape_weight"])
    for sample, geometry, automorphism, shape, prediction, expected_start in examples:
        start, candidate, coordinates = results[sample.package_index]
        expected = selection[sample.package_index]
        quantiles = coordinate_shape_quantiles(
            torch.as_tensor(coordinates, dtype=torch.float64), shape
        )
        target = torch.as_tensor(
            prediction["global_shape_quantiles"], dtype=torch.float64
        )
        shape_mse = float(torch.mean(torch.square(quantiles - target)))
        score = float(candidate["final_energy"]) + weight * shape_mse
        coordinate_error = float(
            np.max(np.abs(coordinates - stored[sample.package_index]))
        )
        energy_error = abs(
            float(candidate["final_energy"])
            - float(expected["scores"][expected_start] - weight * expected["shape_mses"][expected_start])
        )
        shape_error = abs(shape_mse - float(expected["shape_mses"][expected_start]))
        score_error = abs(score - float(expected["scores"][expected_start]))
        coordinate_max = max(coordinate_max, coordinate_error)
        energy_max = max(energy_max, energy_error)
        shape_mse_max = max(shape_mse_max, shape_error)
        score_max = max(score_max, score_error)
        raw = reconstruction_metrics(
            sample,
            geometry,
            build_oracle_targets(sample, geometry),
            coordinates,
        )
        common = {
            "package_index": int(sample.package_index),
            "target_pg": sample.target_pg,
            "selected_start_index": start,
            "selected_energy": float(candidate["final_energy"]),
            "selected_shape_mse": shape_mse,
            "selected_combined_score": score,
            "all_starts_finite": bool(candidate["all_values_finite"]),
        }
        matched, _ = _matched_metrics(
            sample, geometry, automorphism, coordinates, raw
        )
        matched_records.append({**common, **matched})
        records.append(
            {
                "package_index": int(sample.package_index),
                "start_matches": start == expected_start,
                "coordinate_max_abs_difference_angstrom": coordinate_error,
                "energy_abs_difference": energy_error,
                "shape_mse_abs_difference": shape_error,
                "combined_score_abs_difference": score_error,
            }
        )
    values, gate_checks, strata = _reconstruction_checks(
        matched_records, protocol["reconstruction_gate"]
    )
    checks = {
        "molecule_count_exact": len(records) == 32,
        "all_selected_starts_match": all(row["start_matches"] for row in records),
        "coordinate_max_abs_difference_le_1e_5": coordinate_max <= 1e-5,
        "energy_abs_difference_le_1e_8": energy_max <= 1e-8,
        "shape_mse_abs_difference_le_1e_8": shape_mse_max <= 1e-8,
        "combined_score_abs_difference_le_1e_8": score_max <= 1e-8,
        "original_reconstruction_gate_repasses": all(gate_checks.values()),
    }
    passed = all(checks.values())
    output = {
        "schema_version": "pg-orbitflow-m3-selected-start-reproducibility-v1",
        "status": "PASS_M3_REPRODUCIBILITY" if passed else "FAIL_M3_REPRODUCIBILITY",
        "passed": passed,
        "checks": checks,
        "maximum_differences": {
            "coordinate_angstrom": coordinate_max,
            "energy": energy_max,
            "shape_mse": shape_mse_max,
            "combined_score": score_max,
        },
        "records": records,
        "reconstruction_gate": {
            "passed": all(gate_checks.values()),
            "checks": gate_checks,
            "values": values,
            "strata": strata,
        },
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "formal_report_sha256": _sha256(report_path),
            "formal_coordinates_sha256": _sha256(coordinates_path),
            "formal_predictions_sha256": _sha256(predictions_path),
            "auditor_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{output['status']} report={output_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
