"""Recompute every frozen decoder start for one failed M3.3 molecule."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..c0_local_recovery import _sha256
from ..geometry import build_geometry_contract
from ..graph_automorphism import build_graph_automorphism_contract
from ..m0_oracle_reconstruction import (
    build_oracle_targets,
    oracle_energy,
    reconstruction_metrics,
)
from ..m2_torsion_training import _learned_decoder_targets
from ..m3_tier32_training import _decoder_task
from ..m3p1_automorphism_training import _matched_metrics
from ..orbit_kinematics import build_orbit_parameterization
from ..overfit import _panel_samples


def _passes(record: dict, gate: dict, *, all_values_finite: bool) -> bool:
    return bool(
        all_values_finite
        and record["bond_mae_angstrom"] <= gate["bond_mae_max_angstrom"]
        and record["angle_mae_degrees"] <= gate["angle_mae_max_degrees"]
        and record["torsion_circular_mae_degrees"]
        <= gate["torsion_circular_mae_max_degrees"]
        and record["nonplanar_torsion_circular_mae_degrees"]
        <= gate["nonplanar_torsion_circular_mae_max_degrees"]
        and record["ring_closure_mae_angstrom"]
        <= gate["ring_closure_mae_max_angstrom"]
        and record["chirality_preserved_fraction"]
        >= gate["chirality_preserved_fraction_min"]
        and record["collision_free"]
        and record["max_atom_error_angstrom"]
        <= gate["max_operation_atom_error_max_angstrom"]
        and record["kabsch_rmsd_angstrom"] <= gate["kabsch_rmsd_max_angstrom"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prediction-npz", type=Path, required=True)
    parser.add_argument("--package-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    prediction_path = args.prediction_npz.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    sample = next(
        row
        for row in _panel_samples(protocol, tier)
        if row.package_index == args.package_index
    )
    geometry = build_geometry_contract(sample)
    automorphism = build_graph_automorphism_contract(sample, geometry)
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = {
            name: archive[f"{sample.package_index}_{name}"]
            for name in ("bond_lengths", "angle_cosines", "torsion_sincos")
        }
    decoder_targets = _learned_decoder_targets(prediction, geometry)
    specification = build_orbit_parameterization(sample)
    decoder_protocol = {
        key: protocol["decoder"][key]
        for key in ("initialization", "optimization", "energy_weights")
    }
    starts = int(protocol["decoder"]["initialization"]["starts_per_molecule"])
    stride = int(
        protocol["decoder"]["initialization"]["molecule_specific_seed_stride"]
    )
    tasks = [
        (
            sample,
            geometry,
            decoder_targets,
            specification,
            decoder_protocol,
            int(protocol["seed"]) + sample.package_index * stride + start,
            start,
        )
        for start in range(starts)
    ]
    context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(
        max_workers=int(protocol["decoder"]["parallel_workers"]),
        mp_context=context,
    ) as executor:
        for _, start, candidate, coordinates in executor.map(
            _decoder_task, tasks, chunksize=1
        ):
            import torch

            centralizer_energies = []
            for permutation_index, permutation in enumerate(
                automorphism.allowed_atom_permutations
            ):
                terms = oracle_energy(
                    torch.as_tensor(
                        coordinates[permutation], dtype=torch.float64
                    ),
                    sample,
                    geometry,
                    decoder_targets,
                    protocol["decoder"]["energy_weights"],
                )
                centralizer_energies.append(
                    (float(terms["total"]), int(permutation_index))
                )
            invariant_energy, invariant_permutation = min(centralizer_energies)
            set_energies = []
            for permutation_index, torsion_permutation in enumerate(
                automorphism.allowed_torsion_permutations
            ):
                orbit_values = prediction["torsion_sincos"][torsion_permutation]
                variant_targets = replace(
                    decoder_targets,
                    torsion_sincos=orbit_values[
                        np.asarray(geometry.torsion_orbit_id, dtype=np.int64)
                    ],
                )
                terms = oracle_energy(
                    torch.as_tensor(coordinates, dtype=torch.float64),
                    sample,
                    geometry,
                    variant_targets,
                    protocol["decoder"]["energy_weights"],
                )
                set_energies.append((float(terms["total"]), int(permutation_index)))
            set_energy, set_permutation = min(set_energies)
            raw = reconstruction_metrics(
                sample,
                geometry,
                build_oracle_targets(sample, geometry),
                coordinates,
            )
            matched, assignment = _matched_metrics(
                sample, geometry, automorphism, coordinates, raw
            )
            assignment_records = []
            for permutation_index, permutation in enumerate(
                automorphism.allowed_atom_permutations
            ):
                metrics = reconstruction_metrics(
                    sample,
                    geometry,
                    build_oracle_targets(sample, geometry),
                    coordinates[permutation],
                )
                for key in (
                    "collision_free",
                    "collision_threshold_ratio",
                    "minimum_covalent_radius_ratio",
                    "minimum_nonbonded_distance_angstrom",
                    "max_atom_error_angstrom",
                    "max_operation_rms_angstrom",
                    "mean_operation_rms_angstrom",
                ):
                    metrics[key] = raw[key]
                assignment_records.append(
                    {
                        "permutation_index": int(permutation_index),
                        "passes_individual_geometry_gate": _passes(
                            metrics,
                            protocol["reconstruction_gate"],
                            all_values_finite=bool(candidate["all_values_finite"]),
                        ),
                        "metrics": metrics,
                    }
                )
            passing_assignments = [
                row
                for row in assignment_records
                if row["passes_individual_geometry_gate"]
            ]
            results.append(
                {
                    "start": int(start),
                    "seed": int(candidate["seed"]),
                    "learned_target_final_energy": float(candidate["final_energy"]),
                    "centralizer_invariant_learned_energy": invariant_energy,
                    "centralizer_invariant_permutation_index": invariant_permutation,
                    "set_valued_learned_energy": set_energy,
                    "set_valued_torsion_permutation_index": set_permutation,
                    "learned_target_energy_terms": candidate["energy_terms"],
                    "raw_actual_metrics": raw,
                    "matched_actual_metrics": matched,
                    "centralizer_assignment": assignment,
                    "passing_centralizer_assignment_count": len(
                        passing_assignments
                    ),
                    "passing_centralizer_assignment_indices": [
                        row["permutation_index"] for row in passing_assignments
                    ],
                    "centralizer_assignment_records": assignment_records,
                    "passes_individual_geometry_gate": _passes(
                        matched,
                        protocol["reconstruction_gate"],
                        all_values_finite=bool(candidate["all_values_finite"]),
                    ),
                    "coordinates": coordinates,
                }
            )
    results.sort(key=lambda row: row["start"])
    energy_choice = min(
        results, key=lambda row: (row["learned_target_final_energy"], row["start"])
    )
    invariant_choice = min(
        results,
        key=lambda row: (
            row["centralizer_invariant_learned_energy"],
            row["start"],
        ),
    )
    set_choice = min(
        results,
        key=lambda row: (row["set_valued_learned_energy"], row["start"]),
    )
    passing = [row for row in results if row["passes_individual_geometry_gate"]]
    report = {
        "schema_version": "pg-orbitflow-m3p3-decoder-failure-audit-v1",
        "status": (
            "DIAGNOSED_M3P3_DECODER_ENERGY_RANKING_FAILURE"
            if passing and not energy_choice["passes_individual_geometry_gate"]
            else (
                "DIAGNOSED_M3P3_DECODER_START_COVERAGE_FAILURE"
                if not passing
                else "PASS_M3P3_DECODER_SELECTED_START"
            )
        ),
        "package_index": int(sample.package_index),
        "target_pg": sample.target_pg,
        "candidate_count": len(results),
        "passing_candidate_count": len(passing),
        "passing_start_indices": [row["start"] for row in passing],
        "energy_selected_start": energy_choice["start"],
        "energy_selected_passed": energy_choice["passes_individual_geometry_gate"],
        "centralizer_invariant_selected_start": invariant_choice["start"],
        "centralizer_invariant_selected_passed": invariant_choice[
            "passes_individual_geometry_gate"
        ],
        "set_valued_selected_start": set_choice["start"],
        "set_valued_selected_passed": set_choice[
            "passes_individual_geometry_gate"
        ],
        "records": [
            {key: value for key, value in row.items() if key != "coordinates"}
            for row in results
        ],
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "prediction_npz_sha256": _sha256(prediction_path),
            "auditor_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    coordinates = np.stack([row["coordinates"] for row in results])
    np.savez_compressed(
        output_dir / "candidate_coordinates.npz",
        coordinates=coordinates.astype(np.float32),
        starts=np.asarray([row["start"] for row in results], dtype=np.int64),
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{report['status']} report={output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
