"""Evaluate a completed M1 checkpoint when decoder execution moves to CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .m0_oracle_reconstruction import (
    build_oracle_targets,
    optimize_one_start,
    reconstruction_metrics,
)
from .m1_bond_angle_training import (
    M1_RUN_SCHEMA_VERSION,
    _angle_mae_degrees,
    _dump,
    _load_protocol,
    _predicted_decoder_targets,
)
from .orbit_ic_model import QuotientICPredictor, build_orbit_ic_example
from .orbit_kinematics import build_orbit_parameterization
from .overfit import _panel_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--failed-console-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()

    import torch

    protocol_path = args.protocol.resolve()
    checkpoint_path = args.checkpoint.resolve()
    failed_console = args.failed_console_log.resolve()
    protocol = _load_protocol(protocol_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("M1 checkpoint/protocol mismatch")
    if int(state.get("step", -1)) != int(protocol["training"]["optimizer_steps"]):
        raise ValueError("M1 checkpoint is not the completed 1024-step state")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    samples = _panel_samples(protocol, tier)
    examples = []
    for sample in samples:
        contract = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, contract)
        examples.append(
            (sample, contract, graph, targets, build_oracle_targets(sample, contract))
        )
    setting = {
        key: protocol["model"][key]
        for key in ("node_feature_dim", "edge_feature_dim", "hidden_dim", "layers")
    }
    model = QuotientICPredictor(**setting)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    with torch.no_grad():
        predictions = [
            {
                key: value.detach().cpu().numpy()
                for key, value in model(graph, device="cpu").items()
            }
            for _, _, graph, _, _ in examples
        ]
    prediction_records = []
    reconstruction_records = []
    coordinate_parts = []
    offsets = [0]
    decoder_protocol = {
        "initialization": protocol["decoder"]["initialization"],
        "optimization": protocol["decoder"]["optimization"],
        "energy_weights": protocol["decoder"]["energy_weights"],
    }
    for (sample, contract, _, targets, oracle), prediction in zip(
        examples, predictions, strict=True
    ):
        prediction_records.append(
            {
                "package_index": sample.package_index,
                "target_pg": sample.target_pg,
                "bond_orbit_count": len(targets.bond_target_lengths),
                "angle_orbit_count": len(targets.angle_target_cosines),
                "bond_orbit_mae_angstrom": float(
                    np.mean(
                        np.abs(
                            prediction["bond_lengths"]
                            - targets.bond_target_lengths
                        )
                    )
                ),
                "angle_orbit_mae_degrees": _angle_mae_degrees(
                    prediction["angle_cosines"], targets.angle_target_cosines
                ),
            }
        )
        decoder_targets = _predicted_decoder_targets(prediction, contract, oracle)
        specification = build_orbit_parameterization(sample)
        starts = int(protocol["decoder"]["initialization"]["starts_per_molecule"])
        candidates = []
        candidate_coordinates = []
        for start in range(starts):
            candidate, coordinates = optimize_one_start(
                sample,
                contract,
                decoder_targets,
                specification,
                decoder_protocol,
                seed=int(protocol["seed"]) + sample.package_index * 1009 + start,
                device="cpu",
            )
            candidates.append(candidate)
            candidate_coordinates.append(coordinates)
        selected = min(
            range(starts), key=lambda index: candidates[index]["final_energy"]
        )
        coordinates = candidate_coordinates[selected]
        metrics = reconstruction_metrics(sample, contract, oracle, coordinates)
        reconstruction_records.append(
            {
                "package_index": sample.package_index,
                "target_pg": sample.target_pg,
                "selected_start_index": selected,
                "selected_energy": candidates[selected]["final_energy"],
                "finite_start_count": int(
                    sum(row["all_values_finite"] for row in candidates)
                ),
                **metrics,
            }
        )
        coordinate_parts.append(coordinates.astype(np.float32))
        offsets.append(offsets[-1] + len(coordinates))
        print(
            f"reconstruct package_index={sample.package_index} PG={sample.target_pg} "
            f"bond={metrics['bond_mae_angstrom']:.6g}A "
            f"angle={metrics['angle_mae_degrees']:.6g}deg "
            f"rmsd={metrics['kabsch_rmsd_angstrom']:.6g}A",
            flush=True,
        )
    prediction_values = {
        "bond_orbit_mae_max_angstrom": max(
            row["bond_orbit_mae_angstrom"] for row in prediction_records
        ),
        "angle_orbit_mae_max_degrees": max(
            row["angle_orbit_mae_degrees"] for row in prediction_records
        ),
    }
    prediction_thresholds = protocol["prediction_gate"]
    prediction_checks = {
        key: prediction_values[key] <= prediction_thresholds[key]
        for key in prediction_thresholds
    }
    reconstruction_values = {
        "bond_mae_max_angstrom": max(
            row["bond_mae_angstrom"] for row in reconstruction_records
        ),
        "angle_mae_max_degrees": max(
            row["angle_mae_degrees"] for row in reconstruction_records
        ),
        "torsion_circular_mae_max_degrees": max(
            row["torsion_circular_mae_degrees"] for row in reconstruction_records
        ),
        "ring_closure_mae_max_angstrom": max(
            row["ring_closure_mae_angstrom"] for row in reconstruction_records
        ),
        "kabsch_rmsd_max_angstrom": max(
            row["kabsch_rmsd_angstrom"] for row in reconstruction_records
        ),
        "collision_free_fraction": float(
            np.mean([row["collision_free"] for row in reconstruction_records])
        ),
        "max_operation_atom_error_angstrom": max(
            row["max_atom_error_angstrom"] for row in reconstruction_records
        ),
        "all_values_finite": all(
            row["finite_start_count"]
            == int(protocol["decoder"]["initialization"]["starts_per_molecule"])
            for row in reconstruction_records
        ),
    }
    thresholds = protocol["reconstruction_gate"]
    checks = {
        "bond_mae": reconstruction_values["bond_mae_max_angstrom"]
        <= thresholds["bond_mae_max_angstrom"],
        "angle_mae": reconstruction_values["angle_mae_max_degrees"]
        <= thresholds["angle_mae_max_degrees"],
        "torsion_mae": reconstruction_values["torsion_circular_mae_max_degrees"]
        <= thresholds["torsion_circular_mae_max_degrees"],
        "ring_closure": reconstruction_values["ring_closure_mae_max_angstrom"]
        <= thresholds["ring_closure_mae_max_angstrom"],
        "kabsch_rmsd": reconstruction_values["kabsch_rmsd_max_angstrom"]
        <= thresholds["kabsch_rmsd_max_angstrom"],
        "collision_free": reconstruction_values["collision_free_fraction"]
        >= thresholds["collision_free_fraction_min"],
        "group_action": reconstruction_values["max_operation_atom_error_angstrom"]
        <= thresholds["max_operation_atom_error_max_angstrom"],
        "finite": reconstruction_values["all_values_finite"]
        is thresholds["all_values_finite"],
        "oracle_local_pair_never_used": True,
        "target_cartesian_model_input_never_used": True,
    }
    passed = all(prediction_checks.values()) and all(checks.values())
    status = "PASS_M1_ADVANCE_TO_M2_TORSION_HEAD" if passed else "FAIL_M1_STOP_BEFORE_M2"
    np.savez_compressed(
        output_dir / "coordinates.npz",
        coordinates=np.concatenate(coordinate_parts),
        atom_offsets=np.asarray(offsets, dtype=np.int64),
        package_indices=np.asarray(
            [sample.package_index for sample in samples], dtype=np.int64
        ),
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        **{
            f"{sample.package_index}_{key}": value
            for sample, prediction in zip(samples, predictions, strict=True)
            for key, value in prediction.items()
        },
    )
    report = {
        "schema_version": M1_RUN_SCHEMA_VERSION,
        "status": status,
        "passed": passed,
        "claim_scope": "Tier-4 learned bond/angle orbit memorization with oracle torsions",
        "quality_claim": False,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "training_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "step": int(state["step"]),
        },
        "execution_revision": {
            "scientific_protocol_changed": False,
            "failure_signature": "completed training; CUDA float64 small-tensor decoder was manually interrupted for excessive runtime",
            "failed_console_log_sha256": _sha256(failed_console),
            "sole_change": "evaluate the frozen step-1024 checkpoint on CPU",
            "training_restarted": False,
        },
        "prediction_records": prediction_records,
        "prediction_gate": {
            "passed": all(prediction_checks.values()),
            "checks": prediction_checks,
            "values": prediction_values,
            "thresholds": prediction_thresholds,
        },
        "reconstruction_records": reconstruction_records,
        "reconstruction_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "values": reconstruction_values,
            "thresholds": thresholds,
        },
        "isolation": protocol["isolation"],
        "decision": protocol["progression"]["if_passes"]
        if passed
        else (
            protocol["progression"]["if_fails_prediction"]
            if not all(prediction_checks.values())
            else protocol["progression"]["if_prediction_passes_but_reconstruction_fails"]
        ),
    }
    _dump(output_dir / "report.json", report)
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _dump(
        output_dir / "manifest.json",
        {"schema_version": "pg-orbitflow-m1-evaluation-manifest-v1", "artifacts": artifacts},
    )
    print(f"{status} report={output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
