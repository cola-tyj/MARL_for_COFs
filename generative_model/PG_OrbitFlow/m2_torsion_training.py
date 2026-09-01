"""Train M2 bond/angle/torsion orbit heads and run the all-learned IC decoder."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .m0_oracle_reconstruction import (
    OracleTargets,
    build_oracle_targets,
    optimize_one_start,
    reconstruction_metrics,
)
from .m1_bond_angle_training import _angle_mae_degrees
from .orbit_ic_model import QuotientICPredictor, build_orbit_ic_example
from .orbit_kinematics import build_orbit_parameterization
from .overfit import _panel_samples


M2_RUN_SCHEMA_VERSION = "pg-orbitflow-m2-torsion-training-v1"


def _dump(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-m2-torsion-protocol-v1":
        raise ValueError("unsupported M2 protocol schema")
    if _sha256(Path(protocol["package_dir"]) / "manifest.json") != protocol[
        "canonical_manifest_sha256"
    ]:
        raise ValueError("canonical manifest changed after M2 freeze")
    for key in ("base_m1_protocol", "base_m1_summary"):
        if _sha256(Path(protocol[key]["path"])) != protocol[key]["sha256"]:
            raise ValueError(f"{key} changed after M2 freeze")
    checkpoint = protocol["initialization"]["m1_checkpoint"]
    if _sha256(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise ValueError("M1 checkpoint changed after M2 freeze")
    if any(
        protocol["decoder"][key]
        for key in (
            "oracle_bond_used",
            "oracle_angle_used",
            "oracle_torsion_used",
            "oracle_chirality_used",
            "oracle_local_pair_used",
        )
    ):
        raise ValueError("M2 decoder must not use oracle internal coordinates")
    return protocol


def _torsion_mae_degrees(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    cosine = np.sum(predicted * target, axis=-1).clip(-1.0, 1.0)
    sine = predicted[:, 0] * target[:, 1] - predicted[:, 1] * target[:, 0]
    return float(np.mean(np.abs(np.rad2deg(np.arctan2(sine, cosine)))))


def _learned_decoder_targets(prediction, contract) -> OracleTargets:
    """Expand learned orbit outputs; no target-derived geometry enters the decoder."""

    bond = np.asarray(prediction["bond_lengths"], dtype=np.float64)[
        np.asarray(contract.bond_orbit_id, dtype=np.int64)
    ]
    angle = np.asarray(prediction["angle_cosines"], dtype=np.float64)[
        np.asarray(contract.angle_orbit_id, dtype=np.int64)
    ]
    torsion = np.asarray(prediction["torsion_sincos"], dtype=np.float64)[
        np.asarray(contract.torsion_orbit_id, dtype=np.int64)
    ]
    return OracleTargets(
        bond_lengths=bond,
        angle_cosines=angle,
        torsion_sincos=torsion,
        local_pair_lengths=np.zeros(contract.local_pair_index.shape[1], dtype=np.float64),
        ring_lengths=bond[np.asarray(contract.ring_bond_indices, dtype=np.int64)],
        chirality=np.zeros(contract.chirality_index.shape[1], dtype=np.float64),
    )


def _save_checkpoint(path, *, model, optimizer, step, protocol_sha):
    import torch

    torch.save(
        {
            "schema_version": M2_RUN_SCHEMA_VERSION,
            "step": int(step),
            "protocol_sha256": protocol_sha,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def _gate_reconstruction(records: list[dict], thresholds: dict) -> tuple[dict, dict]:
    values = {
        "bond_mae_max_angstrom": max(row["bond_mae_angstrom"] for row in records),
        "angle_mae_max_degrees": max(row["angle_mae_degrees"] for row in records),
        "torsion_circular_mae_max_degrees": max(
            row["torsion_circular_mae_degrees"] for row in records
        ),
        "ring_closure_mae_max_angstrom": max(
            row["ring_closure_mae_angstrom"] for row in records
        ),
        "kabsch_rmsd_max_angstrom": max(row["kabsch_rmsd_angstrom"] for row in records),
        "collision_free_fraction": float(np.mean([row["collision_free"] for row in records])),
        "max_operation_atom_error_angstrom": max(
            row["max_atom_error_angstrom"] for row in records
        ),
        "all_values_finite": all(row["all_starts_finite"] for row in records),
    }
    checks = {
        "bond_mae": values["bond_mae_max_angstrom"] <= thresholds["bond_mae_max_angstrom"],
        "angle_mae": values["angle_mae_max_degrees"] <= thresholds["angle_mae_max_degrees"],
        "torsion_mae": values["torsion_circular_mae_max_degrees"] <= thresholds["torsion_circular_mae_max_degrees"],
        "ring_closure": values["ring_closure_mae_max_angstrom"] <= thresholds["ring_closure_mae_max_angstrom"],
        "kabsch_rmsd": values["kabsch_rmsd_max_angstrom"] <= thresholds["kabsch_rmsd_max_angstrom"],
        "collision_free": values["collision_free_fraction"] >= thresholds["collision_free_fraction_min"],
        "group_action": values["max_operation_atom_error_angstrom"] <= thresholds["max_operation_atom_error_max_angstrom"],
        "finite": values["all_values_finite"] is thresholds["all_values_finite"],
        "oracle_internal_coordinates_never_used": True,
        "target_cartesian_model_input_never_used": True,
    }
    return values, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    import torch

    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    if args.training_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    seed = int(protocol["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)

    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    samples = _panel_samples(protocol, tier)
    examples = []
    for sample in samples:
        contract = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, contract)
        examples.append((sample, contract, graph, targets))

    model_setting = {
        key: protocol["model"][key]
        for key in (
            "node_feature_dim",
            "edge_feature_dim",
            "hidden_dim",
            "layers",
            "predict_torsion",
        )
    }
    model = QuotientICPredictor(**model_setting)
    checkpoint_path = Path(protocol["initialization"]["m1_checkpoint"]["path"])
    parent = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(parent["model"], strict=False)
    expected_missing = {
        "torsion_head.0.weight",
        "torsion_head.0.bias",
        "torsion_head.2.weight",
        "torsion_head.2.bias",
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected M1->M2 state transition: {incompatible}")
    model = model.to(args.training_device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(protocol["model"]["parameter_count_expected"]):
        raise RuntimeError("M2 parameter count changed after protocol freeze")

    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loss_records = []
    steps = int(training["optimizer_steps"])
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), dtype=torch.float32, device=args.training_device)
        bond_mae = angle_mae = torsion_mae = 0.0
        for _, _, graph, targets in examples:
            prediction = model(graph, device=args.training_device)
            target_bond = torch.as_tensor(
                targets.bond_target_lengths, dtype=torch.float32, device=args.training_device
            )
            target_angle = torch.as_tensor(
                targets.angle_target_cosines, dtype=torch.float32, device=args.training_device
            )
            target_torsion = torch.as_tensor(
                targets.torsion_target_sincos, dtype=torch.float32, device=args.training_device
            )
            bond_loss = torch.mean(torch.square(prediction["bond_lengths"] - target_bond))
            angle_loss = torch.mean(torch.square(prediction["angle_cosines"] - target_angle))
            torsion_loss = torch.mean(
                1.0
                - torch.sum(prediction["torsion_sincos"] * target_torsion, dim=-1).clamp(
                    -1.0, 1.0
                )
            )
            total = total + (
                float(training["bond_loss_weight"]) * bond_loss
                + float(training["angle_loss_weight"]) * angle_loss
                + float(training["torsion_circular_loss_weight"]) * torsion_loss
            ) / len(examples)
            bond_mae += float(
                torch.mean(torch.abs(prediction["bond_lengths"] - target_bond)).detach()
            ) / len(examples)
            angle_mae += _angle_mae_degrees(
                prediction["angle_cosines"].detach().cpu().numpy(),
                targets.angle_target_cosines,
            ) / len(examples)
            torsion_mae += _torsion_mae_degrees(
                prediction["torsion_sincos"].detach().cpu().numpy(),
                targets.torsion_target_sincos,
            ) / len(examples)
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(total)) or not bool(
            torch.isfinite(torch.as_tensor(gradient))
        ):
            raise RuntimeError(f"non-finite M2 optimization at step {step}")
        optimizer.step()
        record = {
            "step": step,
            "loss": float(total.detach()),
            "bond_orbit_mae_angstrom": bond_mae,
            "angle_orbit_mae_degrees": angle_mae,
            "torsion_orbit_circular_mae_degrees": torsion_mae,
            "gradient_norm": float(gradient),
        }
        loss_records.append(record)
        if step == 1 or step % int(training["checkpoint_every"]) == 0 or step == steps:
            print(
                f"step={step}/{steps} loss={record['loss']:.6g} "
                f"bond={bond_mae:.6g}A angle={angle_mae:.6g}deg "
                f"torsion={torsion_mae:.6g}deg",
                flush=True,
            )
            _save_checkpoint(
                output_dir / f"step-{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                protocol_sha=_sha256(protocol_path),
            )
    _save_checkpoint(
        output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        step=steps,
        protocol_sha=_sha256(protocol_path),
    )
    np.savez_compressed(
        output_dir / "losses.npz",
        **{
            key: np.asarray([row[key] for row in loss_records])
            for key in loss_records[0]
        },
    )

    model.eval()
    with torch.no_grad():
        predictions = [
            {
                key: value.detach().cpu().numpy()
                for key, value in model(graph, device=args.training_device).items()
            }
            for _, _, graph, _ in examples
        ]
    model.to("cpu")
    if args.training_device == "cuda":
        torch.cuda.empty_cache()

    prediction_records = []
    reconstruction_records = []
    coordinate_parts = []
    offsets = [0]
    decoder_protocol = {
        "initialization": protocol["decoder"]["initialization"],
        "optimization": protocol["decoder"]["optimization"],
        "energy_weights": protocol["decoder"]["energy_weights"],
    }
    for (sample, contract, _, targets), prediction in zip(
        examples, predictions, strict=True
    ):
        prediction_record = {
            "package_index": sample.package_index,
            "target_pg": sample.target_pg,
            "bond_orbit_count": len(targets.bond_target_lengths),
            "angle_orbit_count": len(targets.angle_target_cosines),
            "torsion_orbit_count": len(targets.torsion_target_sincos),
            "bond_orbit_mae_angstrom": float(
                np.mean(np.abs(prediction["bond_lengths"] - targets.bond_target_lengths))
            ),
            "angle_orbit_mae_degrees": _angle_mae_degrees(
                prediction["angle_cosines"], targets.angle_target_cosines
            ),
            "torsion_orbit_circular_mae_degrees": _torsion_mae_degrees(
                prediction["torsion_sincos"], targets.torsion_target_sincos
            ),
        }
        prediction_records.append(prediction_record)
        decoder_targets = _learned_decoder_targets(prediction, contract)
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
                seed=seed + sample.package_index * 1009 + start,
                device=protocol["decoder"]["evaluation_device"],
            )
            candidates.append(candidate)
            candidate_coordinates.append(coordinates)
        selected = min(range(starts), key=lambda index: candidates[index]["final_energy"])
        coordinates = candidate_coordinates[selected]
        evaluation_targets = build_oracle_targets(sample, contract)
        metrics = reconstruction_metrics(sample, contract, evaluation_targets, coordinates)
        reconstruction_records.append(
            {
                "package_index": sample.package_index,
                "target_pg": sample.target_pg,
                "selected_start_index": selected,
                "selected_energy": candidates[selected]["final_energy"],
                "finite_start_count": int(sum(row["all_values_finite"] for row in candidates)),
                "all_starts_finite": all(row["all_values_finite"] for row in candidates),
                **metrics,
            }
        )
        coordinate_parts.append(coordinates.astype(np.float32))
        offsets.append(offsets[-1] + len(coordinates))
        print(
            f"reconstruct package_index={sample.package_index} PG={sample.target_pg} "
            f"bond={metrics['bond_mae_angstrom']:.6g}A "
            f"angle={metrics['angle_mae_degrees']:.6g}deg "
            f"torsion={metrics['torsion_circular_mae_degrees']:.6g}deg "
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
        "torsion_orbit_circular_mae_max_degrees": max(
            row["torsion_orbit_circular_mae_degrees"] for row in prediction_records
        ),
    }
    prediction_gate = protocol["prediction_gate"]
    prediction_checks = {
        key: prediction_values[key] <= prediction_gate[key] for key in prediction_gate
    }
    reconstruction_values, reconstruction_checks = _gate_reconstruction(
        reconstruction_records, protocol["reconstruction_gate"]
    )
    passed = all(prediction_checks.values()) and all(reconstruction_checks.values())
    status = (
        "PASS_M2_ALL_LEARNED_IC_ADVANCE_TO_TIER16_PANEL_AUDIT"
        if passed
        else "FAIL_M2_STOP_BEFORE_TIER16"
    )
    np.savez_compressed(
        output_dir / "coordinates.npz",
        coordinates=np.concatenate(coordinate_parts),
        atom_offsets=np.asarray(offsets, dtype=np.int64),
        package_indices=np.asarray([sample.package_index for sample in samples], dtype=np.int64),
    )
    report = {
        "schema_version": M2_RUN_SCHEMA_VERSION,
        "status": status,
        "passed": passed,
        "claim_scope": "Tier-4 all-learned bond/angle/planar-torsion orbit memorization",
        "claim_limit": protocol["claim_limit"],
        "quality_claim": False,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "initialization": protocol["initialization"],
        "training": {
            "steps": steps,
            "batch_size_molecules": len(examples),
            "parameter_count": parameter_count,
            "training_device": args.training_device,
            "decoder_device": protocol["decoder"]["evaluation_device"],
            "all_finite": True,
        },
        "prediction_records": prediction_records,
        "prediction_gate": {
            "passed": all(prediction_checks.values()),
            "checks": prediction_checks,
            "values": prediction_values,
            "thresholds": prediction_gate,
        },
        "reconstruction_records": reconstruction_records,
        "reconstruction_gate": {
            "passed": all(reconstruction_checks.values()),
            "checks": reconstruction_checks,
            "values": reconstruction_values,
            "thresholds": protocol["reconstruction_gate"],
        },
        "isolation": protocol["isolation"],
        "decision": protocol["progression"]["if_passes"] if passed else (
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
        {"schema_version": "pg-orbitflow-m2-manifest-v1", "artifacts": artifacts},
    )
    print(f"{status} report={output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
