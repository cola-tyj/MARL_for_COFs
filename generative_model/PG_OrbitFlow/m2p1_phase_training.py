"""Train and evaluate phase-aware M2.1 on the frozen Tier-16 panel."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .m0_oracle_reconstruction import (
    build_oracle_targets,
    optimize_one_start,
    reconstruction_metrics,
)
from .m1_bond_angle_training import _angle_mae_degrees
from .m2_torsion_training import (
    _learned_decoder_targets,
    _torsion_mae_degrees,
)
from .orbit_ic_model import QuotientICPredictor, build_orbit_ic_example
from .orbit_kinematics import build_orbit_parameterization
from .overfit import _panel_samples


M2P1_RUN_SCHEMA_VERSION = "pg-orbitflow-m2p1-phase-aware-tier16-training-v1"


def _dump(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != (
        "pg-orbitflow-m2p1-phase-aware-tier16-protocol-v1"
    ):
        raise ValueError("unsupported M2.1 protocol schema")
    if _sha256(Path(protocol["package_dir"]) / "manifest.json") != protocol[
        "canonical_manifest_sha256"
    ]:
        raise ValueError("canonical manifest changed after M2.1 freeze")
    for key in ("base_m2_protocol", "base_m2_summary"):
        if _sha256(Path(protocol[key]["path"])) != protocol[key]["sha256"]:
            raise ValueError(f"{key} changed after M2.1 freeze")
    panel = protocol["panel"]
    if _sha256(Path(panel["path"])) != panel["sha256"]:
        raise ValueError("Tier-16 panel changed after M2.1 freeze")
    if _sha256(Path(panel["audit_path"])) != panel["audit_sha256"]:
        raise ValueError("Tier-16 panel audit changed after M2.1 freeze")
    checkpoint = protocol["initialization"]["m2_checkpoint"]
    if _sha256(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise ValueError("M2 checkpoint changed after M2.1 freeze")
    if not protocol["model"]["geometry_group_phase"]:
        raise ValueError("M2.1 requires geometry group-phase features")
    for name, source in protocol["implementation"].items():
        if _sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"M2.1 implementation changed after freeze: {name}")
    if protocol["isolation"]["oracle_internal_coordinates_used_by_decoder"]:
        raise ValueError("M2.1 decoder must be oracle-free")
    return protocol


def _model_setting(protocol: dict) -> dict:
    return {
        key: protocol["model"][key]
        for key in (
            "node_feature_dim",
            "edge_feature_dim",
            "hidden_dim",
            "layers",
            "predict_torsion",
            "geometry_group_phase",
        )
    }


def _inherit_backbone(model, parent_state: dict, prefixes: tuple[str, ...]) -> dict:
    target = model.state_dict()
    inherited = []
    for key in sorted(target):
        if not key.startswith(prefixes):
            continue
        if key not in parent_state or parent_state[key].shape != target[key].shape:
            raise ValueError(f"missing or incompatible inherited parameter: {key}")
        target[key] = parent_state[key].detach().clone()
        inherited.append(key)
    if not inherited:
        raise ValueError("M2.1 inherited no backbone parameters")
    forbidden = [key for key in inherited if "head" in key]
    if forbidden:
        raise ValueError(f"M2.1 must not inherit resized heads: {forbidden}")
    model.load_state_dict(target, strict=True)
    return {
        "inherited_tensor_count": len(inherited),
        "inherited_parameter_count": int(sum(target[key].numel() for key in inherited)),
        "inherited_keys": inherited,
    }


def _balanced_schedule(examples, *, steps: int, seed: int) -> list[list[int]]:
    groups = {
        name: [index for index, row in enumerate(examples) if row[0].target_pg == name]
        for name in ("C2", "C3")
    }
    if any(len(values) != 8 for values in groups.values()):
        raise ValueError("M2.1 requires exactly eight C2 and eight C3 molecules")
    if steps % 4:
        raise ValueError("M2.1 steps must be divisible by four")
    rng = np.random.default_rng(int(seed))
    schedule = []
    for _ in range(steps // 4):
        shuffled = {name: rng.permutation(values).tolist() for name, values in groups.items()}
        for chunk in range(4):
            batch = (
                shuffled["C2"][2 * chunk : 2 * chunk + 2]
                + shuffled["C3"][2 * chunk : 2 * chunk + 2]
            )
            schedule.append(batch)
    exposures = np.bincount(np.asarray(schedule).reshape(-1), minlength=len(examples))
    if not np.all(exposures == steps // 4):
        raise RuntimeError("M2.1 schedule is not molecule-balanced")
    return schedule


def _nonplanar_mask(target_sincos: np.ndarray) -> np.ndarray:
    values = np.asarray(target_sincos, dtype=np.float64)
    absolute = np.abs(np.rad2deg(np.arctan2(values[:, 0], values[:, 1])))
    return np.minimum(absolute, np.abs(180.0 - absolute)) > 5.0


def _prediction_record(sample, targets, prediction) -> dict:
    nonplanar = _nonplanar_mask(targets.torsion_target_sincos)
    return {
        "package_index": sample.package_index,
        "target_pg": sample.target_pg,
        "bond_orbit_count": len(targets.bond_target_lengths),
        "angle_orbit_count": len(targets.angle_target_cosines),
        "torsion_orbit_count": len(targets.torsion_target_sincos),
        "nonplanar_torsion_orbit_count": int(np.sum(nonplanar)),
        "bond_orbit_mae_angstrom": float(
            np.mean(np.abs(prediction["bond_lengths"] - targets.bond_target_lengths))
        ),
        "angle_orbit_mae_degrees": _angle_mae_degrees(
            prediction["angle_cosines"], targets.angle_target_cosines
        ),
        "torsion_orbit_circular_mae_degrees": _torsion_mae_degrees(
            prediction["torsion_sincos"], targets.torsion_target_sincos
        ),
        "nonplanar_torsion_orbit_circular_mae_degrees": (
            _torsion_mae_degrees(
                prediction["torsion_sincos"][nonplanar],
                targets.torsion_target_sincos[nonplanar],
            )
            if bool(np.any(nonplanar))
            else 0.0
        ),
    }


def _prediction_values(records: list[dict]) -> dict:
    return {
        "bond_orbit_mae_max_angstrom": max(
            row["bond_orbit_mae_angstrom"] for row in records
        ),
        "angle_orbit_mae_max_degrees": max(
            row["angle_orbit_mae_degrees"] for row in records
        ),
        "torsion_orbit_circular_mae_max_degrees": max(
            row["torsion_orbit_circular_mae_degrees"] for row in records
        ),
        "nonplanar_torsion_orbit_circular_mae_max_degrees": max(
            row["nonplanar_torsion_orbit_circular_mae_degrees"] for row in records
        ),
    }


def _prediction_checks(records: list[dict], gate: dict) -> tuple[dict, dict, dict]:
    values = _prediction_values(records)
    numeric = {key: value for key, value in gate.items() if key != "c2_and_c3_each_pass"}
    checks = {key: values[key] <= threshold for key, threshold in numeric.items()}
    strata = {}
    for point_group in ("C2", "C3"):
        subset = [row for row in records if row["target_pg"] == point_group]
        current = _prediction_values(subset)
        strata[point_group] = {
            "values": current,
            "passed": all(current[key] <= threshold for key, threshold in numeric.items()),
        }
    checks["c2_and_c3_each_pass"] = all(row["passed"] for row in strata.values())
    return values, checks, strata


def _reconstruction_values(records: list[dict]) -> dict:
    active = int(sum(row["active_chirality_count"] for row in records))
    chirality_success = sum(
        row["chirality_preserved_fraction"] * row["active_chirality_count"]
        for row in records
    )
    return {
        "bond_mae_max_angstrom": max(row["bond_mae_angstrom"] for row in records),
        "angle_mae_max_degrees": max(row["angle_mae_degrees"] for row in records),
        "torsion_circular_mae_max_degrees": max(
            row["torsion_circular_mae_degrees"] for row in records
        ),
        "nonplanar_torsion_circular_mae_max_degrees": max(
            row["nonplanar_torsion_circular_mae_degrees"] for row in records
        ),
        "ring_closure_mae_max_angstrom": max(
            row["ring_closure_mae_angstrom"] for row in records
        ),
        "kabsch_rmsd_max_angstrom": max(row["kabsch_rmsd_angstrom"] for row in records),
        "collision_free_fraction": float(np.mean([row["collision_free"] for row in records])),
        "max_operation_atom_error_angstrom": max(
            row["max_atom_error_angstrom"] for row in records
        ),
        "active_chirality_count": active,
        "chirality_preserved_fraction": float(chirality_success / active) if active else 1.0,
        "all_values_finite": all(row["all_starts_finite"] for row in records),
    }


def _reconstruction_checks(records: list[dict], gate: dict) -> tuple[dict, dict, dict]:
    values = _reconstruction_values(records)

    def numeric_pass(current, *, include_chirality_count=True):
        checks = {
            "bond_mae": current["bond_mae_max_angstrom"] <= gate["bond_mae_max_angstrom"],
            "angle_mae": current["angle_mae_max_degrees"] <= gate["angle_mae_max_degrees"],
            "torsion_mae": current["torsion_circular_mae_max_degrees"] <= gate["torsion_circular_mae_max_degrees"],
            "nonplanar_torsion_mae": current["nonplanar_torsion_circular_mae_max_degrees"] <= gate["nonplanar_torsion_circular_mae_max_degrees"],
            "ring_closure": current["ring_closure_mae_max_angstrom"] <= gate["ring_closure_mae_max_angstrom"],
            "kabsch_rmsd": current["kabsch_rmsd_max_angstrom"] <= gate["kabsch_rmsd_max_angstrom"],
            "collision_free": current["collision_free_fraction"] >= gate["collision_free_fraction_min"],
            "group_action": current["max_operation_atom_error_angstrom"] <= gate["max_operation_atom_error_max_angstrom"],
            "chirality_preserved": current["chirality_preserved_fraction"] >= gate["chirality_preserved_fraction_min"],
            "finite": current["all_values_finite"] is gate["all_values_finite"],
        }
        if include_chirality_count:
            checks["active_chirality_count"] = current["active_chirality_count"] >= gate["active_chirality_count_min"]
        return checks

    checks = numeric_pass(values)
    strata = {}
    for point_group in ("C2", "C3"):
        subset = [row for row in records if row["target_pg"] == point_group]
        current = _reconstruction_values(subset)
        current_checks = numeric_pass(current, include_chirality_count=False)
        strata[point_group] = {
            "values": current,
            "checks": current_checks,
            "passed": all(current_checks.values()),
        }
    checks["c2_and_c3_each_pass"] = all(row["passed"] for row in strata.values())
    checks["oracle_internal_coordinates_never_used"] = True
    checks["target_cartesian_model_input_never_used"] = True
    return values, checks, strata


def _save_checkpoint(path, *, model, optimizer, step, protocol_sha, inheritance):
    import torch

    torch.save(
        {
            "schema_version": M2P1_RUN_SCHEMA_VERSION,
            "step": int(step),
            "protocol_sha256": protocol_sha,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "inheritance": inheritance,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        help="Stop after the frozen prediction Gate; useful only for diagnosis.",
    )
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

    model = QuotientICPredictor(**_model_setting(protocol))
    parent = torch.load(
        Path(protocol["initialization"]["m2_checkpoint"]["path"]),
        map_location="cpu",
        weights_only=False,
    )
    inheritance = _inherit_backbone(
        model,
        parent["model"],
        tuple(protocol["initialization"]["inherited_parameter_prefixes"]),
    )
    model = model.to(args.training_device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(protocol["model"]["parameter_count_expected"]):
        raise RuntimeError("M2.1 parameter count changed after freeze")

    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps = int(training["optimizer_steps"])
    schedule = _balanced_schedule(examples, steps=steps, seed=seed + 17)
    loss_records = []
    for step, batch in enumerate(schedule, start=1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), dtype=torch.float32, device=args.training_device)
        bond_mae = angle_mae = torsion_mae = 0.0
        for index in batch:
            _, _, graph, targets = examples[index]
            prediction = model(graph, device=args.training_device)
            target_bond = torch.as_tensor(
                targets.bond_target_lengths,
                dtype=torch.float32,
                device=args.training_device,
            )
            target_angle = torch.as_tensor(
                targets.angle_target_cosines,
                dtype=torch.float32,
                device=args.training_device,
            )
            target_torsion = torch.as_tensor(
                targets.torsion_target_sincos,
                dtype=torch.float32,
                device=args.training_device,
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
            ) / len(batch)
            bond_mae += float(
                torch.mean(torch.abs(prediction["bond_lengths"] - target_bond)).detach()
            ) / len(batch)
            angle_mae += _angle_mae_degrees(
                prediction["angle_cosines"].detach().cpu().numpy(),
                targets.angle_target_cosines,
            ) / len(batch)
            torsion_mae += _torsion_mae_degrees(
                prediction["torsion_sincos"].detach().cpu().numpy(),
                targets.torsion_target_sincos,
            ) / len(batch)
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(total)) or not bool(
            torch.isfinite(torch.as_tensor(gradient))
        ):
            raise RuntimeError(f"non-finite M2.1 optimization at step {step}")
        optimizer.step()
        loss_records.append(
            {
                "step": step,
                "loss": float(total.detach()),
                "batch_bond_orbit_mae_angstrom": bond_mae,
                "batch_angle_orbit_mae_degrees": angle_mae,
                "batch_torsion_orbit_circular_mae_degrees": torsion_mae,
                "gradient_norm": float(gradient),
            }
        )
        if step == 1 or step % int(training["checkpoint_every"]) == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                milestone_records = []
                for sample, _, graph, targets in examples:
                    prediction = {
                        key: value.detach().cpu().numpy()
                        for key, value in model(graph, device=args.training_device).items()
                    }
                    milestone_records.append(
                        _prediction_record(sample, targets, prediction)
                    )
            milestone = _prediction_values(milestone_records)
            print(
                f"step={step}/{steps} loss={loss_records[-1]['loss']:.6g} "
                f"bond_max={milestone['bond_orbit_mae_max_angstrom']:.6g}A "
                f"angle_max={milestone['angle_orbit_mae_max_degrees']:.6g}deg "
                f"torsion_max={milestone['torsion_orbit_circular_mae_max_degrees']:.6g}deg "
                f"nonplanar_max={milestone['nonplanar_torsion_orbit_circular_mae_max_degrees']:.6g}deg",
                flush=True,
            )
            _save_checkpoint(
                output_dir / f"step-{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                protocol_sha=_sha256(protocol_path),
                inheritance=inheritance,
            )
    _save_checkpoint(
        output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        step=steps,
        protocol_sha=_sha256(protocol_path),
        inheritance=inheritance,
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
    prediction_records = [
        _prediction_record(sample, targets, prediction)
        for (sample, _, _, targets), prediction in zip(examples, predictions, strict=True)
    ]
    prediction_values, prediction_checks, prediction_strata = _prediction_checks(
        prediction_records, protocol["prediction_gate"]
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        **{
            f"{sample.package_index}_{key}": value
            for (sample, _, _, _), prediction in zip(examples, predictions, strict=True)
            for key, value in prediction.items()
        },
    )
    if args.prediction_only or not all(prediction_checks.values()):
        status = (
            "PASS_M2P1_PREDICTION_ONLY"
            if all(prediction_checks.values())
            else "FAIL_M2P1_PREDICTION_GATE_STOP_BEFORE_DECODER"
        )
        report = {
            "schema_version": M2P1_RUN_SCHEMA_VERSION,
            "status": status,
            "passed": False,
            "prediction_gate_passed": all(prediction_checks.values()),
            "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
            "training": {
                "steps": steps,
                "batch_size_molecules": 4,
                "parameter_count": parameter_count,
                "inheritance": inheritance,
            },
            "prediction_records": prediction_records,
            "prediction_gate": {
                "checks": prediction_checks,
                "values": prediction_values,
                "strata": prediction_strata,
                "thresholds": protocol["prediction_gate"],
            },
            "decision": protocol["progression"]["if_fails_prediction"],
        }
        _dump(output_dir / "report.json", report)
        print(f"{status} report={output_dir / 'report.json'}", flush=True)
        return

    model.to("cpu")
    if args.training_device == "cuda":
        torch.cuda.empty_cache()
    decoder_protocol = {
        "initialization": protocol["decoder"]["initialization"],
        "optimization": protocol["decoder"]["optimization"],
        "energy_weights": protocol["decoder"]["energy_weights"],
    }
    reconstruction_records = []
    coordinate_parts = []
    offsets = [0]
    starts = int(protocol["decoder"]["initialization"]["starts_per_molecule"])
    stride = int(protocol["decoder"]["initialization"]["molecule_specific_seed_stride"])
    for (sample, contract, _, _), prediction in zip(examples, predictions, strict=True):
        decoder_targets = _learned_decoder_targets(prediction, contract)
        specification = build_orbit_parameterization(sample)
        candidates = []
        candidate_coordinates = []
        for start in range(starts):
            candidate, coordinates = optimize_one_start(
                sample,
                contract,
                decoder_targets,
                specification,
                decoder_protocol,
                seed=seed + sample.package_index * stride + start,
                device=protocol["decoder"]["evaluation_device"],
            )
            candidates.append(candidate)
            candidate_coordinates.append(coordinates)
        selected = min(range(starts), key=lambda index: candidates[index]["final_energy"])
        coordinates = candidate_coordinates[selected]
        metrics = reconstruction_metrics(
            sample, contract, build_oracle_targets(sample, contract), coordinates
        )
        record = {
            "package_index": sample.package_index,
            "target_pg": sample.target_pg,
            "selected_start_index": selected,
            "selected_energy": candidates[selected]["final_energy"],
            "finite_start_count": int(sum(row["all_values_finite"] for row in candidates)),
            "all_starts_finite": all(row["all_values_finite"] for row in candidates),
            **metrics,
        }
        reconstruction_records.append(record)
        coordinate_parts.append(coordinates.astype(np.float32))
        offsets.append(offsets[-1] + len(coordinates))
        print(
            f"reconstruct package_index={sample.package_index} PG={sample.target_pg} "
            f"bond={metrics['bond_mae_angstrom']:.6g}A "
            f"angle={metrics['angle_mae_degrees']:.6g}deg "
            f"torsion={metrics['torsion_circular_mae_degrees']:.6g}deg "
            f"nonplanar={metrics['nonplanar_torsion_circular_mae_degrees']:.6g}deg "
            f"chiral={metrics['chirality_preserved_fraction']:.6g} "
            f"rmsd={metrics['kabsch_rmsd_angstrom']:.6g}A",
            flush=True,
        )
    reconstruction_values, reconstruction_checks, reconstruction_strata = (
        _reconstruction_checks(
            reconstruction_records, protocol["reconstruction_gate"]
        )
    )
    passed = all(prediction_checks.values()) and all(reconstruction_checks.values())
    status = (
        "PASS_M2P1_PHASE_AWARE_TIER16_ADVANCE_TO_TIER32_PANEL_AUDIT"
        if passed
        else "FAIL_M2P1_RECONSTRUCTION_GATE_STOP_BEFORE_TIER32"
    )
    np.savez_compressed(
        output_dir / "coordinates.npz",
        coordinates=np.concatenate(coordinate_parts),
        atom_offsets=np.asarray(offsets, dtype=np.int64),
        package_indices=np.asarray([sample.package_index for sample in samples], dtype=np.int64),
    )
    report = {
        "schema_version": M2P1_RUN_SCHEMA_VERSION,
        "status": status,
        "passed": passed,
        "claim_scope": "Tier-16 phase-aware non-planar/chiral memorization Gate",
        "claim_limit": protocol["claim_limit"],
        "quality_claim": False,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "training": {
            "steps": steps,
            "batch_size_molecules": 4,
            "molecule_exposures_total": steps * 4,
            "molecule_exposures_each": steps // 4,
            "parameter_count": parameter_count,
            "training_device": args.training_device,
            "decoder_device": protocol["decoder"]["evaluation_device"],
            "inheritance": inheritance,
            "all_finite": True,
        },
        "prediction_records": prediction_records,
        "prediction_gate": {
            "passed": all(prediction_checks.values()),
            "checks": prediction_checks,
            "values": prediction_values,
            "strata": prediction_strata,
            "thresholds": protocol["prediction_gate"],
        },
        "reconstruction_records": reconstruction_records,
        "reconstruction_gate": {
            "passed": all(reconstruction_checks.values()),
            "checks": reconstruction_checks,
            "values": reconstruction_values,
            "strata": reconstruction_strata,
            "thresholds": protocol["reconstruction_gate"],
        },
        "isolation": protocol["isolation"],
        "decision": protocol["progression"]["if_passes"] if passed else protocol[
            "progression"
        ]["if_prediction_passes_but_reconstruction_fails"],
    }
    _dump(output_dir / "report.json", report)
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _dump(
        output_dir / "manifest.json",
        {"schema_version": "pg-orbitflow-m2p1-manifest-v1", "artifacts": artifacts},
    )
    print(f"{status} report={output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
