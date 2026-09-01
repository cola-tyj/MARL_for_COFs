"""Train and evaluate the frozen nested M3 Tier-32 IC model."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .local_atom_matching import best_terminal_sibling_relabeling
from .local_rotor import (
    build_local_rotor_set_contract,
    optimal_coupled_circular_assignment,
)
from .m0_oracle_reconstruction import build_oracle_targets, optimize_one_start, reconstruction_metrics
from .m1_bond_angle_training import _angle_mae_degrees
from .m2_torsion_training import _learned_decoder_targets
from .m2p1_phase_training import _prediction_checks, _reconstruction_checks
from .m2p2_model import SetValuedQuotientICPredictor
from .m2p2_set_training import _prediction_record
from .orbit_ic_model import build_orbit_ic_example
from .orbit_kinematics import build_orbit_parameterization
from .overfit import _panel_samples


SCHEMA_VERSION = "pg-orbitflow-m3-tier32-training-v1"


def _dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-m3-tier32-protocol-v1":
        raise ValueError("unsupported M3 protocol")
    if _sha256(Path(protocol["package_dir"]) / "manifest.json") != protocol["canonical_manifest_sha256"]:
        raise RuntimeError("canonical manifest changed")
    for item in protocol["parent"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3 parent identity changed")
    if _sha256(Path(protocol["panel_audit"]["path"])) != protocol["panel_audit"]["sha256"]:
        raise RuntimeError("M3 panel audit identity changed")
    if _sha256(Path(protocol["decoder_parallelism_audit"]["path"])) != protocol["decoder_parallelism_audit"]["sha256"]:
        raise RuntimeError("M3 decoder parallelism audit identity changed")
    for item in protocol["implementation"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3 implementation identity changed")
    if len(protocol["panel"]["records"]) != 32:
        raise RuntimeError("M3 requires exactly 32 molecules")
    return protocol


def _model_setting(protocol: dict) -> dict:
    return {
        key: protocol["model"][key]
        for key in ("node_feature_dim", "edge_feature_dim", "hidden_dim", "layers", "predict_torsion", "geometry_group_phase")
    }


def _balanced_schedule(examples, *, steps: int, seed: int):
    groups = {
        name: [index for index, row in enumerate(examples) if row[0].target_pg == name]
        for name in ("C2", "C3")
    }
    if any(len(values) != 16 for values in groups.values()):
        raise ValueError("M3 requires exactly sixteen C2 and sixteen C3 molecules")
    batches_per_cycle = 8
    if steps % batches_per_cycle:
        raise ValueError("M3 steps must be divisible by eight")
    rng = np.random.default_rng(int(seed))
    schedule = []
    for _ in range(steps // batches_per_cycle):
        shuffled = {name: rng.permutation(values).tolist() for name, values in groups.items()}
        for chunk in range(batches_per_cycle):
            schedule.append(
                shuffled["C2"][2 * chunk : 2 * chunk + 2]
                + shuffled["C3"][2 * chunk : 2 * chunk + 2]
            )
    exposures = np.bincount(np.asarray(schedule).reshape(-1), minlength=len(examples))
    if not np.all(exposures == steps // batches_per_cycle):
        raise RuntimeError("M3 schedule is not molecule-balanced")
    return schedule


def _save(path, model, optimizer, step, protocol_sha):
    import torch
    torch.save(
        {"schema_version": SCHEMA_VERSION, "step": int(step), "protocol_sha256": protocol_sha,
         "model": model.state_dict(), "optimizer": optimizer.state_dict()},
        path,
    )


def _decoder_task(task):
    import torch
    torch.set_num_threads(1)
    sample, contract, decoder_targets, specification, decoder_protocol, seed, start = task
    candidate, coordinates = optimize_one_start(
        sample, contract, decoder_targets, specification, decoder_protocol,
        seed=seed, device="cpu",
    )
    return int(sample.package_index), int(start), candidate, coordinates


def _matched_metrics(sample, contract, coordinates, raw_metrics):
    match = best_terminal_sibling_relabeling(sample, coordinates)
    metrics = reconstruction_metrics(
        sample, contract, build_oracle_targets(sample, contract), match["coordinates"]
    )
    for key in (
        "collision_free", "collision_threshold_ratio", "minimum_covalent_radius_ratio",
        "minimum_nonbonded_distance_angstrom", "max_atom_error_angstrom",
        "max_operation_rms_angstrom", "mean_operation_rms_angstrom",
    ):
        metrics[key] = raw_metrics[key]
    assignment = {
        "terminal_sibling_groups": match["terminal_sibling_groups"],
        "assignment_count": int(match["assignment_count"]),
        "nonidentity_assignment": bool(match["nonidentity_assignment"]),
        "permutation": match["permutation"].tolist(),
        "raw_kabsch_rmsd_angstrom": float(raw_metrics["kabsch_rmsd_angstrom"]),
        "matched_kabsch_rmsd_angstrom": float(match["kabsch_rmsd_angstrom"]),
    }
    return metrics, assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prediction-only", action="store_true")
    args = parser.parse_args()
    import torch

    if args.training_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    seed = int(protocol["seed"])
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)

    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    samples = _panel_samples(protocol, tier)
    examples = []
    for sample in samples:
        contract = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, contract)
        local = build_local_rotor_set_contract(sample, contract)
        examples.append((sample, contract, graph, targets, local))

    model = SetValuedQuotientICPredictor(**_model_setting(protocol))
    parent = torch.load(protocol["parent"]["checkpoint"]["path"], map_location="cpu", weights_only=False)
    model.load_state_dict(parent["model"], strict=True)
    model = model.to(args.training_device)
    for parameter in model.parameters(): parameter.requires_grad_(True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != int(protocol["model"]["parameter_count_expected"]) or trainable_count != int(protocol["model"]["trainable_parameter_count_expected"]):
        raise RuntimeError("M3 parameter count changed")
    training = protocol["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    steps = int(training["optimizer_steps"])
    schedule = _balanced_schedule(examples, steps=steps, seed=seed + 17)
    losses = []
    for step, batch in enumerate(schedule, start=1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=args.training_device)
        for index in batch:
            _, _, graph, targets, local = examples[index]
            prediction = model(graph, local, device=args.training_device)
            target_bond = torch.as_tensor(targets.bond_target_lengths, dtype=torch.float32, device=args.training_device)
            target_angle = torch.as_tensor(targets.angle_target_cosines, dtype=torch.float32, device=args.training_device)
            target_torsion = torch.as_tensor(targets.torsion_target_sincos, dtype=torch.float32, device=args.training_device)
            aligned_torsion = optimal_coupled_circular_assignment(prediction["torsion_sincos"], target_torsion, local)
            bond_loss = torch.mean(torch.square(prediction["bond_lengths"] - target_bond))
            angle_loss = torch.mean(torch.square(prediction["angle_cosines"] - target_angle))
            torsion_loss = torch.mean(1.0 - torch.sum(prediction["torsion_sincos"] * aligned_torsion, dim=-1).clamp(-1.0, 1.0))
            total = total + (
                float(training["bond_loss_weight"]) * bond_loss
                + float(training["angle_loss_weight"]) * angle_loss
                + float(training["torsion_circular_loss_weight"]) * torsion_loss
            ) / len(batch)
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
        if not torch.isfinite(total) or not torch.isfinite(torch.as_tensor(gradient)):
            raise RuntimeError(f"non-finite M3 training at step {step}")
        optimizer.step()
        losses.append((step, float(total.detach()), float(gradient)))
        if step == 1 or step % int(training["checkpoint_every"]) == 0 or step == steps:
            model.eval(); milestone = []
            with torch.no_grad():
                for sample, _, graph, targets, local in examples:
                    prediction = {key: value.cpu().numpy() for key, value in model(graph, local, device=args.training_device).items()}
                    milestone.append(_prediction_record(sample, targets, prediction, local))
            values, checks, _ = _prediction_checks(milestone, protocol["prediction_gate"])
            print(
                f"step={step}/{steps} loss={float(total):.6g} "
                f"bond={values['bond_orbit_mae_max_angstrom']:.6g}A "
                f"angle={values['angle_orbit_mae_max_degrees']:.6g}deg "
                f"torsion={values['torsion_orbit_circular_mae_max_degrees']:.6g}deg "
                f"nonplanar={values['nonplanar_torsion_orbit_circular_mae_max_degrees']:.6g}deg "
                f"gate={'PASS' if all(checks.values()) else 'FAIL'}", flush=True,
            )
            _save(output_dir / f"step-{step:06d}.pt", model, optimizer, step, _sha256(protocol_path))
    _save(output_dir / "last.pt", model, optimizer, steps, _sha256(protocol_path))
    np.savez_compressed(output_dir / "losses.npz", step=np.asarray([row[0] for row in losses]), loss=np.asarray([row[1] for row in losses]), gradient_norm=np.asarray([row[2] for row in losses]))

    model.eval(); predictions = []
    with torch.no_grad():
        for _, _, graph, _, local in examples:
            predictions.append({key: value.cpu().numpy() for key, value in model(graph, local, device=args.training_device).items()})
    prediction_records = [_prediction_record(sample, targets, prediction, local) for (sample, _, _, targets, local), prediction in zip(examples, predictions, strict=True)]
    pvalues, pchecks, pstrata = _prediction_checks(prediction_records, protocol["prediction_gate"])
    np.savez_compressed(output_dir / "predictions.npz", **{f"{sample.package_index}_{key}": value for (sample, _, _, _, _), prediction in zip(examples, predictions, strict=True) for key, value in prediction.items()})
    prediction_passed = all(pchecks.values())
    if args.prediction_only or not prediction_passed:
        status = "PASS_M3_PREDICTION_ONLY" if prediction_passed else "FAIL_M3_PREDICTION_GATE_STOP_BEFORE_DECODER"
        report = {"schema_version": SCHEMA_VERSION, "status": status, "passed": False, "prediction_gate_passed": prediction_passed,
                  "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
                  "training": {"steps": steps, "parameter_count": parameter_count, "trainable_parameter_count": trainable_count},
                  "prediction_records": prediction_records,
                  "prediction_gate": {"passed": prediction_passed, "checks": pchecks, "values": pvalues, "strata": pstrata, "thresholds": protocol["prediction_gate"]}}
        _dump(output_dir / "report.json", report); print(f"{status} report={output_dir/'report.json'}", flush=True); return

    model.to("cpu"); torch.cuda.empty_cache()
    decoder_protocol = {key: protocol["decoder"][key] for key in ("initialization", "optimization", "energy_weights")}
    starts = int(protocol["decoder"]["initialization"]["starts_per_molecule"])
    stride = int(protocol["decoder"]["initialization"]["molecule_specific_seed_stride"])
    tasks = []
    for (sample, contract, _, _, _), prediction in zip(examples, predictions, strict=True):
        decoder_targets = _learned_decoder_targets(prediction, contract)
        specification = build_orbit_parameterization(sample)
        for start in range(starts):
            tasks.append((sample, contract, decoder_targets, specification, decoder_protocol, seed + sample.package_index * stride + start, start))
    workers = int(protocol["decoder"]["parallel_workers"])
    context = multiprocessing.get_context("spawn")
    collected = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        for package_index, start, candidate, coordinates in executor.map(_decoder_task, tasks, chunksize=1):
            collected[(package_index, start)] = (candidate, coordinates)

    raw_records = []; matched_records = []; assignments = []; coordinate_parts = []; offsets = [0]
    for sample, contract, _, _, _ in examples:
        candidates = [collected[(sample.package_index, start)][0] for start in range(starts)]
        coordinate_candidates = [collected[(sample.package_index, start)][1] for start in range(starts)]
        selected = min(range(starts), key=lambda index: candidates[index]["final_energy"])
        coordinates = coordinate_candidates[selected]
        raw_metrics = reconstruction_metrics(sample, contract, build_oracle_targets(sample, contract), coordinates)
        common = {"package_index": int(sample.package_index), "target_pg": sample.target_pg,
                  "selected_start_index": selected, "selected_energy": float(candidates[selected]["final_energy"]),
                  "finite_start_count": sum(candidate["all_values_finite"] for candidate in candidates),
                  "all_starts_finite": all(candidate["all_values_finite"] for candidate in candidates)}
        raw_records.append({**common, **raw_metrics})
        matched_metrics, assignment = _matched_metrics(sample, contract, coordinates, raw_metrics)
        matched_records.append({**common, **matched_metrics}); assignments.append({"package_index": int(sample.package_index), **assignment})
        coordinate_parts.append(coordinates.astype(np.float32)); offsets.append(offsets[-1] + len(coordinates))
        print(f"reconstruct package_index={sample.package_index} PG={sample.target_pg} torsion={matched_metrics['torsion_circular_mae_degrees']:.6g} nonplanar={matched_metrics['nonplanar_torsion_circular_mae_degrees']:.6g} rmsd={matched_metrics['kabsch_rmsd_angstrom']:.6g}A", flush=True)
    raw_values, raw_checks, raw_strata = _reconstruction_checks(raw_records, protocol["reconstruction_gate"])
    values, checks, strata = _reconstruction_checks(matched_records, protocol["reconstruction_gate"])
    passed = all(checks.values())
    status = "PASS_M3_TIER32" if passed else "FAIL_M3_RECONSTRUCTION_GATE"
    np.savez_compressed(output_dir / "coordinates.npz", coordinates=np.concatenate(coordinate_parts), atom_offsets=np.asarray(offsets), package_indices=np.asarray([sample.package_index for sample in samples]))
    report = {
        "schema_version": SCHEMA_VERSION, "status": status, "passed": passed,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "training": {"steps": steps, "parameter_count": parameter_count, "trainable_parameter_count": trainable_count},
        "prediction_records": prediction_records,
        "prediction_gate": {"passed": True, "checks": pchecks, "values": pvalues, "strata": pstrata, "thresholds": protocol["prediction_gate"]},
        "raw_reconstruction_records": raw_records,
        "raw_reconstruction_gate": {"passed": all(raw_checks.values()), "checks": raw_checks, "values": raw_values, "strata": raw_strata, "thresholds": protocol["reconstruction_gate"]},
        "terminal_sibling_assignments": assignments,
        "reconstruction_records": matched_records,
        "reconstruction_gate": {"passed": passed, "checks": checks, "values": values, "strata": strata, "thresholds": protocol["reconstruction_gate"]},
    }
    _dump(output_dir / "report.json", report); print(f"{status} report={output_dir/'report.json'}", flush=True)


if __name__ == "__main__":
    main()
