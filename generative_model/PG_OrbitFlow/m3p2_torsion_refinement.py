"""Run M3.2 frozen-feature full-panel torsion-head refinement and Tier-32 Gates."""

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
from .graph_automorphism import build_graph_automorphism_contract, optimal_automorphism_circular_assignment
from .m0_oracle_reconstruction import build_oracle_targets, reconstruction_metrics
from .m2_torsion_training import _learned_decoder_targets
from .m2p1_phase_training import _prediction_checks, _reconstruction_checks
from .m3_tier32_training import _decoder_task, _model_setting, _save
from .m3p1_automorphism_training import _matched_metrics, _prediction_record
from .m3p1_model import AutomorphismSetQuotientICPredictor
from .orbit_ic_model import build_orbit_ic_example
from .orbit_kinematics import build_orbit_parameterization
from .overfit import _panel_samples


SCHEMA_VERSION = "pg-orbitflow-m3p2-refinement-training-v1"


def _dump(path: Path, value: dict) -> None: path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-m3p2-refinement-protocol-v1": raise ValueError("unsupported M3.2 protocol")
    if _sha256(Path(protocol["package_dir"]) / "manifest.json") != protocol["canonical_manifest_sha256"]: raise RuntimeError("canonical manifest changed")
    for item in protocol["parent"].values():
        if _sha256(Path(item["path"])) != item["sha256"]: raise RuntimeError("M3.2 parent identity changed")
    if _sha256(Path(protocol["readiness_audit"]["path"])) != protocol["readiness_audit"]["sha256"]: raise RuntimeError("M3.2 readiness identity changed")
    for item in protocol["implementation"].values():
        if _sha256(Path(item["path"])) != item["sha256"]: raise RuntimeError("M3.2 implementation changed")
    return protocol


def _predict_cached(model, features, automorphism, *, device):
    import torch
    raw = model.parent.base.torsion_head(features); base = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-8); rows = [row for row in base]
    for group in automorphism.groups:
        values = model.predict_group(features, group, device=device)
        for orbit, value in zip(group, values, strict=True): rows[orbit] = value
    return torch.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--training-device", choices=("cpu", "cuda"), default="cuda"); args = parser.parse_args(); import torch
    if args.training_device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    protocol_path = args.protocol.resolve(); protocol = _load_protocol(protocol_path); output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=False); seed = int(protocol["seed"]); random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False); tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}; examples = []
    for sample in _panel_samples(protocol, tier):
        geometry = build_geometry_contract(sample); graph, targets = build_orbit_ic_example(sample, geometry); automorphism = build_graph_automorphism_contract(sample, geometry); examples.append((sample, geometry, graph, targets, automorphism))
    model = AutomorphismSetQuotientICPredictor(**_model_setting(protocol)); parent = torch.load(protocol["parent"]["checkpoint"]["path"], map_location="cpu", weights_only=False); model.load_state_dict(parent["model"], strict=True)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for module in (model.parent.base.torsion_head, model.automorphism_head):
        for parameter in module.parameters(): parameter.requires_grad_(True)
    model = model.to(args.training_device); parameter_count = sum(parameter.numel() for parameter in model.parameters()); trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]; trainable_count = sum(parameter.numel() for parameter in trainable)
    if parameter_count != int(protocol["model"]["parameter_count_expected"]) or trainable_count != int(protocol["model"]["trainable_parameter_count_expected"]): raise RuntimeError("M3.2 parameter count changed")
    cached = []; model.eval()
    with torch.no_grad():
        for _, _, graph, targets, automorphism in examples:
            model(graph, automorphism, device=args.training_device); features = model.parent._captured_torsion_input.detach().clone(); target = torch.as_tensor(targets.torsion_target_sincos, dtype=torch.float32, device=args.training_device); cached.append((features, target, automorphism))
    training = protocol["training"]; optimizer = torch.optim.AdamW(trainable, lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])); steps = int(training["optimizer_steps"]); losses = []
    for step in range(1, steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); total = torch.zeros((), device=args.training_device)
        for features, target, automorphism in cached:
            prediction = _predict_cached(model, features, automorphism, device=args.training_device); aligned = optimal_automorphism_circular_assignment(prediction, target, automorphism); total = total + torch.mean(1.0 - torch.sum(prediction * aligned, dim=-1).clamp(-1, 1)) / len(cached)
        total.backward(); gradient = torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip_norm"]))
        if not torch.isfinite(total) or not torch.isfinite(torch.as_tensor(gradient)): raise RuntimeError(f"non-finite M3.2 training at step {step}")
        optimizer.step(); losses.append((step, float(total.detach()), float(gradient)))
        if step == 1 or step % int(training["checkpoint_every"]) == 0 or step == steps:
            model.eval(); milestone = []
            with torch.no_grad():
                for sample, _, graph, targets, automorphism in examples:
                    prediction = {key: value.cpu().numpy() for key, value in model(graph, automorphism, device=args.training_device).items()}; milestone.append(_prediction_record(sample, targets, prediction, automorphism))
            values, checks, _ = _prediction_checks(milestone, protocol["prediction_gate"]); print(f"step={step}/{steps} loss={float(total):.6g} bond={values['bond_orbit_mae_max_angstrom']:.6g}A angle={values['angle_orbit_mae_max_degrees']:.6g}deg torsion={values['torsion_orbit_circular_mae_max_degrees']:.6g}deg nonplanar={values['nonplanar_torsion_orbit_circular_mae_max_degrees']:.6g}deg gate={'PASS' if all(checks.values()) else 'FAIL'}", flush=True); _save(output_dir / f"step-{step:06d}.pt", model, optimizer, step, _sha256(protocol_path))
    _save(output_dir / "last.pt", model, optimizer, steps, _sha256(protocol_path)); np.savez_compressed(output_dir / "losses.npz", step=np.asarray([row[0] for row in losses]), loss=np.asarray([row[1] for row in losses]), gradient_norm=np.asarray([row[2] for row in losses]))
    model.eval(); predictions = []
    with torch.no_grad():
        for _, _, graph, _, automorphism in examples: predictions.append({key: value.cpu().numpy() for key, value in model(graph, automorphism, device=args.training_device).items()})
    prediction_records = [_prediction_record(sample, targets, prediction, automorphism) for (sample, _, _, targets, automorphism), prediction in zip(examples, predictions, strict=True)]; pvalues, pchecks, pstrata = _prediction_checks(prediction_records, protocol["prediction_gate"]); prediction_passed = all(pchecks.values()); np.savez_compressed(output_dir / "predictions.npz", **{f"{sample.package_index}_{key}": value for (sample, _, _, _, _), prediction in zip(examples, predictions, strict=True) for key, value in prediction.items()})
    if not prediction_passed:
        status = "FAIL_M3P2_PREDICTION_GATE_STOP_BEFORE_DECODER"; report = {"schema_version": SCHEMA_VERSION, "status": status, "passed": False, "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)}, "training": {"steps": steps, "parameter_count": parameter_count, "trainable_parameter_count": trainable_count}, "prediction_records": prediction_records, "prediction_gate": {"passed": False, "checks": pchecks, "values": pvalues, "strata": pstrata, "thresholds": protocol["prediction_gate"]}}; _dump(output_dir / "report.json", report); print(f"{status} report={output_dir/'report.json'}", flush=True); return
    model.to("cpu"); torch.cuda.empty_cache(); decoder_protocol = {key: protocol["decoder"][key] for key in ("initialization", "optimization", "energy_weights")}; starts = int(protocol["decoder"]["initialization"]["starts_per_molecule"]); stride = int(protocol["decoder"]["initialization"]["molecule_specific_seed_stride"]); tasks = []
    for (sample, geometry, _, _, _), prediction in zip(examples, predictions, strict=True):
        decoder_targets = _learned_decoder_targets(prediction, geometry); specification = build_orbit_parameterization(sample)
        for start in range(starts): tasks.append((sample, geometry, decoder_targets, specification, decoder_protocol, seed + sample.package_index * stride + start, start))
    collected = {}; context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(protocol["decoder"]["parallel_workers"]), mp_context=context) as executor:
        for package_index, start, candidate, coordinates in executor.map(_decoder_task, tasks, chunksize=1): collected[(package_index, start)] = (candidate, coordinates)
    raw_records = []; matched_records = []; assignments = []; coordinate_parts = []; offsets = [0]
    for sample, geometry, _, _, automorphism in examples:
        candidates = [collected[(sample.package_index, start)][0] for start in range(starts)]; coordinate_candidates = [collected[(sample.package_index, start)][1] for start in range(starts)]; selected = min(range(starts), key=lambda index: candidates[index]["final_energy"]); coordinates = coordinate_candidates[selected]; raw = reconstruction_metrics(sample, geometry, build_oracle_targets(sample, geometry), coordinates); common = {"package_index": int(sample.package_index), "target_pg": sample.target_pg, "selected_start_index": selected, "selected_energy": float(candidates[selected]["final_energy"]), "finite_start_count": sum(candidate["all_values_finite"] for candidate in candidates), "all_starts_finite": all(candidate["all_values_finite"] for candidate in candidates)}; raw_records.append({**common, **raw}); matched, assignment = _matched_metrics(sample, geometry, automorphism, coordinates, raw); matched_records.append({**common, **matched}); assignments.append({"package_index": int(sample.package_index), **assignment}); coordinate_parts.append(coordinates.astype(np.float32)); offsets.append(offsets[-1] + len(coordinates)); print(f"reconstruct package_index={sample.package_index} PG={sample.target_pg} torsion={matched['torsion_circular_mae_degrees']:.6g} nonplanar={matched['nonplanar_torsion_circular_mae_degrees']:.6g} rmsd={matched['kabsch_rmsd_angstrom']:.6g}A", flush=True)
    raw_values, raw_checks, raw_strata = _reconstruction_checks(raw_records, protocol["reconstruction_gate"]); values, checks, strata = _reconstruction_checks(matched_records, protocol["reconstruction_gate"]); passed = all(checks.values()); status = "PASS_M3_TIER32" if passed else "FAIL_M3P2_RECONSTRUCTION_GATE"; np.savez_compressed(output_dir / "coordinates.npz", coordinates=np.concatenate(coordinate_parts), atom_offsets=np.asarray(offsets), package_indices=np.asarray([sample.package_index for sample, *_ in examples])); report = {"schema_version": SCHEMA_VERSION, "status": status, "passed": passed, "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)}, "training": {"steps": steps, "parameter_count": parameter_count, "trainable_parameter_count": trainable_count}, "prediction_records": prediction_records, "prediction_gate": {"passed": True, "checks": pchecks, "values": pvalues, "strata": pstrata, "thresholds": protocol["prediction_gate"]}, "raw_reconstruction_records": raw_records, "raw_reconstruction_gate": {"passed": all(raw_checks.values()), "checks": raw_checks, "values": raw_values, "strata": raw_strata, "thresholds": protocol["reconstruction_gate"]}, "centralizer_automorphism_assignments": assignments, "reconstruction_records": matched_records, "reconstruction_gate": {"passed": passed, "checks": checks, "values": values, "strata": strata, "thresholds": protocol["reconstruction_gate"]}}; _dump(output_dir / "report.json", report); print(f"{status} report={output_dir/'report.json'}", flush=True)


if __name__ == "__main__": main()
