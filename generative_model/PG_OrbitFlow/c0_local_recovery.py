"""Train and evaluate the frozen C0 multi-noise local-geometry curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .flow import make_training_example, tensorize_sample
from .geometry import build_geometry_contract, geometry_loss_bundle
from .group import operation_error_numpy, project_vectors_numpy
from .losses import overlap_loss, symmetry_mse
from .metrics import collision_audit
from .model import PGOrbitFlow
from .overfit import _fixed_time_evaluation, _gate, _panel_samples, _raw_evaluation
from .train import _model_kwargs


C0_SCHEMA_VERSION = "pg-orbitflow-c0-local-recovery-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-c0-protocol-v2":
        raise ValueError("unsupported C0 protocol schema")
    package_dir = Path(protocol["package_dir"])
    if _sha256(package_dir / "manifest.json") != protocol["canonical_manifest_sha256"]:
        raise ValueError("canonical manifest changed after C0 protocol freeze")
    base = Path(protocol["base_h1_protocol"]["path"])
    if _sha256(base) != protocol["base_h1_protocol"]["sha256"]:
        raise ValueError("H1 base protocol changed after C0 freeze")
    if protocol["training"]["optimizer_steps"] != 1024:
        raise ValueError("C0 must use 1024 optimizer steps")
    if protocol["training"]["batch_size_molecules"] != 4:
        raise ValueError("C0 must use batch size 4")
    if protocol["training"]["short_rollout_steps"] != 4:
        raise ValueError("C0 must use a 4-step differentiable rollout")
    if protocol["curriculum"]["slots"] != [
        "small",
        "small",
        "small",
        "small",
        "medium",
        "medium",
        "medium",
        "large",
        "large",
        "harmonic",
    ]:
        raise ValueError("C0 curriculum exposure slots changed")
    if not protocol["curriculum"].get(
        "point_group_curriculum_independent", False
    ):
        raise ValueError("C0 v2 requires an independent curriculum cursor per PG")
    return protocol


def make_local_recovery_example(
    sample,
    *,
    seed: int,
    sigma_range_angstrom: tuple[float, float],
    device: str,
    coordinate_scale_angstrom: float,
):
    import torch

    lower, upper = (float(value) for value in sigma_range_angstrom)
    if not 0 < lower <= upper <= 0.98:
        raise ValueError("local noise sigma range must satisfy 0<low<=high<=0.98 Å")
    rng = np.random.default_rng(int(seed))
    sigma = float(rng.uniform(lower, upper)) if lower < upper else lower
    target_angstrom = np.asarray(sample.symmetric_target_angstrom, dtype=np.float64)
    raw_noise = rng.normal(size=target_angstrom.shape) * sigma
    symmetric_noise = project_vectors_numpy(
        raw_noise, sample.operation_matrices, sample.permutation_index
    )
    positions_angstrom = target_angstrom + symmetric_noise
    positions_angstrom -= positions_angstrom.mean(axis=0, keepdims=True)
    operation = operation_error_numpy(
        positions_angstrom, sample.operation_matrices, sample.permutation_index
    )
    if operation["max_atom_error_angstrom"] > 1e-5:
        raise RuntimeError("local noisy coordinates left the invariant subspace")
    time_value = float(np.clip(1.0 - sigma, 0.02, 0.95))
    tensors = tensorize_sample(
        sample,
        device=device,
        coordinate_scale_angstrom=coordinate_scale_angstrom,
    )
    target = torch.as_tensor(
        target_angstrom / coordinate_scale_angstrom,
        dtype=torch.float32,
        device=device,
    )
    positions = torch.as_tensor(
        positions_angstrom / coordinate_scale_angstrom,
        dtype=torch.float32,
        device=device,
    )
    time = torch.as_tensor(time_value, dtype=torch.float32, device=device)
    tensors.update(
        {
            "positions_t": positions,
            "time": time,
            "target_velocity": (target - positions) / (1.0 - time),
            "target_positions": target,
            "transport_target_positions": target,
            "geometry_target_positions": target,
            "noise_sigma_angstrom": sigma,
            "curriculum_kind": "local",
        }
    )
    return tensors


def _harmonic_example(sample, *, seed: int, device: str, coordinate_scale: float):
    values = make_training_example(
        sample,
        seed=seed,
        device=device,
        coordinate_scale_angstrom=coordinate_scale,
        prior_type="graph_harmonic",
        harmonic_alpha=1.0,
        transport_type="cn_phase_aligned",
    )
    values["noise_sigma_angstrom"] = float("nan")
    values["curriculum_kind"] = "harmonic"
    return values


def differentiable_short_rollout(model, example: dict, *, steps: int):
    import torch

    if steps <= 0:
        raise ValueError("rollout steps must be positive")
    positions = example["positions_t"]
    start_time = example["time"]
    step_size = (1.0 - start_time) / steps
    kwargs = _model_kwargs(example)
    for index in range(steps):
        time = start_time + index * step_size
        velocity = model(positions=positions, time=time, **kwargs)
        positions = positions + step_size * velocity
        positions = positions - positions.mean(dim=0, keepdim=True)
    if not bool(torch.isfinite(positions).all()):
        raise RuntimeError("short rollout produced NaN/Inf")
    return positions


def _geometry_objective(values: dict, objective: dict, *, early_boost):
    return (
        float(objective["bond_weight"]) * early_boost * values["bond_loss"]
        + float(objective["angle_weight"]) * early_boost * values["angle_loss"]
        + float(objective["torsion_weight"]) * values["torsion_loss"]
        + float(objective["local_pair_weight"]) * values["local_pair_loss"]
        + float(objective["ring_weight"]) * early_boost * values["ring_loss"]
        + float(objective["chirality_weight"]) * values["chirality_loss"]
    )


def c0_loss(model, example: dict, contract, protocol: dict):
    import torch

    objective = protocol["objective"]
    coordinate_scale = float(protocol["data"]["coordinate_scale_angstrom"])
    prediction = model(
        positions=example["positions_t"],
        time=example["time"],
        **_model_kwargs(example),
    )
    flow = torch.mean(torch.square(prediction - example["target_velocity"]))
    endpoint = example["positions_t"] + (1.0 - example["time"]) * prediction
    endpoint_geometry = geometry_loss_bundle(
        endpoint,
        example["geometry_target_positions"],
        contract,
        coordinate_scale_angstrom=coordinate_scale,
    )
    endpoint_overlap = overlap_loss(
        endpoint,
        example["atomic_numbers"],
        example["bond_index"],
        coordinate_scale_angstrom=coordinate_scale,
    )
    endpoint_symmetry, _ = symmetry_mse(
        endpoint, example["operation_matrices"], example["permutation_index"]
    )
    rollout = differentiable_short_rollout(
        model, example, steps=int(protocol["training"]["short_rollout_steps"])
    )
    rollout_geometry = geometry_loss_bundle(
        rollout,
        example["geometry_target_positions"],
        contract,
        coordinate_scale_angstrom=coordinate_scale,
    )
    rollout_overlap = overlap_loss(
        rollout,
        example["atomic_numbers"],
        example["bond_index"],
        coordinate_scale_angstrom=coordinate_scale,
    )
    rollout_symmetry, _ = symmetry_mse(
        rollout, example["operation_matrices"], example["permutation_index"]
    )
    early_boost = 1.0 + float(objective["early_local_boost"]) * torch.square(
        1.0 - example["time"]
    )
    endpoint_objective = _geometry_objective(
        endpoint_geometry, objective, early_boost=early_boost
    )
    rollout_objective = _geometry_objective(
        rollout_geometry, objective, early_boost=early_boost
    )
    total = (
        float(objective["flow_weight"]) * flow
        + endpoint_objective
        + float(objective["overlap_weight"]) * endpoint_overlap
        + float(objective["symmetry_weight"]) * endpoint_symmetry
        + float(objective["short_rollout_weight"])
        * (
            rollout_objective
            + float(objective["overlap_weight"]) * rollout_overlap
            + float(objective["symmetry_weight"]) * rollout_symmetry
        )
    )
    return {
        "loss": total,
        "flow_mse": flow,
        "endpoint_geometry_loss": endpoint_objective,
        "endpoint_bond_mae_angstrom": endpoint_geometry["bond_mae_angstrom"],
        "endpoint_angle_mae_degrees": endpoint_geometry["angle_mae_degrees"],
        "endpoint_overlap_loss": endpoint_overlap,
        "endpoint_symmetry_mse": endpoint_symmetry,
        "rollout_geometry_loss": rollout_objective,
        "rollout_bond_mae_angstrom": rollout_geometry["bond_mae_angstrom"],
        "rollout_angle_mae_degrees": rollout_geometry["angle_mae_degrees"],
        "rollout_overlap_loss": rollout_overlap,
        "rollout_symmetry_mse": rollout_symmetry,
        "endpoint": endpoint,
        "rollout": rollout,
    }


def _save_checkpoint(path: Path, *, model, optimizer, step: int, protocol_sha: str):
    import torch

    torch.save(
        {
            "schema_version": C0_SCHEMA_VERSION,
            "protocol_sha256": protocol_sha,
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        path,
    )


def _local_evaluation(model, samples, contracts, protocol: dict, device: str):
    import torch

    coordinate_scale = float(protocol["data"]["coordinate_scale_angstrom"])
    setting = protocol["local_evaluation"]
    records = []
    coordinate_parts = []
    offsets = [0]
    model.eval()
    with torch.no_grad():
        for sample in samples:
            contract = contracts[sample.package_index]
            for seed_index in range(int(setting["seeds_per_molecule"])):
                seed = (
                    int(protocol["seed"])
                    + 110_000_000
                    + sample.package_index * 1009
                    + seed_index
                )
                sigma = float(setting["sigma_angstrom"])
                example = make_local_recovery_example(
                    sample,
                    seed=seed,
                    sigma_range_angstrom=(sigma, sigma),
                    device=device,
                    coordinate_scale_angstrom=coordinate_scale,
                )
                prediction = model(
                    positions=example["positions_t"],
                    time=example["time"],
                    **_model_kwargs(example),
                )
                endpoint = example["positions_t"] + (1.0 - example["time"]) * prediction
                geometry = geometry_loss_bundle(
                    endpoint,
                    example["geometry_target_positions"],
                    contract,
                    coordinate_scale_angstrom=coordinate_scale,
                )
                coordinates = endpoint.detach().cpu().numpy() * coordinate_scale
                target = example["geometry_target_positions"].detach().cpu().numpy()
                endpoint_rmsd = float(
                    np.sqrt(np.mean(np.sum(np.square(coordinates - target * coordinate_scale), axis=1)))
                )
                collision = collision_audit(
                    coordinates, sample.atomic_numbers, sample.bond_index
                )
                operation = operation_error_numpy(
                    coordinates, sample.operation_matrices, sample.permutation_index
                )
                records.append(
                    {
                        "package_index": sample.package_index,
                        "molecule_id": sample.molecule_id,
                        "target_pg": sample.target_pg,
                        "seed": seed,
                        "endpoint_rmsd_angstrom": endpoint_rmsd,
                        "bond_length_mae_angstrom": float(
                            geometry["bond_mae_angstrom"].item()
                        ),
                        "angle_mae_degrees": float(
                            geometry["angle_mae_degrees"].item()
                        ),
                        "chirality_preserved_fraction": float(
                            geometry["chirality_preserved_fraction"].item()
                        ),
                        "active_chirality_count": int(
                            geometry["active_chirality_count"]
                        ),
                        **collision,
                        **operation,
                    }
                )
                coordinate_parts.append(coordinates.astype(np.float32))
                offsets.append(offsets[-1] + len(coordinates))
    active_count = sum(record["active_chirality_count"] for record in records)
    chirality_fraction = (
        sum(
            record["chirality_preserved_fraction"]
            * record["active_chirality_count"]
            for record in records
        )
        / active_count
        if active_count
        else 1.0
    )
    summary = {
        "case_count": len(records),
        "means": {
            key: float(np.mean([record[key] for record in records]))
            for key in (
                "endpoint_rmsd_angstrom",
                "bond_length_mae_angstrom",
                "angle_mae_degrees",
            )
        },
        "chirality_preserved_fraction": float(chirality_fraction),
        "active_chirality_case_count": int(active_count),
        "collision_free_fraction": float(
            np.mean([record["collision_free"] for record in records])
        ),
        "max_operation_atom_error_angstrom": float(
            max(record["max_atom_error_angstrom"] for record in records)
        ),
        "records": records,
        "posthoc_hard_projection_used": False,
    }
    arrays = {
        "coordinates": np.concatenate(coordinate_parts, axis=0),
        "atom_offsets": np.asarray(offsets, dtype=np.int64),
        "package_indices": np.asarray(
            [record["package_index"] for record in records], dtype=np.int64
        ),
        "seeds": np.asarray([record["seed"] for record in records], dtype=np.int64),
    }
    return summary, arrays


def _local_gate(protocol: dict, evaluation: dict) -> dict:
    thresholds = protocol["local_gate"]
    values = {
        "endpoint_rmsd_mean_angstrom": evaluation["means"]["endpoint_rmsd_angstrom"],
        "bond_length_mae_mean_angstrom": evaluation["means"][
            "bond_length_mae_angstrom"
        ],
        "angle_mae_mean_degrees": evaluation["means"]["angle_mae_degrees"],
        "chirality_preserved_fraction": evaluation[
            "chirality_preserved_fraction"
        ],
        "collision_free_fraction": evaluation["collision_free_fraction"],
        "max_operation_atom_error_angstrom": evaluation[
            "max_operation_atom_error_angstrom"
        ],
    }
    checks = {
        "endpoint_rmsd_mean_max_angstrom": values["endpoint_rmsd_mean_angstrom"]
        <= thresholds["endpoint_rmsd_mean_max_angstrom"],
        "bond_length_mae_mean_max_angstrom": values[
            "bond_length_mae_mean_angstrom"
        ]
        <= thresholds["bond_length_mae_mean_max_angstrom"],
        "angle_mae_mean_max_degrees": values["angle_mae_mean_degrees"]
        <= thresholds["angle_mae_mean_max_degrees"],
        "chirality_preserved_fraction_min": values[
            "chirality_preserved_fraction"
        ]
        >= thresholds["chirality_preserved_fraction_min"],
        "collision_free_fraction_min": values["collision_free_fraction"]
        >= thresholds["collision_free_fraction_min"],
        "max_operation_atom_error_max_angstrom": values[
            "max_operation_atom_error_angstrom"
        ]
        <= thresholds["max_operation_atom_error_max_angstrom"],
        "posthoc_hard_projection_never_used": True,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "values": values,
        "thresholds": thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    import torch

    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_sha = _sha256(protocol_path)
    seed = int(protocol["seed"]) + 4
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)

    tier_config = {
        "records": protocol["panel"]["records"],
        "evaluation": protocol["original_tier4_evaluation"],
        "gate": protocol["original_tier4_gate"],
    }
    samples = _panel_samples(protocol, tier_config)
    contracts = {
        sample.package_index: build_geometry_contract(sample) for sample in samples
    }
    model = PGOrbitFlow(
        **{
            key: protocol["model"][key]
            for key in ("hidden_dim", "layers", "radial_dim", "radial_max")
        }
    ).to(args.device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(protocol["model"]["parameter_count_expected"]):
        raise RuntimeError("model parameter count changed after C0 freeze")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    buckets = {
        pg: [sample for sample in samples if sample.target_pg == pg]
        for pg in ("C2", "C3")
    }
    if len(buckets["C2"]) != len(buckets["C3"]):
        raise ValueError("C0 panel is not PG-balanced")
    rng = np.random.default_rng(seed + 17)
    steps = int(protocol["training"]["optimizer_steps"])
    batch_size = int(protocol["training"]["batch_size_molecules"])
    checkpoint_every = int(protocol["training"]["checkpoint_every"])
    slots = tuple(protocol["curriculum"]["slots"])
    exposures = Counter()
    curriculum_cursors = Counter()
    loss_records = []
    coordinate_scale = float(protocol["data"]["coordinate_scale_angstrom"])
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        metrics = Counter()
        for batch_item in range(batch_size):
            pg = ("C2", "C3")[(step * batch_size + batch_item) % 2]
            sample = buckets[pg][int(rng.integers(len(buckets[pg])))]
            kind = slots[curriculum_cursors[pg] % len(slots)]
            curriculum_cursors[pg] += 1
            example_seed = (
                seed + step * 1_000_003 + batch_item * 10_007 + sample.package_index
            )
            if kind == "harmonic":
                example = _harmonic_example(
                    sample,
                    seed=example_seed,
                    device=args.device,
                    coordinate_scale=coordinate_scale,
                )
            else:
                sigma_range = tuple(
                    protocol["curriculum"][f"{kind}_sigma_angstrom"]
                )
                example = make_local_recovery_example(
                    sample,
                    seed=example_seed,
                    sigma_range_angstrom=sigma_range,
                    device=args.device,
                    coordinate_scale_angstrom=coordinate_scale,
                )
            exposures[f"{pg}_{kind}"] += 1
            losses = c0_loss(
                model, example, contracts[sample.package_index], protocol
            )
            (losses["loss"] / batch_size).backward()
            for key in (
                "loss",
                "flow_mse",
                "endpoint_geometry_loss",
                "endpoint_bond_mae_angstrom",
                "endpoint_angle_mae_degrees",
                "endpoint_overlap_loss",
                "endpoint_symmetry_mse",
                "rollout_geometry_loss",
                "rollout_bond_mae_angstrom",
                "rollout_angle_mae_degrees",
                "rollout_overlap_loss",
                "rollout_symmetry_mse",
            ):
                metrics[key] += float(losses[key].detach().item()) / batch_size
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(protocol["training"]["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise RuntimeError(f"non-finite C0 gradient at step {step}")
        optimizer.step()
        record = {
            "step": step,
            **{key: float(value) for key, value in metrics.items()},
            "gradient_norm": float(gradient_norm),
        }
        if not np.isfinite(list(record.values())).all():
            raise RuntimeError(f"non-finite C0 metric at step {step}")
        loss_records.append(record)
        if step == 1 or step % checkpoint_every == 0 or step == steps:
            print(
                f"step={step}/{steps} loss={record['loss']:.6f} "
                f"flow={record['flow_mse']:.6f} "
                f"bond={record['endpoint_bond_mae_angstrom']:.6f}A "
                f"angle={record['endpoint_angle_mae_degrees']:.3f}deg",
                flush=True,
            )
            _save_checkpoint(
                output_dir / f"step-{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                protocol_sha=protocol_sha,
            )
    _save_checkpoint(
        output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        step=steps,
        protocol_sha=protocol_sha,
    )
    local_evaluation, local_arrays = _local_evaluation(
        model, samples, contracts, protocol, args.device
    )
    local_gate = _local_gate(protocol, local_evaluation)
    fixed = _fixed_time_evaluation(
        model, samples, protocol, tier_config, args.device
    )
    raw, raw_arrays = _raw_evaluation(
        model, samples, protocol, tier_config, args.device
    )
    original_gate = _gate(tier_config, loss_records, fixed, raw)
    np.savez_compressed(
        output_dir / "losses.npz",
        **{
            key: np.asarray([record[key] for record in loss_records])
            for key in loss_records[0]
        },
    )
    np.savez_compressed(output_dir / "local_recovery_coordinates.npz", **local_arrays)
    np.savez_compressed(output_dir / "raw_coordinates.npz", **raw_arrays)
    status = (
        "PASS_C0_LOCAL_RECOVERY_ADVANCE_TO_FULL_PRIOR_CURRICULUM"
        if local_gate["passed"]
        else "FAIL_C0_LOCAL_RECOVERY_STOP_CARTESIAN_BACKBONE"
    )
    report = {
        "schema_version": C0_SCHEMA_VERSION,
        "status": status,
        "passed": local_gate["passed"],
        "claim_scope": "bounded local chemical-geometry recovery only",
        "quality_claim": False,
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "data": {
            "molecule_count": len(samples),
            "package_indices": [sample.package_index for sample in samples],
            "point_group_counts": dict(Counter(sample.target_pg for sample in samples)),
            "molecule_exposures": int(sum(exposures.values())),
            "curriculum_exposures": dict(sorted(exposures.items())),
            "iid_validation_used": False,
            "iid_test_used": False,
            "core_ood_used": False,
        },
        "geometry_contract": {
            str(sample.package_index): {
                "bonds": int(contracts[sample.package_index].bond_index.shape[1]),
                "angles": int(contracts[sample.package_index].angle_index.shape[1]),
                "torsions": int(contracts[sample.package_index].torsion_index.shape[1]),
                "local_pairs": int(
                    contracts[sample.package_index].local_pair_index.shape[1]
                ),
                "ring_bonds": int(
                    len(contracts[sample.package_index].ring_bond_indices)
                ),
                "chirality_candidates": int(
                    contracts[sample.package_index].chirality_index.shape[1]
                ),
            }
            for sample in samples
        },
        "training": {
            "optimizer_steps": steps,
            "batch_size_molecules": batch_size,
            "parameter_count": parameter_count,
            "short_rollout_steps": int(protocol["training"]["short_rollout_steps"]),
            "device": args.device,
            "all_losses_gradients_finite": True,
        },
        "local_recovery_evaluation": local_evaluation,
        "local_gate": local_gate,
        "original_harmonic_fixed_time_evaluation": fixed,
        "original_raw_ode_evaluation": raw,
        "original_tier4_gate_unchanged": original_gate,
        "decision": protocol["progression"][
            "if_local_passes" if local_gate["passed"] else "if_local_fails"
        ],
        "isolation": {
            "posthoc_hard_projection_used": False,
            "f02_used": False,
            "external_pretraining_used": False,
            "historical_checkpoint_used": False,
            "canonical_v2_modified": False,
            "h1_h3_results_modified": False,
        },
        "execution_revision": protocol["execution_revision"],
    }
    _dump(output_dir / "report.json", report)
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _dump(
        output_dir / "manifest.json",
        {"schema_version": "pg-orbitflow-c0-run-manifest-v1", "artifacts": artifacts},
    )
    print(f"{status} report={output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
