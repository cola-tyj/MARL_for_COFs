"""Reproducible training entry point for the isolated PG-OrbitFlow branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# Required by torch deterministic algorithms for CUDA >= 10.2.  This must be
# set before the first cuBLAS operation; using setdefault respects an explicit
# operator choice while making the frozen training command self-contained.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from . import __version__
from .data import PGOrbitFlowDataset, PointGroupBalancedSampler
from .flow import make_training_example, sample_raw_ode
from .group import operation_error_numpy
from .losses import compute_loss
from .model import PGOrbitFlow


TRAINING_SCHEMA_VERSION = "pg-orbitflow-training-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "seed", "data", "model", "training", "objective", "evaluation"}
    if set(config) != required:
        raise ValueError(f"config top-level keys must be exactly {sorted(required)}")
    if config["schema_version"] != "pg-orbitflow-config-v1":
        raise ValueError("unsupported config schema")
    return config


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requested CUDA but torch.cuda.is_available() is false")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    return requested


def _model_kwargs(example: dict) -> dict:
    return {
        key: example[key]
        for key in (
            "atom_types",
            "formal_charges",
            "radical_electrons",
            "target_pg_index",
            "group_features",
            "atom_group_features",
            "invariant_bond_order",
        )
    }


def _loss_for_example(model, example: dict, config: dict):
    prediction = model(
        positions=example["positions_t"],
        time=example["time"],
        **_model_kwargs(example),
    )
    objective = config["objective"]
    return compute_loss(
        predicted_velocity=prediction,
        target_velocity=example["target_velocity"],
        positions_t=example["positions_t"],
        time=example["time"],
        geometry_target_positions=example.get(
            "geometry_target_positions", example["target_positions"]
        ),
        transport_target_positions=example.get(
            "transport_target_positions", example["target_positions"]
        ),
        atomic_numbers=example["atomic_numbers"],
        bond_index=example["bond_index"],
        matrices=example["operation_matrices"],
        permutations=example["permutation_index"],
        coordinate_scale_angstrom=float(config["data"]["coordinate_scale_angstrom"]),
        bond_weight=float(objective["bond_weight"]),
        overlap_weight=float(objective["overlap_weight"]),
        symmetry_weight=float(objective["symmetry_weight"]),
        endpoint_weight=float(objective.get("endpoint_weight", 0.0)),
    )


def _flow_config_kwargs(config: dict) -> dict:
    prior = config.get("prior", {})
    transport = config.get("transport", {})
    return {
        "prior_type": str(prior.get("type", "projected_gaussian")),
        "harmonic_alpha": float(prior.get("harmonic_alpha", 1.0)),
        "transport_type": str(transport.get("type", "independent")),
        "centralizer_phase_count": int(
            transport.get("centralizer_phase_count", 24)
        ),
    }


def _evaluate(model, dataset, config: dict, *, device: str, seed_offset: int) -> dict:
    import torch

    model.eval()
    totals: Counter[str] = Counter()
    by_pg: dict[str, Counter[str]] = {}
    count = 0
    maximum_operation_error = 0.0
    coordinate_scale = float(config["data"]["coordinate_scale_angstrom"])
    times = tuple(float(value) for value in config["evaluation"]["flow_times"])
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            group_metrics = by_pg.setdefault(sample.target_pg, Counter())
            for time_index, fixed_time in enumerate(times):
                example = make_training_example(
                    sample,
                    seed=seed_offset + sample.package_index * 101 + time_index,
                    device=device,
                    coordinate_scale_angstrom=coordinate_scale,
                    time_value=fixed_time,
                    **_flow_config_kwargs(config),
                )
                values = _loss_for_example(model, example, config)
                endpoint = values["endpoint"]
                rmsd = torch.sqrt(
                    torch.mean(
                        torch.sum(
                            torch.square(
                                endpoint - example["geometry_target_positions"]
                            ),
                            dim=1,
                        )
                    )
                ) * coordinate_scale
                for key in (
                    "loss",
                    "flow_mse",
                    "bond_length_mse",
                    "overlap_loss",
                    "symmetry_mse",
                    "endpoint_auxiliary_mse",
                ):
                    scalar = float(values[key].item())
                    totals[key] += scalar
                    group_metrics[key] += scalar
                totals["endpoint_rmsd_angstrom"] += float(rmsd.item())
                group_metrics["endpoint_rmsd_angstrom"] += float(rmsd.item())
                raw_endpoint = endpoint.detach().cpu().numpy() * coordinate_scale
                error = operation_error_numpy(
                    raw_endpoint, sample.operation_matrices, sample.permutation_index
                )["max_atom_error_angstrom"]
                maximum_operation_error = max(maximum_operation_error, error)
                count += 1
    if count == 0:
        raise RuntimeError("validation produced no cases")
    per_pg_count = len(times) * len(dataset)
    # Use actual per-group counts rather than assuming a balanced validation set.
    pg_sample_counts = Counter(dataset[i].target_pg for i in range(len(dataset)))
    return {
        "case_count": count,
        "means": {key: float(value / count) for key, value in totals.items()},
        "by_point_group": {
            pg: {
                "case_count": int(pg_sample_counts[pg] * len(times)),
                "means": {
                    key: float(value / (pg_sample_counts[pg] * len(times)))
                    for key, value in metrics.items()
                },
            }
            for pg, metrics in sorted(by_pg.items())
        },
        "max_raw_operation_atom_error_angstrom": maximum_operation_error,
        "posthoc_hard_projection_used": False,
    }


def _save_checkpoint(path: Path, *, model, optimizer, sampler, step: int, config: dict) -> None:
    import torch

    state = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "pg_orbitflow_version": __version__,
        "step": int(step),
        "config": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, path)


def _load_checkpoint(path: Path, *, model, optimizer, sampler, config: dict, device: str) -> int:
    import torch

    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("schema_version") != TRAINING_SCHEMA_VERSION or state.get("config") != config:
        raise ValueError("checkpoint schema/config mismatch")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    sampler.load_state_dict(state["sampler"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.random.set_rng_state(state["torch_random_state"])
    if device == "cuda" and state["cuda_random_state"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_random_state"])
    return int(state["step"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    import torch

    config = _load_config(args.config.resolve())
    device = _resolve_device(str(config["training"]["device"]))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and args.resume is None:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)

    package_dir = Path(config["data"]["package_dir"]).resolve()
    point_groups = tuple(config["data"]["point_groups"])
    train_dataset = PGOrbitFlowDataset(
        package_dir,
        split="train",
        split_scheme=str(config["data"]["split_scheme"]),
        point_groups=point_groups,
        max_samples=config["data"]["max_train_samples"],
    )
    validation_dataset = PGOrbitFlowDataset(
        package_dir,
        split="val",
        split_scheme=str(config["data"]["split_scheme"]),
        point_groups=point_groups,
        max_samples=config["data"]["max_validation_samples"],
    )
    sampler = PointGroupBalancedSampler(train_dataset, seed=seed + 17)
    sample_iterator = iter(sampler)

    model = PGOrbitFlow(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    start_step = 0
    if args.resume is not None:
        start_step = _load_checkpoint(
            args.resume.resolve(),
            model=model,
            optimizer=optimizer,
            sampler=sampler,
            config=config,
            device=device,
        )
    max_steps = int(config["training"]["max_steps"])
    batch_size = int(config["training"]["batch_size"])
    checkpoint_every = int(config["training"]["checkpoint_every"])
    if not 0 <= start_step < max_steps or min(batch_size, checkpoint_every, max_steps) <= 0:
        raise ValueError("invalid training/resume step settings")

    exposure = Counter()
    unique_packages: set[int] = set()
    loss_records: list[dict] = []
    coordinate_scale = float(config["data"]["coordinate_scale_angstrom"])
    for step in range(start_step + 1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_metrics = Counter()
        for batch_item in range(batch_size):
            dataset_index = next(sample_iterator)
            sample = train_dataset[dataset_index]
            exposure[sample.target_pg] += 1
            unique_packages.add(sample.package_index)
            example_seed = seed + step * 1_000_003 + batch_item * 10_007 + sample.package_index
            example = make_training_example(
                sample,
                seed=example_seed,
                device=device,
                coordinate_scale_angstrom=coordinate_scale,
                **_flow_config_kwargs(config),
            )
            losses = _loss_for_example(model, example, config)
            (losses["loss"] / batch_size).backward()
            for key in (
                "loss",
                "flow_mse",
                "bond_length_mse",
                "overlap_loss",
                "symmetry_mse",
                "endpoint_auxiliary_mse",
            ):
                step_metrics[key] += float(losses[key].detach().item()) / batch_size
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        record = {
            "step": step,
            **{key: float(value) for key, value in step_metrics.items()},
            "gradient_norm": float(gradient_norm),
        }
        loss_records.append(record)
        if step == 1 or step % checkpoint_every == 0 or step == max_steps:
            print(
                f"step={step}/{max_steps} loss={record['loss']:.6f} "
                f"flow={record['flow_mse']:.6f} bond={record['bond_length_mse']:.6f} "
                f"sym={record['symmetry_mse']:.3e}",
                flush=True,
            )
            checkpoint_path = output_dir / f"step-{step:06d}.pt"
            _save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                sampler=sampler,
                step=step,
                config=config,
            )

    last_path = output_dir / "last.pt"
    _save_checkpoint(
        last_path,
        model=model,
        optimizer=optimizer,
        sampler=sampler,
        step=max_steps,
        config=config,
    )
    validation = _evaluate(
        model,
        validation_dataset,
        config,
        device=device,
        seed_offset=seed + 90_000_000,
    )
    # One raw ODE example is an engineering proof that inference does not call
    # the target projection.  Dataset-level generation belongs to the next Gate.
    raw_sample = validation_dataset[0]
    raw_coordinates = sample_raw_ode(
        model,
        raw_sample,
        seed=seed + 80_000_000 + raw_sample.package_index,
        steps=int(config["evaluation"]["ode_steps"]),
        device=device,
        coordinate_scale_angstrom=coordinate_scale,
        method=str(config["evaluation"]["ode_method"]),
        prior_type=str(config.get("prior", {}).get("type", "projected_gaussian")),
        harmonic_alpha=float(
            config.get("prior", {}).get("harmonic_alpha", 1.0)
        ),
    )
    raw_error = operation_error_numpy(
        raw_coordinates, raw_sample.operation_matrices, raw_sample.permutation_index
    )
    np.savez_compressed(
        output_dir / "raw_smoke_sample.npz",
        package_index=np.asarray([raw_sample.package_index], dtype=np.int64),
        coordinates=raw_coordinates.astype(np.float32),
    )
    config_snapshot = output_dir / "config.json"
    _json_dump(config_snapshot, config)
    np.savez_compressed(
        output_dir / "losses.npz",
        **{
            key: np.asarray([record[key] for record in loss_records])
            for key in loss_records[0]
        },
    )
    package_manifest = package_dir / "manifest.json"
    report = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "status": "PASS_ENGINEERING_SMOKE_NOT_SCIENTIFIC_GATE",
        "quality_claim": False,
        "isolation": {
            "our_et_flow_files_modified": False,
            "historical_checkpoint_loaded": False,
            "canonical_v2_read_only": True,
        },
        "data": {
            "package_dir": str(package_dir),
            "canonical_manifest_sha256": _sha256(package_manifest),
            "split_scheme": config["data"]["split_scheme"],
            "point_groups": list(point_groups),
            "train_dataset_size": len(train_dataset),
            "validation_dataset_size": len(validation_dataset),
            "molecule_exposures": int(sum(exposure.values())),
            "unique_train_molecules_seen": len(unique_packages),
            "exposure_by_point_group": dict(sorted(exposure.items())),
            "test_or_core_ood_used": False,
        },
        "training": {
            "steps": max_steps,
            "batch_size_molecules": batch_size,
            "optimizer_updates": max_steps,
            "learning_rate": config["training"]["learning_rate"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "device": device,
            "posthoc_hard_projection_used": False,
        },
        "validation": validation,
        "raw_ode_smoke": {
            "package_index": raw_sample.package_index,
            "steps": config["evaluation"]["ode_steps"],
            "method": config["evaluation"]["ode_method"],
            **raw_error,
            "posthoc_hard_projection_used": False,
        },
    }
    report_path = output_dir / "report.json"
    _json_dump(report_path, report)
    artifact_hashes = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _json_dump(
        output_dir / "manifest.json",
        {
            "schema_version": "pg-orbitflow-run-manifest-v1",
            "artifacts": artifact_hashes,
            "source_config": str(args.config.resolve()),
            "source_config_sha256": _sha256(args.config.resolve()),
        },
    )
    print(f"PASS_ENGINEERING_SMOKE_NOT_SCIENTIFIC_GATE report={report_path}", flush=True)


if __name__ == "__main__":
    main()
