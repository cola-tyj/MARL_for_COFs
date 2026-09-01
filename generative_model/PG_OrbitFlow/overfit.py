"""Run the frozen nested PG-OrbitFlow C2/C3 overfit gate."""

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

from .data import PGOrbitFlowDataset
from .flow import make_training_example, sample_raw_ode
from .metrics import raw_geometry_metrics
from .model import PGOrbitFlow
from .train import _loss_for_example


OVERFIT_SCHEMA_VERSION = "pg-orbitflow-overfit-training-v1"


def _flow_config_kwargs(protocol: dict) -> dict:
    prior = protocol.get("prior", {})
    transport = protocol.get("transport", {})
    return {
        "prior_type": str(prior.get("type", "projected_gaussian")),
        "harmonic_alpha": float(prior.get("harmonic_alpha", 1.0)),
        "transport_type": str(
            transport.get(
                "type",
                "cn_phase_aligned"
                if transport.get("target_phase_aligned_to_prior", False)
                else "independent",
            )
        ),
        "centralizer_phase_count": int(
            transport.get("centralizer_phase_count", 24)
        ),
    }


def _prior_config_kwargs(protocol: dict) -> dict:
    values = _flow_config_kwargs(protocol)
    return {
        "prior_type": values["prior_type"],
        "harmonic_alpha": values["harmonic_alpha"],
    }


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


def _load_protocol(path: Path, tier: str) -> tuple[dict, dict]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-overfit-protocol-v1":
        raise ValueError("unsupported overfit protocol schema")
    if tier not in protocol["tiers"] or tier not in {"4", "16", "32"}:
        raise ValueError("tier must be one of 4,16,32")
    package_dir = Path(protocol["package_dir"])
    if _sha256(package_dir / "manifest.json") != protocol["canonical_manifest_sha256"]:
        raise ValueError("canonical manifest changed after protocol freeze")
    prior = protocol.get("prior", {"type": "projected_gaussian"})
    transport = protocol.get("transport", {"type": "independent"})
    objective = protocol["objective"]
    if prior.get("type") not in {"projected_gaussian", "graph_harmonic"}:
        raise ValueError("unsupported frozen prior")
    if float(prior.get("harmonic_alpha", 1.0)) != 1.0:
        raise ValueError("H1-H3 requires harmonic_alpha=1.0")
    if transport.get("type") not in {
        "independent",
        "cn_phase_aligned",
        "centralizer_averaged",
    }:
        raise ValueError("unsupported frozen transport")
    if (
        transport.get("type") == "centralizer_averaged"
        and prior.get("type") != "graph_harmonic"
    ):
        raise ValueError("centralizer transport requires graph_harmonic prior")
    if int(transport.get("centralizer_phase_count", 24)) != 24:
        raise ValueError("H1-H3 requires 24 centralizer phases")
    endpoint_weight = float(objective.get("endpoint_weight", 0.0))
    if endpoint_weight not in {0.0, 1.0}:
        raise ValueError("endpoint_weight must be frozen to 0.0 or 1.0")
    if endpoint_weight > 0 and transport.get("type") != "centralizer_averaged":
        raise ValueError("endpoint auxiliary requires centralizer transport")
    return protocol, protocol["tiers"][tier]


def _panel_samples(protocol: dict, tier_config: dict):
    dataset = PGOrbitFlowDataset(
        protocol["package_dir"],
        split="train",
        split_scheme="iid",
        point_groups=("C2", "C3"),
    )
    package_to_position = {}
    for position, local_index in enumerate(dataset.local_indices):
        raw = dataset.canonical[int(local_index)]
        package_to_position[int(raw["package_index"])] = position
    requested = [int(record["package_index"]) for record in tier_config["records"]]
    if len(requested) != len(set(requested)) or any(index not in package_to_position for index in requested):
        raise ValueError("protocol panel contains duplicate or non-train package indices")
    samples = [dataset[package_to_position[index]] for index in requested]
    for expected, sample in zip(tier_config["records"], samples, strict=True):
        if (
            sample.molecule_id != expected["molecule_id"]
            or sample.target_pg != expected["target_pg"]
            or len(sample.atom_types) != int(expected["num_atoms"])
        ):
            raise ValueError("protocol panel identity changed")
    return samples


def _save_checkpoint(path: Path, *, model, optimizer, step: int, protocol_sha: str, tier: str):
    import torch

    torch.save(
        {
            "schema_version": OVERFIT_SCHEMA_VERSION,
            "protocol_sha256": protocol_sha,
            "tier": tier,
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def _fixed_time_evaluation(model, samples, protocol: dict, tier_config: dict, device: str):
    import torch

    coordinate_scale = float(protocol["data"]["coordinate_scale_angstrom"])
    objective_config = {
        "data": {"coordinate_scale_angstrom": coordinate_scale},
        "objective": protocol["objective"],
    }
    totals = Counter()
    by_pg: dict[str, Counter] = {"C2": Counter(), "C3": Counter()}
    by_time: dict[str, Counter] = {
        str(float(value)): Counter() for value in tier_config["evaluation"]["flow_times"]
    }
    counts = Counter()
    model.eval()
    with torch.no_grad():
        for sample in samples:
            for time_index, fixed_time in enumerate(tier_config["evaluation"]["flow_times"]):
                example = make_training_example(
                    sample,
                    seed=int(protocol["seed"]) + 70_000_000 + sample.package_index * 101 + time_index,
                    device=device,
                    coordinate_scale_angstrom=coordinate_scale,
                    time_value=float(fixed_time),
                    **_flow_config_kwargs(protocol),
                )
                losses = _loss_for_example(model, example, objective_config)
                endpoint_rmsd = float(
                    (
                        torch.sqrt(
                            torch.mean(
                                torch.sum(
                                    torch.square(
                                        losses["endpoint"]
                                        - example["geometry_target_positions"]
                                    ),
                                    dim=1,
                                )
                            )
                        )
                        * coordinate_scale
                    ).item()
                )
                values = {
                    "loss": float(losses["loss"].item()),
                    "flow_mse": float(losses["flow_mse"].item()),
                    "bond_length_mse": float(losses["bond_length_mse"].item()),
                    "overlap_loss": float(losses["overlap_loss"].item()),
                    "symmetry_mse": float(losses["symmetry_mse"].item()),
                    "endpoint_auxiliary_mse": float(
                        losses["endpoint_auxiliary_mse"].item()
                    ),
                    "endpoint_rmsd_angstrom": endpoint_rmsd,
                }
                counts[sample.target_pg] += 1
                for key, value in values.items():
                    totals[key] += value
                    by_pg[sample.target_pg][key] += value
                    by_time[str(float(fixed_time))][key] += value
    total_count = sum(counts.values())
    return {
        "case_count": total_count,
        "means": {key: float(value / total_count) for key, value in totals.items()},
        "by_point_group": {
            pg: {
                "case_count": int(counts[pg]),
                "means": {
                    key: float(value / counts[pg]) for key, value in by_pg[pg].items()
                },
            }
            for pg in ("C2", "C3")
        },
        "by_flow_time": {
            time: {
                "case_count": len(samples),
                "means": {
                    key: float(value / len(samples)) for key, value in values.items()
                },
            }
            for time, values in by_time.items()
        },
        "posthoc_hard_projection_used": False,
    }


def _raw_evaluation(model, samples, protocol: dict, tier_config: dict, device: str):
    records = []
    coordinate_parts = []
    atom_offsets = [0]
    coordinate_scale = float(protocol["data"]["coordinate_scale_angstrom"])
    for sample in samples:
        for seed_index in range(int(tier_config["evaluation"]["raw_seeds_per_molecule"])):
            seed = int(protocol["seed"]) + 80_000_000 + sample.package_index * 1009 + seed_index
            coordinates = sample_raw_ode(
                model,
                sample,
                seed=seed,
                steps=int(tier_config["evaluation"]["raw_ode_steps"]),
                device=device,
                coordinate_scale_angstrom=coordinate_scale,
                method=str(tier_config["evaluation"]["raw_ode_method"]),
                **_prior_config_kwargs(protocol),
            )
            metrics = raw_geometry_metrics(sample, coordinates)
            records.append(
                {
                    "package_index": sample.package_index,
                    "molecule_id": sample.molecule_id,
                    "target_pg": sample.target_pg,
                    "seed": seed,
                    "atom_count": len(sample.atom_types),
                    **metrics,
                }
            )
            coordinate_parts.append(coordinates.astype(np.float32))
            atom_offsets.append(atom_offsets[-1] + len(coordinates))
    means = {
        key: float(np.mean([record[key] for record in records]))
        for key in (
            "kabsch_rmsd_angstrom",
            "pair_distance_mae_angstrom",
            "bond_length_mae_angstrom",
            "mean_operation_rms_angstrom",
        )
    }
    summary = {
        "case_count": len(records),
        "means": means,
        "collision_free_fraction": float(np.mean([record["collision_free"] for record in records])),
        "max_operation_atom_error_angstrom": float(
            max(record["max_atom_error_angstrom"] for record in records)
        ),
        "by_point_group": {},
        "records": records,
        "posthoc_hard_projection_used": False,
    }
    for pg in ("C2", "C3"):
        subset = [record for record in records if record["target_pg"] == pg]
        summary["by_point_group"][pg] = {
            "case_count": len(subset),
            "kabsch_rmsd_mean_angstrom": float(
                np.mean([record["kabsch_rmsd_angstrom"] for record in subset])
            ),
            "bond_length_mae_mean_angstrom": float(
                np.mean([record["bond_length_mae_angstrom"] for record in subset])
            ),
            "collision_free_fraction": float(
                np.mean([record["collision_free"] for record in subset])
            ),
        }
    arrays = {
        "coordinates": np.concatenate(coordinate_parts, axis=0),
        "atom_offsets": np.asarray(atom_offsets, dtype=np.int64),
        "package_indices": np.asarray([record["package_index"] for record in records], dtype=np.int64),
        "seeds": np.asarray([record["seed"] for record in records], dtype=np.int64),
    }
    return summary, arrays


def _gate(tier_config: dict, loss_records: list[dict], fixed: dict, raw: dict):
    thresholds = tier_config["gate"]
    window = min(100, max(1, len(loss_records) // 4))
    first = float(np.mean([record["loss"] for record in loss_records[:window]]))
    last = float(np.mean([record["loss"] for record in loss_records[-window:]]))
    ratio = last / first if first > 0 else float("inf")
    values = {
        "training_loss_window_ratio": ratio,
        "fixed_time_endpoint_rmsd_mean_angstrom": fixed["means"]["endpoint_rmsd_angstrom"],
        "raw_kabsch_rmsd_mean_angstrom": raw["means"]["kabsch_rmsd_angstrom"],
        "raw_pair_distance_mae_mean_angstrom": raw["means"]["pair_distance_mae_angstrom"],
        "raw_bond_length_mae_mean_angstrom": raw["means"]["bond_length_mae_angstrom"],
        "raw_collision_free_fraction": raw["collision_free_fraction"],
        "raw_max_operation_atom_error_angstrom": raw["max_operation_atom_error_angstrom"],
    }
    checks = {
        "all_losses_gradients_finite": bool(
            all(np.isfinite(list(record.values())).all() for record in loss_records)
        ),
        "training_loss_window_ratio_max": ratio
        <= thresholds["training_loss_window_ratio_max"],
        "fixed_time_endpoint_rmsd_mean_max_angstrom": values[
            "fixed_time_endpoint_rmsd_mean_angstrom"
        ]
        <= thresholds["fixed_time_endpoint_rmsd_mean_max_angstrom"],
        "raw_kabsch_rmsd_mean_max_angstrom": values["raw_kabsch_rmsd_mean_angstrom"]
        <= thresholds["raw_kabsch_rmsd_mean_max_angstrom"],
        "raw_pair_distance_mae_mean_max_angstrom": values[
            "raw_pair_distance_mae_mean_angstrom"
        ]
        <= thresholds["raw_pair_distance_mae_mean_max_angstrom"],
        "raw_bond_length_mae_mean_max_angstrom": values[
            "raw_bond_length_mae_mean_angstrom"
        ]
        <= thresholds["raw_bond_length_mae_mean_max_angstrom"],
        "raw_collision_free_fraction_min": values["raw_collision_free_fraction"]
        >= thresholds["raw_collision_free_fraction_min"],
        "raw_max_operation_atom_error_max_angstrom": values[
            "raw_max_operation_atom_error_angstrom"
        ]
        <= thresholds["raw_max_operation_atom_error_max_angstrom"],
        "posthoc_hard_projection_never_used": True,
    }
    return {"passed": bool(all(checks.values())), "checks": checks, "values": values, "thresholds": thresholds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tier", required=True, choices=("4", "16", "32"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    import torch

    protocol_path = args.protocol.resolve()
    protocol, tier_config = _load_protocol(protocol_path, args.tier)
    if args.tier == "4" and args.init_checkpoint is not None:
        raise ValueError("tier 4 must start from random initialization")
    if args.tier != "4" and args.init_checkpoint is None:
        raise ValueError("tier 16/32 requires the preceding passed checkpoint")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_sha = _sha256(protocol_path)
    seed = int(protocol["seed"]) + int(args.tier)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)

    samples = _panel_samples(protocol, tier_config)
    model = PGOrbitFlow(
        **{
            key: protocol["model"][key]
            for key in ("hidden_dim", "layers", "radial_dim", "radial_max")
        }
    ).to(args.device)
    if sum(parameter.numel() for parameter in model.parameters()) != int(
        protocol["model"]["parameter_count_expected"]
    ):
        raise RuntimeError("model parameter count changed after protocol freeze")
    parent_identity = None
    if args.init_checkpoint is not None:
        parent = torch.load(args.init_checkpoint.resolve(), map_location=args.device, weights_only=False)
        if parent.get("schema_version") != OVERFIT_SCHEMA_VERSION:
            raise ValueError("parent checkpoint schema mismatch")
        expected_parent = "4" if args.tier == "16" else "16"
        if parent.get("tier") != expected_parent or parent.get("protocol_sha256") != protocol_sha:
            raise ValueError("parent checkpoint is not the passed preceding tier/protocol")
        model.load_state_dict(parent["model"], strict=True)
        parent_identity = {
            "path": str(args.init_checkpoint.resolve()),
            "sha256": _sha256(args.init_checkpoint.resolve()),
            "tier": expected_parent,
        }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tier_config["training"]["learning_rate"]),
        weight_decay=float(tier_config["training"]["weight_decay"]),
    )
    buckets = {
        pg: [sample for sample in samples if sample.target_pg == pg] for pg in ("C2", "C3")
    }
    if len(buckets["C2"]) != len(buckets["C3"]):
        raise ValueError("overfit panel is not PG-balanced")
    rng = np.random.default_rng(seed + 17)
    steps = int(tier_config["training"]["additional_optimizer_steps"])
    batch_size = int(tier_config["training"]["batch_size_molecules"])
    checkpoint_every = int(tier_config["training"]["checkpoint_every"])
    objective_config = {
        "data": {"coordinate_scale_angstrom": protocol["data"]["coordinate_scale_angstrom"]},
        "objective": protocol["objective"],
    }
    loss_records = []
    exposure = Counter()
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        metrics = Counter()
        for batch_item in range(batch_size):
            pg = ("C2", "C3")[(step * batch_size + batch_item) % 2]
            sample = buckets[pg][int(rng.integers(len(buckets[pg])))]
            exposure[pg] += 1
            example = make_training_example(
                sample,
                seed=seed + step * 1_000_003 + batch_item * 10_007 + sample.package_index,
                device=args.device,
                coordinate_scale_angstrom=float(protocol["data"]["coordinate_scale_angstrom"]),
                **_flow_config_kwargs(protocol),
            )
            losses = _loss_for_example(model, example, objective_config)
            (losses["loss"] / batch_size).backward()
            for key in (
                "loss",
                "flow_mse",
                "bond_length_mse",
                "overlap_loss",
                "symmetry_mse",
                "endpoint_auxiliary_mse",
            ):
                metrics[key] += float(losses[key].detach().item()) / batch_size
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(tier_config["training"]["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        record = {
            "step": step,
            **{key: float(value) for key, value in metrics.items()},
            "gradient_norm": float(gradient_norm),
        }
        loss_records.append(record)
        if step == 1 or step % checkpoint_every == 0 or step == steps:
            print(
                f"tier={args.tier} step={step}/{steps} loss={record['loss']:.6f} "
                f"flow={record['flow_mse']:.6f} bond={record['bond_length_mse']:.6f} "
                f"sym={record['symmetry_mse']:.3e}",
                flush=True,
            )
            _save_checkpoint(
                output_dir / f"step-{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                protocol_sha=protocol_sha,
                tier=args.tier,
            )
    last_path = output_dir / "last.pt"
    _save_checkpoint(
        last_path,
        model=model,
        optimizer=optimizer,
        step=steps,
        protocol_sha=protocol_sha,
        tier=args.tier,
    )
    fixed = _fixed_time_evaluation(model, samples, protocol, tier_config, args.device)
    raw, raw_arrays = _raw_evaluation(model, samples, protocol, tier_config, args.device)
    gate = _gate(tier_config, loss_records, fixed, raw)
    np.savez_compressed(
        output_dir / "losses.npz",
        **{
            key: np.asarray([record[key] for record in loss_records])
            for key in loss_records[0]
        },
    )
    np.savez_compressed(output_dir / "raw_coordinates.npz", **raw_arrays)
    status = (
        f"PASS_PG_ORBITFLOW_OVERFIT_TIER_{args.tier}_ADVANCE"
        if gate["passed"]
        else f"FAIL_PG_ORBITFLOW_OVERFIT_TIER_{args.tier}_STOP_AND_DIAGNOSE"
    )
    report = {
        "schema_version": OVERFIT_SCHEMA_VERSION,
        "status": status,
        "passed": gate["passed"],
        "claim_scope": "bounded train-panel memorization and raw sampling only",
        "quality_claim": False,
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "transport": protocol.get(
            "transport",
            {
                "type": "independent",
                "target_phase_aligned_to_prior": False,
                "alignment_group": "none",
            },
        ),
        "prior": protocol.get(
            "prior", {"type": "projected_gaussian", "harmonic_alpha": 1.0}
        ),
        "tier": int(args.tier),
        "parent_checkpoint": parent_identity,
        "data": {
            "molecule_count": len(samples),
            "package_indices": [sample.package_index for sample in samples],
            "point_group_counts": dict(Counter(sample.target_pg for sample in samples)),
            "molecule_exposures": int(sum(exposure.values())),
            "exposure_by_point_group": dict(exposure),
            "iid_validation_used": False,
            "iid_test_used": False,
            "core_ood_used": False,
        },
        "training": {
            "additional_optimizer_steps": steps,
            "batch_size_molecules": batch_size,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "device": args.device,
            "all_losses_gradients_finite": gate["checks"]["all_losses_gradients_finite"],
        },
        "fixed_time_evaluation": fixed,
        "raw_ode_evaluation": raw,
        "gate": gate,
        "isolation": {
            "posthoc_hard_projection_used": False,
            "f02_used": False,
            "external_pretraining_used": False,
            "historical_checkpoint_used": False,
            "our_et_flow_modified": False,
        },
    }
    _dump(output_dir / "report.json", report)
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _dump(
        output_dir / "manifest.json",
        {"schema_version": "pg-orbitflow-overfit-run-manifest-v1", "artifacts": artifacts},
    )
    print(f"{status} report={output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
