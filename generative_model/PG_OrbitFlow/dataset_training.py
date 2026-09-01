"""Mini-batch dataset-level training for the orbit-IC + shape model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .data import PGOrbitFlowDataset
from .factorized_automorphism import (
    build_factorized_automorphism_contract,
    factorized_circular_errors_degrees,
    optimal_factorized_circular_assignment,
)
from .factorized_cache import cache_path, load_contract, selection_fingerprint
from .geometry import build_geometry_contract
from .global_shape_fingerprint_model import FingerprintGlobalShapePredictor, graph_wl_fingerprint
from .global_shape_model import build_global_shape_contract, global_shape_target
from .m1_bond_angle_training import _angle_mae_degrees
from .m3_tier32_training import _model_setting
from .orbit_ic_model import build_orbit_ic_example


def _load(path: Path):
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-dataset-training-protocol-v1":
        raise ValueError("unsupported dataset training protocol")
    if _sha256(Path(protocol["package_dir"]) / "manifest.json") != protocol["canonical_manifest_sha256"]:
        raise RuntimeError("canonical manifest changed")
    for section in ("parent_evidence", "implementation"):
        for item in protocol[section].values():
            if _sha256(Path(item["path"])) != item["sha256"]:
                raise RuntimeError(f"dataset protocol {section} identity changed")
    return protocol


def _samples(protocol, *, split: str, records):
    dataset = PGOrbitFlowDataset(
        protocol["package_dir"],
        split=split,
        split_scheme="iid",
        point_groups=("C2", "C3"),
    )
    positions = {}
    for position, local_index in enumerate(dataset.local_indices):
        raw = dataset.canonical[int(local_index)]
        positions[int(raw["package_index"])] = position
    result = []
    for record in records:
        package_index = int(record["package_index"])
        if package_index not in positions:
            raise RuntimeError(f"record is not in frozen {split} split")
        sample = dataset[positions[package_index]]
        if (
            sample.molecule_id != record["molecule_id"]
            or sample.target_pg != record["target_pg"]
            or len(sample.atom_types) != int(record["num_atoms"])
        ):
            raise RuntimeError("dataset training record identity changed")
        result.append(sample)
    return result


def _balanced_schedule(samples, *, steps: int, per_pg: int, seed: int):
    rng = np.random.default_rng(seed)
    buckets = {
        point_group: [index for index, sample in enumerate(samples) if sample.target_pg == point_group]
        for point_group in ("C2", "C3")
    }
    if any(len(values) < per_pg for values in buckets.values()):
        raise ValueError("balanced schedule lacks one point group")
    orders = {key: rng.permutation(values).tolist() for key, values in buckets.items()}
    cursors = {"C2": 0, "C3": 0}
    schedule = []
    for _ in range(steps):
        batch = []
        for point_group in ("C2", "C3"):
            for _ in range(per_pg):
                if cursors[point_group] == len(orders[point_group]):
                    orders[point_group] = rng.permutation(buckets[point_group]).tolist()
                    cursors[point_group] = 0
                batch.append(orders[point_group][cursors[point_group]])
                cursors[point_group] += 1
        schedule.append(tuple(batch))
    return schedule


def _build_example(sample, protocol):
    geometry = build_geometry_contract(sample)
    graph, targets = build_orbit_ic_example(sample, geometry)
    cache = protocol.get("factorized_cache")
    if cache is None:
        factorized = build_factorized_automorphism_contract(sample, geometry)
    else:
        factorized = load_contract(
            cache_path(Path(cache["directory"]), sample.package_index)
        )
        expected_torsions = int(np.max(geometry.torsion_orbit_id)) + 1
        if factorized.allowed_torsion_permutations.shape[1] != expected_torsions:
            raise RuntimeError("factorized cache does not match geometry")
    shape_setting = protocol["global_shape_contract"]
    shape = build_global_shape_contract(
        sample,
        minimum_graph_distance=int(shape_setting["minimum_graph_distance"]),
        quantile_count=int(shape_setting["quantile_count"]),
    )
    shape_target = global_shape_target(sample, shape)
    fingerprint = graph_wl_fingerprint(
        sample,
        bits=int(protocol["model"]["wl_fingerprint_bits"]),
        radius=int(protocol["model"]["wl_fingerprint_radius"]),
    )
    return graph, targets, factorized, shape_target, fingerprint


def _losses(model, example, *, device: str):
    import torch

    graph, targets, factorized, shape_target, fingerprint = example
    prediction = model(graph, factorized, fingerprint, device=device)
    bond = torch.as_tensor(targets.bond_target_lengths, dtype=torch.float32, device=device)
    angle = torch.as_tensor(targets.angle_target_cosines, dtype=torch.float32, device=device)
    torsion = torch.as_tensor(targets.torsion_target_sincos, dtype=torch.float32, device=device)
    shape = torch.as_tensor(shape_target, dtype=torch.float32, device=device)
    aligned = optimal_factorized_circular_assignment(
        prediction["torsion_sincos"], torsion, factorized
    )
    torsion_errors = 1.0 - torch.sum(
        prediction["torsion_sincos"] * aligned, dim=-1
    ).clamp(-1.0, 1.0)
    angles = torch.abs(torch.rad2deg(torch.atan2(aligned[:, 0], aligned[:, 1])))
    nonplanar = torch.minimum(angles, torch.abs(180.0 - angles)) > 5.0
    nonplanar_loss = (
        torsion_errors[nonplanar].mean()
        if bool(nonplanar.any())
        else torsion_errors.sum() * 0.0
    )
    return {
        "bond": torch.mean(torch.square(prediction["bond_lengths"] - bond)),
        "angle": torch.mean(torch.square(prediction["angle_cosines"] - angle)),
        "torsion": torsion_errors.mean(),
        "nonplanar": nonplanar_loss,
        "shape": torch.mean(torch.square(prediction["global_shape_quantiles"] - shape)),
    }


def _prediction_record(model, sample, example, *, device: str):
    import torch

    graph, targets, factorized, shape_target, fingerprint = example
    with torch.no_grad():
        prediction = {
            key: value.cpu().numpy()
            for key, value in model(
                graph, factorized, fingerprint, device=device
            ).items()
        }
    errors = factorized_circular_errors_degrees(
        prediction["torsion_sincos"], targets.torsion_target_sincos, factorized
    )
    angles = np.abs(
        np.rad2deg(
            np.arctan2(
                targets.torsion_target_sincos[:, 0],
                targets.torsion_target_sincos[:, 1],
            )
        )
    )
    nonplanar = np.minimum(angles, np.abs(180.0 - angles)) > 5.0
    return {
        "package_index": int(sample.package_index),
        "target_pg": sample.target_pg,
        "bond_orbit_mae_angstrom": float(np.mean(np.abs(prediction["bond_lengths"] - targets.bond_target_lengths))),
        "angle_orbit_mae_degrees": _angle_mae_degrees(prediction["angle_cosines"], targets.angle_target_cosines),
        "torsion_orbit_circular_mae_degrees": float(np.mean(errors)),
        "nonplanar_torsion_orbit_circular_mae_degrees": float(np.mean(errors[nonplanar])) if np.any(nonplanar) else 0.0,
        "global_shape_quantile_mae_angstrom": float(np.mean(np.abs(prediction["global_shape_quantiles"] - shape_target))),
    }


def _save(path, model, optimizer, *, step: int, protocol_sha: str, history):
    import torch

    torch.save(
        {
            "schema_version": "pg-orbitflow-dataset-training-checkpoint-v2",
            "step": int(step),
            "protocol_sha256": protocol_sha,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": [list(row) for row in history],
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    protocol_path = args.protocol.resolve()
    protocol = _load(protocol_path)
    output_dir = args.output_dir.resolve()
    if args.resume_from is None:
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(protocol["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)

    train_samples = _samples(
        protocol,
        split="train",
        records=protocol["data"]["train_records"],
    )
    validation_samples = _samples(
        protocol,
        split="val",
        records=protocol["data"]["validation_records"],
    )
    if "factorized_cache" in protocol:
        cache = protocol["factorized_cache"]
        manifest_path = Path(cache["manifest"])
        if _sha256(manifest_path) != cache["manifest_sha256"]:
            raise RuntimeError("factorized cache manifest changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = protocol["data"]["train_records"] + protocol["data"]["validation_records"]
        if manifest.get("status") != "PASS_FACTORIZED_CACHE_COMPLETE_WITH_EXPLICIT_UNSUPPORTED":
            raise RuntimeError("factorized cache is incomplete")
        if manifest["cached_selection_fingerprint"] != selection_fingerprint(records):
            raise RuntimeError("factorized cache selection changed")
        for row in manifest["records"]:
            path = cache_path(Path(cache["directory"]), int(row["package_index"]))
            if _sha256(path) != row["sha256"]:
                raise RuntimeError("factorized cache artifact changed")
    if {sample.package_index for sample in train_samples} & {
        sample.package_index for sample in validation_samples
    }:
        raise RuntimeError("train/validation leakage")
    cache = {}

    def example(sample):
        if sample.package_index not in cache:
            cache[sample.package_index] = _build_example(sample, protocol)
        return cache[sample.package_index]

    model = FingerprintGlobalShapePredictor(
        quantile_count=int(protocol["global_shape_contract"]["quantile_count"]),
        fingerprint_bits=int(protocol["model"]["wl_fingerprint_bits"]),
        **_model_setting(protocol),
    )
    model.parent.configure_trainable_parameters()
    model = model.to(args.device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in model.parameters()) != int(protocol["model"]["parameter_count_expected"]):
        raise RuntimeError("dataset model parameter count changed")
    if sum(parameter.numel() for parameter in trainable) != int(protocol["model"]["trainable_parameter_count_expected"]):
        raise RuntimeError("dataset trainable parameter count changed")
    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    per_pg = int(training["point_group_composition_per_batch"]["C2"])
    schedule = _balanced_schedule(
        train_samples,
        steps=int(training["optimizer_steps"]),
        per_pg=per_pg,
        seed=seed + 17,
    )
    weights = training["loss_weights"]
    history = []
    start_step = 0
    if args.resume_from is not None:
        resume_path = args.resume_from.resolve()
        checkpoint = torch.load(resume_path, map_location=args.device, weights_only=False)
        if checkpoint.get("protocol_sha256") != _sha256(protocol_path):
            raise RuntimeError("resume checkpoint protocol mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        history = [tuple(row) for row in checkpoint.get("history", [])]
        if len(history) != start_step:
            raise RuntimeError("resume checkpoint lacks exact loss history")
    for step, batch in enumerate(schedule[start_step:], start=start_step + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rows = [_losses(model, example(train_samples[index]), device=args.device) for index in batch]
        means = {
            key: torch.stack([row[key] for row in rows]).mean()
            for key in rows[0]
        }
        worst_torsion = torch.stack([row["torsion"] for row in rows]).max()
        worst_nonplanar = torch.stack([row["nonplanar"] for row in rows]).max()
        total = (
            float(weights["bond_mse"]) * means["bond"]
            + float(weights["angle_cosine_mse"]) * means["angle"]
            + float(weights["torsion_circular"]) * means["torsion"]
            + float(weights["global_shape_mse"]) * means["shape"]
            + float(weights["worst_molecule_torsion"]) * worst_torsion
            + float(weights["worst_molecule_nonplanar"]) * worst_nonplanar
        )
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            trainable, float(training["gradient_clip_norm"])
        )
        if not torch.isfinite(total) or not torch.isfinite(torch.as_tensor(gradient)):
            raise RuntimeError(f"non-finite dataset training at step {step}")
        optimizer.step()
        history.append(
            (
                step,
                float(total.detach()),
                float(means["bond"].detach()),
                float(means["angle"].detach()),
                float(means["torsion"].detach()),
                float(means["shape"].detach()),
                float(gradient),
            )
        )
        if step == 1 or step % int(training["checkpoint_every"]) == 0:
            print(
                f"step={step}/{training['optimizer_steps']} loss={float(total):.6g} "
                f"bond={float(means['bond']):.6g} angle={float(means['angle']):.6g} "
                f"torsion={float(means['torsion']):.6g} shape={float(means['shape']):.6g}",
                flush=True,
            )
            _save(
                output_dir / f"step-{step:06d}.pt",
                model,
                optimizer,
                step=step,
                protocol_sha=_sha256(protocol_path),
                history=history,
            )
    _save(
        output_dir / "last.pt",
        model,
        optimizer,
        step=int(training["optimizer_steps"]),
        protocol_sha=_sha256(protocol_path),
        history=history,
    )
    np.savez_compressed(
        output_dir / "losses.npz",
        step=np.asarray([row[0] for row in history]),
        loss=np.asarray([row[1] for row in history]),
        bond=np.asarray([row[2] for row in history]),
        angle=np.asarray([row[3] for row in history]),
        torsion=np.asarray([row[4] for row in history]),
        shape=np.asarray([row[5] for row in history]),
        gradient_norm=np.asarray([row[6] for row in history]),
    )

    model.eval()
    validation_records = [
        _prediction_record(model, sample, example(sample), device=args.device)
        for sample in validation_samples
    ]
    reference_sample = validation_samples[0]
    reference = model(
        example(reference_sample)[0],
        example(reference_sample)[2],
        example(reference_sample)[4],
        device=args.device,
    )
    reloaded = FingerprintGlobalShapePredictor(
        quantile_count=int(protocol["global_shape_contract"]["quantile_count"]),
        fingerprint_bits=int(protocol["model"]["wl_fingerprint_bits"]),
        **_model_setting(protocol),
    ).to(args.device)
    checkpoint = torch.load(output_dir / "last.pt", map_location=args.device, weights_only=False)
    reloaded.load_state_dict(checkpoint["model"], strict=True)
    reloaded.eval()
    repeated = reloaded(
        example(reference_sample)[0],
        example(reference_sample)[2],
        example(reference_sample)[4],
        device=args.device,
    )
    reload_difference = max(
        float(torch.max(torch.abs(reference[key] - repeated[key])))
        for key in reference
    )
    first = np.mean([row[1] for row in history[:32]])
    last = np.mean([row[1] for row in history[-32:]])
    ratio = float(last / first)
    checks = {
        "all_losses_and_gradients_finite": bool(np.isfinite(np.asarray(history, dtype=float)).all()),
        "training_loss_last32_over_first32_max": ratio <= float(protocol["smoke_gate"]["training_loss_last32_over_first32_max"]),
        "checkpoint_reload_prediction_max_abs_difference": reload_difference == 0.0,
        "train_validation_overlap_count": not bool({sample.package_index for sample in train_samples} & {sample.package_index for sample in validation_samples}),
        "c2_c3_present_in_every_batch": all(
            sum(train_samples[index].target_pg == "C2" for index in batch) == per_pg
            and sum(train_samples[index].target_pg == "C3" for index in batch) == per_pg
            for batch in schedule
        ),
        "unsupported_records_explicit": len(protocol["data"]["train_unsupported"]) == protocol["data"]["train_total_c2_c3"] - protocol["data"]["train_supported_count"],
        "validation_used_for_gradient_updates": (
            False
            == bool(protocol["smoke_gate"]["validation_used_for_gradient_updates"])
        ),
    }
    passed = all(checks.values())
    status = (
        "PASS_DATASET_TRAINER_SMOKE_READY_FOR_FULL_RUN"
        if passed and protocol["mode"] == "smoke"
        else "PASS_DATASET_TRAINING_EXECUTION"
        if passed
        else "FAIL_DATASET_TRAINER_ENGINEERING_GATE"
    )
    report = {
        "schema_version": "pg-orbitflow-dataset-training-report-v1",
        "status": status,
        "passed": passed,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "training": {
            "steps": int(training["optimizer_steps"]),
            "train_molecule_count": len(train_samples),
            "validation_molecule_count": len(validation_samples),
            "first32_loss_mean": float(first),
            "last32_loss_mean": float(last),
            "loss_window_ratio": ratio,
            "checkpoint_reload_max_abs_difference": reload_difference,
            "cache_molecule_count": len(cache),
        },
        "checks": checks,
        "validation_records_descriptive_only": validation_records,
        "scope": protocol["scope"],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{status} loss_ratio={ratio:.6g} report={output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
