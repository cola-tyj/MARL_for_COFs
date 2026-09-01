"""Train M3.5 global long-range distance quantiles with a frozen M3.3 parent."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .global_shape_model import (
    GlobalShapeAugmentedPredictor,
    build_global_shape_contract,
    global_shape_target,
)
from .graph_automorphism import build_graph_automorphism_contract
from .m2p1_phase_training import _prediction_checks
from .m3_tier32_training import _model_setting, _save
from .m3p1_automorphism_training import _prediction_record
from .orbit_ic_model import build_orbit_ic_example
from .overfit import _panel_samples


SCHEMA_VERSION = "pg-orbitflow-m3p5-global-shape-training-v1"


def _dump(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-m3p5-global-shape-protocol-v1":
        raise ValueError("unsupported M3.5 protocol")
    if _sha256(Path(protocol["package_dir"]) / "manifest.json") != protocol[
        "canonical_manifest_sha256"
    ]:
        raise RuntimeError("canonical manifest changed")
    for item in protocol["parent"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3.5 parent identity changed")
    if _sha256(Path(protocol["m3p4_failure"]["report"]["path"])) != protocol[
        "m3p4_failure"
    ]["report"]["sha256"]:
        raise RuntimeError("M3.4 failure identity changed")
    if _sha256(Path(protocol["decoder_failure_audit"]["path"])) != protocol[
        "decoder_failure_audit"
    ]["sha256"]:
        raise RuntimeError("decoder failure audit changed")
    for item in protocol["implementation"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3.5 implementation changed")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    import torch

    if args.training_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    seed = int(protocol["seed"])
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)

    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    examples = []
    shape_setting = protocol["global_shape_contract"]
    for sample in _panel_samples(protocol, tier):
        geometry = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, geometry)
        automorphism = build_graph_automorphism_contract(sample, geometry)
        shape = build_global_shape_contract(
            sample,
            minimum_graph_distance=int(shape_setting["minimum_graph_distance"]),
            quantile_count=int(shape_setting["quantile_count"]),
        )
        target = global_shape_target(sample, shape)
        examples.append((sample, graph, targets, automorphism, shape, target))

    model = GlobalShapeAugmentedPredictor(
        quantile_count=int(shape_setting["quantile_count"]),
        **_model_setting(protocol),
    )
    state = torch.load(
        protocol["parent"]["checkpoint"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    model.initialize_from_m3p3(state["model"])
    model.configure_shape_head_only()
    model = model.to(args.training_device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if parameter_count != int(protocol["model"]["parameter_count_expected"]):
        raise RuntimeError("M3.5 parameter count changed")
    if trainable_count != int(protocol["model"]["trainable_parameter_count_expected"]):
        raise RuntimeError("M3.5 trainable parameter count changed")

    cached = []
    signatures = {}
    conflicts = []
    model.eval()
    with torch.no_grad():
        for sample, graph, _, automorphism, _, target in examples:
            model(graph, automorphism, device=args.training_device)
            embedding = model._captured_graph_embedding.detach().clone()
            tensor_target = torch.as_tensor(
                target, dtype=torch.float32, device=args.training_device
            )
            key = embedding.detach().cpu().numpy().tobytes()
            if key in signatures and not np.allclose(
                signatures[key][1], target, rtol=0.0, atol=1e-5
            ):
                conflicts.append(
                    [signatures[key][0], int(sample.package_index)]
                )
            else:
                signatures[key] = (int(sample.package_index), target.copy())
            cached.append((embedding, tensor_target))
    if conflicts:
        raise RuntimeError(f"global-shape embedding target conflicts: {conflicts}")

    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    losses = []
    for step in range(1, int(training["optimizer_steps"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        molecule_losses = torch.stack(
            [
                torch.mean(torch.square(model.predict_cached(x) - target))
                for x, target in cached
            ]
        )
        mean_loss = molecule_losses.mean()
        worst_loss = molecule_losses.max()
        total = (
            float(training["mean_molecule_mse_weight"]) * mean_loss
            + float(training["worst_molecule_mse_weight"]) * worst_loss
        )
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            trainable, float(training["gradient_clip_norm"])
        )
        if not torch.isfinite(total) or not torch.isfinite(torch.as_tensor(gradient)):
            raise RuntimeError(f"non-finite M3.5 training at step {step}")
        optimizer.step()
        losses.append(
            (step, float(total), float(mean_loss), float(worst_loss), float(gradient))
        )
        if step == 1 or step % int(training["checkpoint_every"]) == 0:
            maes = [
                float(torch.mean(torch.abs(model.predict_cached(x) - target)).detach())
                for x, target in cached
            ]
            print(
                f"step={step}/{training['optimizer_steps']} loss={float(total):.6g} "
                f"shape_mae_max={max(maes):.6g}A",
                flush=True,
            )
            _save(
                output_dir / f"step-{step:06d}.pt",
                model,
                optimizer,
                step,
                _sha256(protocol_path),
            )
    _save(
        output_dir / "last.pt",
        model,
        optimizer,
        int(training["optimizer_steps"]),
        _sha256(protocol_path),
    )
    np.savez_compressed(
        output_dir / "losses.npz",
        step=np.asarray([row[0] for row in losses]),
        loss=np.asarray([row[1] for row in losses]),
        mean_loss=np.asarray([row[2] for row in losses]),
        worst_loss=np.asarray([row[3] for row in losses]),
        gradient_norm=np.asarray([row[4] for row in losses]),
    )

    model.eval()
    records = []
    parent_records = []
    predictions = []
    with torch.no_grad():
        for sample, graph, targets, automorphism, _, target in examples:
            prediction = {
                key: value.cpu().numpy()
                for key, value in model(
                    graph, automorphism, device=args.training_device
                ).items()
            }
            predictions.append(prediction)
            records.append(
                {
                    "package_index": int(sample.package_index),
                    "target_pg": sample.target_pg,
                    "global_shape_quantile_mae_angstrom": float(
                        np.mean(
                            np.abs(prediction["global_shape_quantiles"] - target)
                        )
                    ),
                }
            )
            parent_records.append(
                _prediction_record(sample, targets, prediction, automorphism)
            )
    threshold = protocol["shape_prediction_gate"][
        "global_shape_quantile_mae_max_angstrom"
    ]
    maximum = max(row["global_shape_quantile_mae_angstrom"] for row in records)
    strata = {
        name: max(
            row["global_shape_quantile_mae_angstrom"]
            for row in records
            if row["target_pg"] == name
        )
        for name in ("C2", "C3")
    }
    parent_values, parent_checks, parent_strata = _prediction_checks(
        parent_records, protocol["prediction_gate"]
    )
    passed = (
        maximum <= threshold
        and all(value <= threshold for value in strata.values())
        and all(parent_checks.values())
        and not conflicts
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        **{
            f"{sample.package_index}_{key}": value
            for (sample, *_), prediction in zip(examples, predictions, strict=True)
            for key, value in prediction.items()
        },
    )
    status = (
        "PASS_M3P5_GLOBAL_SHAPE_GATE_ADVANCE_TO_SELECTION_AUDIT"
        if passed
        else "FAIL_M3P5_GLOBAL_SHAPE_GATE_STOP_BRANCH"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "passed": passed,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "training": {
            "steps": int(training["optimizer_steps"]),
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
        },
        "shape_records": records,
        "shape_gate": {
            "passed": maximum <= threshold,
            "maximum_mae_angstrom": maximum,
            "strata": strata,
            "threshold_angstrom": threshold,
            "exact_embedding_target_conflicts": len(conflicts),
        },
        "parent_prediction_gate": {
            "passed": all(parent_checks.values()),
            "checks": parent_checks,
            "values": parent_values,
            "strata": parent_strata,
        },
    }
    _dump(output_dir / "report.json", report)
    print(f"{status} report={output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
