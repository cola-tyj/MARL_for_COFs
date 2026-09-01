"""Train M3.6 global-shape quantiles from frozen IC features plus a WL fingerprint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .global_shape_fingerprint_model import (
    FingerprintGlobalShapePredictor,
    graph_wl_fingerprint,
)
from .global_shape_model import (
    build_global_shape_contract,
    global_shape_target,
)
from .graph_automorphism import build_graph_automorphism_contract
from .m2p1_phase_training import _prediction_checks
from .m3_tier32_training import _model_setting, _save
from .m3p1_automorphism_training import _prediction_record
from .orbit_ic_model import build_orbit_ic_example
from .overfit import _panel_samples


def _load(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-m3p6-fingerprint-shape-protocol-v1":
        raise ValueError("unsupported M3.6 protocol")
    for item in protocol["parent"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3.6 parent identity changed")
    if _sha256(Path(protocol["m3p5b_failure"]["path"])) != protocol[
        "m3p5b_failure"
    ]["sha256"]:
        raise RuntimeError("M3.5b failure identity changed")
    for item in protocol["implementation"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3.6 implementation changed")
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
    protocol = _load(protocol_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(int(protocol["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(protocol["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=False)
    shape_setting = protocol["global_shape_contract"]
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    examples = []
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
        fingerprint = graph_wl_fingerprint(
            sample,
            bits=int(protocol["model"]["wl_fingerprint_bits"]),
            radius=int(protocol["model"]["wl_fingerprint_radius"]),
        )
        examples.append(
            (sample, graph, targets, automorphism, shape, target, fingerprint)
        )
    model = FingerprintGlobalShapePredictor(
        quantile_count=int(shape_setting["quantile_count"]),
        fingerprint_bits=int(protocol["model"]["wl_fingerprint_bits"]),
        **_model_setting(protocol),
    )
    checkpoint = torch.load(
        protocol["parent"]["checkpoint"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    model.initialize_from_m3p3(checkpoint["model"])
    model.configure_shape_head_only()
    model = model.to(args.training_device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in model.parameters()) != int(
        protocol["model"]["parameter_count_expected"]
    ):
        raise RuntimeError("M3.6 parameter count changed")
    cached = []
    signatures = {}
    conflicts = []
    model.eval()
    with torch.no_grad():
        for sample, graph, _, automorphism, _, target, fingerprint in examples:
            model(graph, automorphism, fingerprint, device=args.training_device)
            shape_input = model._captured_shape_input.detach().clone()
            key = shape_input.cpu().numpy().tobytes()
            if key in signatures and not np.allclose(
                signatures[key][1], target, rtol=0.0, atol=1e-5
            ):
                conflicts.append([signatures[key][0], int(sample.package_index)])
            else:
                signatures[key] = (int(sample.package_index), target.copy())
            cached.append(
                (
                    shape_input,
                    torch.as_tensor(
                        target, dtype=torch.float32, device=args.training_device
                    ),
                )
            )
    if conflicts:
        raise RuntimeError(f"M3.6 exact input target conflicts: {conflicts}")
    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history = []
    for step in range(1, int(training["optimizer_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = torch.stack(
            [
                torch.mean(torch.square(model.predict_cached(x) - target))
                for x, target in cached
            ]
        )
        total = (
            float(training["mean_molecule_mse_weight"]) * losses.mean()
            + float(training["worst_molecule_mse_weight"]) * losses.max()
        )
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            trainable, float(training["gradient_clip_norm"])
        )
        if not torch.isfinite(total) or not torch.isfinite(torch.as_tensor(gradient)):
            raise RuntimeError(f"non-finite M3.6 training at step {step}")
        optimizer.step()
        history.append((step, float(total), float(losses.mean()), float(losses.max())))
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
        step=np.asarray([row[0] for row in history]),
        loss=np.asarray([row[1] for row in history]),
        mean_loss=np.asarray([row[2] for row in history]),
        worst_loss=np.asarray([row[3] for row in history]),
    )
    records = []
    parent_records = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for sample, graph, targets, automorphism, _, target, fingerprint in examples:
            prediction = {
                key: value.cpu().numpy()
                for key, value in model(
                    graph, automorphism, fingerprint, device=args.training_device
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
    passed = maximum <= threshold and all(parent_checks.values()) and not conflicts
    np.savez_compressed(
        output_dir / "predictions.npz",
        **{
            f"{sample.package_index}_{key}": value
            for (sample, *_), prediction in zip(examples, predictions, strict=True)
            for key, value in prediction.items()
        },
    )
    status = (
        "PASS_M3P6_FINGERPRINT_SHAPE_GATE_ADVANCE_TO_SELECTION_AUDIT"
        if passed
        else "FAIL_M3P6_FINGERPRINT_SHAPE_GATE_STOP_BRANCH"
    )
    report = {
        "schema_version": "pg-orbitflow-m3p6-fingerprint-shape-training-v1",
        "status": status,
        "passed": passed,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "shape_records": records,
        "shape_gate": {
            "passed": maximum <= threshold,
            "maximum_mae_angstrom": maximum,
            "strata": strata,
            "threshold_angstrom": threshold,
            "exact_input_target_conflicts": len(conflicts),
        },
        "parent_prediction_gate": {
            "passed": all(parent_checks.values()),
            "checks": parent_checks,
            "values": parent_values,
            "strata": parent_strata,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{status} shape_mae_max={maximum:.6g}A report={output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
