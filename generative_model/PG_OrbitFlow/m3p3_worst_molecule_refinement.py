"""Run M3.3 worst-molecule torsion refinement and the original prediction Gate."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .graph_automorphism import (
    build_graph_automorphism_contract,
    optimal_automorphism_circular_assignment,
)
from .m3_tier32_training import _model_setting, _save
from .m3p1_automorphism_training import _prediction_record
from .m3p1_model import AutomorphismSetQuotientICPredictor
from .m3p2_torsion_refinement import _predict_cached
from .m2p1_phase_training import _prediction_checks
from .orbit_ic_model import build_orbit_ic_example
from .overfit import _panel_samples


SCHEMA_VERSION = "pg-orbitflow-m3p3-refinement-training-v1"


def _dump(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-m3p3-refinement-protocol-v1":
        raise ValueError("unsupported M3.3 protocol")
    if _sha256(Path(protocol["package_dir"]) / "manifest.json") != protocol[
        "canonical_manifest_sha256"
    ]:
        raise RuntimeError("canonical manifest changed")
    for item in protocol["parent"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3.3 parent identity changed")
    readiness = protocol["readiness_audit"]
    if _sha256(Path(readiness["path"])) != readiness["sha256"]:
        raise RuntimeError("M3.3 readiness identity changed")
    for item in protocol["implementation"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3.3 implementation changed")
    return protocol


def _loss_terms(model, cached, *, device):
    import torch

    molecule_losses = []
    nonplanar_losses = []
    for features, target, automorphism in cached:
        prediction = _predict_cached(model, features, automorphism, device=device)
        aligned = optimal_automorphism_circular_assignment(
            prediction, target, automorphism
        )
        errors = 1.0 - torch.sum(prediction * aligned, dim=-1).clamp(-1.0, 1.0)
        molecule_losses.append(errors.mean())
        angles = torch.abs(torch.rad2deg(torch.atan2(aligned[:, 0], aligned[:, 1])))
        nonplanar = torch.minimum(angles, torch.abs(180.0 - angles)) > 5.0
        if bool(nonplanar.any()):
            nonplanar_losses.append(errors[nonplanar].mean())
    stacked = torch.stack(molecule_losses)
    return stacked.mean(), stacked.max(), torch.stack(nonplanar_losses).max()


def _evaluate(model, examples, protocol, *, device):
    model.eval()
    records = []
    predictions = []
    import torch

    with torch.no_grad():
        for sample, _, graph, targets, automorphism in examples:
            prediction = {
                key: value.cpu().numpy()
                for key, value in model(graph, automorphism, device=device).items()
            }
            predictions.append(prediction)
            records.append(
                _prediction_record(sample, targets, prediction, automorphism)
            )
    values, checks, strata = _prediction_checks(
        records, protocol["prediction_gate"]
    )
    return records, predictions, values, checks, strata


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
    for sample in _panel_samples(protocol, tier):
        geometry = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, geometry)
        automorphism = build_graph_automorphism_contract(sample, geometry)
        examples.append((sample, geometry, graph, targets, automorphism))

    model = AutomorphismSetQuotientICPredictor(**_model_setting(protocol))
    parent = torch.load(
        protocol["parent"]["checkpoint"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(parent["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.parent.base.torsion_head, model.automorphism_head):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model = model.to(args.training_device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if parameter_count != int(protocol["model"]["parameter_count_expected"]):
        raise RuntimeError("M3.3 parameter count changed")
    if trainable_count != int(protocol["model"]["trainable_parameter_count_expected"]):
        raise RuntimeError("M3.3 trainable parameter count changed")

    cached = []
    model.eval()
    with torch.no_grad():
        for _, _, graph, targets, automorphism in examples:
            model(graph, automorphism, device=args.training_device)
            features = model.parent._captured_torsion_input.detach().clone()
            target = torch.as_tensor(
                targets.torsion_target_sincos,
                dtype=torch.float32,
                device=args.training_device,
            )
            cached.append((features, target, automorphism))

    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps = int(training["optimizer_steps"])
    losses = []
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mean_loss, worst_loss, worst_nonplanar = _loss_terms(
            model, cached, device=args.training_device
        )
        total = (
            float(training["mean_molecule_torsion_weight"]) * mean_loss
            + float(training["worst_molecule_torsion_weight"]) * worst_loss
            + float(training["worst_molecule_nonplanar_weight"]) * worst_nonplanar
        )
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            trainable, float(training["gradient_clip_norm"])
        )
        if not torch.isfinite(total) or not torch.isfinite(torch.as_tensor(gradient)):
            raise RuntimeError(f"non-finite M3.3 training at step {step}")
        optimizer.step()
        losses.append(
            (
                step,
                float(total.detach()),
                float(mean_loss.detach()),
                float(worst_loss.detach()),
                float(worst_nonplanar.detach()),
                float(gradient),
            )
        )
        if step == 1 or step % int(training["checkpoint_every"]) == 0:
            records, _, values, checks, _ = _evaluate(
                model, examples, protocol, device=args.training_device
            )
            print(
                f"step={step}/{steps} loss={float(total):.6g} "
                f"torsion={values['torsion_orbit_circular_mae_max_degrees']:.6g}deg "
                f"nonplanar={values['nonplanar_torsion_orbit_circular_mae_max_degrees']:.6g}deg "
                f"gate={'PASS' if all(checks.values()) else 'FAIL'}",
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
        steps,
        _sha256(protocol_path),
    )
    np.savez_compressed(
        output_dir / "losses.npz",
        step=np.asarray([row[0] for row in losses]),
        loss=np.asarray([row[1] for row in losses]),
        mean_molecule_loss=np.asarray([row[2] for row in losses]),
        worst_molecule_loss=np.asarray([row[3] for row in losses]),
        worst_nonplanar_loss=np.asarray([row[4] for row in losses]),
        gradient_norm=np.asarray([row[5] for row in losses]),
    )
    records, predictions, values, checks, strata = _evaluate(
        model, examples, protocol, device=args.training_device
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        **{
            f"{sample.package_index}_{key}": value
            for (sample, _, _, _, _), prediction in zip(
                examples, predictions, strict=True
            )
            for key, value in prediction.items()
        },
    )
    passed = all(checks.values())
    status = (
        "PASS_M3P3_PREDICTION_GATE_ADVANCE_TO_DECODER"
        if passed
        else "FAIL_M3P3_PREDICTION_GATE_STOP_BEFORE_DECODER"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "passed": passed,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "training": {
            "steps": steps,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
        },
        "prediction_records": records,
        "prediction_gate": {
            "passed": passed,
            "checks": checks,
            "values": values,
            "strata": strata,
            "thresholds": protocol["prediction_gate"],
        },
    }
    _dump(output_dir / "report.json", report)
    print(f"{status} report={output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
