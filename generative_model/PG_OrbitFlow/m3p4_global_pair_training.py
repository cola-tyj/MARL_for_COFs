"""Train the frozen-parent M3.4 long-range pair-orbit distance head."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .global_pair_model import (
    GlobalPairAugmentedPredictor,
    build_global_pair_graph,
    build_global_pair_targets,
)
from .graph_automorphism import build_graph_automorphism_contract
from .m2p1_phase_training import _prediction_checks
from .m3_tier32_training import _model_setting, _save
from .m3p1_automorphism_training import _prediction_record
from .orbit_ic_model import build_orbit_ic_example
from .overfit import _panel_samples


SCHEMA_VERSION = "pg-orbitflow-m3p4-global-pair-training-v1"


def _dump(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-m3p4-global-pair-protocol-v1":
        raise ValueError("unsupported M3.4 protocol")
    if _sha256(Path(protocol["package_dir"]) / "manifest.json") != protocol[
        "canonical_manifest_sha256"
    ]:
        raise RuntimeError("canonical manifest changed")
    for item in protocol["parent"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3.4 parent identity changed")
    if _sha256(Path(protocol["failure_audit"]["path"])) != protocol[
        "failure_audit"
    ]["sha256"]:
        raise RuntimeError("M3.4 failure audit changed")
    for item in protocol["implementation"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3.4 implementation changed")
    return protocol


def _pair_values(records):
    values = {
        "global_pair_orbit_mae_max_angstrom": max(
            row["global_pair_orbit_mae_angstrom"] for row in records
        )
    }
    strata = {
        name: max(
            row["global_pair_orbit_mae_angstrom"]
            for row in records
            if row["target_pg"] == name
        )
        for name in ("C2", "C3")
    }
    return values, strata


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
        pair_graph = build_global_pair_graph(
            sample,
            minimum_graph_distance=int(
                protocol["global_pair_contract"]["minimum_graph_distance"]
            ),
        )
        pair_targets = build_global_pair_targets(sample, pair_graph)
        examples.append(
            (sample, graph, targets, automorphism, pair_graph, pair_targets)
        )

    model = GlobalPairAugmentedPredictor(**_model_setting(protocol))
    state = torch.load(
        protocol["parent"]["checkpoint"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    model.initialize_from_m3p3(state["model"])
    model.configure_pair_head_only()
    model = model.to(args.training_device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if parameter_count != int(protocol["model"]["parameter_count_expected"]):
        raise RuntimeError("M3.4 parameter count changed")
    if trainable_count != int(protocol["model"]["trainable_parameter_count_expected"]):
        raise RuntimeError("M3.4 trainable parameter count changed")

    cached = []
    model.eval()
    with torch.no_grad():
        for _, graph, _, automorphism, pair_graph, pair_targets in examples:
            model(graph, automorphism, pair_graph, device=args.training_device)
            cached.append(
                (
                    model._captured_global_pair_input.detach().clone(),
                    torch.as_tensor(
                        pair_targets,
                        dtype=torch.float32,
                        device=args.training_device,
                    ),
                )
            )
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
        molecule_losses = []
        for pair_input, target in cached:
            prediction = model.predict_cached(pair_input)
            molecule_losses.append(torch.mean(torch.square(prediction - target)))
        stacked = torch.stack(molecule_losses)
        mean_loss = stacked.mean()
        worst_loss = stacked.max()
        total = (
            float(training["mean_molecule_mse_weight"]) * mean_loss
            + float(training["worst_molecule_mse_weight"]) * worst_loss
        )
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            trainable, float(training["gradient_clip_norm"])
        )
        if not torch.isfinite(total) or not torch.isfinite(torch.as_tensor(gradient)):
            raise RuntimeError(f"non-finite M3.4 training at step {step}")
        optimizer.step()
        losses.append(
            (step, float(total.detach()), float(mean_loss), float(worst_loss), float(gradient))
        )
        if step == 1 or step % int(training["checkpoint_every"]) == 0:
            maes = [
                float(torch.mean(torch.abs(model.predict_cached(x) - y)).detach())
                for x, y in cached
            ]
            print(
                f"step={step}/{training['optimizer_steps']} loss={float(total):.6g} "
                f"pair_mae_max={max(maes):.6g}A",
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
    predictions = []
    parent_records = []
    with torch.no_grad():
        for sample, graph, targets, automorphism, pair_graph, pair_targets in examples:
            prediction = {
                key: value.cpu().numpy()
                for key, value in model(
                    graph, automorphism, pair_graph, device=args.training_device
                ).items()
            }
            predictions.append(prediction)
            records.append(
                {
                    "package_index": int(sample.package_index),
                    "target_pg": sample.target_pg,
                    "global_pair_orbit_count": len(pair_targets),
                    "global_pair_orbit_mae_angstrom": float(
                        np.mean(
                            np.abs(prediction["global_pair_lengths"] - pair_targets)
                        )
                    ),
                }
            )
            parent_records.append(
                _prediction_record(sample, targets, prediction, automorphism)
            )
    values, strata = _pair_values(records)
    threshold = protocol["pair_prediction_gate"][
        "global_pair_orbit_mae_max_angstrom"
    ]
    pair_passed = values["global_pair_orbit_mae_max_angstrom"] <= threshold
    parent_values, parent_checks, parent_strata = _prediction_checks(
        parent_records, protocol["prediction_gate"]
    )
    passed = pair_passed and all(parent_checks.values()) and all(
        value <= threshold for value in strata.values()
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
        "PASS_M3P4_GLOBAL_PAIR_GATE_ADVANCE_TO_DECODER"
        if passed
        else "FAIL_M3P4_GLOBAL_PAIR_GATE_STOP_BEFORE_DECODER"
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
        "global_pair_records": records,
        "global_pair_gate": {
            "passed": pair_passed,
            "values": values,
            "strata": strata,
            "threshold": threshold,
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
