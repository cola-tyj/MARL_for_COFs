"""Refine the M3.5 shape head with deterministic full-batch L-BFGS."""

from __future__ import annotations

import argparse
import json
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


def _load(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-m3p5b-shape-lbfgs-protocol-v1":
        raise ValueError("unsupported M3.5b protocol")
    for item in protocol["parent_shape_run"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3.5b parent identity changed")
    for item in protocol["implementation"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3.5b implementation changed")
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
        examples.append((sample, graph, targets, automorphism, shape, target))
    model = GlobalShapeAugmentedPredictor(
        quantile_count=int(shape_setting["quantile_count"]),
        **_model_setting(protocol),
    )
    checkpoint = torch.load(
        protocol["parent_shape_run"]["checkpoint"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.configure_shape_head_only()
    model = model.to(args.training_device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    cached = []
    model.eval()
    with torch.no_grad():
        for _, graph, _, automorphism, _, target in examples:
            model(graph, automorphism, device=args.training_device)
            cached.append(
                (
                    model._captured_graph_embedding.detach().clone(),
                    torch.as_tensor(
                        target, dtype=torch.float32, device=args.training_device
                    ),
                )
            )
    training = protocol["training"]
    optimizer = torch.optim.LBFGS(
        trainable,
        lr=float(training["learning_rate"]),
        max_iter=int(training["maximum_iterations"]),
        history_size=int(training["history_size"]),
        line_search_fn=training["line_search"],
        tolerance_grad=float(training["tolerance_grad"]),
        tolerance_change=float(training["tolerance_change"]),
    )
    closure_history = []

    def closure():
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
        if not torch.isfinite(total):
            raise RuntimeError("non-finite M3.5b loss")
        total.backward()
        closure_history.append(float(total.detach()))
        return total

    initial = float(closure().detach())
    optimizer.zero_grad(set_to_none=True)
    optimizer.step(closure)
    final = float(closure().detach())
    optimizer.zero_grad(set_to_none=True)
    _save(
        output_dir / "last.pt",
        model,
        optimizer,
        int(training["maximum_iterations"]),
        _sha256(protocol_path),
    )
    np.savez_compressed(
        output_dir / "losses.npz",
        closure=np.arange(len(closure_history), dtype=np.int64),
        loss=np.asarray(closure_history),
    )

    records = []
    parent_records = []
    predictions = []
    model.eval()
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
    passed = maximum <= threshold and all(parent_checks.values())
    np.savez_compressed(
        output_dir / "predictions.npz",
        **{
            f"{sample.package_index}_{key}": value
            for (sample, *_), prediction in zip(examples, predictions, strict=True)
            for key, value in prediction.items()
        },
    )
    status = (
        "PASS_M3P5B_GLOBAL_SHAPE_GATE_ADVANCE_TO_SELECTION_AUDIT"
        if passed
        else "FAIL_M3P5B_GLOBAL_SHAPE_GATE_STOP_BRANCH"
    )
    report = {
        "schema_version": "pg-orbitflow-m3p5b-shape-lbfgs-training-v1",
        "status": status,
        "passed": passed,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "optimization": {
            "initial_loss": initial,
            "final_loss": final,
            "closure_calls": len(closure_history),
        },
        "shape_records": records,
        "shape_gate": {
            "passed": maximum <= threshold,
            "maximum_mae_angstrom": maximum,
            "strata": strata,
            "threshold_angstrom": threshold,
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
    print(
        f"{status} initial={initial:.6g} final={final:.6g} "
        f"shape_mae_max={maximum:.6g}A report={output_dir / 'report.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
