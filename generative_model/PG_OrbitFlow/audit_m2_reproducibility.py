"""Recompute M2 predictions and the four selected deterministic decoder starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import build_geometry_contract
from .m0_oracle_reconstruction import optimize_one_start
from .m1_bond_angle_training import _angle_mae_degrees
from .m2_torsion_training import (
    _learned_decoder_targets,
    _load_protocol,
    _torsion_mae_degrees,
)
from .metrics import kabsch_rmsd
from .orbit_ic_model import QuotientICPredictor, build_orbit_ic_example
from .orbit_kinematics import build_orbit_parameterization
from .overfit import _panel_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prediction-device", choices=("cpu", "cuda"), default="cuda"
    )
    args = parser.parse_args()
    import torch

    if args.prediction_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA prediction reproduction requested but unavailable")

    protocol = _load_protocol(args.protocol.resolve())
    state = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    setting = {
        key: protocol["model"][key]
        for key in (
            "node_feature_dim",
            "edge_feature_dim",
            "hidden_dim",
            "layers",
            "predict_torsion",
        )
    }
    model = QuotientICPredictor(**setting)
    model.load_state_dict(state["model"], strict=True)
    model = model.to(args.prediction_device).eval()
    run_dir = args.run_dir.resolve()
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    archived = np.load(run_dir / "coordinates.npz", allow_pickle=False)
    offsets = archived["atom_offsets"]
    source_prediction = {
        int(row["package_index"]): row for row in report["prediction_records"]
    }
    source_reconstruction = {
        int(row["package_index"]): row for row in report["reconstruction_records"]
    }
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    samples = _panel_samples(protocol, tier)
    decoder_protocol = {
        "initialization": protocol["decoder"]["initialization"],
        "optimization": protocol["decoder"]["optimization"],
        "energy_weights": protocol["decoder"]["energy_weights"],
    }
    records = []
    for index, sample in enumerate(samples):
        contract = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, contract)
        with torch.no_grad():
            prediction = {
                key: value.detach().cpu().numpy()
                for key, value in model(graph, device=args.prediction_device).items()
            }
        metrics = {
            "bond_orbit_mae_angstrom": float(
                np.mean(np.abs(prediction["bond_lengths"] - targets.bond_target_lengths))
            ),
            "angle_orbit_mae_degrees": _angle_mae_degrees(
                prediction["angle_cosines"], targets.angle_target_cosines
            ),
            "torsion_orbit_circular_mae_degrees": _torsion_mae_degrees(
                prediction["torsion_sincos"], targets.torsion_target_sincos
            ),
        }
        prediction_difference = max(
            abs(metrics[key] - float(source_prediction[sample.package_index][key]))
            for key in metrics
        )
        selected = int(
            source_reconstruction[sample.package_index]["selected_start_index"]
        )
        _, repeated = optimize_one_start(
            sample,
            contract,
            _learned_decoder_targets(prediction, contract),
            build_orbit_parameterization(sample),
            decoder_protocol,
            seed=int(protocol["seed"]) + sample.package_index * 1009 + selected,
            device="cpu",
        )
        saved = archived["coordinates"][offsets[index] : offsets[index + 1]]
        max_abs = float(np.max(np.abs(saved.astype(np.float64) - repeated)))
        aligned = kabsch_rmsd(saved, repeated)
        passed = (
            prediction_difference <= 5e-5
            and max_abs <= 5e-7
            and aligned <= 1e-7
        )
        records.append(
            {
                "package_index": sample.package_index,
                "prediction_metric_max_abs_difference": prediction_difference,
                "selected_start_index": selected,
                "coordinate_max_abs_difference_angstrom": max_abs,
                "coordinate_kabsch_rmsd_angstrom": aligned,
                "passed": passed,
            }
        )
    passed = all(row["passed"] for row in records)
    result = {
        "schema_version": "pg-orbitflow-m2-reproducibility-v1",
        "status": "PASS_M2_REPRODUCIBILITY" if passed else "FAIL_M2_REPRODUCIBILITY",
        "passed": passed,
        "prediction_metric_max_abs_tolerance": 5e-5,
        "prediction_device": args.prediction_device,
        "coordinate_max_abs_tolerance_angstrom": 5e-7,
        "coordinate_kabsch_tolerance_angstrom": 1e-7,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit(result["status"])
    print(f"{result['status']} report={args.output.resolve()}")


if __name__ == "__main__":
    main()
