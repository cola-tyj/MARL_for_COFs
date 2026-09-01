"""Recompute M1 predictions and selected CPU decoder starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import build_geometry_contract
from .m0_oracle_reconstruction import (
    build_oracle_targets,
    optimize_one_start,
)
from .m1_bond_angle_training import _load_protocol, _predicted_decoder_targets
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
    args = parser.parse_args()
    import torch

    protocol = _load_protocol(args.protocol.resolve())
    state = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    setting = {
        key: protocol["model"][key]
        for key in ("node_feature_dim", "edge_feature_dim", "hidden_dim", "layers")
    }
    model = QuotientICPredictor(**setting)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    run_dir = args.run_dir.resolve()
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    archived_coordinates = np.load(run_dir / "coordinates.npz", allow_pickle=False)
    archived_predictions = np.load(run_dir / "predictions.npz", allow_pickle=False)
    offsets = archived_coordinates["atom_offsets"]
    source_records = {
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
        graph, _ = build_orbit_ic_example(sample, contract)
        with torch.no_grad():
            prediction = {
                key: value.detach().numpy()
                for key, value in model(graph, device="cpu").items()
            }
        prediction_exact = all(
            np.array_equal(
                value,
                archived_predictions[f"{sample.package_index}_{key}"],
            )
            for key, value in prediction.items()
        )
        oracle = build_oracle_targets(sample, contract)
        decoder_targets = _predicted_decoder_targets(prediction, contract, oracle)
        selected = int(source_records[sample.package_index]["selected_start_index"])
        _, repeated = optimize_one_start(
            sample,
            contract,
            decoder_targets,
            build_orbit_parameterization(sample),
            decoder_protocol,
            seed=int(protocol["seed"]) + sample.package_index * 1009 + selected,
            device="cpu",
        )
        saved = archived_coordinates["coordinates"][
            offsets[index] : offsets[index + 1]
        ]
        max_abs = float(np.max(np.abs(saved.astype(np.float64) - repeated)))
        aligned = kabsch_rmsd(saved, repeated)
        passed = prediction_exact and max_abs <= 5e-7 and aligned <= 1e-7
        records.append(
            {
                "package_index": sample.package_index,
                "prediction_arrays_exact": prediction_exact,
                "selected_start_index": selected,
                "coordinate_max_abs_difference_angstrom": max_abs,
                "coordinate_kabsch_rmsd_angstrom": aligned,
                "passed": passed,
            }
        )
    passed = all(row["passed"] for row in records)
    result = {
        "schema_version": "pg-orbitflow-m1-reproducibility-v1",
        "status": "PASS_M1_REPRODUCIBILITY" if passed else "FAIL_M1_REPRODUCIBILITY",
        "passed": passed,
        "prediction_array_comparison": "exact",
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
