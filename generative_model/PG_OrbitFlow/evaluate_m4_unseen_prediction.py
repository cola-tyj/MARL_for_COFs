"""Evaluate the frozen M3 predictor on an unseen IID-validation panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .global_shape_fingerprint_model import FingerprintGlobalShapePredictor, graph_wl_fingerprint
from .global_shape_model import build_global_shape_contract, global_shape_target
from .graph_automorphism import build_graph_automorphism_contract
from .m2p1_phase_training import _prediction_checks
from .m3_tier32_training import _model_setting
from .m3p1_automorphism_training import _prediction_record
from .orbit_ic_model import build_orbit_ic_example
from .unseen_iid import samples_for_split_records


def _load(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-m4-unseen-prediction-protocol-v1":
        raise ValueError("unsupported M4 protocol")
    if _sha256(Path(protocol["package_dir"]) / "manifest.json") != protocol["canonical_manifest_sha256"]:
        raise RuntimeError("canonical manifest changed")
    for group in ("predictor", "m3_parent"):
        for item in protocol[group].values():
            if _sha256(Path(item["path"])) != item["sha256"]:
                raise RuntimeError(f"M4 {group} identity changed")
    if _sha256(Path(protocol["panel_audit"]["path"])) != protocol["panel_audit"]["sha256"]:
        raise RuntimeError("M4 panel audit changed")
    for item in protocol["implementation"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M4 implementation changed")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    protocol_path = args.protocol.resolve()
    protocol = _load(protocol_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    samples = samples_for_split_records(protocol, split="val")
    setting = protocol["global_shape_contract"]
    examples = []
    for sample in samples:
        geometry = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, geometry)
        automorphism = build_graph_automorphism_contract(sample, geometry)
        shape = build_global_shape_contract(
            sample,
            minimum_graph_distance=int(setting["minimum_graph_distance"]),
            quantile_count=int(setting["quantile_count"]),
        )
        fingerprint = graph_wl_fingerprint(
            sample,
            bits=int(protocol["model"]["wl_fingerprint_bits"]),
            radius=int(protocol["model"]["wl_fingerprint_radius"]),
        )
        examples.append((sample, graph, targets, automorphism, shape, fingerprint))

    model = FingerprintGlobalShapePredictor(
        quantile_count=int(setting["quantile_count"]),
        fingerprint_bits=int(protocol["model"]["wl_fingerprint_bits"]),
        **_model_setting(protocol),
    )
    checkpoint = torch.load(
        protocol["predictor"]["checkpoint"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(args.device).eval()
    records = []
    shape_records = []
    saved = {}
    with torch.no_grad():
        for sample, graph, targets, automorphism, shape, fingerprint in examples:
            prediction = {
                key: value.cpu().numpy()
                for key, value in model(graph, automorphism, fingerprint, device=args.device).items()
            }
            records.append(_prediction_record(sample, targets, prediction, automorphism))
            target_shape = global_shape_target(sample, shape)
            shape_records.append(
                {
                    "package_index": int(sample.package_index),
                    "target_pg": sample.target_pg,
                    "global_shape_quantile_mae_angstrom": float(
                        np.mean(np.abs(prediction["global_shape_quantiles"] - target_shape))
                    ),
                }
            )
            for key, value in prediction.items():
                saved[f"{sample.package_index}_{key}"] = value
    np.savez_compressed(output_dir / "predictions.npz", **saved)

    values, checks, strata = _prediction_checks(records, protocol["prediction_gate"])
    shape_max = max(row["global_shape_quantile_mae_angstrom"] for row in shape_records)
    shape_strata = {
        point_group: max(
            row["global_shape_quantile_mae_angstrom"]
            for row in shape_records
            if row["target_pg"] == point_group
        )
        for point_group in ("C2", "C3")
    }
    shape_threshold = float(
        protocol["shape_prediction_gate"]["global_shape_quantile_mae_max_angstrom"]
    )
    shape_passed = shape_max <= shape_threshold and all(
        value <= shape_threshold for value in shape_strata.values()
    )
    passed = all(checks.values()) and shape_passed
    status = (
        "PASS_M4_UNSEEN_PREDICTION_ADVANCE_TO_DECODER"
        if passed
        else "FAIL_M4_UNSEEN_PREDICTION_PROCEED_TO_DATASET_TRAINING"
    )
    report = {
        "schema_version": "pg-orbitflow-m4-unseen-prediction-evaluation-v1",
        "status": status,
        "passed": passed,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "prediction_records": records,
        "prediction_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "values": values,
            "strata": strata,
            "thresholds": protocol["prediction_gate"],
        },
        "shape_records": shape_records,
        "shape_gate": {
            "passed": shape_passed,
            "maximum_mae_angstrom": shape_max,
            "strata": shape_strata,
            "threshold_angstrom": shape_threshold,
        },
        "scope": protocol["scope"],
        "decision": protocol["decision_rule"][
            "if_prediction_and_shape_pass" if passed else "if_either_fails"
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{status} prediction_passed={all(checks.values())} "
        f"shape_max={shape_max:.6g}A report={output_dir / 'report.json'}"
    )


if __name__ == "__main__":
    main()
