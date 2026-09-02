"""Independently audit a frozen dataset-training checkpoint on IID validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .dataset_training import _build_example, _load, _prediction_record, _samples
from .global_shape_fingerprint_model import FingerprintGlobalShapePredictor
from .m2p1_phase_training import _prediction_checks
from .m3_tier32_training import _model_setting


METRIC_KEYS = (
    "bond_orbit_mae_angstrom",
    "angle_orbit_mae_degrees",
    "torsion_orbit_circular_mae_degrees",
    "nonplanar_torsion_orbit_circular_mae_degrees",
    "global_shape_quantile_mae_angstrom",
)


def _distribution(records: list[dict]) -> dict:
    result = {}
    for key in METRIC_KEYS:
        values = np.asarray([row[key] for row in records], dtype=np.float64)
        result[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
        }
    return result


def _record_reproduction_difference(reference: list[dict], repeated: list[dict]) -> float:
    expected = {int(row["package_index"]): row for row in reference}
    observed = {int(row["package_index"]): row for row in repeated}
    if expected.keys() != observed.keys():
        raise RuntimeError("validation record identity changed")
    return max(
        abs(float(expected[index][key]) - float(observed[index][key]))
        for index in expected
        for key in METRIC_KEYS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--gate-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    training_protocol_path = args.training_protocol.resolve()
    training_run = args.training_run.resolve()
    gate_protocol_path = args.gate_protocol.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    protocol = _load(training_protocol_path)
    if protocol.get("mode") != "full":
        raise RuntimeError("final audit requires the full dataset protocol")
    gate_protocol = json.loads(gate_protocol_path.read_text(encoding="utf-8"))
    if gate_protocol.get("schema_version") != "pg-orbitflow-m3p6-shape-decoder-protocol-v1":
        raise RuntimeError("unsupported frozen Gate protocol")
    if protocol["parent_evidence"]["m3_protocol"]["sha256"] != _sha256(gate_protocol_path):
        raise RuntimeError("dataset training did not bind the supplied M3 Gate protocol")

    m4_report_reference = protocol["parent_evidence"]["m4_unseen_failure"]
    m4_report_path = Path(m4_report_reference["path"])
    m4_report = json.loads(m4_report_path.read_text(encoding="utf-8"))
    if _sha256(m4_report_path) != m4_report_reference["sha256"]:
        raise RuntimeError("bound M4 failure report identity changed")
    m4_protocol_reference = m4_report["protocol"]
    m4_protocol_path = Path(m4_protocol_reference["path"])
    if _sha256(m4_protocol_path) != m4_protocol_reference["sha256"]:
        raise RuntimeError("bound M4 Gate protocol identity changed")
    m4_protocol = json.loads(m4_protocol_path.read_text(encoding="utf-8"))
    if m4_protocol.get("schema_version") != "pg-orbitflow-m4-unseen-prediction-protocol-v1":
        raise RuntimeError("unsupported bound M4 Gate protocol")
    if m4_protocol["m3_parent"]["protocol"]["sha256"] != _sha256(gate_protocol_path):
        raise RuntimeError("M3 and M4 frozen Gate protocols disagree")

    training_report_path = training_run / "report.json"
    checkpoint_path = training_run / "last.pt"
    losses_path = training_run / "losses.npz"
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if training_report.get("status") != "PASS_DATASET_TRAINING_EXECUTION" or not training_report.get("passed"):
        raise RuntimeError("dataset training execution did not pass")
    if training_report["protocol"]["sha256"] != _sha256(training_protocol_path):
        raise RuntimeError("training report protocol identity changed")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_steps = int(protocol["training"]["optimizer_steps"])
    if checkpoint.get("schema_version") != "pg-orbitflow-dataset-training-checkpoint-v2":
        raise RuntimeError("unsupported final checkpoint schema")
    if int(checkpoint.get("step", -1)) != expected_steps:
        raise RuntimeError("final checkpoint did not reach the frozen step count")
    if checkpoint.get("protocol_sha256") != _sha256(training_protocol_path):
        raise RuntimeError("final checkpoint protocol identity changed")
    if len(checkpoint.get("history", [])) != expected_steps:
        raise RuntimeError("final checkpoint loss history is incomplete")

    samples = _samples(
        protocol,
        split="val",
        records=protocol["data"]["validation_records"],
    )
    examples = [_build_example(sample, protocol) for sample in samples]
    model = FingerprintGlobalShapePredictor(
        quantile_count=int(protocol["global_shape_contract"]["quantile_count"]),
        fingerprint_bits=int(protocol["model"]["wl_fingerprint_bits"]),
        **_model_setting(protocol),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(args.device).eval()
    records = [
        _prediction_record(model, sample, example, device=args.device)
        for sample, example in zip(samples, examples, strict=True)
    ]
    reproduction_difference = _record_reproduction_difference(
        training_report["validation_records_descriptive_only"], records
    )

    ic_records = [
        {key: value for key, value in row.items() if key != "global_shape_quantile_mae_angstrom"}
        for row in records
    ]
    prediction_values, prediction_checks, prediction_strata = _prediction_checks(
        ic_records, gate_protocol["prediction_gate"]
    )
    shape_threshold = float(
        m4_protocol["shape_prediction_gate"]["global_shape_quantile_mae_max_angstrom"]
    )
    shape_values = {
        point_group: max(
            row["global_shape_quantile_mae_angstrom"]
            for row in records
            if row["target_pg"] == point_group
        )
        for point_group in ("C2", "C3")
    }
    shape_maximum = max(shape_values.values())
    shape_checks = {
        "global_shape_quantile_mae_max_angstrom": shape_maximum <= shape_threshold,
        "c2_and_c3_each_pass": all(value <= shape_threshold for value in shape_values.values()),
    }
    structural_checks = {
        "training_execution_passed": True,
        "checkpoint_step_exact": int(checkpoint["step"]) == expected_steps,
        "checkpoint_history_complete": len(checkpoint["history"]) == expected_steps,
        "validation_count_exact": len(records) == len(protocol["data"]["validation_records"]),
        "validation_records_reproduced": reproduction_difference <= 1e-7,
        "iid_test_not_used": True,
        "core_ood_not_used": True,
        "hard_projection_not_used": True,
        "f0p2_not_used": True,
    }
    quality_passed = all(prediction_checks.values()) and all(shape_checks.values())
    passed = all(structural_checks.values()) and quality_passed
    status = (
        "PASS_DATASET_VALIDATION_ADVANCE_TO_DECODER_GATE"
        if passed
        else "FAIL_DATASET_VALIDATION_STOP_BEFORE_DECODER"
    )
    report = {
        "schema_version": "pg-orbitflow-dataset-validation-audit-v1",
        "status": status,
        "passed": passed,
        "execution_passed": all(structural_checks.values()),
        "quality_passed": quality_passed,
        "structural_checks": structural_checks,
        "prediction_gate": {
            "passed": all(prediction_checks.values()),
            "checks": prediction_checks,
            "values": prediction_values,
            "strata": prediction_strata,
            "thresholds": gate_protocol["prediction_gate"],
        },
        "shape_gate": {
            "passed": all(shape_checks.values()),
            "checks": shape_checks,
            "maximum_mae_angstrom": shape_maximum,
            "strata": shape_values,
            "threshold_angstrom": shape_threshold,
        },
        "natural_validation_distribution": _distribution(records),
        "records": records,
        "reproduction_max_abs_difference": reproduction_difference,
        "identity": {
            "training_protocol": {
                "path": str(training_protocol_path),
                "sha256": _sha256(training_protocol_path),
            },
            "frozen_gate_protocol": {
                "path": str(gate_protocol_path),
                "sha256": _sha256(gate_protocol_path),
            },
            "frozen_shape_gate_protocol": {
                "path": str(m4_protocol_path),
                "sha256": _sha256(m4_protocol_path),
            },
            "training_report": {
                "path": str(training_report_path),
                "sha256": _sha256(training_report_path),
            },
            "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
            "losses": {"path": str(losses_path), "sha256": _sha256(losses_path)},
            "auditor_sha256": _sha256(Path(__file__).resolve()),
        },
        "scope": {
            "training_performed": False,
            "checkpoint_frozen": True,
            "iid_train_used": False,
            "iid_validation_used_read_only": True,
            "iid_test_used": False,
            "core_ood_used": False,
            "target_coordinates_used_as_model_input": False,
            "target_coordinates_used_for_metrics_only": True,
            "hard_projection_used": False,
            "f0p2_used": False,
        },
        "decision": (
            "advance to frozen decoder Gate"
            if passed
            else "stop before decoder; do not inspect IID-test/Core-OOD and diagnose train-to-validation generalization"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{status} execution={report['execution_passed']} quality={quality_passed} "
        f"reproduction_max={reproduction_difference:.3g} report={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
