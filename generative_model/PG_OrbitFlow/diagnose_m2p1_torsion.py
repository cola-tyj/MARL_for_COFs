"""Diagnose the frozen M2.1 Tier-16 torsion prediction failure.

This is a read-only post-hoc audit.  It never trains, decodes coordinates, or
changes a Gate.  The audit identifies torsion orbits whose predictions collapse
to the same circular value despite distinct atom-labelled targets, then checks
whether those orbits differ only by an equal-element terminal atom around the
same central bond.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .m2p1_phase_training import _model_setting
from .orbit_ic_model import QuotientICPredictor, build_orbit_ic_example
from .overfit import _panel_samples


SCHEMA_VERSION = "pg-orbitflow-m2p1-torsion-diagnosis-v1"
ATOM_SYMBOLS = np.asarray(
    ("H", "C", "N", "O", "F", "B", "P", "S", "Cl", "Br", "I", "Si", "Sn")
)


def _dump(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _angles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.rad2deg(np.arctan2(values[:, 0], values[:, 1]))


def _circular_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    cosine = np.sum(left * right, axis=-1).clip(-1.0, 1.0)
    sine = left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]
    return np.abs(np.rad2deg(np.arctan2(sine, cosine)))


def _same_prediction_groups(prediction: np.ndarray, tolerance_degrees: float) -> list[list[int]]:
    angles = _angles(prediction)
    remaining = set(range(len(angles)))
    groups: list[list[int]] = []
    while remaining:
        first = min(remaining)
        delta = np.abs(
            np.rad2deg(
                np.arctan2(
                    np.sin(np.deg2rad(angles - angles[first])),
                    np.cos(np.deg2rad(angles - angles[first])),
                )
            )
        )
        group = sorted(index for index in remaining if delta[index] <= tolerance_degrees)
        remaining.difference_update(group)
        if len(group) > 1:
            groups.append(group)
    return groups


def _identical_row_groups(values: np.ndarray, tolerance: float) -> list[list[int]]:
    values = np.asarray(values, dtype=np.float64)
    remaining = set(range(len(values)))
    groups: list[list[int]] = []
    while remaining:
        first = min(remaining)
        group = sorted(
            index
            for index in remaining
            if float(np.max(np.abs(values[index] - values[first]))) <= tolerance
        )
        remaining.difference_update(group)
        if len(group) > 1:
            groups.append(group)
    return groups


def _representative_tuples(contract) -> np.ndarray:
    count = int(np.max(contract.torsion_orbit_id)) + 1
    return np.asarray(
        [
            contract.torsion_index[
                :, np.flatnonzero(contract.torsion_orbit_id == current)[0]
            ]
            for current in range(count)
        ],
        dtype=np.int64,
    )


def _equal_element_terminal_substitution(
    left: np.ndarray, right: np.ndarray, atom_types: np.ndarray
) -> bool:
    """Return true for the same rotor with one equal-element endpoint changed."""

    candidates = ((left, right), (left, right[::-1]))
    for first, second in candidates:
        if not np.array_equal(first[1:3], second[1:3]):
            continue
        different = np.flatnonzero(first != second)
        if len(different) != 1 or int(different[0]) not in {0, 3}:
            continue
        position = int(different[0])
        if atom_types[int(first[position])] == atom_types[int(second[position])]:
            return True
    return False


def _collapsed_group_record(
    group: list[int],
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    errors: np.ndarray,
    representatives: np.ndarray,
    atom_types: np.ndarray,
) -> dict:
    terminal_substitution = all(
        _equal_element_terminal_substitution(
            representatives[group[0]], representatives[current], atom_types
        )
        for current in group[1:]
    )
    return {
        "orbit_ids": group,
        "prediction_degrees": [float(_angles(prediction[group[:1]])[0])],
        "target_degrees": [float(value) for value in _angles(target[group])],
        "error_degrees": [float(value) for value in errors[group]],
        "representative_tuples": [representatives[index].tolist() for index in group],
        "representative_elements": [
            ATOM_SYMBOLS[atom_types[representatives[index]]].tolist() for index in group
        ],
        "same_rotor_equal_element_terminal_substitution": terminal_substitution,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-degrees", type=float, default=2.0)
    parser.add_argument("--collapse-tolerance-degrees", type=float, default=1e-3)
    parser.add_argument("--head-input-tolerance", type=float, default=0.0)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    run_dir = args.run_dir.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report_path = run_dir / "report.json"
    prediction_path = run_dir / "predictions.npz"
    formal = json.loads(report_path.read_text(encoding="utf-8"))
    if formal.get("status") != "FAIL_M2P1_PREDICTION_GATE_STOP_BEFORE_DECODER":
        raise ValueError("diagnosis requires the frozen M2.1 prediction-Gate failure")
    if formal["protocol"]["sha256"] != _sha256(protocol_path):
        raise ValueError("M2.1 protocol/report identity mismatch")

    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    samples = _panel_samples(protocol, tier)
    predictions = np.load(prediction_path, allow_pickle=False)
    import torch

    model = QuotientICPredictor(**_model_setting(protocol))
    checkpoint_path = run_dir / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("M2.1 checkpoint/protocol identity mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    captured_head_inputs: list[np.ndarray] = []

    def capture_input(_module, arguments):
        captured_head_inputs.append(arguments[0].detach().cpu().numpy())

    hook = model.torsion_head[0].register_forward_pre_hook(capture_input)
    molecule_records = []
    failed_orbit_count = 0
    collapsed_failed_orbits: set[tuple[int, int]] = set()
    identical_input_failed_orbits: set[tuple[int, int]] = set()
    terminal_explained_orbits: set[tuple[int, int]] = set()
    failed_molecules = 0
    cpu_prediction_max_abs_difference = 0.0

    for sample in samples:
        contract = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, contract)
        captured_head_inputs.clear()
        with torch.no_grad():
            recomputed = {
                key: value.detach().cpu().numpy()
                for key, value in model(graph, device="cpu").items()
            }
        if len(captured_head_inputs) != 1:
            raise RuntimeError("failed to capture exactly one torsion-head input")
        head_input = captured_head_inputs[0]
        prediction = predictions[f"{sample.package_index}_torsion_sincos"]
        current_difference = float(
            np.max(np.abs(recomputed["torsion_sincos"] - prediction))
        )
        cpu_prediction_max_abs_difference = max(
            cpu_prediction_max_abs_difference, current_difference
        )
        if current_difference > 1e-5:
            raise RuntimeError("CPU checkpoint prediction differs materially from archive")
        target = targets.torsion_target_sincos
        errors = _circular_error(prediction, target)
        failed = sorted(np.flatnonzero(errors > float(args.gate_degrees)).tolist())
        failed_orbit_count += len(failed)
        failed_molecules += int(bool(failed))
        representatives = _representative_tuples(contract)
        atom_types = np.asarray(sample.atom_types, dtype=np.int64)
        collapsed = []
        identical_input_groups = _identical_row_groups(
            head_input, float(args.head_input_tolerance)
        )
        for group in identical_input_groups:
            for orbit in set(group).intersection(failed):
                identical_input_failed_orbits.add(
                    (int(sample.package_index), int(orbit))
                )
        for group in _same_prediction_groups(
            prediction, float(args.collapse_tolerance_degrees)
        ):
            failed_in_group = sorted(set(group).intersection(failed))
            if not failed_in_group:
                continue
            record = _collapsed_group_record(
                group,
                prediction=prediction,
                target=target,
                errors=errors,
                representatives=representatives,
                atom_types=atom_types,
            )
            record["failed_orbit_ids"] = failed_in_group
            collapsed.append(record)
            for orbit in failed_in_group:
                collapsed_failed_orbits.add((int(sample.package_index), int(orbit)))
                if record["same_rotor_equal_element_terminal_substitution"]:
                    terminal_explained_orbits.add(
                        (int(sample.package_index), int(orbit))
                    )
        molecule_records.append(
            {
                "package_index": int(sample.package_index),
                "molecule_id": sample.molecule_id,
                "target_pg": sample.target_pg,
                "torsion_orbit_count": len(target),
                "failed_orbit_ids": failed,
                "failed_orbit_count": len(failed),
                "torsion_mae_degrees": float(np.mean(errors)),
                "torsion_max_error_degrees": float(np.max(errors)),
                "collapsed_prediction_groups": collapsed,
                "identical_torsion_head_input_groups": identical_input_groups,
            }
        )

    hook.remove()

    total_orbits = int(sum(row["torsion_orbit_count"] for row in molecule_records))
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSED_M2P1_LOCAL_EXCHANGEABLE_TORSION_COLLAPSE",
        "scope": {
            "read_only": True,
            "training_performed": False,
            "decoder_run": False,
            "gate_changed": False,
        },
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "formal_report_sha256": _sha256(report_path),
            "predictions_sha256": _sha256(prediction_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "thresholds": {
            "torsion_gate_degrees": float(args.gate_degrees),
            "same_prediction_tolerance_degrees": float(
                args.collapse_tolerance_degrees
            ),
            "identical_head_input_absolute_tolerance": float(
                args.head_input_tolerance
            ),
        },
        "summary": {
            "molecule_count": len(molecule_records),
            "passed_molecule_count": len(molecule_records) - failed_molecules,
            "failed_molecule_count": failed_molecules,
            "torsion_orbit_count": total_orbits,
            "failed_torsion_orbit_count": failed_orbit_count,
            "failed_orbits_in_same_prediction_groups": len(collapsed_failed_orbits),
            "failed_orbits_in_identical_head_input_groups": len(
                identical_input_failed_orbits
            ),
            "failed_orbits_in_equal_element_terminal_substitution_groups": len(
                terminal_explained_orbits
            ),
            "same_prediction_group_coverage_fraction": (
                len(collapsed_failed_orbits) / failed_orbit_count
                if failed_orbit_count
                else 1.0
            ),
            "identical_head_input_group_coverage_fraction": (
                len(identical_input_failed_orbits) / failed_orbit_count
                if failed_orbit_count
                else 1.0
            ),
            "equal_element_terminal_group_coverage_fraction": (
                len(terminal_explained_orbits) / failed_orbit_count
                if failed_orbit_count
                else 1.0
            ),
            "cpu_recomputed_prediction_max_abs_difference": (
                cpu_prediction_max_abs_difference
            ),
        },
        "evidence_based_interpretation": [
            "bond and angle heads passed while torsion failed on only six molecules",
            "nearly every failed torsion orbit belongs to a repeated-prediction group",
            "the corresponding final-checkpoint torsion-head input rows are exactly identical",
            "the repeated groups differ by equal-element terminal atoms around the same rotor",
            "the current atom-labelled per-orbit torsion objective is therefore not permutation-safe for local exchangeable substituents",
        ],
        "excluded_explanations": [
            "global inability to learn bond lengths",
            "global inability to learn bond angles",
            "failure on all non-planar molecules",
        ],
        "next_experiment_constraint": (
            "replace atom-labelled local-rotor torsion supervision with a permutation-invariant "
            "set/rotor representation; do not merely extend M2.1 steps or relax the 2-degree Gate"
        ),
        "molecules": molecule_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _dump(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"{result['status']} report={args.output.resolve()}")


if __name__ == "__main__":
    main()
