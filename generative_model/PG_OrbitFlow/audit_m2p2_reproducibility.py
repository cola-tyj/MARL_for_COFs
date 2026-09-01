"""Reproduce M2.2 predictions and each archived selected decoder start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .local_atom_matching import best_terminal_sibling_relabeling
from .local_rotor import build_local_rotor_set_contract
from .m0_oracle_reconstruction import optimize_one_start
from .m2_torsion_training import _learned_decoder_targets
from .m2p1_phase_training import _model_setting
from .m2p2_model import SetValuedQuotientICPredictor
from .m2p2_set_training import _load_protocol, _prediction_record
from .metrics import kabsch_rmsd
from .orbit_ic_model import build_orbit_ic_example
from .orbit_kinematics import build_orbit_parameterization
from .overfit import _panel_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--matching-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    import torch

    if args.prediction_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA prediction reproduction requested but unavailable")
    protocol_path = args.protocol.resolve()
    checkpoint_path = args.checkpoint.resolve()
    run_dir = args.run_dir.resolve()
    matching_path = args.matching_report.resolve()
    protocol = _load_protocol(protocol_path)
    source = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    matching = json.loads(matching_path.read_text(encoding="utf-8"))
    if not source["prediction_gate"]["passed"] or not matching["passed"]:
        raise RuntimeError("reproducibility requires passed prediction and matched reconstruction Gates")
    if matching["source"]["report"]["sha256"] != _sha256(run_dir / "report.json"):
        raise RuntimeError("matching audit does not identify the supplied source report")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state["protocol_sha256"] != _sha256(protocol_path):
        raise RuntimeError("checkpoint does not identify the supplied protocol")
    model = SetValuedQuotientICPredictor(**_model_setting(protocol))
    model.load_state_dict(state["model"], strict=True)
    model = model.to(args.prediction_device).eval()

    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    samples = _panel_samples(protocol, tier)
    source_prediction = {int(row["package_index"]): row for row in source["prediction_records"]}
    source_reconstruction = {int(row["package_index"]): row for row in source["reconstruction_records"]}
    source_assignment = {int(row["package_index"]): row for row in matching["assignments"]}
    with np.load(run_dir / "coordinates.npz", allow_pickle=False) as archive:
        archived_coordinates = np.asarray(archive["coordinates"], dtype=np.float64)
        offsets = np.asarray(archive["atom_offsets"], dtype=np.int64)
        package_indices = np.asarray(archive["package_indices"], dtype=np.int64)
    expected = np.asarray([sample.package_index for sample in samples], dtype=np.int64)
    if not np.array_equal(package_indices, expected):
        raise RuntimeError("archived coordinate molecule order changed")

    decoder_protocol = {
        key: protocol["decoder"][key]
        for key in ("initialization", "optimization", "energy_weights")
    }
    stride = int(protocol["decoder"]["initialization"]["molecule_specific_seed_stride"])
    metric_names = (
        "bond_orbit_mae_angstrom",
        "angle_orbit_mae_degrees",
        "torsion_orbit_circular_mae_degrees",
        "nonplanar_torsion_orbit_circular_mae_degrees",
    )
    records = []
    for position, sample in enumerate(samples):
        contract = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, contract)
        local = build_local_rotor_set_contract(sample, contract)
        with torch.no_grad():
            prediction = {
                key: value.detach().cpu().numpy()
                for key, value in model(graph, local, device=args.prediction_device).items()
            }
        repeated_prediction = _prediction_record(sample, targets, prediction, local)
        prediction_difference = max(
            abs(float(repeated_prediction[name]) - float(source_prediction[sample.package_index][name]))
            for name in metric_names
        )
        selected = int(source_reconstruction[sample.package_index]["selected_start_index"])
        _, repeated = optimize_one_start(
            sample,
            contract,
            _learned_decoder_targets(prediction, contract),
            build_orbit_parameterization(sample),
            decoder_protocol,
            seed=int(protocol["seed"]) + sample.package_index * stride + selected,
            device="cpu",
        )
        saved = archived_coordinates[int(offsets[position]) : int(offsets[position + 1])]
        coordinate_max_abs = float(np.max(np.abs(saved - repeated)))
        coordinate_kabsch = kabsch_rmsd(saved, repeated)
        repeated_match = best_terminal_sibling_relabeling(sample, repeated)
        saved_assignment = source_assignment[sample.package_index]
        assignment_identical = repeated_match["permutation"].tolist() == saved_assignment["permutation"]
        matched_kabsch_difference = abs(
            float(repeated_match["kabsch_rmsd_angstrom"])
            - float(saved_assignment["matched_kabsch_rmsd_angstrom"])
        )
        passed = (
            prediction_difference <= 5e-5
            and coordinate_max_abs <= 5e-7
            and coordinate_kabsch <= 1e-7
            and assignment_identical
            and matched_kabsch_difference <= 5e-7
        )
        records.append(
            {
                "package_index": int(sample.package_index),
                "prediction_metric_max_abs_difference": prediction_difference,
                "selected_start_index": selected,
                "coordinate_max_abs_difference_angstrom": coordinate_max_abs,
                "coordinate_kabsch_rmsd_angstrom": coordinate_kabsch,
                "terminal_sibling_assignment_identical": assignment_identical,
                "matched_kabsch_abs_difference_angstrom": matched_kabsch_difference,
                "passed": passed,
            }
        )
        print(f"reproduce package_index={sample.package_index} passed={passed}", flush=True)

    passed = all(record["passed"] for record in records)
    output = {
        "schema_version": "pg-orbitflow-m2p2-reproducibility-v1",
        "status": "PASS_M2P2_REPRODUCIBILITY" if passed else "FAIL_M2P2_REPRODUCIBILITY",
        "passed": passed,
        "scope": {
            "prediction_recomputed_from_checkpoint": True,
            "selected_decoder_start_recomputed": True,
            "all_decoder_starts_recomputed": False,
            "terminal_sibling_matching_recomputed": True,
        },
        "tolerances": {
            "prediction_metric_max_abs": 5e-5,
            "coordinate_max_abs_angstrom": 5e-7,
            "coordinate_kabsch_angstrom": 1e-7,
            "matched_kabsch_abs_angstrom": 5e-7,
        },
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "source_report_sha256": _sha256(run_dir / "report.json"),
            "source_coordinates_sha256": _sha256(run_dir / "coordinates.npz"),
            "matching_report_sha256": _sha256(matching_path),
            "auditor_sha256": _sha256(Path(__file__).resolve()),
        },
        "prediction_device": args.prediction_device,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{output['status']} report={args.output.resolve()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
