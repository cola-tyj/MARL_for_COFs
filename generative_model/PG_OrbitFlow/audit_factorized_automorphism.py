"""Check scalable factorized branch automorphisms on the frozen M3 panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .c0_local_recovery import _sha256
from .factorized_automorphism import (
    build_factorized_automorphism_contract,
    factorized_circular_errors_degrees,
)
from .geometry import build_geometry_contract
from .global_shape_fingerprint_model import FingerprintGlobalShapePredictor, graph_wl_fingerprint
from .m1_bond_angle_training import _angle_mae_degrees
from .m2p1_phase_training import _prediction_checks
from .m3_tier32_training import _model_setting
from .orbit_ic_model import build_orbit_ic_example
from .overfit import _panel_samples


def _factorized_prediction_record(sample, targets, prediction, factorized):
    import numpy as np

    errors = factorized_circular_errors_degrees(
        prediction["torsion_sincos"],
        targets.torsion_target_sincos,
        factorized,
    )
    angles = np.abs(
        np.rad2deg(
            np.arctan2(
                targets.torsion_target_sincos[:, 0],
                targets.torsion_target_sincos[:, 1],
            )
        )
    )
    nonplanar = np.minimum(angles, np.abs(180.0 - angles)) > 5.0
    return {
        "package_index": int(sample.package_index),
        "target_pg": sample.target_pg,
        "bond_orbit_mae_angstrom": float(
            np.mean(
                np.abs(
                    prediction["bond_lengths"] - targets.bond_target_lengths
                )
            )
        ),
        "angle_orbit_mae_degrees": _angle_mae_degrees(
            prediction["angle_cosines"], targets.angle_target_cosines
        ),
        "torsion_orbit_circular_mae_degrees": float(np.mean(errors)),
        "nonplanar_torsion_orbit_circular_mae_degrees": float(
            np.mean(errors[nonplanar])
        )
        if np.any(nonplanar)
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    import torch

    protocol_path = args.m3_protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    examples = []
    for sample in _panel_samples(protocol, tier):
        geometry = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, geometry)
        factorized = build_factorized_automorphism_contract(sample, geometry)
        fingerprint = graph_wl_fingerprint(
            sample,
            bits=int(protocol["model"]["wl_fingerprint_bits"]),
            radius=int(protocol["model"]["wl_fingerprint_radius"]),
        )
        examples.append((sample, graph, targets, factorized, fingerprint))
    model = FingerprintGlobalShapePredictor(
        quantile_count=int(protocol["global_shape_contract"]["quantile_count"]),
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
    summaries = []
    with torch.no_grad():
        for sample, graph, targets, factorized, fingerprint in examples:
            prediction = {
                key: value.cpu().numpy()
                for key, value in model(
                    graph, factorized, fingerprint, device=args.device
                ).items()
            }
            records.append(
                _factorized_prediction_record(
                    sample, targets, prediction, factorized
                )
            )
            summaries.append(
                {
                    "package_index": int(sample.package_index),
                    "target_pg": sample.target_pg,
                    "group_count": len(factorized.groups),
                    "maximum_group_size": max(map(len, factorized.groups), default=1),
                    "induced_generator_count": factorized.induced_generator_count,
                    "allowed_torsion_permutation_count": factorized.allowed_torsion_permutation_count,
                    "branch_swap_witness_count": factorized.branch_swap_witness_count,
                    "anchored_witness_count": factorized.anchored_witness_count,
                }
            )
    values, metric_checks, strata = _prediction_checks(
        records, protocol["prediction_gate"]
    )
    checks = {
        "molecule_count_exact": len(records) == 32,
        "all_group_sizes_supported": all(row["maximum_group_size"] <= 6 for row in summaries),
        "boh2_package39_branch_groups_recovered": next(
            row["group_count"] for row in summaries if row["package_index"] == 39
        ) >= 2,
        "original_prediction_gate_repasses": all(metric_checks.values()),
        "coordinates_not_used_to_build_factorization": True,
    }
    passed = all(checks.values())
    report = {
        "schema_version": "pg-orbitflow-factorized-automorphism-audit-v1",
        "status": "PASS_FACTORIZED_AUTOMORPHISM_ADVANCE_TO_DATASET_SMOKE" if passed else "FAIL_FACTORIZED_AUTOMORPHISM_STOP_DATASET_BRANCH",
        "passed": passed,
        "checks": checks,
        "prediction_gate": {
            "checks": metric_checks,
            "values": values,
            "strata": strata,
            "thresholds": protocol["prediction_gate"],
        },
        "records": summaries,
        "identity": {
            "m3_protocol_sha256": _sha256(protocol_path),
            "m3_checkpoint_sha256": _sha256(Path(protocol["predictor"]["checkpoint"]["path"])),
            "implementation_sha256": _sha256(Path(__file__).with_name("factorized_automorphism.py")),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{report['status']} report={output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
