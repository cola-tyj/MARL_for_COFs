"""Audit M2.2 reconstruction modulo strictly graph-equivalent terminal atoms.

The decoder output is never changed.  Reference coordinates are used only to
choose among terminal siblings that share one parent and identical canonical
atom/bond attributes.  Raw group-action and collision measurements remain the
authoritative measurements from the unmodified decoder coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .local_atom_matching import best_terminal_sibling_relabeling
from .m0_oracle_reconstruction import build_oracle_targets, reconstruction_metrics
from .m2p1_phase_training import _reconstruction_checks
from .overfit import _panel_samples


SCHEMA_VERSION = "pg-orbitflow-m2p2-terminal-sibling-matching-audit-v1"


def _dump(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_inputs(protocol_path: Path, run_dir: Path) -> tuple[dict, dict]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["protocol"]["sha256"] != _sha256(protocol_path):
        raise RuntimeError("run report does not identify the supplied protocol")
    if report["protocol"]["path"] != str(protocol_path):
        raise RuntimeError("run report protocol path changed")
    if not report["prediction_gate"]["passed"]:
        raise RuntimeError("terminal-sibling audit requires a passed prediction Gate")
    if report["status"] != "FAIL_M2P2_RECONSTRUCTION_GATE":
        raise RuntimeError("terminal-sibling audit is restricted to the frozen v4 failure")
    if len(report["reconstruction_records"]) != len(protocol["panel"]["records"]):
        raise RuntimeError("reconstruction record count changed")
    return protocol, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    run_dir = args.run_dir.resolve()
    output_path = args.output.resolve()
    protocol, source = _load_inputs(protocol_path, run_dir)
    archive_path = run_dir / "coordinates.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        coordinates = np.asarray(archive["coordinates"], dtype=np.float64)
        offsets = np.asarray(archive["atom_offsets"], dtype=np.int64)
        package_indices = np.asarray(archive["package_indices"], dtype=np.int64)

    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    samples = _panel_samples(protocol, tier)
    expected = np.asarray([sample.package_index for sample in samples], dtype=np.int64)
    if not np.array_equal(package_indices, expected):
        raise RuntimeError("coordinate archive molecule order changed")
    if len(offsets) != len(samples) + 1 or offsets[0] != 0 or offsets[-1] != len(coordinates):
        raise RuntimeError("coordinate archive offsets are invalid")

    raw_by_index = {
        int(record["package_index"]): record
        for record in source["reconstruction_records"]
    }
    matched_records = []
    assignment_records = []
    raw_physical_keys = (
        "collision_free",
        "collision_threshold_ratio",
        "minimum_covalent_radius_ratio",
        "minimum_nonbonded_distance_angstrom",
        "max_atom_error_angstrom",
        "max_operation_rms_angstrom",
        "mean_operation_rms_angstrom",
    )
    for position, sample in enumerate(samples):
        raw = raw_by_index[int(sample.package_index)]
        start, stop = int(offsets[position]), int(offsets[position + 1])
        decoded = coordinates[start:stop]
        match = best_terminal_sibling_relabeling(sample, decoded)
        contract = build_geometry_contract(sample)
        metrics = reconstruction_metrics(
            sample,
            contract,
            build_oracle_targets(sample, contract),
            match["coordinates"],
        )
        # Relabeling may not commute with the stored atom-index group action.
        # These physical properties are therefore copied from raw coordinates.
        for key in raw_physical_keys:
            metrics[key] = raw[key]
        matched_records.append(
            {
                "package_index": int(sample.package_index),
                "target_pg": sample.target_pg,
                "selected_start_index": int(raw["selected_start_index"]),
                "selected_energy": float(raw["selected_energy"]),
                "finite_start_count": int(raw["finite_start_count"]),
                "all_starts_finite": bool(raw["all_starts_finite"]),
                **metrics,
            }
        )
        assignment_records.append(
            {
                "package_index": int(sample.package_index),
                "terminal_sibling_groups": match["terminal_sibling_groups"],
                "assignment_count": int(match["assignment_count"]),
                "nonidentity_assignment": bool(match["nonidentity_assignment"]),
                "permutation": match["permutation"].tolist(),
                "raw_kabsch_rmsd_angstrom": float(raw["kabsch_rmsd_angstrom"]),
                "matched_kabsch_rmsd_angstrom": float(match["kabsch_rmsd_angstrom"]),
            }
        )

    values, checks, strata = _reconstruction_checks(
        matched_records, protocol["reconstruction_gate"]
    )
    passed = all(checks.values())
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "PASS_M2P2_GRAPH_EQUIVALENT_RECONSTRUCTION_GATE"
            if passed
            else "FAIL_M2P2_GRAPH_EQUIVALENT_RECONSTRUCTION_GATE"
        ),
        "passed": passed,
        "scope": {
            "decoder_coordinates_modified": False,
            "decoder_rerun": False,
            "training_or_checkpoint_modified": False,
            "gate_thresholds_modified": False,
            "reference_coordinates_used_only_for_evaluation_matching": True,
            "allowed_matching": "same-parent degree-one siblings with identical atom type, formal charge, radical count, and bond type",
            "free_element_hungarian_used": False,
            "raw_group_action_and_collision_metrics_preserved": True,
        },
        "source": {
            "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
            "report": {"path": str(run_dir / "report.json"), "sha256": _sha256(run_dir / "report.json")},
            "coordinates": {"path": str(archive_path), "sha256": _sha256(archive_path)},
            "checkpoint": {"path": str(run_dir / "last.pt"), "sha256": _sha256(run_dir / "last.pt")},
            "auditor": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        },
        "assignment_summary": {
            "molecule_count": len(samples),
            "nonidentity_molecule_count": sum(record["nonidentity_assignment"] for record in assignment_records),
            "total_terminal_sibling_group_count": sum(len(record["terminal_sibling_groups"]) for record in assignment_records),
        },
        "assignments": assignment_records,
        "matched_reconstruction_records": matched_records,
        "matched_reconstruction_gate": {
            "passed": passed,
            "checks": checks,
            "values": values,
            "strata": strata,
            "thresholds": protocol["reconstruction_gate"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _dump(output_path, output)
    print(f"{output['status']} report={output_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
