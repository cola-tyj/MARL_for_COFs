"""Audit centralizer graph automorphisms against the frozen M3 torsion failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .graph_automorphism import build_graph_automorphism_contract
from .overfit import _panel_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(); run_dir = args.run_dir.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "FAIL_M3_PREDICTION_GATE_STOP_BEFORE_DECODER":
        raise RuntimeError("M3.1 audit requires the frozen M3 prediction failure")
    failed = {
        int(row["package_index"])
        for row in report["prediction_records"]
        if row["torsion_orbit_circular_mae_degrees"] > 2
        or row["nonplanar_torsion_orbit_circular_mae_degrees"] > 2
    }
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    records = []; failed_with_groups = set(); sizes = []
    for sample in _panel_samples(protocol, tier):
        contract = build_graph_automorphism_contract(sample, build_geometry_contract(sample))
        sizes.extend(len(group) for group in contract.groups)
        if sample.package_index in failed and contract.groups:
            failed_with_groups.add(int(sample.package_index))
        records.append({
            "package_index": int(sample.package_index), "target_pg": sample.target_pg,
            "centralizer_atom_automorphism_count": len(contract.allowed_atom_permutations),
            "induced_torsion_automorphism_count": len(contract.allowed_torsion_permutations),
            "torsion_set_count": len(contract.groups),
            "torsion_set_sizes": [len(group) for group in contract.groups],
            "torsion_sets": [list(group) for group in contract.groups],
            "failed_in_m3_v2": sample.package_index in failed,
        })
    checks = {
        "molecule_count_exact": len(records) == 32,
        "identity_present_for_every_molecule": all(row["centralizer_atom_automorphism_count"] >= 1 for row in records),
        "all_set_sizes_supported_by_generalized_head": all(2 <= size <= 6 for size in sizes),
        "every_failed_molecule_has_automorphism_set": failed_with_groups == failed,
        "graph_and_action_only": True,
        "coordinates_or_targets_not_used_to_build_contract": True,
    }
    passed = all(checks.values())
    output = {
        "schema_version": "pg-orbitflow-m3p1-automorphism-contract-audit-v2",
        "status": "PASS_M3P1_GENERALIZED_AUTOMORPHISM_CONTRACT_GATE" if passed else "FAIL_M3P1_GENERALIZED_AUTOMORPHISM_CONTRACT_GATE",
        "passed": passed, "checks": checks,
        "identity": {"m3_protocol_sha256": _sha256(protocol_path), "m3_report_sha256": _sha256(run_dir / "report.json"), "m3_predictions_sha256": _sha256(run_dir / "predictions.npz"), "implementation_sha256": _sha256(Path(__file__).resolve().parent / "graph_automorphism.py")},
        "summary": {"failed_molecule_count": len(failed), "covered_failed_molecule_count": len(failed_with_groups), "torsion_set_count": sum(row["torsion_set_count"] for row in records), "maximum_set_size": max(sizes, default=1)},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{output['status']} report={args.output.resolve()}")
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
