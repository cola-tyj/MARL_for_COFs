"""Audit and freeze the nested M3 Tier-32 C2/C3 panel."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .audit_m2_tier16_panel import _geometry_support
from .c0_local_recovery import _sha256
from .data import PGOrbitFlowDataset
from .local_rotor import build_local_rotor_set_contract


def _dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _eligible(sample) -> tuple[bool, dict]:
    support = _geometry_support(sample)
    eligible = (
        len(sample.atom_types) <= 21
        and support["undefined_torsion_orbit_count"] == 0
        and support["nonplanar_torsion_orbit_count"] > 0
        and support["active_chirality_count"] > 0
    )
    return eligible, support


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-matching-report", type=Path, required=True)
    parser.add_argument("--m2-reproducibility-report", type=Path, required=True)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("generative_model/data/processed/v2"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-output", type=Path, required=True)
    args = parser.parse_args()

    matching_path = args.m2_matching_report.resolve()
    reproduction_path = args.m2_reproducibility_report.resolve()
    matching = json.loads(matching_path.read_text(encoding="utf-8"))
    reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
    if not matching.get("passed") or not reproduction.get("passed"):
        raise RuntimeError("M3 panel audit requires completed M2 matching and reproducibility Gates")
    m2_protocol_path = Path(matching["source"]["protocol"]["path"])
    if _sha256(m2_protocol_path) != matching["source"]["protocol"]["sha256"]:
        raise RuntimeError("M2 protocol identity changed")
    m2_protocol = json.loads(m2_protocol_path.read_text(encoding="utf-8"))
    m2_indices = tuple(int(row["package_index"]) for row in m2_protocol["panel"]["records"])
    if len(m2_indices) != 16 or len(set(m2_indices)) != 16:
        raise RuntimeError("M3 requires the frozen 16-molecule M2 parent panel")

    package_dir = args.package_dir.resolve()
    dataset = PGOrbitFlowDataset(
        package_dir, split="train", split_scheme="iid", point_groups=("C2", "C3")
    )
    samples = {dataset[index].package_index: dataset[index] for index in range(len(dataset))}
    eligible_by_pg: dict[str, list] = defaultdict(list)
    support_by_index = {}
    for sample in samples.values():
        eligible, support = _eligible(sample)
        support_by_index[int(sample.package_index)] = support
        if eligible and sample.package_index not in m2_indices:
            eligible_by_pg[sample.target_pg].append(sample)
    for values in eligible_by_pg.values():
        values.sort(key=lambda sample: int(sample.package_index))

    # C3 has exactly eight eligible additions under the frozen scientific
    # criteria.  C2 is more numerous, so select the smallest package index per
    # eligible core; this maximizes core coverage without geometry-based ranking.
    c3_added = tuple(sample.package_index for sample in eligible_by_pg["C3"])
    c2_by_core = defaultdict(list)
    for sample in eligible_by_pg["C2"]:
        c2_by_core[sample.core_name].append(sample)
    c2_added = tuple(
        sorted(min(values, key=lambda sample: sample.package_index).package_index for values in c2_by_core.values())
    )
    added = c2_added + c3_added
    selected = m2_indices + added
    if any(index not in samples for index in selected):
        raise RuntimeError("selected M3 molecule is not in IID train")

    records = []
    local_group_count = 0
    operation_component_count = 0
    for package_index in selected:
        sample = samples[package_index]
        support = dict(support_by_index[package_index])
        contract = support.pop("contract")
        local = build_local_rotor_set_contract(sample, contract)
        local_group_count += len(local.groups)
        operation_component_count += len(local.coupled_group_components)
        records.append(
            {
                "package_index": int(sample.package_index),
                "molecule_id": sample.molecule_id,
                "target_pg": sample.target_pg,
                "core": sample.core_name,
                "arm": sample.arm_name,
                "num_atoms": len(sample.atom_types),
                "contains_sn": bool(np.any(sample.atomic_numbers == 50)),
                "in_m2_tier16": package_index in m2_indices,
                "local_rotor_group_count": len(local.groups),
                "operation_coupled_component_count": len(local.coupled_group_components),
                **support,
            }
        )

    counts = Counter(record["target_pg"] for record in records)
    added_records = [record for record in records if not record["in_m2_tier16"]]
    checks = {
        "molecule_count_exact": len(records) == 32,
        "c2_c3_balanced": counts == {"C2": 16, "C3": 16},
        "m2_tier16_strictly_nested": tuple(record["package_index"] for record in records[:16]) == m2_indices,
        "iid_train_only": True,
        "added_molecule_count_exact": len(added_records) == 16,
        "added_c3_is_complete_eligible_population": len(c3_added) == 8,
        "added_c2_one_minimum_index_per_eligible_core": len(c2_added) == 8,
        "all_added_have_nonplanar_torsion": all(record["nonplanar_torsion_orbit_count"] > 0 for record in added_records),
        "all_added_have_active_chirality": all(record["active_chirality_count"] > 0 for record in added_records),
        "no_undefined_circular_targets": all(record["undefined_torsion_orbit_count"] == 0 for record in records),
        "atom_count_max_21": max(record["num_atoms"] for record in records) <= 21,
        "operation_coupled_contract_constructed": True,
        "coordinates_not_used_for_panel_ranking": True,
    }
    passed = all(checks.values())
    panel = {
        "schema_version": "pg-orbitflow-m3-tier32-panel-v1",
        "status": "FROZEN_FOR_M3_TIER32" if passed else "NOT_FROZEN_AUDIT_FAILED",
        "m2_parent": {
            "matching_report": {"path": str(matching_path), "sha256": _sha256(matching_path)},
            "reproducibility_report": {"path": str(reproduction_path), "sha256": _sha256(reproduction_path)},
            "protocol": {"path": str(m2_protocol_path), "sha256": _sha256(m2_protocol_path)},
        },
        "package_dir": str(package_dir),
        "canonical_manifest_sha256": _sha256(package_dir / "manifest.json"),
        "selection_rule": (
            "strictly retain the passed M2 Tier-16 panel; among remaining IID-train C2/C3, "
            "require N<=21, nonplanar torsion, active chirality, and no undefined circular target; "
            "take the complete eight-molecule eligible C3 population and the minimum package index "
            "from each of the eight eligible C2 cores"
        ),
        "molecule_count": len(records),
        "records": records,
    }
    _dump(args.panel_output.resolve(), panel)
    result = {
        "schema_version": "pg-orbitflow-m3-tier32-panel-audit-v1",
        "status": "PASS_M3_TIER32_PANEL_AUDIT" if passed else "FAIL_M3_TIER32_PANEL_AUDIT",
        "passed": passed,
        "checks": checks,
        "selected_summary": {
            "point_group_counts": dict(sorted(counts.items())),
            "added_c2_package_indices": list(c2_added),
            "added_c3_package_indices": list(c3_added),
            "added_c2_core_count": len(c2_by_core),
            "nonplanar_torsion_orbit_count": sum(record["nonplanar_torsion_orbit_count"] for record in records),
            "active_chirality_count": sum(record["active_chirality_count"] for record in records),
            "local_rotor_group_count": local_group_count,
            "operation_coupled_component_count": operation_component_count,
            "max_atom_count": max(record["num_atoms"] for record in records),
        },
        "panel": {"path": str(args.panel_output.resolve()), "sha256": _sha256(args.panel_output.resolve())},
        "decision": "freeze M3 Tier-32 training protocol" if passed else "stop before M3 training",
    }
    _dump(args.output.resolve(), result)
    print(f"{result['status']} report={args.output.resolve()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
