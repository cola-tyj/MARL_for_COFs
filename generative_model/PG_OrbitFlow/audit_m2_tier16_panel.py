"""Audit and freeze a balanced Tier-16 panel with non-planar/chiral support."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .data import PGOrbitFlowDataset
from .geometry import _chirality_values, _torsion_sincos, build_geometry_contract
from .orbit_ic_model import build_orbit_ic_example


FROZEN_PACKAGE_INDICES = (
    864,
    1045,
    879,
    892,
    875,
    1075,
    1768,
    406,
    1135,
    2391,
    1222,
    1124,
    1143,
    2389,
    2403,
    62,
)
M2_TIER4_INDICES = (864, 1045, 1135, 2391)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _geometry_support(sample) -> dict:
    import torch

    contract = build_geometry_contract(sample)
    coordinates = torch.as_tensor(sample.symmetric_target_angstrom, dtype=torch.float64)
    torsion = _torsion_sincos(
        coordinates, torch.as_tensor(contract.torsion_index, dtype=torch.long)
    ).numpy()
    chirality = _chirality_values(
        coordinates, torch.as_tensor(contract.chirality_index, dtype=torch.long)
    ).numpy()
    orbit_torsions = []
    undefined = 0
    for current in np.unique(contract.torsion_orbit_id):
        mean = torsion[contract.torsion_orbit_id == current].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-8:
            undefined += 1
        else:
            orbit_torsions.append(mean / norm)
    orbit_torsions = np.asarray(orbit_torsions, dtype=np.float64).reshape(-1, 2)
    absolute = np.abs(
        np.rad2deg(np.arctan2(orbit_torsions[:, 0], orbit_torsions[:, 1]))
    )
    distance_to_planar = np.minimum(absolute, np.abs(180.0 - absolute))
    return {
        "contract": contract,
        "torsion_orbit_count": int(len(np.unique(contract.torsion_orbit_id))),
        "undefined_torsion_orbit_count": int(undefined),
        "nonplanar_torsion_orbit_count": int(np.sum(distance_to_planar > 5.0)),
        "max_distance_to_planar_degrees": float(distance_to_planar.max(initial=0.0)),
        "chirality_candidate_count": int(len(chirality)),
        "active_chirality_count": int(np.sum(np.abs(chirality) >= 0.05)),
    }


def _head_feature_conflicts(sample, contract) -> dict:
    graph, targets = build_orbit_ic_example(sample, contract)

    def groups(keys, values, circular=False):
        buckets: dict[tuple, list[np.ndarray]] = {}
        for key, value in zip(keys, values, strict=True):
            buckets.setdefault(key, []).append(np.asarray(value, dtype=np.float64))
        conflicts = 0
        for rows in buckets.values():
            if len(rows) < 2:
                continue
            anchor = rows[0]
            for current in rows[1:]:
                if circular:
                    error = abs(
                        np.rad2deg(
                            np.arctan2(
                                current[0] * anchor[1] - current[1] * anchor[0],
                                np.dot(current, anchor),
                            )
                        )
                    )
                    conflicts += int(error > 1.0)
                else:
                    conflicts += int(float(np.max(np.abs(current - anchor))) > 1e-4)
        return conflicts

    bond_keys = [
        tuple(sorted(graph.bond_orbit_endpoints[:, index].tolist()))
        + tuple(np.round(graph.bond_orbit_features[index], 6))
        for index in range(graph.bond_orbit_endpoints.shape[1])
    ]
    angle_keys = [
        (
            min(int(graph.angle_orbit_nodes[0, index]), int(graph.angle_orbit_nodes[2, index])),
            int(graph.angle_orbit_nodes[1, index]),
            max(int(graph.angle_orbit_nodes[0, index]), int(graph.angle_orbit_nodes[2, index])),
            *tuple(np.round(graph.angle_orbit_features[index], 6)),
        )
        for index in range(graph.angle_orbit_nodes.shape[1])
    ]
    phase_aware_bond_keys = [
        bond_keys[index] + tuple(np.round(graph.bond_phase_features[index], 6))
        for index in range(len(bond_keys))
    ]
    phase_aware_angle_keys = [
        angle_keys[index] + tuple(np.round(graph.angle_phase_features[index], 6))
        for index in range(len(angle_keys))
    ]
    torsion_keys = [
        tuple(
            min(
                tuple(graph.torsion_orbit_nodes[:, index].tolist()),
                tuple(reversed(graph.torsion_orbit_nodes[:, index].tolist())),
            )
        )
        + tuple(np.round(graph.torsion_orbit_features[index], 6))
        for index in range(graph.torsion_orbit_nodes.shape[1])
    ]
    phase_aware_torsion_keys = [
        torsion_keys[index]
        + tuple(np.round(graph.torsion_phase_features[index], 6))
        for index in range(len(torsion_keys))
    ]
    return {
        "bond_head_exact_feature_conflicts": groups(
            bond_keys, targets.bond_target_lengths
        ),
        "phase_aware_bond_head_exact_feature_conflicts": groups(
            phase_aware_bond_keys, targets.bond_target_lengths
        ),
        "angle_head_exact_feature_conflicts": groups(
            angle_keys, targets.angle_target_cosines
        ),
        "phase_aware_angle_head_exact_feature_conflicts": groups(
            phase_aware_angle_keys, targets.angle_target_cosines
        ),
        "torsion_head_exact_feature_conflicts": groups(
            torsion_keys, targets.torsion_target_sincos, circular=True
        ),
        "phase_aware_torsion_head_exact_feature_conflicts": groups(
            phase_aware_torsion_keys,
            targets.torsion_target_sincos,
            circular=True,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m2-summary",
        type=Path,
        default=Path(
            "generative_model/PG_OrbitFlow/reports/m2_torsion_tier4_v1_summary.json"
        ),
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("generative_model/data/processed/v2"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.m2_summary.resolve()
    summary = _json(summary_path)
    if not summary.get("passed"):
        raise ValueError("Tier-16 audit requires passed M2")
    dataset = PGOrbitFlowDataset(
        args.package_dir.resolve(), split="train", point_groups=("C2", "C3")
    )
    samples = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        samples[sample.package_index] = sample
    if any(index not in samples for index in FROZEN_PACKAGE_INDICES):
        raise ValueError("frozen Tier-16 record is not in IID train")

    population = Counter()
    undefined_records = []
    for sample in samples.values():
        support = _geometry_support(sample)
        population[f"{sample.target_pg}_molecules"] += 1
        if support["undefined_torsion_orbit_count"]:
            population[f"{sample.target_pg}_undefined_torsion_molecules"] += 1
            undefined_records.append(sample.package_index)
            continue
        if support["nonplanar_torsion_orbit_count"]:
            population[f"{sample.target_pg}_nonplanar_molecules"] += 1
        if support["active_chirality_count"]:
            population[f"{sample.target_pg}_active_chirality_molecules"] += 1

    records = []
    for package_index in FROZEN_PACKAGE_INDICES:
        sample = samples[package_index]
        support = _geometry_support(sample)
        conflicts = _head_feature_conflicts(sample, support.pop("contract"))
        records.append(
            {
                "package_index": sample.package_index,
                "molecule_id": sample.molecule_id,
                "target_pg": sample.target_pg,
                "core": sample.core_name,
                "arm": sample.arm_name,
                "num_atoms": len(sample.atom_types),
                "contains_sn": bool(np.any(sample.atomic_numbers == 50)),
                "in_m2_tier4": package_index in M2_TIER4_INDICES,
                **support,
                **conflicts,
            }
        )
    added = [row for row in records if not row["in_m2_tier4"]]
    group_counts = Counter(row["target_pg"] for row in records)
    conflict_total = sum(
        row[key]
        for row in records
        for key in (
            "bond_head_exact_feature_conflicts",
            "angle_head_exact_feature_conflicts",
            "torsion_head_exact_feature_conflicts",
        )
    )
    phase_aware_conflict_total = sum(
        row[key]
        for row in records
        for key in (
            "phase_aware_bond_head_exact_feature_conflicts",
            "phase_aware_angle_head_exact_feature_conflicts",
            "phase_aware_torsion_head_exact_feature_conflicts",
        )
    )
    checks = {
        "molecule_count_exact": len(records) == 16,
        "c2_c3_balanced": group_counts == {"C2": 8, "C3": 8},
        "m2_tier4_nested": set(M2_TIER4_INDICES).issubset(FROZEN_PACKAGE_INDICES),
        "iid_train_only": True,
        "all_added_have_nonplanar_torsion": all(
            row["nonplanar_torsion_orbit_count"] > 0 for row in added
        ),
        "all_added_have_active_chirality": all(
            row["active_chirality_count"] > 0 for row in added
        ),
        "no_undefined_circular_targets": all(
            row["undefined_torsion_orbit_count"] == 0 for row in records
        ),
        "current_m2_aliasing_exposed": conflict_total > 0,
        "phase_aware_geometry_aliasing_resolved": phase_aware_conflict_total == 0,
        "atom_count_max_21": max(row["num_atoms"] for row in records) <= 21,
    }
    passed = all(checks.values())
    panel = {
        "schema_version": "pg-orbitflow-m2-tier16-panel-v1",
        "status": "FROZEN_FOR_PHASE_AWARE_M2P1" if passed else "NOT_FROZEN_AUDIT_FAILED",
        "base_m2_summary": {
            "path": str(summary_path),
            "sha256": _sha256(summary_path),
        },
        "package_dir": str(args.package_dir.resolve()),
        "canonical_manifest_sha256": _sha256(
            args.package_dir.resolve() / "manifest.json"
        ),
        "selection_rule": (
            "retain the four passed M2 molecules; add six per PG from IID train, "
            "requiring non-planar torsion and active chirality, avoiding undefined "
            "circular orbit targets, keeping N<=21 and deterministic explicit indices"
        ),
        "molecule_count": len(records),
        "records": records,
    }
    _dump(args.panel_output.resolve(), panel)
    result = {
        "schema_version": "pg-orbitflow-m2-tier16-panel-audit-v1",
        "status": "PASS_M2_TIER16_PANEL_AUDIT" if passed else "FAIL_M2_TIER16_PANEL_AUDIT",
        "passed": passed,
        "checks": checks,
        "population": dict(sorted(population.items())),
        "undefined_torsion_record_count": len(undefined_records),
        "undefined_torsion_package_indices": undefined_records,
        "selected_summary": {
            "point_group_counts": dict(group_counts),
            "added_molecule_count": len(added),
            "added_nonplanar_molecule_count": sum(
                row["nonplanar_torsion_orbit_count"] > 0 for row in added
            ),
            "added_active_chirality_molecule_count": sum(
                row["active_chirality_count"] > 0 for row in added
            ),
            "selected_nonplanar_torsion_orbit_count": sum(
                row["nonplanar_torsion_orbit_count"] for row in records
            ),
            "selected_active_chirality_count": sum(
                row["active_chirality_count"] for row in records
            ),
            "max_atom_count": max(row["num_atoms"] for row in records),
            "exact_feature_conflict_count": conflict_total,
            "phase_aware_geometry_feature_conflict_count": phase_aware_conflict_total,
        },
        "panel": {
            "path": str(args.panel_output.resolve()),
            "sha256": _sha256(args.panel_output.resolve()),
        },
        "decision": (
            "freeze a phase-aware M2.1 training/evaluation protocol for Tier-16; "
            "add an explicit chirality evaluation Gate before training"
            if passed
            else "revise panel selection before any Tier-16 training"
        ),
    }
    _dump(args.output.resolve(), result)
    if not passed:
        raise SystemExit(result["status"])
    print(f"{result['status']} report={args.output.resolve()}")


if __name__ == "__main__":
    main()
