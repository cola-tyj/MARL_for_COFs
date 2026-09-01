"""Layered chemical and geometric validity audit for Cn our_ET_Flow outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from generative_model.data.cn_graph_dataset import CnGraphDataset
from generative_model.evaluation.chemistry import (
    canonical_graph_smiles, fingerprint_generator, reconstruct_molecule,
)
from generative_model.evaluation.schema import EvaluationSample
from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT, _atomic_json, _sha256,
)


SCHEMA_VERSION = "our-etflow-cn-layered-validity-audit-v1"
DEFAULT_INPUT = ROOT / "generative_model/data/Cn"
DEFAULT_RESULTS = ROOT / "generative_model/results/Cn"
COLLISION_THRESHOLD_ANGSTROM = 0.6
BOND_RATIO_MIN = 0.65
BOND_RATIO_MAX = 1.35
UFF_STABILITY_RMSD_ANGSTROM = 0.5
UFF_MAX_ITERATIONS = 500


def _kabsch_rmsd(reference: np.ndarray, moving: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(moving, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("Kabsch 输入必须为同 shape [N,3]")
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    u, _, vt = np.linalg.svd(right.T @ left)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = right @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - left) ** 2, axis=1))))


def _pair_distances(positions: np.ndarray) -> tuple[float, np.ndarray]:
    pairs = np.asarray(list(combinations(range(len(positions)), 2)), dtype=np.int64)
    distances = np.linalg.norm(
        positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1
    )
    return float(distances.min()), distances


def _nonbonded_minimum(
    positions: np.ndarray, bond_index: np.ndarray
) -> float:
    bonded = {tuple(map(int, pair)) for pair in bond_index.T}
    pairs = [
        pair for pair in combinations(range(len(positions)), 2) if pair not in bonded
    ]
    if not pairs:
        return float("inf")
    values = np.asarray(pairs, dtype=np.int64)
    return float(np.linalg.norm(
        positions[values[:, 0]] - positions[values[:, 1]], axis=1
    ).min())


def _bond_ratio_range(
    atomic_numbers: np.ndarray, positions: np.ndarray, bond_index: np.ndarray
) -> tuple[float, float]:
    table = Chem.GetPeriodicTable()
    radii = np.asarray(
        [table.GetRcovalent(int(number)) for number in atomic_numbers],
        dtype=np.float64,
    )
    begin, end = bond_index
    lengths = np.linalg.norm(positions[begin] - positions[end], axis=1)
    expected = radii[begin] + radii[end]
    if np.any(expected <= 0):
        raise ValueError("共价半径缺失；禁止 bond-ratio fallback")
    ratios = lengths / expected
    return float(ratios.min()), float(ratios.max())


def _unconstrained_uff_screen(molecule: Chem.Mol) -> dict[str, Any]:
    working = Chem.Mol(molecule)
    if not AllChem.UFFHasAllMoleculeParams(working):
        return {"supported": False, "status": "unsupported_parameters"}
    force_field = AllChem.UFFGetMoleculeForceField(working, confId=0)
    if force_field is None:
        return {"supported": False, "status": "setup_failed"}
    initial_positions = np.asarray(
        working.GetConformer().GetPositions(), dtype=np.float64
    )
    initial_energy = float(force_field.CalcEnergy())
    optimize_status = int(AllChem.UFFOptimizeMolecule(
        working, maxIters=UFF_MAX_ITERATIONS
    ))
    final_force_field = AllChem.UFFGetMoleculeForceField(working, confId=0)
    if final_force_field is None:
        return {"supported": True, "status": "final_setup_failed"}
    final_positions = np.asarray(
        working.GetConformer().GetPositions(), dtype=np.float64
    )
    final_energy = float(final_force_field.CalcEnergy())
    rmsd = _kabsch_rmsd(initial_positions, final_positions)
    return {
        "supported": True,
        "status": {0: "converged", 1: "not_converged"}.get(
            optimize_status, f"status_{optimize_status}"
        ),
        "initial_energy_kcal_mol": initial_energy,
        "final_energy_kcal_mol": final_energy,
        "energy_change_kcal_mol": final_energy - initial_energy,
        "kabsch_rmsd_angstrom": rmsd,
    }


def audit_validity(input_root: Path, results_root: Path) -> dict[str, Any]:
    generation = json.loads(
        (results_root / "batch_report.json").read_text(encoding="utf-8")
    )
    point_group = json.loads(
        (results_root / "point_group_batch_report.json").read_text(encoding="utf-8")
    )
    if not generation["passed_generation"] or not point_group["passed"]:
        raise RuntimeError("Cn generation/point-group 前置审计未通过")
    dataset = CnGraphDataset(input_root)
    generator = fingerprint_generator(2, 2048, True)
    records: list[dict[str, Any]] = []
    for sample in dataset.records:
        group = sample["metadata"]["Source_Group"]
        record_dir = results_root / group / "records" / sample["molecule_id"]
        source_report = json.loads(
            (record_dir / "report.json").read_text(encoding="utf-8")
        )
        pg_report = json.loads(
            (record_dir / "point_group_report.json").read_text(encoding="utf-8")
        )
        coordinates_path = record_dir / "coordinates.npz"
        if source_report["identity"]["coordinates_sha256"] != _sha256(coordinates_path):
            raise RuntimeError(f"coordinates SHA 不一致: {sample['molecule_id']}")
        with np.load(coordinates_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        graph_exact = all((
            np.array_equal(arrays["atomic_numbers"], sample["atomic_numbers"]),
            np.array_equal(arrays["bond_index"], sample["bond_index"]),
            np.array_equal(arrays["bond_types"], sample["bond_types"]),
        ))
        positions = np.asarray(arrays["final_positions"], dtype=np.float64)
        evaluation_sample = EvaluationSample.from_mapping({
            "molecule_id": sample["molecule_id"],
            "atomic_numbers": sample["atomic_numbers"],
            "formal_charges": sample["formal_charges"],
            "positions": positions,
            "bond_index": sample["bond_index"],
            "bond_types": sample["bond_types"],
            "target_pg": "C3",
            "actual_pg": pg_report["actual_pg"],
            "pg_compatible": pg_report["pg_compatible"],
        })
        chemistry = reconstruct_molecule(evaluation_sample, generator=generator)
        source_molecule = Chem.MolFromSmiles(sample["metadata"]["SMILES"])
        source_canonical = (
            None if source_molecule is None else canonical_graph_smiles(source_molecule)
        )
        smiles_exact = bool(
            chemistry.valid and chemistry.canonical_smiles == source_canonical
        )
        connected = bool(
            chemistry.valid
            and len(Chem.GetMolFrags(chemistry.molecule, asMols=False)) == 1
        )
        finite = bool(np.isfinite(positions).all())
        centroid_norm = float(np.linalg.norm(positions.mean(axis=0)))
        minimum_pair, _ = _pair_distances(positions)
        minimum_nonbonded = _nonbonded_minimum(
            positions, sample["bond_index"]
        )
        minimum_bond_ratio, maximum_bond_ratio = _bond_ratio_range(
            sample["atomic_numbers"], positions, sample["bond_index"]
        )
        uff = (
            _unconstrained_uff_screen(chemistry.molecule)
            if chemistry.valid else {"supported": False, "status": "sanitize_failed"}
        )
        checks = {
            "graph_arrays_exact": graph_exact,
            "rdkit_sanitize_valid": chemistry.valid,
            "connected": connected,
            "source_smiles_exact": smiles_exact,
            "finite_coordinates": finite,
            "centroid_centered": centroid_norm <= 1e-8,
            "all_pair_collision_free_at_0p6": (
                minimum_pair >= COLLISION_THRESHOLD_ANGSTROM
            ),
            "nonbonded_collision_free_at_0p6": (
                minimum_nonbonded >= COLLISION_THRESHOLD_ANGSTROM
            ),
            "no_active_f02_repulsion_pairs": (
                source_report["selected"]["final_active_repulsion_pairs"] == 0
            ),
            "optimizer_termination_acceptable": bool(
                source_report["selected"]["acceptable_optimizer_termination"]
            ),
            "c3_compatible": bool(pg_report["pg_compatible"]),
            "bond_ratio_screen": (
                minimum_bond_ratio >= BOND_RATIO_MIN
                and maximum_bond_ratio <= BOND_RATIO_MAX
            ),
            "uff_supported": bool(uff["supported"]),
            "unconstrained_uff_energy_nonincreasing": bool(
                uff.get("energy_change_kcal_mol", 1.0) <= 1e-7
            ),
            "unconstrained_uff_stability_rmsd": bool(
                uff.get("kabsch_rmsd_angstrom", float("inf"))
                <= UFF_STABILITY_RMSD_ANGSTROM
            ),
        }
        chemical_graph_pass = all(checks[name] for name in (
            "graph_arrays_exact", "rdkit_sanitize_valid", "connected",
            "source_smiles_exact",
        ))
        pipeline_geometry_pass = all(checks[name] for name in (
            "finite_coordinates", "centroid_centered",
            "all_pair_collision_free_at_0p6",
            "nonbonded_collision_free_at_0p6",
            "optimizer_termination_acceptable", "c3_compatible",
        ))
        bond_geometry_pass = bool(checks["bond_ratio_screen"])
        uff_stability_pass = all(checks[name] for name in (
            "uff_supported", "unconstrained_uff_energy_nonincreasing",
            "unconstrained_uff_stability_rmsd",
        ))
        records.append({
            "package_index": sample["package_index"],
            "molecule_id": sample["molecule_id"],
            "source_group": group,
            "smiles": sample["metadata"]["SMILES"],
            "actual_pg": pg_report["actual_pg"],
            "chemical_graph_pass": chemical_graph_pass,
            "pipeline_geometry_pass": pipeline_geometry_pass,
            "bond_geometry_pass": bond_geometry_pass,
            "uff_stability_pass": uff_stability_pass,
            "strict_screen_pass": bool(
                chemical_graph_pass and pipeline_geometry_pass
                and bond_geometry_pass and uff_stability_pass
            ),
            "atom_count": len(sample["atomic_numbers"]),
            "minimum_pair_distance_angstrom": minimum_pair,
            "minimum_nonbonded_distance_angstrom": minimum_nonbonded,
            "minimum_bond_covalent_radius_ratio": minimum_bond_ratio,
            "maximum_bond_covalent_radius_ratio": maximum_bond_ratio,
            "centroid_norm_angstrom": centroid_norm,
            "uff_status": uff["status"],
            "uff_initial_energy_kcal_mol": uff.get("initial_energy_kcal_mol"),
            "uff_final_energy_kcal_mol": uff.get("final_energy_kcal_mol"),
            "uff_relaxation_kabsch_rmsd_angstrom": uff.get("kabsch_rmsd_angstrom"),
            "checks": checks,
        })

    csv_fields = [key for key in records[0] if key != "checks"]
    csv_path = results_root / "validity_records.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows([{key: value[key] for key in csv_fields} for value in records])
    warning_rows = [{
        "molecule_id": value["molecule_id"],
        "source_group": value["source_group"],
        "smiles": value["smiles"],
        "actual_pg": value["actual_pg"],
        "failed_checks": ";".join(
            name for name, passed in value["checks"].items() if not passed
        ),
        "minimum_pair_distance_angstrom": value["minimum_pair_distance_angstrom"],
        "minimum_bond_covalent_radius_ratio": value[
            "minimum_bond_covalent_radius_ratio"
        ],
        "maximum_bond_covalent_radius_ratio": value[
            "maximum_bond_covalent_radius_ratio"
        ],
        "uff_relaxation_kabsch_rmsd_angstrom": value[
            "uff_relaxation_kabsch_rmsd_angstrom"
        ],
    } for value in records if not value["strict_screen_pass"]]
    warning_path = results_root / "validity_warnings.csv"
    with warning_path.open("w", encoding="utf-8", newline="") as handle:
        warning_fields = list(warning_rows[0]) if warning_rows else [
            "molecule_id", "source_group", "smiles", "actual_pg",
            "failed_checks", "minimum_pair_distance_angstrom",
            "minimum_bond_covalent_radius_ratio",
            "maximum_bond_covalent_radius_ratio",
            "uff_relaxation_kabsch_rmsd_angstrom",
        ]
        writer = csv.DictWriter(handle, fieldnames=warning_fields)
        writer.writeheader(); writer.writerows(warning_rows)
    failed_checks = Counter(
        name for record in records for name, passed in record["checks"].items()
        if not passed
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "PASS_OUR_ETFLOW_CN_LAYERED_VALIDITY_SCREEN"
            if all(value["strict_screen_pass"] for value in records)
            else "COMPLETE_OUR_ETFLOW_CN_VALIDITY_WITH_WARNINGS"
        ),
        "molecule_count": len(records),
        "chemical_graph_pass_count": sum(
            value["chemical_graph_pass"] for value in records
        ),
        "pipeline_geometry_pass_count": sum(
            value["pipeline_geometry_pass"] for value in records
        ),
        "bond_geometry_pass_count": sum(
            value["bond_geometry_pass"] for value in records
        ),
        "uff_stability_pass_count": sum(
            value["uff_stability_pass"] for value in records
        ),
        "strict_screen_pass_count": sum(
            value["strict_screen_pass"] for value in records
        ),
        "warning_molecules": warning_rows,
        "failed_check_counts": dict(sorted(failed_checks.items())),
        "thresholds": {
            "collision_distance_angstrom": COLLISION_THRESHOLD_ANGSTROM,
            "bond_to_covalent_radius_sum_ratio": [BOND_RATIO_MIN, BOND_RATIO_MAX],
            "unconstrained_uff_kabsch_rmsd_angstrom": UFF_STABILITY_RMSD_ANGSTROM,
            "unconstrained_uff_max_iterations": UFF_MAX_ITERATIONS,
        },
        "disclaimer": (
            "This is a graph/geometry/UFF screening result, not proof of "
            "thermodynamic stability; xTB/DFT is required for stronger claims."
        ),
        "records": records,
        "identity": {
            "generation_report_sha256": _sha256(results_root / "batch_report.json"),
            "point_group_report_sha256": _sha256(
                results_root / "point_group_batch_report.json"
            ),
            "records_csv_sha256": _sha256(csv_path),
            "warnings_csv_sha256": _sha256(warning_path),
            "auditor_sha256": _sha256(Path(__file__)),
        },
    }
    _atomic_json(results_root / "validity_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    report = audit_validity(args.input_root.resolve(), args.results_root.resolve())
    print(json.dumps({
        key: report[key] for key in (
            "status", "molecule_count", "chemical_graph_pass_count",
            "pipeline_geometry_pass_count", "bond_geometry_pass_count",
            "uff_stability_pass_count", "strict_screen_pass_count",
            "failed_check_counts",
        )
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
