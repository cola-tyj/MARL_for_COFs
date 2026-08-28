"""Generate a Target_PG 3D conformer from a canonical known graph.

This practical route uses ET-Flow only as a learned initial-conformer prior,
recovers a C2/C3 graph action without coordinates, applies the validated E3
symmetry projection, and refines inside the orbit subspace with the F0.2
UFF+repulsion backend.  It is deliberately reported separately from raw
learned Target_PG generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from generative_model.conformer.etflow_bridge import ETFlowRuntime
from generative_model.conformer.etflow_ef1_projection import project_etflow_ef1
from generative_model.data.build_cof_package import save_deterministic_npz
from generative_model.data.cof_graph_dataset import COFGraphDataset
from generative_model.models.graph_pg_3d_projection import (
    minimum_pair_distance,
    operation_errors,
)
from generative_model.optimization.orbit_force_field_repulsion import (
    optimize_orbit_uff_with_repulsion,
)
from generative_model.symmetry.graph_action import recover_cyclic_graph_action


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT / "generative_model/data/processed/v2"
DEFAULT_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_protocol_v1.json"
DEFAULT_CACHE = ROOT / "generative_model/checkpoints/etflow"
DEFAULT_OUTPUT = ROOT / "generative_model/runs/etflow_e3f02_inference"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("protocol_fingerprint", None)
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    def encode(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, default=encode
    ) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _symbols(atomic_numbers: np.ndarray) -> list[str]:
    table = Chem.GetPeriodicTable()
    result = [table.GetElementSymbol(int(number)) for number in atomic_numbers]
    if any(not value for value in result):
        raise ValueError("atomic number 无法严格转为元素符号")
    return result


def _write_xyz(
    path: Path,
    symbols: list[str],
    positions: np.ndarray,
    comment: str,
) -> None:
    values = np.asarray(positions, dtype=np.float64)
    if values.shape != (len(symbols), 3) or not np.isfinite(values).all():
        raise ValueError("XYZ coordinates 必须为 finite [N,3]")
    lines = [str(len(symbols)), comment]
    lines.extend(
        f"{symbol:<2s} {x: .10f} {y: .10f} {z: .10f}"
        for symbol, (x, y, z) in zip(symbols, values, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_from_canonical(
    canonical: dict[str, Any], target_pg: str
) -> dict[str, Any]:
    action = recover_cyclic_graph_action(
        canonical["atomic_numbers"],
        canonical["formal_charges"],
        canonical["radical_electrons"],
        canonical["bond_index"],
        canonical["bond_types"],
        str(target_pg),
    )
    aromaticity = np.zeros(len(canonical["atomic_numbers"]), dtype=np.bool_)
    for (left, right), kind in zip(
        canonical["bond_index"].T, canonical["bond_types"], strict=True
    ):
        if int(kind) == 4:
            aromaticity[int(left)] = True
            aromaticity[int(right)] = True
    return {
        "package_index": int(canonical["package_index"]),
        "molecule_id": str(canonical["molecule_id"]),
        "target_pg": str(target_pg),
        "atomic_numbers": np.asarray(canonical["atomic_numbers"], dtype=np.int64),
        "formal_charges": np.asarray(canonical["formal_charges"], dtype=np.int64),
        "radical_electrons": np.asarray(canonical["radical_electrons"], dtype=np.int64),
        "bond_index": np.asarray(canonical["bond_index"], dtype=np.int64),
        "bond_types": np.asarray(canonical["bond_types"], dtype=np.int64),
        "aromaticity": aromaticity,
        "target_operation_matrices": action["operation_matrices"],
        "target_permutation_index": action["permutation_index"],
        "target_orbit_id": action["orbit_id"],
        "target_orbit_size": action["orbit_size"],
        "graph_action_audit": {
            "recovery": action["recovery_audit"],
            "validation": action["validation"],
        },
    }


def _finish_candidate(
    sample: dict[str, Any],
    raw_positions: np.ndarray,
    candidate_id: int,
    source_seed: int,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    projection = project_etflow_ef1(
        raw_positions,
        sample,
        match_radius_of_gyration=bool(
            protocol["projection"]["match_radius_of_gyration"]
        ),
    )
    projected = projection["positions"]
    optimizer = protocol["optimizer"]
    optimized = optimize_orbit_uff_with_repulsion(
        sample,
        projected,
        cutoff_angstrom=float(optimizer["repulsion_cutoff_angstrom"]),
        force_constant_kcal_mol_angstrom2=float(
            optimizer["repulsion_force_constant_kcal_mol_angstrom2"]
        ),
        maximum_iterations=int(optimizer["maximum_iterations"]),
        function_tolerance=float(optimizer["function_tolerance"]),
        gradient_tolerance=float(optimizer["gradient_tolerance"]),
    )
    final = optimized.pop("positions")
    projected_errors = projection["operation_errors"]
    final_errors = operation_errors(
        final,
        sample["target_operation_matrices"],
        sample["target_permutation_index"],
    )
    threshold = protocol["gate_thresholds"]
    acceptable_termination = bool(
        optimized["optimizer_success"]
        or optimized["final_projected_gradient_norm"]
        <= float(threshold["acceptable_gradient_norm_max"])
    )
    final_minimum = minimum_pair_distance(final)
    passed = bool(
        acceptable_termination
        and final_minimum >= float(threshold["minimum_pair_distance_angstrom"])
        and final_errors["max_atom_error_angstrom"]
        <= float(threshold["maximum_operation_error_angstrom"])
    )
    record = {
        "candidate_id": int(candidate_id),
        "source_seed": int(source_seed),
        "passed_geometry_gate": passed,
        "acceptable_optimizer_termination": acceptable_termination,
        "raw_minimum_pair_distance_angstrom": minimum_pair_distance(raw_positions),
        "projected_minimum_pair_distance_angstrom": minimum_pair_distance(projected),
        "final_minimum_pair_distance_angstrom": final_minimum,
        "projection_rmsd_angstrom": float(projection["projection_rmsd_angstrom"]),
        "orientation_id": int(projection["orientation_id"]),
        "projected_maximum_operation_error_angstrom": float(
            projected_errors["max_atom_error_angstrom"]
        ),
        "final_maximum_operation_error_angstrom": float(
            final_errors["max_atom_error_angstrom"]
        ),
        "resonance_audit": projection["resonance_audit"],
        **optimized,
    }
    return record, {
        "raw": np.asarray(raw_positions, dtype="<f8"),
        "projected": np.asarray(projected, dtype="<f8"),
        "final": np.asarray(final, dtype="<f8"),
    }


def _select_candidate(records: list[dict[str, Any]]) -> int:
    eligible = [index for index, value in enumerate(records) if value["passed_geometry_gate"]]
    if not eligible:
        raise RuntimeError("所有 ET-Flow candidates 均未通过 E3/F0.2 geometry Gate")
    return min(eligible, key=lambda index: (
        records[index]["final_total_objective_kcal_mol"],
        records[index]["candidate_id"],
    ))


def generate(
    *,
    package: Path,
    package_index: int,
    target_pg: str,
    seed: int,
    protocol_path: Path,
    output_dir: Path,
    cache: Path,
    device: str,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_fingerprint"] != _fingerprint(protocol):
        raise RuntimeError("ET-Flow E3/F0.2 inference protocol fingerprint 不一致")
    for relative, expected in protocol["identity"]["source_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"inference source hash 不一致: {relative}")
    checkpoint = cache / "drugs-o3.ckpt"
    if _sha256(checkpoint) != protocol["identity"]["checkpoint_sha256"]:
        raise RuntimeError("ET-Flow drugs-o3 checkpoint SHA 不一致")
    if target_pg not in protocol["supported_target_pgs"]:
        raise ValueError(f"当前严格推理仅支持 {protocol['supported_target_pgs']}")

    dataset = COFGraphDataset(package)
    canonical = dataset[int(package_index)]
    sample = _sample_from_canonical(canonical, target_pg)
    runtime = ETFlowRuntime(
        model_name=str(protocol["etflow"]["model_name"]),
        device=device,
        cache=str(cache),
    )
    prediction = runtime.predict(
        sample,
        num_samples=int(protocol["etflow"]["candidate_count"]),
        n_timesteps=int(protocol["etflow"]["n_timesteps"]),
        seed=int(seed),
    )
    records = []
    coordinate_records = []
    failures = []
    for candidate_id, raw in enumerate(prediction["positions"]):
        try:
            record, coordinates = _finish_candidate(
                sample, raw, candidate_id, int(seed), protocol
            )
            records.append(record)
            coordinate_records.append(coordinates)
        except Exception as error:
            failures.append({
                "candidate_id": int(candidate_id),
                "failure": f"{type(error).__name__}: {error}",
            })
    selected_index = _select_candidate(records)
    selected_record = records[selected_index]
    selected = coordinate_records[selected_index]
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "atomic_numbers": sample["atomic_numbers"].astype("<i8"),
        "bond_index": sample["bond_index"].astype("<i8"),
        "bond_types": sample["bond_types"].astype("<i8"),
        "operation_matrices": np.asarray(
            sample["target_operation_matrices"], dtype="<f8"
        ),
        "permutation_index": np.asarray(
            sample["target_permutation_index"], dtype="<i8"
        ),
        "orbit_id": np.asarray(sample["target_orbit_id"], dtype="<i8"),
        "orbit_size": np.asarray(sample["target_orbit_size"], dtype="<i8"),
        "raw_positions": selected["raw"],
        "projected_positions": selected["projected"],
        "final_positions": selected["final"],
    }
    coordinates_path = output_dir / "coordinates.npz"
    save_deterministic_npz(coordinates_path, arrays)
    symbols = _symbols(sample["atomic_numbers"])
    for stage in ("raw", "projected", "final"):
        _write_xyz(
            output_dir / f"{stage}.xyz",
            symbols,
            selected[stage],
            (
                f"molecule_id={sample['molecule_id']} package_index={package_index} "
                f"target_pg={target_pg} stage={stage} seed={seed}"
            ),
        )
    report = {
        "schema_version": "etflow-e3f02-known-graph-inference-v1",
        "status": "PASS_ETFLOW_E3F02_XYZ_AWAIT_INDEPENDENT_POINT_GROUP_AUDIT",
        "passed_geometry_gate": True,
        "package_index": int(package_index),
        "molecule_id": sample["molecule_id"],
        "source_split": str(canonical["metadata"]["Split_IID"]),
        "requested_target_pg": str(target_pg),
        "canonical_dataset_target_pg": str(canonical["metadata"]["Target_PG"]),
        "atom_count": len(sample["atomic_numbers"]),
        "sn_count": int(np.sum(sample["atomic_numbers"] == 50)),
        "seed": int(seed),
        "candidate_count_requested": int(protocol["etflow"]["candidate_count"]),
        "candidate_count_completed": len(records),
        "selected_candidate_id": selected_record["candidate_id"],
        "candidate_selection": (
            "minimum final orbit-constrained UFF+repulsion objective among "
            "strict geometry-passing candidates, then candidate id"
        ),
        "candidate_records": records,
        "candidate_failures": failures,
        "selected": selected_record,
        "bridge_audit": prediction["bridge_audit"],
        "feature_audit": prediction["feature_audit"],
        "graph_action_audit": sample["graph_action_audit"],
        "scope": {
            "known_canonical_graph": True,
            "requested_target_pg": True,
            "reference_xyz_used": False,
            "stored_symmetry_action_used": False,
            "etkdg_used": False,
            "etflow_used_as_initial_conformer_prior": True,
            "hard_symmetry_projection_used": True,
            "orbit_constrained_force_field_used": True,
            "raw_learned_target_pg_claim": False,
        },
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "checkpoint_sha256": _sha256(checkpoint),
            "coordinates_sha256": _sha256(coordinates_path),
        },
        "outputs": {
            "coordinates_npz": "coordinates.npz",
            "raw_xyz": "raw.xyz",
            "projected_xyz": "projected.xyz",
            "final_xyz": "final.xyz",
        },
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--package-index", type=int, required=True)
    parser.add_argument("--target-pg", choices=("C2", "C3"), required=True)
    parser.add_argument("--seed", type=int, default=2026082701)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = generate(
        package=args.package.resolve(),
        package_index=int(args.package_index),
        target_pg=str(args.target_pg),
        seed=int(args.seed),
        protocol_path=args.protocol.resolve(),
        output_dir=args.output_dir.resolve(),
        cache=args.cache.resolve(),
        device=str(args.device),
    )
    print(json.dumps({
        "status": report["status"],
        "package_index": report["package_index"],
        "requested_target_pg": report["requested_target_pg"],
        "selected_candidate_id": report["selected_candidate_id"],
        "final_minimum_pair_distance_angstrom": report["selected"][
            "final_minimum_pair_distance_angstrom"
        ],
        "final_maximum_operation_error_angstrom": report["selected"][
            "final_maximum_operation_error_angstrom"
        ],
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
