"""M0 oracle reconstruction in stabilizer-aware atom-orbit coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import (
    _angle_cosines,
    _chirality_values,
    _lengths,
    _orbit_mean,
    _torsion_sincos,
    build_geometry_contract,
)
from .group import operation_error_numpy
from .metrics import collision_audit, kabsch_rmsd, pair_distance_mae
from .orbit_kinematics import (
    build_orbit_parameterization,
    encode_orbit_parameters,
    lift_orbit_parameters,
)
from .overfit import _panel_samples


M0_RUN_SCHEMA_VERSION = "pg-orbitflow-m0-oracle-reconstruction-v1"

_COVALENT_RADII = {
    1: 0.31,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    35: 1.20,
    50: 1.40,
    53: 1.39,
}


@dataclass(frozen=True)
class OracleTargets:
    bond_lengths: np.ndarray
    angle_cosines: np.ndarray
    torsion_sincos: np.ndarray
    local_pair_lengths: np.ndarray
    ring_lengths: np.ndarray
    chirality: np.ndarray


def _broadcast_orbit_mean(values: np.ndarray, orbit_id: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for current in np.unique(orbit_id):
        mask = orbit_id == current
        mean = np.asarray(values[mask], dtype=np.float64).mean(axis=0)
        if mean.ndim and mean.shape == (2,):
            norm = float(np.linalg.norm(mean))
            if norm < 1e-8:
                raise ValueError("torsion orbit has an undefined circular mean")
            mean = mean / norm
        result[mask] = mean
    return result


def build_oracle_targets(sample, contract) -> OracleTargets:
    import torch

    target = torch.as_tensor(sample.symmetric_target_angstrom, dtype=torch.float64)

    def indices(values):
        return torch.as_tensor(values, dtype=torch.long)

    bond = _lengths(target, indices(contract.bond_index)).numpy()
    angle = _angle_cosines(target, indices(contract.angle_index)).numpy()
    torsion = _torsion_sincos(target, indices(contract.torsion_index)).numpy()
    local_pair = _lengths(target, indices(contract.local_pair_index)).numpy()
    chirality = _chirality_values(target, indices(contract.chirality_index)).numpy()
    ring = bond[contract.ring_bond_indices]
    return OracleTargets(
        bond_lengths=_broadcast_orbit_mean(bond, contract.bond_orbit_id),
        angle_cosines=_broadcast_orbit_mean(angle, contract.angle_orbit_id),
        torsion_sincos=_broadcast_orbit_mean(torsion, contract.torsion_orbit_id),
        local_pair_lengths=_broadcast_orbit_mean(
            local_pair, contract.local_pair_orbit_id
        ),
        ring_lengths=_broadcast_orbit_mean(ring, contract.ring_bond_orbit_id),
        # Chirality is an oriented pseudoscalar tied to the canonical neighbor
        # ordering.  Symmetry-related centers can legitimately carry opposite
        # signs, so averaging it as an invariant scalar would erase or corrupt
        # the target.  Keep the per-center canonical values; orbit balancing is
        # still applied to the squared errors in ``oracle_energy``.
        chirality=chirality,
    )


def _nonbonded_pairs(sample) -> tuple[np.ndarray, np.ndarray]:
    atom_count = len(sample.atomic_numbers)
    bonded = {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in np.asarray(sample.bond_index).T
    }
    pairs = [
        (left, right)
        for left in range(atom_count)
        for right in range(left + 1, atom_count)
        if (left, right) not in bonded
    ]
    radii = np.asarray(
        [_COVALENT_RADII[int(number)] for number in sample.atomic_numbers],
        dtype=np.float64,
    )
    thresholds = np.asarray(
        [0.60 * (radii[left] + radii[right]) for left, right in pairs],
        dtype=np.float64,
    )
    return np.asarray(pairs, dtype=np.int64).T, thresholds


def oracle_energy(coordinates, sample, contract, targets, weights: dict) -> dict:
    import torch

    device = coordinates.device

    def indices(values):
        return torch.as_tensor(values, dtype=torch.long, device=device)

    def target(values):
        return torch.as_tensor(values, dtype=coordinates.dtype, device=device)

    bond = _lengths(coordinates, indices(contract.bond_index))
    angle = _angle_cosines(coordinates, indices(contract.angle_index))
    torsion = _torsion_sincos(coordinates, indices(contract.torsion_index))
    local_pair = _lengths(coordinates, indices(contract.local_pair_index))
    bond_loss = _orbit_mean(
        torch.square(bond - target(targets.bond_lengths)), contract.bond_orbit_id
    )
    angle_loss = _orbit_mean(
        torch.square(angle - target(targets.angle_cosines)), contract.angle_orbit_id
    )
    if torsion.numel():
        torsion_loss = _orbit_mean(
            1.0
            - torch.sum(torsion * target(targets.torsion_sincos), dim=-1).clamp(
                -1.0, 1.0
            ),
            contract.torsion_orbit_id,
        )
    else:
        torsion_loss = torch.zeros((), dtype=coordinates.dtype, device=device)
    local_pair_loss = _orbit_mean(
        torch.square(local_pair - target(targets.local_pair_lengths)),
        contract.local_pair_orbit_id,
    )
    if len(contract.ring_bond_indices):
        ring_values = bond[
            torch.as_tensor(
                contract.ring_bond_indices, dtype=torch.long, device=device
            )
        ]
        ring_loss = _orbit_mean(
            torch.square(ring_values - target(targets.ring_lengths)),
            contract.ring_bond_orbit_id,
        )
    else:
        ring_loss = torch.zeros((), dtype=coordinates.dtype, device=device)
    chirality_index = indices(contract.chirality_index)
    chirality = _chirality_values(coordinates, chirality_index)
    target_chirality = target(targets.chirality)
    active = torch.abs(target_chirality) >= 0.05
    if bool(active.any()):
        chirality_loss = _orbit_mean(
            torch.square(chirality[active] - target_chirality[active]),
            contract.chirality_orbit_id[active.detach().cpu().numpy()],
        )
    else:
        chirality_loss = torch.zeros((), dtype=coordinates.dtype, device=device)
    pair_index, thresholds = _nonbonded_pairs(sample)
    nonbonded = _lengths(coordinates, indices(pair_index))
    collision_loss = torch.mean(
        torch.square(torch.relu(target(thresholds) - nonbonded))
    )
    terms = {
        "bond": bond_loss,
        "angle": angle_loss,
        "torsion": torsion_loss,
        "local_pair": local_pair_loss,
        "ring": ring_loss,
        "chirality": chirality_loss,
        "collision": collision_loss,
    }
    total = sum(float(weights[name]) * value for name, value in terms.items())
    return {"total": total, **terms}


def _torsion_mae_degrees(coordinates, contract, targets) -> float:
    import torch

    if not contract.torsion_index.shape[1]:
        return 0.0
    values = _torsion_sincos(
        torch.as_tensor(coordinates, dtype=torch.float64),
        torch.as_tensor(contract.torsion_index, dtype=torch.long),
    ).numpy()
    target_values = targets.torsion_sincos
    cosine = np.sum(values * target_values, axis=-1).clip(-1.0, 1.0)
    sine = values[:, 0] * target_values[:, 1] - values[:, 1] * target_values[:, 0]
    return float(np.mean(np.abs(np.rad2deg(np.arctan2(sine, cosine)))))


def reconstruction_metrics(sample, contract, targets, coordinates) -> dict:
    import torch

    values = np.asarray(coordinates, dtype=np.float64)
    candidate = torch.as_tensor(values, dtype=torch.float64)
    bond = _lengths(
        candidate, torch.as_tensor(contract.bond_index, dtype=torch.long)
    ).numpy()
    angles = _angle_cosines(
        candidate, torch.as_tensor(contract.angle_index, dtype=torch.long)
    ).numpy()
    torsions = _torsion_sincos(
        candidate, torch.as_tensor(contract.torsion_index, dtype=torch.long)
    ).numpy()
    angle_error = np.rad2deg(
        np.abs(
            np.arccos(angles.clip(-1.0, 1.0))
            - np.arccos(targets.angle_cosines.clip(-1.0, 1.0))
        )
    )
    ring_error = (
        float(
            np.mean(
                np.abs(
                    bond[contract.ring_bond_indices] - targets.ring_lengths
                )
            )
        )
        if len(contract.ring_bond_indices)
        else 0.0
    )
    collision = collision_audit(values, sample.atomic_numbers, sample.bond_index)
    operation = operation_error_numpy(
        values, sample.operation_matrices, sample.permutation_index
    )
    torsion_cosine = np.sum(torsions * targets.torsion_sincos, axis=-1).clip(
        -1.0, 1.0
    )
    torsion_sine = (
        torsions[:, 0] * targets.torsion_sincos[:, 1]
        - torsions[:, 1] * targets.torsion_sincos[:, 0]
    )
    torsion_errors = np.abs(
        np.rad2deg(np.arctan2(torsion_sine, torsion_cosine))
    )
    target_angles = np.abs(
        np.rad2deg(
            np.arctan2(
                targets.torsion_sincos[:, 0], targets.torsion_sincos[:, 1]
            )
        )
    )
    nonplanar = np.minimum(target_angles, np.abs(180.0 - target_angles)) > 5.0
    chirality = _chirality_values(
        candidate, torch.as_tensor(contract.chirality_index, dtype=torch.long)
    ).numpy()
    active_chirality = np.abs(targets.chirality) >= 0.05
    chirality_preserved = (
        float(np.mean(chirality[active_chirality] * targets.chirality[active_chirality] > 0))
        if bool(np.any(active_chirality))
        else 1.0
    )
    return {
        "bond_mae_angstrom": float(np.mean(np.abs(bond - targets.bond_lengths))),
        "angle_mae_degrees": float(np.mean(angle_error)),
        "torsion_circular_mae_degrees": _torsion_mae_degrees(
            values, contract, targets
        ),
        "nonplanar_torsion_count": int(np.sum(nonplanar)),
        "nonplanar_torsion_circular_mae_degrees": (
            float(np.mean(torsion_errors[nonplanar])) if bool(np.any(nonplanar)) else 0.0
        ),
        "active_chirality_count": int(np.sum(active_chirality)),
        "chirality_preserved_fraction": chirality_preserved,
        "ring_closure_mae_angstrom": ring_error,
        "kabsch_rmsd_angstrom": kabsch_rmsd(
            sample.symmetric_target_angstrom, values
        ),
        "pair_distance_mae_angstrom": pair_distance_mae(
            sample.symmetric_target_angstrom, values
        ),
        **collision,
        **operation,
    }


def optimize_one_start(
    sample,
    contract,
    targets,
    specification,
    protocol: dict,
    *,
    seed: int,
    device: str,
) -> tuple[dict, np.ndarray]:
    import torch

    rng = np.random.default_rng(int(seed))
    scale = float(protocol["initialization"]["coordinate_standard_deviation_angstrom"])
    initial = rng.normal(0.0, scale, size=specification.parameter_count)
    parameters = torch.nn.Parameter(
        torch.as_tensor(initial, dtype=torch.float64, device=device)
    )
    setting = protocol["optimization"]
    optimizer = torch.optim.Adam(
        [parameters], lr=float(setting["adam_learning_rate"])
    )
    all_finite = True
    for _ in range(int(setting["adam_steps"])):
        optimizer.zero_grad(set_to_none=True)
        coordinates = lift_orbit_parameters(parameters, specification)
        energy = oracle_energy(
            coordinates, sample, contract, targets, protocol["energy_weights"]
        )["total"]
        if not bool(torch.isfinite(energy)):
            all_finite = False
            break
        energy.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            [parameters], float(setting["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient))):
            all_finite = False
            break
        optimizer.step()
    if all_finite:
        optimizer_lbfgs = torch.optim.LBFGS(
            [parameters],
            lr=float(setting["lbfgs_learning_rate"]),
            max_iter=int(setting["lbfgs_max_iterations"]),
            history_size=int(setting["lbfgs_history_size"]),
            line_search_fn=setting["line_search"],
            tolerance_grad=1e-12,
            tolerance_change=1e-15,
        )

        def closure():
            optimizer_lbfgs.zero_grad(set_to_none=True)
            current = oracle_energy(
                lift_orbit_parameters(parameters, specification),
                sample,
                contract,
                targets,
                protocol["energy_weights"],
            )["total"]
            current.backward()
            return current

        optimizer_lbfgs.step(closure)
    with torch.no_grad():
        coordinates = lift_orbit_parameters(parameters, specification)
        terms = oracle_energy(
            coordinates, sample, contract, targets, protocol["energy_weights"]
        )
    values = coordinates.detach().cpu().numpy()
    record = {
        "seed": int(seed),
        "all_values_finite": bool(
            all_finite
            and np.isfinite(values).all()
            and all(torch.isfinite(value) for value in terms.values())
        ),
        "final_energy": float(terms["total"].item()),
        "energy_terms": {
            key: float(value.item()) for key, value in terms.items() if key != "total"
        },
        **reconstruction_metrics(sample, contract, targets, values),
    }
    return record, values


def _dump(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-m0-oracle-protocol-v1":
        raise ValueError("unsupported M0 protocol schema")
    package_dir = Path(protocol["package_dir"])
    if _sha256(package_dir / "manifest.json") != protocol["canonical_manifest_sha256"]:
        raise ValueError("canonical manifest changed after M0 freeze")
    base = Path(protocol["base_c0_protocol"]["path"])
    if _sha256(base) != protocol["base_c0_protocol"]["sha256"]:
        raise ValueError("formal C0 protocol changed after M0 freeze")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    import torch

    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True, warn_only=False)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    samples = _panel_samples(protocol, tier)
    records = []
    all_start_records = []
    coordinate_parts = []
    offsets = [0]
    starts = int(protocol["initialization"]["starts_per_molecule"])
    stride = int(protocol["initialization"]["molecule_specific_seed_stride"])
    for sample in samples:
        contract = build_geometry_contract(sample)
        targets = build_oracle_targets(sample, contract)
        specification = build_orbit_parameterization(sample)
        encoded = encode_orbit_parameters(
            sample.symmetric_target_angstrom, specification
        )
        roundtrip = lift_orbit_parameters(
            torch.as_tensor(encoded, dtype=torch.float64, device=args.device),
            specification,
        ).detach().cpu().numpy()
        roundtrip_error = float(
            np.max(np.abs(roundtrip - sample.symmetric_target_angstrom))
        )
        candidate_records = []
        candidate_coordinates = []
        for start in range(starts):
            seed = int(protocol["seed"]) + sample.package_index * stride + start
            record, coordinates = optimize_one_start(
                sample,
                contract,
                targets,
                specification,
                protocol,
                seed=seed,
                device=args.device,
            )
            record["start_index"] = start
            candidate_records.append(record)
            candidate_coordinates.append(coordinates)
        best_index = min(
            range(starts), key=lambda index: candidate_records[index]["final_energy"]
        )
        best = dict(candidate_records[best_index])
        best.update(
            {
                "package_index": sample.package_index,
                "molecule_id": sample.molecule_id,
                "target_pg": sample.target_pg,
                "atom_count": len(sample.atom_types),
                "atom_orbit_count": specification.orbit_count,
                "orbit_parameter_count": specification.parameter_count,
                "fixed_dimensions": specification.fixed_dimensions.tolist(),
                "roundtrip_max_abs_error_angstrom": roundtrip_error,
                "selected_start_index": best_index,
                "successful_start_count": int(
                    sum(record["all_values_finite"] for record in candidate_records)
                ),
            }
        )
        records.append(best)
        all_start_records.append(
            {
                "package_index": sample.package_index,
                "records": candidate_records,
            }
        )
        coordinates = candidate_coordinates[best_index].astype(np.float32)
        coordinate_parts.append(coordinates)
        offsets.append(offsets[-1] + len(coordinates))
        print(
            f"package_index={sample.package_index} PG={sample.target_pg} "
            f"energy={best['final_energy']:.6g} "
            f"bond={best['bond_mae_angstrom']:.6g}A "
            f"angle={best['angle_mae_degrees']:.6g}deg "
            f"torsion={best['torsion_circular_mae_degrees']:.6g}deg "
            f"rmsd={best['kabsch_rmsd_angstrom']:.6g}A",
            flush=True,
        )
    gate = protocol["gate"]
    values = {
        "roundtrip_max_abs_error_angstrom": max(
            record["roundtrip_max_abs_error_angstrom"] for record in records
        ),
        "bond_mae_max_angstrom": max(record["bond_mae_angstrom"] for record in records),
        "angle_mae_max_degrees": max(record["angle_mae_degrees"] for record in records),
        "torsion_circular_mae_max_degrees": max(
            record["torsion_circular_mae_degrees"] for record in records
        ),
        "ring_closure_mae_max_angstrom": max(
            record["ring_closure_mae_angstrom"] for record in records
        ),
        "kabsch_rmsd_max_angstrom": max(
            record["kabsch_rmsd_angstrom"] for record in records
        ),
        "collision_free_fraction": float(
            np.mean([record["collision_free"] for record in records])
        ),
        "max_operation_atom_error_angstrom": max(
            record["max_atom_error_angstrom"] for record in records
        ),
        "all_optimization_values_finite": all(
            record["all_values_finite"] for record in records
        ),
    }
    checks = {
        "roundtrip_max_abs_error": values["roundtrip_max_abs_error_angstrom"]
        <= gate["roundtrip_max_abs_error_max_angstrom"],
        "bond_mae": values["bond_mae_max_angstrom"]
        <= gate["bond_mae_max_angstrom"],
        "angle_mae": values["angle_mae_max_degrees"]
        <= gate["angle_mae_max_degrees"],
        "torsion_mae": values["torsion_circular_mae_max_degrees"]
        <= gate["torsion_circular_mae_max_degrees"],
        "ring_closure": values["ring_closure_mae_max_angstrom"]
        <= gate["ring_closure_mae_max_angstrom"],
        "kabsch_rmsd": values["kabsch_rmsd_max_angstrom"]
        <= gate["kabsch_rmsd_max_angstrom"],
        "collision_free": values["collision_free_fraction"]
        >= gate["collision_free_fraction_min"],
        "group_action": values["max_operation_atom_error_angstrom"]
        <= gate["max_operation_atom_error_max_angstrom"],
        "finite": values["all_optimization_values_finite"]
        is gate["all_optimization_values_finite"],
        "target_cartesian_initialization_never_used": True,
        "target_cartesian_coordinate_loss_never_used": True,
        "posthoc_hard_projection_never_used": True,
    }
    passed = all(checks.values())
    status = (
        "PASS_M0_ORACLE_ADVANCE_TO_M1_ORBIT_IC_HEADS"
        if passed
        else "FAIL_M0_ORACLE_STOP_BEFORE_M1"
    )
    np.savez_compressed(
        output_dir / "coordinates.npz",
        coordinates=np.concatenate(coordinate_parts, axis=0),
        atom_offsets=np.asarray(offsets, dtype=np.int64),
        package_indices=np.asarray(
            [record["package_index"] for record in records], dtype=np.int64
        ),
    )
    _dump(output_dir / "all_starts.json", all_start_records)
    report = {
        "schema_version": M0_RUN_SCHEMA_VERSION,
        "status": status,
        "passed": passed,
        "claim_scope": "oracle internal-coordinate reconstruction only",
        "quality_claim": False,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "molecule_count": len(records),
        "records": records,
        "gate": {"passed": passed, "checks": checks, "values": values, "thresholds": gate},
        "decision": protocol["progression"]["if_passes" if passed else "if_fails"],
        "isolation": protocol["isolation"],
    }
    _dump(output_dir / "report.json", report)
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _dump(
        output_dir / "manifest.json",
        {"schema_version": "pg-orbitflow-m0-manifest-v1", "artifacts": artifacts},
    )
    print(f"{status} report={output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
