"""Paired, reference-free route helpers for the our_ET_Flow ablation.

The four routes share one canonical graph, one base seed and one requested
point group.  Reference coordinates are accepted only by the metric function;
they are never used to generate or select a candidate.
"""

from __future__ import annotations

import hashlib
from time import perf_counter
from typing import Any

import numpy as np
from rdkit.Chem import AllChem

from generative_model.conformer.etflow_ef1_projection import (
    kabsch_rmsd,
    pair_distance_mae,
    project_etflow_ef1,
)
from generative_model.inference.generate_etflow_symmetric_xyz import (
    _finish_candidate,
    _select_candidate,
)
from generative_model.inference.generate_etflow_symmetric_xyz_v3 import (
    projection_rank_key,
)
from generative_model.inference.s4_inertia_selection import (
    mass_weighted_inertia_separation,
    select_s4_inertia_stable_candidate,
)
from generative_model.models.graph_pg_3d_projection import (
    bond_length_mae,
    minimum_pair_distance,
    operation_errors,
)
from generative_model.optimization.etkdg_orbit_initializer import embed_etkdg
from generative_model.optimization.force_field_support import (
    build_strict_rdkit_molecule,
)


ROUTES = (
    "etkdg_v3_best_of_n",
    "etflow_raw",
    "etflow_hard_projection",
    "etflow_hard_projection_f02",
)


def derived_seed(base_seed: int, route: str, candidate_id: int) -> int:
    """Derive a stable signed-32-bit seed without Python hash randomization."""

    if candidate_id == 0:
        return int(base_seed) % (2**31 - 1)
    payload = f"{int(base_seed)}\0{route}\0{int(candidate_id)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def single_point_uff_energy(
    sample: dict[str, Any], positions: np.ndarray
) -> float:
    """Return a strict UFF single-point energy without geometry optimization."""

    molecule = build_strict_rdkit_molecule({**sample, "positions": positions})
    if not AllChem.UFFHasAllMoleculeParams(molecule):
        raise RuntimeError("UFF parameters incomplete; fallback is forbidden")
    force_field = AllChem.UFFGetMoleculeForceField(molecule, confId=0)
    if force_field is None:
        raise RuntimeError("UFF force-field setup failed")
    energy = float(force_field.CalcEnergy())
    if not np.isfinite(energy):
        raise RuntimeError("UFF energy is not finite")
    return energy


def _pair_counts(positions: np.ndarray) -> tuple[int, int]:
    values = np.asarray(positions, dtype=np.float64)
    distances = np.linalg.norm(
        values[:, None, :] - values[None, :, :], axis=-1
    )[np.triu_indices(len(values), k=1)]
    return int(np.sum(distances < 0.6)), int(np.sum(distances < 0.8))


def candidate_diversity(candidates: list[np.ndarray]) -> dict[str, Any]:
    """Internal diversity for candidates of the same immutable graph."""

    values = [np.asarray(value, dtype=np.float64) for value in candidates]
    kabsch, pair_mae = [], []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            kabsch.append(kabsch_rmsd(values[left], values[right]))
            pair_mae.append(pair_distance_mae(values[left], values[right]))
    return {
        "candidate_count": len(values),
        "pair_count": len(kabsch),
        "mean_pairwise_kabsch_rmsd_angstrom": (
            None if not kabsch else float(np.mean(kabsch))
        ),
        "mean_pairwise_distance_mae_angstrom": (
            None if not pair_mae else float(np.mean(pair_mae))
        ),
    }


def route_geometry_metrics(
    sample: dict[str, Any], positions: np.ndarray, reference_positions: np.ndarray
) -> dict[str, Any]:
    """Metrics that do not affect candidate generation or selection."""

    values = np.asarray(positions, dtype=np.float64)
    reference = np.asarray(reference_positions, dtype=np.float64)
    energy = single_point_uff_energy(sample, values)
    return {
        "minimum_pair_distance_angstrom": minimum_pair_distance(values),
        "collision_free_at_0p6": bool(minimum_pair_distance(values) >= 0.6),
        "bond_length_mae_to_canonical_angstrom": bond_length_mae(
            reference, values, np.asarray(sample["bond_index"], dtype=np.int64)
        ),
        "kabsch_rmsd_to_canonical_angstrom": kabsch_rmsd(reference, values),
        "pair_distance_mae_to_canonical_angstrom": pair_distance_mae(
            reference, values
        ),
        "uff_single_point_energy_kcal_mol": energy,
        "uff_single_point_energy_per_atom_kcal_mol": energy / len(values),
    }


def _lowest_energy_candidate(
    sample: dict[str, Any], candidates: list[np.ndarray]
) -> tuple[int, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for candidate_id, positions in enumerate(candidates):
        try:
            energy = single_point_uff_energy(sample, positions)
            records.append({
                "candidate_id": candidate_id,
                "success": True,
                "uff_single_point_energy_kcal_mol": energy,
            })
        except Exception as error:
            records.append({
                "candidate_id": candidate_id,
                "success": False,
                "failure": f"{type(error).__name__}: {error}",
            })
    eligible = [record for record in records if record["success"]]
    if not eligible:
        raise RuntimeError("all candidates failed strict UFF single-point setup")
    selected = min(
        eligible,
        key=lambda record: (
            float(record["uff_single_point_energy_kcal_mol"]),
            int(record["candidate_id"]),
        ),
    )
    return int(selected["candidate_id"]), records


def run_etkdg_route(
    sample: dict[str, Any], *, base_seed: int, settings: dict[str, Any]
) -> dict[str, Any]:
    started = perf_counter()
    candidates, failures, seeds = [], [], []
    for candidate_id in range(int(settings["candidate_count"])):
        seed = derived_seed(base_seed, "etkdg_v3", candidate_id)
        seeds.append(seed)
        try:
            candidates.append(embed_etkdg(
                sample,
                seed=seed,
                maximum_iterations=int(settings["maximum_iterations"]),
                use_random_coordinates=bool(settings["use_random_coordinates"]),
            ))
        except Exception as error:
            failures.append({
                "candidate_id": candidate_id,
                "seed": seed,
                "failure": f"{type(error).__name__}: {error}",
            })
    if len(candidates) != int(settings["candidate_count"]):
        raise RuntimeError(
            f"ETKDG strict best-of-N incomplete: {len(candidates)}/"
            f"{settings['candidate_count']}; failures={failures}"
        )
    selected_id, energy_records = _lowest_energy_candidate(sample, candidates)
    return {
        "route": ROUTES[0],
        "positions": candidates[selected_id],
        "selected_candidate_id": selected_id,
        "candidate_seeds": seeds,
        "candidate_records": energy_records,
        "diversity": candidate_diversity(candidates),
        "runtime_seconds": perf_counter() - started,
        "target_pg_used_by_generator": False,
        "reference_coordinates_used": False,
        "hard_projection_used": False,
        "f02_used": False,
    }


def run_etflow_raw_route(
    sample: dict[str, Any], raw_positions: list[np.ndarray], *, prior_seconds: float
) -> dict[str, Any]:
    started = perf_counter()
    candidates = [np.asarray(value, dtype=np.float64) for value in raw_positions]
    selected_id, energy_records = _lowest_energy_candidate(sample, candidates)
    post_seconds = perf_counter() - started
    return {
        "route": ROUTES[1],
        "positions": candidates[selected_id],
        "selected_candidate_id": selected_id,
        "candidate_records": energy_records,
        "diversity": candidate_diversity(candidates),
        "shared_prior_runtime_seconds": float(prior_seconds),
        "postprocess_runtime_seconds": post_seconds,
        "runtime_seconds": float(prior_seconds) + post_seconds,
        "target_pg_used_by_generator": False,
        "reference_coordinates_used": False,
        "hard_projection_used": False,
        "f02_used": False,
    }


def _projected_candidates_by_raw(
    raw_positions: list[np.ndarray],
    samples: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed, failures = [], []
    for raw_id, raw in enumerate(raw_positions):
        choices = []
        for action_id, sample in enumerate(samples):
            try:
                result = project_etflow_ef1(
                    raw,
                    sample,
                    match_radius_of_gyration=bool(
                        protocol["projection"]["match_radius_of_gyration"]
                    ),
                )
                positions = np.asarray(result["positions"], dtype=np.float64)
                below_06, below_08 = _pair_counts(positions)
                choices.append({
                    "raw_candidate_id": raw_id,
                    "action_candidate_id": action_id,
                    "positions": positions,
                    "pair_count_below_0p6_angstrom": below_06,
                    "pair_count_below_0p8_angstrom": below_08,
                    "minimum_pair_distance_angstrom": minimum_pair_distance(positions),
                    "projection_rmsd_angstrom": float(result["projection_rmsd_angstrom"]),
                    "orientation_id": int(result["orientation_id"]),
                    "maximum_operation_error_angstrom": float(
                        result["operation_errors"]["max_atom_error_angstrom"]
                    ),
                })
            except Exception as error:
                failures.append({
                    "raw_candidate_id": raw_id,
                    "action_candidate_id": action_id,
                    "failure": f"{type(error).__name__}: {error}",
                })
        if not choices:
            continue
        completed.append(min(choices, key=projection_rank_key))
    return completed, failures


def run_projection_route(
    samples: list[dict[str, Any]],
    raw_positions: list[np.ndarray],
    target_pg: str,
    protocol: dict[str, Any],
    *,
    prior_seconds: float,
) -> dict[str, Any]:
    """Select one reference-free projected action per raw ET-Flow candidate."""

    started = perf_counter()
    completed, failures = _projected_candidates_by_raw(raw_positions, samples, protocol)
    if not completed:
        raise RuntimeError(f"all hard projections failed: {failures}")
    eligible = list(range(len(completed)))
    inertia_values = None
    if target_pg == "S4":
        minimum = float(protocol["s4_inertia_selection"]["minimum_separation"])
        inertia_values = [
            mass_weighted_inertia_separation(
                samples[item["action_candidate_id"]]["atomic_numbers"], item["positions"]
            )
            for item in completed
        ]
        eligible = [
            index for index, value in enumerate(inertia_values) if value >= minimum
        ]
        if not eligible:
            raise RuntimeError("all projected S4 candidates failed the frozen inertia guard")
    selected_index = min(
        eligible,
        key=lambda index: projection_rank_key(completed[index]),
    )
    selected = completed[selected_index]
    action_sample = samples[int(selected["action_candidate_id"])]
    selected["operation_errors"] = operation_errors(
        selected["positions"],
        action_sample["target_operation_matrices"],
        action_sample["target_permutation_index"],
    )
    post_seconds = perf_counter() - started
    return {
        "route": ROUTES[2],
        "positions": selected["positions"],
        "selected_candidate_id": int(selected["raw_candidate_id"]),
        "selected_action_candidate_id": int(selected["action_candidate_id"]),
        "candidate_records": completed,
        "projection_failures": failures,
        "s4_inertia_separations": inertia_values,
        "diversity": candidate_diversity(
            [record["positions"] for record in completed]
        ),
        "shared_prior_runtime_seconds": float(prior_seconds),
        "postprocess_runtime_seconds": post_seconds,
        "runtime_seconds": float(prior_seconds) + post_seconds,
        "target_pg_used_by_generator": True,
        "reference_coordinates_used": False,
        "hard_projection_used": True,
        "f02_used": False,
        "selected_operation_max_atom_error_angstrom": float(
            selected["operation_errors"]["max_atom_error_angstrom"]
        ),
    }


def run_full_f02_route(
    samples: list[dict[str, Any]],
    raw_positions: list[np.ndarray],
    target_pg: str,
    seed: int,
    protocol: dict[str, Any],
    *,
    prior_seconds: float,
) -> dict[str, Any]:
    """Apply the frozen v5 combination ranking, projection and F0.2 logic."""

    from generative_model.inference.generate_etflow_symmetric_xyz_v3 import (
        _rank_combinations,
    )

    started = perf_counter()
    ranked, projection_failures = _rank_combinations(raw_positions, samples, protocol)
    optimize_count = (
        int(protocol["rare_action_selection"][target_pg]["optimize_top_combinations"])
        if target_pg in {"S4", "D6h"}
        else len(ranked)
    )
    completed, coordinates, failures = [], [], []
    for rank, choice in enumerate(ranked[:optimize_count]):
        try:
            record, values = _finish_candidate(
                samples[int(choice["action_candidate_id"])],
                raw_positions[int(choice["raw_candidate_id"])],
                rank,
                int(seed),
                protocol,
            )
            record.update({
                "preselection_rank": rank,
                "raw_candidate_id": int(choice["raw_candidate_id"]),
                "action_candidate_id": int(choice["action_candidate_id"]),
                "preselection_metrics": choice,
            })
            completed.append(record)
            coordinates.append(values)
        except Exception as error:
            failures.append({
                "preselection_rank": rank,
                "raw_candidate_id": int(choice["raw_candidate_id"]),
                "action_candidate_id": int(choice["action_candidate_id"]),
                "failure": f"{type(error).__name__}: {error}",
            })
    if target_pg == "S4":
        selected_index, inertia_values = select_s4_inertia_stable_candidate(
            completed,
            coordinates,
            samples[0]["atomic_numbers"],
            minimum_separation=float(
                protocol["s4_inertia_selection"]["minimum_separation"]
            ),
        )
    else:
        selected_index, inertia_values = _select_candidate(completed), None
    selected = completed[selected_index]
    selected_coordinates = coordinates[selected_index]
    post_seconds = perf_counter() - started
    return {
        "route": ROUTES[3],
        "positions": selected_coordinates["final"],
        "projected_positions": selected_coordinates["projected"],
        "selected_candidate_id": int(selected["raw_candidate_id"]),
        "selected_action_candidate_id": int(selected["action_candidate_id"]),
        "candidate_records": completed,
        "projection_failures": projection_failures,
        "optimization_failures": failures,
        "s4_inertia_separations": inertia_values,
        "selected_f02": selected,
        "diversity": candidate_diversity(
            [value["final"] for value in coordinates]
        ),
        "shared_prior_runtime_seconds": float(prior_seconds),
        "postprocess_runtime_seconds": post_seconds,
        "runtime_seconds": float(prior_seconds) + post_seconds,
        "target_pg_used_by_generator": True,
        "reference_coordinates_used": False,
        "hard_projection_used": True,
        "f02_used": True,
        "selected_operation_max_atom_error_angstrom": float(
            selected["final_maximum_operation_error_angstrom"]
        ),
    }


def strip_positions(record: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe route record while keeping arrays in the NPZ."""

    result = dict(record)
    result.pop("positions", None)
    result.pop("projected_positions", None)
    for key in ("candidate_records",):
        cleaned = []
        for item in result.get(key, []):
            value = dict(item)
            value.pop("positions", None)
            cleaned.append(value)
        if key in result:
            result[key] = cleaned
    return result
