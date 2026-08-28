"""Orbit-constrained UFF with an explicit short-range nonbonded barrier."""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit.Chem import AllChem
from scipy.optimize import minimize

from generative_model.models.symmetry_by_construction import compress_orbit_representatives
from .force_field_support import build_strict_rdkit_molecule
from .orbit_force_field import _expansion_matrix


SCHEMA_VERSION = "orbit-constrained-uff-short-range-repulsion-v1"


def nonbonded_pairs(sample: dict[str, Any]) -> np.ndarray:
    atom_count = len(sample["atomic_numbers"])
    bonded = {tuple(sorted(map(int, edge))) for edge in np.asarray(sample["bond_index"]).T}
    pairs = [
        (left, right)
        for left in range(atom_count)
        for right in range(left + 1, atom_count)
        if (left, right) not in bonded
    ]
    return np.asarray(pairs, dtype=np.int64)


def short_range_repulsion(
    positions: np.ndarray,
    pairs: np.ndarray,
    *,
    cutoff_angstrom: float,
    force_constant_kcal_mol_angstrom2: float,
) -> tuple[float, np.ndarray, int]:
    """Harmonic barrier ``0.5*k*(cutoff-r)^2`` for nonbonded ``r<cutoff``."""

    values = np.asarray(positions, dtype=np.float64)
    pair_index = np.asarray(pairs, dtype=np.int64)
    if cutoff_angstrom <= 0 or force_constant_kcal_mol_angstrom2 <= 0:
        raise ValueError("short-range repulsion 参数必须为正")
    if pair_index.ndim != 2 or pair_index.shape[1] != 2:
        raise ValueError("nonbonded pairs 必须为 [P,2]")
    delta = values[pair_index[:, 0]] - values[pair_index[:, 1]]
    distances = np.linalg.norm(delta, axis=1)
    active = distances < float(cutoff_angstrom)
    if bool(np.any(distances[active] <= 1e-10)):
        raise RuntimeError("非键原子完全重合，repulsion 梯度方向未定义")
    gradient = np.zeros_like(values)
    if not bool(np.any(active)):
        return 0.0, gradient, 0
    gaps = float(cutoff_angstrom) - distances[active]
    energy = 0.5 * float(force_constant_kcal_mol_angstrom2) * float(np.sum(gaps**2))
    directions = delta[active] / distances[active, None]
    pair_gradient = -float(force_constant_kcal_mol_angstrom2) * gaps[:, None] * directions
    active_pairs = pair_index[active]
    np.add.at(gradient, active_pairs[:, 0], pair_gradient)
    np.add.at(gradient, active_pairs[:, 1], -pair_gradient)
    return energy, gradient, int(np.sum(active))


def optimize_orbit_uff_with_repulsion(
    sample: dict[str, Any],
    initial_positions: np.ndarray,
    *,
    cutoff_angstrom: float,
    force_constant_kcal_mol_angstrom2: float,
    maximum_iterations: int,
    function_tolerance: float,
    gradient_tolerance: float,
) -> dict[str, Any]:
    initial = np.asarray(initial_positions, dtype=np.float64)
    compressed = compress_orbit_representatives(
        initial,
        sample["target_orbit_id"],
        sample["target_operation_matrices"],
        sample["target_permutation_index"],
    )
    representatives = compressed["representative_positions"]
    expansion = _expansion_matrix(sample, len(representatives))
    initial_vector = representatives.reshape(-1)
    reconstructed = (expansion @ initial_vector).reshape(initial.shape)
    if float(np.max(np.abs(reconstructed - initial))) > 2e-5:
        raise ValueError("initial_positions 不在冻结 orbit subspace")
    molecule = build_strict_rdkit_molecule({**sample, "positions": reconstructed})
    if not AllChem.UFFHasAllMoleculeParams(molecule):
        raise RuntimeError("UFF 参数不完整；禁止 fallback")
    force_field = AllChem.UFFGetMoleculeForceField(molecule, confId=0)
    if force_field is None:
        raise RuntimeError("UFF force field setup 失败")
    pairs = nonbonded_pairs(sample)
    evaluation_count = 0

    def components(values: np.ndarray) -> tuple[float, float, np.ndarray, int]:
        coordinates = (expansion @ values).reshape(initial.shape)
        flat = tuple(map(float, coordinates.reshape(-1)))
        uff_energy = float(force_field.CalcEnergy(flat))
        uff_gradient = np.asarray(force_field.CalcGrad(flat), dtype=np.float64).reshape(initial.shape)
        barrier_energy, barrier_gradient, active_count = short_range_repulsion(
            coordinates,
            pairs,
            cutoff_angstrom=cutoff_angstrom,
            force_constant_kcal_mol_angstrom2=force_constant_kcal_mol_angstrom2,
        )
        atom_gradient = uff_gradient + barrier_gradient
        return uff_energy, barrier_energy, expansion.T @ atom_gradient.reshape(-1), active_count

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluation_count
        uff_energy, barrier_energy, gradient, _ = components(values)
        evaluation_count += 1
        total = uff_energy + barrier_energy
        if not np.isfinite(total) or not np.isfinite(gradient).all():
            raise RuntimeError("UFF+repulsion objective/gradient 非有限")
        return total, gradient

    initial_uff, initial_barrier, initial_gradient, initial_active = components(initial_vector)
    result = minimize(
        objective,
        initial_vector,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(maximum_iterations),
            "ftol": float(function_tolerance),
            "gtol": float(gradient_tolerance),
            "maxls": 40,
        },
    )
    final_vector = np.asarray(result.x, dtype=np.float64)
    final_positions = (expansion @ final_vector).reshape(initial.shape)
    final_uff, final_barrier, final_gradient, final_active = components(final_vector)
    return {
        "schema_version": SCHEMA_VERSION,
        "positions": final_positions,
        "initial_uff_energy_kcal_mol": initial_uff,
        "final_uff_energy_kcal_mol": final_uff,
        "initial_repulsion_energy_kcal_mol": initial_barrier,
        "final_repulsion_energy_kcal_mol": final_barrier,
        "initial_total_objective_kcal_mol": initial_uff + initial_barrier,
        "final_total_objective_kcal_mol": final_uff + final_barrier,
        "initial_active_repulsion_pairs": initial_active,
        "final_active_repulsion_pairs": final_active,
        "initial_projected_gradient_norm": float(np.linalg.norm(initial_gradient)),
        "final_projected_gradient_norm": float(np.linalg.norm(final_gradient)),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective_evaluations": evaluation_count,
        "orbit_count": len(representatives),
        "nonbonded_pair_count": len(pairs),
    }
