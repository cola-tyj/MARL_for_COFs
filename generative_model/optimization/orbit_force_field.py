"""UFF minimization in atomic-orbit representative coordinate space."""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit.Chem import AllChem
from scipy.optimize import minimize

from generative_model.models.symmetry_by_construction import (
    compress_orbit_representatives,
    expand_orbit_representatives,
)
from .force_field_support import build_strict_rdkit_molecule


SCHEMA_VERSION = "orbit-constrained-uff-optimizer-v1"


def _set_positions(molecule, positions: np.ndarray) -> None:
    conformer = molecule.GetConformer(0)
    for index, coordinate in enumerate(positions):
        conformer.SetAtomPosition(index, tuple(map(float, coordinate)))


def _expansion_matrix(sample: dict[str, Any], orbit_count: int) -> np.ndarray:
    atom_count = len(sample["atomic_numbers"])
    matrix = np.empty((3 * atom_count, 3 * orbit_count), dtype=np.float64)
    for column in range(3 * orbit_count):
        basis = np.zeros((orbit_count, 3), dtype=np.float64)
        basis.reshape(-1)[column] = 1.0
        expanded = expand_orbit_representatives(
            basis,
            sample["target_orbit_id"],
            sample["target_operation_matrices"],
            sample["target_permutation_index"],
        )
        matrix[:, column] = expanded.reshape(-1)
    return matrix


def optimize_orbit_uff(
    sample: dict[str, Any],
    initial_positions: np.ndarray,
    *,
    maximum_iterations: int = 200,
    function_tolerance: float = 1e-9,
    gradient_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Optimize UFF energy while every evaluated geometry has exact symmetry."""

    initial = np.asarray(initial_positions, dtype=np.float64)
    if initial.shape != (len(sample["atomic_numbers"]), 3) or not np.isfinite(initial).all():
        raise ValueError("initial_positions 非法")
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

    evaluation_count = 0

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluation_count
        coordinates = (expansion @ values).reshape(initial.shape)
        flat_coordinates = tuple(map(float, coordinates.reshape(-1)))
        # RDKit ForceField caches its own point vector. Passing coordinates is
        # mandatory: mutating only the Mol conformer can yield a stale gradient.
        energy = float(force_field.CalcEnergy(flat_coordinates))
        atom_gradient = np.asarray(
            force_field.CalcGrad(flat_coordinates), dtype=np.float64
        )
        gradient = expansion.T @ atom_gradient
        evaluation_count += 1
        if not np.isfinite(energy) or not np.isfinite(gradient).all():
            raise RuntimeError("UFF objective/gradient 非有限")
        return energy, gradient

    initial_energy, initial_gradient = objective(initial_vector)
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
    final_positions = (expansion @ np.asarray(result.x, dtype=np.float64)).reshape(initial.shape)
    _set_positions(molecule, final_positions)
    final_flat = tuple(map(float, final_positions.reshape(-1)))
    final_energy = float(force_field.CalcEnergy(final_flat))
    if not np.isfinite(final_positions).all() or not np.isfinite(final_energy):
        raise RuntimeError("UFF optimizer 输出非有限")
    return {
        "schema_version": SCHEMA_VERSION,
        "positions": final_positions,
        "initial_energy_kcal_mol": initial_energy,
        "final_energy_kcal_mol": final_energy,
        "initial_projected_gradient_norm": float(np.linalg.norm(initial_gradient)),
        "final_projected_gradient_norm": float(np.linalg.norm(
            expansion.T @ np.asarray(force_field.CalcGrad(final_flat), dtype=np.float64)
        )),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective_evaluations": evaluation_count,
        "orbit_count": len(representatives),
    }
