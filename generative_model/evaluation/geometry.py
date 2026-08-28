"""与模型无关的三维几何和 MMFF 松弛指标。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .schema import EvaluationSample


def bond_lengths(sample: EvaluationSample) -> np.ndarray:
    if sample.bond_index.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    begin, end = sample.bond_index
    return np.linalg.norm(sample.positions[begin] - sample.positions[end], axis=1)


def bond_angles_degrees(sample: EvaluationSample) -> np.ndarray:
    neighbors: list[list[int]] = [[] for _ in sample.atomic_numbers]
    for begin, end in sample.bond_index.T:
        neighbors[int(begin)].append(int(end))
        neighbors[int(end)].append(int(begin))
    angles: list[float] = []
    for center, adjacent in enumerate(neighbors):
        for left, right in combinations(sorted(adjacent), 2):
            first = sample.positions[left] - sample.positions[center]
            second = sample.positions[right] - sample.positions[center]
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator == 0:
                angles.append(float("nan"))
                continue
            cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
            angles.append(float(np.degrees(np.arccos(cosine))))
    return np.asarray(angles, dtype=np.float64)


def nonbonded_distances(sample: EvaluationSample) -> np.ndarray:
    atom_count = len(sample.atomic_numbers)
    if atom_count < 2:
        return np.empty(0, dtype=np.float64)
    bonded = {tuple(map(int, edge)) for edge in sample.bond_index.T}
    pairs = [pair for pair in combinations(range(atom_count), 2) if pair not in bonded]
    if not pairs:
        return np.empty(0, dtype=np.float64)
    begin = np.fromiter((pair[0] for pair in pairs), dtype=np.int64)
    end = np.fromiter((pair[1] for pair in pairs), dtype=np.int64)
    return np.linalg.norm(sample.positions[begin] - sample.positions[end], axis=1)


@dataclass(frozen=True)
class MMFFResult:
    status: str
    initial_energy_kcal_mol: float | None = None
    final_energy_kcal_mol: float | None = None
    centered_rms_displacement_angstrom: float | None = None


def mmff_relax(
    molecule: Chem.Mol,
    *,
    variant: str,
    max_iterations: int,
) -> MMFFResult:
    """在副本上确定性松弛，不覆盖输入坐标。"""

    working = Chem.Mol(molecule)
    if not AllChem.MMFFHasAllMoleculeParams(working):
        return MMFFResult(status="unsupported_parameters")
    properties = AllChem.MMFFGetMoleculeProperties(working, mmffVariant=variant)
    if properties is None:
        return MMFFResult(status="setup_failed")
    force_field = AllChem.MMFFGetMoleculeForceField(working, properties)
    if force_field is None:
        return MMFFResult(status="setup_failed")
    initial_energy = float(force_field.CalcEnergy())
    initial = np.asarray(working.GetConformer().GetPositions(), dtype=np.float64)
    optimize_status = int(
        AllChem.MMFFOptimizeMolecule(
            working,
            mmffVariant=variant,
            maxIters=max_iterations,
        )
    )
    final_properties = AllChem.MMFFGetMoleculeProperties(working, mmffVariant=variant)
    final_force_field = (
        None
        if final_properties is None
        else AllChem.MMFFGetMoleculeForceField(working, final_properties)
    )
    if final_force_field is None:
        return MMFFResult(status="setup_failed")
    final_energy = float(final_force_field.CalcEnergy())
    final = np.asarray(working.GetConformer().GetPositions(), dtype=np.float64)
    initial_centered = initial - initial.mean(axis=0)
    final_centered = final - final.mean(axis=0)
    displacement = float(
        np.sqrt(np.mean(np.sum(np.square(final_centered - initial_centered), axis=1)))
    )
    status = {0: "converged", 1: "not_converged", -1: "setup_failed"}.get(
        optimize_status, f"unknown_status_{optimize_status}"
    )
    return MMFFResult(
        status=status,
        initial_energy_kcal_mol=initial_energy,
        final_energy_kcal_mol=final_energy,
        centered_rms_displacement_angstrom=displacement,
    )

