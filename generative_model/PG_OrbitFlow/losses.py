"""Flow, chemistry-geometry and group-action objectives."""

from __future__ import annotations

import torch


LOSS_SCHEMA_VERSION = "pg-orbitflow-objective-v1"

# Pyykko-style covalent radii are used only to define a conservative overlap
# floor.  Every frozen-vocabulary element is explicit; unknown Z fails.
_COVALENT_RADII_ANGSTROM = {
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


def _bond_lengths(positions, bond_index):
    if bond_index.ndim != 2 or bond_index.shape[0] != 2:
        raise ValueError("bond_index must be [2,M]")
    return torch.linalg.vector_norm(
        positions[bond_index[0]] - positions[bond_index[1]], dim=-1
    )


def symmetry_mse(positions, matrices, permutations):
    errors = []
    for rotation, row in zip(matrices, permutations):
        errors.append(torch.mean(torch.square(positions @ rotation.T - positions[row])))
    values = torch.stack(errors)
    return values.mean(), values.max()


def overlap_loss(positions, atomic_numbers, bond_index, *, coordinate_scale_angstrom: float):
    atom_count = len(positions)
    radii = []
    for number in atomic_numbers.detach().cpu().tolist():
        if int(number) not in _COVALENT_RADII_ANGSTROM:
            raise ValueError(f"no collision radius for atomic number {int(number)}")
        radii.append(_COVALENT_RADII_ANGSTROM[int(number)] / coordinate_scale_angstrom)
    radii_tensor = torch.as_tensor(radii, dtype=positions.dtype, device=positions.device)
    bonded = torch.zeros((atom_count, atom_count), dtype=torch.bool, device=positions.device)
    bonded[bond_index[0], bond_index[1]] = True
    bonded[bond_index[1], bond_index[0]] = True
    upper = torch.triu_indices(atom_count, atom_count, offset=1, device=positions.device)
    keep = ~bonded[upper[0], upper[1]]
    left, right = upper[0][keep], upper[1][keep]
    if not len(left):
        return torch.zeros((), dtype=positions.dtype, device=positions.device)
    distances = torch.linalg.vector_norm(positions[left] - positions[right], dim=-1)
    minimum = 0.60 * (radii_tensor[left] + radii_tensor[right])
    return torch.mean(torch.square(torch.relu(minimum - distances)))


def compute_loss(
    *,
    predicted_velocity,
    target_velocity,
    positions_t,
    time,
    geometry_target_positions,
    transport_target_positions,
    atomic_numbers,
    bond_index,
    matrices,
    permutations,
    coordinate_scale_angstrom: float,
    bond_weight: float,
    overlap_weight: float,
    symmetry_weight: float,
    endpoint_weight: float = 0.0,
):
    if endpoint_weight < 0:
        raise ValueError("endpoint_weight must be non-negative")
    flow = torch.mean(torch.square(predicted_velocity - target_velocity))
    endpoint = positions_t + (1.0 - time) * predicted_velocity
    bond = torch.mean(
        torch.square(
            _bond_lengths(endpoint, bond_index)
            - _bond_lengths(geometry_target_positions, bond_index)
        )
    )
    overlap = overlap_loss(
        endpoint,
        atomic_numbers,
        bond_index,
        coordinate_scale_angstrom=coordinate_scale_angstrom,
    )
    symmetry, worst_symmetry = symmetry_mse(endpoint, matrices, permutations)
    endpoint_auxiliary = torch.mean(
        torch.square(endpoint - transport_target_positions)
    )
    total = (
        flow
        + bond_weight * bond
        + overlap_weight * overlap
        + symmetry_weight * symmetry
        + endpoint_weight * endpoint_auxiliary
    )
    return {
        "loss": total,
        "flow_mse": flow,
        "bond_length_mse": bond,
        "overlap_loss": overlap,
        "symmetry_mse": symmetry,
        "worst_operation_mse": worst_symmetry,
        "endpoint_auxiliary_mse": endpoint_auxiliary,
        "endpoint": endpoint,
    }
