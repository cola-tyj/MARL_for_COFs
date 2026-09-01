"""Strict fixed-graph internal-geometry contracts and differentiable losses."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .data import OrbitFlowSample


GEOMETRY_SCHEMA_VERSION = "pg-orbitflow-fixed-graph-geometry-v1"


@dataclass(frozen=True)
class GeometryContract:
    bond_index: np.ndarray
    bond_orbit_id: np.ndarray
    angle_index: np.ndarray
    angle_orbit_id: np.ndarray
    torsion_index: np.ndarray
    torsion_orbit_id: np.ndarray
    local_pair_index: np.ndarray
    local_pair_orbit_id: np.ndarray
    ring_bond_indices: np.ndarray
    ring_bond_orbit_id: np.ndarray
    chirality_index: np.ndarray
    chirality_orbit_id: np.ndarray


def _normal_bond(values) -> tuple[int, int]:
    left, right = (int(value) for value in values)
    return (left, right) if left < right else (right, left)


def _normal_angle(values) -> tuple[int, int, int]:
    left, center, right = (int(value) for value in values)
    return (left, center, right) if left < right else (right, center, left)


def _normal_torsion(values) -> tuple[int, int, int, int]:
    forward = tuple(int(value) for value in values)
    reverse = tuple(reversed(forward))
    return min(forward, reverse)


def _tuple_orbits(
    tuples: list[tuple[int, ...]],
    permutations: np.ndarray,
    normalizer,
) -> np.ndarray:
    if not tuples:
        return np.empty((0,), dtype=np.int64)
    available = set(tuples)
    representatives = []
    for values in tuples:
        transformed = []
        for permutation in permutations:
            current = normalizer(tuple(int(permutation[index]) for index in values))
            if current not in available:
                raise ValueError("geometry tuple set is not closed under group action")
            transformed.append(current)
        representatives.append(min(transformed))
    unique = {value: index for index, value in enumerate(sorted(set(representatives)))}
    return np.asarray([unique[value] for value in representatives], dtype=np.int64)


def _bridge_bond_indices(atom_count: int, bonds: list[tuple[int, int]]) -> set[int]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(atom_count)]
    for edge_index, (left, right) in enumerate(bonds):
        adjacency[left].append((right, edge_index))
        adjacency[right].append((left, edge_index))
    discovery = np.full(atom_count, -1, dtype=np.int64)
    low = np.full(atom_count, -1, dtype=np.int64)
    bridges: set[int] = set()
    clock = 0

    def visit(node: int, parent_edge: int) -> None:
        nonlocal clock
        discovery[node] = low[node] = clock
        clock += 1
        for neighbor, edge_index in adjacency[node]:
            if edge_index == parent_edge:
                continue
            if discovery[neighbor] < 0:
                visit(neighbor, edge_index)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    bridges.add(edge_index)
            else:
                low[node] = min(low[node], discovery[neighbor])

    visit(0, -1)
    if np.any(discovery < 0):
        raise ValueError("geometry contract requires a connected molecular graph")
    return bridges


def build_geometry_contract(sample: OrbitFlowSample) -> GeometryContract:
    """Enumerate graph-local coordinates without mutating canonical topology."""

    atom_count = len(sample.atom_types)
    raw_bonds = np.asarray(sample.bond_index, dtype=np.int64)
    if raw_bonds.ndim != 2 or raw_bonds.shape[0] != 2 or not raw_bonds.shape[1]:
        raise ValueError("bond_index must be non-empty [2,M]")
    bonds = [_normal_bond(edge) for edge in raw_bonds.T]
    if len(bonds) != len(set(bonds)):
        raise ValueError("canonical bond graph contains duplicate undirected edges")
    bonds = sorted(bonds)
    neighbors = [set() for _ in range(atom_count)]
    for left, right in bonds:
        if not 0 <= left < atom_count or not 0 <= right < atom_count or left == right:
            raise ValueError("invalid canonical bond")
        neighbors[left].add(right)
        neighbors[right].add(left)

    angles = sorted(
        _normal_angle((left, center, right))
        for center in range(atom_count)
        for left, right in combinations(sorted(neighbors[center]), 2)
    )
    torsions = sorted(
        {
            _normal_torsion((left, center_left, center_right, right))
            for center_left, center_right in bonds
            for left in neighbors[center_left] - {center_right}
            for right in neighbors[center_right] - {center_left}
            if len({left, center_left, center_right, right}) == 4
        }
    )

    distances = np.full((atom_count, atom_count), atom_count + 1, dtype=np.int64)
    for source in range(atom_count):
        distances[source, source] = 0
        queue = [source]
        for node in queue:
            for neighbor in neighbors[node]:
                if distances[source, neighbor] > distances[source, node] + 1:
                    distances[source, neighbor] = distances[source, node] + 1
                    queue.append(neighbor)
    local_pairs = [
        (left, right)
        for left in range(atom_count)
        for right in range(left + 1, atom_count)
        if 1 <= distances[left, right] <= 3
    ]
    if not local_pairs:
        raise ValueError("geometry contract has no graph-distance<=3 pairs")

    bridges = _bridge_bond_indices(atom_count, bonds)
    ring_indices = np.asarray(
        [index for index in range(len(bonds)) if index not in bridges], dtype=np.int64
    )
    chirality = [
        (center, *tuple(sorted(neighbors[center])[:3]))
        for center in range(atom_count)
        if len(neighbors[center]) >= 3
    ]
    permutations = np.asarray(sample.permutation_index, dtype=np.int64)
    bond_orbits = _tuple_orbits(bonds, permutations, _normal_bond)
    angle_orbits = _tuple_orbits(angles, permutations, _normal_angle)
    torsion_orbits = _tuple_orbits(torsions, permutations, _normal_torsion)
    pair_orbits = _tuple_orbits(local_pairs, permutations, _normal_bond)
    ring_orbits = bond_orbits[ring_indices]
    chirality_orbits = np.asarray(
        [sample.orbit_id[values[0]] for values in chirality], dtype=np.int64
    )
    return GeometryContract(
        bond_index=np.asarray(bonds, dtype=np.int64).T,
        bond_orbit_id=bond_orbits,
        angle_index=np.asarray(angles, dtype=np.int64).reshape(-1, 3).T,
        angle_orbit_id=angle_orbits,
        torsion_index=np.asarray(torsions, dtype=np.int64).reshape(-1, 4).T,
        torsion_orbit_id=torsion_orbits,
        local_pair_index=np.asarray(local_pairs, dtype=np.int64).T,
        local_pair_orbit_id=pair_orbits,
        ring_bond_indices=ring_indices,
        ring_bond_orbit_id=ring_orbits,
        chirality_index=np.asarray(chirality, dtype=np.int64).reshape(-1, 4).T,
        chirality_orbit_id=chirality_orbits,
    )


def _orbit_mean(values, orbit_id):
    import torch

    if values.numel() == 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    identifiers = torch.as_tensor(orbit_id, dtype=torch.long, device=values.device)
    return torch.stack(
        [values[identifiers == current].mean() for current in torch.unique(identifiers)]
    ).mean()


def _lengths(positions, indices):
    return (positions[indices[0]] - positions[indices[1]]).norm(dim=-1)


def _angle_cosines(positions, indices):
    left = positions[indices[0]] - positions[indices[1]]
    right = positions[indices[2]] - positions[indices[1]]
    denominator = left.norm(dim=-1) * right.norm(dim=-1)
    return (left * right).sum(dim=-1) / denominator.clamp_min(1e-8)


def _torsion_sincos(positions, indices):
    import torch

    first = positions[indices[0]] - positions[indices[1]]
    axis = positions[indices[2]] - positions[indices[1]]
    last = positions[indices[3]] - positions[indices[2]]
    normal_left = torch.linalg.cross(first, axis, dim=-1)
    normal_right = torch.linalg.cross(axis, last, dim=-1)
    left_unit = normal_left / normal_left.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    right_unit = normal_right / normal_right.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis_unit = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cosine = (left_unit * right_unit).sum(dim=-1)
    sine = (
        torch.linalg.cross(left_unit, right_unit, dim=-1) * axis_unit
    ).sum(dim=-1)
    return torch.stack((sine, cosine), dim=-1)


def _chirality_values(positions, indices):
    import torch

    if not indices.shape[1]:
        return torch.empty((0,), dtype=positions.dtype, device=positions.device)
    center = positions[indices[0]]
    directions = [positions[indices[row]] - center for row in (1, 2, 3)]
    directions = [
        values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        for values in directions
    ]
    return (
        directions[0]
        * torch.linalg.cross(directions[1], directions[2], dim=-1)
    ).sum(dim=-1)


def geometry_loss_bundle(
    candidate,
    target,
    contract: GeometryContract,
    *,
    coordinate_scale_angstrom: float,
):
    """Return orbit-balanced differentiable losses and direct diagnostics."""

    import torch
    import torch.nn.functional as functional

    if candidate.shape != target.shape or candidate.ndim != 2 or candidate.shape[1] != 3:
        raise ValueError("candidate/target must have equal [N,3] shape")
    if coordinate_scale_angstrom <= 0:
        raise ValueError("coordinate_scale_angstrom must be positive")
    device = candidate.device

    def indices(values):
        return torch.as_tensor(values, dtype=torch.long, device=device)

    bond_index = indices(contract.bond_index)
    angle_index = indices(contract.angle_index)
    torsion_index = indices(contract.torsion_index)
    pair_index = indices(contract.local_pair_index)
    chirality_index = indices(contract.chirality_index)
    candidate_angstrom = candidate * coordinate_scale_angstrom
    target_angstrom = target * coordinate_scale_angstrom

    bond_difference = _lengths(candidate_angstrom, bond_index) - _lengths(
        target_angstrom, bond_index
    )
    bond_errors = functional.smooth_l1_loss(
        bond_difference, torch.zeros_like(bond_difference), beta=0.05, reduction="none"
    )
    bond_loss = _orbit_mean(bond_errors, contract.bond_orbit_id)

    candidate_angles = _angle_cosines(candidate, angle_index)
    target_angles = _angle_cosines(target, angle_index)
    angle_loss = _orbit_mean(
        torch.square(candidate_angles - target_angles), contract.angle_orbit_id
    )

    if torsion_index.shape[1]:
        candidate_torsions = _torsion_sincos(candidate, torsion_index)
        target_torsions = _torsion_sincos(target, torsion_index)
        torsion_errors = 1.0 - torch.sum(
            candidate_torsions * target_torsions, dim=-1
        ).clamp(-1.0, 1.0)
        torsion_loss = _orbit_mean(torsion_errors, contract.torsion_orbit_id)
    else:
        torsion_loss = torch.zeros((), dtype=candidate.dtype, device=device)

    pair_difference = _lengths(candidate_angstrom, pair_index) - _lengths(
        target_angstrom, pair_index
    )
    pair_errors = functional.smooth_l1_loss(
        pair_difference, torch.zeros_like(pair_difference), beta=0.10, reduction="none"
    )
    local_pair_loss = _orbit_mean(pair_errors, contract.local_pair_orbit_id)

    if len(contract.ring_bond_indices):
        ring_difference = bond_difference[
            torch.as_tensor(contract.ring_bond_indices, dtype=torch.long, device=device)
        ]
        ring_errors = functional.smooth_l1_loss(
            ring_difference,
            torch.zeros_like(ring_difference),
            beta=0.05,
            reduction="none",
        )
        ring_loss = _orbit_mean(ring_errors, contract.ring_bond_orbit_id)
    else:
        ring_loss = torch.zeros((), dtype=candidate.dtype, device=device)

    candidate_chirality = _chirality_values(candidate, chirality_index)
    target_chirality = _chirality_values(target, chirality_index)
    active_chirality = torch.abs(target_chirality) >= 0.05
    if bool(active_chirality.any()):
        chirality_errors = torch.square(
            candidate_chirality[active_chirality]
            - target_chirality[active_chirality]
        )
        chirality_loss = _orbit_mean(
            chirality_errors,
            contract.chirality_orbit_id[
                active_chirality.detach().cpu().numpy()
            ],
        )
        chirality_preserved = torch.mean(
            (
                candidate_chirality[active_chirality]
                * target_chirality[active_chirality]
                > 0
            ).to(candidate.dtype)
        )
        active_count = int(active_chirality.sum().item())
    else:
        chirality_loss = torch.zeros((), dtype=candidate.dtype, device=device)
        chirality_preserved = torch.ones((), dtype=candidate.dtype, device=device)
        active_count = 0

    angle_difference = torch.rad2deg(
        torch.abs(
            torch.acos(candidate_angles.clamp(-1.0, 1.0))
            - torch.acos(target_angles.clamp(-1.0, 1.0))
        )
    )
    return {
        "bond_loss": bond_loss,
        "angle_loss": angle_loss,
        "torsion_loss": torsion_loss,
        "local_pair_loss": local_pair_loss,
        "ring_loss": ring_loss,
        "chirality_loss": chirality_loss,
        "bond_mae_angstrom": torch.mean(torch.abs(bond_difference)),
        "angle_mae_degrees": torch.mean(angle_difference),
        "chirality_preserved_fraction": chirality_preserved,
        "active_chirality_count": active_count,
    }
