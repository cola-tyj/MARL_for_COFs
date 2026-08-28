"""Orbit-representative coordinates expanded by an explicit finite group action.

The graph, atom order and group action are immutable.  A downstream model only
predicts one 3D vector per atomic orbit; this module expands those vectors to all
atoms and applies a final Reynolds projection.  No atom assignment, element
fallback or learned symmetry classification occurs here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from generative_model.models.graph_pg_3d_projection import (
    center_positions,
    operation_errors,
    project_reynolds,
    validate_group_action,
)


SCHEMA_VERSION = "orbit-representative-hard-symmetry-v1"


def validate_orbit_action(
    orbit_id: np.ndarray,
    operation_matrices: np.ndarray,
    permutation_index: np.ndarray,
) -> dict[str, Any]:
    orbit_ids = np.asarray(orbit_id, dtype=np.int64)
    if orbit_ids.ndim != 1 or not len(orbit_ids):
        raise ValueError("orbit_id 必须为非空 [N]")
    unique = np.unique(orbit_ids)
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
        raise ValueError("orbit_id 必须从 0 开始连续编号")
    matrices = np.asarray(operation_matrices, dtype=np.float64)
    permutations = np.asarray(permutation_index, dtype=np.int64)
    validate_group_action(matrices, permutations, atom_count=len(orbit_ids))
    for permutation in permutations:
        if not np.array_equal(orbit_ids, orbit_ids[permutation]):
            raise ValueError("group permutation 跨 orbit 映射")

    representatives = []
    orbit_sizes = []
    stabilizer_sizes = []
    mapping_counts = []
    for current_orbit in unique:
        members = np.flatnonzero(orbit_ids == current_orbit)
        representative = int(members[0])
        reached = np.unique(permutations[:, representative])
        if not np.array_equal(np.sort(reached), members):
            raise ValueError("operations 对 orbit 的作用不传递")
        counts = [int(np.sum(permutations[:, representative] == atom)) for atom in members]
        if min(counts) <= 0:
            raise ValueError("orbit atom 缺少 representative mapping operation")
        representatives.append(representative)
        orbit_sizes.append(len(members))
        stabilizer_sizes.append(counts[0])
        mapping_counts.append(counts)
    return {
        "schema_version": SCHEMA_VERSION,
        "atom_count": len(orbit_ids),
        "orbit_count": len(unique),
        "representative_indices": np.asarray(representatives, dtype=np.int64),
        "orbit_sizes": np.asarray(orbit_sizes, dtype=np.int64),
        "stabilizer_sizes": np.asarray(stabilizer_sizes, dtype=np.int64),
        "mapping_counts": mapping_counts,
    }


def compress_orbit_representatives(
    positions: np.ndarray,
    orbit_id: np.ndarray,
    operation_matrices: np.ndarray,
    permutation_index: np.ndarray,
) -> dict[str, np.ndarray]:
    values = center_positions(positions)
    audit = validate_orbit_action(orbit_id, operation_matrices, permutation_index)
    if len(values) != audit["atom_count"]:
        raise ValueError("positions 与 orbit atom_count 不一致")
    indices = audit["representative_indices"]
    return {
        "representative_indices": indices.copy(),
        "representative_positions": values[indices].copy(),
        "orbit_sizes": audit["orbit_sizes"].copy(),
    }


def expand_orbit_representatives(
    representative_positions: np.ndarray,
    orbit_id: np.ndarray,
    operation_matrices: np.ndarray,
    permutation_index: np.ndarray,
    *,
    reynolds_iterations: int = 1,
) -> np.ndarray:
    """Expand one vector per orbit under ``X @ R.T = X[permutation]``."""

    orbit_ids = np.asarray(orbit_id, dtype=np.int64)
    matrices = np.asarray(operation_matrices, dtype=np.float64)
    permutations = np.asarray(permutation_index, dtype=np.int64)
    representatives = np.asarray(representative_positions, dtype=np.float64)
    audit = validate_orbit_action(orbit_ids, matrices, permutations)
    if representatives.shape != (audit["orbit_count"], 3):
        raise ValueError("representative_positions 必须为 [orbit_count,3]")
    if not np.isfinite(representatives).all():
        raise ValueError("representative_positions 含 NaN/Inf")
    expanded = np.zeros((audit["atom_count"], 3), dtype=np.float64)
    for current_orbit, representative_atom in enumerate(audit["representative_indices"]):
        members = np.flatnonzero(orbit_ids == current_orbit)
        stabilizer = permutations[:, representative_atom] == representative_atom
        stabilized = np.mean(
            np.einsum("j,gij->gi", representatives[current_orbit], matrices[stabilizer]),
            axis=0,
        )
        for atom in members:
            mappings = permutations[:, representative_atom] == atom
            expanded[atom] = np.mean(
                np.einsum("j,gij->gi", stabilized, matrices[mappings]),
                axis=0,
            )
    expanded = project_reynolds(
        expanded, matrices, permutations, iterations=reynolds_iterations
    )
    errors = operation_errors(expanded, matrices, permutations)
    if not np.isfinite(expanded).all():
        raise RuntimeError("orbit-expanded coordinates 含 NaN/Inf")
    if errors["max_atom_error_angstrom"] > 1e-5:
        raise RuntimeError("orbit expansion 未满足冻结 group action")
    return expanded


def expand_orbit_representatives_torch(
    representative_positions,
    orbit_id,
    operation_matrices,
    permutation_index,
):
    """Differentiable Torch expansion for downstream coordinate models."""

    import torch

    if representative_positions.ndim != 2 or representative_positions.shape[-1] != 3:
        raise ValueError("representative_positions 必须为 [O,3]")
    if not representative_positions.is_floating_point() or not bool(
        torch.isfinite(representative_positions).all()
    ):
        raise ValueError("representative_positions 必须为有限浮点 tensor")
    orbit_ids_np = np.asarray(orbit_id.detach().cpu(), dtype=np.int64)
    matrices_np = np.asarray(operation_matrices.detach().cpu(), dtype=np.float64)
    permutations_np = np.asarray(permutation_index.detach().cpu(), dtype=np.int64)
    audit = validate_orbit_action(orbit_ids_np, matrices_np, permutations_np)
    if representative_positions.shape != (audit["orbit_count"], 3):
        raise ValueError("representative_positions orbit_count 不一致")
    matrices = operation_matrices.to(
        dtype=representative_positions.dtype, device=representative_positions.device
    )
    permutations = permutation_index.to(device=representative_positions.device)
    orbit_ids = orbit_id.to(device=representative_positions.device)
    rows = [None] * audit["atom_count"]
    for current_orbit, representative_atom in enumerate(audit["representative_indices"]):
        representative_atom = int(representative_atom)
        members = torch.nonzero(orbit_ids == current_orbit, as_tuple=False).flatten()
        stabilizer = permutations[:, representative_atom] == representative_atom
        stabilized = torch.mean(
            torch.einsum(
                "j,gij->gi", representative_positions[current_orbit], matrices[stabilizer]
            ),
            dim=0,
        )
        for atom_tensor in members:
            atom = int(atom_tensor.item())
            mappings = permutations[:, representative_atom] == atom
            rows[atom] = torch.mean(
                torch.einsum("j,gij->gi", stabilized, matrices[mappings]), dim=0
            )
    expanded = torch.stack(rows, dim=0)
    accumulator = torch.zeros_like(expanded)
    for matrix, permutation in zip(matrices, permutations):
        accumulator = accumulator.index_add(0, permutation, expanded @ matrix.T)
    expanded = accumulator / len(matrices)
    return expanded - expanded.mean(dim=0, keepdim=True)
