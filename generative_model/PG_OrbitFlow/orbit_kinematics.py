"""Stabilizer-aware orbit coordinates and exact finite-group lifting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .group import operation_error_numpy, validate_group_action


ORBIT_KINEMATICS_SCHEMA_VERSION = "pg-orbitflow-orbit-kinematics-v1"


@dataclass(frozen=True)
class OrbitParameterization:
    representative_indices: np.ndarray
    orbit_id: np.ndarray
    fixed_bases: tuple[np.ndarray, ...]
    parameter_offsets: np.ndarray
    member_operation_index: np.ndarray
    operation_matrices: np.ndarray
    permutation_index: np.ndarray

    @property
    def orbit_count(self) -> int:
        return len(self.representative_indices)

    @property
    def parameter_count(self) -> int:
        return int(self.parameter_offsets[-1])

    @property
    def fixed_dimensions(self) -> np.ndarray:
        return np.diff(self.parameter_offsets)


def build_orbit_parameterization(sample, *, tolerance: float = 2e-5) -> OrbitParameterization:
    """Build one free representative per atom orbit with stabilizer constraints."""

    matrices = np.asarray(sample.operation_matrices, dtype=np.float64)
    permutations = np.asarray(sample.permutation_index, dtype=np.int64)
    orbit_id = np.asarray(sample.orbit_id, dtype=np.int64)
    audit = validate_group_action(
        matrices,
        permutations,
        atomic_numbers=sample.atomic_numbers,
        orbit_id=orbit_id,
    )
    representatives = np.asarray(
        [np.flatnonzero(orbit_id == current)[0] for current in range(audit.orbit_count)],
        dtype=np.int64,
    )
    bases: list[np.ndarray] = []
    offsets = [0]
    member_operations = np.full(audit.atom_count, -1, dtype=np.int64)
    eye = np.eye(3, dtype=np.float64)
    for current, representative in enumerate(representatives):
        stabilizer = np.flatnonzero(permutations[:, representative] == representative)
        if not len(stabilizer):
            raise RuntimeError("atom-orbit representative has an empty stabilizer")
        projector = matrices[stabilizer].mean(axis=0)
        projector = 0.5 * (projector + projector.T)
        eigenvalues, eigenvectors = np.linalg.eigh(projector)
        keep = eigenvalues >= 1.0 - tolerance
        basis = eigenvectors[:, keep]
        if basis.shape[1] == 0:
            # A representative fixed by inversion can only occupy the origin.
            basis = np.empty((3, 0), dtype=np.float64)
        if basis.size:
            residual = max(
                float(np.max(np.abs(matrix @ basis - basis)))
                for matrix in matrices[stabilizer]
            )
            if residual > tolerance:
                raise ValueError("stabilizer fixed-subspace basis is inconsistent")
            if float(np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1])))) > tolerance:
                raise RuntimeError("stabilizer basis is not orthonormal")
        bases.append(basis)
        offsets.append(offsets[-1] + basis.shape[1])
        members = np.flatnonzero(orbit_id == current)
        for member in members:
            candidates = np.flatnonzero(permutations[:, representative] == member)
            if not len(candidates):
                raise ValueError("orbit member is unreachable from its representative")
            chosen = int(candidates[0])
            member_operations[member] = chosen
            for alternate in candidates[1:]:
                if basis.size:
                    error = float(
                        np.max(
                            np.abs(
                                matrices[chosen] @ basis
                                - matrices[int(alternate)] @ basis
                            )
                        )
                    )
                    if error > tolerance:
                        raise ValueError("group lift depends on coset representative")
    if np.any(member_operations < 0):
        raise RuntimeError("group lift did not assign every atom")
    return OrbitParameterization(
        representative_indices=representatives,
        orbit_id=orbit_id,
        fixed_bases=tuple(bases),
        parameter_offsets=np.asarray(offsets, dtype=np.int64),
        member_operation_index=member_operations,
        operation_matrices=matrices,
        permutation_index=permutations,
    )


def lift_orbit_parameters(parameters, specification: OrbitParameterization):
    """Differentiably lift independent orbit parameters to centered full coordinates."""

    import torch

    if parameters.ndim != 1 or len(parameters) != specification.parameter_count:
        raise ValueError("orbit parameter vector has the wrong shape")
    atom_rows = []
    for atom, current_orbit in enumerate(specification.orbit_id):
        current = int(current_orbit)
        start, stop = specification.parameter_offsets[current : current + 2]
        basis = torch.as_tensor(
            specification.fixed_bases[current],
            dtype=parameters.dtype,
            device=parameters.device,
        )
        representative_column = basis @ parameters[int(start) : int(stop)]
        rotation = torch.as_tensor(
            specification.operation_matrices[
                specification.member_operation_index[atom]
            ],
            dtype=parameters.dtype,
            device=parameters.device,
        )
        atom_rows.append(rotation @ representative_column)
    coordinates = torch.stack(atom_rows, dim=0)
    coordinates = coordinates - coordinates.mean(dim=0, keepdim=True)
    if not bool(torch.isfinite(coordinates).all()):
        raise RuntimeError("group lifting produced NaN/Inf")
    return coordinates


def encode_orbit_parameters(
    coordinates: np.ndarray,
    specification: OrbitParameterization,
    *,
    tolerance: float = 2e-5,
) -> np.ndarray:
    """Encode a centered invariant conformation into representative parameters."""

    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (len(specification.orbit_id), 3) or not np.isfinite(values).all():
        raise ValueError("coordinates must be finite [N,3]")
    values = values - values.mean(axis=0, keepdims=True)
    action = operation_error_numpy(
        values, specification.operation_matrices, specification.permutation_index
    )
    if action["max_atom_error_angstrom"] > tolerance:
        raise ValueError("cannot encode coordinates outside the supplied group action")
    parameters = np.zeros(specification.parameter_count, dtype=np.float64)
    for current, representative in enumerate(specification.representative_indices):
        basis = specification.fixed_bases[current]
        start, stop = specification.parameter_offsets[current : current + 2]
        parameters[int(start) : int(stop)] = basis.T @ values[representative]
        residual = values[representative] - basis @ parameters[int(start) : int(stop)]
        if float(np.max(np.abs(residual), initial=0.0)) > tolerance:
            raise ValueError("representative coordinate violates its stabilizer")
    return parameters

