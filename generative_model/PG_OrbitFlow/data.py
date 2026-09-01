"""Frozen-v2 data adapter for PG-OrbitFlow.

Only this module knows about the canonical package.  It never writes back to
v2 and never imports the historical our_ET_Flow inference branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from generative_model.data.graph_pg_3d_dataset import GraphPG3DDataset

from .group import (
    group_features,
    operation_error_numpy,
    project_vectors_numpy,
    validate_group_action,
)
from .quotient import QuotientGraph, build_quotient_graph, symmetry_invariant_bond_order


DATA_SCHEMA_VERSION = "pg-orbitflow-frozen-v2-adapter-v1"
POINT_GROUPS = ("C2", "C3", "S4", "D6h")


@dataclass(frozen=True)
class OrbitFlowSample:
    package_index: int
    molecule_id: str
    core_name: str
    arm_name: str
    target_pg: str
    target_pg_index: int
    atom_types: np.ndarray
    atomic_numbers: np.ndarray
    formal_charges: np.ndarray
    radical_electrons: np.ndarray
    bond_index: np.ndarray
    bond_types: np.ndarray
    target_positions_angstrom: np.ndarray
    symmetric_target_angstrom: np.ndarray
    operation_matrices: np.ndarray
    permutation_index: np.ndarray
    orbit_id: np.ndarray
    orbit_size: np.ndarray
    group_features: np.ndarray
    atom_group_features: np.ndarray
    invariant_bond_order: np.ndarray
    quotient_graph: QuotientGraph
    target_projection_rmsd_angstrom: float


def _rmsd(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(left - right), axis=1))))


def make_orbitflow_sample(raw: dict) -> OrbitFlowSample:
    target_pg = str(raw["target_pg"])
    if target_pg not in POINT_GROUPS:
        raise ValueError(f"unsupported target point group: {target_pg}")
    atom_types = np.asarray(raw["atom_types"], dtype=np.int64)
    atomic_numbers = np.asarray(raw["atomic_numbers"], dtype=np.int64)
    charges = np.asarray(raw["formal_charges"], dtype=np.int64)
    radicals = np.asarray(raw["radical_electrons"], dtype=np.int64)
    positions = np.asarray(raw["target_positions"], dtype=np.float64)
    bonds = np.asarray(raw["bond_index"], dtype=np.int64)
    bond_types = np.asarray(raw["bond_types"], dtype=np.int64)
    matrices = np.asarray(raw["target_operation_matrices"], dtype=np.float64)
    permutations = np.asarray(raw["target_permutation_index"], dtype=np.int64)
    orbit_id = np.asarray(raw["target_orbit_id"], dtype=np.int64)
    orbit_size = np.asarray(raw["target_orbit_size"], dtype=np.int64)
    atom_count = len(atom_types)
    if atom_types.shape != (atom_count,) or np.any((atom_types < 0) | (atom_types >= 13)):
        raise ValueError("atom type vocabulary must be the frozen 13-element indices")
    if any(field.shape != (atom_count,) for field in (atomic_numbers, charges, radicals, orbit_size)):
        raise ValueError("atom field length mismatch")
    if positions.shape != (atom_count, 3) or not np.isfinite(positions).all():
        raise ValueError("target positions must be finite [N,3]")
    if np.any((charges < -4) | (charges > 4)):
        raise ValueError("formal charge outside explicit model vocabulary [-4,4]")
    if np.any((radicals < 0) | (radicals > 4)):
        raise ValueError("radical count outside explicit model vocabulary [0,4]")
    audit = validate_group_action(
        matrices,
        permutations,
        atomic_numbers=atomic_numbers,
        orbit_id=orbit_id,
    )
    if audit.atom_count != atom_count:
        raise ValueError("group action atom count mismatch")
    expected_orbit_size = np.bincount(orbit_id)[orbit_id]
    if not np.array_equal(orbit_size, expected_orbit_size):
        raise ValueError("orbit_size is inconsistent with orbit_id")

    centered = positions - positions.mean(axis=0, keepdims=True)
    symmetric_target = project_vectors_numpy(centered, matrices, permutations)
    target_error = operation_error_numpy(symmetric_target, matrices, permutations)
    if target_error["max_atom_error_angstrom"] > 1e-5:
        raise RuntimeError("projected training target is not group invariant")
    quotient = build_quotient_graph(
        atom_count=atom_count,
        bond_index=bonds,
        bond_types=bond_types,
        orbit_id=orbit_id,
        operation_matrices=matrices,
        permutation_index=permutations,
    )
    global_group, atom_group = group_features(matrices, permutations, orbit_id)
    invariant_bonds = symmetry_invariant_bond_order(
        atom_count, bonds, bond_types, permutations
    )
    # The feature itself must commute with every atom permutation.
    for row in permutations:
        inverse = np.argsort(row)
        if float(np.max(np.abs(invariant_bonds - invariant_bonds[np.ix_(inverse, inverse)]))) > 1e-5:
            raise RuntimeError("symmetrized bond-order feature breaks the group action")

    metadata = raw.get("metadata", {})
    return OrbitFlowSample(
        package_index=int(raw["package_index"]),
        molecule_id=str(raw["molecule_id"]),
        core_name=str(metadata.get("Core", "")),
        arm_name=str(metadata.get("Arm", "")),
        target_pg=target_pg,
        target_pg_index=POINT_GROUPS.index(target_pg),
        atom_types=atom_types,
        atomic_numbers=atomic_numbers,
        formal_charges=charges,
        radical_electrons=radicals,
        bond_index=bonds,
        bond_types=bond_types,
        target_positions_angstrom=centered.astype(np.float32),
        symmetric_target_angstrom=symmetric_target.astype(np.float32),
        operation_matrices=matrices.astype(np.float32),
        permutation_index=permutations,
        orbit_id=orbit_id,
        orbit_size=orbit_size,
        group_features=global_group,
        atom_group_features=atom_group,
        invariant_bond_order=invariant_bonds,
        quotient_graph=quotient,
        target_projection_rmsd_angstrom=_rmsd(centered, symmetric_target),
    )


class PGOrbitFlowDataset:
    """Filtered strict view of the frozen IID/Core-OOD split."""

    def __init__(
        self,
        package_dir: str | Path,
        *,
        split: str,
        split_scheme: str = "iid",
        point_groups: tuple[str, ...] = ("C2", "C3"),
        max_samples: int | None = None,
    ) -> None:
        unknown = sorted(set(point_groups) - set(POINT_GROUPS))
        if unknown:
            raise ValueError(f"unsupported point groups: {unknown}")
        canonical = GraphPG3DDataset(package_dir, split=split, split_scheme=split_scheme)
        selected: list[int] = []
        buckets: dict[str, list[int]] = {name: [] for name in point_groups}
        for local_index in range(len(canonical)):
            raw = canonical[local_index]
            if raw["target_pg"] in point_groups:
                selected.append(local_index)
                buckets[raw["target_pg"]].append(local_index)
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            # A bounded engineering panel must not silently become C2-only just
            # because canonical indices are class-blocked.  Select groups in a
            # deterministic round-robin; the full dataset keeps canonical order.
            selected = []
            cursor = {name: 0 for name in point_groups}
            while len(selected) < max_samples:
                progressed = False
                for name in point_groups:
                    if cursor[name] < len(buckets[name]) and len(selected) < max_samples:
                        selected.append(buckets[name][cursor[name]])
                        cursor[name] += 1
                        progressed = True
                if not progressed:
                    break
        if not selected:
            raise ValueError("PG-OrbitFlow dataset selection is empty")
        self.canonical = canonical
        self.local_indices = np.asarray(selected, dtype=np.int64)
        self.package_dir = Path(package_dir)
        self.split = split
        self.split_scheme = split_scheme
        self.point_groups = tuple(point_groups)

    def __len__(self) -> int:
        return len(self.local_indices)

    def __getitem__(self, item: int) -> OrbitFlowSample:
        raw = dict(self.canonical[int(self.local_indices[item])])
        raw["metadata"] = self.canonical.canonical.metadata.iloc[
            int(raw["package_index"])
        ].to_dict()
        return make_orbitflow_sample(raw)


class PointGroupBalancedSampler:
    """Deterministic infinite sampler with uniform exposure over selected PGs."""

    def __init__(self, dataset: PGOrbitFlowDataset, *, seed: int) -> None:
        buckets: dict[str, list[int]] = {name: [] for name in dataset.point_groups}
        for index in range(len(dataset)):
            target_pg = dataset.canonical[int(dataset.local_indices[index])]["target_pg"]
            buckets[target_pg].append(index)
        if any(not values for values in buckets.values()):
            raise ValueError("balanced sampler requires every selected PG to have samples")
        self.buckets = buckets
        self.groups = tuple(dataset.point_groups)
        self.rng = np.random.default_rng(int(seed))
        self.cursor = 0

    def __iter__(self) -> Iterator[int]:
        while True:
            group = self.groups[self.cursor % len(self.groups)]
            self.cursor += 1
            yield int(self.rng.choice(self.buckets[group]))

    def state_dict(self) -> dict:
        return {"cursor": self.cursor, "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        if set(state) != {"cursor", "rng_state"}:
            raise ValueError("balanced sampler state schema mismatch")
        self.cursor = int(state["cursor"])
        self.rng.bit_generator.state = state["rng_state"]


def make_symmetric_noise(
    sample: OrbitFlowSample,
    *,
    seed: int,
    coordinate_scale_angstrom: float,
    prior_type: str = "projected_gaussian",
    harmonic_alpha: float = 1.0,
) -> np.ndarray:
    """Draw an input prior inside the target action's invariant subspace."""

    if coordinate_scale_angstrom <= 0:
        raise ValueError("coordinate_scale_angstrom must be positive")
    if prior_type not in {"projected_gaussian", "graph_harmonic"}:
        raise ValueError(f"unsupported prior_type: {prior_type}")
    if not np.isfinite(harmonic_alpha) or harmonic_alpha <= 0:
        raise ValueError("harmonic_alpha must be finite and positive")
    rng = np.random.default_rng(int(seed))
    eigenvalues = eigenvectors = None
    if prior_type == "graph_harmonic":
        laplacian = graph_laplacian(sample, alpha=harmonic_alpha)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
            raise RuntimeError("graph Laplacian eigendecomposition produced NaN/Inf")
        if float(eigenvalues.min()) < -1e-10:
            raise RuntimeError("graph Laplacian is not positive semidefinite")
        zero_modes = int(np.count_nonzero(eigenvalues <= 1e-8))
        if zero_modes != 1:
            raise ValueError(
                "graph_harmonic prior requires one connected component and one zero mode"
            )
        inverse_sqrt = np.zeros_like(eigenvalues)
        inverse_sqrt[eigenvalues > 1e-8] = 1.0 / np.sqrt(
            eigenvalues[eigenvalues > 1e-8]
        )
    for _ in range(8):
        white = rng.normal(size=(len(sample.atom_types), 3))
        if prior_type == "graph_harmonic":
            noise = eigenvectors @ (inverse_sqrt[:, None] * white)
        else:
            noise = white
        noise = project_vectors_numpy(noise, sample.operation_matrices, sample.permutation_index)
        noise -= noise.mean(axis=0, keepdims=True)
        rms = float(np.sqrt(np.mean(np.sum(np.square(noise), axis=1))))
        if rms > 1e-6:
            # Preserve the Gaussian radial law for graph-harmonic sampling so
            # its Laplacian density can be used by centralizer-averaged flow.
            values = noise if prior_type == "graph_harmonic" else noise / rms
            error = operation_error_numpy(
                values, sample.operation_matrices, sample.permutation_index
            )
            if error["max_atom_error_angstrom"] > 1e-5:
                raise RuntimeError("symmetric prior construction failed")
            return values.astype(np.float32)
    raise RuntimeError("unable to draw a non-degenerate symmetric prior")


def graph_laplacian(sample: OrbitFlowSample, *, alpha: float = 1.0) -> np.ndarray:
    """Return the strict unweighted canonical graph Laplacian.

    The Laplacian must commute with every supplied atom permutation.  This is
    a hard contract: a topology/action mismatch is never repaired or ignored.
    """

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    atom_count = len(sample.atom_types)
    bonds = np.asarray(sample.bond_index, dtype=np.int64)
    if bonds.ndim != 2 or bonds.shape[0] != 2:
        raise ValueError("bond_index must be [2,M]")
    adjacency = np.zeros((atom_count, atom_count), dtype=np.float64)
    for left, right in bonds.T:
        left, right = int(left), int(right)
        if not 0 <= left < atom_count or not 0 <= right < atom_count or left == right:
            raise ValueError("invalid canonical bond endpoint")
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    if not adjacency.any():
        raise ValueError("graph_harmonic prior requires at least one bond")
    laplacian = float(alpha) * (np.diag(adjacency.sum(axis=1)) - adjacency)
    if float(np.max(np.abs(laplacian - laplacian.T))) > 1e-12:
        raise RuntimeError("graph Laplacian is not symmetric")
    for row in np.asarray(sample.permutation_index, dtype=np.int64):
        transformed = laplacian[np.ix_(row, row)]
        if float(np.max(np.abs(laplacian - transformed))) > 1e-10:
            raise ValueError("graph Laplacian does not commute with group permutation")
    return laplacian
