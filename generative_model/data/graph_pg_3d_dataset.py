"""Model-independent dataset contract for known graph + Target_PG -> 3D.

The canonical molecular graph is an immutable condition.  A model predicts
coordinates only; target symmetry operations/orbits are supplied as optional
constraints and evaluation metadata.  No element or point-group fallback exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .v2_dataset import COFSymmetryDataset


TARGET_POINT_GROUPS = ("C2", "C3", "S4", "D6h")
TARGET_PG_TO_INDEX = {name: index for index, name in enumerate(TARGET_POINT_GROUPS)}
GRAPH_PG_3D_SCHEMA_VERSION = "graph-pg-to-3d-condition-v1"


def _validate_permutations(
    permutations: np.ndarray,
    atomic_numbers: np.ndarray,
) -> None:
    atom_count = len(atomic_numbers)
    expected = np.arange(atom_count, dtype=np.int64)
    if permutations.ndim != 2 or permutations.shape[1] != atom_count:
        raise ValueError("target symmetry permutation shape 非法")
    for permutation in permutations:
        if not np.array_equal(np.sort(permutation.astype(np.int64)), expected):
            raise ValueError("target symmetry permutation 不是原子双射")
        if not np.array_equal(atomic_numbers, atomic_numbers[permutation]):
            raise ValueError("target symmetry permutation 跨元素映射")


def graph_pg_3d_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert one canonical v2 sample to the strict coordinate-task contract."""

    target_pg = str(sample["target_pg"])
    if target_pg not in TARGET_PG_TO_INDEX:
        raise ValueError(f"不支持的 Target_PG: {target_pg}")
    atom_types = np.asarray(sample["atom_types"], dtype=np.int64)
    atomic_numbers = np.asarray(sample["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(sample["positions"], dtype=np.float32)
    bond_index = np.asarray(sample["bond_index"], dtype=np.int64)
    bond_types = np.asarray(sample["bond_types"], dtype=np.int64)
    atom_count = len(atom_types)
    if positions.shape != (atom_count, 3) or not np.isfinite(positions).all():
        raise ValueError("graph+PG->3D target coordinates 非法")
    if float(np.abs(positions.mean(axis=0)).max()) > 1e-4:
        raise ValueError("graph+PG->3D target coordinates 未质心归零")
    if bond_index.ndim != 2 or bond_index.shape[0] != 2:
        raise ValueError("known sparse bond_index 必须为 [2,M]")
    if bond_index.shape[1] != len(bond_types):
        raise ValueError("known sparse bond_index/bond_types 长度不一致")
    if len(bond_types) and (
        int(bond_index.min()) < 0 or int(bond_index.max()) >= atom_count
    ):
        raise ValueError("known sparse bond_index 越界")
    if np.any((bond_types < 1) | (bond_types > 4)):
        raise ValueError("known sparse bond type 超出 1..4")

    symmetry = sample["target_symmetry"]
    operations = np.asarray(symmetry["operation_matrices"], dtype=np.float32)
    permutations = np.asarray(symmetry["permutation_index"], dtype=np.int64)
    rms_error = np.asarray(symmetry["operation_rms_error"], dtype=np.float32)
    max_error = np.asarray(symmetry["operation_max_error"], dtype=np.float32)
    if operations.ndim != 3 or operations.shape[1:] != (3, 3) or not len(operations):
        raise ValueError("target symmetry operations 必须为 [G,3,3]")
    if len(permutations) != len(operations):
        raise ValueError("target symmetry operations/permutations 数量不一致")
    if rms_error.shape != (len(operations),) or max_error.shape != (len(operations),):
        raise ValueError("target symmetry error shape 非法")
    if not all(np.isfinite(value).all() for value in (operations, rms_error, max_error)):
        raise ValueError("target symmetry annotation 含 NaN/Inf")
    _validate_permutations(permutations, atomic_numbers)

    orbit_id = np.asarray(sample["target_orbit_id"], dtype=np.int64)
    orbit_size = np.asarray(sample["target_orbit_size"], dtype=np.int64)
    if orbit_id.shape != (atom_count,) or orbit_size.shape != (atom_count,):
        raise ValueError("target orbit annotation shape 非法")
    if np.any(orbit_id < 0) or np.any(orbit_size <= 0):
        raise ValueError("target orbit annotation 值非法")

    return {
        "schema_version": GRAPH_PG_3D_SCHEMA_VERSION,
        "package_index": int(sample["package_index"]),
        "molecule_id": str(sample["molecule_id"]),
        "atom_types": atom_types,
        "atomic_numbers": atomic_numbers,
        "formal_charges": np.asarray(sample["formal_charges"], dtype=np.int64),
        "radical_electrons": np.asarray(sample["radical_electrons"], dtype=np.int64),
        "bond_index": bond_index,
        "bond_types": bond_types,
        "target_pg": target_pg,
        "target_pg_index": TARGET_PG_TO_INDEX[target_pg],
        "target_pg_one_hot": np.eye(len(TARGET_POINT_GROUPS), dtype=np.float32)[
            TARGET_PG_TO_INDEX[target_pg]
        ],
        "target_positions": positions,
        "target_operation_matrices": operations,
        "target_permutation_index": permutations,
        "target_operation_rms_error": rms_error,
        "target_operation_max_error": max_error,
        "target_orbit_id": orbit_id,
        "target_orbit_size": orbit_size,
    }


class GraphPG3DDataset:
    """Strict v2 view for graph-conditioned, point-group-conditioned 3D models."""

    def __init__(
        self,
        package_dir: str | Path,
        split: str | None = None,
        split_scheme: str = "iid",
    ) -> None:
        self.canonical = COFSymmetryDataset(
            package_dir, split=split, split_scheme=split_scheme
        )
        self.package_dir = self.canonical.package_dir
        self.indices = self.canonical.indices
        self.manifest = self.canonical.manifest
        self.split = split
        self.split_scheme = split_scheme

    def __len__(self) -> int:
        return len(self.canonical)

    def __getitem__(self, item: int) -> dict[str, Any]:
        return graph_pg_3d_sample(self.canonical[item])


def collate_graph_pg_3d(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pack variable-size graphs and symmetry actions without dense N x N bonds."""

    if not samples:
        raise ValueError("不能拼接空 graph+PG->3D batch")
    if any(sample.get("schema_version") != GRAPH_PG_3D_SCHEMA_VERSION for sample in samples):
        raise ValueError("graph+PG->3D sample schema 不一致")
    atom_counts = np.asarray([len(sample["atom_types"]) for sample in samples], dtype=np.int64)
    atom_offsets = np.concatenate(([0], np.cumsum(atom_counts)))
    bond_parts = [
        sample["bond_index"] + atom_offsets[index]
        for index, sample in enumerate(samples)
    ]
    operation_counts = np.asarray(
        [len(sample["target_operation_matrices"]) for sample in samples], dtype=np.int64
    )
    operation_offsets = np.concatenate(([0], np.cumsum(operation_counts)))
    permutation_parts = []
    permutation_offsets = [0]
    for sample in samples:
        for permutation in sample["target_permutation_index"]:
            permutation_parts.append(np.asarray(permutation, dtype=np.int64))
            permutation_offsets.append(permutation_offsets[-1] + len(permutation))
    return {
        "schema_version": GRAPH_PG_3D_SCHEMA_VERSION,
        "package_indices": np.asarray([sample["package_index"] for sample in samples], dtype=np.int64),
        "molecule_ids": [sample["molecule_id"] for sample in samples],
        "atom_types": np.concatenate([sample["atom_types"] for sample in samples]),
        "atomic_numbers": np.concatenate([sample["atomic_numbers"] for sample in samples]),
        "formal_charges": np.concatenate([sample["formal_charges"] for sample in samples]),
        "radical_electrons": np.concatenate([sample["radical_electrons"] for sample in samples]),
        "bond_index": np.concatenate(bond_parts, axis=1),
        "bond_types": np.concatenate([sample["bond_types"] for sample in samples]),
        "target_positions": np.concatenate([sample["target_positions"] for sample in samples]),
        "target_pg_indices": np.asarray([sample["target_pg_index"] for sample in samples], dtype=np.int64),
        "target_pg_one_hot": np.stack([sample["target_pg_one_hot"] for sample in samples]),
        "target_operation_matrices": np.concatenate(
            [sample["target_operation_matrices"] for sample in samples]
        ),
        "target_permutation_index": np.concatenate(permutation_parts),
        "target_operation_rms_error": np.concatenate(
            [sample["target_operation_rms_error"] for sample in samples]
        ),
        "target_operation_max_error": np.concatenate(
            [sample["target_operation_max_error"] for sample in samples]
        ),
        "target_orbit_id": np.concatenate([sample["target_orbit_id"] for sample in samples]),
        "target_orbit_size": np.concatenate([sample["target_orbit_size"] for sample in samples]),
        "atom_counts": atom_counts,
        "atom_offsets": atom_offsets,
        "batch_index": np.repeat(np.arange(len(samples), dtype=np.int64), atom_counts),
        "operation_counts": operation_counts,
        "operation_offsets": operation_offsets,
        "permutation_offsets": np.asarray(permutation_offsets, dtype=np.int64),
    }
