"""Deterministic, pickle-free storage for factorized automorphism contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .factorized_automorphism import FactorizedAutomorphismContract


SCHEMA_VERSION = "pg-orbitflow-factorized-contract-cache-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_fingerprint(records: list[dict]) -> str:
    identity = [
        {
            "package_index": int(row["package_index"]),
            "molecule_id": row["molecule_id"],
            "target_pg": row["target_pg"],
            "num_atoms": int(row["num_atoms"]),
        }
        for row in records
    ]
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def cache_path(directory: Path, package_index: int) -> Path:
    return directory / f"{int(package_index):06d}.npz"


def save_contract(path: Path, contract: FactorizedAutomorphismContract) -> None:
    groups = tuple(tuple(int(value) for value in group) for group in contract.groups)
    offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    if groups:
        offsets[1:] = np.cumsum([len(group) for group in groups])
        values = np.asarray([value for group in groups for value in group], dtype=np.int64)
    else:
        values = np.empty(0, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(SCHEMA_VERSION),
            group_offsets=offsets,
            group_values=values,
            representative_tuples=np.asarray(contract.representative_tuples, dtype=np.int64),
            allowed_torsion_permutations=np.asarray(
                contract.allowed_torsion_permutations, dtype=np.int64
            ),
            induced_generator_count=np.asarray(contract.induced_generator_count, dtype=np.int64),
            allowed_torsion_permutation_count=np.asarray(
                contract.allowed_torsion_permutation_count, dtype=np.int64
            ),
            branch_swap_witness_count=np.asarray(
                contract.branch_swap_witness_count, dtype=np.int64
            ),
            anchored_witness_count=np.asarray(
                contract.anchored_witness_count, dtype=np.int64
            ),
        )
    temporary.replace(path)


def load_contract(path: Path) -> FactorizedAutomorphismContract:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema_version"].item()) != SCHEMA_VERSION:
            raise ValueError("unsupported factorized contract cache schema")
        offsets = np.asarray(archive["group_offsets"], dtype=np.int64)
        values = np.asarray(archive["group_values"], dtype=np.int64)
        if offsets.ndim != 1 or not len(offsets) or offsets[0] != 0:
            raise ValueError("invalid cached group offsets")
        if offsets[-1] != len(values) or np.any(np.diff(offsets) < 0):
            raise ValueError("invalid cached group storage")
        groups = tuple(
            tuple(int(value) for value in values[offsets[index] : offsets[index + 1]])
            for index in range(len(offsets) - 1)
        )
        result = FactorizedAutomorphismContract(
            groups=groups,
            representative_tuples=np.asarray(archive["representative_tuples"], dtype=np.int64),
            allowed_torsion_permutations=np.asarray(
                archive["allowed_torsion_permutations"], dtype=np.int64
            ),
            induced_generator_count=int(archive["induced_generator_count"].item()),
            allowed_torsion_permutation_count=int(
                archive["allowed_torsion_permutation_count"].item()
            ),
            branch_swap_witness_count=int(archive["branch_swap_witness_count"].item()),
            anchored_witness_count=int(archive["anchored_witness_count"].item()),
        )
    permutations = result.allowed_torsion_permutations
    if permutations.ndim != 2 or not len(permutations):
        raise ValueError("cached torsion permutations must be non-empty 2D")
    if result.allowed_torsion_permutation_count != len(permutations):
        raise ValueError("cached torsion permutation count mismatch")
    if result.representative_tuples.shape[0] != permutations.shape[1]:
        raise ValueError("cached torsion support mismatch")
    return result
