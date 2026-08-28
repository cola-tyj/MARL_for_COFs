"""v2 canonical + symmetry + RDKit 衍生特征加载器。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .cof_graph_dataset import COFGraphDataset


class COFSymmetryDataset(COFGraphDataset):
    """按 molecule index 联合读取 canonical、symmetry 和 RDKit feature layers。"""

    def __init__(self, package_dir: str | Path, split: str | None = None, split_scheme: str = "iid") -> None:
        super().__init__(package_dir, split=split, split_scheme=split_scheme)
        with np.load(self.package_dir / "symmetry_data.npz", allow_pickle=False) as archive:
            self.symmetry = {name: archive[name] for name in archive.files}
        with np.load(self.package_dir / "rdkit_features.npz", allow_pickle=False) as archive:
            self.features = {name: archive[name] for name in archive.files}
        self.symmetry_vocab = json.loads(
            (self.package_dir / "symmetry_vocab.json").read_text(encoding="utf-8")
        )
        self.feature_vocab = json.loads(
            (self.package_dir / "feature_vocab.json").read_text(encoding="utf-8")
        )
        if not np.array_equal(self.arrays["atom_offsets"], self.symmetry["atom_offsets"]):
            raise ValueError("symmetry_data 与 canonical atom_offsets 不一致")
        if not np.array_equal(self.arrays["atom_offsets"], self.features["atom_offsets"]):
            raise ValueError("rdkit_features 与 canonical atom_offsets 不一致")

    def _operation_block(self, key: str, molecule_index: int, atom_count: int) -> dict[str, np.ndarray]:
        operation_start, operation_end = map(
            int, self.symmetry[f"{key}_operation_offsets"][molecule_index : molecule_index + 2]
        )
        permutation_offsets = self.symmetry[f"{key}_permutation_offsets"]
        permutation_start = int(permutation_offsets[operation_start])
        permutation_end = int(permutation_offsets[operation_end])
        permutations = self.symmetry[f"{key}_permutation_index"][permutation_start:permutation_end]
        return {
            "operation_matrices": self.symmetry[f"{key}_operation_matrices"][operation_start:operation_end],
            "permutation_index": permutations.reshape(operation_end - operation_start, atom_count),
            "operation_rms_error": self.symmetry[f"{key}_operation_rms_error"][operation_start:operation_end],
            "operation_max_error": self.symmetry[f"{key}_operation_max_error"][operation_start:operation_end],
        }

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = super().__getitem__(item)
        molecule_index = sample["package_index"]
        atom_start, atom_end = map(
            int, self.arrays["atom_offsets"][molecule_index : molecule_index + 2]
        )
        bond_start, bond_end = map(
            int, self.arrays["bond_offsets"][molecule_index : molecule_index + 2]
        )
        point_groups = self.symmetry_vocab["point_groups"]
        sample.update(
            {
                "target_pg": point_groups[int(self.symmetry["target_pg"][molecule_index])],
                "analyzer_pg": point_groups[int(self.symmetry["analyzer_pg"][molecule_index])],
                "actual_pg": point_groups[int(self.symmetry["actual_pg"][molecule_index])],
                "pg_exact_match": bool(self.symmetry["pg_exact_match"][molecule_index]),
                "pg_compatible": bool(self.symmetry["pg_compatible"][molecule_index]),
                "orbit_id": self.symmetry["actual_orbit_id"][atom_start:atom_end],
                "orbit_size": self.symmetry["actual_orbit_size"][atom_start:atom_end],
                "target_orbit_id": self.symmetry["target_orbit_id"][atom_start:atom_end],
                "target_orbit_size": self.symmetry["target_orbit_size"][atom_start:atom_end],
                "actual_symmetry": self._operation_block("actual", molecule_index, atom_end - atom_start),
                "target_symmetry": self._operation_block("target", molecule_index, atom_end - atom_start),
                "hybridization": self.features["hybridization"][atom_start:atom_end],
                "aromaticity": self.features["aromaticity"][atom_start:atom_end],
                "chirality": self.features["chirality"][atom_start:atom_end],
                "degree": self.features["degree"][atom_start:atom_end],
                "explicit_h_neighbor_count": self.features["explicit_h_neighbor_count"][atom_start:atom_end],
                "ring_membership": self.features["ring_membership"][atom_start:atom_end],
                "bond_stereo": self.features["bond_stereo"][bond_start:bond_end],
                "bond_conjugation": self.features["bond_conjugation"][bond_start:bond_end],
                "bond_ring_membership": self.features["bond_ring_membership"][bond_start:bond_end],
            }
        )
        return sample
