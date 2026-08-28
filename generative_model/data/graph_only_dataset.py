"""Canonical molecular graph loader that never loads coordinate arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GRAPH_ARRAY_NAMES = (
    "atom_offsets", "bond_offsets", "atomic_numbers", "formal_charges",
    "radical_electrons", "bond_index", "bond_types",
)


class CanonicalGraphOnlyDataset:
    """Load graph ground truth while making coordinate access impossible."""

    def __init__(self, package_dir: str | Path) -> None:
        self.package_dir = Path(package_dir)
        self.metadata = pd.read_csv(self.package_dir / "metadata.csv")
        with np.load(self.package_dir / "cof_graphs.npz", allow_pickle=False) as archive:
            missing = sorted(set(GRAPH_ARRAY_NAMES) - set(archive.files))
            if missing:
                raise ValueError(f"canonical graph archive 缺字段: {missing}")
            self.arrays = {name: archive[name] for name in GRAPH_ARRAY_NAMES}
        if set(self.arrays) != set(GRAPH_ARRAY_NAMES):
            raise RuntimeError("graph-only loader 意外加载额外数组")
        self._validate()

    @property
    def loaded_array_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.arrays))

    def _validate(self) -> None:
        molecule_count = len(self.metadata)
        atoms = self.arrays["atom_offsets"]; bonds = self.arrays["bond_offsets"]
        if len(atoms) != molecule_count + 1 or len(bonds) != molecule_count + 1:
            raise ValueError("graph-only offsets 与 metadata 不一致")
        if int(atoms[0]) != 0 or int(bonds[0]) != 0:
            raise ValueError("graph-only offsets 必须从零开始")
        if np.any(np.diff(atoms) < 0) or np.any(np.diff(bonds) < 0):
            raise ValueError("graph-only offsets 非单调")
        if int(atoms[-1]) != len(self.arrays["atomic_numbers"]):
            raise ValueError("graph-only atom offsets 不匹配")
        if int(bonds[-1]) != len(self.arrays["bond_index"]):
            raise ValueError("graph-only bond offsets 不匹配")
        if any(
            len(self.arrays[name]) != len(self.arrays["atomic_numbers"])
            for name in ("formal_charges", "radical_electrons")
        ):
            raise ValueError("graph-only atom feature 长度不匹配")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, package_index: int) -> dict[str, Any]:
        index = int(package_index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        atom_start, atom_end = map(
            int, self.arrays["atom_offsets"][index : index + 2]
        )
        bond_start, bond_end = map(
            int, self.arrays["bond_offsets"][index : index + 2]
        )
        row = self.metadata.iloc[index]
        return {
            "package_index": index,
            "molecule_id": str(row["Molecule_ID"]),
            "target_pg": str(row["Target_PG"]),
            "atomic_numbers": self.arrays["atomic_numbers"][atom_start:atom_end],
            "formal_charges": self.arrays["formal_charges"][atom_start:atom_end],
            "radical_electrons": self.arrays["radical_electrons"][atom_start:atom_end],
            "bond_index": self.arrays["bond_index"][bond_start:bond_end].T,
            "bond_types": self.arrays["bond_types"][bond_start:bond_end],
            "metadata": row.to_dict(),
        }
