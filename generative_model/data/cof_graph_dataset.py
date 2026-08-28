"""轻量级、与训练框架无关的 COF 分子图数据加载器。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class COFGraphDataset:
    """读取由 :mod:`build_cof_package` 生成的数据包。

    ``split`` 为 ``train``、``val``、``test`` 或 ``None``；``split_scheme``
    支持 ``iid`` 与 ``core_ood``。返回的 bond_index 采用常见的 ``[2, M]``
    形状，且每条无向化学键只出现一次。
    """

    def __init__(
        self,
        package_dir: str | Path,
        split: str | None = None,
        split_scheme: str = "iid",
    ) -> None:
        self.package_dir = Path(package_dir)
        self.metadata = pd.read_csv(self.package_dir / "metadata.csv")
        # 压缩 NPZ 不支持真正的 mmap；一次解压进内存可避免每次取样重复解压整个数组。
        with np.load(self.package_dir / "cof_graphs.npz", allow_pickle=False) as archive:
            self.arrays = {name: archive[name] for name in archive.files}
        with (self.package_dir / "vocab.json").open(encoding="utf-8") as handle:
            self.vocab = json.load(handle)
        with (self.package_dir / "statistics.json").open(encoding="utf-8") as handle:
            self.statistics = json.load(handle)
        with (self.package_dir / "manifest.json").open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        self._validate_structure()
        if split is None:
            self.indices = np.arange(len(self.metadata), dtype=np.int64)
        else:
            if split not in {"train", "val", "test"}:
                raise ValueError(f"未知 split: {split}")
            split_file = self.package_dir / f"split_{split_scheme}.json"
            if not split_file.exists():
                raise ValueError(f"未知 split_scheme: {split_scheme}")
            with split_file.open(encoding="utf-8") as handle:
                split_data = json.load(handle)
            self.indices = np.asarray(split_data["indices"][split], dtype=np.int64)

    def _validate_structure(self) -> None:
        atom_offsets = self.arrays["atom_offsets"]
        bond_offsets = self.arrays["bond_offsets"]
        expected = len(self.metadata) + 1
        if len(atom_offsets) != expected or len(bond_offsets) != expected:
            raise ValueError("offset 数量与 metadata 行数不一致")
        if atom_offsets[0] != 0 or bond_offsets[0] != 0:
            raise ValueError("offset 必须从 0 开始")
        if np.any(np.diff(atom_offsets) < 0) or np.any(np.diff(bond_offsets) < 0):
            raise ValueError("offset 必须单调递增")
        if atom_offsets[-1] != len(self.arrays["positions"]):
            raise ValueError("atom_offsets 与 positions 长度不一致")
        if bond_offsets[-1] != len(self.arrays["bond_index"]):
            raise ValueError("bond_offsets 与 bond_index 长度不一致")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        package_index = int(self.indices[item])
        atom_start, atom_end = self.arrays["atom_offsets"][package_index : package_index + 2]
        bond_start, bond_end = self.arrays["bond_offsets"][package_index : package_index + 2]
        atom_start, atom_end = int(atom_start), int(atom_end)
        bond_start, bond_end = int(bond_start), int(bond_end)
        row = self.metadata.iloc[package_index]
        return {
            "package_index": package_index,
            "molecule_id": row["Molecule_ID"],
            "atom_types": self.arrays["atom_types"][atom_start:atom_end],
            "atomic_numbers": self.arrays["atomic_numbers"][atom_start:atom_end],
            "formal_charges": self.arrays["formal_charges"][atom_start:atom_end],
            "radical_electrons": self.arrays["radical_electrons"][atom_start:atom_end],
            "positions": self.arrays["positions"][atom_start:atom_end],
            "centroid": self.arrays["centroids"][package_index],
            "bond_index": self.arrays["bond_index"][bond_start:bond_end].T,
            "bond_types": self.arrays["bond_types"][bond_start:bond_end],
            "metadata": row.to_dict(),
        }

    def close(self) -> None:
        """保留上下文管理接口；数组已在初始化时完整载入内存。"""

    def __enter__(self) -> "COFGraphDataset":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def collate_graphs(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """将若干样本拼成训练 batch，并把局部键索引偏移为 batch 全局索引。"""
    if not samples:
        raise ValueError("不能拼接空 batch")
    atom_counts = np.asarray([len(sample["atom_types"]) for sample in samples], dtype=np.int64)
    atom_offsets = np.concatenate(([0], np.cumsum(atom_counts)))
    bond_parts = [sample["bond_index"] + atom_offsets[index] for index, sample in enumerate(samples)]
    return {
        "molecule_ids": [sample["molecule_id"] for sample in samples],
        "atom_types": np.concatenate([sample["atom_types"] for sample in samples]),
        "atomic_numbers": np.concatenate([sample["atomic_numbers"] for sample in samples]),
        "formal_charges": np.concatenate([sample["formal_charges"] for sample in samples]),
        "radical_electrons": np.concatenate([sample["radical_electrons"] for sample in samples]),
        "positions": np.concatenate([sample["positions"] for sample in samples]),
        "centroids": np.stack([sample["centroid"] for sample in samples]),
        "bond_index": np.concatenate(bond_parts, axis=1),
        "bond_types": np.concatenate([sample["bond_types"] for sample in samples]),
        "atom_counts": atom_counts,
        "atom_offsets": atom_offsets,
        "batch_index": np.repeat(np.arange(len(samples), dtype=np.int64), atom_counts),
    }
