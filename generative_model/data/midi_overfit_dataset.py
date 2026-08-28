"""冻结 MiDi overfit 数据包的严格 PyG 加载器。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_ARRAYS = {
    "atom_offsets",
    "edge_offsets",
    "atom_types",
    "formal_charges",
    "positions",
    "edge_index",
    "edge_types",
    "package_indices",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_sha(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        encoded_name = name.encode("utf-8")
        encoded_dtype = array.dtype.str.encode("ascii")
        digest.update(len(encoded_name).to_bytes(2, "little"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(2, "little"))
        digest.update(encoded_dtype)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


class MiDiOverfitDataset:
    """读取确定性 MiDi 数据包，并按 split 返回官方语义的 PyG ``Data``。

    ``split`` 在 overfit gate 中故意令 train/val/test 相同；该类只允许读取 manifest
    明确声明为 overfit 的包，防止其被误用作泛化实验。
    """

    def __init__(
        self,
        package_dir: str | Path,
        split: str = "train",
        *,
        verify_hashes: bool = True,
    ) -> None:
        self.package_dir = Path(package_dir).resolve()
        manifest_path = self.package_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"缺少 MiDi manifest: {manifest_path}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "PASS":
            raise ValueError("MiDi data manifest 未标记为 PASS")
        if "overfit" not in str(self.manifest.get("purpose", "")).lower():
            raise ValueError("MiDiOverfitDataset 拒绝读取非 overfit 数据包")
        if self.manifest.get("edge_storage") != (
            "directed sparse PyG semantics; each canonical undirected bond stored twice"
        ):
            raise ValueError("未知的 MiDi edge storage 约定")

        if verify_hashes:
            for name, record in self.manifest.get("files", {}).items():
                path = self.package_dir / name
                if not path.is_file():
                    raise FileNotFoundError(f"manifest 所列文件不存在: {path}")
                if path.stat().st_size != int(record["bytes"]):
                    raise ValueError(f"{name} 字节数与 manifest 不一致")
                if _sha256_file(path) != record["sha256"]:
                    raise ValueError(f"{name} SHA-256 与 manifest 不一致")

        graph_path = self.package_dir / "midi_graphs.npz"
        with np.load(graph_path, allow_pickle=False) as archive:
            if set(archive.files) != REQUIRED_ARRAYS:
                raise ValueError(
                    f"midi_graphs.npz schema 不一致: {sorted(archive.files)}"
                )
            self.arrays = {name: archive[name] for name in archive.files}
        if _semantic_sha(self.arrays) != self.manifest.get("semantic_sha256"):
            raise ValueError("midi_graphs.npz semantic SHA-256 不一致")

        self.metadata: list[dict[str, Any]] = json.loads(
            (self.package_dir / "metadata.json").read_text(encoding="utf-8")
        )
        split_data = json.loads(
            (self.package_dir / "overfit_split.json").read_text(encoding="utf-8")
        )
        if split not in split_data.get("indices", {}):
            raise ValueError(f"未知 MiDi split: {split}")
        self.indices = list(map(int, split_data["indices"][split]))
        self.split = split
        self._validate()

    def _validate(self) -> None:
        arrays = self.arrays
        molecule_count = int(self.manifest["molecule_count"])
        if molecule_count != len(self.metadata) or molecule_count != len(arrays["package_indices"]):
            raise ValueError("manifest、metadata 与 package_indices 分子数不一致")
        if not np.array_equal(
            arrays["package_indices"],
            np.asarray(self.manifest["ordered_package_indices"], dtype=np.int64),
        ):
            raise ValueError("package_indices 与 manifest 顺序不一致")
        for key, total in (
            ("atom_offsets", len(arrays["atom_types"])),
            ("edge_offsets", len(arrays["edge_types"])),
        ):
            offsets = arrays[key]
            if offsets.dtype != np.int64 or offsets.shape != (molecule_count + 1,):
                raise ValueError(f"{key} dtype/shape 错误")
            if offsets[0] != 0 or offsets[-1] != total or np.any(np.diff(offsets) < 0):
                raise ValueError(f"{key} 非法")
        if arrays["positions"].shape != (len(arrays["atom_types"]), 3):
            raise ValueError("positions shape 错误")
        if arrays["edge_index"].shape != (2, len(arrays["edge_types"])):
            raise ValueError("edge_index shape 错误")
        if not np.isfinite(arrays["positions"]).all():
            raise ValueError("positions 含 NaN/Inf")
        if np.any(arrays["atom_types"] >= len(self.manifest["atom_vocabulary"])):
            raise ValueError("atom type index 越界")
        if not set(map(int, arrays["formal_charges"])).issubset(
            set(map(int, self.manifest["formal_charge_vocabulary"]))
        ):
            raise ValueError("formal charge 超出 vocabulary")
        if np.any((arrays["edge_types"] < 1) | (arrays["edge_types"] > 4)):
            raise ValueError("sparse edge type 只允许 1..4")
        if any(index < 0 or index >= molecule_count for index in self.indices):
            raise ValueError("split index 越界")

        for local_index in range(molecule_count):
            atom_start, atom_end = map(
                int, arrays["atom_offsets"][local_index : local_index + 2]
            )
            edge_start, edge_end = map(
                int, arrays["edge_offsets"][local_index : local_index + 2]
            )
            coordinates = arrays["positions"][atom_start:atom_end]
            if len(coordinates) == 0 or np.max(np.abs(coordinates.mean(axis=0))) >= 1e-3:
                raise ValueError(f"local index {local_index} 坐标未质心归零")
            edges = arrays["edge_index"][:, edge_start:edge_end]
            kinds = arrays["edge_types"][edge_start:edge_end]
            atom_count = atom_end - atom_start
            if edges.size and (edges.min() < 0 or edges.max() >= atom_count):
                raise ValueError(f"local index {local_index} edge index 越界")
            directed: dict[tuple[int, int], int] = {}
            for pair, kind in zip(edges.T, kinds):
                begin, end = map(int, pair)
                key = (begin, end)
                if begin == end or key in directed:
                    raise ValueError(f"local index {local_index} 含 self-loop/重复边")
                directed[key] = int(kind)
            for (begin, end), kind in directed.items():
                if directed.get((end, begin)) != kind:
                    raise ValueError(f"local index {local_index} 双向边不一致")
            if len(directed) % 2:
                raise ValueError(f"local index {local_index} 有向边数不是偶数")
            record = self.metadata[local_index]
            if int(record["local_index"]) != local_index:
                raise ValueError("metadata local_index 顺序不一致")
            if int(record["package_index"]) != int(arrays["package_indices"][local_index]):
                raise ValueError("metadata package_index 不一致")
            if int(record["atoms"]) != atom_count or int(record["directed_edges"]) != len(kinds):
                raise ValueError("metadata 原子/边统计不一致")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        import torch
        from torch_geometric.data import Data

        local_index = self.indices[item]
        atom_start, atom_end = map(
            int, self.arrays["atom_offsets"][local_index : local_index + 2]
        )
        edge_start, edge_end = map(
            int, self.arrays["edge_offsets"][local_index : local_index + 2]
        )
        record = self.metadata[local_index]
        return Data(
            x=torch.as_tensor(
                self.arrays["atom_types"][atom_start:atom_end].astype(np.int64)
            ),
            charges=torch.as_tensor(
                self.arrays["formal_charges"][atom_start:atom_end].astype(np.int64)
            ),
            pos=torch.as_tensor(
                self.arrays["positions"][atom_start:atom_end].astype(np.float32)
            ),
            edge_index=torch.as_tensor(
                self.arrays["edge_index"][:, edge_start:edge_end].astype(np.int64)
            ),
            edge_attr=torch.as_tensor(
                self.arrays["edge_types"][edge_start:edge_end].astype(np.int64)
            ),
            smiles=str(record["smiles"]),
            molecule_id=str(record["molecule_id"]),
            package_index=int(record["package_index"]),
            target_pg=str(record["target_pg"]),
        )

