"""从同一个 v2 canonical dataset 动态构造不同生成模型需要的表示。"""

from __future__ import annotations

from copy import copy
from typing import Any

import numpy as np

from .schema import ATOM_SYMBOLS
from .v2_dataset import COFSymmetryDataset

UAE_GEOM_VOCAB = (
    "H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi"
)
UAE_EXTENDED_VOCAB = (*UAE_GEOM_VOCAB, "Sn")
MIDI_GEOM_VOCAB = UAE_GEOM_VOCAB
MIDI_EXTENDED_VOCAB = (*MIDI_GEOM_VOCAB, "Sn")
MIDI_CHARGE_TO_INDEX = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5}

# SemlaFlow 官方 build_vocab() 顺序；0/1 是特殊 token，真实元素从 2 开始。
SEMLAFLOW_VOCAB = (
    "<PAD>", "<MASK>", "H", "C", "N", "O", "F", "P", "S", "Cl",
    "Br", "B", "Al", "Si", "As", "I", "Hg", "Bi",
)
SEMLAFLOW_CHARGE_TO_INDEX = {0: 0, 1: 1, 2: 2, 3: 3, -1: 4, -2: 5, -3: 6}
SEMLAFLOW_GEOM_COORD_STD = 2.407038688659668


def full_pair_bond_types(sample: dict[str, Any]) -> np.ndarray:
    """动态生成 [N,N] 键类别；0=none，1/2/3/4=single/double/triple/aromatic。"""
    atom_count = len(sample["atom_types"])
    dense = np.zeros((atom_count, atom_count), dtype=np.uint8)
    if sample["bond_index"].size:
        begin, end = sample["bond_index"]
        dense[begin, end] = sample["bond_types"]
        dense[end, begin] = sample["bond_types"]
    return dense


def _renumber_orbits(orbit_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(map(int, orbit_id)))
    mapping = {old: new for new, old in enumerate(unique)}
    renumbered = np.asarray([mapping[int(old)] for old in orbit_id], dtype=np.uint16)
    counts = np.bincount(renumbered, minlength=len(unique))
    return renumbered, counts[renumbered].astype(np.uint16)


def _filter_explicit_h(sample: dict[str, Any]) -> dict[str, Any]:
    keep = sample["atomic_numbers"] != 1
    if keep.all():
        return sample
    old_to_new = np.full(len(keep), -1, dtype=np.int32)
    old_to_new[keep] = np.arange(np.count_nonzero(keep), dtype=np.int32)
    result = copy(sample)
    atom_fields = (
        "atom_types", "atomic_numbers", "formal_charges", "radical_electrons", "positions",
        "hybridization", "aromaticity", "chirality", "degree", "explicit_h_neighbor_count",
        "ring_membership", "orbit_id", "orbit_size", "target_orbit_id", "target_orbit_size",
    )
    for field in atom_fields:
        if field in sample:
            result[field] = sample[field][keep]
    edge_keep = keep[sample["bond_index"][0]] & keep[sample["bond_index"][1]]
    result["bond_index"] = old_to_new[sample["bond_index"][:, edge_keep]]
    for field in ("bond_types", "bond_stereo", "bond_conjugation", "bond_ring_membership"):
        if field in sample:
            result[field] = sample[field][edge_keep]
    for orbit_field in ("orbit_id", "target_orbit_id"):
        if orbit_field in result:
            size_field = "orbit_size" if orbit_field == "orbit_id" else "target_orbit_size"
            result[orbit_field], result[size_field] = _renumber_orbits(result[orbit_field])
    kept_old_indices = np.flatnonzero(keep)
    for symmetry_field in ("actual_symmetry", "target_symmetry"):
        if symmetry_field not in sample:
            continue
        block = copy(sample[symmetry_field])
        old_permutations = block["permutation_index"][:, kept_old_indices]
        if np.any(~keep[old_permutations]):
            raise ValueError("symmetry permutation 将重原子映射到 H，数据损坏")
        block["permutation_index"] = old_to_new[old_permutations]
        result[symmetry_field] = block
    result["explicit_h_removed"] = True
    return result


class SparseGraphAdapter:
    """EQGAT 类模型：保留 canonical sparse graph。"""

    def __init__(self, dataset: COFSymmetryDataset, include_h: bool = True) -> None:
        self.dataset = dataset
        self.include_h = include_h

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = self.dataset[item]
        return sample if self.include_h else _filter_explicit_h(sample)


class FullPairAdapter(SparseGraphAdapter):
    """MiDi/UAE 类模型：加载时构造 full-pair bond tensor，不改变磁盘 canonical 数据。"""

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = super().__getitem__(item)
        sample["full_pair_bond_types"] = full_pair_bond_types(sample)
        sample["self_loop_mask"] = np.eye(len(sample["atom_types"]), dtype=np.bool_)
        return sample


class UAE3DAdapter(FullPairAdapter):
    """对齐 UAE-3D 官方 GEOM vocabulary，显式区分 compatible/extended 两种模式。"""

    def __init__(
        self,
        dataset: COFSymmetryDataset,
        mode: str = "compatible_subset",
        include_h: bool = True,
    ) -> None:
        super().__init__(dataset, include_h=include_h)
        if mode not in {"compatible_subset", "extended_vocabulary"}:
            raise ValueError(f"未知 UAE mode: {mode}")
        self.mode = mode
        self.vocabulary = UAE_GEOM_VOCAB if mode == "compatible_subset" else UAE_EXTENDED_VOCAB
        self.symbol_to_index = {symbol: index for index, symbol in enumerate(self.vocabulary)}

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = super().__getitem__(item)
        symbols = [ATOM_SYMBOLS[int(index)] for index in sample["atom_types"]]
        missing = sorted(set(symbols) - set(self.symbol_to_index))
        if missing:
            if "Sn" in missing and self.mode == "compatible_subset":
                raise ValueError(
                    "该分子含 Sn，而 UAE-3D 官方 GEOM vocabulary 不含 Sn；"
                    "请使用 compatible subset indices 或 extended_vocabulary，禁止映射到 C/Si"
                )
            raise ValueError(f"UAE vocabulary 缺少元素: {missing}")
        sample["uae_atom_types"] = np.asarray(
            [self.symbol_to_index[symbol] for symbol in symbols], dtype=np.uint8
        )
        sample["uae_vocabulary"] = self.vocabulary
        sample["uae_mode"] = self.mode
        return sample


class MiDiAdapter(FullPairAdapter):
    """严格对齐官方 MiDi GEOM atom/charge/full-pair 表示。

    官方模型没有 radical-electron head。预训练兼容模式拒绝 Sn；扩展词表模式显式把 Sn
    添加为第 17 类，但需要从头训练对应输入/输出层，绝不映射成 C 或 Si。
    """

    def __init__(
        self,
        dataset: COFSymmetryDataset,
        mode: str = "pretrained_compatible",
        include_h: bool = True,
    ) -> None:
        super().__init__(dataset, include_h=include_h)
        if mode not in {"pretrained_compatible", "extended_vocabulary"}:
            raise ValueError(f"未知 MiDi mode: {mode}")
        self.mode = mode
        self.vocabulary = (
            MIDI_GEOM_VOCAB if mode == "pretrained_compatible" else MIDI_EXTENDED_VOCAB
        )
        self.symbol_to_index = {symbol: index for index, symbol in enumerate(self.vocabulary)}

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = super().__getitem__(item)
        symbols = [ATOM_SYMBOLS[int(index)] for index in sample["atom_types"]]
        missing = sorted(set(symbols) - set(self.symbol_to_index))
        if missing:
            if "Sn" in missing and self.mode == "pretrained_compatible":
                raise ValueError(
                    "MiDi 官方 GEOM vocabulary 不含 Sn；请使用 compatible subset 或 "
                    "extended_vocabulary，禁止映射到 C/Si"
                )
            raise ValueError(f"MiDi vocabulary 缺少元素: {missing}")
        radical_count = int(np.count_nonzero(sample["radical_electrons"]))
        if radical_count:
            raise ValueError(
                f"MiDi 官方模型没有 radical-electron head；该分子有 {radical_count} 个"
                "非零自由基原子，禁止静默置零"
            )
        charges = [int(charge) for charge in sample["formal_charges"]]
        unsupported_charges = sorted(set(charges) - set(MIDI_CHARGE_TO_INDEX))
        if unsupported_charges:
            raise ValueError(f"MiDi charge vocabulary 不支持: {unsupported_charges}")
        sample["midi_atom_types"] = np.asarray(
            [self.symbol_to_index[symbol] for symbol in symbols], dtype=np.uint8
        )
        sample["midi_charge_types"] = np.asarray(
            [MIDI_CHARGE_TO_INDEX[charge] for charge in charges], dtype=np.uint8
        )
        sample["midi_positions"] = sample["positions"].astype(np.float32, copy=False)
        sample["midi_atom_mask"] = np.ones(len(symbols), dtype=np.bool_)
        sample["midi_vocabulary"] = self.vocabulary
        sample["midi_mode"] = self.mode
        return sample


class SemlaFlowAdapter(FullPairAdapter):
    """对齐 SemlaFlow 官方 GEOM-Drugs token、charge 与坐标约定。

    官方模型没有 radical-electron 输出头。``pretrained_compatible`` 模式因此对 Sn 和
    非零自由基严格失败；后续若扩展模型 head，应使用新 mode 和新 checkpoint schema。
    """

    def __init__(
        self,
        dataset: COFSymmetryDataset,
        mode: str = "pretrained_compatible",
        include_h: bool = True,
    ) -> None:
        if mode != "pretrained_compatible":
            raise ValueError(f"尚未实现的 SemlaFlow mode: {mode}")
        if not include_h:
            raise ValueError("SemlaFlow GEOM pretrained-compatible 路径要求保留显式 H")
        super().__init__(dataset, include_h=True)
        self.mode = mode
        self.vocabulary = SEMLAFLOW_VOCAB
        self.symbol_to_index = {symbol: index for index, symbol in enumerate(self.vocabulary)}

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = super().__getitem__(item)
        symbols = [ATOM_SYMBOLS[int(index)] for index in sample["atom_types"]]
        missing = sorted(set(symbols) - set(self.symbol_to_index))
        if missing:
            if "Sn" in missing:
                raise ValueError(
                    "SemlaFlow 官方 GEOM vocabulary 不含 Sn；禁止映射到 C/Si，"
                    "请使用严格 compatible indices 或实现扩展 vocabulary/head"
                )
            raise ValueError(f"SemlaFlow vocabulary 缺少元素: {missing}")
        radical_count = int(np.count_nonzero(sample["radical_electrons"]))
        if radical_count:
            raise ValueError(
                f"SemlaFlow 官方模型没有 radical-electron head；该分子有 {radical_count} 个"
                "非零自由基原子，禁止静默置零"
            )
        charges = [int(charge) for charge in sample["formal_charges"]]
        unsupported_charges = sorted(set(charges) - set(SEMLAFLOW_CHARGE_TO_INDEX))
        if unsupported_charges:
            raise ValueError(f"SemlaFlow charge vocabulary 不支持: {unsupported_charges}")

        sample["semlaflow_atom_types"] = np.asarray(
            [self.symbol_to_index[symbol] for symbol in symbols], dtype=np.uint8
        )
        sample["semlaflow_charge_types"] = np.asarray(
            [SEMLAFLOW_CHARGE_TO_INDEX[charge] for charge in charges], dtype=np.uint8
        )
        sample["semlaflow_positions"] = (
            sample["positions"].astype(np.float32, copy=False) / SEMLAFLOW_GEOM_COORD_STD
        )
        sample["semlaflow_atom_mask"] = np.ones(len(symbols), dtype=np.bool_)
        sample["semlaflow_vocabulary"] = self.vocabulary
        sample["semlaflow_coordinate_std"] = SEMLAFLOW_GEOM_COORD_STD
        sample["semlaflow_mode"] = self.mode
        return sample
