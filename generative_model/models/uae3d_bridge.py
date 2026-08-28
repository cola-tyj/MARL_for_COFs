"""Canonical v2 与锁定 UAE-3D 官方 ``UnifiedAutoEncoder`` 的严格桥接。"""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np

from generative_model.data.model_adapters import UAE_GEOM_VOCAB
from generative_model.data.schema import ATOM_SYMBOLS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "generative_model/external/UAE-3D"
OFFICIAL_COMMIT = "4327dd5d8233c74e0f045be481f24d0f1d4c0c09"
OFFICIAL_GEOM_POSITION_STD = 2.3859
OFFICIAL_NODE_DIM = 55
OFFICIAL_ENCODER_NODE_DIM = OFFICIAL_NODE_DIM + 3
OFFICIAL_EDGE_DIM = 4


def uae3d_model_args(profile: str = "official") -> SimpleNamespace:
    """返回锁定 UAE-3D autoencoder 架构及官方重构损失配置。"""

    if profile not in {"mini", "official"}:
        raise ValueError(f"未知 UAE-3D profile: {profile}")
    return SimpleNamespace(
        node_dim=OFFICIAL_ENCODER_NODE_DIM,
        edge_dim=OFFICIAL_EDGE_DIM,
        n_atom_types=16,
        n_bond_types=4,
        encoder_hidden_dim=64,
        encoder_n_heads=8,
        encoder_blocks=2 if profile == "mini" else 6,
        latent_dim=16,
        decoder_hidden_dim=64,
        decoder_n_heads=8,
        decoder_blocks=2 if profile == "mini" else 4,
        dropout=0.1,
        atom_loss_weight=1.0,
        bond_loss_weight=1.0,
        center_prediction=False,
        align_prediction=False,
        coordinate_loss_weight=1.0,
        dist_loss_weight=1.0,
        bond_dist_loss_weight=10.0,
        kld_weight=1e-8,
    )


def assert_uae3d_official_source(source_root: Path = DEFAULT_SOURCE) -> str:
    """拒绝错误 commit 或带本地修改的上游源码。"""

    source_root = source_root.resolve()
    if not (source_root / ".git").is_dir():
        raise FileNotFoundError(
            f"缺少 UAE-3D checkout: {source_root}；"
            "请先运行 bootstrap_sources --model uae3d"
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if head != OFFICIAL_COMMIT:
        raise RuntimeError(f"UAE-3D HEAD={head}，source lock 要求 {OFFICIAL_COMMIT}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source_root, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("UAE-3D official checkout 有本地修改，拒绝继续")
    return head


def is_uae3d_three_head_compatible(sample: dict[str, Any]) -> bool:
    """是否能由官方 atom/bond/coordinate 三个 head 无损表示。"""

    symbols = {ATOM_SYMBOLS[int(index)] for index in sample["atom_types"]}
    return bool(
        symbols.issubset(UAE_GEOM_VOCAB)
        and np.all(np.asarray(sample["formal_charges"]) == 0)
        and np.all(np.asarray(sample["radical_electrons"]) == 0)
    )


def _validate_canonical_sample(sample: dict[str, Any]) -> None:
    required = (
        "positions", "atom_types", "atomic_numbers", "formal_charges",
        "radical_electrons", "bond_index", "bond_types", "metadata",
    )
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"canonical sample 缺少字段: {missing}")
    atom_count = len(sample["atom_types"])
    positions = np.asarray(sample["positions"])
    if positions.shape != (atom_count, 3) or not np.isfinite(positions).all():
        raise ValueError("positions 必须为有限 [N,3] 数组")
    if np.max(np.abs(positions.mean(axis=0))) >= 1e-3:
        raise ValueError("UAE-3D 输入坐标质心未归零")
    bond_index = np.asarray(sample["bond_index"])
    bond_types = np.asarray(sample["bond_types"])
    if bond_index.ndim != 2 or bond_index.shape[0] != 2:
        raise ValueError("bond_index 必须为 [2,E]")
    if bond_index.shape[1] != len(bond_types):
        raise ValueError("bond_index 与 bond_types 长度不一致")
    if bond_index.size and (bond_index.min() < 0 or bond_index.max() >= atom_count):
        raise ValueError("bond_index 越界")
    if np.any(bond_index[0] >= bond_index[1]):
        raise ValueError("canonical bond_index 必须每条无向键只存一次且 begin < end")
    if np.any((bond_types < 1) | (bond_types > 4)):
        raise ValueError("bond_types 只允许 1/2/3/4")

    symbols = [ATOM_SYMBOLS[int(index)] for index in sample["atom_types"]]
    unsupported = sorted(set(symbols) - set(UAE_GEOM_VOCAB))
    if unsupported:
        if "Sn" in unsupported:
            raise ValueError("UAE-3D 官方 GEOM vocabulary 不含 Sn，禁止映射到 C/Si")
        raise ValueError(f"UAE-3D vocabulary 缺少元素: {unsupported}")
    nonzero_charge = int(np.count_nonzero(sample["formal_charges"]))
    if nonzero_charge:
        raise ValueError(
            f"UAE-3D 官方 decoder 没有 formal-charge head；该分子有 "
            f"{nonzero_charge} 个非零形式电荷原子"
        )
    nonzero_radical = int(np.count_nonzero(sample["radical_electrons"]))
    if nonzero_radical:
        raise ValueError(
            f"UAE-3D 官方 decoder 没有 radical-electron head；该分子有 "
            f"{nonzero_radical} 个非零自由基原子"
        )


def _one_k_encoding(value: Any, choices: list[Any]) -> list[int]:
    encoding = [0] * (len(choices) + 1)
    encoding[choices.index(value) if value in choices else -1] = 1
    return encoding


def _rdkit_molecule_and_features(sample: dict[str, Any]):
    """按锁定源码 ``featurize_mol(..., geom_with_h_1)`` 的公式生成 55 维特征。"""

    from rdkit import Chem

    smiles = str(sample["metadata"].get("SMILES", ""))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit 无法解析 canonical SMILES: {smiles}")
    molecule = Chem.AddHs(molecule)
    atom_count = len(sample["atom_types"])
    if molecule.GetNumAtoms() != atom_count:
        raise ValueError(
            f"SMILES AddHs 原子数 {molecule.GetNumAtoms()} != canonical {atom_count}"
        )
    rdkit_atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.uint8
    )
    if not np.array_equal(rdkit_atomic_numbers, np.asarray(sample["atomic_numbers"])):
        raise ValueError("SMILES/RDKit 与 canonical XYZ 原子索引不一致")

    bond_type_to_index = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    rdkit_bonds: dict[tuple[int, int], int] = {}
    for bond in molecule.GetBonds():
        begin, end = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if bond.GetBondType() not in bond_type_to_index:
            raise ValueError(f"UAE-3D 不支持 RDKit bond type: {bond.GetBondType()}")
        rdkit_bonds[(begin, end)] = bond_type_to_index[bond.GetBondType()]
    canonical_bonds = {
        (int(begin), int(end)): int(kind)
        for (begin, end), kind in zip(
            np.asarray(sample["bond_index"]).T, np.asarray(sample["bond_types"])
        )
    }
    if rdkit_bonds != canonical_bonds:
        raise ValueError("SMILES/RDKit bond graph 与 canonical bond graph 不一致")

    atom_encoder = {symbol: index for index, symbol in enumerate(UAE_GEOM_VOCAB)}
    ring = molecule.GetRingInfo()
    rows: list[list[float]] = []
    for index, atom in enumerate(molecule.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol not in atom_encoder:
            raise ValueError(f"UAE-3D vocabulary 缺少元素: {symbol}")
        atom_one_hot = [0] * len(atom_encoder)
        atom_one_hot[atom_encoder[symbol]] = 1
        derived: list[float] = [float(atom.GetAtomicNum()), float(atom.GetIsAromatic())]
        derived.extend(_one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        derived.extend(_one_k_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]))
        derived.extend(_one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
        derived.extend(_one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
        derived.extend(float(ring.IsAtomInRingOfSize(index, size)) for size in range(3, 9))
        derived.extend(_one_k_encoding(int(ring.NumAtomRings(index)), [0, 1, 2, 3]))
        row = [*map(float, atom_one_hot), *derived]
        if len(row) != OFFICIAL_NODE_DIM:
            raise RuntimeError(f"UAE-3D node feature 维度 {len(row)} != {OFFICIAL_NODE_DIM}")
        rows.append(row)
    return molecule, np.asarray(rows, dtype=np.float32)


def canonical_to_uae3d_data(
    sample: dict[str, Any],
    source_root: Path = DEFAULT_SOURCE,
    position_std: float = OFFICIAL_GEOM_POSITION_STD,
    *,
    verify_source: bool = True,
):
    """构造官方 UAE-3D encoder 接受的 full-pair PyG ``Data``。"""

    if verify_source:
        assert_uae3d_official_source(source_root)
    _validate_canonical_sample(sample)
    if not np.isfinite(position_std) or position_std <= 0:
        raise ValueError("position_std 必须为有限正数")

    import torch
    from torch_geometric.data import Data

    molecule, node_features = _rdkit_molecule_and_features(sample)
    atom_count = molecule.GetNumAtoms()
    begin = torch.arange(atom_count, dtype=torch.long).repeat_interleave(atom_count)
    end = torch.arange(atom_count, dtype=torch.long).repeat(atom_count)
    full_edge_index = torch.stack((begin, end), dim=0)
    full_edge_attr = torch.zeros((atom_count * atom_count, 5), dtype=torch.float32)
    for (left, right), canonical_kind in zip(
        np.asarray(sample["bond_index"]).T, np.asarray(sample["bond_types"])
    ):
        official_kind = int(canonical_kind) - 1
        full_edge_attr[int(left) * atom_count + int(right), official_kind] = 1.0
        full_edge_attr[int(right) * atom_count + int(left), official_kind] = 1.0
    diagonal = torch.arange(atom_count, dtype=torch.long)
    full_edge_attr[diagonal * atom_count + diagonal, 4] = 1.0
    positions = np.asarray(sample["positions"], dtype=np.float32) / np.float32(position_std)
    metadata = sample["metadata"]
    return Data(
        x=torch.as_tensor(node_features, dtype=torch.float32),
        z=torch.as_tensor(np.asarray(sample["atomic_numbers"]), dtype=torch.long),
        edge_index=full_edge_index,
        edge_attr=full_edge_attr,
        pos=torch.as_tensor(positions, dtype=torch.float32),
        idx=torch.tensor(int(sample.get("package_index", -1)), dtype=torch.long),
        molecule_id=str(sample.get("molecule_id", metadata.get("Molecule_ID", ""))),
        smiles=str(metadata.get("SMILES", "")),
        position_std=float(position_std),
    )


def uae3d_data_to_canonical(
    data,
    position_std: float = OFFICIAL_GEOM_POSITION_STD,
) -> dict[str, np.ndarray]:
    """从官方输入 tensor 恢复三个 head 的 canonical ground truth。"""

    import torch

    atom_count = int(data.x.shape[0])
    if data.x.shape[1] != OFFICIAL_NODE_DIM:
        raise ValueError(f"UAE-3D node dim {data.x.shape[1]} != {OFFICIAL_NODE_DIM}")
    atom_one_hot = data.x[:, :len(UAE_GEOM_VOCAB)]
    if not torch.all((atom_one_hot == 0) | (atom_one_hot == 1)):
        raise ValueError("UAE-3D atom one-hot 含非 0/1 值")
    if not torch.all(atom_one_hot.sum(dim=1) == 1):
        raise ValueError("UAE-3D atom one-hot 每行必须恰有一个类别")
    official_indices = atom_one_hot.argmax(dim=1).cpu().numpy()
    canonical_by_symbol = {symbol: index for index, symbol in enumerate(ATOM_SYMBOLS)}
    symbols = [UAE_GEOM_VOCAB[int(index)] for index in official_indices]
    unsupported = sorted(set(symbols) - set(canonical_by_symbol))
    if unsupported:
        raise ValueError(f"canonical 13-element vocabulary 无法表示 UAE 输出: {unsupported}")

    expected_begin = torch.arange(atom_count).repeat_interleave(atom_count)
    expected_end = torch.arange(atom_count).repeat(atom_count)
    expected_edge_index = torch.stack((expected_begin, expected_end), dim=0)
    if not torch.equal(data.edge_index.cpu(), expected_edge_index):
        raise ValueError("UAE-3D full edge index 不是确定性 row-major N² 顺序")
    edge_attr = data.edge_attr.detach().cpu().numpy()
    if edge_attr.shape != (atom_count * atom_count, 5):
        raise ValueError("UAE-3D edge_attr 必须为 [N²,5]")
    dense = edge_attr.reshape(atom_count, atom_count, 5)
    expected_self = np.eye(atom_count, dtype=np.float32)
    if not np.array_equal(dense[:, :, 4], expected_self):
        raise ValueError("UAE-3D self-loop channel 错误")
    bond_channels = dense[:, :, :4]
    if np.any((bond_channels != 0) & (bond_channels != 1)):
        raise ValueError("UAE-3D bond one-hot 含非 0/1 值")
    if np.any(bond_channels.sum(axis=2) > 1):
        raise ValueError("UAE-3D bond pair 含多个键类别")
    if not np.array_equal(bond_channels, bond_channels.transpose(1, 0, 2)):
        raise ValueError("UAE-3D full-pair bond tensor 非对称")
    if np.any(bond_channels[np.arange(atom_count), np.arange(atom_count)] != 0):
        raise ValueError("UAE-3D 对角 pair 不得同时含化学键")

    bonds: list[tuple[int, int, int]] = []
    for left in range(atom_count):
        for right in range(left + 1, atom_count):
            present = np.flatnonzero(bond_channels[left, right])
            if len(present):
                bonds.append((left, right, int(present[0]) + 1))
    bond_index = (
        np.asarray([(left, right) for left, right, _ in bonds], dtype=np.int32).T
        if bonds else np.empty((2, 0), dtype=np.int32)
    )
    return {
        "positions": (
            data.pos.detach().cpu().numpy().astype(np.float32, copy=False)
            * np.float32(position_std)
        ),
        "atom_types": np.asarray(
            [canonical_by_symbol[symbol] for symbol in symbols], dtype=np.uint8
        ),
        "atomic_numbers": data.z.detach().cpu().numpy().astype(np.uint8, copy=False),
        "formal_charges": np.zeros(atom_count, dtype=np.int8),
        "radical_electrons": np.zeros(atom_count, dtype=np.uint8),
        "bond_index": bond_index,
        "bond_types": np.asarray([kind for _, _, kind in bonds], dtype=np.uint8),
    }
