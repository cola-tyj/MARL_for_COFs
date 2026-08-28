"""canonical v2 与锁定官方 MiDi PyG ``Data`` 的严格桥接。"""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np

from generative_model.data.model_adapters import MIDI_CHARGE_TO_INDEX, MIDI_GEOM_VOCAB
from generative_model.data.schema import ATOM_SYMBOLS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "generative_model/external/midi"
OFFICIAL_COMMIT = "775b731c38967a1e49615b2ad70ac6b5db24909a"


def assert_official_source(source_root: Path = DEFAULT_SOURCE) -> str:
    source_root = source_root.resolve()
    if not (source_root / ".git").is_dir():
        raise FileNotFoundError(
            f"缺少 MiDi checkout: {source_root}；请先运行 bootstrap_sources --model midi"
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if head != OFFICIAL_COMMIT:
        raise RuntimeError(f"MiDi HEAD={head}，source lock 要求 {OFFICIAL_COMMIT}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source_root, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("MiDi official checkout 有本地修改，拒绝继续")
    return head


def _validate_sample(sample: dict) -> None:
    required = (
        "positions", "atom_types", "atomic_numbers", "formal_charges",
        "radical_electrons", "bond_index", "bond_types",
    )
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"canonical sample 缺少字段: {missing}")
    atom_count = len(sample["atom_types"])
    if np.asarray(sample["positions"]).shape != (atom_count, 3):
        raise ValueError("positions 必须为 [N,3]")
    if not np.isfinite(sample["positions"]).all():
        raise ValueError("positions 含 NaN/Inf")
    bond_index = np.asarray(sample["bond_index"])
    if bond_index.ndim != 2 or bond_index.shape[0] != 2:
        raise ValueError("bond_index 必须为 [2,E]")
    if bond_index.shape[1] != len(sample["bond_types"]):
        raise ValueError("bond_index 与 bond_types 长度不一致")
    if bond_index.size and (bond_index.min() < 0 or bond_index.max() >= atom_count):
        raise ValueError("bond_index 越界")
    if np.any(bond_index[0] >= bond_index[1]):
        raise ValueError("canonical bond_index 必须每条无向键只存一次且 begin < end")
    bond_types = np.asarray(sample["bond_types"])
    if np.any((bond_types < 1) | (bond_types > 4)):
        raise ValueError("bond_types 只允许 1/2/3/4")
    symbols = [ATOM_SYMBOLS[int(index)] for index in sample["atom_types"]]
    unsupported = sorted(set(symbols) - set(MIDI_GEOM_VOCAB))
    if unsupported:
        if "Sn" in unsupported:
            raise ValueError("MiDi 官方 GEOM vocabulary 不含 Sn，禁止映射到 C/Si")
        raise ValueError(f"MiDi vocabulary 缺少元素: {unsupported}")
    if np.any(np.asarray(sample["radical_electrons"]) != 0):
        raise ValueError("MiDi 官方模型没有 radical-electron head")
    charges = set(map(int, sample["formal_charges"]))
    unsupported_charges = sorted(charges - set(MIDI_CHARGE_TO_INDEX))
    if unsupported_charges:
        raise ValueError(f"MiDi charge vocabulary 不支持: {unsupported_charges}")


def canonical_to_midi_data(sample: dict, source_root: Path = DEFAULT_SOURCE):
    """转换成官方训练代码接受的 PyG ``Data``，不改变 canonical ground truth。

    MiDi 的 sparse PyG 输入把每条无向键保存为两个方向；``edge_attr`` 仍使用
    1/2/3/4，0=none 只在官方 ``to_dense`` 阶段动态补齐。
    """

    assert_official_source(source_root)
    _validate_sample(sample)
    import torch
    from torch_geometric.data import Data

    symbol_to_index = {symbol: index for index, symbol in enumerate(MIDI_GEOM_VOCAB)}
    symbols = [ATOM_SYMBOLS[int(index)] for index in sample["atom_types"]]
    atom_types = np.asarray([symbol_to_index[symbol] for symbol in symbols], dtype=np.int64)
    undirected = np.asarray(sample["bond_index"], dtype=np.int64)
    directed = np.concatenate((undirected, undirected[::-1]), axis=1)
    bond_types = np.asarray(sample["bond_types"], dtype=np.int64)
    directed_types = np.concatenate((bond_types, bond_types))
    positions = np.asarray(sample["positions"], dtype=np.float32)
    # canonical 已归零；只允许浮点累计误差，不在桥接中静默重新平移。
    if np.max(np.abs(positions.mean(axis=0))) >= 1e-3:
        raise ValueError("MiDi 输入坐标质心未归零")
    metadata = sample.get("metadata", {})
    return Data(
        x=torch.as_tensor(atom_types, dtype=torch.long),
        edge_index=torch.as_tensor(directed.copy(), dtype=torch.long),
        edge_attr=torch.as_tensor(directed_types, dtype=torch.long),
        pos=torch.as_tensor(positions, dtype=torch.float32),
        charges=torch.as_tensor(np.asarray(sample["formal_charges"]), dtype=torch.long),
        smiles=str(metadata.get("SMILES", metadata.get("smiles", ""))),
        molecule_id=str(sample.get("molecule_id", metadata.get("Molecule_ID", ""))),
        package_index=int(sample.get("package_index", -1)),
    )


def midi_data_to_canonical(data) -> dict[str, np.ndarray]:
    """从 PyG 输入恢复 canonical 字段，并严格验证双向边完全一致。"""

    atom_indices = data.x.detach().cpu().numpy().astype(np.int64, copy=False)
    if np.any((atom_indices < 0) | (atom_indices >= len(MIDI_GEOM_VOCAB))):
        raise ValueError("MiDi atom index 越界")
    symbol_to_canonical = {symbol: index for index, symbol in enumerate(ATOM_SYMBOLS)}
    output_symbols = [MIDI_GEOM_VOCAB[int(index)] for index in atom_indices]
    unsupported = sorted(set(output_symbols) - set(symbol_to_canonical))
    if unsupported:
        raise ValueError(f"canonical 13-element vocabulary 无法表示 MiDi 输出: {unsupported}")
    canonical_atom_types = np.asarray(
        [symbol_to_canonical[symbol] for symbol in output_symbols],
        dtype=np.uint8,
    )
    atomic_numbers_by_symbol = {
        "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Al": 13,
        "Si": 14, "P": 15, "S": 16, "Cl": 17, "As": 33, "Br": 35,
        "I": 53, "Hg": 80, "Bi": 83,
    }
    atomic_numbers = np.asarray(
        [atomic_numbers_by_symbol[MIDI_GEOM_VOCAB[int(index)]] for index in atom_indices],
        dtype=np.uint8,
    )
    edge_index = data.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    edge_attr = data.edge_attr.detach().cpu().numpy().astype(np.int64, copy=False)
    directed: dict[tuple[int, int], int] = {}
    for (begin, end), bond_type in zip(edge_index.T, edge_attr):
        key = (int(begin), int(end))
        if begin == end or key in directed:
            raise ValueError("MiDi edge_index 含 self-loop 或重复有向边")
        directed[key] = int(bond_type)
    bonds: list[tuple[int, int, int]] = []
    for (begin, end), bond_type in sorted(directed.items()):
        reverse = directed.get((end, begin))
        if reverse != bond_type:
            raise ValueError("MiDi 双向边缺失或 bond type 不一致")
        if begin < end:
            bonds.append((begin, end, bond_type))
    compact_index = (
        np.asarray([(begin, end) for begin, end, _ in bonds], dtype=np.int32).T
        if bonds else np.empty((2, 0), dtype=np.int32)
    )
    return {
        "positions": data.pos.detach().cpu().numpy().astype(np.float32, copy=False),
        "atom_types": canonical_atom_types,
        "atomic_numbers": atomic_numbers,
        "formal_charges": data.charges.detach().cpu().numpy().astype(np.int8, copy=False),
        "bond_index": compact_index,
        "bond_types": np.asarray([kind for _, _, kind in bonds], dtype=np.uint8),
    }
