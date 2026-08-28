"""v2 canonical sample 与锁定 SemlaFlow ``GeometricMol`` 的严格桥接。"""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from generative_model.data.model_adapters import (
    SEMLAFLOW_CHARGE_TO_INDEX,
    SEMLAFLOW_VOCAB,
)
from generative_model.data.schema import ATOM_SYMBOLS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "generative_model/external/semla-flow"
OFFICIAL_COMMIT = "3f43103d3af138b86dbe9f29fe8085e83f9a6283"


def assert_official_source(source_root: Path = DEFAULT_SOURCE) -> str:
    """拒绝错误、缺失或被本地修改的上游 checkout。"""
    source_root = source_root.resolve()
    if not (source_root / ".git").is_dir():
        raise FileNotFoundError(
            f"缺少 SemlaFlow checkout: {source_root}；请先运行 bootstrap_sources"
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if head != OFFICIAL_COMMIT:
        raise RuntimeError(f"SemlaFlow HEAD={head}，source lock 要求 {OFFICIAL_COMMIT}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=source_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("SemlaFlow official checkout 有本地修改，拒绝继续")
    return head


def _official_geometric_mol_class(source_root: Path = DEFAULT_SOURCE):
    assert_official_source(source_root)
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("semlaflow.util.molrepr")
    return module.GeometricMol


def _validate_canonical_sample(sample: dict[str, Any]) -> None:
    required = (
        "positions",
        "atomic_numbers",
        "atom_types",
        "bond_index",
        "bond_types",
        "formal_charges",
        "radical_electrons",
    )
    missing_fields = [field for field in required if field not in sample]
    if missing_fields:
        raise KeyError(f"canonical sample 缺少字段: {missing_fields}")
    atom_count = len(sample["atomic_numbers"])
    if sample["positions"].shape != (atom_count, 3):
        raise ValueError("positions 必须为 [N,3]")
    if sample["bond_index"].ndim != 2 or sample["bond_index"].shape[0] != 2:
        raise ValueError("bond_index 必须为 [2,E]")
    if len(sample["bond_types"]) != sample["bond_index"].shape[1]:
        raise ValueError("bond_types 与 bond_index 长度不一致")
    if not np.isfinite(sample["positions"]).all():
        raise ValueError("positions 含 NaN/Inf")

    symbols = [ATOM_SYMBOLS[int(index)] for index in sample["atom_types"]]
    unsupported = sorted(set(symbols) - set(SEMLAFLOW_VOCAB))
    if unsupported:
        raise ValueError(f"SemlaFlow 官方 vocabulary 缺少元素: {unsupported}")
    charges = set(map(int, sample["formal_charges"]))
    unsupported_charges = sorted(charges - set(SEMLAFLOW_CHARGE_TO_INDEX))
    if unsupported_charges:
        raise ValueError(f"SemlaFlow charge vocabulary 不支持: {unsupported_charges}")
    if np.any(np.asarray(sample["radical_electrons"]) != 0):
        raise ValueError("SemlaFlow 官方模型没有 radical-electron head")


def canonical_to_geometric_mol(
    sample: dict[str, Any],
    source_root: Path = DEFAULT_SOURCE,
    *,
    str_id: str | None = None,
):
    """构造官方 raw ``GeometricMol``；此处不缩放、不旋转、不把电荷改成类别索引。

    ``str_id`` 在 SemlaFlow 的训练/评测代码里按 SMILES 解释。普通接口为了兼容旧调用
    仍默认使用 ``Molecule_ID``；导出训练集时必须显式传入 canonical SMILES。
    """
    _validate_canonical_sample(sample)
    geometric_mol = _official_geometric_mol_class(source_root)
    import torch

    if str_id is None:
        str_id = str(sample.get("metadata", {}).get("Molecule_ID", "")) or None
    return geometric_mol(
        torch.as_tensor(np.asarray(sample["positions"]), dtype=torch.float32),
        torch.as_tensor(np.asarray(sample["atomic_numbers"]), dtype=torch.long),
        bond_indices=torch.as_tensor(np.asarray(sample["bond_index"]).T.copy(), dtype=torch.long),
        bond_types=torch.as_tensor(np.asarray(sample["bond_types"]), dtype=torch.long),
        charges=torch.as_tensor(np.asarray(sample["formal_charges"]), dtype=torch.long),
        str_id=str_id,
    )


def geometric_mol_to_canonical(molecule: Any) -> dict[str, np.ndarray]:
    """从官方对象取回 canonical ground-truth 字段，不猜测或重建化学键。"""
    return {
        "positions": molecule.coords.detach().cpu().numpy().astype(np.float32, copy=False),
        "atomic_numbers": molecule.atomics.detach().cpu().numpy().astype(np.uint8, copy=False),
        "bond_index": molecule.bond_indices.detach().cpu().numpy().T.astype(np.int32, copy=False),
        "bond_types": molecule.bond_types.detach().cpu().numpy().astype(np.uint8, copy=False),
        "formal_charges": molecule.charges.detach().cpu().numpy().astype(np.int8, copy=False),
    }


def roundtrip_official_bytes(
    sample: dict[str, Any], source_root: Path = DEFAULT_SOURCE
) -> dict[str, np.ndarray]:
    """通过官方 protocol-4 bytes 序列化往返，用于接口 smoke test。"""
    molecule = canonical_to_geometric_mol(sample, source_root)
    geometric_mol = _official_geometric_mol_class(source_root)
    restored = geometric_mol.from_bytes(molecule.to_bytes())
    return geometric_mol_to_canonical(restored)
