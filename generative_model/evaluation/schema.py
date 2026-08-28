"""模型无关的生成分子评测输入与报告 schema。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EVALUATION_SCHEMA_VERSION = "1.1"
REPORT_SCHEMA_PATH = Path(__file__).with_name("evaluation_report.schema.json")


class FailureReason(str, Enum):
    """可以出现在报告中的、含义稳定的输入失败原因。"""

    MISSING_FIELD = "missing_field"
    EMPTY_MOLECULE = "empty_molecule"
    INVALID_SHAPE = "invalid_shape"
    INVALID_DTYPE = "invalid_dtype"
    LENGTH_MISMATCH = "length_mismatch"
    INVALID_ATOMIC_NUMBER = "invalid_atomic_number"
    INVALID_BOND_INDEX = "invalid_bond_index"
    INVALID_BOND_TYPE = "invalid_bond_type"
    SELF_BOND = "self_bond"
    NON_CANONICAL_BOND = "non_canonical_bond"
    DUPLICATE_BOND = "duplicate_bond"


class EvaluationInputError(ValueError):
    """输入违反评测契约；只捕获此类预期错误，程序缺陷应直接抛出。"""

    def __init__(self, reason: FailureReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise EvaluationInputError(FailureReason.MISSING_FIELD, f"missing required field: {key}")
    return mapping[key]


def _integer_array(value: Any, key: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise EvaluationInputError(FailureReason.INVALID_DTYPE, f"{key} must have an integer dtype")
    return array.astype(np.int64, copy=False)


@dataclass(frozen=True)
class EvaluationSample:
    """一个待评测分子的统一稀疏无向图表示。

    ``bond_index`` 必须是 ``[2, M]``，每条无向键只出现一次且满足 ``u < v``。
    ``bond_types`` 只接受 1--4；0=none 只属于模型内部 full-pair 表示。
    非有限坐标属于应被度量的生成失败，不属于结构 schema 错误。
    """

    molecule_id: str
    atomic_numbers: np.ndarray
    formal_charges: np.ndarray
    positions: np.ndarray
    bond_index: np.ndarray
    bond_types: np.ndarray
    package_index: int | None = None
    target_pg: str | None = None
    actual_pg: str | None = None
    pg_compatible: bool | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationSample":
        molecule_id = str(_require(value, "molecule_id"))
        atomic_numbers = _integer_array(_require(value, "atomic_numbers"), "atomic_numbers")
        formal_charges = _integer_array(_require(value, "formal_charges"), "formal_charges")
        positions = np.asarray(_require(value, "positions"))
        bond_index = _integer_array(_require(value, "bond_index"), "bond_index")
        bond_types = _integer_array(_require(value, "bond_types"), "bond_types")

        if atomic_numbers.ndim != 1:
            raise EvaluationInputError(FailureReason.INVALID_SHAPE, "atomic_numbers must have shape [N]")
        atom_count = len(atomic_numbers)
        if atom_count == 0:
            raise EvaluationInputError(FailureReason.EMPTY_MOLECULE, "molecule contains no atoms")
        if formal_charges.shape != (atom_count,):
            raise EvaluationInputError(FailureReason.LENGTH_MISMATCH, "formal_charges must have shape [N]")
        if positions.shape != (atom_count, 3):
            raise EvaluationInputError(FailureReason.INVALID_SHAPE, "positions must have shape [N, 3]")
        if positions.dtype.kind not in "fiu":
            raise EvaluationInputError(FailureReason.INVALID_DTYPE, "positions must have a numeric dtype")
        if np.any(atomic_numbers <= 0):
            raise EvaluationInputError(FailureReason.INVALID_ATOMIC_NUMBER, "atomic_numbers must be positive")
        if bond_index.ndim != 2 or bond_index.shape[0] != 2:
            raise EvaluationInputError(FailureReason.INVALID_SHAPE, "bond_index must have shape [2, M]")
        if bond_types.shape != (bond_index.shape[1],):
            raise EvaluationInputError(FailureReason.LENGTH_MISMATCH, "bond_types length must equal M")
        if np.any((bond_types < 1) | (bond_types > 4)):
            raise EvaluationInputError(FailureReason.INVALID_BOND_TYPE, "sparse bond_types must be in [1, 4]")
        if bond_index.size and (np.any(bond_index < 0) or np.any(bond_index >= atom_count)):
            raise EvaluationInputError(FailureReason.INVALID_BOND_INDEX, "bond_index is outside [0, N)")
        if bond_index.size and np.any(bond_index[0] == bond_index[1]):
            raise EvaluationInputError(FailureReason.SELF_BOND, "self bonds are not allowed")
        if bond_index.size and np.any(bond_index[0] > bond_index[1]):
            raise EvaluationInputError(
                FailureReason.NON_CANONICAL_BOND,
                "undirected sparse bonds must be stored once with u < v",
            )
        if bond_index.shape[1]:
            pairs = bond_index.T
            if len(np.unique(pairs, axis=0)) != len(pairs):
                raise EvaluationInputError(FailureReason.DUPLICATE_BOND, "duplicate bonds are not allowed")

        package_index = value.get("package_index")
        return cls(
            molecule_id=molecule_id,
            atomic_numbers=atomic_numbers,
            formal_charges=formal_charges,
            positions=positions.astype(np.float64, copy=False),
            bond_index=bond_index,
            bond_types=bond_types,
            package_index=None if package_index is None else int(package_index),
            target_pg=None if value.get("target_pg") is None else str(value["target_pg"]),
            actual_pg=None if value.get("actual_pg") is None else str(value["actual_pg"]),
            pg_compatible=None if value.get("pg_compatible") is None else bool(value["pg_compatible"]),
        )
