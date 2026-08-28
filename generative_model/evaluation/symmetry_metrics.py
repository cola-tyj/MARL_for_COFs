"""在生成坐标上重跑与 v2 相同的严格点群协议。"""

from __future__ import annotations

from typing import Any

import numpy as np
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from rdkit import Chem

from generative_model.data.symmetry import (
    EXPECTED_GROUP_ORDER,
    SymmetryProtocol,
    _assign_operation,
    _canonical_generators,
    _classify_group,
    _largest_valid_subgroup,
    _orbits,
    _safe_closure,
    analyze_symmetry,
)

SUPPORTED_TARGET_POINT_GROUPS = frozenset({"C2", "C3", "S4", "D6h"})


def atomic_symbols(atomic_numbers: np.ndarray) -> list[str]:
    table = Chem.GetPeriodicTable()
    symbols = [str(table.GetElementSymbol(int(number))) for number in atomic_numbers]
    if any(not symbol for symbol in symbols):
        raise ValueError("无法将 atomic number 转换为元素符号")
    return symbols


def _actual_only(
    symbols: list[str],
    positions: np.ndarray,
    protocol: SymmetryProtocol,
) -> dict[str, Any]:
    """复用 v2 内部有限群构造，但不要求无条件样本具有 target PG。"""

    analyzer = PointGroupAnalyzer(
        Molecule(symbols, positions),
        tolerance=protocol.tolerance_angstrom,
        eigen_tolerance=protocol.eigen_tolerance,
        matrix_tolerance=protocol.matrix_tolerance,
    )
    candidate_pg = analyzer.sch_symbol
    if candidate_pg not in EXPECTED_GROUP_ORDER:
        raise ValueError(f"pymatgen 返回当前 schema 不支持的点群: {candidate_pg}")
    expected_order = EXPECTED_GROUP_ORDER[candidate_pg]
    operations = _safe_closure(
        _canonical_generators(analyzer, candidate_pg),
        protocol.matrix_tolerance,
        maximum_size=expected_order,
    )
    assignments = [_assign_operation(symbols, positions, matrix) for matrix in operations]
    permutations = [item[0] for item in assignments]
    rms_errors = np.asarray([item[1] for item in assignments], dtype=np.float64)
    max_errors = np.asarray([item[2] for item in assignments], dtype=np.float64)
    orthogonality = np.asarray(
        [np.max(np.abs(matrix.T @ matrix - np.eye(3))) for matrix in operations]
    )
    valid = (rms_errors <= protocol.tolerance_angstrom + 1e-12) & (orthogonality <= 1e-6)
    if len(operations) == expected_order and bool(valid.all()):
        actual_indices = tuple(range(len(operations)))
        actual_pg = candidate_pg
    else:
        actual_indices = _largest_valid_subgroup(operations, valid, protocol.matrix_tolerance)
        actual_pg = _classify_group([operations[index] for index in actual_indices])
    selected_permutations = [permutations[index] for index in actual_indices]
    selected_rms = rms_errors[list(actual_indices)]
    selected_max = max_errors[list(actual_indices)]
    orbit_id, orbit_size = _orbits(selected_permutations, len(symbols))
    return {
        "analyzer_pg": candidate_pg,
        "actual_pg": actual_pg,
        "actual": {
            "rms_errors": selected_rms,
            "max_errors": selected_max,
            "mean_rms_error": float(selected_rms.mean()),
            "max_rms_error": float(selected_rms.max()),
            "max_atom_error": float(selected_max.max()),
            "orbit_id": orbit_id,
            "orbit_size": orbit_size,
        },
    }


def analyze_evaluation_symmetry(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    target_pg: str | None,
    protocol: SymmetryProtocol,
) -> dict[str, Any]:
    if not np.isfinite(positions).all():
        raise ValueError("symmetry analyzer 收到无效坐标")
    centered = np.asarray(positions, dtype=np.float64) - np.asarray(
        positions, dtype=np.float64
    ).mean(axis=0)
    symbols = atomic_symbols(atomic_numbers)
    if target_pg is None:
        return _actual_only(symbols, centered, protocol)
    if target_pg not in SUPPORTED_TARGET_POINT_GROUPS:
        raise ValueError(f"不支持的 target_pg: {target_pg}")
    return analyze_symmetry(symbols, centered, target_pg, protocol)

