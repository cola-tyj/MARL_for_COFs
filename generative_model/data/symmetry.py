"""可复现的三维点群、操作、原子置换与轨道分析。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
import re
from typing import Any

import numpy as np
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from scipy.optimize import linear_sum_assignment


EXPECTED_GROUP_ORDER = {
    "C1": 1,
    "Cs": 2,
    "Ci": 2,
    "C2": 2,
    "C2v": 4,
    "C2h": 4,
    "D2": 4,
    "D2h": 8,
    "D2d": 8,
    "C3": 3,
    "C3v": 6,
    "C3h": 6,
    "D3": 6,
    "D3h": 12,
    "D3d": 12,
    "S4": 4,
    "D6": 12,
    "D6h": 24,
}


@dataclass(frozen=True)
class SymmetryProtocol:
    analyzer: str = "pymatgen.symmetry.analyzer.PointGroupAnalyzer"
    tolerance_angstrom: float = 0.3
    eigen_tolerance: float = 0.01
    matrix_tolerance: float = 0.1
    assignment: str = "scipy.optimize.linear_sum_assignment, element-blocked"
    assignment_metric: str = "Euclidean distance in centroid-centered Cartesian coordinates"
    operation_acceptance: str = "operation RMS assigned distance <= tolerance_angstrom"


def _matrix_close(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    return bool(np.max(np.abs(left - right)) <= tolerance)


def _deduplicate(matrices: list[np.ndarray], tolerance: float) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for matrix in matrices:
        matrix = np.asarray(matrix, dtype=np.float64)
        if not any(_matrix_close(matrix, present, tolerance) for present in unique):
            unique.append(matrix)
    return unique


def _sort_operations(matrices: list[np.ndarray]) -> list[np.ndarray]:
    identity = np.eye(3)
    return sorted(
        matrices,
        key=lambda matrix: (
            0 if np.allclose(matrix, identity, atol=1e-6) else 1,
            tuple(np.round(matrix, 8).ravel()),
        ),
    )


def _normalize(vector: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        raise ValueError(f"无法归一化 {label}")
    return vector / norm


def _rotation(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis = _normalize(axis, "rotation axis")
    x, y, z = axis
    angle = np.radians(angle_degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return cosine * np.eye(3) + (1.0 - cosine) * np.outer(axis, axis) + sine * cross


def _reflection(normal: np.ndarray) -> np.ndarray:
    normal = _normalize(normal, "reflection normal")
    return np.eye(3) - 2.0 * np.outer(normal, normal)


def _axis_from_rotation(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(matrix)
    index = int(np.argmin(np.abs(values - 1.0)))
    return _normalize(np.real(vectors[:, index]), "C2 axis")


def _normal_from_reflection(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(matrix)
    index = int(np.argmin(np.abs(values + 1.0)))
    return _normalize(np.real(vectors[:, index]), "mirror normal")


def _perpendicular(vector: np.ndarray, axis: np.ndarray, label: str) -> np.ndarray:
    axis = _normalize(axis, "main axis")
    projected = np.asarray(vector, dtype=np.float64) - np.dot(vector, axis) * axis
    return _normalize(projected, label)


def _canonical_generators(analyzer: PointGroupAnalyzer, point_group: str) -> list[np.ndarray]:
    """从 pymatgen 检出的方向重建满足精确有限群关系的生成元。"""
    rotations = [(np.asarray(axis, dtype=np.float64), int(order)) for axis, order in analyzer.rot_sym if order > 1]
    if not rotations:
        if point_group in {"C1", "Cs", "Ci"}:
            rotations = []
        else:
            raise ValueError(f"{point_group} 缺少非平凡旋转轴")
    number_match = re.search(r"\d+", point_group)
    n = int(number_match.group()) if number_match else 1
    main_axis = _normalize(max(rotations, key=lambda item: item[1])[0], "main axis") if rotations else None

    improper_ops = [
        np.asarray(operation.rotation_matrix, dtype=np.float64)
        for operation in analyzer.symmops
        if np.linalg.det(operation.rotation_matrix) < 0
    ]
    generators: list[np.ndarray] = []
    if point_group == "C1":
        return generators
    if point_group == "Ci":
        return [-np.eye(3)]
    if point_group == "Cs":
        if not improper_ops:
            raise ValueError("Cs 缺少镜面操作")
        return [_reflection(_normal_from_reflection(improper_ops[0]))]

    if point_group.startswith("S"):
        generators.append(_rotation(main_axis, 360.0 / n) @ _reflection(main_axis))
        return generators

    generators.append(_rotation(main_axis, 360.0 / n))

    if point_group.startswith("D"):
        secondary_axes: list[np.ndarray] = []
        for axis, order in rotations:
            if order != 2:
                continue
            try:
                secondary_axes.append(_perpendicular(axis, main_axis, "secondary C2 axis"))
            except ValueError:
                continue
        if not secondary_axes:
            # pymatgen 的 symmetric-top 路径可能用 order=1 轴误报 D2；保留真实 C2 供后续降级。
            if point_group == "D2":
                return generators
            if point_group == "D2h":
                return [*generators, _reflection(main_axis)]
            if point_group == "D2d":
                return [_rotation(main_axis, 90.0) @ _reflection(main_axis)]
            raise ValueError(f"{point_group} 缺少垂直于主轴的 C2 轴")
        secondary = secondary_axes[0]
        if point_group == "D2d":
            return [_rotation(main_axis, 90.0) @ _reflection(main_axis), _rotation(secondary, 180.0)]
        generators.append(_rotation(secondary, 180.0))
        if point_group.endswith("h"):
            generators.append(_reflection(main_axis))
        elif point_group.endswith("d"):
            if point_group == "D3d":
                generators.append(-np.eye(3))
            else:
                raise ValueError(f"尚未定义 {point_group} 的标准生成元")
        return generators

    if point_group.endswith("h"):
        generators.append(_reflection(main_axis))
    elif point_group.endswith("v"):
        if not improper_ops:
            raise ValueError(f"{point_group} 缺少 vertical mirror")
        normals: list[np.ndarray] = []
        for operation in improper_ops:
            try:
                normals.append(
                    _perpendicular(
                        _normal_from_reflection(operation), main_axis, "vertical mirror normal"
                    )
                )
            except ValueError:
                continue
        if not normals:
            raise ValueError(f"{point_group} 无法确定 vertical mirror normal")
        generators.append(_reflection(normals[0]))
    return generators


def _safe_closure(
    generators: list[np.ndarray], matrix_tolerance: float, maximum_size: int
) -> list[np.ndarray]:
    operations = _deduplicate([np.eye(3), *generators], matrix_tolerance)
    changed = True
    while changed:
        changed = False
        snapshot = operations[:]
        for left in snapshot:
            for right in snapshot:
                product = left @ right
                if not any(_matrix_close(product, item, matrix_tolerance) for item in operations):
                    if len(operations) >= maximum_size:
                        raise ValueError(
                            f"操作闭包超过声明群阶 {maximum_size}；生成元不满足有限群关系"
                        )
                    operations.append(product)
                    changed = True
    return _sort_operations(operations)


def _operation_order(matrix: np.ndarray, tolerance: float = 0.1) -> int:
    power = np.eye(3)
    for order in range(1, 25):
        power = power @ matrix
        if np.allclose(power, np.eye(3), atol=tolerance):
            return order
    return 0


def _classify_group(operations: list[np.ndarray]) -> str:
    """按有限 O(3) 操作集合分类项目当前可能出现的 Schoenflies 群。"""
    size = len(operations)
    dets = [int(round(np.linalg.det(matrix))) for matrix in operations]
    orders = [_operation_order(matrix) for matrix in operations]
    proper = [matrix for matrix, determinant in zip(operations, dets) if determinant == 1]
    proper_orders = [_operation_order(matrix) for matrix in proper]
    has_inversion = any(np.allclose(matrix, -np.eye(3), atol=1e-5) for matrix in operations)
    has_improper_order4 = any(det == -1 and order == 4 for det, order in zip(dets, orders))
    abelian = all(
        np.allclose(left @ right, right @ left, atol=0.1)
        for left in operations
        for right in operations
    )
    max_proper = max(proper_orders, default=1)

    if size == 1:
        return "C1"
    if size == 2:
        if len(proper) == 2:
            return "C2"
        return "Ci" if has_inversion else "Cs"
    if size == 3 and len(proper) == 3:
        return "C3"
    if size == 4:
        if len(proper) == 4:
            return "C4" if max_proper == 4 else "D2"
        if has_improper_order4:
            return "S4"
        return "C2h" if has_inversion else "C2v"
    if size == 6:
        if len(proper) == 6:
            return "C6" if max_proper == 6 else "D3"
        return "C3h" if abelian else "C3v"
    if size == 8 and len(proper) == 4:
        if has_inversion:
            return "D2h" if max_proper == 2 else "C4h"
        if has_improper_order4 and max_proper == 2:
            return "D2d"
        return "C4v"
    if size == 12 and len(proper) == 6:
        if max_proper == 3:
            return "D3d" if has_inversion else "D3h"
        return "C6h" if abelian else "C6v"
    if size == 12 and len(proper) == 12 and max_proper == 6:
        return "D6"
    if size == 24 and len(proper) == 12 and max_proper == 6:
        return "D6h" if has_inversion else "D6d"
    raise ValueError(
        f"无法分类验证操作群：size={size}, proper={len(proper)}, orders={sorted(orders)}"
    )


def _assign_operation(
    symbols: list[str], positions: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, float, float]:
    transformed = positions @ matrix.T
    permutation = np.empty(len(symbols), dtype=np.int32)
    distances = np.empty(len(symbols), dtype=np.float64)
    symbol_array = np.asarray(symbols)
    for symbol in sorted(set(symbols)):
        indices = np.flatnonzero(symbol_array == symbol)
        delta = transformed[indices, None, :] - positions[indices][None, :, :]
        cost = np.linalg.norm(delta, axis=2)
        source_rows, target_columns = linear_sum_assignment(cost)
        if len(source_rows) != len(indices):
            raise ValueError(f"Hungarian assignment 未覆盖元素 {symbol} 的全部原子")
        source = indices[source_rows]
        target = indices[target_columns]
        permutation[source] = target
        distances[source] = cost[source_rows, target_columns]
    if len(np.unique(permutation)) != len(permutation):
        raise ValueError("操作对应的原子 permutation 不是双射")
    rms = float(np.sqrt(np.mean(np.square(distances))))
    return permutation, rms, float(distances.max(initial=0.0))


def _find_matrix_index(
    matrix: np.ndarray, operations: list[np.ndarray], tolerance: float
) -> int | None:
    errors = [float(np.max(np.abs(matrix - item))) for item in operations]
    index = int(np.argmin(errors))
    return index if errors[index] <= tolerance else None


def _subgroup_indices(
    seeds: tuple[int, ...], operations: list[np.ndarray], tolerance: float
) -> tuple[int, ...] | None:
    identity = _find_matrix_index(np.eye(3), operations, tolerance)
    if identity is None:
        return None
    selected = {identity, *seeds}
    changed = True
    while changed:
        changed = False
        snapshot = sorted(selected)
        for left in snapshot:
            for right in snapshot:
                index = _find_matrix_index(operations[left] @ operations[right], operations, tolerance)
                if index is None:
                    return None
                if index not in selected:
                    selected.add(index)
                    changed = True
    return tuple(sorted(selected))


def _largest_valid_subgroup(
    operations: list[np.ndarray], valid: np.ndarray, matrix_tolerance: float
) -> tuple[int, ...]:
    valid_indices = tuple(int(index) for index in np.flatnonzero(valid))
    candidates: set[tuple[int, ...]] = set()
    for left, right in combinations_with_replacement(valid_indices, 2):
        subgroup = _subgroup_indices((left, right), operations, matrix_tolerance)
        if subgroup is not None and all(valid[index] for index in subgroup):
            candidates.add(subgroup)
    if not candidates:
        identity = _find_matrix_index(np.eye(3), operations, matrix_tolerance)
        if identity is None or not valid[identity]:
            raise ValueError("identity 操作未通过 assignment")
        return (identity,)
    return max(candidates, key=lambda indices: (len(indices), tuple(-index for index in indices)))


def _target_subgroup(
    target_pg: str,
    operations: list[np.ndarray],
    rms_errors: np.ndarray,
    matrix_tolerance: float,
) -> tuple[int, ...]:
    expected = EXPECTED_GROUP_ORDER[target_pg]
    if target_pg == "D6h":
        if len(operations) != expected or _classify_group(operations) != "D6h":
            raise ValueError("候选操作中不存在完整 D6h 目标群")
        return tuple(range(len(operations)))

    desired_det = -1 if target_pg == "S4" else 1
    desired_order = {"C2": 2, "C3": 3, "S4": 4}[target_pg]
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for index, matrix in enumerate(operations):
        if int(round(np.linalg.det(matrix))) != desired_det:
            continue
        if _operation_order(matrix) != desired_order:
            continue
        subgroup = _subgroup_indices((index,), operations, matrix_tolerance)
        if subgroup is None or len(subgroup) != expected:
            continue
        score = float(np.mean(rms_errors[list(subgroup)]))
        candidates.append((score, subgroup))
    if not candidates:
        raise ValueError(f"候选操作中不存在完整 {target_pg} 目标子群")
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def _orbits(permutations: list[np.ndarray], atom_count: int) -> tuple[np.ndarray, np.ndarray]:
    parent = np.arange(atom_count, dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root

    for permutation in permutations:
        for source, target in enumerate(permutation):
            union(source, int(target))
    roots = np.asarray([find(index) for index in range(atom_count)], dtype=np.int32)
    unique_roots = sorted(set(int(root) for root in roots))
    root_to_orbit = {root: orbit for orbit, root in enumerate(unique_roots)}
    orbit_id = np.asarray([root_to_orbit[int(root)] for root in roots], dtype=np.uint16)
    counts = np.bincount(orbit_id, minlength=len(unique_roots))
    return orbit_id, counts[orbit_id].astype(np.uint16)


def analyze_symmetry(
    symbols: list[str],
    positions: np.ndarray,
    target_pg: str,
    protocol: SymmetryProtocol,
) -> dict[str, Any]:
    if target_pg not in {"C2", "C3", "S4", "D6h"}:
        raise ValueError(f"不支持的 target_pg: {target_pg}")
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (len(symbols), 3) or not np.isfinite(positions).all():
        raise ValueError("symmetry analyzer 收到无效坐标")
    if not np.allclose(positions.mean(axis=0), 0.0, atol=2e-5):
        raise ValueError("symmetry analyzer 要求 canonical 质心归零坐标")

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
    generators = _canonical_generators(analyzer, candidate_pg)

    candidate_operations = _safe_closure(
        generators, protocol.matrix_tolerance, maximum_size=expected_order
    )
    assignments = [_assign_operation(symbols, positions, matrix) for matrix in candidate_operations]
    permutations = [item[0] for item in assignments]
    rms_errors = np.asarray([item[1] for item in assignments], dtype=np.float64)
    max_errors = np.asarray([item[2] for item in assignments], dtype=np.float64)
    orthogonality = np.asarray(
        [np.max(np.abs(matrix.T @ matrix - np.eye(3))) for matrix in candidate_operations]
    )
    valid = (rms_errors <= protocol.tolerance_angstrom + 1e-12) & (orthogonality <= 1e-6)

    if len(candidate_operations) == expected_order and bool(valid.all()):
        actual_indices = tuple(range(len(candidate_operations)))
        actual_pg = candidate_pg
    else:
        actual_indices = _largest_valid_subgroup(
            candidate_operations, valid, protocol.matrix_tolerance
        )
        actual_pg = _classify_group([candidate_operations[index] for index in actual_indices])

    target_indices = _target_subgroup(
        target_pg, candidate_operations, rms_errors, protocol.matrix_tolerance
    )
    compatible = all(
        any(
            _matrix_close(candidate_operations[target], candidate_operations[actual], protocol.matrix_tolerance)
            for actual in actual_indices
        )
        for target in target_indices
    )

    def collect(indices: tuple[int, ...]) -> dict[str, Any]:
        matrices = [candidate_operations[index] for index in indices]
        perms = [permutations[index] for index in indices]
        rms = rms_errors[list(indices)]
        maximum = max_errors[list(indices)]
        orbit_id, orbit_size = _orbits(perms, len(symbols))
        return {
            "matrices": matrices,
            "permutations": perms,
            "rms_errors": rms,
            "max_errors": maximum,
            "mean_rms_error": float(rms.mean()),
            "max_rms_error": float(rms.max()),
            "max_atom_error": float(maximum.max()),
            "orbit_id": orbit_id,
            "orbit_size": orbit_size,
        }

    return {
        "target_pg": target_pg,
        "analyzer_pg": candidate_pg,
        "actual_pg": actual_pg,
        "pg_exact_match": actual_pg == target_pg,
        "pg_compatible": compatible,
        "actual": collect(actual_indices),
        "target": collect(target_indices),
    }
