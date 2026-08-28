"""确定性的模型无关评测入口。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator
import numpy as np
from rdkit import Chem

from generative_model.data.symmetry import SymmetryProtocol

from .chemistry import (
    BOND_TYPE_NAMES,
    EvaluationReference,
    fingerprint_generator,
    nearest_similarity,
    reconstruct_molecule,
)
from .geometry import bond_angles_degrees, bond_lengths, mmff_relax, nonbonded_distances
from .schema import (
    EVALUATION_SCHEMA_VERSION,
    REPORT_SCHEMA_PATH,
    EvaluationInputError,
    EvaluationSample,
)
from .symmetry_metrics import analyze_evaluation_symmetry


METRIC_SUITE_VERSION = "p1.0"


@dataclass(frozen=True)
class EvaluationConfig:
    """统一 P1 指标的全部可复现参数。"""

    coordinate_unit: str = "angstrom"
    collision_distance_angstrom: float = 0.6
    centering_tolerance_angstrom: float = 2e-5
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    fingerprint_include_chirality: bool = False
    run_mmff: bool = True
    mmff_variant: str = "MMFF94s"
    mmff_max_iterations: int = 200
    run_symmetry: bool = True
    symmetry_tolerance_angstrom: float = 0.3
    symmetry_eigen_tolerance: float = 0.01
    symmetry_matrix_tolerance: float = 0.1

    def validate(self) -> None:
        if self.coordinate_unit != "angstrom":
            raise ValueError("当前评测只接受 angstrom 坐标")
        if self.collision_distance_angstrom <= 0:
            raise ValueError("collision_distance_angstrom 必须为正")
        if self.centering_tolerance_angstrom < 0:
            raise ValueError("centering_tolerance_angstrom 不能为负")
        if self.fingerprint_radius <= 0 or self.fingerprint_bits <= 0:
            raise ValueError("Morgan fingerprint 参数必须为正")
        if self.mmff_variant not in {"MMFF94", "MMFF94s"}:
            raise ValueError("mmff_variant 只支持 MMFF94 或 MMFF94s")
        if self.mmff_max_iterations <= 0:
            raise ValueError("mmff_max_iterations 必须为正")
        if min(
            self.symmetry_tolerance_angstrom,
            self.symmetry_eigen_tolerance,
            self.symmetry_matrix_tolerance,
        ) <= 0:
            raise ValueError("symmetry tolerance 必须为正")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _summary(values: list[float | int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("distribution summary 收到非有限值")
    return {
        "count": len(values),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _connected_components(atom_count: int, bond_index: np.ndarray) -> int:
    parent = np.arange(atom_count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for begin, end in bond_index.T:
        root_begin, root_end = find(int(begin)), find(int(end))
        if root_begin != root_end:
            parent[root_end] = root_begin
    return len({find(index) for index in range(atom_count)})


def _pair_distances(positions: np.ndarray) -> np.ndarray:
    atom_count = len(positions)
    if atom_count < 2:
        return np.empty(0, dtype=np.float64)
    begin, end = np.triu_indices(atom_count, k=1)
    return np.linalg.norm(positions[begin] - positions[end], axis=1)


def evaluate(
    samples: Iterable[Mapping[str, Any] | EvaluationSample],
    *,
    provenance: Mapping[str, Any],
    report_type: str,
    config: EvaluationConfig | None = None,
    reference: EvaluationReference | None = None,
) -> dict[str, Any]:
    """用相同契约评测真实分子或模型生成分子。

    输入 schema 错误进入顶层 ``failures``。化学、MMFF 和 symmetry 失败属于模型质量结果，
    分别进入对应 metric，不会修改原子、键、电荷或坐标。意外程序异常不会被吞掉。
    """

    config = config or EvaluationConfig()
    config.validate()
    required_provenance = {"source_type", "source_fingerprint", "coordinate_unit"}
    missing = sorted(required_provenance - set(provenance))
    if missing:
        raise ValueError(f"provenance 缺少字段: {', '.join(missing)}")
    if provenance["coordinate_unit"] != config.coordinate_unit:
        raise ValueError("provenance 与 EvaluationConfig 的坐标单位不一致")
    if reference is not None and (
        reference.fingerprint_radius != config.fingerprint_radius
        or reference.fingerprint_bits != config.fingerprint_bits
        or reference.include_chirality != config.fingerprint_include_chirality
    ):
        raise ValueError("reference 与 EvaluationConfig 的 fingerprint 参数不一致")

    fp_generator = fingerprint_generator(
        config.fingerprint_radius,
        config.fingerprint_bits,
        config.fingerprint_include_chirality,
    )
    symmetry_protocol = SymmetryProtocol(
        tolerance_angstrom=config.symmetry_tolerance_angstrom,
        eigen_tolerance=config.symmetry_eigen_tolerance,
        matrix_tolerance=config.symmetry_matrix_tolerance,
    )
    periodic_table = Chem.GetPeriodicTable()

    atom_counts: list[int] = []
    bond_counts: list[int] = []
    component_counts: list[int] = []
    max_radii: list[float] = []
    min_distances: list[float] = []
    finite_count = centered_count = connected_count = collision_count = 0
    provided_target_pg: Counter[str] = Counter()
    provided_actual_pg: Counter[str] = Counter()
    provided_compatible: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    failure_records: list[dict[str, Any]] = []

    element_counts: Counter[str] = Counter()
    bond_type_counts: Counter[str] = Counter()
    chemistry_valid_count = 0
    sanitize_failures: Counter[str] = Counter()
    canonical_smiles: list[str] = []
    ring_counts: list[int] = []
    nearest_similarities: list[float] = []
    novel_count = 0

    all_bond_lengths: list[float] = []
    bond_lengths_by_type: dict[str, list[float]] = {
        name: [] for name in BOND_TYPE_NAMES.values()
    }
    all_bond_angles: list[float] = []
    degenerate_angle_count = 0
    min_nonbonded_distances: list[float] = []
    nonbonded_collision_count = 0
    mmff_status: Counter[str] = Counter()
    mmff_initial_energy: list[float] = []
    mmff_final_energy: list[float] = []
    mmff_energy_change: list[float] = []
    mmff_displacement: list[float] = []

    symmetry_analyzed_count = 0
    symmetry_failures: Counter[str] = Counter()
    recomputed_analyzer_pg: Counter[str] = Counter()
    recomputed_actual_pg: Counter[str] = Counter()
    actual_label_comparison_count = actual_label_match_count = 0
    conditioned_symmetry_count = target_exact_count = target_compatible_count = 0
    symmetry_mean_rms: list[float] = []
    symmetry_max_rms: list[float] = []
    symmetry_max_atom_error: list[float] = []
    orbit_sizes: list[int] = []
    symmetry_by_target: dict[str, dict[str, Any]] = {}
    input_count = 0

    for sample_index, raw_sample in enumerate(samples):
        input_count += 1
        try:
            sample = raw_sample if isinstance(raw_sample, EvaluationSample) else EvaluationSample.from_mapping(raw_sample)
        except EvaluationInputError as error:
            molecule_id = None
            if isinstance(raw_sample, Mapping) and raw_sample.get("molecule_id") is not None:
                molecule_id = str(raw_sample["molecule_id"])
            failure_counts[error.reason.value] += 1
            failure_records.append(
                {
                    "sample_index": sample_index,
                    "molecule_id": molecule_id,
                    "reason": error.reason.value,
                    "message": str(error),
                }
            )
            continue

        atom_count = len(sample.atomic_numbers)
        atom_counts.append(atom_count)
        bond_counts.append(len(sample.bond_types))
        components = _connected_components(atom_count, sample.bond_index)
        component_counts.append(components)
        connected_count += int(components == 1)
        for atomic_number in sample.atomic_numbers:
            element_counts[str(periodic_table.GetElementSymbol(int(atomic_number)))] += 1
        for bond_type in sample.bond_types:
            bond_type_counts[BOND_TYPE_NAMES[int(bond_type)]] += 1

        if sample.target_pg is not None:
            provided_target_pg[sample.target_pg] += 1
        if sample.actual_pg is not None:
            provided_actual_pg[sample.actual_pg] += 1
        provided_compatible[
            "unknown" if sample.pg_compatible is None else str(sample.pg_compatible).lower()
        ] += 1

        positions_finite = bool(np.isfinite(sample.positions).all())
        if positions_finite:
            finite_count += 1
            centroid_norm = float(np.linalg.norm(sample.positions.mean(axis=0)))
            centered_count += int(centroid_norm <= config.centering_tolerance_angstrom)
            max_radii.append(float(np.linalg.norm(sample.positions, axis=1).max()))
            distances = _pair_distances(sample.positions)
            if len(distances):
                minimum = float(distances.min())
                min_distances.append(minimum)
                collision_count += int(minimum < config.collision_distance_angstrom)

            lengths = bond_lengths(sample)
            all_bond_lengths.extend(map(float, lengths))
            for bond_type, length in zip(sample.bond_types, lengths):
                bond_lengths_by_type[BOND_TYPE_NAMES[int(bond_type)]].append(float(length))
            angles = bond_angles_degrees(sample)
            finite_angles = angles[np.isfinite(angles)]
            all_bond_angles.extend(map(float, finite_angles))
            degenerate_angle_count += int(len(angles) - len(finite_angles))
            nonbonded = nonbonded_distances(sample)
            if len(nonbonded):
                minimum_nonbonded = float(nonbonded.min())
                min_nonbonded_distances.append(minimum_nonbonded)
                nonbonded_collision_count += int(
                    minimum_nonbonded < config.collision_distance_angstrom
                )

        chemistry = reconstruct_molecule(sample, generator=fp_generator)
        if chemistry.valid:
            chemistry_valid_count += 1
            if chemistry.canonical_smiles is None or chemistry.fingerprint is None:
                raise RuntimeError("valid ChemistryResult 缺少 SMILES 或 fingerprint")
            canonical_smiles.append(chemistry.canonical_smiles)
            ring_counts.append(int(chemistry.ring_count or 0))
            if reference is not None:
                novel_count += int(chemistry.canonical_smiles not in reference.canonical_smiles)
                similarity = nearest_similarity(chemistry.fingerprint, reference)
                if similarity is not None:
                    nearest_similarities.append(similarity)
            if config.run_mmff and positions_finite:
                if chemistry.molecule is None:
                    raise RuntimeError("valid ChemistryResult 缺少 RDKit molecule")
                relaxed = mmff_relax(
                    chemistry.molecule,
                    variant=config.mmff_variant,
                    max_iterations=config.mmff_max_iterations,
                )
                mmff_status[relaxed.status] += 1
                if relaxed.initial_energy_kcal_mol is not None:
                    mmff_initial_energy.append(relaxed.initial_energy_kcal_mol)
                if relaxed.final_energy_kcal_mol is not None:
                    mmff_final_energy.append(relaxed.final_energy_kcal_mol)
                if (
                    relaxed.initial_energy_kcal_mol is not None
                    and relaxed.final_energy_kcal_mol is not None
                ):
                    mmff_energy_change.append(
                        relaxed.final_energy_kcal_mol - relaxed.initial_energy_kcal_mol
                    )
                if relaxed.centered_rms_displacement_angstrom is not None:
                    mmff_displacement.append(relaxed.centered_rms_displacement_angstrom)
        else:
            sanitize_failures[str(chemistry.sanitize_error)] += 1

        if config.run_symmetry and positions_finite:
            try:
                result = analyze_evaluation_symmetry(
                    sample.atomic_numbers,
                    sample.positions,
                    sample.target_pg,
                    symmetry_protocol,
                )
            except ValueError as error:
                reason = "unsupported_target_pg" if "不支持的 target_pg" in str(error) else "analysis_error"
                symmetry_failures[reason] += 1
            else:
                symmetry_analyzed_count += 1
                recomputed_analyzer_pg[str(result["analyzer_pg"])] += 1
                recomputed_actual_pg[str(result["actual_pg"])] += 1
                if sample.actual_pg is not None:
                    actual_label_comparison_count += 1
                    actual_label_match_count += int(result["actual_pg"] == sample.actual_pg)
                if sample.target_pg is not None:
                    conditioned_symmetry_count += 1
                    exact = int(result["actual_pg"] == sample.target_pg)
                    compatible = int(bool(result["pg_compatible"]))
                    target_exact_count += exact
                    target_compatible_count += compatible
                    target_block = symmetry_by_target.setdefault(
                        sample.target_pg,
                        {
                            "analyzed": 0,
                            "exact": 0,
                            "compatible": 0,
                            "mean_rms": [],
                            "max_rms": [],
                            "orbit_sizes": [],
                        },
                    )
                    target_block["analyzed"] += 1
                    target_block["exact"] += exact
                    target_block["compatible"] += compatible
                actual = result["actual"]
                symmetry_mean_rms.append(float(actual["mean_rms_error"]))
                symmetry_max_rms.append(float(actual["max_rms_error"]))
                symmetry_max_atom_error.append(float(actual["max_atom_error"]))
                ids = np.asarray(actual["orbit_id"])
                sizes = np.asarray(actual["orbit_size"])
                for orbit_id in np.unique(ids):
                    orbit_sizes.append(int(sizes[np.flatnonzero(ids == orbit_id)[0]]))
                if sample.target_pg is not None:
                    target_block["mean_rms"].append(float(actual["mean_rms_error"]))
                    target_block["max_rms"].append(float(actual["max_rms_error"]))
                    for orbit_id in np.unique(ids):
                        target_block["orbit_sizes"].append(
                            int(sizes[np.flatnonzero(ids == orbit_id)[0]])
                        )

    evaluated_count = len(atom_counts)
    failed_count = input_count - evaluated_count

    def rate(count: int, denominator: int = evaluated_count) -> float | None:
        return None if denominator == 0 else count / denominator

    unique_smiles = len(set(canonical_smiles))
    mmff_requested = sum(mmff_status.values())
    mmff_parameterized = mmff_status["converged"] + mmff_status["not_converged"]
    report: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "metric_suite_version": METRIC_SUITE_VERSION,
        "report_type": report_type,
        "provenance": dict(provenance),
        "reference": None if reference is None else reference.describe(),
        "config": asdict(config),
        "summary": {
            "status": "PASS" if failed_count == 0 else "FAIL",
            "input_molecules": input_count,
            "evaluated_molecules": evaluated_count,
            "failed_molecules": failed_count,
        },
        "metrics": {
            "size": {
                "atom_count": _summary(atom_counts),
                "bond_count": _summary(bond_counts),
            },
            "numerical": {
                "finite_coordinate_molecules": finite_count,
                "finite_coordinate_rate": rate(finite_count),
                "centered_molecules": centered_count,
                "centered_rate": rate(centered_count),
                "collision_molecules": collision_count,
                "collision_rate": rate(collision_count),
                "max_radius_angstrom": _summary(max_radii),
                "min_interatomic_distance_angstrom": _summary(min_distances),
            },
            "graph": {
                "connected_molecules": connected_count,
                "connected_rate": rate(connected_count),
                "component_count": _summary(component_counts),
            },
            "chemistry": {
                "rdkit_sanitize_valid_molecules": chemistry_valid_count,
                "rdkit_sanitize_valid_rate": rate(chemistry_valid_count),
                "sanitize_failure_counts": dict(sorted(sanitize_failures.items())),
            },
            "two_dimensional": {
                "element_counts": dict(sorted(element_counts.items())),
                "bond_type_counts": dict(sorted(bond_type_counts.items())),
                "ring_count": _summary(ring_counts),
                "unique_canonical_smiles": unique_smiles,
                "uniqueness_rate": rate(unique_smiles, chemistry_valid_count),
                "novel_molecules": novel_count if reference is not None else None,
                "novelty_rate": (
                    rate(novel_count, chemistry_valid_count) if reference is not None else None
                ),
                "nearest_train_tanimoto": _summary(nearest_similarities),
            },
            "three_dimensional": {
                "bond_length_angstrom": _summary(all_bond_lengths),
                "bond_length_by_type_angstrom": {
                    name: _summary(values) for name, values in sorted(bond_lengths_by_type.items())
                },
                "bond_angle_degrees": _summary(all_bond_angles),
                "degenerate_angle_count": degenerate_angle_count,
                "min_nonbonded_distance_angstrom": _summary(min_nonbonded_distances),
                "nonbonded_collision_molecules": nonbonded_collision_count,
                "nonbonded_collision_rate": rate(nonbonded_collision_count),
                "mmff": {
                    "requested_molecules": mmff_requested,
                    "parameterized_molecules": mmff_parameterized,
                    "status_counts": dict(sorted(mmff_status.items())),
                    "converged_rate_among_parameterized": rate(
                        mmff_status["converged"], mmff_parameterized
                    ),
                    "initial_energy_kcal_mol": _summary(mmff_initial_energy),
                    "final_energy_kcal_mol": _summary(mmff_final_energy),
                    "energy_change_kcal_mol": _summary(mmff_energy_change),
                    "centered_rms_displacement_angstrom": _summary(mmff_displacement),
                },
            },
            "symmetry": {
                "analyzed_molecules": symmetry_analyzed_count,
                "analyzed_rate": rate(symmetry_analyzed_count),
                "failure_counts": dict(sorted(symmetry_failures.items())),
                "analyzer_pg_counts": dict(sorted(recomputed_analyzer_pg.items())),
                "actual_pg_counts": dict(sorted(recomputed_actual_pg.items())),
                "actual_label_comparison_molecules": actual_label_comparison_count,
                "actual_label_match_rate": rate(
                    actual_label_match_count, actual_label_comparison_count
                ),
                "conditioned_molecules": conditioned_symmetry_count,
                "target_exact_rate": rate(target_exact_count, conditioned_symmetry_count),
                "target_compatible_rate": rate(
                    target_compatible_count, conditioned_symmetry_count
                ),
                "actual_mean_operation_rms_angstrom": _summary(symmetry_mean_rms),
                "actual_max_operation_rms_angstrom": _summary(symmetry_max_rms),
                "actual_max_atom_error_angstrom": _summary(symmetry_max_atom_error),
                "orbit_size": _summary(orbit_sizes),
                "by_target_pg": {
                    point_group: {
                        "analyzed_molecules": block["analyzed"],
                        "target_exact_rate": rate(block["exact"], block["analyzed"]),
                        "target_compatible_rate": rate(
                            block["compatible"], block["analyzed"]
                        ),
                        "actual_mean_operation_rms_angstrom": _summary(block["mean_rms"]),
                        "actual_max_operation_rms_angstrom": _summary(block["max_rms"]),
                        "orbit_size": _summary(block["orbit_sizes"]),
                    }
                    for point_group, block in sorted(symmetry_by_target.items())
                },
            },
            "provided_conditions": {
                "target_pg_counts": dict(sorted(provided_target_pg.items())),
                "actual_pg_counts": dict(sorted(provided_actual_pg.items())),
                "pg_compatible_counts": dict(sorted(provided_compatible.items())),
            },
        },
        "failures": {
            "counts": dict(sorted(failure_counts.items())),
            "records": failure_records,
        },
    }
    report["report_fingerprint"] = _sha256_json(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    """校验 JSON Schema、计数不变量与报告内容指纹。"""

    schema = json.loads(REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(dict(report))
    summary = report["summary"]
    if summary["input_molecules"] != summary["evaluated_molecules"] + summary["failed_molecules"]:
        raise ValueError("evaluation report 的分子计数不守恒")
    failures = report["failures"]
    if sum(failures["counts"].values()) != summary["failed_molecules"]:
        raise ValueError("evaluation report 的失败计数不守恒")
    fingerprint = report["report_fingerprint"]
    payload = dict(report)
    del payload["report_fingerprint"]
    if fingerprint != _sha256_json(payload):
        raise ValueError("evaluation report fingerprint 不匹配")


def write_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    """以固定 JSON 格式写出报告。"""

    validate_report(report)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")
