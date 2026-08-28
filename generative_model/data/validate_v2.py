"""严格验证 v2 canonical、symmetry annotation、splits、相似度与哈希。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .build_cof_package import REPO_ROOT, sha256_file, write_json
from .schema import ATOM_SYMBOLS
from .symmetry import EXPECTED_GROUP_ORDER


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _similarity_distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    result = _distribution(values)
    result.update(
        {
            "equal_1_count": int(np.count_nonzero(np.isclose(values, 1.0, atol=1e-12))),
            "ge_0_9_count": int(np.count_nonzero(values >= 0.9)),
            "ge_0_8_count": int(np.count_nonzero(values >= 0.8)),
        }
    )
    return result


def _counts(values: Any) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items(), key=lambda x: str(x[0]))}


def _validate_operation_layer(
    key: str,
    canonical: dict[str, np.ndarray],
    symmetry: dict[str, np.ndarray],
    point_group_labels: np.ndarray,
    matrix_tolerance: float,
) -> None:
    matrices = symmetry[f"{key}_operation_matrices"]
    operation_offsets = symmetry[f"{key}_operation_offsets"]
    permutations = symmetry[f"{key}_permutation_index"]
    permutation_offsets = symmetry[f"{key}_permutation_offsets"]
    stored_rms = symmetry[f"{key}_operation_rms_error"]
    stored_max = symmetry[f"{key}_operation_max_error"]
    if len(operation_offsets) != len(canonical["atom_offsets"]):
        raise ValueError(f"{key} operation_offsets 与分子数不一致")
    if len(permutation_offsets) != len(matrices) + 1:
        raise ValueError(f"{key} permutation_offsets 与操作数不一致")
    if int(permutation_offsets[-1]) != len(permutations):
        raise ValueError(f"{key} permutation_offsets 未覆盖 permutation_index")
    if len(stored_rms) != len(matrices) or len(stored_max) != len(matrices):
        raise ValueError(f"{key} operation error 数量不一致")

    atom_types = canonical["atom_types"]
    positions = canonical["positions"].astype(np.float64)
    for molecule_index in range(len(operation_offsets) - 1):
        atom_start, atom_end = map(int, canonical["atom_offsets"][molecule_index : molecule_index + 2])
        local_positions = positions[atom_start:atom_end]
        local_types = atom_types[atom_start:atom_end]
        operation_start, operation_end = map(
            int, operation_offsets[molecule_index : molecule_index + 2]
        )
        expected_order = EXPECTED_GROUP_ORDER[str(point_group_labels[molecule_index])]
        if operation_end - operation_start != expected_order:
            raise ValueError(
                f"{key} 分子 {molecule_index} 操作数不是 {point_group_labels[molecule_index]} 的完整群阶"
            )
        if operation_end <= operation_start:
            raise ValueError(f"{key} 分子 {molecule_index} 没有 symmetry operation")
        local_matrices = matrices[operation_start:operation_end].astype(np.float64)
        local_permutations: list[np.ndarray] = []
        for operation_index in range(operation_start, operation_end):
            permutation_start, permutation_end = map(
                int, permutation_offsets[operation_index : operation_index + 2]
            )
            permutation = permutations[permutation_start:permutation_end]
            if len(permutation) != atom_end - atom_start:
                raise ValueError(f"{key} 分子 {molecule_index} permutation 长度错误")
            if not np.array_equal(np.sort(permutation), np.arange(len(permutation))):
                raise ValueError(f"{key} 分子 {molecule_index} permutation 不是双射")
            if not np.array_equal(local_types, local_types[permutation]):
                raise ValueError(f"{key} 分子 {molecule_index} permutation 跨元素映射")
            matrix = matrices[operation_index].astype(np.float64)
            if not np.allclose(matrix.T @ matrix, np.eye(3), atol=2e-5):
                raise ValueError(f"{key} 分子 {molecule_index} 操作矩阵不属于 O(3)")
            distances = np.linalg.norm(
                local_positions @ matrix.T - local_positions[permutation], axis=1
            )
            rms = float(np.sqrt(np.mean(np.square(distances))))
            maximum = float(distances.max(initial=0.0))
            if not np.isclose(rms, float(stored_rms[operation_index]), atol=2e-5):
                raise ValueError(f"{key} 分子 {molecule_index} RMS error 与存储值不一致")
            if not np.isclose(maximum, float(stored_max[operation_index]), atol=2e-5):
                raise ValueError(f"{key} 分子 {molecule_index} max error 与存储值不一致")
            local_permutations.append(permutation)
        # 同时验证矩阵闭包及 permutation 对群乘法的同态一致性。
        for left in range(len(local_matrices)):
            for right in range(len(local_matrices)):
                product = local_matrices[left] @ local_matrices[right]
                errors = np.max(np.abs(local_matrices - product[None, :, :]), axis=(1, 2))
                product_index = int(np.argmin(errors))
                if errors[product_index] > matrix_tolerance:
                    raise ValueError(f"{key} 分子 {molecule_index} 操作矩阵不闭合")
                composed = local_permutations[left][local_permutations[right]]
                if not np.array_equal(local_permutations[product_index], composed):
                    raise ValueError(f"{key} 分子 {molecule_index} permutation 不满足群乘法")


def _orbit_distribution(
    metadata: pd.DataFrame,
    atom_offsets: np.ndarray,
    orbit_id: np.ndarray,
    orbit_size: np.ndarray,
) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[int]] = defaultdict(Counter)
    for molecule_index, actual_pg in enumerate(metadata["Actual_PG"]):
        start, end = map(int, atom_offsets[molecule_index : molecule_index + 2])
        ids = orbit_id[start:end]
        sizes = orbit_size[start:end]
        for local_orbit in np.unique(ids):
            members = np.flatnonzero(ids == local_orbit)
            size = int(sizes[members[0]])
            if len(members) != size or np.any(sizes[members] != size):
                raise ValueError(f"分子 {molecule_index} orbit_size 与 orbit_id 不一致")
            result[str(actual_pg)][size] += 1
    return {
        point_group: {str(size): int(count) for size, count in sorted(counter.items())}
        for point_group, counter in sorted(result.items())
    }


def validate_v2(package_dir: Path, write_report: bool = True) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "2.0":
        raise ValueError(f"不支持的 v2 schema: {manifest['schema_version']}")
    for name, expected in manifest["package_files"].items():
        path = package_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"package hash 不一致: {name}")
    source_csv = REPO_ROOT / manifest["source_csv"]["path"]
    if sha256_file(source_csv) != manifest["source_csv"]["sha256"]:
        raise ValueError("source CSV hash 不一致")
    for item in manifest["source_xyz"]:
        path = REPO_ROOT / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"source XYZ hash 不一致: {item['path']}")

    metadata = pd.read_csv(package_dir / "metadata.csv")
    with np.load(package_dir / "cof_graphs.npz", allow_pickle=False) as archive:
        canonical = {name: archive[name] for name in archive.files}
    with np.load(package_dir / "symmetry_data.npz", allow_pickle=False) as archive:
        symmetry = {name: archive[name] for name in archive.files}
    with np.load(package_dir / "rdkit_features.npz", allow_pickle=False) as archive:
        features = {name: archive[name] for name in archive.files}
    symmetry_vocab = json.loads((package_dir / "symmetry_vocab.json").read_text(encoding="utf-8"))
    point_groups = symmetry_vocab["point_groups"]

    molecule_count = len(metadata)
    atom_count = len(canonical["atom_types"])
    bond_count = len(canonical["bond_types"])
    expected_counts = manifest["counts"]
    if (molecule_count, atom_count, bond_count) != (
        expected_counts["molecules"], expected_counts["atoms"], expected_counts["bonds"]
    ):
        raise ValueError("canonical counts 与 manifest 不一致")
    if molecule_count != 2532:
        raise ValueError(f"预期 2532 个分子，实际 {molecule_count}")
    if not np.array_equal(canonical["atom_offsets"], symmetry["atom_offsets"]):
        raise ValueError("symmetry atom_offsets 与 canonical 不一致")
    if not np.array_equal(canonical["atom_offsets"], features["atom_offsets"]):
        raise ValueError("RDKit atom_offsets 与 canonical 不一致")
    if not np.array_equal(canonical["bond_offsets"], features["bond_offsets"]):
        raise ValueError("RDKit bond_offsets 与 canonical 不一致")

    decoded = {
        field: np.asarray([point_groups[int(index)] for index in symmetry[field]])
        for field in ("target_pg", "analyzer_pg", "actual_pg")
    }
    if not np.array_equal(decoded["target_pg"], metadata["Target_PG"].astype(str)):
        raise ValueError("target_pg 与 metadata 不一致")
    if not np.array_equal(decoded["analyzer_pg"], metadata["Analyzer_PG"].astype(str)):
        raise ValueError("analyzer_pg 与 metadata 不一致")
    if not np.array_equal(decoded["actual_pg"], metadata["Actual_PG"].astype(str)):
        raise ValueError("actual_pg 与 metadata 不一致")
    if not np.array_equal(symmetry["pg_exact_match"], metadata["PG_Exact_Match"].to_numpy(bool)):
        raise ValueError("pg_exact_match 与 metadata 不一致")
    if not np.array_equal(symmetry["pg_compatible"], metadata["PG_Compatible"].to_numpy(bool)):
        raise ValueError("pg_compatible 与 metadata 不一致")

    matrix_tolerance = float(manifest["symmetry_protocol"]["matrix_tolerance"])
    _validate_operation_layer(
        "actual", canonical, symmetry, decoded["actual_pg"], matrix_tolerance
    )
    _validate_operation_layer(
        "target", canonical, symmetry, decoded["target_pg"], matrix_tolerance
    )
    actual_orbits = _orbit_distribution(
        metadata, canonical["atom_offsets"], symmetry["actual_orbit_id"], symmetry["actual_orbit_size"]
    )
    _orbit_distribution(
        metadata, canonical["atom_offsets"], symmetry["target_orbit_id"], symmetry["target_orbit_size"]
    )

    for filename in ("split_iid.json", "split_core_ood.json"):
        split = json.loads((package_dir / filename).read_text(encoding="utf-8"))["indices"]
        sets = [set(split[key]) for key in ("train", "val", "test")]
        if set.union(*sets) != set(range(molecule_count)) or any(
            sets[left] & sets[right] for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError(f"{filename} 未互斥完整覆盖数据集")
        if filename == "split_core_ood.json":
            cores = [set(metadata.iloc[list(indices)]["Core"]) for indices in sets]
            if any(cores[left] & cores[right] for left, right in ((0, 1), (0, 2), (1, 2))):
                raise ValueError("Core-OOD split 存在 Core 泄漏")

    element_symbols = [ATOM_SYMBOLS[int(index)] for index in canonical["atom_types"]]
    sn_rows = metadata[metadata["Contains_Sn"].astype(bool)]
    sn_records = []
    for _, row in sn_rows.iterrows():
        index = int(row["Package_Index"])
        start, end = map(int, canonical["atom_offsets"][index : index + 2])
        sn_records.append(
            {
                "package_index": index,
                "molecule_id": row["Molecule_ID"],
                "smiles": row["SMILES"],
                "sn_atoms": int(np.count_nonzero(canonical["atomic_numbers"][start:end] == 50)),
            }
        )
    pair_counts = metadata.groupby(["Target_PG", "Actual_PG", "PG_Compatible"]).size()
    similarity = pd.read_csv(package_dir / "similarity_audit.csv")
    similarity_summary = {
        scheme: _similarity_distribution(group["Max_Morgan_Tanimoto"].to_numpy())
        for scheme, group in similarity.groupby("Split_Scheme", sort=True)
    }
    split_statistics = {
        scheme: json.loads((package_dir / filename).read_text(encoding="utf-8"))["summary"]
        for scheme, filename in (("iid", "split_iid.json"), ("core_ood", "split_core_ood.json"))
    }
    hashes = {
        name: sha256_file(package_dir / name)
        for name in ("cof_graphs.npz", "symmetry_data.npz", "rdkit_features.npz", "manifest.json")
    }
    analyzer_consistency = metadata["Analyzer_PG"].eq(metadata["Legacy_Point_Group"])
    actual_consistency = metadata["Actual_PG"].eq(metadata["Legacy_Point_Group"])
    report: dict[str, Any] = {
        "schema_version": "2.0-validation-1.0",
        "status": "PASS",
        "counts": {"molecules": molecule_count, "atoms": atom_count, "bonds": bond_count},
        "point_group_recalculation": {
            "analyzer_vs_legacy_matches": int(analyzer_consistency.sum()),
            "analyzer_vs_legacy_rate": float(analyzer_consistency.mean()),
            "validated_actual_vs_legacy_matches": int(actual_consistency.sum()),
            "validated_actual_vs_legacy_rate": float(actual_consistency.mean()),
            "target_exact_matches": int(symmetry["pg_exact_match"].sum()),
            "target_exact_rate": float(symmetry["pg_exact_match"].mean()),
            "target_compatible": int(symmetry["pg_compatible"].sum()),
            "target_compatible_rate": float(symmetry["pg_compatible"].mean()),
        },
        "target_pg_distribution": _counts(metadata["Target_PG"]),
        "analyzer_pg_distribution": _counts(metadata["Analyzer_PG"]),
        "actual_pg_distribution": _counts(metadata["Actual_PG"]),
        "target_actual_compatible_distribution": {
            f"{target}|{actual}|compatible={bool(compatible)}": int(count)
            for (target, actual, compatible), count in pair_counts.items()
        },
        "actual_orbit_size_distribution_by_point_group": actual_orbits,
        "symmetry_error_angstrom": {
            "actual_operation_rms": _distribution(symmetry["actual_operation_rms_error"]),
            "actual_operation_max_atom": _distribution(symmetry["actual_operation_max_error"]),
            "target_operation_rms": _distribution(symmetry["target_operation_rms_error"]),
            "target_operation_max_atom": _distribution(symmetry["target_operation_max_error"]),
        },
        "elements": _counts(element_symbols),
        "sn": {
            "molecules": len(sn_rows),
            "atoms": int(np.count_nonzero(canonical["atomic_numbers"] == 50)),
            "records": sn_records,
            "uae_official_vocab_compatible": False,
            "fallback_mapping_used": False,
        },
        "splits": split_statistics,
        "core_ood_d6h": "low-support/exploratory; excluded from primary quantitative conclusions",
        "train_test_max_morgan_tanimoto": similarity_summary,
        "hashes": hashes,
    }
    if write_report:
        write_json(package_dir / "validation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        default=REPO_ROOT / "generative_model/data/processed/v2",
    )
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    report = validate_v2(args.package, write_report=not args.no_write_report)
    print(
        f"v2 验证通过：{report['counts']}; "
        f"analyzer/legacy={report['point_group_recalculation']['analyzer_vs_legacy_rate']:.4%}; "
        f"target-compatible={report['point_group_recalculation']['target_compatible_rate']:.4%}"
    )


if __name__ == "__main__":
    main()
