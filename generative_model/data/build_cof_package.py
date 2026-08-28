"""从 final_dataset.csv 构建可复现的联合分子图 + 3D 坐标数据包。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem

from .schema import (
    ATOM_SYMBOLS,
    ATOM_TO_INDEX,
    BOND_TYPE_NAMES,
    BOND_TYPE_TO_INDEX,
    SCHEMA_VERSION,
    atomic_numbers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "cof_symmetry_pipeline/output/final_dataset.csv"
DEFAULT_OUTPUT = REPO_ROOT / "generative_model/data/processed/v1"
REQUIRED_COLUMNS = {"SMILES", "Core", "Arm", "Target_PG", "Point_Group", "XYZ_Path"}
SPLIT_SEED = 20260809


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """写出字节级可复现的压缩 NPZ（固定成员顺序和 ZIP 时间戳）。"""
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[name], allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue(), compresslevel=9)


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_xyz(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path.resolve()
    marker = "cof_symmetry_pipeline/"
    normalized = raw_path.replace("\\", "/")
    if marker in normalized:
        candidate = REPO_ROOT / marker / normalized.split(marker, 1)[1]
        if candidate.is_file():
            return candidate.resolve()
    candidate = REPO_ROOT / raw_path
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"XYZ 不存在: {raw_path}")


def parse_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ 文件过短: {path}")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"XYZ 首行不是原子数: {path}") from exc
    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(f"XYZ 原子行不足: {path}")
    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for line_number, line in enumerate(atom_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"XYZ 第 {line_number} 行格式错误: {path}")
        symbols.append(fields[0])
        try:
            coordinates.append([float(fields[1]), float(fields[2]), float(fields[3])])
        except ValueError as exc:
            raise ValueError(f"XYZ 第 {line_number} 行坐标错误: {path}") from exc
    positions = np.asarray(coordinates, dtype=np.float64)
    if not np.isfinite(positions).all():
        raise ValueError(f"XYZ 含 NaN/Inf: {path}")
    return symbols, positions


def make_iid_split(labels: Iterable[str], seed: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    for label in sorted(groups):
        indices = groups[label][:]
        rng.shuffle(indices)
        count = len(indices)
        n_val = max(1, round(0.1 * count))
        n_test = max(1, round(0.1 * count))
        if n_val + n_test >= count:
            raise ValueError(f"类别 {label} 样本太少，无法进行 IID 三划分")
        split["val"].extend(indices[:n_val])
        split["test"].extend(indices[n_val : n_val + n_test])
        split["train"].extend(indices[n_val + n_test :])
    for values in split.values():
        values.sort()
    return split


def closest_group_subset(
    groups: list[tuple[str, int]], target: int, excluded: set[str] | None = None
) -> set[str]:
    """用确定性的子集和选择最接近目标样本数的一组 Core。"""
    excluded = excluded or set()
    candidates = [(name, count) for name, count in groups if name not in excluded]
    states: dict[int, tuple[str, ...]] = {0: ()}
    for name, count in candidates:
        additions: dict[int, tuple[str, ...]] = {}
        for total, selected in sorted(states.items()):
            new_total = total + count
            proposal = selected + (name,)
            current = states.get(new_total, additions.get(new_total))
            if current is None or proposal < current:
                additions[new_total] = proposal
        states.update(additions)
    viable = [(total, selected) for total, selected in states.items() if selected]
    if not viable:
        return set()
    _, best = min(viable, key=lambda item: (abs(item[0] - target), item[0] > target, item[0], item[1]))
    return set(best)


def make_core_ood_split(frame: pd.DataFrame) -> tuple[dict[str, list[int]], list[str]]:
    core_target_counts = frame.groupby(["Target_PG", "Core"]).size()
    val_cores: set[str] = set()
    test_cores: set[str] = set()
    notes: list[str] = []
    for label in sorted(frame["Target_PG"].unique()):
        groups = sorted(
            [(core, int(count)) for (_, core), count in core_target_counts.items() if _ == label]
        )
        target = max(1, round(0.1 * int((frame["Target_PG"] == label).sum())))
        if len(groups) == 1:
            notes.append(f"{label} 只有 1 个 Core，全部保留在 train。")
            continue
        if len(groups) == 2:
            chosen = closest_group_subset(groups, target)
            test_cores.update(chosen)
            notes.append(f"{label} 只有 2 个 Core：一个用于 test，无法同时构造 Core-OOD val。")
            continue
        chosen_test = closest_group_subset(groups, target)
        chosen_val = closest_group_subset(groups, target, excluded=chosen_test)
        test_cores.update(chosen_test)
        val_cores.update(chosen_val)

    split = {"train": [], "val": [], "test": []}
    for index, core in enumerate(frame["Core"]):
        if core in test_cores:
            split["test"].append(index)
        elif core in val_cores:
            split["val"].append(index)
        else:
            split["train"].append(index)
    return split, notes


def validate_split(split: dict[str, list[int]], size: int, frame: pd.DataFrame, grouped: bool) -> None:
    sets = {name: set(values) for name, values in split.items()}
    if any(sets[a] & sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("数据划分存在重叠")
    if set.union(*sets.values()) != set(range(size)):
        raise ValueError("数据划分没有完整覆盖数据集")
    if grouped:
        core_sets = {name: set(frame.iloc[values]["Core"]) for name, values in split.items()}
        if any(core_sets[a] & core_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError("Core-OOD 划分发生 Core 泄漏")


def split_summary(split: dict[str, list[int]], frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, indices in split.items():
        part = frame.iloc[indices]
        result[name] = {
            "molecules": len(indices),
            "cores": int(part["Core"].nunique()),
            "target_pg": {key: int(value) for key, value in part["Target_PG"].value_counts().sort_index().items()},
        }
    return result


def _counter(values: Iterable[object]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


def build_package(source_csv: Path, output_dir: Path, seed: int = SPLIT_SEED) -> dict[str, object]:
    source_csv = source_csv.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source_csv)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {sorted(missing)}")
    if frame["SMILES"].isna().any() or frame["XYZ_Path"].isna().any():
        raise ValueError("SMILES/XYZ_Path 不允许为空")
    if frame["SMILES"].duplicated().any():
        raise ValueError("SMILES 必须唯一，避免同一分子跨 split 泄漏")
    if frame["XYZ_Path"].duplicated().any():
        raise ValueError("XYZ_Path 必须唯一")

    all_atom_types: list[np.ndarray] = []
    all_atomic_numbers: list[np.ndarray] = []
    all_charges: list[np.ndarray] = []
    all_radicals: list[np.ndarray] = []
    all_positions: list[np.ndarray] = []
    all_edges: list[np.ndarray] = []
    all_bond_types: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    atom_offsets = [0]
    bond_offsets = [0]
    metadata_rows: list[dict[str, object]] = []
    xyz_manifest: list[dict[str, object]] = []
    periodic_numbers = dict(zip(ATOM_SYMBOLS, atomic_numbers()))

    for package_index, (_, row) in enumerate(frame.iterrows()):
        xyz_path = resolve_xyz(str(row["XYZ_Path"]))
        xyz_symbols, raw_positions = parse_xyz(xyz_path)
        unknown = sorted(set(xyz_symbols) - set(ATOM_SYMBOLS))
        if unknown:
            raise ValueError(f"{xyz_path} 含未知元素 {unknown}；禁止回退映射")

        molecule = Chem.MolFromSmiles(str(row["SMILES"]))
        if molecule is None:
            raise ValueError(f"SMILES 无法解析（行 {package_index}）: {row['SMILES']}")
        molecule = Chem.AddHs(molecule)
        smiles_symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        if smiles_symbols != xyz_symbols:
            mismatch = next(
                (i for i, pair in enumerate(zip(smiles_symbols, xyz_symbols)) if pair[0] != pair[1]),
                min(len(smiles_symbols), len(xyz_symbols)),
            )
            raise ValueError(
                f"SMILES/XYZ 原子索引不一致（行 {package_index}, 首个差异 {mismatch}）: {xyz_path}"
            )

        atom_types = np.asarray([ATOM_TO_INDEX[symbol] for symbol in xyz_symbols], dtype=np.uint8)
        numbers = np.asarray([periodic_numbers[symbol] for symbol in xyz_symbols], dtype=np.uint8)
        charges = np.asarray([atom.GetFormalCharge() for atom in molecule.GetAtoms()], dtype=np.int8)
        radicals = np.asarray([atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()], dtype=np.uint8)
        centroid = raw_positions.mean(axis=0)
        positions = (raw_positions - centroid).astype(np.float32)
        if not np.allclose(positions.mean(axis=0), 0.0, atol=2e-5):
            raise ValueError(f"质心归零失败: {xyz_path}")

        edges: list[tuple[int, int]] = []
        types: list[int] = []
        for bond in molecule.GetBonds():
            begin, end = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            try:
                bond_type = BOND_TYPE_TO_INDEX[bond.GetBondType()]
            except KeyError as exc:
                raise ValueError(f"不支持的键类型 {bond.GetBondType()}: {row['SMILES']}") from exc
            edges.append((begin, end))
            types.append(bond_type)
        if len(edges) != len(set(edges)):
            raise ValueError(f"重复化学键（行 {package_index}）")
        order = np.argsort([begin * len(xyz_symbols) + end for begin, end in edges])
        edge_array = np.asarray(edges, dtype=np.int32).reshape(-1, 2)[order]
        type_array = np.asarray(types, dtype=np.uint8)[order]

        all_atom_types.append(atom_types)
        all_atomic_numbers.append(numbers)
        all_charges.append(charges)
        all_radicals.append(radicals)
        all_positions.append(positions)
        all_edges.append(edge_array)
        all_bond_types.append(type_array)
        centroids.append(centroid.astype(np.float32))
        atom_offsets.append(atom_offsets[-1] + len(atom_types))
        bond_offsets.append(bond_offsets[-1] + len(types))

        relative_path = repo_relative(xyz_path)
        metadata = {key: value for key, value in row.to_dict().items() if key != "XYZ_Path"}
        metadata.update(
            {
                "Package_Index": package_index,
                "Molecule_ID": f"cof_{package_index:06d}",
                "XYZ_RelPath": relative_path,
                "Num_Atoms": len(atom_types),
                "Num_Heavy_Atoms": int(np.count_nonzero(numbers > 1)),
                "Num_Bonds": len(types),
                "Net_Formal_Charge": int(charges.sum()),
                "Num_Radical_Electrons": int(radicals.sum()),
            }
        )
        metadata_rows.append(metadata)
        xyz_manifest.append(
            {"package_index": package_index, "path": relative_path, "sha256": sha256_file(xyz_path)}
        )

    arrays = {
        "atom_types": np.concatenate(all_atom_types),
        "atomic_numbers": np.concatenate(all_atomic_numbers),
        "formal_charges": np.concatenate(all_charges),
        "radical_electrons": np.concatenate(all_radicals),
        "positions": np.concatenate(all_positions),
        "centroids": np.stack(centroids),
        "bond_index": np.concatenate(all_edges),
        "bond_types": np.concatenate(all_bond_types),
        "atom_offsets": np.asarray(atom_offsets, dtype=np.int64),
        "bond_offsets": np.asarray(bond_offsets, dtype=np.int64),
    }
    save_deterministic_npz(output_dir / "cof_graphs.npz", arrays)

    metadata_frame = pd.DataFrame(metadata_rows)
    iid_split = make_iid_split(metadata_frame["Target_PG"], seed)
    core_split, core_notes = make_core_ood_split(metadata_frame)
    validate_split(iid_split, len(metadata_frame), metadata_frame, grouped=False)
    validate_split(core_split, len(metadata_frame), metadata_frame, grouped=True)
    metadata_frame["Split_IID"] = ""
    metadata_frame["Split_Core_OOD"] = ""
    for name, indices in iid_split.items():
        metadata_frame.loc[indices, "Split_IID"] = name
    for name, indices in core_split.items():
        metadata_frame.loc[indices, "Split_Core_OOD"] = name
    metadata_frame.to_csv(output_dir / "metadata.csv", index=False)

    split_common = {"schema_version": SCHEMA_VERSION, "seed": seed}
    write_json(
        output_dir / "split_iid.json",
        {**split_common, "method": "Target_PG-stratified 80/10/10", "indices": iid_split, "summary": split_summary(iid_split, metadata_frame)},
    )
    write_json(
        output_dir / "split_core_ood.json",
        {
            **split_common,
            "method": "Core-grouped, approximately 80/10/10 by molecule count within Target_PG",
            "indices": core_split,
            "summary": split_summary(core_split, metadata_frame),
            "notes": core_notes,
        },
    )
    write_json(
        output_dir / "vocab.json",
        {
            "schema_version": SCHEMA_VERSION,
            "atom_symbols": list(ATOM_SYMBOLS),
            "atom_to_index": ATOM_TO_INDEX,
            "atomic_numbers": {symbol: periodic_numbers[symbol] for symbol in ATOM_SYMBOLS},
            "bond_type_names": list(BOND_TYPE_NAMES),
            "bond_type_to_index": {name: index for index, name in enumerate(BOND_TYPE_NAMES)},
            "conventions": {"atom_index_base": 0, "undirected_bonds_stored_once": True, "positions": "angstrom, per-molecule centroid removed"},
        },
    )

    atom_counts = np.diff(arrays["atom_offsets"])
    heavy_counts = metadata_frame["Num_Heavy_Atoms"].to_numpy()
    radii = np.concatenate([np.linalg.norm(part, axis=1) for part in all_positions])
    statistics = {
        "schema_version": SCHEMA_VERSION,
        "molecules": len(metadata_frame),
        "atoms": int(len(arrays["atom_types"])),
        "bonds": int(len(arrays["bond_types"])),
        "elements": _counter(symbol for part in all_atom_types for symbol in (ATOM_SYMBOLS[i] for i in part)),
        "bond_types": _counter(BOND_TYPE_NAMES[int(index)] for index in arrays["bond_types"]),
        "formal_charges": _counter(arrays["formal_charges"]),
        "radical_electrons_per_atom": _counter(arrays["radical_electrons"]),
        "target_pg": _counter(metadata_frame["Target_PG"]),
        "point_group": _counter(metadata_frame["Point_Group"]),
        "atom_count": {"min": int(atom_counts.min()), "median": float(np.median(atom_counts)), "p95": float(np.percentile(atom_counts, 95)), "max": int(atom_counts.max())},
        "heavy_atom_count": {"min": int(heavy_counts.min()), "median": float(np.median(heavy_counts)), "p95": float(np.percentile(heavy_counts, 95)), "max": int(heavy_counts.max())},
        "distance_to_centroid_angstrom": {"mean": float(radii.mean()), "p95": float(np.percentile(radii, 95)), "max": float(radii.max())},
        "splits": {"iid": split_summary(iid_split, metadata_frame), "core_ood": split_summary(core_split, metadata_frame)},
    }
    write_json(output_dir / "statistics.json", statistics)

    (output_dir / "README.md").write_text(
        "# COF graph + 3D dataset v1\n\n"
        "该目录由 `python -m generative_model.data.build_cof_package` 自动生成。\n"
        "不要手工修改；字段、加载方法和重建命令见 `generative_model/README.md`。\n",
        encoding="utf-8",
    )

    package_files = ["cof_graphs.npz", "metadata.csv", "split_iid.json", "split_core_ood.json", "vocab.json", "statistics.json", "README.md"]
    source_hash = sha256_file(source_csv)
    fingerprint_payload = source_hash + "".join(item["sha256"] for item in xyz_manifest) + SCHEMA_VERSION
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "builder": repo_relative(Path(__file__)),
        "source_csv": {"path": repo_relative(source_csv), "sha256": source_hash},
        "source_xyz": xyz_manifest,
        "dataset_fingerprint": hashlib.sha256(fingerprint_payload.encode("ascii")).hexdigest(),
        "package_files": {name: sha256_file(output_dir / name) for name in package_files},
        "counts": {"molecules": len(metadata_frame), "atoms": int(len(arrays["atom_types"])), "bonds": int(len(arrays["bond_types"]))},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def verify_package(output_dir: Path, source_csv: Path | None = None) -> dict[str, object]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = source_csv.resolve() if source_csv else REPO_ROOT / manifest["source_csv"]["path"]
    if sha256_file(source) != manifest["source_csv"]["sha256"]:
        raise ValueError("源 CSV 的 SHA-256 与 manifest 不一致")
    for item in manifest["source_xyz"]:
        path = REPO_ROOT / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"源 XYZ 的 SHA-256 与 manifest 不一致: {item['path']}")
    for name, expected in manifest["package_files"].items():
        if sha256_file(output_dir / name) != expected:
            raise ValueError(f"数据包文件的 SHA-256 与 manifest 不一致: {name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_package(args.output, args.source)
        print(f"验证通过：{manifest['counts']}")
    else:
        manifest = build_package(args.source, args.output, args.seed)
        verify_package(args.output, args.source)
        print(f"构建并验证通过：{manifest['counts']}")
        print(f"dataset_fingerprint={manifest['dataset_fingerprint']}")


if __name__ == "__main__":
    main()
