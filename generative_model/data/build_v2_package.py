"""构建 v2：canonical 分子数据 + symmetry annotation + 模型适配审计。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

from .build_cof_package import (
    DEFAULT_SOURCE,
    REPO_ROOT,
    SPLIT_SEED,
    build_package,
    save_deterministic_npz,
    sha256_file,
    write_json,
)
from .schema import ATOM_SYMBOLS, BOND_TYPE_TO_INDEX, SCHEMA_VERSION as CANONICAL_SCHEMA_VERSION
from .symmetry import SymmetryProtocol, analyze_symmetry

V2_SCHEMA_VERSION = "2.0"
SYMMETRY_SCHEMA_VERSION = "1.0"
ELEMENT_VOCAB_VERSION = "cof-explicit-h-13-v1"
DEFAULT_OUTPUT = REPO_ROOT / "generative_model/data/processed/v2"

UAE_OFFICIAL_REPOSITORY = "https://github.com/lyc0930/UAE-3D"
UAE_OFFICIAL_COMMIT = "4327dd5d8233c74e0f045be481f24d0f1d4c0c09"
UAE_GEOM_VOCAB = (
    "H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi"
)
UAE_EXTENDED_VOCAB = (*UAE_GEOM_VOCAB, "Sn")


def _version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _analyze_worker(payload: tuple[int, list[str], np.ndarray, str, dict[str, Any]]) -> dict[str, Any]:
    molecule_index, symbols, positions, target_pg, protocol_data = payload
    try:
        return analyze_symmetry(symbols, positions, target_pg, SymmetryProtocol(**protocol_data))
    except Exception as exc:
        raise ValueError(f"molecule index {molecule_index}: {exc}") from exc


def _flatten_operation_set(
    results: list[dict[str, Any]], key: str
) -> dict[str, np.ndarray]:
    matrices: list[np.ndarray] = []
    permutations: list[np.ndarray] = []
    rms_errors: list[np.ndarray] = []
    max_errors: list[np.ndarray] = []
    operation_offsets = [0]
    permutation_offsets = [0]
    for result in results:
        block = result[key]
        for matrix, permutation, rms, maximum in zip(
            block["matrices"], block["permutations"], block["rms_errors"], block["max_errors"]
        ):
            matrices.append(np.asarray(matrix, dtype=np.float32))
            permutation_array = np.asarray(permutation, dtype=np.int32)
            permutations.append(permutation_array)
            rms_errors.append(np.asarray(rms, dtype=np.float32))
            max_errors.append(np.asarray(maximum, dtype=np.float32))
            permutation_offsets.append(permutation_offsets[-1] + len(permutation_array))
        operation_offsets.append(operation_offsets[-1] + len(block["matrices"]))
    return {
        f"{key}_operation_matrices": np.stack(matrices),
        f"{key}_permutation_index": np.concatenate(permutations),
        f"{key}_operation_rms_error": np.asarray(rms_errors, dtype=np.float32),
        f"{key}_operation_max_error": np.asarray(max_errors, dtype=np.float32),
        f"{key}_operation_offsets": np.asarray(operation_offsets, dtype=np.int64),
        f"{key}_permutation_offsets": np.asarray(permutation_offsets, dtype=np.int64),
    }


def _build_symmetry_arrays(
    results: list[dict[str, Any]], atom_offsets: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    point_groups = sorted(
        {result[field] for result in results for field in ("target_pg", "analyzer_pg", "actual_pg")}
    )
    pg_to_index = {point_group: index for index, point_group in enumerate(point_groups)}
    arrays: dict[str, np.ndarray] = {
        "atom_offsets": atom_offsets.astype(np.int64, copy=True),
        "target_pg": np.asarray([pg_to_index[result["target_pg"]] for result in results], dtype=np.uint8),
        "analyzer_pg": np.asarray([pg_to_index[result["analyzer_pg"]] for result in results], dtype=np.uint8),
        "actual_pg": np.asarray([pg_to_index[result["actual_pg"]] for result in results], dtype=np.uint8),
        "pg_exact_match": np.asarray([result["pg_exact_match"] for result in results], dtype=np.bool_),
        "pg_compatible": np.asarray([result["pg_compatible"] for result in results], dtype=np.bool_),
    }
    for key in ("actual", "target"):
        arrays.update(_flatten_operation_set(results, key))
        arrays[f"{key}_orbit_id"] = np.concatenate([result[key]["orbit_id"] for result in results])
        arrays[f"{key}_orbit_size"] = np.concatenate([result[key]["orbit_size"] for result in results])
        arrays[f"{key}_mean_rms_error"] = np.asarray(
            [result[key]["mean_rms_error"] for result in results], dtype=np.float32
        )
        arrays[f"{key}_max_rms_error"] = np.asarray(
            [result[key]["max_rms_error"] for result in results], dtype=np.float32
        )
        arrays[f"{key}_max_atom_error"] = np.asarray(
            [result[key]["max_atom_error"] for result in results], dtype=np.float32
        )
    vocab = {
        "schema_version": SYMMETRY_SCHEMA_VERSION,
        "point_groups": point_groups,
        "point_group_to_index": pg_to_index,
        "permutation_convention": "permutation_index[source_atom] = target_atom, local 0-based indices",
        "operation_equation": "positions @ R.T ~= positions[permutation_index]",
        "orbit_convention": "local contiguous orbit ids ordered by smallest atom index",
    }
    return arrays, vocab


def _derive_rdkit_features(
    metadata: pd.DataFrame,
    canonical: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[Chem.Mol]]:
    atom_records: list[dict[str, Any]] = []
    bond_records: list[dict[str, Any]] = []
    molecules: list[Chem.Mol] = []
    for molecule_index, row in metadata.iterrows():
        molecule = Chem.MolFromSmiles(row["SMILES"])
        if molecule is None:
            raise ValueError(f"RDKit 无法解析 SMILES（{molecule_index}）")
        molecule = Chem.AddHs(molecule)
        atom_start, atom_end = canonical["atom_offsets"][molecule_index : molecule_index + 2]
        atom_start, atom_end = int(atom_start), int(atom_end)
        symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        canonical_symbols = [ATOM_SYMBOLS[int(index)] for index in canonical["atom_types"][atom_start:atom_end]]
        if symbols != canonical_symbols:
            raise ValueError(f"RDKit feature 原子顺序与 canonical 不一致（{molecule_index}）")
        for atom in molecule.GetAtoms():
            atom_records.append(
                {
                    "hybridization": str(atom.GetHybridization()),
                    "aromatic": atom.GetIsAromatic(),
                    "chirality": str(atom.GetChiralTag()),
                    "degree": atom.GetDegree(),
                    "h_count": sum(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()),
                    "in_ring": atom.IsInRing(),
                }
            )
        local_bonds: list[tuple[int, int, Any]] = []
        for bond in molecule.GetBonds():
            begin, end = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            local_bonds.append((begin, end, bond))
        local_bonds.sort(key=lambda item: (item[0], item[1]))
        bond_start, bond_end = canonical["bond_offsets"][molecule_index : molecule_index + 2]
        expected_edges = canonical["bond_index"][int(bond_start) : int(bond_end)]
        if not np.array_equal(np.asarray([(a, b) for a, b, _ in local_bonds]), expected_edges):
            raise ValueError(f"RDKit feature 键顺序与 canonical 不一致（{molecule_index}）")
        for _, _, bond in local_bonds:
            expected_type = BOND_TYPE_TO_INDEX.get(bond.GetBondType())
            if expected_type is None:
                raise ValueError(f"不支持的 RDKit 键类型（{molecule_index}）")
            bond_records.append(
                {
                    "stereo": str(bond.GetStereo()),
                    "conjugated": bond.GetIsConjugated(),
                    "in_ring": bond.IsInRing(),
                }
            )
        molecules.append(molecule)

    hybridization_vocab = sorted({record["hybridization"] for record in atom_records})
    chirality_vocab = sorted({record["chirality"] for record in atom_records})
    stereo_vocab = sorted({record["stereo"] for record in bond_records})
    hybridization_index = {value: index for index, value in enumerate(hybridization_vocab)}
    chirality_index = {value: index for index, value in enumerate(chirality_vocab)}
    stereo_index = {value: index for index, value in enumerate(stereo_vocab)}
    arrays = {
        "atom_offsets": canonical["atom_offsets"].astype(np.int64, copy=True),
        "bond_offsets": canonical["bond_offsets"].astype(np.int64, copy=True),
        "hybridization": np.asarray([hybridization_index[r["hybridization"]] for r in atom_records], dtype=np.uint8),
        "aromaticity": np.asarray([r["aromatic"] for r in atom_records], dtype=np.bool_),
        "chirality": np.asarray([chirality_index[r["chirality"]] for r in atom_records], dtype=np.uint8),
        "degree": np.asarray([r["degree"] for r in atom_records], dtype=np.uint8),
        "explicit_h_neighbor_count": np.asarray([r["h_count"] for r in atom_records], dtype=np.uint8),
        "ring_membership": np.asarray([r["in_ring"] for r in atom_records], dtype=np.bool_),
        "bond_stereo": np.asarray([stereo_index[r["stereo"]] for r in bond_records], dtype=np.uint8),
        "bond_conjugation": np.asarray([r["conjugated"] for r in bond_records], dtype=np.bool_),
        "bond_ring_membership": np.asarray([r["in_ring"] for r in bond_records], dtype=np.bool_),
    }
    vocab = {
        "hybridization": hybridization_vocab,
        "chirality": chirality_vocab,
        "bond_stereo": stereo_vocab,
        "ground_truth_boundary": "all arrays here are deterministic RDKit derivatives; atomic number, formal charge, sparse bond graph, and XYZ remain canonical",
    }
    return arrays, vocab, molecules


def _similarity_audit(
    metadata: pd.DataFrame, molecules: list[Chem.Mol], output_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    fingerprints = [generator.GetFingerprint(Chem.RemoveHs(molecule)) for molecule in molecules]
    records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for scheme, filename in (("iid", "split_iid.json"), ("core_ood", "split_core_ood.json")):
        split = json.loads((output_dir / filename).read_text(encoding="utf-8"))["indices"]
        train_indices = list(map(int, split["train"]))
        similarities_for_scheme: list[float] = []
        for test_index in map(int, split["test"]):
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprints[test_index], [fingerprints[index] for index in train_indices]
            )
            best_position = int(np.argmax(similarities))
            nearest_index = train_indices[best_position]
            similarity = float(similarities[best_position])
            similarities_for_scheme.append(similarity)
            records.append(
                {
                    "Split_Scheme": scheme,
                    "Test_Package_Index": test_index,
                    "Test_Molecule_ID": metadata.iloc[test_index]["Molecule_ID"],
                    "Nearest_Train_Package_Index": nearest_index,
                    "Nearest_Train_Molecule_ID": metadata.iloc[nearest_index]["Molecule_ID"],
                    "Max_Morgan_Tanimoto": similarity,
                }
            )
        values = np.asarray(similarities_for_scheme)
        summaries[scheme] = {
            "count": len(values),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
            "min": float(values.min()),
            "max": float(values.max()),
            "equal_1_count": int(np.count_nonzero(np.isclose(values, 1.0, atol=1e-12))),
            "ge_0_9_count": int(np.count_nonzero(values >= 0.9)),
            "ge_0_8_count": int(np.count_nonzero(values >= 0.8)),
        }
    return pd.DataFrame(records), summaries


def _dtype_manifest(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: str(array.dtype) for name, array in sorted(arrays.items())}


def build_v2(
    source_csv: Path,
    output_dir: Path,
    workers: int = 12,
    protocol: SymmetryProtocol | None = None,
) -> dict[str, Any]:
    protocol = protocol or SymmetryProtocol()
    # 直接从唯一源索引重建 canonical v1 schema，避免依赖可变的中间数据包。
    base_manifest = build_package(source_csv, output_dir, SPLIT_SEED)
    metadata = pd.read_csv(output_dir / "metadata.csv")
    with np.load(output_dir / "cof_graphs.npz", allow_pickle=False) as archive:
        canonical = {name: archive[name] for name in archive.files}

    payloads: list[tuple[int, list[str], np.ndarray, str, dict[str, Any]]] = []
    for index, row in metadata.iterrows():
        atom_start, atom_end = canonical["atom_offsets"][index : index + 2]
        atom_start, atom_end = int(atom_start), int(atom_end)
        symbols = [ATOM_SYMBOLS[int(value)] for value in canonical["atom_types"][atom_start:atom_end]]
        positions = canonical["positions"][atom_start:atom_end].astype(np.float64)
        payloads.append((index, symbols, positions, str(row["Target_PG"]), asdict(protocol)))
    if workers == 1:
        symmetry_results = [_analyze_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            symmetry_results = list(executor.map(_analyze_worker, payloads, chunksize=8))

    symmetry_arrays, symmetry_vocab = _build_symmetry_arrays(
        symmetry_results, canonical["atom_offsets"]
    )
    save_deterministic_npz(output_dir / "symmetry_data.npz", symmetry_arrays)
    write_json(output_dir / "symmetry_vocab.json", symmetry_vocab)

    feature_arrays, feature_vocab, molecules = _derive_rdkit_features(metadata, canonical)
    canonical_smiles = [Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True) for molecule in molecules]
    if len(set(canonical_smiles)) != len(canonical_smiles):
        raise ValueError(
            "当前 v2 声明 one_conformer_per_smiles=true，但 RDKit canonical SMILES 存在重复"
        )
    save_deterministic_npz(output_dir / "rdkit_features.npz", feature_arrays)
    write_json(output_dir / "feature_vocab.json", feature_vocab)

    for index, result in enumerate(symmetry_results):
        metadata.loc[index, "Legacy_Point_Group"] = metadata.loc[index, "Point_Group"]
        metadata.loc[index, "Analyzer_PG"] = result["analyzer_pg"]
        metadata.loc[index, "Actual_PG"] = result["actual_pg"]
        metadata.loc[index, "PG_Exact_Match"] = result["pg_exact_match"]
        metadata.loc[index, "PG_Compatible"] = result["pg_compatible"]
        metadata.loc[index, "Actual_Mean_RMS_Error"] = result["actual"]["mean_rms_error"]
        metadata.loc[index, "Actual_Max_RMS_Error"] = result["actual"]["max_rms_error"]
        metadata.loc[index, "Actual_Max_Atom_Error"] = result["actual"]["max_atom_error"]
        metadata.loc[index, "Target_Mean_RMS_Error"] = result["target"]["mean_rms_error"]
        metadata.loc[index, "Target_Max_RMS_Error"] = result["target"]["max_rms_error"]
        metadata.loc[index, "Target_Max_Atom_Error"] = result["target"]["max_atom_error"]

    sn_mask = np.asarray([any(atom.GetSymbol() == "Sn" for atom in molecule.GetAtoms()) for molecule in molecules])
    uae_compatible = ~sn_mask
    metadata["Contains_Sn"] = sn_mask
    metadata["UAE_GEOM_Pretrained_Compatible"] = uae_compatible
    metadata["Core_OOD_Support"] = np.where(metadata["Target_PG"].eq("D6h"), "low-support/exploratory", "standard")
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    compatible_indices = np.flatnonzero(uae_compatible).astype(int).tolist()
    sn_indices = np.flatnonzero(sn_mask).astype(int).tolist()
    write_json(
        output_dir / "uae_compatibility.json",
        {
            "official_repository": UAE_OFFICIAL_REPOSITORY,
            "official_commit_audited": UAE_OFFICIAL_COMMIT,
            "official_geom_vocabulary": list(UAE_GEOM_VOCAB),
            "sn_supported_by_official_vocabulary": False,
            "compatible_subset": {"indices": compatible_indices, "molecules": len(compatible_indices)},
            "excluded_sn_indices": sn_indices,
            "extended_vocabulary": list(UAE_EXTENDED_VOCAB),
            "extended_sn_index": UAE_EXTENDED_VOCAB.index("Sn"),
            "policy": "Sn is never mapped to C or Si; use compatible_subset or expand pretrained atom embedding/head",
        },
    )

    similarity_frame, similarity_summary = _similarity_audit(metadata, molecules, output_dir)
    similarity_frame.to_csv(output_dir / "similarity_audit.csv", index=False)

    readme = (
        "# COF graph + 3D + symmetry dataset v2\n\n"
        "`cof_graphs.npz` 保持模型无关的 canonical molecular dataset；"
        "`symmetry_data.npz` 通过 package/molecule index 一一对应。\n\n"
        "重新构建与验证：\n\n"
        "```bash\n"
        "python -m generative_model.data.build_v2_package\n"
        "python -m generative_model.data.validate_v2 --package generative_model/data/processed/v2\n"
        "```\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    package_files = [
        "README.md", "cof_graphs.npz", "metadata.csv", "vocab.json", "statistics.json",
        "split_iid.json", "split_core_ood.json", "symmetry_data.npz", "symmetry_vocab.json",
        "rdkit_features.npz", "feature_vocab.json", "uae_compatibility.json", "similarity_audit.csv",
    ]
    builder_files = [
        Path(__file__), Path(__file__).with_name("symmetry.py"),
        Path(__file__).with_name("build_cof_package.py"), Path(__file__).with_name("schema.py"),
        Path(__file__).with_name("validate_v2.py"),
    ]
    fingerprint_payload = json.dumps(
        {
            "base": base_manifest["dataset_fingerprint"],
            "schema": V2_SCHEMA_VERSION,
            "protocol": asdict(protocol),
            "element_vocab": ELEMENT_VOCAB_VERSION,
            "uae_commit": UAE_OFFICIAL_COMMIT,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest: dict[str, Any] = {
        "schema_version": V2_SCHEMA_VERSION,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "symmetry_schema_version": SYMMETRY_SCHEMA_VERSION,
        "element_vocabulary_version": ELEMENT_VOCAB_VERSION,
        "one_conformer_per_smiles": True,
        "smiles_uniqueness": "2,532/2,532 unique after RDKit canonical isomeric SMILES; no conformer was deleted",
        "coordinate": {"unit": "angstrom", "centering": "arithmetic centroid per molecule", "dtype": "float32"},
        "canonical_ground_truth": ["atomic_numbers", "formal_charges", "bond_index", "bond_types", "positions"],
        "builder_git_commit": _git_commit(),
        "builder_files": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in builder_files},
        "environment": {
            "python": platform.python_version(), "numpy": _version("numpy"), "pandas": _version("pandas"),
            "rdkit": _version("rdkit"), "scipy": _version("scipy"), "pymatgen": _version("pymatgen"),
        },
        "source_csv": base_manifest["source_csv"],
        "source_xyz": base_manifest["source_xyz"],
        "base_dataset_fingerprint": base_manifest["dataset_fingerprint"],
        "dataset_fingerprint": hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest(),
        "counts": base_manifest["counts"],
        "dtypes": {"cof_graphs.npz": _dtype_manifest(canonical), "symmetry_data.npz": _dtype_manifest(symmetry_arrays), "rdkit_features.npz": _dtype_manifest(feature_arrays)},
        "split": {
            "seed": SPLIT_SEED,
            "iid": "Target_PG-stratified 80/10/10",
            "core_ood": "Core-grouped approximate 80/10/10 within Target_PG",
            "d6h": "low-support/exploratory; not for primary quantitative conclusions",
        },
        "symmetry_protocol": asdict(protocol),
        "symmetry_label_policy": "actual_pg is the largest complete subgroup of pymatgen candidate operations whose operation RMS errors pass tolerance; analyzer_pg preserves raw pymatgen label",
        "morgan_similarity": {"radius": 2, "fp_size": 2048, "include_chirality": True, "use_explicit_h": False, "summary": similarity_summary},
        "uae_3d": {"repository": UAE_OFFICIAL_REPOSITORY, "audited_commit": UAE_OFFICIAL_COMMIT, "official_geom_vocabulary": list(UAE_GEOM_VOCAB), "sn_policy": "compatible subset or append Sn at index 16; never remap"},
        "package_files": {name: sha256_file(output_dir / name) for name in package_files},
        "validation_report": "validation_report.json (deterministic audit output; contains manifest SHA-256 and is therefore not self-hashed by manifest)",
    }
    write_json(output_dir / "manifest.json", manifest)

    from .validate_v2 import validate_v2

    validate_v2(output_dir, write_report=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=0.3)
    parser.add_argument("--eigen-tolerance", type=float, default=0.01)
    parser.add_argument("--matrix-tolerance", type=float, default=0.1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers 必须 >= 1")
    protocol = SymmetryProtocol(
        tolerance_angstrom=args.tolerance,
        eigen_tolerance=args.eigen_tolerance,
        matrix_tolerance=args.matrix_tolerance,
    )
    manifest = build_v2(args.source, args.output, args.workers, protocol)
    print(f"v2 构建与验证通过：{manifest['counts']}")
    print(f"dataset_fingerprint={manifest['dataset_fingerprint']}")


if __name__ == "__main__":
    main()
