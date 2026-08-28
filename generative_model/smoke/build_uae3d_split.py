"""冻结 UAE-3D 首轮严格三头兼容 1/4/32 smoke split。"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from generative_model.data import COFSymmetryDataset
from generative_model.data.model_adapters import UAE_GEOM_VOCAB
from generative_model.data.schema import ATOM_SYMBOLS
from generative_model.models.uae3d_bridge import (
    OFFICIAL_COMMIT,
    is_uae3d_three_head_compatible,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT / "generative_model/data/processed/v2"
DEFAULT_OUTPUT = ROOT / "generative_model/smoke/splits/uae3d_v1.json"
UAE3D_REPOSITORY = "https://github.com/lyc0930/UAE-3D"
TARGET_QUOTAS = {"C2": 24, "C3": 6, "S4": 1, "D6h": 1}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_compatible(dataset: COFSymmetryDataset, package_index: int) -> bool:
    return is_uae3d_three_head_compatible(dataset[package_index])


def _spread_indices(dataset: COFSymmetryDataset, indices: list[int], count: int) -> list[int]:
    ordered = sorted(indices, key=lambda index: (int(dataset.metadata.iloc[index]["Num_Atoms"]), index))
    if len(ordered) < count:
        raise ValueError(f"兼容候选只有 {len(ordered)} 条，无法选择 {count} 条")
    if count == 1:
        return [ordered[len(ordered) // 2]]
    positions = [(rank * (len(ordered) - 1)) // (count - 1) for rank in range(count)]
    if len(set(positions)) != count:
        raise RuntimeError("spread selection 产生重复位置")
    return [ordered[position] for position in positions]


def build_uae3d_split(package_dir: Path = DEFAULT_PACKAGE) -> dict[str, Any]:
    dataset = COFSymmetryDataset(package_dir)
    split_path = package_dir / "split_iid.json"
    train_indices = list(map(
        int, json.loads(split_path.read_text(encoding="utf-8"))["indices"]["train"]
    ))
    compatible_all = [index for index in range(len(dataset)) if _is_compatible(dataset, index)]
    compatible_train = [index for index in train_indices if _is_compatible(dataset, index)]

    selected_by_pg: dict[str, list[int]] = {}
    for point_group, quota in TARGET_QUOTAS.items():
        candidates = [
            index for index in compatible_train
            if str(dataset.metadata.iloc[index]["Target_PG"]) == point_group
        ]
        selected_by_pg[point_group] = _spread_indices(dataset, candidates, quota)
    ordered_indices = [
        index for point_group in TARGET_QUOTAS for index in selected_by_pg[point_group]
    ]
    if len(ordered_indices) != 32 or len(set(ordered_indices)) != 32:
        raise RuntimeError("UAE-3D smoke split 必须包含 32 个唯一分子")
    tier_indices = {
        "1": [selected_by_pg["C2"][0]],
        "4": [selected_by_pg[point_group][0] for point_group in TARGET_QUOTAS],
        "32": ordered_indices,
    }
    if not set(tier_indices["1"]).issubset(tier_indices["4"]):
        raise RuntimeError("tier-1 必须是 tier-4 子集")
    if not set(tier_indices["4"]).issubset(tier_indices["32"]):
        raise RuntimeError("tier-4 必须是 tier-32 子集")

    atom_counts = [int(dataset.metadata.iloc[index]["Num_Atoms"]) for index in ordered_indices]
    elements: Counter[str] = Counter()
    for index in ordered_indices:
        start, end = map(int, dataset.arrays["atom_offsets"][index : index + 2])
        elements.update(ATOM_SYMBOLS[int(value)] for value in dataset.arrays["atom_types"][start:end])
    max_train_atoms = max(int(dataset.metadata.iloc[index]["Num_Atoms"]) for index in compatible_train)
    max_all_atoms = max(int(dataset.metadata.iloc[index]["Num_Atoms"]) for index in compatible_all)
    stress_train = min(
        index for index in compatible_train
        if int(dataset.metadata.iloc[index]["Num_Atoms"]) == max_train_atoms
    )
    stress_all = min(
        index for index in compatible_all
        if int(dataset.metadata.iloc[index]["Num_Atoms"]) == max_all_atoms
    )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "model_family": "UAE-3D",
        "official_repository": UAE3D_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
        "source_split": "iid/train",
        "source_split_sha256": _sha256_file(split_path),
        "selection_algorithm": (
            "filter official UAE GEOM atom vocabulary, formal_charges=0 and "
            "radical_electrons=0; fixed Target_PG quotas; within each PG sort by "
            "(Num_Atoms, Package_Index) and select evenly spaced ranks"
        ),
        "target_quotas": TARGET_QUOTAS,
        "compatible_policy": {
            "explicit_h": True,
            "allowed_tokens": list(UAE_GEOM_VOCAB),
            "formal_charges": 0,
            "radical_electrons": 0,
            "official_output_heads": ["atom_types", "bond_types", "coordinates"],
            "sn_policy": "excluded; never remap to C or Si",
        },
        "compatible_counts": {
            "full_dataset": len(compatible_all),
            "iid_train": len(compatible_train),
            "by_target_pg_full": dict(sorted(Counter(
                str(dataset.metadata.iloc[index]["Target_PG"]) for index in compatible_all
            ).items())),
            "by_target_pg_iid_train": dict(sorted(Counter(
                str(dataset.metadata.iloc[index]["Target_PG"]) for index in compatible_train
            ).items())),
        },
        "ordered_indices": ordered_indices,
        "indices_by_target_pg": selected_by_pg,
        "tier_indices": tier_indices,
        "stress_indices": {
            "max_iid_train": stress_train,
            "max_iid_train_atoms": max_train_atoms,
            "max_full_dataset": stress_all,
            "max_full_dataset_atoms": max_all_atoms,
        },
        "molecule_ids": [str(dataset.metadata.iloc[index]["Molecule_ID"]) for index in ordered_indices],
        "statistics": {
            "molecules": len(ordered_indices),
            "atom_count": {
                "min": min(atom_counts),
                "median": float(np.median(atom_counts)),
                "max": max(atom_counts),
            },
            "element_counts": dict(sorted(elements.items())),
            "target_pg_counts": dict(sorted(Counter(
                str(dataset.metadata.iloc[index]["Target_PG"]) for index in ordered_indices
            ).items())),
        },
    }
    report["split_fingerprint"] = _sha256_json(report)
    return report


def main() -> None:
    report = build_uae3d_split()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(
        f"wrote {len(report['ordered_indices'])} molecules to {DEFAULT_OUTPUT}; "
        f"compatible={report['compatible_counts']['full_dataset']}; "
        f"fingerprint={report['split_fingerprint']}"
    )


if __name__ == "__main__":
    main()
