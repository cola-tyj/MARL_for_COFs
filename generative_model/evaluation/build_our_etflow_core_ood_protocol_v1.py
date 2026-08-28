"""Freeze a fresh C2/C3 Core-OOD paired evaluation panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT,
    _atomic_json,
    _fingerprint,
    _sha256,
)


PACKAGE = ROOT / "generative_model/data/processed/v2"
PARENT = ROOT / "generative_model/evaluation/our_etflow_ablation_protocol_v1.json"
OUTPUT = ROOT / "generative_model/evaluation/our_etflow_core_ood_protocol_v1.json"


def _canonical_and_fp(smiles: str, generator: Any) -> tuple[str, Any]:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    molecule = Chem.RemoveHs(molecule)
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False),
        generator.GetFingerprint(molecule),
    )


def _seed(package_index: int, target_pg: str) -> int:
    payload = f"our-etflow-core-ood-v1\0{package_index}\0{target_pg}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def build() -> dict[str, Any]:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    metadata = pd.read_csv(PACKAGE / "metadata.csv")
    excluded = {int(item["package_index"]) for item in parent["panel"]["records"]}
    for path in (
        ROOT / "generative_model/inference/etflow_e3f02_iidtest_protocol_v1.json",
        ROOT / "generative_model/inference/etflow_e3f02_rare_targets_protocol_v3.json",
    ):
        excluded.update(json.loads(path.read_text(encoding="utf-8"))["panel"]["package_indices"])
    train = metadata[metadata["Split_Core_OOD"] == "train"]
    train_cores = set(map(str, train["Core"]))
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    train_canonical, train_fps = [], []
    for smiles in train["SMILES"]:
        canonical, fingerprint = _canonical_and_fp(smiles, generator)
        train_canonical.append(canonical)
        train_fps.append(fingerprint)
    train_smiles = set(train_canonical)
    candidate_audits: dict[int, dict[str, Any]] = {}
    excluded_exact_fingerprint = 0
    selected: list[int] = []
    for target_pg in ("C2", "C3"):
        candidates = [
            int(index)
            for index, row in metadata.iterrows()
            if row["Split_Core_OOD"] == "test"
            and row["Target_PG"] == target_pg
            and int(index) not in excluded
        ]
        candidates.sort(key=lambda index: hashlib.sha256(
            f"our-etflow-core-ood-panel-v1\0{target_pg}\0"
            f"{metadata.iloc[index]['Molecule_ID']}".encode()
        ).hexdigest())
        eligible = []
        for package_index in candidates:
            row = metadata.iloc[package_index]
            canonical, fingerprint = _canonical_and_fp(row["SMILES"], generator)
            similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps)
            nearest_local = int(np.argmax(similarities))
            nearest_similarity = float(similarities[nearest_local])
            audit = {
                "canonical": canonical,
                "nearest_similarity": nearest_similarity,
                "nearest_package_index": int(train.index[nearest_local]),
            }
            candidate_audits[package_index] = audit
            if canonical in train_smiles:
                raise RuntimeError("Core-OOD split contains an exact train SMILES overlap")
            if str(row["Core"]) in train_cores:
                raise RuntimeError("Core-OOD split contains a train Core overlap")
            if nearest_similarity >= 1.0 - 1e-12:
                excluded_exact_fingerprint += 1
                continue
            eligible.append(package_index)
        if len(eligible) < 16:
            raise RuntimeError(f"fresh Core-OOD {target_pg} insufficient")
        selected.extend(eligible[:16])

    records = []
    for package_index in selected:
        row = metadata.iloc[package_index]
        candidate_audit = candidate_audits[package_index]
        nearest_similarity = float(candidate_audit["nearest_similarity"])
        nearest_package_index = int(candidate_audit["nearest_package_index"])
        target_pg = str(row["Target_PG"])
        records.append({
            "package_index": package_index,
            "molecule_id": str(row["Molecule_ID"]),
            "target_pg": target_pg,
            "seed": _seed(package_index, target_pg),
            "atom_count": int(row["Num_Atoms"]),
            "core": str(row["Core"]),
            "split_iid": str(row["Split_IID"]),
            "split_core_ood": str(row["Split_Core_OOD"]),
            "contains_sn": bool(row["Contains_Sn"]),
            "canonical_smiles_overlap_with_core_ood_train": False,
            "core_overlap_with_core_ood_train": False,
            "exact_morgan_fingerprint_overlap_with_core_ood_train": False,
            "nearest_core_ood_train_tanimoto": nearest_similarity,
            "nearest_core_ood_train_package_index": nearest_package_index,
        })
    similarities = [item["nearest_core_ood_train_tanimoto"] for item in records]
    protocol: dict[str, Any] = {
        **{key: value for key, value in parent.items() if key != "protocol_fingerprint"},
        "schema_version": "our-etflow-core-ood-paired-protocol-v1",
        "status": "FROZEN_BEFORE_CORE_OOD_COORDINATE_GENERATION",
        "purpose": (
            "Evaluate the same four routes on fresh Core-OOD test graphs with "
            "strict train-set SMILES/Core/exact-fingerprint nonoverlap."
        ),
        "panel": {
            "selection": (
                "SHA-256-ranked 16 C2 + 16 C3 Core-OOD test records after excluding "
                "all paired-ablation and frozen v5 completion panels"
            ),
            "molecule_count": len(records),
            "target_pg_counts": {"C2": 16, "C3": 16},
            "unique_core_count": len({item["core"] for item in records}),
            "excluded_prior_panel_count": len(excluded),
            "excluded_exact_morgan_fingerprint_overlap_count": (
                excluded_exact_fingerprint
            ),
            "nearest_train_tanimoto": {
                "minimum": float(np.min(similarities)),
                "mean": float(np.mean(similarities)),
                "median": float(np.median(similarities)),
                "maximum": float(np.max(similarities)),
            },
            "records": records,
        },
        "scope": {
            **parent["scope"],
            "target_pgs": ["C2", "C3"],
            "core_ood_test_only": True,
            "core_ood_train_used_for_nearest_neighbor_audit_only": True,
            "core_ood_train_coordinates_used": False,
            "unseen_graph_claim": True,
            "external_graph_claim": False,
        },
        "identity": {
            **parent["identity"],
            "parent_ablation_protocol_sha256": _sha256(PARENT),
            "core_ood_split_sha256": _sha256(PACKAGE / "split_core_ood.json"),
        },
    }
    protocol["protocol_fingerprint"] = _fingerprint(protocol)
    _atomic_json(OUTPUT, protocol)
    return protocol


def main() -> None:
    protocol = build()
    print(json.dumps({
        "status": protocol["status"],
        "output": str(OUTPUT),
        "panel": protocol["panel"]["target_pg_counts"],
        "unique_core_count": protocol["panel"]["unique_core_count"],
        "nearest_train_tanimoto": protocol["panel"]["nearest_train_tanimoto"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "sha256": _sha256(OUTPUT),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
