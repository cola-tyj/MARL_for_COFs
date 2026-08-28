"""Build and freeze a small external-to-COF symmetric graph package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from generative_model.conformer.etflow_bridge import build_mapped_explicit_h_smiles
from generative_model.data.build_cof_package import save_deterministic_npz
from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT,
    _atomic_json,
    _fingerprint,
    _sha256,
)
from generative_model.inference.generate_etflow_symmetric_xyz_v4 import (
    action_samples_v4,
)
from generative_model.optimization.force_field_support import (
    build_strict_rdkit_molecule,
    force_field_support,
)


CANDIDATES = ROOT / "generative_model/evaluation/external_symmetric_candidates_v1.json"
CANONICAL = ROOT / "generative_model/data/processed/v2"
OUTPUT = ROOT / "generative_model/data/external_symmetric_v1"
PROTOCOL_OUTPUT = ROOT / "generative_model/evaluation/our_etflow_external_protocol_v1.json"
PARENT = ROOT / "generative_model/evaluation/our_etflow_ablation_protocol_v1.json"
V5_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_protocol_v5.json"
BOND_IDS = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}
QUOTAS = {"C2": 4, "C3": 4, "S4": 2, "D6h": 2}


def _canonical_molecule(smiles: str) -> tuple[Chem.Mol, str]:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("RDKit parse failed")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    reparsed = Chem.MolFromSmiles(canonical)
    if reparsed is None:
        raise RuntimeError("canonical SMILES reparse failed")
    return reparsed, canonical


def _scaffold(molecule: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        return "<ACYCLIC>"
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def _graph(molecule: Chem.Mol, external_index: int, name: str) -> dict[str, Any]:
    explicit = Chem.AddHs(Chem.Mol(molecule))
    numbers = np.asarray([atom.GetAtomicNum() for atom in explicit.GetAtoms()], dtype=np.int64)
    charges = np.asarray([atom.GetFormalCharge() for atom in explicit.GetAtoms()], dtype=np.int64)
    radicals = np.asarray(
        [atom.GetNumRadicalElectrons() for atom in explicit.GetAtoms()], dtype=np.int64
    )
    edges, kinds = [], []
    for bond in explicit.GetBonds():
        kind = BOND_IDS.get(bond.GetBondType())
        if kind is None:
            raise ValueError(f"unsupported bond type: {bond.GetBondType()}")
        left, right = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edges.append((left, right))
        kinds.append(kind)
    order = np.argsort([left * len(numbers) + right for left, right in edges])
    bond_index = np.asarray(edges, dtype=np.int64)[order].T
    bond_types = np.asarray(kinds, dtype=np.int64)[order]
    return {
        "package_index": external_index,
        "molecule_id": f"external_{external_index:04d}_{name}",
        "atomic_numbers": numbers,
        "formal_charges": charges,
        "radical_electrons": radicals,
        "bond_index": bond_index,
        "bond_types": bond_types,
    }


def _reference_positions(molecule: Chem.Mol, seed: int) -> np.ndarray:
    working = Chem.AddHs(Chem.Mol(molecule))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed)
    parameters.maxIterations = 1000
    parameters.enforceChirality = True
    code = int(AllChem.EmbedMolecule(working, parameters))
    if code != 0:
        raise RuntimeError(f"evaluation-reference ETKDG failed: {code}")
    if not AllChem.UFFHasAllMoleculeParams(working):
        raise RuntimeError("evaluation-reference UFF parameters incomplete")
    status = int(AllChem.UFFOptimizeMolecule(working, maxIters=500))
    if status not in (0, 1):
        raise RuntimeError(f"evaluation-reference UFF failed: {status}")
    values = np.asarray(working.GetConformer().GetPositions(), dtype=np.float64)
    return values - values.mean(axis=0, keepdims=True)


def _seed(name: str, purpose: str) -> int:
    digest = hashlib.sha256(f"external-v1\0{name}\0{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def build() -> dict[str, Any]:
    pool = json.loads(CANDIDATES.read_text(encoding="utf-8"))["candidates"]
    canonical_metadata = pd.read_csv(CANONICAL / "metadata.csv")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    existing_smiles, existing_scaffolds, existing_fps = set(), set(), []
    for smiles in canonical_metadata["SMILES"]:
        molecule, canonical = _canonical_molecule(smiles)
        existing_smiles.add(canonical)
        existing_scaffolds.add(_scaffold(molecule))
        existing_fps.append(generator.GetFingerprint(molecule))
    vocab = json.loads((CANONICAL / "vocab.json").read_text(encoding="utf-8"))
    allowed_numbers = set(map(int, vocab["atomic_numbers"].values()))
    v5 = json.loads(V5_PROTOCOL.read_text(encoding="utf-8"))
    accepted, rejected, seen_external = [], [], set()
    for candidate in pool:
        name, target_pg = str(candidate["name"]), str(candidate["target_pg"])
        try:
            molecule, canonical = _canonical_molecule(candidate["smiles"])
            scaffold = _scaffold(molecule)
            fingerprint = generator.GetFingerprint(molecule)
            nearest = float(max(DataStructs.BulkTanimotoSimilarity(
                fingerprint, existing_fps
            )))
            checks = {
                "canonical_smiles_not_in_cof_v2": canonical not in existing_smiles,
                "murcko_scaffold_not_in_cof_v2": scaffold not in existing_scaffolds,
                "exact_morgan_fingerprint_not_in_cof_v2": nearest < 1.0 - 1e-12,
                "unique_within_external_pool": canonical not in seen_external,
            }
            if not all(checks.values()):
                raise ValueError(f"nonoverlap checks failed: {checks}")
            graph = _graph(molecule, len(accepted), name)
            if not set(map(int, graph["atomic_numbers"])).issubset(allowed_numbers):
                raise ValueError("element outside canonical 13-element vocabulary")
            reference = _reference_positions(molecule, _seed(name, "reference"))
            aromaticity = np.zeros(len(graph["atomic_numbers"]), dtype=np.bool_)
            for (left, right), kind in zip(
                graph["bond_index"].T, graph["bond_types"], strict=True
            ):
                if int(kind) == 4:
                    aromaticity[int(left)] = aromaticity[int(right)] = True
            strict_sample = {**graph, "positions": reference, "aromaticity": aromaticity}
            support = force_field_support(build_strict_rdkit_molecule(strict_sample))
            if not support["uff_setup_success"]:
                raise ValueError("strict UFF setup unavailable")
            bridge_smiles = build_mapped_explicit_h_smiles(strict_sample)
            actions = action_samples_v4(graph, target_pg, v5)
            seen_external.add(canonical)
            accepted.append({
                **graph,
                "name": name,
                "input_smiles": str(candidate["smiles"]),
                "canonical_smiles": canonical,
                "murcko_scaffold_smiles": scaffold,
                "target_pg": target_pg,
                "reference_positions": reference,
                "nearest_cof_v2_tanimoto": nearest,
                "graph_action_candidate_count": len(actions),
                "bridge_smiles_sha256": hashlib.sha256(bridge_smiles.encode()).hexdigest(),
                "nonoverlap_checks": checks,
            })
        except Exception as error:
            rejected.append({
                **candidate,
                "failure": f"{type(error).__name__}: {error}",
            })
    selected = []
    for target_pg, count in QUOTAS.items():
        candidates = [item for item in accepted if item["target_pg"] == target_pg]
        candidates.sort(key=lambda item: hashlib.sha256(
            f"external-select-v1\0{target_pg}\0{item['name']}".encode()
        ).hexdigest())
        if len(candidates) < count:
            raise RuntimeError(
                f"external eligible {target_pg} insufficient: {len(candidates)} < {count}; "
                f"rejected={rejected}"
            )
        selected.extend(candidates[:count])
    # Reindex only after deterministic quota selection.
    for index, item in enumerate(selected):
        item["package_index"] = index
        item["molecule_id"] = f"external_{index:04d}_{item['name']}"

    atom_offsets, bond_offsets = [0], [0]
    atomic_numbers, atom_types, charges, radicals, positions = [], [], [], [], []
    bond_indices, bond_types = [], []
    rows = []
    for item in selected:
        numbers = item["atomic_numbers"]
        symbols = [Chem.GetPeriodicTable().GetElementSymbol(int(value)) for value in numbers]
        atomic_numbers.append(numbers)
        atom_types.append(np.asarray([vocab["atom_to_index"][symbol] for symbol in symbols]))
        charges.append(item["formal_charges"])
        radicals.append(item["radical_electrons"])
        positions.append(item["reference_positions"])
        bond_indices.append(item["bond_index"].T)
        bond_types.append(item["bond_types"])
        atom_offsets.append(atom_offsets[-1] + len(numbers))
        bond_offsets.append(bond_offsets[-1] + len(item["bond_types"]))
        rows.append({
            "Molecule_ID": item["molecule_id"],
            "SMILES": item["canonical_smiles"],
            "Core": item["murcko_scaffold_smiles"],
            "Target_PG": item["target_pg"],
            "Num_Atoms": len(numbers),
            "Contains_Sn": bool(np.any(numbers == 50)),
            "Split_IID": "external",
            "Split_Core_OOD": "external",
            "Nearest_COF_v2_Tanimoto": item["nearest_cof_v2_tanimoto"],
            "Reference_Coordinates_Role": "evaluation-only ETKDGv3+UFF; never generator input",
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_deterministic_npz(OUTPUT / "cof_graphs.npz", {
        "atom_offsets": np.asarray(atom_offsets, dtype="<i8"),
        "bond_offsets": np.asarray(bond_offsets, dtype="<i8"),
        "atomic_numbers": np.concatenate(atomic_numbers).astype("<i8"),
        "atom_types": np.concatenate(atom_types).astype("<i8"),
        "formal_charges": np.concatenate(charges).astype("<i8"),
        "radical_electrons": np.concatenate(radicals).astype("<i8"),
        "positions": np.concatenate(positions).astype("<f8"),
        "centroids": np.zeros((len(selected), 3), dtype="<f8"),
        "bond_index": np.concatenate(bond_indices).astype("<i8"),
        "bond_types": np.concatenate(bond_types).astype("<i8"),
    })
    pd.DataFrame(rows).to_csv(OUTPUT / "metadata.csv", index=False, lineterminator="\n")
    _atomic_json(OUTPUT / "vocab.json", vocab)
    _atomic_json(OUTPUT / "statistics.json", {
        "molecule_count": len(selected),
        "target_pg_counts": QUOTAS,
        "role": "external-to-COF evaluation only",
    })
    candidate_audit = OUTPUT / "candidate_audit.json"
    _atomic_json(candidate_audit, {
        "accepted_before_quota": [item["name"] for item in accepted],
        "selected": [item["name"] for item in selected],
        "rejected": rejected,
    })
    manifest = {
        "schema_version": "external-symmetric-graph-package-v1",
        "molecule_count": len(selected),
        "source_candidate_pool_sha256": _sha256(CANDIDATES),
        "canonical_cof_v2_manifest_sha256": _sha256(CANONICAL / "manifest.json"),
        "strict_nonoverlap": {
            "canonical_smiles": True,
            "bemis_murcko_scaffold": True,
            "morgan_radius": 2,
            "morgan_bits": 2048,
            "exact_morgan_fingerprint": True,
        },
        "reference_coordinates": (
            "deterministic ETKDGv3+UFF, evaluation-only; never used by generation "
            "or candidate selection"
        ),
    }
    _atomic_json(OUTPUT / "manifest.json", manifest)
    package_hashes = {
        filename: _sha256(OUTPUT / filename)
        for filename in (
            "cof_graphs.npz", "metadata.csv", "vocab.json", "statistics.json",
            "candidate_audit.json", "manifest.json",
        )
    }
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    protocol: dict[str, Any] = {
        **{key: value for key, value in parent.items() if key != "protocol_fingerprint"},
        "schema_version": "our-etflow-external-paired-protocol-v1",
        "status": "FROZEN_BEFORE_EXTERNAL_COORDINATE_GENERATION",
        "purpose": "Four-route evaluation on graphs outside canonical COF v2.",
        "panel": {
            "selection": "strict nonoverlap screen followed by SHA-ranked fixed PG quotas",
            "molecule_count": len(selected),
            "target_pg_counts": QUOTAS,
            "candidate_pool_count": len(pool),
            "accepted_before_quota_count": len(accepted),
            "rejected_count": len(rejected),
            "records": [{
                "package_index": item["package_index"],
                "molecule_id": item["molecule_id"],
                "target_pg": item["target_pg"],
                "seed": _seed(item["name"], "generation"),
                "atom_count": len(item["atomic_numbers"]),
                "name": item["name"],
                "canonical_smiles": item["canonical_smiles"],
                "murcko_scaffold_smiles": item["murcko_scaffold_smiles"],
                "nearest_cof_v2_tanimoto": item["nearest_cof_v2_tanimoto"],
                "nonoverlap_checks": item["nonoverlap_checks"],
            } for item in selected],
        },
        "metrics": {
            **parent["metrics"],
            "external_reference_note": (
                "bond/RMSD metrics use evaluation-only ETKDGv3+UFF coordinates and "
                "must not be interpreted as experimental conformer truth"
            ),
        },
        "scope": {
            **parent["scope"],
            "core_ood_test_only": False,
            "external_to_canonical_cof_v2": True,
            "external_graph_claim": True,
            "guaranteed_unseen_to_official_geom_training": False,
        },
        "identity": {
            **parent["identity"],
            "parent_ablation_protocol_sha256": _sha256(PARENT),
            "candidate_pool_sha256": _sha256(CANDIDATES),
            "external_package_sha256": package_hashes,
            "source_sha256": {
                **parent["identity"]["source_sha256"],
                "generative_model/evaluation/build_our_etflow_external_package_v1.py": _sha256(Path(__file__)),
                "generative_model/evaluation/run_our_etflow_external.py": _sha256(
                    ROOT / "generative_model/evaluation/run_our_etflow_external.py"
                ),
            },
        },
    }
    protocol["protocol_fingerprint"] = _fingerprint(protocol)
    _atomic_json(PROTOCOL_OUTPUT, protocol)
    return protocol


def main() -> None:
    protocol = build()
    print(json.dumps({
        "status": protocol["status"],
        "package": str(OUTPUT),
        "protocol": str(PROTOCOL_OUTPUT),
        "panel": protocol["panel"]["target_pg_counts"],
        "selected": [item["name"] for item in protocol["panel"]["records"]],
        "protocol_sha256": _sha256(PROTOCOL_OUTPUT),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
