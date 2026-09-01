"""Strict adapter for the external Cn heavy-atom graph collection.

The uploaded Cn package stores only heavy atoms.  This adapter validates the
numeric graph against the supplied SMILES, then uses RDKit solely to make the
implicit hydrogens explicit.  No element, bond, charge, or radical fallback is
allowed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem


CN_DATASET_SCHEMA_VERSION = "cn-heavy-graph-explicit-h-adapter-v1"
SUPPORTED_GROUPS = ("C3k2", "C3k3")
EXPECTED_ATOM_VOCAB = ("C", "N", "O", "F")
SOURCE_BOND_TYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.AROMATIC,
    3: Chem.BondType.DOUBLE,
    4: Chem.BondType.TRIPLE,
}
CANONICAL_BOND_TYPES = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decode_vocab(values: np.ndarray) -> tuple[str, ...]:
    result = tuple(
        value.decode("ascii") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    )
    if result != EXPECTED_ATOM_VOCAB:
        raise ValueError(
            f"Cn atom_vocab 必须严格为 {EXPECTED_ATOM_VOCAB}，收到 {result}"
        )
    return result


def _parse_labels(path: Path) -> list[tuple[str, int, int, str]]:
    records: list[tuple[str, int, int, str]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number} 必须为 5 列 TSV")
        label_id, original_id, order, orbit_count, smiles = fields
        if original_id != "-":
            raise ValueError(f"{path}:{line_number} 原种子列必须为 '-'")
        records.append((label_id, int(order), int(orbit_count), smiles))
    if not records:
        raise ValueError(f"Cn labels 为空: {path}")
    return records


def _molecule_from_source_graph(
    x_idx: np.ndarray, adjacency: np.ndarray, vocab: tuple[str, ...]
) -> Chem.Mol:
    indices = np.asarray(x_idx, dtype=np.int64)
    matrix = np.asarray(adjacency, dtype=np.int64)
    atom_count = len(indices)
    if indices.shape != (atom_count,) or atom_count == 0:
        raise ValueError("Cn x_idx 必须为非空 [N]")
    if np.any((indices < 0) | (indices >= len(vocab))):
        raise ValueError("Cn x_idx 越界；不允许元素 fallback")
    if matrix.shape != (atom_count, atom_count):
        raise ValueError("Cn adj_corr shape 与 x_idx 不一致")
    if not np.array_equal(matrix, matrix.T) or np.any(np.diag(matrix) != 0):
        raise ValueError("Cn adj_corr 必须对称且对角为 0")
    if not np.isin(matrix, np.asarray([0, 1, 2, 3, 4])).all():
        raise ValueError("Cn adj_corr 含未知键型")

    editable = Chem.RWMol()
    for index in indices:
        editable.AddAtom(Chem.Atom(vocab[int(index)]))
    for left in range(atom_count):
        for right in range(left + 1, atom_count):
            source_kind = int(matrix[left, right])
            if source_kind == 0:
                continue
            editable.AddBond(left, right, SOURCE_BOND_TYPES[source_kind])
            if source_kind == 2:
                bond = editable.GetBondBetweenAtoms(left, right)
                bond.SetIsAromatic(True)
                editable.GetAtomWithIdx(left).SetIsAromatic(True)
                editable.GetAtomWithIdx(right).SetIsAromatic(True)
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    if Chem.GetMolFrags(molecule, asMols=False, sanitizeFrags=False) and len(
        Chem.GetMolFrags(molecule, asMols=False, sanitizeFrags=False)
    ) != 1:
        raise ValueError("Cn adj_corr 分子图不连通")
    return molecule


def _canonical_arrays(molecule: Chem.Mol) -> dict[str, np.ndarray]:
    explicit = Chem.AddHs(molecule)
    numbers = np.asarray(
        [atom.GetAtomicNum() for atom in explicit.GetAtoms()], dtype=np.int64
    )
    charges = np.asarray(
        [atom.GetFormalCharge() for atom in explicit.GetAtoms()], dtype=np.int64
    )
    radicals = np.asarray(
        [atom.GetNumRadicalElectrons() for atom in explicit.GetAtoms()], dtype=np.int64
    )
    bonds: list[tuple[tuple[int, int], int]] = []
    for bond in explicit.GetBonds():
        kind = CANONICAL_BOND_TYPES.get(bond.GetBondType())
        if kind is None:
            raise ValueError(f"Cn RDKit 分子含未知键型: {bond.GetBondType()}")
        pair = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        bonds.append((pair, kind))
    bonds.sort()
    if not bonds:
        raise ValueError("Cn 分子没有化学键")
    return {
        "atomic_numbers": numbers,
        "formal_charges": charges,
        "radical_electrons": radicals,
        "bond_index": np.asarray([value[0] for value in bonds], dtype=np.int64).T,
        "bond_types": np.asarray([value[1] for value in bonds], dtype=np.int64),
    }


class CnGraphDataset:
    """Load C3k2/C3k3 records as strict explicit-H canonical graph dicts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Cn 数据目录不存在: {self.root}")
        self.records: list[dict[str, Any]] = []
        for group in SUPPORTED_GROUPS:
            directory = self.root / group
            archive_path = directory / f"graphs_{group}.npz"
            labels_path = directory / f"labels_{group}.txt"
            if not archive_path.is_file() or not labels_path.is_file():
                raise FileNotFoundError(f"Cn {group} 缺少 NPZ 或 labels")
            labels = _parse_labels(labels_path)
            # The local package intentionally stores object-string arrays.  We
            # read them only to prove that NPZ and TSV indices agree.
            with np.load(archive_path, allow_pickle=True) as archive:
                required = {
                    "x", "x_idx", "adj_corr", "n_atoms", "smiles", "order",
                    "orbit", "atom_vocab",
                }
                missing = sorted(required - set(archive.files))
                if missing:
                    raise ValueError(f"Cn {group} NPZ 缺字段: {missing}")
                arrays = {name: archive[name] for name in required}
            count = len(arrays["x_idx"])
            record_arrays = (
                "x", "x_idx", "adj_corr", "n_atoms", "smiles", "order", "orbit"
            )
            if len(labels) != count or any(
                len(arrays[name]) != count for name in record_arrays
            ):
                raise ValueError(f"Cn {group} 数组/labels 记录数不一致")
            vocab = _decode_vocab(arrays["atom_vocab"])
            for local_index in range(count):
                label_id, order, orbit_count, label_smiles = labels[local_index]
                npz_smiles = str(arrays["smiles"][local_index])
                if npz_smiles != label_smiles:
                    raise ValueError(f"Cn {group}/{label_id} NPZ/labels SMILES 不一致")
                expected_orbits = 2 if group == "C3k2" else 3
                if (
                    order != 3
                    or orbit_count != expected_orbits
                    or int(arrays["order"][local_index]) != order
                    or int(arrays["orbit"][local_index]) != orbit_count
                ):
                    raise ValueError(f"Cn {group}/{label_id} C3/orbit 标签不一致")
                x = np.asarray(arrays["x"][local_index], dtype=np.float64)
                x_idx = np.asarray(arrays["x_idx"][local_index], dtype=np.int64)
                if x.shape != (len(x_idx), len(vocab)) or not np.isfinite(x).all():
                    raise ValueError(f"Cn {group}/{label_id} x shape/finite 非法")
                if not np.array_equal(np.argmax(x, axis=1), x_idx):
                    raise ValueError(f"Cn {group}/{label_id} x_idx != argmax(x)")
                if int(arrays["n_atoms"][local_index]) != len(x_idx):
                    raise ValueError(f"Cn {group}/{label_id} n_atoms 不一致")

                molecule = _molecule_from_source_graph(
                    x_idx, arrays["adj_corr"][local_index], vocab
                )
                smiles_molecule = Chem.MolFromSmiles(label_smiles)
                if smiles_molecule is None:
                    raise ValueError(f"Cn {group}/{label_id} SMILES 无法解析")
                graph_key = Chem.MolToInchiKey(molecule)
                smiles_key = Chem.MolToInchiKey(smiles_molecule)
                if not graph_key or graph_key != smiles_key:
                    raise ValueError(
                        f"Cn {group}/{label_id} adj_corr/SMILES InChIKey 不一致"
                    )
                canonical = _canonical_arrays(molecule)
                global_index = len(self.records)
                molecule_name = f"mol_{local_index + 1:03d}_{group}"
                self.records.append({
                    "package_index": global_index,
                    "molecule_id": molecule_name,
                    **canonical,
                    "metadata": {
                        "Molecule_ID": molecule_name,
                        "SMILES": label_smiles,
                        "Target_PG": "C3",
                        "Split_IID": "external_Cn",
                        "Source_Group": group,
                        "Source_Label_ID": label_id,
                        "Source_Orbit_Count": orbit_count,
                    },
                    "adapter_audit": {
                        "schema_version": CN_DATASET_SCHEMA_VERSION,
                        "source_group": group,
                        "source_local_index": local_index,
                        "source_label_id": label_id,
                        "source_heavy_atom_count": len(x_idx),
                        "explicit_h_atom_count": int(np.sum(canonical["atomic_numbers"] == 1)),
                        "output_atom_count": len(canonical["atomic_numbers"]),
                        "inchi_key": graph_key,
                        "npz_smiles_matches_label": True,
                        "x_idx_matches_argmax": True,
                        "adj_corr_smiles_inchi_key_matches": True,
                        "implicit_h_expansion_used": True,
                        "element_fallback_used": False,
                        "graph_repair_used": False,
                    },
                    "source_identity": {
                        "archive_relative_path": str(archive_path.relative_to(self.root)),
                        "archive_sha256": _sha256(archive_path),
                        "labels_relative_path": str(labels_path.relative_to(self.root)),
                        "labels_sha256": _sha256(labels_path),
                    },
                })
        if len(self.records) != 98:
            raise ValueError(f"Cn 数据应为 98 条，实际 {len(self.records)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, Any]:
        return self.records[item]
