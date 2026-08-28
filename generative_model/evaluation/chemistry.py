"""从统一稀疏图严格重建 RDKit 分子并计算 2D 化学表示。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from .schema import EvaluationSample


BOND_TYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}
BOND_TYPE_NAMES = {1: "single", 2: "double", 3: "triple", 4: "aromatic"}


@dataclass(frozen=True)
class ChemistryResult:
    molecule: Chem.Mol | None
    sanitize_error: str | None
    canonical_smiles: str | None
    fingerprint: Any | None
    ring_count: int | None

    @property
    def valid(self) -> bool:
        return self.molecule is not None and self.sanitize_error is None


@dataclass(frozen=True)
class EvaluationReference:
    """训练集 novelty/Tanimoto 参考，构建时严格禁止跳过无效 SMILES。"""

    source_fingerprint: str
    molecule_count: int
    canonical_smiles: frozenset[str]
    fingerprints: tuple[Any, ...]
    fingerprint_radius: int
    fingerprint_bits: int
    include_chirality: bool

    def describe(self) -> dict[str, Any]:
        return {
            "source_fingerprint": self.source_fingerprint,
            "molecule_count": self.molecule_count,
            "unique_canonical_smiles": len(self.canonical_smiles),
            "fingerprint": {
                "type": "Morgan",
                "radius": self.fingerprint_radius,
                "bits": self.fingerprint_bits,
                "include_chirality": self.include_chirality,
                "explicit_h": False,
            },
        }


def fingerprint_generator(radius: int, bits: int, include_chirality: bool) -> Any:
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=bits,
        includeChirality=include_chirality,
    )


def canonical_graph_smiles(molecule: Chem.Mol) -> str:
    """生成适用于图生成模型的无显式 H、非立体 canonical SMILES。"""

    heavy = Chem.RemoveHs(Chem.Mol(molecule))
    return Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=False)


def build_reference(
    smiles: Iterable[str],
    *,
    radius: int,
    bits: int,
    include_chirality: bool,
    identity: dict[str, Any],
) -> EvaluationReference:
    generator = fingerprint_generator(radius, bits, include_chirality)
    canonical: list[str] = []
    fingerprints: list[Any] = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"reference SMILES {index} 无法由 RDKit 解析")
        molecule = Chem.RemoveHs(molecule)
        canonical.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))
        fingerprints.append(generator.GetFingerprint(molecule))
    fingerprint_payload = {
        "identity": identity,
        "canonical_smiles": canonical,
        "radius": radius,
        "bits": bits,
        "include_chirality": include_chirality,
    }
    source_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return EvaluationReference(
        source_fingerprint=source_fingerprint,
        molecule_count=len(canonical),
        canonical_smiles=frozenset(canonical),
        fingerprints=tuple(fingerprints),
        fingerprint_radius=radius,
        fingerprint_bits=bits,
        include_chirality=include_chirality,
    )


def reconstruct_molecule(
    sample: EvaluationSample,
    *,
    generator: Any,
) -> ChemistryResult:
    """从 canonical ground-truth 字段重建；失败时不改变键型或电荷。"""

    editable = Chem.RWMol()
    for atomic_number, formal_charge in zip(sample.atomic_numbers, sample.formal_charges):
        atom = Chem.Atom(int(atomic_number))
        atom.SetFormalCharge(int(formal_charge))
        # canonical dataset 已显式包含 H；禁止 RDKit 为缺价结构静默补隐式 H。
        atom.SetNoImplicit(True)
        editable.AddAtom(atom)
    for (begin, end), bond_type in zip(sample.bond_index.T, sample.bond_types):
        editable.AddBond(int(begin), int(end), BOND_TYPES[int(bond_type)])
        if int(bond_type) == 4:
            bond = editable.GetBondBetweenAtoms(int(begin), int(end))
            if bond is None:
                raise RuntimeError("RDKit 未返回刚添加的 aromatic bond")
            bond.SetIsAromatic(True)
            editable.GetAtomWithIdx(int(begin)).SetIsAromatic(True)
            editable.GetAtomWithIdx(int(end)).SetIsAromatic(True)
    molecule = editable.GetMol()
    conformer = Chem.Conformer(len(sample.atomic_numbers))
    for index, position in enumerate(sample.positions):
        conformer.SetAtomPosition(index, tuple(float(value) for value in position))
    molecule.AddConformer(conformer, assignId=True)

    sanitize_flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if sanitize_flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return ChemistryResult(
            molecule=None,
            sanitize_error=str(sanitize_flag),
            canonical_smiles=None,
            fingerprint=None,
            ring_count=None,
        )
    smiles = canonical_graph_smiles(molecule)
    heavy = Chem.RemoveHs(Chem.Mol(molecule))
    return ChemistryResult(
        molecule=molecule,
        sanitize_error=None,
        canonical_smiles=smiles,
        fingerprint=generator.GetFingerprint(heavy),
        ring_count=int(rdMolDescriptors.CalcNumRings(heavy)),
    )


def nearest_similarity(fingerprint: Any, reference: EvaluationReference) -> float | None:
    if not reference.fingerprints:
        return None
    return float(max(DataStructs.BulkTanimotoSimilarity(fingerprint, list(reference.fingerprints))))
