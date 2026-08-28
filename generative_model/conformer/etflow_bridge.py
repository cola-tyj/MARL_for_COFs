"""Strict canonical-graph bridge for the external ET-Flow package.

ET-Flow is an alternative *initial conformer provider*.  It is not a point-group
conditioned model and it is not part of the deterministic ETKDG E2/E3 route.
Atom-map numbers are used only as a lossless transport index.  Elements,
charges, radicals, typed bonds and explicit hydrogens must round-trip exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem


ETFLOW_BRIDGE_SCHEMA_VERSION = "etflow-canonical-bridge-v1"
_BOND_TYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}
_BOND_TYPE_IDS = {value: key for key, value in _BOND_TYPES.items()}


def _canonical_arrays(sample: dict[str, Any]) -> tuple[np.ndarray, ...]:
    atomic_numbers = np.asarray(sample["atomic_numbers"], dtype=np.int64)
    formal_charges = np.asarray(sample["formal_charges"], dtype=np.int64)
    radicals = np.asarray(sample["radical_electrons"], dtype=np.int64)
    bond_index = np.asarray(sample["bond_index"], dtype=np.int64)
    bond_types = np.asarray(sample["bond_types"], dtype=np.int64)
    atom_count = len(atomic_numbers)
    if atom_count == 0:
        raise ValueError("canonical graph 不能为空")
    if formal_charges.shape != (atom_count,) or radicals.shape != (atom_count,):
        raise ValueError("canonical atom fields 长度不一致")
    if bond_index.shape != (2, len(bond_types)):
        raise ValueError("canonical sparse bond shape 非法")
    if np.any((atomic_numbers <= 0) | (atomic_numbers >= 119)):
        raise ValueError("canonical atomic number 超出 1..118")
    seen: set[tuple[int, int]] = set()
    for edge, kind in zip(bond_index.T, bond_types, strict=True):
        left, right = map(int, edge)
        if not 0 <= left < right < atom_count:
            raise ValueError("canonical bonds 必须为有序无向 pair")
        if (left, right) in seen:
            raise ValueError("canonical graph 含重复 bond")
        if int(kind) not in _BOND_TYPES:
            raise ValueError(f"canonical bond type 非法: {int(kind)}")
        seen.add((left, right))
    return atomic_numbers, formal_charges, radicals, bond_index, bond_types


def _typed_bond_set(
    bond_index: np.ndarray, bond_types: np.ndarray
) -> set[tuple[int, int, int]]:
    return {
        (int(left), int(right), int(kind))
        for (left, right), kind in zip(bond_index.T, bond_types, strict=True)
    }


def build_mapped_explicit_h_smiles(sample: dict[str, Any]) -> str:
    """Serialize the immutable explicit-H graph with map number ``index + 1``."""

    atomic_numbers, charges, radicals, bond_index, bond_types = _canonical_arrays(sample)
    editable = Chem.RWMol()
    aromatic_atoms: set[int] = set()
    for (left, right), kind in zip(bond_index.T, bond_types, strict=True):
        if int(kind) == 4:
            aromatic_atoms.update((int(left), int(right)))
    for index, (number, charge, radical) in enumerate(
        zip(atomic_numbers, charges, radicals, strict=True)
    ):
        atom = Chem.Atom(int(number))
        atom.SetFormalCharge(int(charge))
        atom.SetNumRadicalElectrons(int(radical))
        atom.SetNoImplicit(True)
        atom.SetAtomMapNum(index + 1)
        atom.SetIsAromatic(index in aromatic_atoms)
        editable.AddAtom(atom)
    for (left, right), kind in zip(bond_index.T, bond_types, strict=True):
        editable.AddBond(int(left), int(right), _BOND_TYPES[int(kind)])
        if int(kind) == 4:
            editable.GetBondBetweenAtoms(int(left), int(right)).SetIsAromatic(True)
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    smiles = Chem.MolToSmiles(
        molecule,
        canonical=False,
        isomericSmiles=True,
        allHsExplicit=True,
    )
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise RuntimeError("mapped explicit-H SMILES 无法被 RDKit 重新解析")
    audit_model_molecule(sample, Chem.AddHs(parsed), maximum_atomic_number=100)
    return smiles


def audit_model_molecule(
    sample: dict[str, Any],
    model_molecule: Chem.Mol,
    *,
    maximum_atomic_number: int,
) -> dict[str, Any]:
    """Prove that a model-side RDKit molecule is the same canonical graph."""

    if model_molecule is None:
        raise ValueError("model molecule 为空")
    atomic_numbers, charges, radicals, bond_index, bond_types = _canonical_arrays(sample)
    atom_count = len(atomic_numbers)
    if maximum_atomic_number <= 1:
        raise ValueError("maximum_atomic_number 非法")
    if int(atomic_numbers.max()) >= int(maximum_atomic_number):
        raise ValueError(
            f"ET-Flow atomic-number embedding 不支持 Z={int(atomic_numbers.max())}; "
            f"要求 Z < {int(maximum_atomic_number)}"
        )
    if model_molecule.GetNumAtoms() != atom_count:
        raise ValueError(
            f"ET-Flow explicit-H atom count 改变: {model_molecule.GetNumAtoms()} != {atom_count}"
        )
    observed_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in model_molecule.GetAtoms()], dtype=np.int64
    )
    map_numbers = np.asarray(
        [atom.GetAtomMapNum() for atom in model_molecule.GetAtoms()], dtype=np.int64
    )
    # Official ET-Flow parses with Chem.MolFromSmiles and then Chem.AddHs. RDKit
    # consequently keeps heavy-atom maps but reconstructs H with map number 0.
    # Lift those H indices using only typed-graph adjacency; coordinates are
    # never used and ambiguous/mismatched neighborhoods fail.
    model_to_canonical = np.full(atom_count, -1, dtype=np.int64)
    canonical_heavy = np.flatnonzero(atomic_numbers != 1)
    model_heavy = np.flatnonzero(observed_numbers != 1)
    if len(model_heavy) != len(canonical_heavy):
        raise ValueError("ET-Flow heavy-atom count 改变")
    expected_heavy_maps = canonical_heavy + 1
    if not np.array_equal(np.sort(map_numbers[model_heavy]), np.sort(expected_heavy_maps)):
        raise ValueError("ET-Flow heavy-atom maps 与 canonical indices 不一致")
    model_to_canonical[model_heavy] = map_numbers[model_heavy] - 1

    canonical_h_by_parent: dict[int, list[int]] = {int(index): [] for index in canonical_heavy}
    for hydrogen in np.flatnonzero(atomic_numbers == 1):
        incident = []
        for (left, right), kind in zip(bond_index.T, bond_types, strict=True):
            if int(left) == int(hydrogen) or int(right) == int(hydrogen):
                incident.append((int(right) if int(left) == int(hydrogen) else int(left), int(kind)))
        if len(incident) != 1 or incident[0][1] != 1 or atomic_numbers[incident[0][0]] == 1:
            raise ValueError("canonical explicit H 必须以 single bond 连接唯一重原子")
        canonical_h_by_parent[incident[0][0]].append(int(hydrogen))
    for values in canonical_h_by_parent.values():
        values.sort()

    for model_parent in model_heavy:
        canonical_parent = int(model_to_canonical[model_parent])
        model_h = sorted(
            neighbor.GetIdx()
            for neighbor in model_molecule.GetAtomWithIdx(int(model_parent)).GetNeighbors()
            if neighbor.GetAtomicNum() == 1
        )
        canonical_h = canonical_h_by_parent[canonical_parent]
        if len(model_h) != len(canonical_h):
            raise ValueError(
                f"ET-Flow explicit-H neighborhood 改变: parent={canonical_parent}, "
                f"model={len(model_h)}, canonical={len(canonical_h)}"
            )
        for model_index, canonical_index in zip(model_h, canonical_h, strict=True):
            if model_to_canonical[model_index] != -1:
                raise ValueError("ET-Flow H lifting 重复赋值")
            model_to_canonical[model_index] = canonical_index
    if not np.array_equal(np.sort(model_to_canonical), np.arange(atom_count)):
        raise ValueError("ET-Flow graph-only atom lifting 未得到 N 元双射")
    observed_charges = np.asarray(
        [atom.GetFormalCharge() for atom in model_molecule.GetAtoms()], dtype=np.int64
    )
    observed_radicals = np.asarray(
        [atom.GetNumRadicalElectrons() for atom in model_molecule.GetAtoms()], dtype=np.int64
    )
    for name, observed, expected in (
        ("atomic numbers", observed_numbers, atomic_numbers),
        ("formal charges", observed_charges, charges),
        ("radical electrons", observed_radicals, radicals),
    ):
        canonical_order = np.empty_like(observed)
        canonical_order[model_to_canonical] = observed
        if not np.array_equal(canonical_order, expected):
            raise ValueError(f"ET-Flow bridge 修改了 {name}")

    model_edges: set[tuple[int, int, int]] = set()
    for bond in model_molecule.GetBonds():
        kind = _BOND_TYPE_IDS.get(bond.GetBondType())
        if kind is None:
            raise ValueError(f"ET-Flow model molecule 含未知 bond type: {bond.GetBondType()}")
        left = int(model_to_canonical[bond.GetBeginAtomIdx()])
        right = int(model_to_canonical[bond.GetEndAtomIdx()])
        model_edges.add((min(left, right), max(left, right), int(kind)))
    expected_edges = _typed_bond_set(bond_index, bond_types)
    if model_edges != expected_edges:
        missing = sorted(expected_edges - model_edges)[:10]
        extra = sorted(model_edges - expected_edges)[:10]
        raise ValueError(f"ET-Flow bridge 修改 typed bonds: missing={missing}, extra={extra}")
    sn_indices = np.flatnonzero(atomic_numbers == 50)
    return {
        "schema_version": ETFLOW_BRIDGE_SCHEMA_VERSION,
        "atom_count": atom_count,
        "bond_count": len(expected_edges),
        "explicit_h_count": int(np.sum(atomic_numbers == 1)),
        "sn_count": int(len(sn_indices)),
        "sn_preserved": bool(
            np.all(observed_numbers[np.flatnonzero(np.isin(model_to_canonical, sn_indices))] == 50)
        ),
        "maximum_atomic_number": int(atomic_numbers.max()),
        "model_max_z_exclusive": int(maximum_atomic_number),
        "model_to_canonical": model_to_canonical,
        "element_fallback_used": False,
        "graph_mutation_used": False,
    }


def audit_official_node_features(node_attr: Any, *, atom_count: int) -> dict[str, Any]:
    """Report ET-Flow's upstream ``misc`` buckets instead of hiding them.

    This follows the current official ten-column OGB-style feature layout.  A
    future package changing that layout fails explicitly and requires a new
    bridge schema.
    """

    values = np.asarray(node_attr.detach().cpu() if hasattr(node_attr, "detach") else node_attr)
    if values.shape != (atom_count, 10) or not np.isfinite(values).all():
        raise ValueError(f"ET-Flow node_attr schema changed: observed {values.shape}, expected {(atom_count, 10)}")
    misc_indices = np.asarray([4, 11, 11, 7, 10, 6, 5], dtype=np.int64)
    rounded = np.rint(values[:, :7]).astype(np.int64)
    if not np.allclose(values[:, :7], rounded):
        raise ValueError("ET-Flow categorical node features 不是整数索引")
    counts = {
        name: int(np.sum(rounded[:, column] == misc_indices[column]))
        for column, name in enumerate(
            ("chirality", "degree", "formal_charge", "implicit_valence", "total_h", "hybridization", "radical_electrons")
        )
    }
    return {
        "schema_version": "etflow-official-node-feature-audit-v1",
        "misc_counts": counts,
        "atoms_with_any_misc": int(np.sum(np.any(rounded == misc_indices[None, :], axis=1))),
        "misc_is_reported_not_silenced": True,
        "element_fallback_used": False,
    }


def reorder_positions_to_canonical(
    positions: np.ndarray,
    model_to_canonical: np.ndarray,
    *,
    atom_count: int,
) -> np.ndarray:
    """Reorder ``[N,3]`` or ``[K,N,3]`` model coordinates by atom maps."""

    values = np.asarray(positions, dtype=np.float64)
    mapping = np.asarray(model_to_canonical, dtype=np.int64)
    if mapping.shape != (atom_count,) or not np.array_equal(
        np.sort(mapping), np.arange(atom_count, dtype=np.int64)
    ):
        raise ValueError("model_to_canonical 不是 N 元双射")
    if values.ndim == 2:
        if values.shape != (atom_count, 3):
            raise ValueError("ET-Flow positions 必须为 [N,3]")
        result = np.empty_like(values); result[mapping] = values
    elif values.ndim == 3:
        if values.shape[1:] != (atom_count, 3):
            raise ValueError("ET-Flow sampled positions 必须为 [K,N,3]")
        result = np.empty_like(values); result[:, mapping] = values
    else:
        raise ValueError("ET-Flow positions rank 必须为 2 或 3")
    if not np.isfinite(result).all():
        raise ValueError("ET-Flow positions 含 NaN/Inf")
    return result


class ETFlowRuntime:
    """Lazy official-package runtime; construction may download a checkpoint."""

    def __init__(
        self,
        *,
        model_name: str = "drugs-o3",
        device: str = "cpu",
        cache: str | None = None,
    ) -> None:
        try:
            from etflow import BaseFlow
            from etflow.commons.featurization import MoleculeFeaturizer, get_mol_from_smiles
        except ImportError as error:
            raise RuntimeError(
                "ET-Flow 未安装；请在独立 env_etflow 中安装，禁止修改冻结的项目环境"
            ) from error
        self.model_name = str(model_name)
        self.device = str(device)
        self.featurizer = MoleculeFeaturizer()
        self._get_mol_from_smiles = get_mol_from_smiles
        kwargs = {"model": self.model_name, "device": self.device}
        if cache is not None:
            kwargs["cache"] = str(cache)
        self.model = BaseFlow.from_default(**kwargs)
        self.model.to(self.device)
        cache_directory = Path(cache).expanduser() if cache is not None else Path("~/.cache/etflow").expanduser()
        self.checkpoint_path = cache_directory / f"{self.model_name}.ckpt"
        if not self.checkpoint_path.is_file():
            raise RuntimeError(f"ET-Flow checkpoint 加载后仍找不到文件: {self.checkpoint_path}")

    def predict(
        self,
        sample: dict[str, Any],
        *,
        num_samples: int,
        n_timesteps: int,
        seed: int,
    ) -> dict[str, Any]:
        if num_samples <= 0 or n_timesteps <= 0:
            raise ValueError("num_samples/n_timesteps 必须为正")
        smiles = build_mapped_explicit_h_smiles(sample)
        model_molecule = self._get_mol_from_smiles(smiles)
        bridge = audit_model_molecule(sample, model_molecule, maximum_atomic_number=100)
        graph = self.featurizer.get_data_from_smiles(smiles)
        feature_audit = audit_official_node_features(
            graph.node_attr, atom_count=bridge["atom_count"]
        )
        output = self.model.predict(
            [smiles],
            max_batch_size=1,
            num_samples=int(num_samples),
            n_timesteps=int(n_timesteps),
            seed=int(seed),
            device=self.device,
            as_mol=False,
        )
        if smiles not in output:
            raise RuntimeError("ET-Flow output 缺少输入 SMILES key")
        positions = reorder_positions_to_canonical(
            output[smiles], bridge["model_to_canonical"], atom_count=bridge["atom_count"]
        )
        return {
            "schema_version": "etflow-canonical-prediction-v1",
            "model_name": self.model_name,
            "smiles": smiles,
            "positions": positions,
            "bridge_audit": bridge,
            "feature_audit": feature_audit,
        }
