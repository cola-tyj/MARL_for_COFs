"""
COF Building Block Dataset for diffusion training.

Loads original + augmented cjson files and converts to PyG Data format
with symmetry and connectivity labels for conditional diffusion training.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset

from ..symmetry.symmetry_encoder import (
    MOLECULAR_POINT_GROUPS,
    POINT_GROUP_TO_IDX,
)


# Extended atom vocabulary for COF
COF_ATOM_SYMBOLS = [
    "H", "C", "N", "O", "F", "Cl", "Br", "S", "Q", "X",
]
SYMBOL_TO_IDX_COF = {s: i for i, s in enumerate(COF_ATOM_SYMBOLS)}
NUM_COF_ATOM_TYPES = len(COF_ATOM_SYMBOLS)

# Functional group types
FUNC_GROUP_TYPES = [
    "H", "CHO", "NH2", "COOH", "CN", "OH", "Cl", "Br", "CH3", "NO2",
]
FUNC_GROUP_TO_IDX = {fg: i for i, fg in enumerate(FUNC_GROUP_TYPES)}

# COF symmetry type → point group mapping
COF_SYMMETRY_TO_POINT_GROUP = {
    "L2": "C2v",
    "T3": "D3h",
    "S4": "D4h",
    "H6": "D6h",
    "D4": "D4h",
    "R4": "D4h",
    "C2": "C2v",
    "C3": "C3",
    "C4": "C4",
    "C6": "C6",
}

# Symmetry type → expected connector count
SYMMETRY_CONNECTORS = {
    "L2": 2, "C2": 2,
    "T3": 3, "C3": 3,
    "S4": 4, "C4": 4, "D4": 4, "R4": 4,
    "H6": 6, "C6": 6,
}


class COFBBDataModule:
    """
    Data module for COF building block conditional diffusion training.

    Loads cjson files, computes symmetry labels, and creates DataLoaders.
    """

    def __init__(
        self,
        data_dir: str = "/home/tianyajun/MARL_for_COFs/pycofbuilder/data",
        augmented_dir: Optional[str] = None,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        self.data_dir = Path(data_dir)
        self.augmented_dir = Path(augmented_dir) if augmented_dir else None
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def setup(self) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Load and split all data.

        Returns:
            (train_data, val_data, test_data) lists of PyG Data objects
        """
        all_data = []

        # Load original cores
        core_dir = self.data_dir / "core"
        if core_dir.exists():
            for sym_dir in core_dir.iterdir():
                if sym_dir.is_dir():
                    for cjson_file in sym_dir.glob("*.cjson"):
                        try:
                            data = self._load_cjson_to_data(
                                str(cjson_file), sym_dir.name
                            )
                            if data is not None:
                                all_data.append(data)
                        except Exception:
                            pass

        # Load augmented data if available
        if self.augmented_dir and self.augmented_dir.exists():
            for sym_dir in self.augmented_dir.iterdir():
                if sym_dir.is_dir():
                    for cjson_file in sym_dir.glob("*.cjson"):
                        try:
                            data = self._load_cjson_to_data(
                                str(cjson_file), sym_dir.name
                            )
                            if data is not None:
                                all_data.append(data)
                        except Exception:
                            pass

        print(f"Loaded {len(all_data)} total building blocks")

        # Shuffle and split
        indices = torch.randperm(len(all_data)).tolist()
        train_end = int(len(all_data) * self.train_ratio)
        val_end = int(len(all_data) * (self.train_ratio + self.val_ratio))

        train_data = [all_data[i] for i in indices[:train_end]]
        val_data = [all_data[i] for i in indices[train_end:val_end]]
        test_data = [all_data[i] for i in indices[val_end:]]

        return train_data, val_data, test_data

    def _load_cjson_to_data(self, filepath: str, sym_type: str) -> Optional[Data]:
        """Load a cjson file and convert to PyG Data with labels."""
        with open(filepath) as f:
            cjson = json.load(f)

        elements = cjson["atoms"]["elements"]["type"]
        coords_flat = cjson["atoms"]["coords"]["3d"]
        n_atoms = len(elements)

        # One-hot atom types
        x = torch.zeros(n_atoms, NUM_COF_ATOM_TYPES)
        for i, elem in enumerate(elements):
            idx = SYMBOL_TO_IDX_COF.get(elem, 1)  # default to C
            x[i, idx] = 1.0

        # Positions
        positions = torch.tensor(coords_flat, dtype=torch.float32).reshape(n_atoms, 3)

        # Edges (fully connected for small molecules)
        if n_atoms > 1:
            row = torch.arange(n_atoms).unsqueeze(1).expand(-1, n_atoms).reshape(-1)
            col = torch.arange(n_atoms).unsqueeze(0).expand(n_atoms, -1).reshape(-1)
            keep = row != col
            edge_index = torch.stack([row[keep], col[keep]], dim=0)

            # Infer bond types from distances
            edge_attr = self._infer_bonds(positions, edge_index, elements)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, 5)

        # Symmetry label
        point_group = COF_SYMMETRY_TO_POINT_GROUP.get(sym_type, "C1")
        symm_idx = POINT_GROUP_TO_IDX.get(point_group, 0)

        # Connector count
        num_connectors = SYMMETRY_CONNECTORS.get(sym_type, 0)

        # Count actual Q/X atoms
        q_count = sum(1 for e in elements if e in ("Q", "X"))

        # Functional group type
        connector_type = cjson.get("properties", {}).get("connector_type", "CHO")
        if isinstance(connector_type, list):
            connector_type = connector_type[0] if connector_type else "CHO"
        fg_idx = FUNC_GROUP_TO_IDX.get(connector_type, 1)  # default CHO

        return Data(
            x=x,
            positions=positions,
            edge_index=edge_index,
            edge_attr=edge_attr,
            symm_idx=torch.tensor(symm_idx, dtype=torch.long),
            num_connectors=torch.tensor(num_connectors, dtype=torch.long),
            func_group_type=torch.tensor(fg_idx, dtype=torch.long),
            q_count=torch.tensor(q_count, dtype=torch.long),
            num_atoms=n_atoms,
            sym_type=sym_type,
        )

    @staticmethod
    def _infer_bonds(
        positions: Tensor, edge_index: Tensor, elements: List[str]
    ) -> Tensor:
        """Infer bond types from pairwise distances."""
        row, col = edge_index[0], edge_index[1]
        dist = (positions[row] - positions[col]).norm(dim=-1)

        # Simple covalent radii
        radii = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
                 "Cl": 1.02, "Br": 1.20, "S": 1.05, "Q": 0.7, "X": 0.7}

        num_edges = edge_index.size(1)
        bond_attr = torch.zeros(num_edges, 5)

        for e in range(num_edges):
            d = dist[e].item()
            ri = radii.get(elements[row[e].item()], 0.7)
            rj = radii.get(elements[col[e].item()], 0.7)
            r_sum = ri + rj

            if d < r_sum * 0.7:
                bond_attr[e, 3] = 1.0   # triple
            elif d < r_sum * 0.9:
                bond_attr[e, 2] = 1.0   # double
            elif d < r_sum * 1.2:
                bond_attr[e, 1] = 1.0   # single
            elif d < r_sum * 1.5:
                bond_attr[e, 4] = 1.0   # aromatic
            else:
                bond_attr[e, 0] = 1.0   # none

        return bond_attr

    def compute_marginals(self, data_list: List[Data]) -> Tuple[Tensor, Tensor]:
        """Compute empirical marginals for discrete diffusion."""
        atom_counts = torch.zeros(NUM_COF_ATOM_TYPES)
        bond_counts = torch.zeros(5)

        for data in data_list:
            atom_idx = data.x.argmax(dim=-1)
            for a in atom_idx:
                atom_counts[a] += 1
            if data.edge_attr is not None:
                bond_idx = data.edge_attr.argmax(dim=-1)
                for b in bond_idx:
                    bond_counts[b] += 1

        atom_marginals = atom_counts / atom_counts.sum().clamp(min=1)
        bond_marginals = bond_counts / bond_counts.sum().clamp(min=1)

        atom_marginals = atom_marginals.clamp(min=1e-5)
        bond_marginals = bond_marginals.clamp(min=1e-5)

        return (atom_marginals / atom_marginals.sum(),
                bond_marginals / bond_marginals.sum())
