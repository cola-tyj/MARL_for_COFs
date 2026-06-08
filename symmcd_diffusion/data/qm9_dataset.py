"""
QM9 dataset loader for molecular diffusion pre-training.

QM9 dataset: ~130K small organic molecules with up to 9 heavy atoms
(C, N, O, F) plus hydrogen. Each molecule has 3D geometry optimized at
the B3LYP/6-31G(2df,p) level of theory.

We use the PyTorch Geometric QM9 dataset and convert to our format.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.datasets import QM9 as PyGQM9
from torch_geometric.loader import DataLoader

# Atom type mapping for QM9
# QM9 uses atomic numbers: H=1, C=6, N=7, O=8, F=9
QM9_ATOMIC_NUMBERS = [1, 6, 7, 8, 9]
QM9_ATOM_TO_IDX = {z: i for i, z in enumerate(QM9_ATOMIC_NUMBERS)}
QM9_NUM_ATOM_TYPES = len(QM9_ATOMIC_NUMBERS)

# Bond type mapping
BOND_TYPES = ["none", "single", "double", "triple", "aromatic"]
BOND_TYPE_TO_IDX = {b: i for i, b in enumerate(BOND_TYPES)}
NUM_BOND_TYPES = len(BOND_TYPES)


class QM9Dataset:
    """
    Wrapper around PyG's QM9 dataset with pre-processing for diffusion.

    Converts raw QM9 data to the format expected by our diffusion model:
    - x: one-hot atom types
    - positions: 3D coordinates
    - edge_index: fully-connected graph edges
    - edge_attr: bond types (inferred from distances)
    """

    def __init__(
        self,
        root: str = "./data/qm9",
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        max_molecules: Optional[int] = None,
        remove_h: bool = False,
    ):
        """
        Args:
            root: data directory
            split: "train", "val", or "test"
            train_ratio: fraction for training
            val_ratio: fraction for validation
            max_molecules: limit dataset size (for debugging)
            remove_h: if True, remove hydrogen atoms
        """
        self.root = root
        self.split = split
        self.remove_h = remove_h

        # Load QM9
        dataset = PyGQM9(root=root)

        # Create train/val/test split
        n = len(dataset)
        indices = torch.randperm(n).tolist()
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        split_indices = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }

        self.indices = split_indices[split]
        if max_molecules is not None:
            self.indices = self.indices[:max_molecules]

        self.dataset = dataset

        # Compute marginals
        self.atom_marginals = self._compute_atom_marginals()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        """Get a single molecule in diffusion-compatible format."""
        data = self.dataset[self.indices[idx]]

        # Convert to our format
        z = data.z  # atomic numbers
        pos = data.pos  # 3D coordinates

        # Filter hydrogens if requested
        if self.remove_h:
            keep = z != 1
            z = z[keep]
            pos = pos[keep]

        # One-hot atom types
        x = torch.zeros(len(z), QM9_NUM_ATOM_TYPES)
        for i, atomic_num in enumerate(z.tolist()):
            if atomic_num in QM9_ATOM_TO_IDX:
                x[i, QM9_ATOM_TO_IDX[atomic_num]] = 1.0
            else:
                # Unknown atom type, map to carbon as fallback
                x[i, QM9_ATOM_TO_IDX[6]] = 1.0

        # Build fully-connected edges
        n = len(z)
        if n > 1:
            row = torch.arange(n).unsqueeze(1).expand(-1, n).reshape(-1)
            col = torch.arange(n).unsqueeze(0).expand(n, -1).reshape(-1)
            edge_index = torch.stack([row, col], dim=0)
            # Remove self-loops
            keep = row != col
            edge_index = edge_index[:, keep]

            # Infer bond types from distances
            edge_attr = self._infer_bonds(pos, edge_index, z)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, NUM_BOND_TYPES)

        return Data(
            x=x,
            positions=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_atoms=n,
        )

    def _infer_bonds(
        self, pos: Tensor, edge_index: Tensor, z: Tensor
    ) -> Tensor:
        """
        Infer bond types from pairwise distances and atom types.

        Uses simple distance-based heuristics with element-specific radii.
        """
        row, col = edge_index[0], edge_index[1]
        dist = (pos[row] - pos[col]).norm(dim=-1)

        # Covalent radii (simplified, in Angstroms)
        radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}

        bond_attr = torch.zeros(edge_index.size(1), NUM_BOND_TYPES)

        for e in range(edge_index.size(1)):
            d = dist[e].item()
            zi = z[row[e]].item()
            zj = z[col[e]].item()

            r_sum = radii.get(zi, 0.7) + radii.get(zj, 0.7)

            # Simple heuristic
            if d < r_sum * 0.7:
                bond_type = "triple"
            elif d < r_sum * 0.9:
                bond_type = "double"
            elif d < r_sum * 1.2:
                bond_type = "single"
            elif d < r_sum * 1.5:
                bond_type = "aromatic"
            else:
                bond_type = "none"

            bond_attr[e, BOND_TYPE_TO_IDX[bond_type]] = 1.0

        return bond_attr

    def _compute_atom_marginals(self) -> Tensor:
        """Compute empirical atom type distribution for discrete diffusion."""
        counts = torch.zeros(QM9_NUM_ATOM_TYPES)

        for idx in self.indices[:1000]:  # Sample for efficiency
            data = self.dataset[idx]
            for atomic_num in data.z.tolist():
                if atomic_num in QM9_ATOM_TO_IDX:
                    counts[QM9_ATOM_TO_IDX[atomic_num]] += 1

        marginals = counts / counts.sum()
        marginals = marginals.clamp(min=1e-5)  # Avoid log(0)
        return marginals / marginals.sum()

    def get_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> DataLoader:
        """Create a DataLoader for this dataset split."""

        class _QM9Iterable(torch.utils.data.IterableDataset):
            def __init__(self, parent):
                self.parent = parent

            def __iter__(self):
                for i in range(len(self.parent)):
                    yield self.parent[i]

        dataset = _QM9Iterable(self)
        # Actually use a regular list-based approach for simplicity
        samples = [self[i] for i in range(len(self))]

        return DataLoader(
            samples,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch):
        """Custom collate function for variable-size molecular graphs."""
        from torch_geometric.data import Batch
        return Batch.from_data_list(batch)
