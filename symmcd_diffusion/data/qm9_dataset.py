"""
QM9 dataset loader for molecular diffusion pre-training.

QM9 dataset: ~130K small organic molecules with up to 9 heavy atoms
(C, N, O, F) plus hydrogen. Each molecule has 3D geometry optimized at
the B3LYP/6-31G(2df,p) level of theory.

We use the PyTorch Geometric QM9 dataset and convert to our format.
"""

import os
import sys
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

# Radii for bond inference
_COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}


class QM9Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping PyG's QM9 for lazy-loading diffusion training.

    Converts raw QM9 data to the format expected by our diffusion model:
    - x: one-hot atom types
    - positions: 3D coordinates
    - edge_index: fully-connected graph edges
    - edge_attr: bond types (inferred from distances)

    Inherits from torch.utils.data.Dataset so DataLoader iterates lazily
    instead of materializing all ~107K samples into memory at once.
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
        super().__init__()
        self.root = root
        self.split = split
        self.remove_h = remove_h

        # Load QM9
        print(f"  Loading QM9 from {root}...", flush=True)
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
        print(f"  {split} split: {len(self.indices)} molecules", flush=True)

        # Pre-cache atom types and positions as tensors to avoid repeated conversions
        print(f"  Pre-caching atom type tensors...", flush=True)
        self._z_cache = [self.dataset[idx].z for idx in self.indices]
        self._pos_cache = [self.dataset[idx].pos for idx in self.indices]
        print(f"  Pre-cache complete ({len(self._z_cache)} entries)", flush=True)

        # Compute marginals
        self.atom_marginals = self._compute_atom_marginals()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        """Get a single molecule in diffusion-compatible format."""
        z = self._z_cache[idx]
        pos = self._pos_cache[idx]

        # Filter hydrogens if requested
        if self.remove_h:
            keep = z != 1
            z = z[keep]
            pos = pos[keep]

        n = len(z)

        # One-hot atom types (vectorized)
        x = torch.zeros(n, QM9_NUM_ATOM_TYPES)
        for i, atomic_num in enumerate(z.tolist()):
            at_idx = QM9_ATOM_TO_IDX.get(atomic_num, QM9_ATOM_TO_IDX[6])
            x[i, at_idx] = 1.0

        # Build fully-connected edges
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

        bond_attr = torch.zeros(edge_index.size(1), NUM_BOND_TYPES)

        # Vectorized bond type assignment
        r_i = torch.tensor([_COVALENT_RADII.get(int(zi), 0.7) for zi in z[row].tolist()])
        r_j = torch.tensor([_COVALENT_RADII.get(int(zj), 0.7) for zj in z[col].tolist()])
        r_sum = r_i + r_j

        # triple: d < 0.7 * r_sum
        triple_mask = dist < r_sum * 0.7
        # double: 0.7*r_sum <= d < 0.9*r_sum
        double_mask = (~triple_mask) & (dist < r_sum * 0.9)
        # single: 0.9*r_sum <= d < 1.2*r_sum
        single_mask = (~triple_mask) & (~double_mask) & (dist < r_sum * 1.2)
        # aromatic: 1.2*r_sum <= d < 1.5*r_sum
        aromatic_mask = (~triple_mask) & (~double_mask) & (~single_mask) & (dist < r_sum * 1.5)
        # none: d >= 1.5*r_sum
        none_mask = ~(triple_mask | double_mask | single_mask | aromatic_mask)

        bond_attr[triple_mask, BOND_TYPE_TO_IDX["triple"]] = 1.0
        bond_attr[double_mask, BOND_TYPE_TO_IDX["double"]] = 1.0
        bond_attr[single_mask, BOND_TYPE_TO_IDX["single"]] = 1.0
        bond_attr[aromatic_mask, BOND_TYPE_TO_IDX["aromatic"]] = 1.0
        bond_attr[none_mask, BOND_TYPE_TO_IDX["none"]] = 1.0

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
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Create a DataLoader for this dataset split.

        Uses lazy iteration via torch.utils.data.Dataset — samples are processed
        on-the-fly by DataLoader workers rather than materialized upfront.

        IMPORTANT: num_workers defaults to 0 because PyG Data objects use
        non-trivial pickling that can cause hangs with multiprocessing.
        For better performance, use InMemoryDataset or pre-process to disk.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            # Pin memory speeds up CPU→GPU transfer when using CUDA
            pin_memory=True,
        )


def _collate_fn(batch):
    """Custom collate function for variable-size molecular graphs."""
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)
