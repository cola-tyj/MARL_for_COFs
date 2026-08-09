"""
CJSON I/O: Convert between diffusion model output and pycofbuilder cjson format.

The cjson (Chemical JSON) format is used by pycofbuilder to represent
molecular building blocks. This module converts:
- Diffusion output (atom types, positions, bonds) → cjson file
- cjson file → PyG Data for diffusion training
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data


# Atom type mapping for extended COF vocabulary
# 0:H, 1:C, 2:N, 3:O, 4:F, 5:Cl, 6:Br, 7:S, 8:Q(connector), 9:X(placeholder)
IDX_TO_SYMBOL = {
    0: "H", 1: "C", 2: "N", 3: "O", 4: "F",
    5: "Cl", 6: "Br", 7: "S", 8: "Q", 9: "X",
}
IDX_TO_ATOMIC_NUM = {
    0: 1, 1: 6, 2: 7, 3: 8, 4: 9,
    5: 17, 6: 35, 7: 16, 8: 0, 9: 0,  # Q/X = dummy (0)
}
SYMBOL_TO_IDX = {v: k for k, v in IDX_TO_SYMBOL.items()}

# Bond type mapping
BOND_IDX_TO_TYPE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}  # none, single, double, triple, aromatic


class CJSONExporter:
    """
    Export diffusion-generated molecules to pycofbuilder-compatible cjson files.
    """

    def __init__(self, output_dir: str = "/home/tianyajun/MARL_for_COFs/data"):
        self.output_dir = Path(output_dir)

    def export(
        self,
        atom_types: Tensor,
        positions: Tensor,
        bonds: Optional[Tensor],
        name: str,
        symmetry_type: str,
        connector_type: str = "CHO",
        functional_groups: Optional[List[str]] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Dict:
        """
        Convert diffusion output to cjson dictionary.

        Args:
            atom_types: (N,) or (N, num_types) atom type indices or one-hot
            positions: (N, 3) cartesian coordinates
            bonds: (E,) or (E, num_bond_types) bond indices or one-hot
            name: molecule name (e.g., "gen_D3h_001")
            symmetry_type: COF symmetry type (L2, T3, S4, H6)
            connector_type: functional group type at connectors
            functional_groups: list of functional group names
            edge_index: (2, E) edge indices

        Returns:
            cjson_dict following the ChemJSON format
        """
        # Convert to hard assignments
        if atom_types.dim() == 2:
            atom_idx = atom_types.argmax(dim=-1).cpu().numpy()
        else:
            atom_idx = atom_types.cpu().numpy()

        pos = positions.cpu().numpy()

        # Build atom type and coordinate lists
        elements = [IDX_TO_SYMBOL.get(a, "C") for a in atom_idx]
        atomic_numbers = [IDX_TO_ATOMIC_NUM.get(a, 6) for a in atom_idx]

        # Flatten coordinates to 1D list (x1,y1,z1, x2,y2,z2, ...)
        coords_3d = pos.flatten().tolist()

        # Generate SMILES via RDKit if available
        smiles = ""
        xsmiles = ""
        xsmiles_label = ""
        try:
            from rdkit import Chem
            mol = self._build_rdkit_mol(atom_idx, pos, bonds, edge_index)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                # Convert to xsmiles (replace * with [*])
                xsmiles = smiles.replace("*", "[*]")
        except Exception:
            smiles = f"[generated_{symmetry_type}]"
            xsmiles = smiles

        # Build cjson dict
        cjson = {
            "chemical json": 1,
            "name": name,
            "formula": self._compute_formula(elements),
            "atoms": {
                "elements": {
                    "type": elements,
                    "number": atomic_numbers,
                },
                "coords": {
                    "3d": coords_3d,
                },
            },
            "properties": {
                "smiles": smiles,
                "code": name.split("_")[0] if "_" in name else name,
                "xsmiles": xsmiles,
                "xsmiles_label": xsmiles_label,
                "symmetry_type": symmetry_type,
                "connector_type": connector_type,
            },
        }

        if functional_groups:
            cjson["properties"]["functional_groups"] = functional_groups

        return cjson

    def save_to_core(
        self, cjson_dict: Dict, symmetry_type: str
    ) -> str:
        """
        Save cjson to pycofbuilder core data directory.

        Args:
            cjson_dict: cjson dictionary from export()
            symmetry_type: e.g., "T3", "L2", "S4", "H6"

        Returns:
            filepath of saved file
        """
        core_dir = self.output_dir / "core" / symmetry_type
        core_dir.mkdir(parents=True, exist_ok=True)

        name = cjson_dict["name"]
        filepath = core_dir / f"{name}.cjson"

        with open(filepath, "w") as f:
            json.dump(cjson_dict, f, indent=4)

        return str(filepath)

    def save_to_all(self, cjson_dict: Dict) -> str:
        """Save cjson to the shared data/all/ directory."""
        all_dir = self.output_dir / "all"
        all_dir.mkdir(parents=True, exist_ok=True)

        name = cjson_dict["name"]
        filepath = all_dir / f"{name}.cjson"

        with open(filepath, "w") as f:
            json.dump(cjson_dict, f, indent=4)

        return str(filepath)

    def _build_rdkit_mol(
        self,
        atom_idx: np.ndarray,
        pos: np.ndarray,
        bonds: Optional[Tensor],
        edge_index: Optional[Tensor],
    ):
        """Build RDKit molecule from atom types, positions, and bonds."""
        try:
            from rdkit import Chem
        except ImportError:
            return None

        mol = Chem.RWMol()
        idx_to_mol = {}

        for i, a in enumerate(atom_idx):
            symbol = IDX_TO_SYMBOL.get(a, "C")
            if symbol in ("Q", "X"):
                atom = Chem.Atom(0)  # dummy atom
                atom.SetProp("atomLabel", symbol)
            else:
                atom = Chem.Atom(symbol)
            idx = mol.AddAtom(atom)
            idx_to_mol[i] = idx

        # Add bonds from edge_index or distance heuristics
        if edge_index is not None and bonds is not None:
            ei = edge_index.cpu().numpy()
            if bonds.dim() == 2:
                bond_idx = bonds.argmax(dim=-1).cpu().numpy()
            else:
                bond_idx = bonds.cpu().numpy()

            for e in range(ei.shape[1]):
                i, j = ei[0, e], ei[1, e]
                if i < j and i in idx_to_mol and j in idx_to_mol:
                    bt = bond_idx[e] if e < len(bond_idx) else 1
                    rdkit_bond = {
                        0: None,
                        1: Chem.BondType.SINGLE,
                        2: Chem.BondType.DOUBLE,
                        3: Chem.BondType.TRIPLE,
                        4: Chem.BondType.AROMATIC,
                    }.get(bt, Chem.BondType.SINGLE)
                    if rdkit_bond is not None:
                        mol.AddBond(idx_to_mol[i], idx_to_mol[j], rdkit_bond)
        else:
            # Distance-based bond inference
            n = len(atom_idx)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(pos[i] - pos[j])
                    zi = IDX_TO_ATOMIC_NUM.get(atom_idx[i], 6)
                    zj = IDX_TO_ATOMIC_NUM.get(atom_idx[j], 6)
                    if zi > 0 and zj > 0 and dist < 2.0:
                        try:
                            r_i = Chem.GetPeriodicTable().GetRcovalent(zi)
                            r_j = Chem.GetPeriodicTable().GetRcovalent(zj)
                            if dist < (r_i + r_j) * 1.2:
                                mol.AddBond(idx_to_mol[i], idx_to_mol[j],
                                          Chem.BondType.SINGLE)
                        except Exception:
                            pass

        return mol.GetMol()

    @staticmethod
    def _compute_formula(elements: List[str]) -> str:
        """Compute molecular formula from element list."""
        from collections import Counter
        counts = Counter(elements)
        formula_parts = []
        # Standard order: C, H, then alphabetical
        for elem in ["C", "H"]:
            if elem in counts:
                n = counts.pop(elem)
                formula_parts.append(f"{elem}{n if n > 1 else ''}")
        for elem in sorted(counts.keys()):
            n = counts[elem]
            formula_parts.append(f"{elem}{n if n > 1 else ''}")
        return "".join(formula_parts)


class CJSONImporter:
    """
    Import cjson files to PyG Data format for diffusion training.
    """

    def __init__(self, data_dir: str = "/home/tianyajun/MARL_for_COFs/pycofbuilder/data"):
        self.data_dir = Path(data_dir)

    def load_building_block(self, cjson_path: str) -> Data:
        """
        Load a single cjson file and convert to PyG Data.

        Args:
            cjson_path: path to .cjson file

        Returns:
            PyG Data with x (atom types), positions, edge_index, edge_attr
        """
        with open(cjson_path) as f:
            cjson = json.load(f)

        elements = cjson["atoms"]["elements"]["type"]
        coords_flat = cjson["atoms"]["coords"]["3d"]

        # Convert to tensors
        atom_types = torch.zeros(len(elements), len(IDX_TO_SYMBOL))
        for i, elem in enumerate(elements):
            idx = SYMBOL_TO_IDX.get(elem, SYMBOL_TO_IDX["C"])
            atom_types[i, idx] = 1.0

        n = len(elements)
        positions = torch.tensor(coords_flat).reshape(n, 3).float()

        # Build edges
        if n > 1:
            row = torch.arange(n).unsqueeze(1).expand(-1, n).reshape(-1)
            col = torch.arange(n).unsqueeze(0).expand(n, -1).reshape(-1)
            keep = row != col
            edge_index = torch.stack([row[keep], col[keep]], dim=0)

            # Infer bonds
            edge_attr = torch.zeros(edge_index.size(1), 5)  # 5 bond types
            for e in range(edge_index.size(1)):
                i, j = edge_index[0, e].item(), edge_index[1, e].item()
                dist = (positions[i] - positions[j]).norm().item()
                if dist < 1.2:
                    edge_attr[e, 2] = 1.0  # double
                elif dist < 1.6:
                    edge_attr[e, 1] = 1.0  # single
                elif dist < 2.0:
                    edge_attr[e, 4] = 1.0  # aromatic
                else:
                    edge_attr[e, 0] = 1.0  # none
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, 5)

        # Extract properties
        symmetry_type = cjson.get("properties", {}).get("symmetry_type", "unknown")
        connector_type = cjson.get("properties", {}).get("connector_type", "unknown")
        smiles = cjson.get("properties", {}).get("smiles", "")

        return Data(
            x=atom_types,
            positions=positions,
            edge_index=edge_index,
            edge_attr=edge_attr,
            symmetry_type=symmetry_type,
            connector_type=connector_type,
            smiles=smiles,
            num_atoms=n,
        )

    def load_all_cores(self) -> List[Data]:
        """Load all core building blocks from pycofbuilder data."""
        cores = []
        core_dir = self.data_dir / "core"
        if not core_dir.exists():
            return cores

        for sym_dir in core_dir.iterdir():
            if sym_dir.is_dir():
                for cjson_file in sym_dir.glob("*.cjson"):
                    try:
                        data = self.load_building_block(str(cjson_file))
                        data.symmetry_type = sym_dir.name
                        cores.append(data)
                    except Exception as e:
                        print(f"Failed to load {cjson_file}: {e}")

        return cores
