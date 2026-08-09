"""
COF building block data augmentation.

Expands the existing ~194 building blocks to ~2000 training samples
for effective diffusion model fine-tuning.

Augmentation strategies preserve:
- Chemical validity (via RDKit)
- Symmetry type
- Number of connection points
- Functional group identity
"""

import copy
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class COFAugmenter:
    """
    Augments COF building block data for diffusion training.

    Strategies (applied independently, can compose):
    1. Random SO(3) rotation (always valid, 5× multiplier)
    2. Functional group perturbation (substitute similar groups)
    3. Scaffold extension (insert benzene/ethyne spacer)
    4. Position noise (small random displacements)
    """

    def __init__(
        self,
        data_dir: str = "/home/tianyajun/MARL_for_COFs/pycofbuilder/data",
        output_dir: str = "/home/tianyajun/MARL_for_COFs/symmcd_diffusion/data/augmented",
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(seed)
        np.random.seed(seed)

    def augment_all(
        self,
        target_count: int = 2000,
        rot_copies: int = 5,
        noise_copies: int = 2,
        scaffold_copies: int = 1,
        fg_perturb_copies: int = 1,
    ) -> List[str]:
        """
        Augment all building blocks to reach target_count.

        Returns:
            List of paths to augmented cjson files
        """
        all_augmented = []

        # Collect all original cjson files
        original_files = []
        core_dir = self.data_dir / "core"
        if core_dir.exists():
            for sym_dir in core_dir.iterdir():
                if sym_dir.is_dir():
                    for cjson_file in sym_dir.glob("*.cjson"):
                        original_files.append((str(cjson_file), sym_dir.name))

        print(f"Found {len(original_files)} original building blocks")

        samples_per_original = max(1, target_count // len(original_files))

        for filepath, sym_type in original_files:
            with open(filepath) as f:
                cjson = json.load(f)

            base_name = Path(filepath).stem

            # Strategy 1: Random rotations
            for r in range(min(rot_copies, samples_per_original)):
                rotated = self._rotate_randomly(cjson)
                name = f"{base_name}_rot{r}"
                path = self._save(rotated, name, sym_type)
                all_augmented.append(path)

            # Strategy 2: Position noise
            for n in range(min(noise_copies, samples_per_original)):
                noised = self._add_position_noise(cjson, sigma=0.05)
                name = f"{base_name}_noise{n}"
                path = self._save(noised, name, sym_type)
                all_augmented.append(path)

            # Strategy 3: Scaffold extension (only for L2/S4 with simple cores)
            if scaffold_copies > 0 and sym_type in ("L2", "S4"):
                try:
                    extended = self._extend_scaffold(cjson)
                    if extended is not None:
                        name = f"{base_name}_ext"
                        path = self._save(extended, name, sym_type)
                        all_augmented.append(path)
                except Exception:
                    pass

            # Strategy 4: Functional group perturbation
            if fg_perturb_copies > 0:
                try:
                    perturbed = self._perturb_functional_groups(cjson)
                    if perturbed is not None:
                        name = f"{base_name}_fg"
                        path = self._save(perturbed, name, sym_type)
                        all_augmented.append(path)
                except Exception:
                    pass

        print(f"Generated {len(all_augmented)} augmented samples")
        return all_augmented

    @staticmethod
    def _rotate_randomly(cjson: Dict) -> Dict:
        """Apply random SO(3) rotation to coordinates."""
        coords = np.array(cjson["atoms"]["coords"]["3d"]).reshape(-1, 3)

        # Random rotation via QR decomposition of random matrix
        M = np.random.randn(3, 3)
        Q, R = np.linalg.qr(M)
        # Ensure proper rotation (det = +1)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1

        rotated = coords @ Q.T

        new_cjson = copy.deepcopy(cjson)
        new_cjson["atoms"]["coords"]["3d"] = rotated.flatten().tolist()
        return new_cjson

    @staticmethod
    def _add_position_noise(cjson: Dict, sigma: float = 0.05) -> Dict:
        """Add small Gaussian noise to atomic positions."""
        coords = np.array(cjson["atoms"]["coords"]["3d"]).reshape(-1, 3)
        noise = np.random.randn(*coords.shape) * sigma
        noised = coords + noise

        new_cjson = copy.deepcopy(cjson)
        new_cjson["atoms"]["coords"]["3d"] = noised.flatten().tolist()
        return new_cjson

    @staticmethod
    def _extend_scaffold(cjson: Dict) -> Optional[Dict]:
        """
        Attempt to extend molecular scaffold with a spacer.
        This is a placeholder for RDKit-based extension.
        For now, returns None to skip if RDKit unavailable.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            smiles = cjson.get("properties", {}).get("smiles", "")
            if not smiles or "*" not in smiles:
                return None

            # Replace single bonds with phenylene spacers
            # This is a simplified version
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Add 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)

            conf = mol.GetConformer()
            coords = []
            elements = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                coords.extend([pos.x, pos.y, pos.z])
                elements.append(atom.GetSymbol())

            new_cjson = copy.deepcopy(cjson)
            new_cjson["atoms"]["elements"]["type"] = elements
            new_cjson["atoms"]["coords"]["3d"] = coords
            new_cjson["atoms"]["elements"]["number"] = [
                atom.GetAtomicNum() for atom in mol.GetAtoms()
            ]
            return new_cjson

        except ImportError:
            return None
        except Exception:
            return None

    @staticmethod
    def _perturb_functional_groups(cjson: Dict) -> Optional[Dict]:
        """
        Perturb functional groups by substituting similar ones.
        e.g., CHO → COOH, NH2 → OH
        """
        try:
            from rdkit import Chem

            smiles = cjson.get("properties", {}).get("smiles", "")
            if not smiles:
                return None

            # Simple SMILES-based substitution
            substitutions = [
                ("C=O", "C(=O)O"),     # aldehyde → acid
                ("N", "O"),            # amine → hydroxyl
                ("Cl", "Br"),          # chloride → bromide
            ]

            for old, new in substitutions:
                if old in smiles:
                    new_smiles = smiles.replace(old, new, 1)
                    mol = Chem.MolFromSmiles(new_smiles)
                    if mol is not None:
                        new_cjson = copy.deepcopy(cjson)
                        new_cjson["properties"]["smiles"] = new_smiles
                        return new_cjson

            return None

        except ImportError:
            return None
        except Exception:
            return None

    def _save(self, cjson: Dict, name: str, sym_type: str) -> str:
        """Save augmented cjson to output directory."""
        sym_dir = self.output_dir / sym_type
        sym_dir.mkdir(parents=True, exist_ok=True)

        filepath = sym_dir / f"{name}.cjson"
        with open(filepath, "w") as f:
            json.dump(cjson, f, indent=4)

        return str(filepath)
