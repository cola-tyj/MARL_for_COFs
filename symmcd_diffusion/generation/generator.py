"""
COF Building Block Generator.

End-to-end generation pipeline using the symmetry-conditioned diffusion model.
Combines diffusion sampling with legality filtering and cjson export.

Usage:
    generator = COFBBGenerator(denoiser, diffusion, config)
    building_blocks = generator.generate(
        point_group="D3h", num_connectors=3,
        func_group="CHO", num_samples=10
    )
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..data.cjson_io import CJSONExporter
from ..filters.legality_filter import FullFilterResult, LegalityFilter
from ..models.conditional_denoiser import SymmetryConditionedDenoiser
from ..models.diffusion_process import DiffusionProcess
from ..symmetry.symmetry_encoder import (
    IDX_TO_POINT_GROUP,
    POINT_GROUP_TO_IDX,
)


@dataclass
class GenerationResult:
    """Result from a generation request."""
    cjson_paths: List[str]
    num_requested: int
    num_generated: int
    filter_stats: Dict[str, Dict[str, int]]
    elapsed_time: float
    success_rate: float


class COFBBGenerator:
    """
    End-to-end COF building block generator.

    Combines:
    1. Diffusion sampling with symmetry conditioning
    2. Five-layer legality filtering
    3. CJSON export for pycofbuilder compatibility
    """

    def __init__(
        self,
        denoiser: SymmetryConditionedDenoiser,
        diffusion: DiffusionProcess,
        atom_marginals: Tensor,
        bond_marginals: Tensor,
        exporter: Optional[CJSONExporter] = None,
        device: str = "cuda",
        max_atoms: int = 80,
        min_atoms: int = 5,
    ):
        self.denoiser = denoiser
        self.diffusion = diffusion
        self.atom_marginals = atom_marginals
        self.bond_marginals = bond_marginals
        self.exporter = exporter or CJSONExporter()
        self.device = device
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms

        self.filter = LegalityFilter()

        # Symmetry to COF type mapping
        self.symmetry_to_cof_type = {
            "C2v": "L2",
            "D3h": "T3",
            "D4h": "S4",
            "D6h": "H6",
        }
        self.cof_type_to_symmetry = {v: k for k, v in self.symmetry_to_cof_type.items()}

        # Functional group mapping
        self.func_group_names = [
            "H", "CHO", "NH2", "COOH", "CN", "OH", "Cl", "Br",
            "CH3", "NO2",
        ]
        self.func_group_to_idx = {fg: i for i, fg in enumerate(self.func_group_names)}

        # Generator stats
        self.generation_count = 0
        self.total_generated = 0

    def generate(
        self,
        symmetry_type: str,
        num_connectors: int,
        func_group: str = "CHO",
        num_samples: int = 10,
        max_retries_per_sample: int = 20,
    ) -> GenerationResult:
        """
        Generate COF building blocks with specified properties.

        Args:
            symmetry_type: COF symmetry ("L2", "T3", "S4", "H6")
            num_connectors: number of connection points (2, 3, 4, 6)
            func_group: functional group at connectors ("CHO", "NH2", etc.)
            num_samples: desired number of valid building blocks
            max_retries_per_sample: max diffusion samples per valid BB

        Returns:
            GenerationResult with paths to saved cjson files
        """
        start_time = time.time()
        self.filter.reset_stats()

        point_group = self.cof_type_to_symmetry.get(symmetry_type, "D3h")
        pg_idx = torch.tensor(
            [POINT_GROUP_TO_IDX[point_group]], device=self.device
        )
        conn_idx = torch.tensor([num_connectors], device=self.device)
        fg_idx = torch.tensor(
            [self.func_group_to_idx.get(func_group, 0)], device=self.device
        )

        cjson_paths = []
        total_attempts = 0
        max_total = num_samples * max_retries_per_sample

        while len(cjson_paths) < num_samples and total_attempts < max_total:
            # Estimate molecule size based on symmetry type
            base_atoms = {
                "L2": 10, "T3": 15, "S4": 20, "H6": 30,
            }.get(symmetry_type, 15)
            num_atoms = np.random.randint(
                max(self.min_atoms, base_atoms - 5),
                min(self.max_atoms, base_atoms + 10),
            )

            # Sample from diffusion model
            try:
                result = self.diffusion.sample(
                    denoiser=self.denoiser,
                    num_atoms=num_atoms,
                    atom_marginals=self.atom_marginals,
                    bond_marginals=self.bond_marginals,
                    condition=None,   # handled via FiLM in denoiser
                    device=self.device,
                )
                total_attempts += 1
            except Exception as e:
                total_attempts += 1
                continue

            # Run legality filter
            filter_result = self.filter.check(
                atom_types=result["atom_types"],
                positions=result["positions"],
                bonds=result.get("bond_types"),
                expected_symmetry=point_group,
                expected_connectors=num_connectors,
                expected_func_group=func_group,
            )

            if filter_result.passed:
                # Export to cjson
                name = f"gen_{symmetry_type}_{func_group}_{len(cjson_paths):03d}"
                cjson_dict = self.exporter.export(
                    atom_types=result["atom_types"],
                    positions=result["positions"],
                    bonds=result.get("bond_types"),
                    name=name,
                    symmetry_type=symmetry_type,
                    connector_type=func_group,
                    edge_index=result.get("edge_index"),
                )

                path = self.exporter.save_to_core(cjson_dict, symmetry_type)
                cjson_paths.append(path)

            self.total_generated += 1

        elapsed = time.time() - start_time
        success_rate = len(cjson_paths) / max(total_attempts, 1)

        return GenerationResult(
            cjson_paths=cjson_paths,
            num_requested=num_samples,
            num_generated=len(cjson_paths),
            filter_stats=self.filter.get_stats(),
            elapsed_time=elapsed,
            success_rate=success_rate,
        )

    def generate_pair(
        self,
        topology: str,
        stacking: str = "AA",
        func_group_a: str = "CHO",
        func_group_b: str = "NH2",
        num_samples: int = 5,
    ) -> List[Tuple[str, str, str]]:
        """
        Generate compatible building block pairs for a COF topology.

        Args:
            topology: COF topology (e.g., "HCB_A", "KGD", "SQL")
            stacking: stacking mode
            func_group_a: functional group for BB-A
            func_group_b: functional group for BB-B
            num_samples: number of pairs to generate

        Returns:
            List of (bb_a_path, bb_b_path, cof_name) tuples
        """
        # Topology → symmetry types
        topology_symmetries = {
            "HCB_A": ("T3", "L2"),
            "HCB": ("T3", "T3"),
            "SQL": ("S4", "S4"),
            "SQL_A": ("S4", "L2"),
            "KGD": ("H6", "T3"),
            "HXL_A": ("H6", "L2"),
            "KGM": ("T3", "L2"),
            "KGM_A": ("T3", "L2"),
            "FXT": ("S4", "S4"),
            "FXT_A": ("S4", "S4"),
            "LON_A": ("S4", "L2"),
            "DIA": ("S4", "S4"),
            "DIA_A": ("S4", "S4"),
            "BOR": ("S4", "S4"),
        }

        sym_a, sym_b = topology_symmetries.get(
            topology, ("T3", "L2")
        )

        # Num connectors from symmetry
        sym_connectors = {"L2": 2, "T3": 3, "S4": 4, "H6": 6}
        conn_a = sym_connectors.get(sym_a, 3)
        conn_b = sym_connectors.get(sym_b, 2)

        # Generate BB-A
        result_a = self.generate(
            symmetry_type=sym_a,
            num_connectors=conn_a,
            func_group=func_group_a,
            num_samples=num_samples,
        )

        # Generate BB-B
        result_b = self.generate(
            symmetry_type=sym_b,
            num_connectors=conn_b,
            func_group=func_group_b,
            num_samples=num_samples,
        )

        # Pair them up
        pairs = []
        for i in range(min(len(result_a.cjson_paths), len(result_b.cjson_paths))):
            cof_name = f"{sym_a}_{func_group_a}-{sym_b}_{func_group_b}-{topology}-{stacking}"
            pairs.append((
                result_a.cjson_paths[i],
                result_b.cjson_paths[i],
                cof_name,
            ))

        return pairs
