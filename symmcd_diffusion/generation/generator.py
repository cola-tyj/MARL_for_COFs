"""
COF Core Generator (Phase 3).

Generates molecular Cores using the symmetry-conditioned diffusion model,
then optionally assembles complete Building Blocks via pycofbuilder.

Pipeline:
  Diffusion Model → Core (Q+R points) → [pycofbuilder → Complete BB]

Usage:
    generator = CoreGenerator(denoiser, diffusion, atom_marginals, bond_marginals)
    cores = generator.generate(symmetry_type="L2", num_samples=10)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ..data.cjson_io import CJSONExporter
from ..models.conditional_denoiser import SymmetryConditionedDenoiser
from ..models.diffusion_process import DiffusionProcess
from ..symmetry.symmetry_encoder import IDX_TO_POINT_GROUP, POINT_GROUP_TO_IDX


# ── Symmetry specifications ────────────────────────────────────────────────

SYMMETRY_SPEC = {
    "L2": {"point_group": "C2v", "q_count": 2, "r_count_range": (4, 8)},
    "T3": {"point_group": "D3h", "q_count": 3, "r_count_range": (3, 9)},
    "S4": {"point_group": "D4h", "q_count": 4, "r_count_range": (4, 8)},
    "H6": {"point_group": "D6h", "q_count": 6, "r_count_range": (6, 12)},
    "D4": {"point_group": "D4h", "q_count": 4, "r_count_range": (4, 8)},
    "R4": {"point_group": "D4h", "q_count": 4, "r_count_range": (4, 8)},
}

# Q atom index in Core vocabulary (12 types: H,C,N,O,F,S,B,Cu,Zn,Co,Q,R)
Q_ATOM_IDX = 10
R_ATOM_IDX = 11


@dataclass
class GenerationResult:
    """Result from a generation request."""
    cjson_paths: List[str] = field(default_factory=list)
    num_requested: int = 0
    num_generated: int = 0
    num_valid: int = 0
    total_attempts: int = 0
    elapsed_time: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.num_valid / max(self.num_requested, 1)


class CoreGenerator:
    """
    Symmetry-conditioned Core generator.

    Generates molecular Cores from diffusion sampling, with optional
    validation and cjson export. Complete BB assembly is handled by
    pycofbuilder (deterministic chemistry, not learned).
    """

    def __init__(
        self,
        denoiser: SymmetryConditionedDenoiser,
        diffusion: DiffusionProcess,
        atom_marginals: Tensor,
        bond_marginals: Tensor,
        exporter: Optional[CJSONExporter] = None,
        device: str = "cuda",
        max_atoms: int = 50,
        min_atoms: int = 8,
        ddim_steps: int = 0,
    ):
        self.denoiser = denoiser
        self.diffusion = diffusion
        self.atom_marginals = atom_marginals
        self.bond_marginals = bond_marginals
        self.exporter = exporter or CJSONExporter()
        self.device = device
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms
        self.ddim_steps = ddim_steps  # 0=ancestral (1000), 50=DDIM
        self.generation_count = 0

    def generate(
        self,
        symmetry_type: str = "L2",
        num_samples: int = 10,
        max_attempts_per_sample: int = 20,
        validate: bool = True,
        output_dir: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate Cores with specified symmetry.

        Args:
            symmetry_type: COF symmetry (L2, T3, S4, H6, D4, R4)
            num_samples: desired number of valid Cores
            max_attempts_per_sample: max diffusion samples per valid Core
            validate: if True, apply geometric validation
            output_dir: directory to save cjson files

        Returns:
            GenerationResult with paths and statistics
        """
        if symmetry_type not in SYMMETRY_SPEC:
            raise ValueError(f"Unknown symmetry: {symmetry_type}. "
                             f"Choose from {list(SYMMETRY_SPEC.keys())}")

        spec = SYMMETRY_SPEC[symmetry_type]
        point_group = spec["point_group"]
        pg_idx = POINT_GROUP_TO_IDX.get(point_group, 0)
        expected_q = spec["q_count"]

        start_time = time.time()
        result = GenerationResult(num_requested=num_samples)
        max_total = num_samples * max_attempts_per_sample

        # Condition tensor (batch_size=1 for sampling)
        cond = self.denoiser.encode_condition(
            torch.tensor([pg_idx], device=self.device)
        )

        while result.num_valid < num_samples and result.total_attempts < max_total:
            # Estimate Core size based on symmetry
            num_atoms = self._estimate_size(symmetry_type)

            # Sample from diffusion model (use DDIM if ddim_steps specified)
            try:
                if getattr(self, 'ddim_steps', None) and self.ddim_steps > 0:
                    sample = self.diffusion.sample_ddim(
                        denoiser=self.denoiser.denoiser,
                        num_atoms=num_atoms,
                        atom_marginals=self.atom_marginals,
                        bond_marginals=self.bond_marginals,
                        condition=cond,
                        device=self.device,
                        ddim_steps=self.ddim_steps,
                    )
                else:
                    sample = self.diffusion.sample(
                        denoiser=self.denoiser.denoiser,
                        num_atoms=num_atoms,
                        atom_marginals=self.atom_marginals,
                        bond_marginals=self.bond_marginals,
                        condition=cond,
                        device=self.device,
                    )
                result.total_attempts += 1
            except Exception:
                result.total_attempts += 1
                continue

            # ── Validation ──────────────────────────────────────────────
            if validate:
                if not self._validate_core(sample["atom_types"], expected_q):
                    continue

            # ── Export ─────────────────────────────────────────────────
            name = f"gen_{symmetry_type}_{result.num_valid:03d}"
            try:
                path = self.exporter.export(
                    atom_types=sample["atom_types"],
                    positions=sample["positions"],
                    bonds=sample.get("bond_types"),
                    name=name,
                    symmetry_type=symmetry_type,
                    edge_index=sample.get("edge_index"),
                    output_dir=output_dir,
                )
                result.cjson_paths.append(path)
                result.num_valid += 1
                result.num_generated += 1
            except Exception:
                continue

        result.elapsed_time = time.time() - start_time
        self.generation_count += 1
        return result

    def _validate_core(self, atom_types: Tensor, expected_q: int) -> bool:
        """Quick geometric validation of a generated Core."""
        # Check Q atom count
        if atom_types.dim() == 2:  # one-hot
            atom_idx = atom_types.argmax(dim=-1)
        else:
            atom_idx = atom_types
        q_count = (atom_idx == Q_ATOM_IDX).sum().item()

        # Allow ±1 tolerance (diffusion may not produce exact counts)
        if q_count < expected_q - 1 or q_count > expected_q + 1:
            return False

        # Check R atom count (should be reasonable)
        r_count = (atom_idx == R_ATOM_IDX).sum().item()
        if r_count < 2:
            return False

        return True

    def _estimate_size(self, symmetry_type: str) -> int:
        """Estimate reasonable atom count for a Core of given symmetry."""
        base = {"L2": 20, "T3": 15, "S4": 25, "H6": 35, "D4": 20, "R4": 20}
        n = base.get(symmetry_type, 20)
        return np.random.randint(
            max(self.min_atoms, n - 5),
            min(self.max_atoms, n + 10),
        )

    def assemble_bb(
        self,
        core_path: str,
        connector: str = "COOH",
        func_group: str = "H",
    ) -> Optional[str]:
        """
        Assemble a complete Building Block from a generated Core.

        Uses pycofbuilder to attach connector + functional group.
        """
        try:
            import json, os
            from pycofbuilder.building_block import BuildingBlock

            with open(core_path) as f:
                core_data = json.load(f)

            symmetry = core_data.get("properties", {}).get("symmetry_type", "L2")
            core_name = os.path.splitext(os.path.basename(core_path))[0]

            # Build name: Symmetry_CoreName_Connector_FG
            bb_name = f"{symmetry}_{core_name}_{connector}_{func_group}"
            bb = BuildingBlock()
            bb.from_name(bb_name)

            # Save assembled BB
            bb_path = core_path.replace(".cjson", f"_{connector}_{func_group}.cjson")
            bb.save(bb_path)
            return bb_path
        except Exception:
            return None
