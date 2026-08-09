"""
Diffusion Generator Wrapper for MARL environment.

Provides caching and batch generation to efficiently integrate
the diffusion model into the MARL training loop.

Key optimization: pre-generation and caching of building blocks
for frequently requested (symmetry, connector, func_group) combinations.
"""

from typing import Dict, List, Optional, Tuple

import torch

from ..generation.generator import COFBBGenerator, GenerationResult


class DiffusionGeneratorWrapper:
    """
    Caching wrapper around COFBBGenerator for MARL integration.

    Since diffusion sampling is expensive (~seconds per molecule),
    we:
    1. Cache generated BBs by (symmetry, connectors, func_group)
    2. Pre-generate pools for common specifications
    3. Allow fallback to fixed vocabulary if generation fails
    """

    def __init__(
        self,
        generator: COFBBGenerator,
        cache_size: int = 200,
        pre_generate: bool = True,
    ):
        self.generator = generator
        self.cache_size = cache_size

        # Cache: (symmetry, conn_count, func_group) → list of cjson paths
        self.cache: Dict[Tuple[str, int, str], List[str]] = {}

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_generated = 0

        # Pre-generate pools for common specs
        if pre_generate:
            self._pre_generate_common()

    def _pre_generate_common(self):
        """Pre-generate building blocks for common specifications."""
        common_specs = [
            ("T3", 3, "CHO"), ("T3", 3, "NH2"),
            ("L2", 2, "CHO"), ("L2", 2, "NH2"),
            ("S4", 4, "CHO"), ("S4", 4, "COOH"),
            ("H6", 6, "CHO"), ("H6", 6, "NH2"),
        ]

        for sym, conn, fg in common_specs:
            try:
                result = self.generator.generate(
                    symmetry_type=sym,
                    num_connectors=conn,
                    func_group=fg,
                    num_samples=max(5, self.cache_size // len(common_specs)),
                    max_retries_per_sample=10,
                )
                self.cache[(sym, conn, fg)] = result.cjson_paths
                self.total_generated += result.num_generated
            except Exception as e:
                print(f"Pre-generation failed for {(sym, conn, fg)}: {e}")
                self.cache[(sym, conn, fg)] = []

    def get_or_generate(
        self,
        symmetry_type: str,
        num_connectors: int,
        func_group: str,
        num_samples: int = 5,
    ) -> List[str]:
        """
        Get building block cjson paths, generating if needed.

        Args:
            symmetry_type: "L2", "T3", "S4", "H6"
            num_connectors: 2, 3, 4, 6
            func_group: "CHO", "NH2", etc.
            num_samples: number of BBs to return

        Returns:
            List of cjson file paths
        """
        key = (symmetry_type, num_connectors, func_group)

        # Check cache
        if key in self.cache and len(self.cache[key]) >= num_samples:
            self.cache_hits += 1
            return self.cache[key][:num_samples]

        self.cache_misses += 1

        # Generate new BBs
        result = self.generator.generate(
            symmetry_type=symmetry_type,
            num_connectors=num_connectors,
            func_group=func_group,
            num_samples=num_samples,
            max_retries_per_sample=15,
        )

        # Update cache
        paths = result.cjson_paths
        if key not in self.cache:
            self.cache[key] = []
        self.cache[key].extend(paths)

        # Limit cache size
        if len(self.cache[key]) > self.cache_size:
            self.cache[key] = self.cache[key][-self.cache_size:]

        self.total_generated += result.num_generated
        return paths[:num_samples]

    def generate_pair_for_topology(
        self,
        topology: str,
        stacking: str = "AA",
        func_group_a: str = "CHO",
        func_group_b: str = "NH2",
    ) -> List[Tuple[str, str]]:
        """
        Generate compatible BB pairs for a COF topology.

        Returns:
            List of (bb_a_path, bb_b_path) tuples
        """
        pairs = self.generator.generate_pair(
            topology=topology,
            stacking=stacking,
            func_group_a=func_group_a,
            func_group_b=func_group_b,
            num_samples=3,
        )
        return [(pa, pb) for pa, pb, _ in pairs]

    def get_stats(self) -> Dict:
        """Get wrapper statistics."""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": sum(len(v) for v in self.cache.values()),
            "total_generated": self.total_generated,
            "hit_rate": (
                self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            ),
        }
