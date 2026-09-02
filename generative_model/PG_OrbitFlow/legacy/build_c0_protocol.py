"""Freeze the Tier-4 multi-noise local-geometry recovery curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-protocol",
        type=Path,
        default=Path(
            "generative_model/PG_OrbitFlow/configs/overfit_protocol_h1_harmonic.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_path = args.base_protocol.resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base["prior"]["type"] != "graph_harmonic":
        raise ValueError("C0 base must use graph_harmonic raw prior")
    if base["transport"]["type"] != "cn_phase_aligned":
        raise ValueError("C0 base must use the H1 single-phase transport")
    tier = base["tiers"]["4"]
    protocol = {
        "schema_version": "pg-orbitflow-c0-protocol-v2",
        "scientific_question": (
            "Can the unchanged Cartesian PG-OrbitFlow backbone first learn strict "
            "local chemical-geometry recovery under a multi-noise curriculum?"
        ),
        "base_h1_protocol": {
            "path": str(base_path),
            "sha256": _sha256(base_path),
        },
        "package_dir": base["package_dir"],
        "canonical_manifest_sha256": base["canonical_manifest_sha256"],
        "seed": base["seed"],
        "model": base["model"],
        "data": base["data"],
        "panel": {
            "molecule_count": tier["molecule_count"],
            "records": tier["records"],
        },
        "training": {
            **tier["training"],
            "optimizer_steps": tier["training"]["additional_optimizer_steps"],
            "short_rollout_steps": 4,
        },
        "curriculum": {
            "period_exposures": 10,
            "slots": ["small", "small", "small", "small", "medium", "medium", "medium", "large", "large", "harmonic"],
            "small_sigma_angstrom": [0.05, 0.15],
            "medium_sigma_angstrom": [0.15, 0.40],
            "large_sigma_angstrom": [0.40, 0.98],
            "harmonic_prior_fraction": 0.10,
            "balance_scope": "independent 10-slot cursor within each point group",
            "point_group_curriculum_independent": True,
            "local_time_definition": "clip(1-sigma_angstrom,0.02,0.95)",
            "noise_projection": "Reynolds projection before training; not posthoc output projection",
        },
        "objective": {
            "flow_weight": 1.0,
            "bond_weight": 1.0,
            "angle_weight": 0.5,
            "torsion_weight": 0.25,
            "local_pair_weight": 0.25,
            "ring_weight": 0.5,
            "chirality_weight": 0.25,
            "overlap_weight": 0.1,
            "symmetry_weight": 1.0,
            "short_rollout_weight": 0.25,
            "early_local_boost": 2.0,
            "endpoint_weight": 0.0,
        },
        "prior": base["prior"],
        "transport": base["transport"],
        "local_evaluation": {
            "sigma_angstrom": 0.10,
            "seeds_per_molecule": 8,
        },
        "local_gate": {
            "endpoint_rmsd_mean_max_angstrom": 0.10,
            "bond_length_mae_mean_max_angstrom": 0.03,
            "angle_mae_mean_max_degrees": 5.0,
            "chirality_preserved_fraction_min": 1.0,
            "collision_free_fraction_min": 1.0,
            "max_operation_atom_error_max_angstrom": 1e-4,
        },
        "original_tier4_evaluation": tier["evaluation"],
        "original_tier4_gate": tier["gate"],
        "progression": {
            "local_gate_controls_next_step": True,
            "original_raw_gate_changed": False,
            "if_local_passes": "advance to a separately frozen full-prior curriculum gate",
            "if_local_fails": "stop Cartesian backbone and implement internal-coordinate decoder",
            "tier16_or_tier32_allowed": False,
        },
        "execution_revision": {
            "supersedes_protocol": "c0_local_recovery_v1.json",
            "scientific_objective_changed": False,
            "reason": (
                "v1 used one global slot cursor correlated with alternating C2/C3 "
                "exposure, assigning every harmonic slot to C3"
            ),
            "sole_change": "advance the same frozen 10-slot curriculum independently within C2 and C3",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_PG_ORBITFLOW_C0_PROTOCOL {args.output.resolve()}")


if __name__ == "__main__":
    main()
