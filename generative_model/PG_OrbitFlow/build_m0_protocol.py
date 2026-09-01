"""Freeze the M0 oracle orbit-space reconstruction protocol."""

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
            "generative_model/PG_OrbitFlow/configs/c0_local_recovery_v2_pg_balanced.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_path = args.base_protocol.resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base["schema_version"] != "pg-orbitflow-c0-protocol-v2":
        raise ValueError("M0 base must be the formal C0-v2 protocol")
    package_dir = Path(base["package_dir"])
    protocol = {
        "schema_version": "pg-orbitflow-m0-oracle-protocol-v1",
        "scientific_question": (
            "Can an exact group-lifted orbit-space solver reconstruct the frozen "
            "Tier-4 conformations from oracle internal-coordinate orbits without "
            "using target Cartesian coordinates for initialization?"
        ),
        "base_c0_protocol": {"path": str(base_path), "sha256": _sha256(base_path)},
        "package_dir": str(package_dir.resolve()),
        "canonical_manifest_sha256": _sha256(package_dir / "manifest.json"),
        "seed": int(base["seed"]) + 500,
        "panel": base["panel"],
        "dtype": "float64",
        "oracle_targets": {
            "source": "symmetric canonical target XYZ",
            "orbit_broadcast": True,
            "features": [
                "bond_length",
                "angle_cosine",
                "torsion_sine_cosine",
                "graph_distance_le_3_pair_distance",
                "ring_bond_length",
                "chirality",
            ],
            "target_cartesian_coordinate_loss_used": False,
            "target_cartesian_initialization_used": False,
        },
        "initialization": {
            "type": "deterministic independent Gaussian orbit parameters",
            "starts_per_molecule": 16,
            "coordinate_standard_deviation_angstrom": 2.0,
            "molecule_specific_seed_stride": 1009,
        },
        "optimization": {
            "adam_steps": 1200,
            "adam_learning_rate": 0.03,
            "gradient_clip_norm": 100.0,
            "lbfgs_max_iterations": 200,
            "lbfgs_learning_rate": 0.8,
            "lbfgs_history_size": 50,
            "line_search": "strong_wolfe",
            "select": "lowest final oracle energy across deterministic starts",
        },
        "energy_weights": {
            "bond": 10.0,
            "angle": 2.0,
            "torsion": 1.0,
            "local_pair": 2.0,
            "ring": 10.0,
            "chirality": 1.0,
            "collision": 2.0,
        },
        "gate": {
            "roundtrip_max_abs_error_max_angstrom": 1e-5,
            "bond_mae_max_angstrom": 1e-3,
            "angle_mae_max_degrees": 0.2,
            "torsion_circular_mae_max_degrees": 2.0,
            "ring_closure_mae_max_angstrom": 1e-3,
            "kabsch_rmsd_max_angstrom": 0.10,
            "collision_free_fraction_min": 1.0,
            "max_operation_atom_error_max_angstrom": 1e-6,
            "all_optimization_values_finite": True,
        },
        "isolation": {
            "neural_network_used": False,
            "m0_checkpoint_used": False,
            "midi_semlaflow_uae_or_etflow_used": False,
            "etkdg_used": False,
            "mds_used": False,
            "nerf_used": False,
            "posthoc_hard_projection_used": False,
            "f02_used": False,
            "canonical_v2_modified": False,
            "iid_validation_test_or_core_ood_used": False,
        },
        "progression": {
            "if_passes": "freeze M0 solver and implement M1 learned bond/angle orbit heads",
            "if_fails": "diagnose exact lifting versus non-convex reconstruction; do not train M1",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_PG_ORBITFLOW_M0_PROTOCOL {args.output.resolve()}")


if __name__ == "__main__":
    main()
