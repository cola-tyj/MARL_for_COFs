"""Freeze the M1 learned bond/angle orbit-head Tier-4 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .orbit_ic_model import QuotientICPredictor


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-protocol",
        type=Path,
        default=Path(
            "generative_model/PG_OrbitFlow/configs/m0_oracle_reconstruction_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_path = args.base_protocol.resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base["schema_version"] != "pg-orbitflow-m0-oracle-protocol-v1":
        raise ValueError("M1 base must be the passed M0 protocol")
    model_setting = {
        "node_feature_dim": 29,
        "edge_feature_dim": 5,
        "hidden_dim": 96,
        "layers": 4,
    }
    model = QuotientICPredictor(**model_setting)
    model_setting["parameter_count_expected"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    energy_weights = dict(base["energy_weights"])
    energy_weights["local_pair"] = 0.0
    protocol = {
        "schema_version": "pg-orbitflow-m1-bond-angle-protocol-v1",
        "scientific_question": (
            "Can a quotient-multigraph network memorize Tier-4 bond/angle orbit "
            "targets accurately enough for the passed M0 decoder, while torsions "
            "remain oracle and no Cartesian target enters the model?"
        ),
        "base_m0_protocol": {"path": str(base_path), "sha256": _sha256(base_path)},
        "base_m0_summary": {
            "path": str(
                Path("generative_model/PG_OrbitFlow/reports/m0_oracle_reconstruction_v1_summary.json").resolve()
            ),
            "sha256": _sha256(
                Path("generative_model/PG_OrbitFlow/reports/m0_oracle_reconstruction_v1_summary.json")
            ),
        },
        "package_dir": base["package_dir"],
        "canonical_manifest_sha256": base["canonical_manifest_sha256"],
        "seed": int(base["seed"]) + 100,
        "panel": base["panel"],
        "model": model_setting,
        "training": {
            "optimizer_steps": 1024,
            "batch_size_molecules": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "checkpoint_every": 256,
            "bond_loss_weight": 10.0,
            "angle_loss_weight": 2.0,
        },
        "decoder": {
            "oracle_torsion_used": True,
            "oracle_chirality_used": True,
            "oracle_local_pair_used": False,
            "predicted_ring_bond_lengths_used": True,
            "energy_weights": energy_weights,
            "initialization": base["initialization"],
            "optimization": base["optimization"],
        },
        "prediction_gate": {
            "bond_orbit_mae_max_angstrom": 0.03,
            "angle_orbit_mae_max_degrees": 5.0,
        },
        "reconstruction_gate": {
            "bond_mae_max_angstrom": 0.03,
            "angle_mae_max_degrees": 5.0,
            "torsion_circular_mae_max_degrees": 2.0,
            "ring_closure_mae_max_angstrom": 0.03,
            "kabsch_rmsd_max_angstrom": 0.20,
            "collision_free_fraction_min": 1.0,
            "max_operation_atom_error_max_angstrom": 1e-6,
            "all_values_finite": True,
        },
        "isolation": {
            "iid_train_only": True,
            "iid_validation_test_or_core_ood_used": False,
            "target_cartesian_model_input_used": False,
            "target_cartesian_coordinate_loss_used": False,
            "oracle_bond_or_angle_used_by_decoder": False,
            "oracle_local_pair_used_by_decoder": False,
            "oracle_torsion_used_by_decoder": True,
            "mds_nerf_etkdg_or_external_checkpoint_used": False,
            "posthoc_hard_projection_or_f02_used": False,
        },
        "progression": {
            "if_passes": "implement M2 learned torsion orbit head and ring/chirality constraints",
            "if_fails_prediction": "diagnose quotient feature collisions or head capacity before changing decoder",
            "if_prediction_passes_but_reconstruction_fails": "diagnose predicted-constraint inconsistency or multi-start initialization",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_PG_ORBITFLOW_M1_PROTOCOL {args.output.resolve()}")


if __name__ == "__main__":
    main()
