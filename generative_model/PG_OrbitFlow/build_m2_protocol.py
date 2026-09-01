"""Freeze the M2 learned bond/angle/torsion Tier-4 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .c0_local_recovery import _sha256
from .orbit_ic_model import QuotientICPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-protocol",
        type=Path,
        default=Path(
            "generative_model/PG_OrbitFlow/configs/m1_bond_angle_tier4_v1.json"
        ),
    )
    parser.add_argument(
        "--base-summary",
        type=Path,
        default=Path(
            "generative_model/PG_OrbitFlow/reports/"
            "m1_bond_angle_tier4_v1_summary.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_path = args.base_protocol.resolve()
    summary_path = args.base_summary.resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base["schema_version"] != "pg-orbitflow-m1-bond-angle-protocol-v1":
        raise ValueError("M2 base must be the passed M1 protocol")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("passed"):
        raise ValueError("M2 requires a passed M1 summary")

    model_setting = {
        "node_feature_dim": 29,
        "edge_feature_dim": 5,
        "hidden_dim": 96,
        "layers": 4,
        "predict_torsion": True,
    }
    model = QuotientICPredictor(**model_setting)
    model_setting["parameter_count_expected"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    m1_checkpoint = Path(
        "generative_model/PG_OrbitFlow/runs/m1_bond_angle_tier4_v1/"
        "step-001024.pt"
    ).resolve()
    if not m1_checkpoint.is_file():
        raise FileNotFoundError(m1_checkpoint)
    energy_weights = dict(base["decoder"]["energy_weights"])
    energy_weights["local_pair"] = 0.0
    energy_weights["chirality"] = 0.0
    protocol = {
        "schema_version": "pg-orbitflow-m2-torsion-protocol-v1",
        "scientific_question": (
            "Can one quotient-multigraph network memorize all Tier-4 bond, angle, "
            "and torsion orbit targets accurately enough for reconstruction without "
            "any oracle internal-coordinate constraint in the decoder?"
        ),
        "claim_limit": (
            "The frozen Tier-4 panel contains only planar 0/pi torsions; a pass "
            "validates the all-learned interface but not non-planar torsion "
            "generalization."
        ),
        "base_m1_protocol": {"path": str(base_path), "sha256": _sha256(base_path)},
        "base_m1_summary": {
            "path": str(summary_path),
            "sha256": _sha256(summary_path),
        },
        "package_dir": base["package_dir"],
        "canonical_manifest_sha256": base["canonical_manifest_sha256"],
        "seed": int(base["seed"]) + 100,
        "panel": base["panel"],
        "model": model_setting,
        "initialization": {
            "shared_encoder_bond_angle_from_m1": True,
            "m1_checkpoint": {
                "path": str(m1_checkpoint),
                "sha256": _sha256(m1_checkpoint),
            },
            "torsion_head": "deterministic random initialization from protocol seed",
            "optimizer_state_reused": False,
        },
        "training": {
            "optimizer_steps": 1024,
            "batch_size_molecules": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "checkpoint_every": 256,
            "bond_loss_weight": 10.0,
            "angle_loss_weight": 2.0,
            "torsion_circular_loss_weight": 2.0,
        },
        "decoder": {
            "oracle_bond_used": False,
            "oracle_angle_used": False,
            "oracle_torsion_used": False,
            "oracle_chirality_used": False,
            "oracle_local_pair_used": False,
            "predicted_ring_bond_lengths_used": True,
            "energy_weights": energy_weights,
            "initialization": base["decoder"]["initialization"],
            "optimization": base["decoder"]["optimization"],
            "evaluation_device": "cpu",
        },
        "prediction_gate": {
            "bond_orbit_mae_max_angstrom": 0.03,
            "angle_orbit_mae_max_degrees": 5.0,
            "torsion_orbit_circular_mae_max_degrees": 2.0,
        },
        "reconstruction_gate": base["reconstruction_gate"],
        "isolation": {
            "iid_train_only": True,
            "iid_validation_test_or_core_ood_used": False,
            "target_cartesian_model_input_used": False,
            "target_cartesian_coordinate_loss_used": False,
            "oracle_internal_coordinates_used_by_decoder": False,
            "oracle_local_pair_or_chirality_used_by_decoder": False,
            "mds_nerf_etkdg_or_external_checkpoint_used": False,
            "posthoc_hard_projection_or_f02_used": False,
        },
        "progression": {
            "if_passes": (
                "freeze the M2 interface, then audit/select a Tier-16 panel with "
                "non-planar and chiral examples before any generalization claim"
            ),
            "if_fails_prediction": (
                "diagnose planar torsion feature collisions or circular-head capacity"
            ),
            "if_prediction_passes_but_reconstruction_fails": (
                "diagnose learned-constraint consistency and multi-start optimization"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_PG_ORBITFLOW_M2_PROTOCOL {args.output.resolve()}")


if __name__ == "__main__":
    main()
