"""Freeze phase-aware M2.1 Tier-16 training and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..c0_local_recovery import _sha256
from ..orbit_ic_model import QuotientICPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path(
            "generative_model/PG_OrbitFlow/configs/"
            "m2_tier16_panel_v2_phase_aware.json"
        ),
    )
    parser.add_argument(
        "--panel-audit",
        type=Path,
        default=Path(
            "generative_model/PG_OrbitFlow/reports/"
            "m2_tier16_panel_audit_v2_phase_aware.json"
        ),
    )
    parser.add_argument(
        "--m2-protocol",
        type=Path,
        default=Path(
            "generative_model/PG_OrbitFlow/configs/m2_torsion_tier4_v1.json"
        ),
    )
    parser.add_argument(
        "--m2-summary",
        type=Path,
        default=Path(
            "generative_model/PG_OrbitFlow/reports/m2_torsion_tier4_v1_summary.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panel_path = args.panel.resolve()
    audit_path = args.panel_audit.resolve()
    m2_protocol_path = args.m2_protocol.resolve()
    m2_summary_path = args.m2_summary.resolve()
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    m2 = json.loads(m2_protocol_path.read_text(encoding="utf-8"))
    summary = json.loads(m2_summary_path.read_text(encoding="utf-8"))
    if panel.get("status") != "FROZEN_FOR_PHASE_AWARE_M2P1":
        raise ValueError("M2.1 requires a frozen phase-aware Tier-16 panel")
    if not audit.get("passed") or not summary.get("passed"):
        raise ValueError("M2.1 requires passed panel audit and M2")
    if audit["panel"]["sha256"] != _sha256(panel_path):
        raise ValueError("panel changed after audit")

    checkpoint = Path(
        "generative_model/PG_OrbitFlow/runs/m2_torsion_tier4_v1/last.pt"
    ).resolve()
    model_setting = {
        "node_feature_dim": 29,
        "edge_feature_dim": 5,
        "hidden_dim": 96,
        "layers": 4,
        "predict_torsion": True,
        "geometry_group_phase": True,
    }
    model = QuotientICPredictor(**model_setting)
    model_setting["parameter_count_expected"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    decoder = dict(m2["decoder"])
    decoder["initialization"] = dict(decoder["initialization"])
    decoder["optimization"] = dict(decoder["optimization"])
    decoder["energy_weights"] = dict(decoder["energy_weights"])
    implementation_paths = {
        "orbit_ic_model": Path(
            "generative_model/PG_OrbitFlow/orbit_ic_model.py"
        ).resolve(),
        "orbit_kinematics": Path(
            "generative_model/PG_OrbitFlow/orbit_kinematics.py"
        ).resolve(),
        "oracle_free_solver": Path(
            "generative_model/PG_OrbitFlow/m0_oracle_reconstruction.py"
        ).resolve(),
        "training_runner": Path(
            "generative_model/PG_OrbitFlow/m2p1_phase_training.py"
        ).resolve(),
    }
    protocol = {
        "schema_version": "pg-orbitflow-m2p1-phase-aware-tier16-protocol-v1",
        "scientific_question": (
            "Can action-only group-phase features learn non-planar/chiral Tier-16 "
            "bond, angle, and signed torsion orbits and reconstruct symmetric 3D "
            "without any oracle decoder geometry?"
        ),
        "claim_limit": (
            "This is a 16-molecule memorization/representation Gate, not unseen "
            "molecule or dataset-scale generation evidence."
        ),
        "package_dir": panel["package_dir"],
        "canonical_manifest_sha256": panel["canonical_manifest_sha256"],
        "seed": int(m2["seed"]) + 100,
        "base_m2_protocol": {
            "path": str(m2_protocol_path),
            "sha256": _sha256(m2_protocol_path),
        },
        "base_m2_summary": {
            "path": str(m2_summary_path),
            "sha256": _sha256(m2_summary_path),
        },
        "panel": {
            "path": str(panel_path),
            "sha256": _sha256(panel_path),
            "audit_path": str(audit_path),
            "audit_sha256": _sha256(audit_path),
            "molecule_count": panel["molecule_count"],
            "records": panel["records"],
        },
        "model": model_setting,
        "implementation": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in implementation_paths.items()
        },
        "initialization": {
            "m2_checkpoint": {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
            },
            "inherited_parameter_prefixes": [
                "node_projection.",
                "messages.",
                "updates.",
                "norms.",
            ],
            "reinitialized_modules": ["bond_head", "angle_head", "torsion_head"],
            "reason": "all IC head input dimensions change when group-phase features are enabled",
            "optimizer_state_reused": False,
        },
        "training": {
            "optimizer_steps": 2048,
            "batch_size_molecules": 4,
            "point_group_composition_per_batch": {"C2": 2, "C3": 2},
            "deterministic_shuffle_cycle_steps": 4,
            "molecule_exposures_total": 8192,
            "molecule_exposures_each": 512,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "checkpoint_every": 512,
            "bond_loss_weight": 10.0,
            "angle_loss_weight": 2.0,
            "torsion_circular_loss_weight": 2.0,
        },
        "decoder": decoder,
        "prediction_gate": {
            "bond_orbit_mae_max_angstrom": 0.03,
            "angle_orbit_mae_max_degrees": 5.0,
            "torsion_orbit_circular_mae_max_degrees": 2.0,
            "nonplanar_torsion_orbit_circular_mae_max_degrees": 2.0,
            "c2_and_c3_each_pass": True,
        },
        "reconstruction_gate": {
            **m2["reconstruction_gate"],
            "nonplanar_torsion_circular_mae_max_degrees": 2.0,
            "active_chirality_count_min": int(
                audit["selected_summary"]["selected_active_chirality_count"]
            ),
            "chirality_preserved_fraction_min": 1.0,
            "c2_and_c3_each_pass": True,
        },
        "isolation": {
            "iid_train_only": True,
            "iid_validation_test_or_core_ood_used": False,
            "target_cartesian_model_input_used": False,
            "target_cartesian_coordinate_loss_used": False,
            "oracle_internal_coordinates_used_by_decoder": False,
            "oracle_local_pair_or_chirality_used_by_decoder": False,
            "group_phase_uses_only_operation_matrices_and_permutations": True,
            "mds_nerf_etkdg_or_external_checkpoint_used": False,
            "posthoc_hard_projection_or_f02_used": False,
        },
        "progression": {
            "if_passes": (
                "freeze M2.1 and design a nested Tier-32 panel; do not claim unseen "
                "generalization before IID-test/Core-OOD"
            ),
            "if_fails_prediction": (
                "stop before decoder and diagnose phase-feature/head capacity without "
                "changing the frozen panel or Gate"
            ),
            "if_prediction_passes_but_reconstruction_fails": (
                "diagnose signed-torsion consistency, chirality, ring closure, and "
                "multi-start selection without adding oracle constraints"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_PG_ORBITFLOW_M2P1_PROTOCOL {args.output.resolve()}")


if __name__ == "__main__":
    main()
