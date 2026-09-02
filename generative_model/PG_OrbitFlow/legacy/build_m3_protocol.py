"""Freeze the nested M3 Tier-32 operation-aware IC experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..c0_local_recovery import _sha256
from ..m2p2_model import SetValuedQuotientICPredictor
from ..m2p1_phase_training import _model_setting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-audit", type=Path, required=True)
    parser.add_argument("--m2-protocol", type=Path, required=True)
    parser.add_argument("--m2-run-dir", type=Path, required=True)
    parser.add_argument("--m2-matching-report", type=Path, required=True)
    parser.add_argument("--m2-reproducibility-report", type=Path, required=True)
    parser.add_argument("--decoder-parallelism-report", type=Path, required=True)
    parser.add_argument("--failed-v1-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("generative_model/PG_OrbitFlow").resolve()
    paths = {name: value.resolve() for name, value in vars(args).items() if isinstance(value, Path)}
    panel = json.loads(paths["panel"].read_text(encoding="utf-8"))
    audit = json.loads(paths["panel_audit"].read_text(encoding="utf-8"))
    m2_protocol = json.loads(paths["m2_protocol"].read_text(encoding="utf-8"))
    m2_report_path = paths["m2_run_dir"] / "report.json"
    m2_checkpoint_path = paths["m2_run_dir"] / "last.pt"
    m2_report = json.loads(m2_report_path.read_text(encoding="utf-8"))
    matching = json.loads(paths["m2_matching_report"].read_text(encoding="utf-8"))
    reproduction = json.loads(paths["m2_reproducibility_report"].read_text(encoding="utf-8"))
    parallelism = json.loads(paths["decoder_parallelism_report"].read_text(encoding="utf-8"))
    if not audit.get("passed") or not matching.get("passed") or not reproduction.get("passed") or not parallelism.get("passed"):
        raise RuntimeError("M3 protocol requires passed panel, M2 completion, and decoder parallelism Gates")
    if panel.get("status") != "FROZEN_FOR_M3_TIER32" or len(panel["records"]) != 32:
        raise RuntimeError("M3 panel is not frozen Tier-32")
    if not m2_report["prediction_gate"]["passed"]:
        raise RuntimeError("M3 parent prediction Gate did not pass")

    settings = _model_setting(m2_protocol)
    model = SetValuedQuotientICPredictor(**settings)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    settings["parameter_count_expected"] = parameter_count
    settings["trainable_parameter_count_expected"] = parameter_count
    sources = {
        name: root / filename
        for name, filename in {
            "runner": "m3_tier32_training.py",
            "m2p2_model": "m2p2_model.py",
            "local_rotor": "local_rotor.py",
            "local_atom_matching": "local_atom_matching.py",
            "orbit_ic_model": "orbit_ic_model.py",
            "solver": "m0_oracle_reconstruction.py",
            "kinematics": "orbit_kinematics.py",
        }.items()
    }
    reconstruction_gate = dict(m2_protocol["reconstruction_gate"])
    reconstruction_gate["active_chirality_count_min"] = int(
        audit["selected_summary"]["active_chirality_count"]
    )
    protocol = {
        "schema_version": "pg-orbitflow-m3-tier32-protocol-v1",
        "seed": int(m2_protocol["seed"]) + 100,
        "package_dir": panel["package_dir"],
        "canonical_manifest_sha256": panel["canonical_manifest_sha256"],
        "panel": panel,
        "panel_audit": {"path": str(paths["panel_audit"]), "sha256": _sha256(paths["panel_audit"])},
        "decoder_parallelism_audit": {"path": str(paths["decoder_parallelism_report"]), "sha256": _sha256(paths["decoder_parallelism_report"])},
        "model": settings,
        "parent": {
            "protocol": {"path": str(paths["m2_protocol"]), "sha256": _sha256(paths["m2_protocol"])},
            "report": {"path": str(m2_report_path), "sha256": _sha256(m2_report_path)},
            "checkpoint": {"path": str(m2_checkpoint_path), "sha256": _sha256(m2_checkpoint_path)},
            "matching_report": {"path": str(paths["m2_matching_report"]), "sha256": _sha256(paths["m2_matching_report"])},
            "reproducibility_report": {"path": str(paths["m2_reproducibility_report"]), "sha256": _sha256(paths["m2_reproducibility_report"])},
        },
        "implementation": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in sources.items()
        },
        "initialization": {
            "complete_m2_v5_state_inherited": True,
            "all_model_parameters_unfrozen": True,
            "optimizer_reused": False,
        },
        "training": {
            "optimizer_steps": 4096,
            "batch_size_molecules": 4,
            "point_group_composition_per_batch": {"C2": 2, "C3": 2},
            "molecule_exposures_each": 512,
            "molecule_exposures_total": 16384,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "checkpoint_every": 512,
            "bond_loss_weight": 10.0,
            "angle_loss_weight": 2.0,
            "torsion_circular_loss_weight": 2.0,
            "torsion_assignment": "one group-action-conjugated terminal-sibling permutation per operation-coupled component",
        },
        "decoder": {**m2_protocol["decoder"], "parallel_workers": 8},
        "prediction_gate": m2_protocol["prediction_gate"],
        "reconstruction_gate": reconstruction_gate,
        "evaluation_semantics": {
            "chemical_geometry_metrics": "strict graph-equivalent terminal-sibling matching",
            "raw_group_action_and_collision_metrics_preserved": True,
            "reference_coordinates_used_only_for_evaluation_matching": True,
            "free_element_hungarian_used": False,
        },
        "isolation": {
            "iid_test_used": False,
            "core_ood_used": False,
            "hard_projection_or_f02_used": False,
            "oracle_internal_coordinates_used": False,
            "target_cartesian_model_input_used": False,
        },
        "progression": {
            "if_passes": "freeze M3 Tier-32 and run selected-start reproducibility audit",
            "if_prediction_fails": "stop before decoder and diagnose without relaxing Gate",
            "if_reconstruction_fails": "diagnose exact failing records without relaxing Gate",
        },
    }
    if args.failed_v1_log:
        failed = paths["failed_v1_log"]
        protocol["execution_revision"] = {
            "failed_v1_log": {"path": str(failed), "sha256": _sha256(failed)},
            "failure_signature": "ValueError: M2.1 requires exactly eight C2 and eight C3 molecules",
            "optimizer_steps_before_failure": 0,
            "checkpoint_written_before_failure": False,
            "failed_run_reused": False,
            "restart_policy": "new output directory, deterministic restart from step 1",
            "scientific_protocol_changed": False,
            "sole_code_change": "replace the reused hard-coded 8+8 M2.1 scheduler with the same deterministic 2+2 algorithm generalized to the frozen 16+16 M3 panel",
        }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_M3_PROTOCOL {paths['output']}")


if __name__ == "__main__":
    main()
