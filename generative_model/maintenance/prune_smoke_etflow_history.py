"""Prune superseded smoke experiments and pre-v5 ET-Flow branches.

Dry-run is the default.  The retained set is the complete executable closure
of ET-Flow/E3/F0.2 v5, the frozen four-route ablation and the minimal closed
SemlaFlow/MiDi/UAE entry points.  Frozen reports are not selected for deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "generative_model/maintenance/smoke_etflow_cleanup_manifest_20260828.json"
)

KEEP_SMOKE = {
    "__init__.py",
    "audit_uae3d_bond_capacity.py",
    "audit_uae3d_c23_reconstruction.py",
    "audit_uae3d_gate_a.py",
    "build_uae3d_c23_readonly_protocol.py",
    "build_uae3d_gate_a_protocol.py",
    "build_uae3d_split.py",
    "finalize_uae3d_c23_readonly_audit.py",
    "run_midi_forward.py",
    "run_semlaflow_forward.py",
    "run_uae3d_forward.py",
    "train_uae3d_reconstruction.py",
}

DELETE_MODELS = {
    "coordinate_flow.py",
    "coordinate_flow_runtime.py",
    "etflow_pg_collision_aware.py",
    "etflow_pg_collision_aware_v2.py",
    "etflow_pg_conditioned.py",
    "etflow_pg_conditioned_v2.py",
    "etflow_pg_dual_basis_v1.py",
    "etflow_pg_dual_basis_v2.py",
    "etflow_pg_dual_basis_v3.py",
    "etflow_pg_operation_orbit_v1.py",
    "etflow_pg_trajectory_risk_v1.py",
    "graph_pg_3d_denoiser.py",
    "graph_pg_3d_denoiser_full_pair.py",
    "graph_pg_3d_denoiser_multichannel.py",
    "graph_pg_3d_eqgat_backbone.py",
    "graph_pg_3d_semla_backbone.py",
    "graph_pg_3d_semla_flow.py",
    "orbit_coordinate_denoiser.py",
}

DELETE_TESTS = {
    "test_coordinate_flow.py",
    "test_coordinate_flow_fast32_protocol.py",
    "test_coordinate_flow_runtime.py",
    "test_etflow_ef2d4_execution_fix_v2.py",
    "test_etflow_pg_collision_aware.py",
    "test_etflow_pg_collision_aware_v2.py",
    "test_etflow_pg_conditioned.py",
    "test_etflow_pg_conditioned_v2.py",
    "test_etflow_pg_dual_basis_v1.py",
    "test_etflow_pg_dual_basis_v2.py",
    "test_etflow_pg_dual_basis_v3.py",
    "test_etflow_pg_operation_orbit_v1.py",
    "test_etflow_pg_trajectory_risk_v1.py",
    "test_etkdg_orbit_e2.py",
    "test_graph_pg_3d_denoiser.py",
    "test_graph_pg_3d_denoiser_full_pair.py",
    "test_graph_pg_3d_denoiser_multichannel.py",
    "test_graph_pg_3d_eqgat_backbone.py",
    "test_graph_pg_3d_semla_backbone.py",
    "test_graph_pg_3d_semla_flow.py",
    "test_orbit_coordinate_denoiser.py",
    "test_orbit_coordinate_o1_protocol.py",
    "test_orbit_force_field_coverage.py",
    "test_orbit_force_field_f01_point_group_artifacts.py",
    "test_orbit_force_field_f02_collision_repair.py",
}

DELETE_DATA = {
    "etflow_pg_adapter.py",
    "etflow_pg_inference.py",
}

DELETE_INFERENCE = {
    "audit_graph_action_extended.py",
    "build_etflow_e3f02_protocol.py",
    "build_etflow_e3f02_protocol_v2.py",
    "build_etflow_e3f02_rare_targets_protocol.py",
    "build_etflow_e3f02_rare_targets_protocol_v2.py",
    "build_graph_action_extended_protocol.py",
    "build_graph_action_extended_protocol_v2.py",
    "diagnose_d6h_action_candidates.py",
    "diagnose_etflow_e3f02_v2_failure.py",
    "diagnose_s4_action_candidates.py",
    "run_etflow_e3f02_rare_targets_v2.py",
}

DELETE_RUNS = {
    "etflow_c7rs_balanced_a2_0256_v1",
    "etflow_c7rs_raw_confirmation_v1",
    "etflow_e3f02_examples",
    "etflow_e3f02_iidtest_pilot8_v1",
    "etflow_e3f02_rare_targets_v1",
    "etflow_e3f02_s4_diagnosis",
    "etflow_e3f02_v2_examples",
    "etflow_e3f02_v3_examples",
    "etflow_e3f02_v4_examples",
    "etflow_ef0b_official",
    "etflow_ef1_zeroshot",
    "etflow_ef2a2_endpoint_interleaved_0256",
    "etflow_ef2a_adapter_0256",
    "etflow_ef2b_raw_sampling_v1",
    "etflow_ef2c2_overlap_v2_0256",
    "etflow_ef2c3_raw_sampling_v1",
    "etflow_ef2c4_repeat_v1",
    "etflow_ef2c6_disjoint_iidval64_v1",
    "etflow_ef2d2_operation_orbit_v2_0256",
    "etflow_ef2d4_trajectory_cvar_v2_0064",
    "etkdg_orbit_e2",
    "graph_pg_3d",
    "graph_pg_3d_g2",
    "graph_pg_3d_g21",
    "our_etflow_xtb_preflight1_v1",
    "our_etflow_xtb_preflight1_v2",
}

# v5 protocol builders use these three v2 files as frozen scientific evidence.
RARE_V2_KEEP = {
    "geometry_report.json",
    "point_group_report.json",
    "s4_inertia_boundary_diagnosis.json",
}


def _inside_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or ROOT not in resolved.parents:
        raise RuntimeError(f"refusing unsafe target: {resolved}")
    return resolved


def _size(path: Path) -> int:
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size
    return sum(item.lstat().st_size for item in path.rglob("*") if item.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidates() -> list[Path]:
    paths: list[Path] = []
    smoke = ROOT / "generative_model/smoke"
    paths.extend(path for path in smoke.glob("*.py") if path.name not in KEEP_SMOKE)
    paths.extend(ROOT / "generative_model/models" / name for name in DELETE_MODELS)
    paths.extend(ROOT / "generative_model/tests" / name for name in DELETE_TESTS)
    paths.extend(ROOT / "generative_model/data" / name for name in DELETE_DATA)
    paths.extend(ROOT / "generative_model/inference" / name for name in DELETE_INFERENCE)
    paths.extend(ROOT / "generative_model/runs" / name for name in DELETE_RUNS)

    rare_v2 = ROOT / "generative_model/runs/etflow_e3f02_rare_targets_v2"
    if rare_v2.exists():
        paths.extend(path for path in rare_v2.iterdir() if path.name not in RARE_V2_KEEP)

    paths.extend((ROOT / "generative_model").rglob("__pycache__"))
    paths.extend((ROOT / "generative_model").rglob("*.py[co]"))

    unique: dict[str, Path] = {}
    for path in paths:
        resolved = _inside_root(path)
        if resolved.exists() or resolved.is_symlink():
            unique[str(resolved)] = resolved
    ordered = sorted(unique.values(), key=lambda item: (len(item.parts), str(item)))
    selected: list[Path] = []
    for path in ordered:
        if not any(parent == path or parent in path.parents for parent in selected):
            selected.append(path)
    return selected


def _remove(path: Path) -> None:
    _inside_root(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    selected = candidates()
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "kind": "directory" if path.is_dir() else "file",
            "bytes": _size(path),
        }
        for path in selected
    ]
    payload = {
        "schema_version": "smoke-etflow-cleanup-v1",
        "status": "APPLIED" if args.apply else "DRY_RUN",
        "candidate_count": len(records),
        "reclaimable_bytes": sum(record["bytes"] for record in records),
        "policy": {
            "etflow_v5_executable_closure_preserved": True,
            "four_route_ablation_and_supplements_preserved": True,
            "official_etflow_checkpoint_preserved": True,
            "frozen_smoke_reports_preserved": True,
            "rare_v2_evidence_files_preserved": sorted(RARE_V2_KEEP),
            "legacy_minimal_entrypoints_preserved": True,
        },
        "retained_smoke_scripts": sorted(KEEP_SMOKE),
        "targets": records,
        "cleanup_script_sha256": _sha256(Path(__file__)),
    }
    if args.apply:
        for path in selected:
            _remove(path)
        OUTPUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
