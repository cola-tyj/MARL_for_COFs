"""Add fixed-protocol symmetry analysis and finalize the UAE C2/C3 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from generative_model.data import ATOM_SYMBOLS
from generative_model.data.symmetry import SymmetryProtocol, analyze_symmetry
from generative_model.models.graph_pg_3d_projection import minimum_pair_distance


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "generative_model/smoke/reports/uae3d_c23_readonly_protocol_v1.json"
DEFAULT_INTERMEDIATE = ROOT / "generative_model/runs/uae3d_c23_readonly/reconstruction_v1.json"
DEFAULT_OUTPUT = ROOT / "generative_model/smoke/reports/uae3d_c23_readonly_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(records: list[dict]) -> dict:
    count = len(records)
    return {
        "molecule_count": count,
        "atom_exact_molecules": sum(r["atom_exact"] for r in records),
        "bond_exact_molecules": sum(r["bond_exact"] for r in records),
        "categorical_exact_molecules": sum(r["categorical_exact"] for r in records),
        "sanitize_valid_molecules": sum(r["sanitize_valid"] for r in records),
        "connected_molecules": sum(r["connected"] for r in records),
        "collision_free_molecules": sum(r["collision_free"] for r in records),
        "analyzer_success_molecules": sum(r["analyzer_success"] for r in records),
        "pg_exact_molecules": sum(r["pg_exact_match"] for r in records),
        "pg_compatible_molecules": sum(r["pg_compatible"] for r in records),
        "joint_success_molecules": sum(r["joint_success"] for r in records),
        "coordinate_rmsd_mean_angstrom": float(np.mean([r["coordinate_rmsd_angstrom"] for r in records])),
        "coordinate_rmsd_max_angstrom": float(np.max([r["coordinate_rmsd_angstrom"] for r in records])),
        "minimum_pair_distance_angstrom": float(np.min([r["minimum_pair_distance_angstrom"] for r in records])),
    }


def run(args) -> dict:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    intermediate = json.loads(args.intermediate.read_text(encoding="utf-8"))
    if intermediate.get("status") != "PASS_UAE_C23_READONLY_RECONSTRUCTION_EXECUTION":
        raise RuntimeError("UAE C23 reconstruction intermediate 未通过")
    if intermediate["protocol_sha256"] != _sha256(args.protocol):
        raise RuntimeError("UAE C23 intermediate 未绑定当前 protocol")
    coords_path = Path(intermediate["coordinate_artifact"]["path"])
    if _sha256(coords_path) != intermediate["coordinate_artifact"]["sha256"]:
        raise RuntimeError("UAE C23 coordinate artifact SHA 不匹配")
    arrays = np.load(coords_path, allow_pickle=False)
    indices = arrays["package_indices"].astype(int).tolist()
    offsets = arrays["atom_offsets"].astype(int)
    expected = [int(item["package_index"]) for item in protocol["panel"]["records"]]
    if indices != expected or [r["package_index"] for r in intermediate["records"]] != expected:
        raise RuntimeError("UAE C23 panel/coordinate 顺序变化")
    symmetry_protocol = SymmetryProtocol(**protocol["identity"]["symmetry_analyzer"])
    joint = protocol["joint_success_definition"]
    records = []
    for item, base in enumerate(intermediate["records"]):
        begin, end = int(offsets[item]), int(offsets[item + 1])
        positions = np.asarray(arrays["predicted_positions"][begin:end], dtype=np.float64)
        centroid = positions.mean(axis=0)
        centered_positions = positions - centroid
        atom_types = arrays["atom_types"][begin:end].astype(int)
        symbols = [ATOM_SYMBOLS[index] for index in atom_types]
        analyzer_success = True
        analyzer_error = None
        actual_pg = None
        exact = compatible = False
        try:
            analysis = analyze_symmetry(
                symbols, centered_positions, base["target_pg"], symmetry_protocol
            )
            actual_pg = analysis["actual_pg"]
            exact = bool(analysis["pg_exact_match"])
            compatible = bool(analysis["pg_compatible"])
        except Exception as error:
            analyzer_success = False
            analyzer_error = f"{type(error).__name__}: {error}"
        min_distance = minimum_pair_distance(positions)
        collision_free = min_distance >= float(joint["minimum_pair_distance_angstrom_min"])
        joint_success = (
            bool(base["atom_exact"]) and bool(base["bond_exact"])
            and bool(base["sanitize_valid"]) and bool(base["connected"])
            and float(base["coordinate_rmsd_angstrom"]) <= float(joint["coordinate_rmsd_angstrom_max"])
            and collision_free and compatible
        )
        records.append({
            **base,
            "predicted_centroid_norm_angstrom": float(np.linalg.norm(centroid)),
            "symmetry_analysis_coordinate_normalization": "subtract predicted centroid only",
            "minimum_pair_distance_angstrom": min_distance,
            "collision_free": collision_free,
            "analyzer_success": analyzer_success,
            "analyzer_error": analyzer_error,
            "reconstructed_actual_pg": actual_pg,
            "pg_exact_match": exact,
            "pg_compatible": compatible,
            "joint_success": joint_success,
        })
    overall = _summary(records)
    by_pg = {pg: _summary([r for r in records if r["target_pg"] == pg]) for pg in ("C2", "C3")}
    size_names = ("small_le_16", "medium_17_40", "large_gt_40")
    by_size = {name: _summary([r for r in records if r["size_bin"] == name]) for name in size_names if any(r["size_bin"] == name for r in records)}
    t = protocol["descriptive_evidence_thresholds"]
    strong_checks = {
        "joint_success_molecules_min": overall["joint_success_molecules"] >= t["strong_joint_success_molecules_min"],
        "atom_exact_molecules_min": overall["atom_exact_molecules"] >= t["strong_atom_exact_molecules_min"],
        "bond_exact_molecules_min": overall["bond_exact_molecules"] >= t["strong_bond_exact_molecules_min"],
        "sanitize_connected_molecules_min": min(overall["sanitize_valid_molecules"], overall["connected_molecules"]) >= t["strong_sanitize_connected_molecules_min"],
        "pg_compatible_molecules_min": overall["pg_compatible_molecules"] >= t["strong_pg_compatible_molecules_min"],
    }
    if all(strong_checks.values()):
        status = "UAE_C23_READONLY_STRONG_RECONSTRUCTION_EVIDENCE_NO_BRANCH_REOPEN"
    elif overall["joint_success_molecules"] >= t["partial_joint_success_molecules_min"]:
        status = "UAE_C23_READONLY_PARTIAL_RECONSTRUCTION_EVIDENCE_NO_BRANCH_REOPEN"
    else:
        status = "UAE_C23_READONLY_WEAK_RECONSTRUCTION_EVIDENCE_KEEP_BRANCH_CLOSED"
    report = {
        "schema_version": "uae3d-c23-readonly-audit-v1",
        "status": status,
        "purpose": protocol["purpose"],
        "protocol": {"path": str(args.protocol.resolve()), "sha256": _sha256(args.protocol), "panel": protocol["panel"], "joint_success_definition": joint, "thresholds": t},
        "identity": {**protocol["identity"], "checkpoint_sha256": protocol["source"]["checkpoint_sha256"], "intermediate_sha256": _sha256(args.intermediate), "coordinate_artifact_sha256": _sha256(coords_path), "model_state_fingerprint": intermediate["model_state_fingerprint"]},
        "model_state_unchanged": intermediate["model_state_unchanged"],
        "weights_updated": False,
        "overall": overall,
        "by_target_pg": by_pg,
        "by_size_bin": by_size,
        "strong_evidence_checks": strong_checks,
        "records": records,
        "decision_scope": protocol["decision_scope"],
        "conclusion": (
            "This is reconstruction evidence only: UAE does not consume Target_PG. "
            "The pre-existing FAIL_GATE_A_STOP_UAE_BRANCH decision is unchanged."
        ),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "symmetry_analyzer": protocol["identity"]["symmetry_analyzer"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--intermediate", type=Path, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "overall": report["overall"], "by_target_pg": report["by_target_pg"], "by_size_bin": report["by_size_bin"], "output": str(args.output)}, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
