"""Freeze a fresh size-stratified IID-test evaluation for E3/F0.2 inference."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "generative_model/data/processed/v2"
METADATA = PACKAGE / "metadata.csv"
GRAPHS = PACKAGE / "cof_graphs.npz"
SPLIT = PACKAGE / "split_iid.json"
MANIFEST = PACKAGE / "manifest.json"
INFERENCE_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_protocol_v1.json"
EF1_PROTOCOL = ROOT / "generative_model/smoke/reports/etflow_ef1_zeroshot_protocol_v2.json"
OUTPUT = ROOT / "generative_model/inference/etflow_e3f02_iidtest_protocol_v1.json"
SELECTION_SEED = 20260827
EXAMINED_EXAMPLES = (902, 1160, 2400)
SOURCES = (
    "generative_model/inference/generate_etflow_symmetric_xyz.py",
    "generative_model/inference/run_etflow_e3f02_iidtest.py",
    "generative_model/inference/audit_etflow_e3f02_iidtest.py",
    "generative_model/inference/build_etflow_e3f02_iidtest_protocol.py",
    "generative_model/inference/audit_etflow_e3f02_point_group.py",
    "generative_model/tests/test_etflow_e3f02_iidtest.py",
    "generative_model/conformer/etflow_bridge.py",
    "generative_model/conformer/etflow_ef1_projection.py",
    "generative_model/symmetry/graph_action.py",
    "generative_model/optimization/orbit_force_field_repulsion.py",
    "generative_model/evaluation/symmetry_metrics.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: dict) -> str:
    payload = dict(value)
    payload.pop("protocol_fingerprint", None)
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _rank(package_index: int) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}:{int(package_index)}".encode()
    ).hexdigest()


def select_size_stratified(
    candidates: list[int], atom_counts: dict[int, int], count: int
) -> tuple[list[int], list[dict]]:
    if count % 4:
        raise ValueError("size-stratified quota 必须能被 4 整除")
    ordered = sorted(candidates, key=lambda index: (atom_counts[index], index))
    strata = np.array_split(np.asarray(ordered, dtype=np.int64), 4)
    quota = count // 4
    selected: list[int] = []
    audit: list[dict] = []
    for stratum_id, values in enumerate(strata):
        members = list(map(int, values.tolist()))
        chosen = sorted(members, key=_rank)[:quota]
        if len(chosen) != quota:
            raise RuntimeError("IID-test size stratum 样本不足")
        selected.extend(chosen)
        audit.append({
            "stratum_id": int(stratum_id),
            "available_count": len(members),
            "selected_count": len(chosen),
            "minimum_atom_count": min(atom_counts[index] for index in members),
            "maximum_atom_count": max(atom_counts[index] for index in members),
            "selected_package_indices": chosen,
        })
    return selected, audit


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    inference = json.loads(INFERENCE_PROTOCOL.read_text(encoding="utf-8"))
    ef1 = json.loads(EF1_PROTOCOL.read_text(encoding="utf-8"))
    if inference["status"] != "FROZEN_KNOWN_GRAPH_TARGET_PG_TO_XYZ":
        raise RuntimeError("缺少冻结的 E3/F0.2 inference protocol")
    if inference["protocol_fingerprint"] != _fingerprint(inference):
        raise RuntimeError("E3/F0.2 inference protocol fingerprint 不一致")

    with METADATA.open(encoding="utf-8", newline="") as handle:
        rows = {int(row["Package_Index"]): row for row in csv.DictReader(handle)}
    with np.load(GRAPHS, allow_pickle=False) as archive:
        offsets = np.asarray(archive["atom_offsets"], dtype=np.int64)
    atom_counts = {
        index: int(offsets[index + 1] - offsets[index])
        for index in range(len(offsets) - 1)
    }
    test_indices = set(map(int, split["indices"]["test"]))
    excluded = set(map(int, ef1["panel"]["package_indices"])) | set(EXAMINED_EXAMPLES)
    available = {
        point_group: [
            index for index in sorted(test_indices - excluded)
            if rows[index]["Target_PG"] == point_group
        ]
        for point_group in ("C2", "C3")
    }
    selected_by_pg: dict[str, list[int]] = {}
    strata: dict[str, list[dict]] = {}
    for point_group, count in (("C2", 48), ("C3", 16)):
        selected_by_pg[point_group], strata[point_group] = select_size_stratified(
            available[point_group], atom_counts, count
        )
    panel = sorted(
        selected_by_pg["C2"] + selected_by_pg["C3"], key=_rank
    )
    if len(panel) != 64 or len(set(panel)) != 64:
        raise RuntimeError("IID-test panel 必须包含 64 个唯一分子")
    if set(panel) & excluded or not set(panel).issubset(test_indices):
        raise RuntimeError("IID-test panel 与既有示例重叠或 split 泄漏")

    protocol = {
        "schema_version": "etflow-e3f02-iidtest-protocol-v1",
        "status": "FROZEN_BEFORE_FRESH_IID_TEST_INFERENCE",
        "route": "ET-Flow official prior + graph-only E3 hard projection + F0.2",
        "purpose": (
            "Measure end-to-end known-graph + requested C2/C3 -> compatible XYZ "
            "success on a fresh size-stratified IID-test panel."
        ),
        "panel": {
            "package_indices": panel,
            "molecule_ids": [rows[index]["Molecule_ID"] for index in panel],
            "molecule_count": 64,
            "target_pg_counts": {"C2": 48, "C3": 16},
            "selected_by_target_pg": selected_by_pg,
            "size_strata": strata,
            "available_fresh_counts": {
                key: len(value) for key, value in available.items()
            },
            "source_split": "IID-test only",
            "selection": (
                "Within each Target_PG, sort by (atom_count, package_index), "
                "split into four equal-count strata, then take a fixed quota by "
                "SHA256 rank of '20260827:package_index'."
            ),
            "selection_seed": SELECTION_SEED,
            "excluded_prior_examples": sorted(excluded),
            "overlap_with_prior_examples": 0,
            "minimum_atom_count": min(atom_counts[index] for index in panel),
            "maximum_atom_count": max(atom_counts[index] for index in panel),
            "contains_sn_count": sum(
                rows[index]["Contains_Sn"] == "True" for index in panel
            ),
        },
        "sampling": {
            "model_name": "drugs-o3",
            "base_seed": 2026082800,
            "seed_formula": "base_seed + package_index",
            "candidates_per_molecule": 4,
            "n_timesteps": 50,
            "sampler_type": "ODE",
            "resume_policy": "immutable per-molecule JSON/NPZ keyed by protocol SHA",
        },
        "geometry_gate_thresholds": {
            "graph_action_recovery_fraction_min": 1.0,
            "end_to_end_success_fraction_min": 0.90,
            "c2_success_fraction_min": 0.90,
            "c3_success_fraction_min": 0.875,
            "minimum_pair_distance_angstrom": 0.6,
            "maximum_operation_error_angstrom": 1e-5,
        },
        "point_group_gate_thresholds": {
            "analyzer_success_fraction_min": 0.90,
            "pg_compatible_fraction_min": 0.90,
            "c2_pg_compatible_fraction_min": 0.90,
            "c3_pg_compatible_fraction_min": 0.875,
        },
        "primary_gate_policy": {
            "all_fractions_use_full_panel_denominator": True,
            "failed_generation_counts_as_analyzer_and_compatible_failure": True,
            "exact_match_is_descriptive_only": True,
            "thresholds_frozen_before_coordinates": True,
            "no_posthoc_candidate_or_threshold_change": True,
        },
        "decision": {
            "geometry_pass_status": "PASS_ETFLOW_E3F02_IIDTEST_GEOMETRY_AWAIT_POINT_GROUP_AUDIT",
            "final_pass_status": "PASS_ETFLOW_E3F02_IIDTEST_KNOWN_GRAPH_TARGET_PG_TO_XYZ",
            "fail_status": "FAIL_ETFLOW_E3F02_IIDTEST_DIAGNOSE_FAILURES",
            "pass_action": "freeze C2/C3 composite route and proceed to S4/D6h feasibility",
            "fail_action": "diagnose action, ET-Flow candidate, optimizer and PG failures without lowering thresholds",
        },
        "scope": {
            "known_graph": True,
            "requested_target_pg": True,
            "reference_xyz_used": False,
            "stored_symmetry_action_used": False,
            "etkdg_used": False,
            "hard_symmetry_projection_used": True,
            "orbit_constrained_force_field_used": True,
            "raw_learned_target_pg_claim": False,
            "iid_test_used": True,
            "core_ood_used": False,
        },
        "identity": {
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "manifest_sha256": _sha256(MANIFEST),
            "metadata_sha256": _sha256(METADATA),
            "graphs_sha256": _sha256(GRAPHS),
            "iid_split_sha256": _sha256(SPLIT),
            "inference_protocol_sha256": _sha256(INFERENCE_PROTOCOL),
            "checkpoint_sha256": inference["identity"]["checkpoint_sha256"],
            "symmetry_protocol": manifest["symmetry_protocol"],
            "source_sha256": {
                relative: _sha256(ROOT / relative) for relative in SOURCES
            },
        },
    }
    protocol["protocol_fingerprint"] = _fingerprint(protocol)
    OUTPUT.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"FROZEN protocol={OUTPUT}")
    print(f"sha256={_sha256(OUTPUT)}")
    print(json.dumps(protocol["panel"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
