"""Freeze the 32-molecule paired our_ET_Flow ablation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT,
    _atomic_json,
    _fingerprint,
    _sha256,
)


OUTPUT = ROOT / "generative_model/evaluation/our_etflow_ablation_protocol_v1.json"
PACKAGE = ROOT / "generative_model/data/processed/v2"
V5_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_protocol_v5.json"
IID_PANEL = ROOT / "generative_model/inference/etflow_e3f02_iidtest_protocol_v1.json"
RARE_PANEL = ROOT / "generative_model/inference/etflow_e3f02_rare_targets_protocol_v3.json"


def _seed(package_index: int, target_pg: str) -> int:
    payload = f"our-etflow-ablation-v1\0{package_index}\0{target_pg}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _select(
    indices: list[int], metadata: pd.DataFrame, target_pg: str, count: int
) -> list[int]:
    candidates = [
        int(index)
        for index in indices
        if str(metadata.iloc[int(index)]["Target_PG"]) == target_pg
    ]
    ranked = sorted(
        candidates,
        key=lambda index: hashlib.sha256(
            f"our-etflow-ablation-panel-v1\0{target_pg}\0"
            f"{metadata.iloc[index]['Molecule_ID']}".encode()
        ).hexdigest(),
    )
    if len(ranked) < count:
        raise RuntimeError(f"{target_pg} panel insufficient: {len(ranked)} < {count}")
    return ranked[:count]


def build() -> dict[str, Any]:
    metadata = pd.read_csv(PACKAGE / "metadata.csv")
    iid = json.loads(IID_PANEL.read_text(encoding="utf-8"))["panel"]["package_indices"]
    rare = json.loads(RARE_PANEL.read_text(encoding="utf-8"))["panel"]["package_indices"]
    indices: list[int] = []
    for target_pg in ("C2", "C3"):
        indices.extend(_select(iid, metadata, target_pg, 8))
    for target_pg in ("S4", "D6h"):
        indices.extend(_select(rare, metadata, target_pg, 8))
    records = []
    for package_index in indices:
        row = metadata.iloc[package_index]
        target_pg = str(row["Target_PG"])
        records.append({
            "package_index": package_index,
            "molecule_id": str(row["Molecule_ID"]),
            "target_pg": target_pg,
            "seed": _seed(package_index, target_pg),
            "atom_count": int(row["Num_Atoms"]),
            "split_iid": str(row["Split_IID"]),
            "split_core_ood": str(row["Split_Core_OOD"]),
            "contains_sn": bool(row["Contains_Sn"]),
        })
    source_files = (
        "generative_model/evaluation/our_etflow_ablation.py",
        "generative_model/evaluation/run_our_etflow_ablation.py",
        "generative_model/evaluation/audit_our_etflow_ablation_point_group.py",
        "generative_model/inference/generate_etflow_symmetric_xyz.py",
        "generative_model/inference/generate_etflow_symmetric_xyz_v3.py",
        "generative_model/inference/generate_etflow_symmetric_xyz_v4.py",
        "generative_model/inference/generate_etflow_symmetric_xyz_v5.py",
        "generative_model/conformer/etflow_bridge.py",
        "generative_model/conformer/etflow_ef1_projection.py",
        "generative_model/optimization/etkdg_orbit_initializer.py",
        "generative_model/optimization/orbit_force_field_repulsion.py",
        "generative_model/optimization/force_field_support.py",
    )
    protocol: dict[str, Any] = {
        "schema_version": "our-etflow-paired-ablation-protocol-v1",
        "status": "FROZEN_BEFORE_COORDINATE_GENERATION",
        "purpose": (
            "Paired decomposition of ETKDGv3, official ET-Flow raw prior, "
            "graph-action Reynolds hard projection, and frozen F0.2 refinement."
        ),
        "panel": {
            "selection": (
                "SHA-256-ranked 8 per Target_PG from the already frozen C2/C3 "
                "IID-test and all-S4/D6h panels; coordinates and outcome metrics "
                "were not used for selection"
            ),
            "molecule_count": len(records),
            "target_pg_counts": {
                target_pg: sum(item["target_pg"] == target_pg for item in records)
                for target_pg in ("C2", "C3", "S4", "D6h")
            },
            "records": records,
        },
        "paired_contract": {
            "same_canonical_graph": True,
            "same_base_seed": True,
            "same_requested_target_pg": True,
            "etflow_raw_coordinates_shared_by_routes": [
                "etflow_raw",
                "etflow_hard_projection",
                "etflow_hard_projection_f02",
            ],
            "reference_xyz_used_for_generation": False,
            "reference_xyz_used_for_selection": False,
            "reference_xyz_role": "post-hoc bond/RMSD metrics only",
            "full_panel_denominator_for_failures": True,
        },
        "routes": {
            "etkdg_v3_best_of_n": {
                "candidate_count": 4,
                "maximum_iterations": 1000,
                "use_random_coordinates": False,
                "selection": "lowest strict UFF single-point energy, then candidate id",
                "target_pg_used": False,
                "unconstrained_force_field_optimization_used": False,
            },
            "etflow_raw": {
                "candidate_count": 4,
                "n_timesteps": 50,
                "selection": "lowest strict UFF single-point energy, then candidate id",
                "target_pg_used": False,
            },
            "etflow_hard_projection": {
                "candidate_count": 4,
                "action_source": "coordinate-free typed-graph automorphism recovery",
                "selection": "frozen reference-free v5 projection rank per raw candidate",
                "hard_projection": "24 PCA orientations plus one Reynolds projection",
                "f02_used": False,
            },
            "etflow_hard_projection_f02": {
                "definition": "frozen production v5 combination ranking and F0.2 selection",
                "candidate_count": 4,
                "f02_used": True,
            },
        },
        "metrics": {
            "primary": [
                "independent pymatgen PG-compatible fraction",
                "collision-free fraction at 0.6 angstrom",
                "bond-length MAE to canonical coordinates",
                "strict UFF single-point energy per atom",
                "descriptive wall runtime",
                "within-graph candidate diversity",
            ],
            "secondary": [
                "PG exact match",
                "Kabsch RMSD to the one canonical conformer",
                "all-pair distance MAE to the one canonical conformer",
            ],
            "energy_note": (
                "UFF energies are compared only within paired routes of the same graph; "
                "cross-molecule means are additionally normalized per atom"
            ),
            "runtime_note": (
                "wall timing is descriptive and machine-dependent; the ET-Flow prior "
                "time is measured once and attributed to each ET-Flow end-to-end route"
            ),
        },
        "symmetry_protocol": {
            "analyzer": "pymatgen.symmetry.analyzer.PointGroupAnalyzer",
            "tolerance_angstrom": 0.3,
            "eigen_tolerance": 0.01,
            "matrix_tolerance": 0.1,
            "assignment": "scipy.optimize.linear_sum_assignment, element-blocked",
            "assignment_metric": (
                "Euclidean distance in centroid-centered Cartesian coordinates"
            ),
            "operation_acceptance": (
                "operation RMS assigned distance <= tolerance_angstrom"
            ),
        },
        "scope": {
            "known_canonical_graph": True,
            "all_atoms_explicit_including_hydrogen": True,
            "target_pgs": ["C2", "C3", "S4", "D6h"],
            "new_graph_generation_claim": False,
            "unseen_graph_claim": False,
            "raw_etflow_target_pg_conditioning_claim": False,
            "quality_claim_before_audit": False,
        },
        "identity": {
            "checkpoint_path": "generative_model/checkpoints/etflow/drugs-o3.ckpt",
            "checkpoint_sha256": _sha256(
                ROOT / "generative_model/checkpoints/etflow/drugs-o3.ckpt"
            ),
            "v5_protocol_sha256": _sha256(V5_PROTOCOL),
            "dataset_manifest_sha256": _sha256(PACKAGE / "manifest.json"),
            "dataset_graphs_sha256": _sha256(PACKAGE / "cof_graphs.npz"),
            "dataset_metadata_sha256": _sha256(PACKAGE / "metadata.csv"),
            "iid_panel_protocol_sha256": _sha256(IID_PANEL),
            "rare_panel_protocol_sha256": _sha256(RARE_PANEL),
            "source_sha256": {
                relative: _sha256(ROOT / relative) for relative in source_files
            },
        },
    }
    protocol["protocol_fingerprint"] = _fingerprint(protocol)
    _atomic_json(OUTPUT, protocol)
    return protocol


def main() -> None:
    protocol = build()
    print(json.dumps({
        "status": protocol["status"],
        "output": str(OUTPUT),
        "panel": protocol["panel"]["target_pg_counts"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "sha256": _sha256(OUTPUT),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
