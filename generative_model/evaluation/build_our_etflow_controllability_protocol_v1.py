"""Freeze a same-graph, shared-raw C2/C3/D6h controllability panel."""

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


OUTPUT = ROOT / "generative_model/evaluation/our_etflow_controllability_protocol_v1.json"
PACKAGE = ROOT / "generative_model/data/processed/v2"
V5 = ROOT / "generative_model/inference/etflow_e3f02_protocol_v5.json"


def build() -> dict[str, Any]:
    metadata = pd.read_csv(PACKAGE / "metadata.csv")
    candidates = [
        int(index) for index, row in metadata.iterrows() if row["Target_PG"] == "D6h"
    ]
    candidates.sort(key=lambda index: hashlib.sha256(
        f"our-etflow-control-v1\0{metadata.iloc[index]['Molecule_ID']}".encode()
    ).hexdigest())
    selected = candidates[:8]
    records = []
    for package_index in selected:
        row = metadata.iloc[package_index]
        seed = int.from_bytes(hashlib.sha256(
            f"our-etflow-control-seed-v1\0{package_index}".encode()
        ).digest()[:8], "big") % (2**31 - 1)
        records.append({
            "package_index": package_index,
            "molecule_id": str(row["Molecule_ID"]),
            "seed": seed,
            "atom_count": int(row["Num_Atoms"]),
            "canonical_target_pg": "D6h",
            "requested_targets": ["C2", "C3", "D6h"],
        })
    source_files = (
        "generative_model/evaluation/run_our_etflow_controllability.py",
        "generative_model/evaluation/audit_our_etflow_controllability.py",
        "generative_model/evaluation/our_etflow_ablation.py",
        "generative_model/conformer/etflow_bridge.py",
        "generative_model/conformer/etflow_ef1_projection.py",
        "generative_model/inference/generate_etflow_symmetric_xyz.py",
        "generative_model/inference/generate_etflow_symmetric_xyz_v3.py",
        "generative_model/inference/generate_etflow_symmetric_xyz_v4.py",
        "generative_model/optimization/orbit_force_field_repulsion.py",
        "generative_model/symmetry/graph_action.py",
        "generative_model/symmetry/graph_action_extended.py",
        "generative_model/symmetry/graph_action_candidates_v2.py",
    )
    v5 = json.loads(V5.read_text(encoding="utf-8"))
    protocol: dict[str, Any] = {
        "schema_version": "our-etflow-target-pg-controllability-protocol-v1",
        "status": "FROZEN_BEFORE_CONTROLLABILITY_GENERATION",
        "purpose": (
            "Hold graph, seed and all four raw ET-Flow samples fixed while changing "
            "only the legal requested point group C2/C3/D6h."
        ),
        "targets": ["C2", "C3", "D6h"],
        "panel": {
            "selection": "SHA-256-ranked 8 canonical D6h graphs",
            "molecule_count": len(records),
            "records": records,
        },
        "shared_input_contract": {
            "same_graph": True,
            "same_seed": True,
            "same_raw_etflow_candidate_tensor": True,
            "only_changed_variable": "requested Target_PG and its recovered graph action",
            "reference_xyz_used_for_generation_or_selection": False,
        },
        "response_gate": {
            "minimum_kabsch_rmsd_angstrom": 0.05,
            "minimum_responsive_fraction_per_target_pair": 0.75,
            "primary_metric": (
                "fraction of same-graph target pairs whose final structures differ "
                "after proper-rotation Kabsch alignment"
            ),
            "compatible_under_requested_pg_required": True,
            "cross_action_errors_descriptive": True,
            "actual_pg_exact_match_descriptive": True,
        },
        "symmetry_protocol": v5["identity"]["symmetry_protocol"],
        "scope": {
            "known_canonical_graph": True,
            "raw_etflow_target_pg_conditioning_claim": False,
            "composite_target_pg_controllability_claim": True,
            "hard_projection_used": True,
            "f02_used": True,
        },
        "identity": {
            "checkpoint_sha256": v5["identity"]["checkpoint_sha256"],
            "v5_protocol_sha256": _sha256(V5),
            "dataset_manifest_sha256": _sha256(PACKAGE / "manifest.json"),
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
        "panel": [item["package_index"] for item in protocol["panel"]["records"]],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "sha256": _sha256(OUTPUT),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
