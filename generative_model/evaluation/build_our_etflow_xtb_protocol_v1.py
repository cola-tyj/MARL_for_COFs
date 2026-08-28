"""Freeze the 32-molecule unconstrained GFN2-xTB relaxation protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT,
    _atomic_json,
    _fingerprint,
    _sha256,
)


OUTPUT = ROOT / "generative_model/evaluation/our_etflow_xtb_protocol_v1.json"
ABLATION_PROTOCOL = ROOT / "generative_model/evaluation/our_etflow_ablation_protocol_v1.json"
ABLATION_RUN = ROOT / "generative_model/runs/our_etflow_ablation_v1"


def build() -> dict[str, Any]:
    parent_protocol = json.loads(ABLATION_PROTOCOL.read_text(encoding="utf-8"))
    parent_report = json.loads((ABLATION_RUN / "report.json").read_text(encoding="utf-8"))
    records = []
    for item, result in zip(
        parent_protocol["panel"]["records"], parent_report["records"], strict=True
    ):
        route = result["routes"]["etflow_hard_projection_f02"]
        if not route["success"]:
            raise RuntimeError(
                f"frozen F0.2 source failed for package_index={item['package_index']}"
            )
        records.append({
            "panel_id": int(result["panel_id"]),
            "package_index": int(item["package_index"]),
            "molecule_id": str(item["molecule_id"]),
            "target_pg": str(item["target_pg"]),
            "atom_count": int(item["atom_count"]),
        })
    source_files = (
        "generative_model/evaluation/our_etflow_xtb_relaxation.py",
        "generative_model/evaluation/run_our_etflow_xtb_relaxation.py",
        "generative_model/evaluation/audit_our_etflow_xtb_point_group.py",
        "generative_model/evaluation/summarize_our_etflow_xtb.py",
    )
    protocol: dict[str, Any] = {
        "schema_version": "our-etflow-gfn2-xtb-relaxation-protocol-v1",
        "status": "FROZEN_BEFORE_GFN2_XTB_RELAXATION",
        "purpose": (
            "Test whether the 32 frozen F0.2 coordinates remain collision-free and "
            "Target_PG-compatible after unconstrained GFN2-xTB relaxation."
        ),
        "panel": {
            "molecule_count": len(records),
            "target_pg_counts": {
                target: sum(record["target_pg"] == target for record in records)
                for target in ("C2", "C3", "S4", "D6h")
            },
            "records": records,
            "selection": "exact 32-molecule paired-ablation panel; no result-based reselection",
        },
        "relaxation": {
            "calculator": "tblite.ase.TBLite",
            "tblite_version": "0.7.0",
            "ase_version": "3.24.0",
            "method": "GFN2-xTB",
            "optimizer": "ase.optimize.LBFGS",
            "fmax_ev_per_angstrom": 0.1,
            "maximum_geometry_steps": 100,
            "maximum_scf_iterations": 250,
            "accuracy": 1.0,
            "electronic_temperature_kelvin": 300.0,
            "calculator_verbosity": 0,
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            "periodic_boundary_conditions": False,
            "constraints": None,
            "hard_projection_during_relaxation": False,
            "posthoc_force_field": False,
            "charge_source": "sum of canonical formal_charges",
            "multiplicity_source": "1 + sum of canonical radical_electrons",
        },
        "metrics": {
            "energy_unit": "eV",
            "force_unit": "eV/angstrom",
            "coordinate_unit": "angstrom",
            "collision_threshold_angstrom": 0.6,
            "reported": [
                "pre/post GFN2-xTB energy and energy per atom",
                "pre/post maximum force",
                "optimizer convergence and step count",
                "pre/post collision state",
                "pre/post bond-length error",
                "pre/post Kabsch RMSD and pair-distance MAE",
                "independent pre/post actual-PG and target compatibility",
                "wall runtime",
            ],
        },
        "gate": {
            "calculation_success_fraction_min": 0.875,
            "energy_nonincreasing_fraction_among_success_min": 0.95,
            "collision_free_after_fraction_full_panel_min": 0.90,
            "pg_compatible_after_fraction_full_panel_min": 0.75,
            "median_kabsch_rmsd_pre_to_post_angstrom_max": 0.50,
            "all_successful_values_finite": True,
            "interpretation": (
                "A diagnostic physical-stability gate. Passing does not establish "
                "quantum accuracy or thermodynamic stability."
            ),
        },
        "symmetry_protocol": parent_protocol["symmetry_protocol"],
        "scope": {
            "source_coordinates": "etflow_hard_projection_f02",
            "all_atoms_explicit_including_hydrogen": True,
            "unconstrained_relaxation": True,
            "target_pg_used_by_calculator": False,
            "hard_projection_used_by_calculator": False,
            "full_panel_denominator_for_failures": True,
            "quality_claim_before_audit": False,
        },
        "identity": {
            "parent_ablation_protocol_sha256": _sha256(ABLATION_PROTOCOL),
            "parent_ablation_report_sha256": _sha256(ABLATION_RUN / "report.json"),
            "source_coordinates_sha256": _sha256(ABLATION_RUN / "coordinates.npz"),
            "source_point_group_report_sha256": _sha256(
                ABLATION_RUN / "point_group_report.json"
            ),
            "dataset_manifest_sha256": _sha256(
                ROOT / "generative_model/data/processed/v2/manifest.json"
            ),
            "source_sha256": {relative: _sha256(ROOT / relative) for relative in source_files},
        },
    }
    protocol["protocol_fingerprint"] = _fingerprint(protocol)
    _atomic_json(OUTPUT, protocol)
    return protocol


def main() -> None:
    result = build()
    print(json.dumps({
        "status": result["status"],
        "panel": result["panel"]["target_pg_counts"],
        "protocol_fingerprint": result["protocol_fingerprint"],
        "output": str(OUTPUT),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
