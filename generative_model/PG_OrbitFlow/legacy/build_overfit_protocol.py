"""Freeze nested C2/C3 4->16->32 overfit panels and scientific gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_diverse(rows: pd.DataFrame, count: int) -> list[dict]:
    ordered = rows.sort_values(["Num_Atoms", "Package_Index"], kind="stable")
    selected = []
    used_indices: set[int] = set()
    # First cover distinct cores using the smallest molecule from each core.
    core_minima = (
        ordered.groupby("Core", sort=True, as_index=False)
        .first()
        .sort_values(["Num_Atoms", "Package_Index"], kind="stable")
    )
    for _, row in core_minima.iterrows():
        selected.append(row)
        used_indices.add(int(row["Package_Index"]))
        if len(selected) == count:
            break
    if len(selected) < count:
        for _, row in ordered.iterrows():
            if int(row["Package_Index"]) in used_indices:
                continue
            selected.append(row)
            used_indices.add(int(row["Package_Index"]))
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError(f"unable to select {count} diverse molecules")
    return [
        {
            "package_index": int(row["Package_Index"]),
            "molecule_id": str(row["Molecule_ID"]),
            "target_pg": str(row["Target_PG"]),
            "core": str(row["Core"]),
            "arm": str(row["Arm"]),
            "num_atoms": int(row["Num_Atoms"]),
            "contains_sn": bool(row["Contains_Sn"]),
        }
        for row in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir", type=Path, default=Path("generative_model/data/processed/v2")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--transport",
        choices=("independent", "cn_phase_aligned", "centralizer_averaged"),
        default="independent",
    )
    parser.add_argument(
        "--prior",
        choices=("projected_gaussian", "graph_harmonic"),
        default="projected_gaussian",
    )
    parser.add_argument("--harmonic-alpha", type=float, default=1.0)
    parser.add_argument("--centralizer-phase-count", type=int, default=24)
    parser.add_argument("--endpoint-weight", type=float, default=0.0)
    args = parser.parse_args()
    if args.harmonic_alpha != 1.0:
        raise ValueError("H1-H3 freeze requires harmonic_alpha=1.0")
    if args.centralizer_phase_count != 24:
        raise ValueError("H1-H3 freeze requires 24 centralizer phases")
    if args.endpoint_weight not in {0.0, 1.0}:
        raise ValueError("H1-H3 endpoint_weight must be exactly 0.0 or 1.0")
    if args.transport == "centralizer_averaged" and args.prior != "graph_harmonic":
        raise ValueError("centralizer transport requires graph_harmonic prior")
    if args.endpoint_weight > 0 and args.transport != "centralizer_averaged":
        raise ValueError("endpoint auxiliary is cumulative on centralizer transport")
    package_dir = args.package_dir.resolve()
    metadata = pd.read_csv(package_dir / "metadata.csv")
    eligible = metadata[
        (metadata["Split_IID"] == "train")
        & metadata["Target_PG"].isin(["C2", "C3"])
    ].copy()
    ordered = {
        point_group: _ordered_diverse(
            eligible[eligible["Target_PG"] == point_group], 16
        )
        for point_group in ("C2", "C3")
    }
    tiers = {}
    specifications = {
        "4": {
            "per_pg": 2,
            "steps": 1024,
            "gate": {
                "training_loss_window_ratio_max": 0.50,
                "fixed_time_endpoint_rmsd_mean_max_angstrom": 0.75,
                "raw_kabsch_rmsd_mean_max_angstrom": 1.00,
                "raw_pair_distance_mae_mean_max_angstrom": 0.60,
                "raw_bond_length_mae_mean_max_angstrom": 0.20,
                "raw_collision_free_fraction_min": 0.75,
                "raw_max_operation_atom_error_max_angstrom": 1e-4,
            },
        },
        "16": {
            "per_pg": 8,
            "steps": 2048,
            "gate": {
                "training_loss_window_ratio_max": 0.60,
                "fixed_time_endpoint_rmsd_mean_max_angstrom": 0.90,
                "raw_kabsch_rmsd_mean_max_angstrom": 1.20,
                "raw_pair_distance_mae_mean_max_angstrom": 0.70,
                "raw_bond_length_mae_mean_max_angstrom": 0.23,
                "raw_collision_free_fraction_min": 0.75,
                "raw_max_operation_atom_error_max_angstrom": 1e-4,
            },
        },
        "32": {
            "per_pg": 16,
            "steps": 4096,
            "gate": {
                "training_loss_window_ratio_max": 0.65,
                "fixed_time_endpoint_rmsd_mean_max_angstrom": 1.00,
                "raw_kabsch_rmsd_mean_max_angstrom": 1.30,
                "raw_pair_distance_mae_mean_max_angstrom": 0.75,
                "raw_bond_length_mae_mean_max_angstrom": 0.25,
                "raw_collision_free_fraction_min": 0.70,
                "raw_max_operation_atom_error_max_angstrom": 1e-4,
            },
        },
    }
    for tier, specification in specifications.items():
        per_pg = specification["per_pg"]
        records = ordered["C2"][:per_pg] + ordered["C3"][:per_pg]
        tiers[tier] = {
            "molecule_count": len(records),
            "records": records,
            "training": {
                "additional_optimizer_steps": specification["steps"],
                "batch_size_molecules": 4,
                "learning_rate": 0.0003,
                "weight_decay": 0.000001,
                "gradient_clip_norm": 5.0,
                "checkpoint_every": 256,
                "point_group_balanced_exposure": True,
            },
            "evaluation": {
                "flow_times": [0.1, 0.25, 0.5, 0.75, 0.9],
                "raw_seeds_per_molecule": 2,
                "raw_ode_steps": 50,
                "raw_ode_method": "heun",
            },
            "gate": specification["gate"],
        }
    protocol = {
        "schema_version": "pg-orbitflow-overfit-protocol-v1",
        "scientific_question": "Can the native group-constrained vector field memorize and raw-sample nested C2/C3 panels before any external pretraining?",
        "package_dir": str(package_dir),
        "canonical_manifest_sha256": _sha256(package_dir / "manifest.json"),
        "model": {
            "hidden_dim": 128,
            "layers": 6,
            "radial_dim": 16,
            "radial_max": 6.0,
            "parameter_count_expected": 740424,
        },
        "data": {
            "split": "iid_train_only",
            "coordinate_scale_angstrom": 3.0,
            "point_groups": ["C2", "C3"],
            "test_used": False,
            "validation_used": False,
            "core_ood_used": False,
            "nested_panels": True,
        },
        "objective": {
            "bond_weight": 0.25,
            "overlap_weight": 0.1,
            "symmetry_weight": 1.0,
            "endpoint_weight": args.endpoint_weight,
        },
        "prior": {
            "type": args.prior,
            "harmonic_alpha": args.harmonic_alpha,
            "per_sample_rms_normalization": args.prior == "projected_gaussian",
            "coordinate_scale_angstrom": 3.0,
        },
        "transport": {
            "type": args.transport,
            "target_phase_aligned_to_prior": args.transport == "cn_phase_aligned",
            "alignment_group": (
                "SO(2) centralizer of the supplied Cn action"
                if args.transport == "cn_phase_aligned"
                else "full SO(3) centralizer Haar quadrature"
                if args.transport == "centralizer_averaged"
                else "none"
            ),
            "centralizer_phase_count": args.centralizer_phase_count,
            "c2_candidate_count": 2 * args.centralizer_phase_count,
            "c3_candidate_count": args.centralizer_phase_count,
            "posterior_density": "graph_laplacian_quadratic"
            if args.transport == "centralizer_averaged"
            else "not_applicable",
        },
        "seed": 20260828,
        "progression": {
            "order": [4, 16, 32],
            "stop_on_failure": True,
            "tier_4_initialization": "random",
            "tier_16_initialization": "tier_4_last_model_state",
            "tier_32_initialization": "tier_16_last_model_state",
            "optimizer_reinitialized_each_tier": True,
        },
        "tiers": tiers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE_PG_ORBITFLOW_OVERFIT_PROTOCOL {args.output.resolve()}")


if __name__ == "__main__":
    main()
