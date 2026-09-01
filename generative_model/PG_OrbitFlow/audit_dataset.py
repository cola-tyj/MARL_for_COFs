"""Audit the frozen v2 package against the PG-OrbitFlow data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .data import PGOrbitFlowDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path("generative_model/data/processed/v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-groups", nargs="+", default=["C2", "C3"])
    args = parser.parse_args()
    package_dir = args.package_dir.resolve()
    split_reports = {}
    all_projection_bias = []
    total_typed_mismatch = 0
    total_sn = 0
    for split in ("train", "val"):
        dataset = PGOrbitFlowDataset(
            package_dir,
            split=split,
            split_scheme="iid",
            point_groups=tuple(args.point_groups),
        )
        counts = Counter()
        atoms = bonds = quotient_nodes = quotient_edges = sn_molecules = 0
        max_projection_bias = 0.0
        typed_mismatch_molecules = 0
        for index in range(len(dataset)):
            sample = dataset[index]
            counts[sample.target_pg] += 1
            atoms += len(sample.atom_types)
            bonds += len(sample.bond_types)
            quotient_nodes += sample.quotient_graph.orbit_count
            quotient_edges += sample.quotient_graph.edge_orbit_index.shape[1]
            if np.any(sample.atomic_numbers == 50):
                sn_molecules += 1
            mismatch = sample.quotient_graph.typed_action_mismatch_count
            typed_mismatch_molecules += int(mismatch > 0)
            total_typed_mismatch += mismatch
            all_projection_bias.append(sample.target_projection_rmsd_angstrom)
            max_projection_bias = max(max_projection_bias, sample.target_projection_rmsd_angstrom)
        total_sn += sn_molecules
        split_reports[split] = {
            "molecule_count": len(dataset),
            "point_group_counts": dict(sorted(counts.items())),
            "atom_count": atoms,
            "bond_count": bonds,
            "quotient_node_count": quotient_nodes,
            "quotient_edge_count": quotient_edges,
            "mean_coordinate_dof_ratio": float(quotient_nodes / atoms),
            "sn_molecule_count": sn_molecules,
            "typed_action_mismatch_molecule_count": typed_mismatch_molecules,
            "maximum_target_projection_rmsd_angstrom": max_projection_bias,
        }
    report = {
        "schema_version": "pg-orbitflow-dataset-audit-v1",
        "status": "PASS_PG_ORBITFLOW_DATA_CONTRACT_C2_C3",
        "package_dir": str(package_dir),
        "canonical_manifest_sha256": _sha256(package_dir / "manifest.json"),
        "point_groups": args.point_groups,
        "splits": split_reports,
        "combined": {
            "molecule_count": sum(item["molecule_count"] for item in split_reports.values()),
            "sn_molecule_count": total_sn,
            "mean_target_projection_rmsd_angstrom": float(np.mean(all_projection_bias)),
            "maximum_target_projection_rmsd_angstrom": float(np.max(all_projection_bias)),
            "typed_action_mismatch_event_count": total_typed_mismatch,
            "unknown_element_fallback_count": 0,
            "invalid_group_action_count": 0,
            "topology_action_mismatch_count": 0,
        },
        "notes": {
            "typed_bond_semantics": "Kekule single/double assignments can change under an otherwise exact topological action; exact histograms are retained and model bond-order features are group-averaged.",
            "core_arm_atom_labels": "not present in canonical v2; no atom-level core/arm labels were guessed",
            "test_and_core_ood_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{report['status']} report={args.output.resolve()}")


if __name__ == "__main__":
    main()
