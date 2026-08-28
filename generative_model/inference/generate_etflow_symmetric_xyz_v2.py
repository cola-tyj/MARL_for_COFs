"""All-target known-graph ET-Flow -> E3 -> F0.2 XYZ inference (v2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from generative_model.conformer.etflow_bridge import ETFlowRuntime
from generative_model.data.build_cof_package import save_deterministic_npz
from generative_model.data.cof_graph_dataset import COFGraphDataset
from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT,
    DEFAULT_CACHE,
    DEFAULT_PACKAGE,
    _atomic_json,
    _finish_candidate,
    _fingerprint,
    _select_candidate,
    _sha256,
    _symbols,
    _write_xyz,
)
from generative_model.symmetry.graph_action_extended import (
    SUPPORTED_TARGETS,
    recover_full_target_graph_action,
)


DEFAULT_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_protocol_v2.json"
DEFAULT_OUTPUT = ROOT / "generative_model/runs/etflow_e3f02_inference_v2"


def sample_from_canonical_v2(
    canonical: dict[str, Any], target_pg: str
) -> dict[str, Any]:
    action = recover_full_target_graph_action(
        canonical["atomic_numbers"],
        canonical["formal_charges"],
        canonical["radical_electrons"],
        canonical["bond_index"],
        canonical["bond_types"],
        str(target_pg),
    )
    aromaticity = np.zeros(len(canonical["atomic_numbers"]), dtype=np.bool_)
    for (left, right), kind in zip(
        canonical["bond_index"].T, canonical["bond_types"], strict=True
    ):
        if int(kind) == 4:
            aromaticity[int(left)] = True
            aromaticity[int(right)] = True
    return {
        "package_index": int(canonical["package_index"]),
        "molecule_id": str(canonical["molecule_id"]),
        "target_pg": str(target_pg),
        "atomic_numbers": np.asarray(canonical["atomic_numbers"], dtype=np.int64),
        "formal_charges": np.asarray(canonical["formal_charges"], dtype=np.int64),
        "radical_electrons": np.asarray(
            canonical["radical_electrons"], dtype=np.int64
        ),
        "bond_index": np.asarray(canonical["bond_index"], dtype=np.int64),
        "bond_types": np.asarray(canonical["bond_types"], dtype=np.int64),
        "aromaticity": aromaticity,
        "target_operation_matrices": action["operation_matrices"],
        "target_permutation_index": action["permutation_index"],
        "target_orbit_id": action["orbit_id"],
        "target_orbit_size": action["orbit_size"],
        "graph_action_audit": {
            "recovery": action["recovery_audit"],
            "validation": action["validation"],
        },
    }


def generate_v2(
    *,
    package: Path,
    package_index: int,
    target_pg: str,
    seed: int,
    protocol_path: Path,
    output_dir: Path,
    cache: Path,
    device: str,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_fingerprint"] != _fingerprint(protocol):
        raise RuntimeError("ET-Flow E3/F0.2 v2 protocol fingerprint 不一致")
    for relative, expected in protocol["identity"]["source_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"v2 inference source hash 不一致: {relative}")
    checkpoint = cache / "drugs-o3.ckpt"
    if _sha256(checkpoint) != protocol["identity"]["checkpoint_sha256"]:
        raise RuntimeError("ET-Flow drugs-o3 checkpoint SHA 不一致")
    if target_pg not in protocol["supported_target_pgs"]:
        raise ValueError(f"v2 严格推理仅支持 {protocol['supported_target_pgs']}")

    dataset = COFGraphDataset(package)
    canonical = dataset[int(package_index)]
    sample = sample_from_canonical_v2(canonical, target_pg)
    runtime = ETFlowRuntime(
        model_name=str(protocol["etflow"]["model_name"]),
        device=device,
        cache=str(cache),
    )
    prediction = runtime.predict(
        sample,
        num_samples=int(protocol["etflow"]["candidate_count"]),
        n_timesteps=int(protocol["etflow"]["n_timesteps"]),
        seed=int(seed),
    )
    records = []
    coordinates = []
    failures = []
    for candidate_id, raw in enumerate(prediction["positions"]):
        try:
            record, values = _finish_candidate(
                sample, raw, candidate_id, int(seed), protocol
            )
            records.append(record)
            coordinates.append(values)
        except Exception as error:
            failures.append({
                "candidate_id": int(candidate_id),
                "failure": f"{type(error).__name__}: {error}",
            })
    selected_index = _select_candidate(records)
    selected_record = records[selected_index]
    selected = coordinates[selected_index]
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "atomic_numbers": sample["atomic_numbers"].astype("<i8"),
        "bond_index": sample["bond_index"].astype("<i8"),
        "bond_types": sample["bond_types"].astype("<i8"),
        "operation_matrices": np.asarray(
            sample["target_operation_matrices"], dtype="<f8"
        ),
        "permutation_index": np.asarray(
            sample["target_permutation_index"], dtype="<i8"
        ),
        "orbit_id": np.asarray(sample["target_orbit_id"], dtype="<i8"),
        "orbit_size": np.asarray(sample["target_orbit_size"], dtype="<i8"),
        "raw_positions": np.asarray(selected["raw"], dtype="<f8"),
        "projected_positions": np.asarray(selected["projected"], dtype="<f8"),
        "final_positions": np.asarray(selected["final"], dtype="<f8"),
    }
    coordinates_path = output_dir / "coordinates.npz"
    save_deterministic_npz(coordinates_path, arrays)
    symbols = _symbols(sample["atomic_numbers"])
    for stage in ("raw", "projected", "final"):
        _write_xyz(
            output_dir / f"{stage}.xyz",
            symbols,
            selected[stage],
            (
                f"molecule_id={sample['molecule_id']} package_index={package_index} "
                f"target_pg={target_pg} stage={stage} seed={seed}"
            ),
        )
    report = {
        "schema_version": "etflow-e3f02-known-graph-inference-v2",
        "status": "PASS_ETFLOW_E3F02_V2_XYZ_AWAIT_INDEPENDENT_POINT_GROUP_AUDIT",
        "passed_geometry_gate": True,
        "package_index": int(package_index),
        "molecule_id": sample["molecule_id"],
        "source_split": str(canonical["metadata"]["Split_IID"]),
        "requested_target_pg": str(target_pg),
        "canonical_dataset_target_pg": str(canonical["metadata"]["Target_PG"]),
        "atom_count": len(sample["atomic_numbers"]),
        "sn_count": int(np.sum(sample["atomic_numbers"] == 50)),
        "seed": int(seed),
        "candidate_count_requested": int(protocol["etflow"]["candidate_count"]),
        "candidate_count_completed": len(records),
        "selected_candidate_id": selected_record["candidate_id"],
        "candidate_selection": protocol["candidate_selection"],
        "candidate_records": records,
        "candidate_failures": failures,
        "selected": selected_record,
        "bridge_audit": prediction["bridge_audit"],
        "feature_audit": prediction["feature_audit"],
        "graph_action_audit": sample["graph_action_audit"],
        "scope": protocol["scope"],
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "checkpoint_sha256": _sha256(checkpoint),
            "coordinates_sha256": _sha256(coordinates_path),
        },
        "outputs": {
            "coordinates_npz": "coordinates.npz",
            "raw_xyz": "raw.xyz",
            "projected_xyz": "projected.xyz",
            "final_xyz": "final.xyz",
        },
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--package-index", type=int, required=True)
    parser.add_argument("--target-pg", choices=SUPPORTED_TARGETS, required=True)
    parser.add_argument("--seed", type=int, default=2026082901)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = generate_v2(
        package=args.package.resolve(),
        package_index=int(args.package_index),
        target_pg=str(args.target_pg),
        seed=int(args.seed),
        protocol_path=args.protocol.resolve(),
        output_dir=args.output_dir.resolve(),
        cache=args.cache.resolve(),
        device=str(args.device),
    )
    print(json.dumps({
        "status": report["status"],
        "package_index": report["package_index"],
        "requested_target_pg": report["requested_target_pg"],
        "selected_candidate_id": report["selected_candidate_id"],
        "final_minimum_pair_distance_angstrom": report["selected"][
            "final_minimum_pair_distance_angstrom"
        ],
        "final_maximum_operation_error_angstrom": report["selected"][
            "final_maximum_operation_error_angstrom"
        ],
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
