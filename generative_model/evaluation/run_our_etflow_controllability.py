"""Same-graph, shared-raw Target_PG controllability experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from generative_model.conformer.etflow_bridge import ETFlowRuntime
from generative_model.conformer.etflow_ef1_projection import kabsch_rmsd, pair_distance_mae
from generative_model.data.build_cof_package import save_deterministic_npz
from generative_model.data.cof_graph_dataset import COFGraphDataset
from generative_model.evaluation.our_etflow_ablation import (
    route_geometry_metrics,
    run_full_f02_route,
    strip_positions,
)
from generative_model.inference.generate_etflow_symmetric_xyz import (
    DEFAULT_CACHE,
    DEFAULT_PACKAGE,
    ROOT,
    _atomic_json,
    _fingerprint,
    _sha256,
)
from generative_model.inference.generate_etflow_symmetric_xyz_v4 import action_samples_v4
from generative_model.models.graph_pg_3d_projection import operation_errors


DEFAULT_PROTOCOL = ROOT / "generative_model/evaluation/our_etflow_controllability_protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "generative_model/runs/our_etflow_controllability_v1"


def run(
    protocol_path: Path,
    package_path: Path,
    output_dir: Path,
    cache: Path,
    device: str,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_fingerprint"] != _fingerprint(protocol):
        raise RuntimeError("controllability protocol fingerprint mismatch")
    for relative, expected in protocol["identity"]["source_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"controllability source hash mismatch: {relative}")
    v5_path = ROOT / "generative_model/inference/etflow_e3f02_protocol_v5.json"
    if _sha256(v5_path) != protocol["identity"]["v5_protocol_sha256"]:
        raise RuntimeError("v5 protocol hash mismatch")
    v5 = json.loads(v5_path.read_text(encoding="utf-8"))
    dataset = COFGraphDataset(package_path)
    runtime = ETFlowRuntime(
        model_name=v5["etflow"]["model_name"], device=device, cache=str(cache)
    )
    panel = protocol["panel"]["records"]
    targets = tuple(protocol["targets"])
    atom_counts = np.asarray(
        [len(dataset[int(item["package_index"])]["atomic_numbers"]) for item in panel],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(atom_counts))).astype("<i8")
    total_atoms = int(offsets[-1])
    arrays: dict[str, np.ndarray] = {
        "atom_offsets": offsets,
        "atomic_numbers": np.empty(total_atoms, dtype="<i8"),
        "canonical_positions": np.empty((total_atoms, 3), dtype="<f8"),
    }
    for target in targets:
        arrays[f"final_positions_{target}"] = np.full(
            (total_atoms, 3), np.nan, dtype="<f8"
        )
        arrays[f"success_{target}"] = np.zeros(len(panel), dtype=np.bool_)
    records = []
    started_all = perf_counter()
    for panel_id, item in enumerate(panel):
        canonical = dataset[int(item["package_index"])]
        start, end = map(int, offsets[panel_id : panel_id + 2])
        arrays["atomic_numbers"][start:end] = canonical["atomic_numbers"]
        arrays["canonical_positions"][start:end] = canonical["positions"]
        samples = {
            target: action_samples_v4(canonical, target, v5) for target in targets
        }
        prior_started = perf_counter()
        prediction = runtime.predict(
            samples[targets[0]][0],
            num_samples=int(v5["etflow"]["candidate_count"]),
            n_timesteps=int(v5["etflow"]["n_timesteps"]),
            seed=int(item["seed"]),
        )
        prior_seconds = perf_counter() - prior_started
        raw = [np.asarray(value, dtype=np.float64) for value in prediction["positions"]]
        record: dict[str, Any] = {
            "panel_id": panel_id,
            "package_index": int(item["package_index"]),
            "molecule_id": str(canonical["molecule_id"]),
            "seed": int(item["seed"]),
            "shared_raw_candidate_count": len(raw),
            "shared_raw_fingerprint": hashlib_arrays(raw),
            "targets": {},
            "pairwise_target_response": {},
        }
        outputs: dict[str, dict[str, Any]] = {}
        selected_samples: dict[str, dict[str, Any]] = {}
        for target in targets:
            try:
                output = run_full_f02_route(
                    samples[target], raw, target, int(item["seed"]), v5,
                    prior_seconds=prior_seconds,
                )
                metrics = route_geometry_metrics(
                    samples[target][0], output["positions"], canonical["positions"]
                )
                outputs[target] = output
                selected_samples[target] = samples[target][
                    int(output["selected_action_candidate_id"])
                ]
                arrays[f"final_positions_{target}"][start:end] = output["positions"]
                arrays[f"success_{target}"][panel_id] = True
                record["targets"][target] = {
                    "success": True,
                    **strip_positions(output),
                    **metrics,
                }
            except Exception as error:
                record["targets"][target] = {
                    "success": False,
                    "failure": f"{type(error).__name__}: {error}",
                }
        for left_id, left in enumerate(targets):
            for right in targets[left_id + 1 :]:
                key = f"{left}_vs_{right}"
                if left not in outputs or right not in outputs:
                    record["pairwise_target_response"][key] = {"success": False}
                    continue
                left_positions = outputs[left]["positions"]
                right_positions = outputs[right]["positions"]
                left_on_right = operation_errors(
                    left_positions,
                    selected_samples[right]["target_operation_matrices"],
                    selected_samples[right]["target_permutation_index"],
                )
                right_on_left = operation_errors(
                    right_positions,
                    selected_samples[left]["target_operation_matrices"],
                    selected_samples[left]["target_permutation_index"],
                )
                record["pairwise_target_response"][key] = {
                    "success": True,
                    "kabsch_rmsd_angstrom": kabsch_rmsd(left_positions, right_positions),
                    "pair_distance_mae_angstrom": pair_distance_mae(
                        left_positions, right_positions
                    ),
                    f"{left}_output_error_under_{right}_action_angstrom": float(
                        left_on_right["max_atom_error_angstrom"]
                    ),
                    f"{right}_output_error_under_{left}_action_angstrom": float(
                        right_on_left["max_atom_error_angstrom"]
                    ),
                }
        records.append(record)
        print(json.dumps({
            "completed": panel_id + 1,
            "total": len(panel),
            "package_index": record["package_index"],
            "target_success": {
                target: record["targets"][target]["success"] for target in targets
            },
        }, ensure_ascii=False), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = output_dir / "coordinates.npz"
    save_deterministic_npz(coordinates_path, arrays)
    pair_values: dict[str, list[float]] = {}
    for record in records:
        for pair, value in record["pairwise_target_response"].items():
            if value.get("success"):
                pair_values.setdefault(pair, []).append(value["kabsch_rmsd_angstrom"])
    threshold = float(protocol["response_gate"]["minimum_kabsch_rmsd_angstrom"])
    report = {
        "schema_version": "our-etflow-target-pg-controllability-run-v1",
        "status": "PASS_EXECUTION_AWAIT_INDEPENDENT_POINT_GROUP_AUDIT",
        "quality_claim": False,
        "records": records,
        "summary_pre_point_group": {
            "target_generation_success": {
                target: sum(record["targets"][target]["success"] for record in records)
                for target in targets
            },
            "pairwise_response": {
                pair: {
                    "count": len(values),
                    "mean_kabsch_rmsd_angstrom": float(np.mean(values)),
                    "minimum_kabsch_rmsd_angstrom": float(np.min(values)),
                    "responsive_fraction": float(np.mean(np.asarray(values) >= threshold)),
                }
                for pair, values in pair_values.items()
            },
            "minimum_response_threshold_angstrom": threshold,
            "raw_coordinates_shared_exactly_by_construction": True,
        },
        "runtime": {"total_wall_seconds": perf_counter() - started_all},
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "coordinates_sha256": _sha256(coordinates_path),
            "checkpoint_sha256": protocol["identity"]["checkpoint_sha256"],
        },
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def hashlib_arrays(arrays: list[np.ndarray]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in arrays:
        normalized = np.asarray(value, dtype="<f8")
        digest.update(np.asarray(normalized.shape, dtype="<i8").tobytes())
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run(
        args.protocol.resolve(), args.package.resolve(), args.output_dir.resolve(),
        args.cache.resolve(), args.device,
    )
    print(json.dumps({
        "status": report["status"],
        "summary": report["summary_pre_point_group"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
