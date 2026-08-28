"""All-target inference v4 with S4 and D6h graph-action selection."""

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
    ROOT, DEFAULT_CACHE, DEFAULT_PACKAGE, _atomic_json, _finish_candidate,
    _fingerprint, _select_candidate, _sha256, _symbols, _write_xyz,
)
from generative_model.inference.generate_etflow_symmetric_xyz_v2 import sample_from_canonical_v2
from generative_model.inference.generate_etflow_symmetric_xyz_v3 import (
    _rank_combinations, _sample_with_action,
)
from generative_model.symmetry.graph_action_candidates_v2 import (
    recover_d6h_graph_action_candidates, recover_s4_graph_action_candidates,
)
from generative_model.symmetry.graph_action_extended import SUPPORTED_TARGETS


DEFAULT_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_protocol_v4.json"
DEFAULT_OUTPUT = ROOT / "generative_model/runs/etflow_e3f02_inference_v4"


def action_samples_v4(
    canonical: dict[str, Any], target_pg: str, protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    base = sample_from_canonical_v2(canonical, target_pg)
    if target_pg not in {"S4", "D6h"}:
        return [base]
    settings = protocol["rare_action_selection"][target_pg]
    recover = (
        recover_s4_graph_action_candidates
        if target_pg == "S4" else recover_d6h_graph_action_candidates
    )
    actions = recover(
        canonical["atomic_numbers"], canonical["formal_charges"],
        canonical["radical_electrons"], canonical["bond_index"],
        canonical["bond_types"],
        maximum_candidates=int(settings["maximum_actions"]),
        maximum_search_matches=int(settings["maximum_search_matches"]),
    )
    return [
        _sample_with_action(base, action, action_id)
        for action_id, action in enumerate(actions)
    ]


def run_prediction_from_samples(
    samples: list[dict[str, Any]], target_pg: str, seed: int,
    runtime: ETFlowRuntime, protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not samples:
        raise ValueError("action samples 不能为空")
    prediction = runtime.predict(
        samples[0], num_samples=int(protocol["etflow"]["candidate_count"]),
        n_timesteps=int(protocol["etflow"]["n_timesteps"]), seed=int(seed),
    )
    ranked, projection_failures = _rank_combinations(
        list(prediction["positions"]), samples, protocol
    )
    optimize_count = (
        int(protocol["rare_action_selection"][target_pg]["optimize_top_combinations"])
        if target_pg in {"S4", "D6h"} else len(ranked)
    )
    completed, coordinates, optimization_failures = [], [], []
    for rank, choice in enumerate(ranked[:optimize_count]):
        try:
            record, values = _finish_candidate(
                samples[choice["action_candidate_id"]],
                prediction["positions"][choice["raw_candidate_id"]],
                rank, int(seed), protocol,
            )
            record.update({
                "preselection_rank": rank,
                "raw_candidate_id": int(choice["raw_candidate_id"]),
                "action_candidate_id": int(choice["action_candidate_id"]),
                "preselection_metrics": choice,
            })
            completed.append(record)
            coordinates.append(values)
        except Exception as error:
            optimization_failures.append({
                "preselection_rank": rank,
                "raw_candidate_id": int(choice["raw_candidate_id"]),
                "action_candidate_id": int(choice["action_candidate_id"]),
                "failure": f"{type(error).__name__}: {error}",
            })
    selected_index = _select_candidate(completed)
    selected_record = completed[selected_index]
    selected_sample = samples[selected_record["action_candidate_id"]]
    audit = {
        "action_candidate_count": len(samples),
        "raw_candidate_count": len(prediction["positions"]),
        "projection_combination_count": len(ranked),
        "optimized_combination_count": len(completed),
        "projection_failures": projection_failures,
        "optimization_failures": optimization_failures,
        "ranked_projection_top": ranked[:min(20, len(ranked))],
        "completed_records": completed,
        "selected": selected_record,
        "bridge_audit": prediction["bridge_audit"],
        "feature_audit": prediction["feature_audit"],
        "selected_graph_action_audit": selected_sample["graph_action_audit"],
    }
    values = coordinates[selected_index]
    arrays = {
        "atomic_numbers": selected_sample["atomic_numbers"].astype("<i8"),
        "bond_index": selected_sample["bond_index"].astype("<i8"),
        "bond_types": selected_sample["bond_types"].astype("<i8"),
        "operation_matrices": np.asarray(selected_sample["target_operation_matrices"], dtype="<f8"),
        "permutation_index": np.asarray(selected_sample["target_permutation_index"], dtype="<i8"),
        "orbit_id": np.asarray(selected_sample["target_orbit_id"], dtype="<i8"),
        "orbit_size": np.asarray(selected_sample["target_orbit_size"], dtype="<i8"),
        "raw_positions": np.asarray(values["raw"], dtype="<f8"),
        "projected_positions": np.asarray(values["projected"], dtype="<f8"),
        "final_positions": np.asarray(values["final"], dtype="<f8"),
    }
    return audit, arrays


def run_prediction_v4(
    canonical: dict[str, Any], target_pg: str, seed: int,
    runtime: ETFlowRuntime, protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    samples = action_samples_v4(canonical, target_pg, protocol)
    return run_prediction_from_samples(samples, target_pg, seed, runtime, protocol)


def generate_v4(
    *, package: Path, package_index: int, target_pg: str, seed: int,
    protocol_path: Path, output_dir: Path, cache: Path, device: str,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_fingerprint"] != _fingerprint(protocol):
        raise RuntimeError("ET-Flow E3/F0.2 v4 protocol fingerprint 不一致")
    for relative, expected in protocol["identity"]["source_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"v4 inference source hash 不一致: {relative}")
    checkpoint = cache / "drugs-o3.ckpt"
    if _sha256(checkpoint) != protocol["identity"]["checkpoint_sha256"]:
        raise RuntimeError("ET-Flow checkpoint SHA 不一致")
    if target_pg not in protocol["supported_target_pgs"]:
        raise ValueError(f"v4 严格推理仅支持 {protocol['supported_target_pgs']}")
    canonical = COFGraphDataset(package)[int(package_index)]
    runtime = ETFlowRuntime(
        model_name=protocol["etflow"]["model_name"], device=device, cache=str(cache)
    )
    audit, arrays = run_prediction_v4(canonical, target_pg, seed, runtime, protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = output_dir / "coordinates.npz"
    save_deterministic_npz(coordinates_path, arrays)
    symbols = _symbols(arrays["atomic_numbers"])
    for stage in ("raw", "projected", "final"):
        _write_xyz(
            output_dir / f"{stage}.xyz", symbols, arrays[f"{stage}_positions"],
            f"molecule_id={canonical['molecule_id']} package_index={package_index} target_pg={target_pg} stage={stage} seed={seed}",
        )
    selected = audit["selected"]
    report = {
        "schema_version": "etflow-e3f02-known-graph-inference-v4",
        "status": "PASS_ETFLOW_E3F02_V4_XYZ_AWAIT_INDEPENDENT_POINT_GROUP_AUDIT",
        "passed_geometry_gate": True,
        "package_index": int(package_index),
        "molecule_id": str(canonical["molecule_id"]),
        "source_split": str(canonical["metadata"]["Split_IID"]),
        "requested_target_pg": target_pg,
        "canonical_dataset_target_pg": str(canonical["metadata"]["Target_PG"]),
        "atom_count": len(arrays["atomic_numbers"]),
        "sn_count": int(np.sum(arrays["atomic_numbers"] == 50)),
        "seed": int(seed),
        "selected_candidate_id": int(selected["candidate_id"]),
        "selected": selected,
        "action_selection_audit": audit,
        "scope": protocol["scope"],
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "checkpoint_sha256": _sha256(checkpoint),
            "coordinates_sha256": _sha256(coordinates_path),
        },
        "outputs": {
            "coordinates_npz": "coordinates.npz", "raw_xyz": "raw.xyz",
            "projected_xyz": "projected.xyz", "final_xyz": "final.xyz",
        },
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--package-index", type=int, required=True)
    parser.add_argument("--target-pg", choices=SUPPORTED_TARGETS, required=True)
    parser.add_argument("--seed", type=int, default=2026083201)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = generate_v4(
        package=args.package.resolve(), package_index=args.package_index,
        target_pg=args.target_pg, seed=args.seed,
        protocol_path=args.protocol.resolve(), output_dir=args.output_dir.resolve(),
        cache=args.cache.resolve(), device=args.device,
    )
    print(json.dumps({
        "status": report["status"], "package_index": report["package_index"],
        "requested_target_pg": report["requested_target_pg"],
        "action_candidate_count": report["action_selection_audit"]["action_candidate_count"],
        "selected_action_candidate_id": report["selected"]["action_candidate_id"],
        "final_minimum_pair_distance_angstrom": report["selected"]["final_minimum_pair_distance_angstrom"],
        "final_maximum_operation_error_angstrom": report["selected"]["final_maximum_operation_error_angstrom"],
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
