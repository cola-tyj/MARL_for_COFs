"""Resumable dataset-level ET-Flow -> E3 -> F0.2 IID-test runner."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from generative_model.conformer.etflow_bridge import ETFlowRuntime
from generative_model.data.build_cof_package import save_deterministic_npz
from generative_model.data.cof_graph_dataset import COFGraphDataset
from generative_model.inference.generate_etflow_symmetric_xyz import (
    _atomic_json,
    _finish_candidate,
    _sample_from_canonical,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT / "generative_model/data/processed/v2"
DEFAULT_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_iidtest_protocol_v1.json"
DEFAULT_INFERENCE_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_protocol_v1.json"
DEFAULT_CACHE = ROOT / "generative_model/checkpoints/etflow"
DEFAULT_OUTPUT = ROOT / "generative_model/runs/etflow_e3f02_iidtest_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("protocol_fingerprint", None)
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _geometry_pass(record: dict[str, Any]) -> bool:
    return bool(record.get("status") == "success")


def summarize_geometry(
    records: list[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, bool]]:
    expected = list(map(int, protocol["panel"]["package_indices"]))
    if [int(record["package_index"]) for record in records] != expected:
        raise RuntimeError("geometry record order 与冻结 panel 不一致")
    by_pg = {point_group: [] for point_group in ("C2", "C3")}
    successes = []
    action_successes = []
    failure_counts: Counter[str] = Counter()
    for record in records:
        point_group = str(record["requested_target_pg"])
        passed = _geometry_pass(record)
        by_pg[point_group].append(passed)
        successes.append(passed)
        action_successes.append(bool(record["graph_action_recovered"]))
        if not passed:
            failure_counts[str(record["failure_stage"])] += 1
    metrics = {
        "molecule_count": len(records),
        "graph_action_recovery_fraction": float(np.mean(action_successes)),
        "end_to_end_success_fraction": float(np.mean(successes)),
        "successful_molecule_count": int(sum(successes)),
        "failed_molecule_count": int(len(successes) - sum(successes)),
        "c2_success_fraction": float(np.mean(by_pg["C2"])),
        "c3_success_fraction": float(np.mean(by_pg["C3"])),
        "failure_stage_counts": dict(sorted(failure_counts.items())),
    }
    thresholds = protocol["geometry_gate_thresholds"]
    checks = {
        "molecule_count_exact": len(records) == protocol["panel"]["molecule_count"],
        "graph_action_recovery_fraction_min": (
            metrics["graph_action_recovery_fraction"]
            >= thresholds["graph_action_recovery_fraction_min"]
        ),
        "end_to_end_success_fraction_min": (
            metrics["end_to_end_success_fraction"]
            >= thresholds["end_to_end_success_fraction_min"]
        ),
        "c2_success_fraction_min": (
            metrics["c2_success_fraction"] >= thresholds["c2_success_fraction_min"]
        ),
        "c3_success_fraction_min": (
            metrics["c3_success_fraction"] >= thresholds["c3_success_fraction_min"]
        ),
        "reference_xyz_never_used": all(
            not record["scope"]["reference_xyz_used"] for record in records
        ),
        "stored_action_never_used": all(
            not record["scope"]["stored_symmetry_action_used"] for record in records
        ),
        "etkdg_never_used": all(
            not record["scope"]["etkdg_used"] for record in records
        ),
    }
    return metrics, checks


def _success_arrays(sample: dict[str, Any], values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "atomic_numbers": np.asarray(sample["atomic_numbers"], dtype="<i8"),
        "bond_index": np.asarray(sample["bond_index"], dtype="<i8"),
        "bond_types": np.asarray(sample["bond_types"], dtype="<i8"),
        "operation_matrices": np.asarray(
            sample["target_operation_matrices"], dtype="<f8"
        ),
        "permutation_index": np.asarray(
            sample["target_permutation_index"], dtype="<i8"
        ),
        "orbit_id": np.asarray(sample["target_orbit_id"], dtype="<i8"),
        "orbit_size": np.asarray(sample["target_orbit_size"], dtype="<i8"),
        "raw_positions": np.asarray(values["raw"], dtype="<f8"),
        "projected_positions": np.asarray(values["projected"], dtype="<f8"),
        "final_positions": np.asarray(values["final"], dtype="<f8"),
    }


def run_molecule(
    canonical: dict[str, Any],
    runtime: ETFlowRuntime,
    batch_protocol: dict[str, Any],
    inference_protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    package_index = int(canonical["package_index"])
    target_pg = str(canonical["metadata"]["Target_PG"])
    base = {
        "package_index": package_index,
        "molecule_id": str(canonical["molecule_id"]),
        "requested_target_pg": target_pg,
        "source_split": str(canonical["metadata"]["Split_IID"]),
        "atom_count": len(canonical["atomic_numbers"]),
        "seed": int(batch_protocol["sampling"]["base_seed"] + package_index),
        "scope": dict(batch_protocol["scope"]),
        "graph_action_recovered": False,
        "status": "failure",
        "failure_stage": None,
        "failure": None,
        "candidate_records": [],
        "candidate_failures": [],
    }
    try:
        sample = _sample_from_canonical(canonical, target_pg)
        base["graph_action_recovered"] = True
        base["graph_action_audit"] = sample["graph_action_audit"]
    except Exception as error:
        base["failure_stage"] = "graph_action"
        base["failure"] = f"{type(error).__name__}: {error}"
        return base, None
    try:
        prediction = runtime.predict(
            sample,
            num_samples=int(batch_protocol["sampling"]["candidates_per_molecule"]),
            n_timesteps=int(batch_protocol["sampling"]["n_timesteps"]),
            seed=int(base["seed"]),
        )
        base["bridge_audit"] = prediction["bridge_audit"]
        base["feature_audit"] = prediction["feature_audit"]
    except Exception as error:
        base["failure_stage"] = "etflow_sampling"
        base["failure"] = f"{type(error).__name__}: {error}"
        return base, None

    completed_records: list[dict[str, Any]] = []
    coordinate_records: list[dict[str, np.ndarray]] = []
    for candidate_id, raw in enumerate(prediction["positions"]):
        try:
            record, coordinates = _finish_candidate(
                sample,
                raw,
                candidate_id,
                int(base["seed"]),
                inference_protocol,
            )
            completed_records.append(record)
            coordinate_records.append(coordinates)
        except Exception as error:
            base["candidate_failures"].append({
                "candidate_id": int(candidate_id),
                "failure": f"{type(error).__name__}: {error}",
            })
    base["candidate_records"] = completed_records
    eligible = [
        index for index, record in enumerate(completed_records)
        if record["passed_geometry_gate"]
    ]
    if not eligible:
        base["failure_stage"] = "strict_candidate_gate"
        base["failure"] = "no ET-Flow candidate passed E3/F0.2 geometry thresholds"
        return base, None
    selected = min(eligible, key=lambda index: (
        completed_records[index]["final_total_objective_kcal_mol"],
        completed_records[index]["candidate_id"],
    ))
    base["status"] = "success"
    base["failure_stage"] = None
    base["failure"] = None
    base["selected_candidate_id"] = int(
        completed_records[selected]["candidate_id"]
    )
    base["selected"] = completed_records[selected]
    return base, _success_arrays(sample, coordinate_records[selected])


def run(
    package: Path,
    protocol_path: Path,
    inference_protocol_path: Path,
    output_dir: Path,
    cache: Path,
    device: str,
    maximum_molecules: int | None = None,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_fingerprint"] != _fingerprint(protocol):
        raise RuntimeError("IID-test protocol fingerprint 不一致")
    if _sha256(inference_protocol_path) != protocol["identity"]["inference_protocol_sha256"]:
        raise RuntimeError("IID-test/inference protocol SHA 不一致")
    for relative, expected in protocol["identity"]["source_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"IID-test source hash 不一致: {relative}")
    checkpoint = cache / "drugs-o3.ckpt"
    if _sha256(checkpoint) != protocol["identity"]["checkpoint_sha256"]:
        raise RuntimeError("ET-Flow checkpoint SHA 不一致")

    output_dir.mkdir(parents=True, exist_ok=True)
    molecule_dir = output_dir / "molecules"
    molecule_dir.mkdir(exist_ok=True)
    protocol_sha = _sha256(protocol_path)
    dataset = COFGraphDataset(package)
    runtime = ETFlowRuntime(model_name="drugs-o3", device=device, cache=str(cache))
    panel = list(map(int, protocol["panel"]["package_indices"]))
    if maximum_molecules is not None:
        panel = panel[: int(maximum_molecules)]
    records: list[dict[str, Any]] = []
    for progress, package_index in enumerate(panel, start=1):
        json_path = molecule_dir / f"{package_index:06d}.json"
        npz_path = molecule_dir / f"{package_index:06d}.npz"
        if json_path.exists():
            record = json.loads(json_path.read_text(encoding="utf-8"))
            if record.get("protocol_sha256") != protocol_sha:
                raise RuntimeError(f"resume protocol SHA 不一致: {package_index}")
            if record["status"] == "success":
                if not npz_path.exists() or record.get("coordinates_sha256") != _sha256(npz_path):
                    raise RuntimeError(f"resume coordinate identity 不一致: {package_index}")
            elif npz_path.exists():
                raise RuntimeError(f"failed record 不应具有 NPZ: {package_index}")
        else:
            record, arrays = run_molecule(
                dataset[package_index], runtime, protocol, inference
            )
            record["protocol_sha256"] = protocol_sha
            if arrays is not None:
                save_deterministic_npz(npz_path, arrays)
                record["coordinates_sha256"] = _sha256(npz_path)
            else:
                record["coordinates_sha256"] = None
            _atomic_json(json_path, record)
        records.append(record)
        print(json.dumps({
            "completed": progress,
            "total": len(panel),
            "package_index": package_index,
            "target_pg": record["requested_target_pg"],
            "status": record["status"],
            "failure_stage": record["failure_stage"],
        }, ensure_ascii=False), flush=True)

    if len(panel) != protocol["panel"]["molecule_count"]:
        partial = {
            "schema_version": "etflow-e3f02-iidtest-partial-v1",
            "status": "PARTIAL_ETFLOW_E3F02_IIDTEST_RESUMABLE",
            "completed_count": len(records),
            "total_count": protocol["panel"]["molecule_count"],
            "protocol_sha256": protocol_sha,
        }
        _atomic_json(output_dir / "partial_status.json", partial)
        return partial

    metrics, checks = summarize_geometry(records, protocol)
    passed = all(checks.values())
    report = {
        "schema_version": "etflow-e3f02-iidtest-geometry-v1",
        "status": (
            protocol["decision"]["geometry_pass_status"]
            if passed else protocol["decision"]["fail_status"]
        ),
        "passed_geometry_gate": passed,
        "metrics": metrics,
        "checks": checks,
        "records": records,
        "scope": protocol["scope"],
        "identity": {
            "protocol_sha256": protocol_sha,
            "inference_protocol_sha256": _sha256(inference_protocol_path),
            "checkpoint_sha256": _sha256(checkpoint),
            "runner_source_sha256": _sha256(Path(__file__)),
        },
    }
    _atomic_json(output_dir / "geometry_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--inference-protocol", type=Path, default=DEFAULT_INFERENCE_PROTOCOL
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-molecules", type=int)
    args = parser.parse_args()
    report = run(
        args.package.resolve(),
        args.protocol.resolve(),
        args.inference_protocol.resolve(),
        args.output_dir.resolve(),
        args.cache.resolve(),
        str(args.device),
        args.maximum_molecules,
    )
    print(json.dumps({
        key: report[key] for key in report
        if key in {"status", "passed_geometry_gate", "metrics", "checks", "completed_count", "total_count"}
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
