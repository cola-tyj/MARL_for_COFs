"""Resumable ET-Flow -> E3 -> F0.2 runner for all canonical S4/D6h."""

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
from generative_model.inference.generate_etflow_symmetric_xyz import _atomic_json
from generative_model.inference.generate_etflow_symmetric_xyz_v3 import run_prediction


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT / "generative_model/data/processed/v2"
DEFAULT_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_rare_targets_protocol_v1.json"
DEFAULT_INFERENCE_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_protocol_v3.json"
DEFAULT_CACHE = ROOT / "generative_model/checkpoints/etflow"
DEFAULT_OUTPUT = ROOT / "generative_model/runs/etflow_e3f02_rare_targets_v1"
TARGETS = ("S4", "D6h")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("protocol_fingerprint", None)
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def summarize_geometry(
    records: list[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, bool]]:
    expected = list(map(int, protocol["panel"]["package_indices"]))
    if [int(record["package_index"]) for record in records] != expected:
        raise RuntimeError("rare-target record order 与冻结 panel 不一致")
    by_pg = {point_group: [] for point_group in TARGETS}
    success, action_success = [], []
    failures: Counter[str] = Counter()
    for record in records:
        point_group = str(record["requested_target_pg"])
        passed = record["status"] == "success"
        by_pg[point_group].append(passed)
        success.append(passed)
        action_success.append(bool(record["graph_action_recovered"]))
        if not passed:
            failures[str(record["failure_stage"])] += 1
    metrics = {
        "molecule_count": len(records),
        "graph_action_recovery_fraction": float(np.mean(action_success)),
        "end_to_end_success_fraction": float(np.mean(success)),
        "successful_molecule_count": int(sum(success)),
        "failed_molecule_count": int(len(success) - sum(success)),
        "s4_success_fraction": float(np.mean(by_pg["S4"])),
        "d6h_success_fraction": float(np.mean(by_pg["D6h"])),
        "failure_stage_counts": dict(sorted(failures.items())),
    }
    threshold = protocol["geometry_gate_thresholds"]
    checks = {
        "molecule_count_exact": len(records) == protocol["panel"]["molecule_count"],
        "graph_action_recovery_fraction_min": metrics["graph_action_recovery_fraction"] >= threshold["graph_action_recovery_fraction_min"],
        "end_to_end_success_fraction_min": metrics["end_to_end_success_fraction"] >= threshold["end_to_end_success_fraction_min"],
        "s4_success_fraction_min": metrics["s4_success_fraction"] >= threshold["s4_success_fraction_min"],
        "d6h_success_fraction_min": metrics["d6h_success_fraction"] >= threshold["d6h_success_fraction_min"],
        "reference_xyz_never_used": all(not record["scope"]["reference_xyz_used"] for record in records),
        "stored_action_never_used": all(not record["scope"]["stored_symmetry_action_used"] for record in records),
        "etkdg_never_used": all(not record["scope"]["etkdg_used"] for record in records),
    }
    return metrics, checks


def run_molecule(
    canonical: dict[str, Any], runtime: ETFlowRuntime,
    batch_protocol: dict[str, Any], inference_protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    package_index = int(canonical["package_index"])
    target_pg = str(canonical["metadata"]["Target_PG"])
    record = {
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
    }
    try:
        audit, arrays = run_prediction(
            canonical, target_pg, int(record["seed"]), runtime, inference_protocol
        )
        record.update({
            "graph_action_recovered": True,
            "status": "success",
            "selected_candidate_id": int(audit["selected"]["candidate_id"]),
            "selected": audit["selected"],
            "action_selection_audit": audit,
        })
        return record, arrays
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        record["failure_stage"] = (
            "graph_action" if "automorphism" in message.lower()
            or "action" in message.lower() else "strict_inference"
        )
        record["failure"] = message
        return record, None


def run(
    package: Path, protocol_path: Path, inference_protocol_path: Path,
    output_dir: Path, cache: Path, device: str,
    maximum_molecules: int | None = None,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_fingerprint"] != _fingerprint(protocol):
        raise RuntimeError("rare-target protocol fingerprint 不一致")
    if _sha256(inference_protocol_path) != protocol["identity"]["inference_protocol_sha256"]:
        raise RuntimeError("rare-target/inference protocol SHA 不一致")
    for relative, expected in protocol["identity"]["source_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"rare-target source hash 不一致: {relative}")
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
        panel = panel[:int(maximum_molecules)]
    records = []
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
            record, arrays = run_molecule(dataset[package_index], runtime, protocol, inference)
            record["protocol_sha256"] = protocol_sha
            if arrays is not None:
                save_deterministic_npz(npz_path, arrays)
                record["coordinates_sha256"] = _sha256(npz_path)
            else:
                record["coordinates_sha256"] = None
            _atomic_json(json_path, record)
        records.append(record)
        print(json.dumps({
            "completed": progress, "total": len(panel),
            "package_index": package_index,
            "target_pg": record["requested_target_pg"],
            "status": record["status"], "failure_stage": record["failure_stage"],
        }, ensure_ascii=False), flush=True)

    if len(panel) != protocol["panel"]["molecule_count"]:
        partial = {
            "schema_version": "etflow-e3f02-rare-targets-partial-v1",
            "status": "PARTIAL_ETFLOW_E3F02_RARE_TARGETS_RESUMABLE",
            "completed_count": len(records),
            "total_count": protocol["panel"]["molecule_count"],
            "protocol_sha256": protocol_sha,
        }
        _atomic_json(output_dir / "partial_status.json", partial)
        return partial

    metrics, checks = summarize_geometry(records, protocol)
    passed = all(checks.values())
    report = {
        "schema_version": "etflow-e3f02-rare-targets-geometry-v1",
        "status": protocol["decision"]["geometry_pass_status"] if passed else protocol["decision"]["fail_status"],
        "passed_geometry_gate": passed,
        "metrics": metrics, "checks": checks, "records": records,
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
    parser.add_argument("--inference-protocol", type=Path, default=DEFAULT_INFERENCE_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-molecules", type=int)
    args = parser.parse_args()
    report = run(
        args.package.resolve(), args.protocol.resolve(),
        args.inference_protocol.resolve(), args.output_dir.resolve(),
        args.cache.resolve(), args.device, args.maximum_molecules,
    )
    print(json.dumps({
        key: report[key] for key in report
        if key in {"status", "passed_geometry_gate", "metrics", "checks", "completed_count", "total_count"}
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
