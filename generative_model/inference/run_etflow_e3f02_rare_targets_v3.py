"""Resumable all-52 runner using v5 S4 inertia-stable inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from generative_model.conformer.etflow_bridge import ETFlowRuntime
from generative_model.data.build_cof_package import save_deterministic_npz
from generative_model.data.cof_graph_dataset import COFGraphDataset
from generative_model.inference.generate_etflow_symmetric_xyz import _atomic_json
from generative_model.inference.generate_etflow_symmetric_xyz_v4 import action_samples_v4
from generative_model.inference.generate_etflow_symmetric_xyz_v5 import run_prediction_v5
from generative_model.inference.run_etflow_e3f02_rare_targets import summarize_geometry


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT / "generative_model/data/processed/v2"
DEFAULT_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_rare_targets_protocol_v3.json"
DEFAULT_INFERENCE_PROTOCOL = ROOT / "generative_model/inference/etflow_e3f02_protocol_v5.json"
DEFAULT_CACHE = ROOT / "generative_model/checkpoints/etflow"
DEFAULT_OUTPUT = ROOT / "generative_model/runs/etflow_e3f02_rare_targets_v3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("protocol_fingerprint", None)
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def run_molecule(
    canonical: dict[str, Any], runtime: ETFlowRuntime,
    batch_protocol: dict[str, Any], inference_protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    package_index = int(canonical["package_index"])
    target_pg = str(canonical["metadata"]["Target_PG"])
    record = {
        "package_index": package_index, "molecule_id": str(canonical["molecule_id"]),
        "requested_target_pg": target_pg,
        "source_split": str(canonical["metadata"]["Split_IID"]),
        "atom_count": len(canonical["atomic_numbers"]),
        "seed": int(batch_protocol["sampling"]["base_seed"] + package_index),
        "scope": dict(batch_protocol["scope"]),
        "graph_action_recovered": False, "action_candidate_count": 0,
        "status": "failure", "failure_stage": None, "failure": None,
    }
    try:
        # Explicit preflight keeps graph-action accounting independent of later
        # ET-Flow/projection/optimization failures.  v5 repeats this deterministic
        # graph-only enumeration; no coordinate or stored action is consulted.
        samples = action_samples_v4(canonical, target_pg, inference_protocol)
        record["graph_action_recovered"] = True
        record["action_candidate_count"] = len(samples)
    except Exception as error:
        record["failure_stage"] = "graph_action"
        record["failure"] = f"{type(error).__name__}: {error}"
        return record, None
    try:
        audit, arrays = run_prediction_v5(
            canonical, target_pg, int(record["seed"]), runtime, inference_protocol
        )
        record.update({
            "status": "success",
            "selected_candidate_id": int(audit["selected"]["candidate_id"]),
            "selected": audit["selected"],
            "action_selection_audit": audit,
        })
        return record, arrays
    except Exception as error:
        record["failure_stage"] = "strict_inference"
        record["failure"] = f"{type(error).__name__}: {error}"
        return record, None


def run(
    package: Path, protocol_path: Path, inference_protocol_path: Path,
    output_dir: Path, cache: Path, device: str,
    maximum_molecules: int | None = None,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_fingerprint"] != _fingerprint(protocol):
        raise RuntimeError("rare-target v3 protocol fingerprint 不一致")
    if _sha256(inference_protocol_path) != protocol["identity"]["inference_protocol_sha256"]:
        raise RuntimeError("rare-target v3/inference v5 protocol SHA 不一致")
    for relative, expected in protocol["identity"]["source_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"rare-target v3 source hash 不一致: {relative}")
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
            "package_index": package_index, "target_pg": record["requested_target_pg"],
            "action_candidates": record["action_candidate_count"],
            "inertia_separation": record.get("selected", {}).get("mass_weighted_inertia_separation"),
            "status": record["status"], "failure_stage": record["failure_stage"],
        }, ensure_ascii=False), flush=True)

    if len(panel) != protocol["panel"]["molecule_count"]:
        partial = {
            "schema_version": "etflow-e3f02-rare-targets-partial-v3",
            "status": "PARTIAL_ETFLOW_E3F02_RARE_TARGETS_V3_RESUMABLE",
            "completed_count": len(records),
            "total_count": protocol["panel"]["molecule_count"],
            "protocol_sha256": protocol_sha,
        }
        _atomic_json(output_dir / "partial_status.json", partial)
        return partial

    metrics, checks = summarize_geometry(records, protocol)
    s4_separations = [
        float(record["selected"]["mass_weighted_inertia_separation"])
        for record in records if record["status"] == "success"
        and record["requested_target_pg"] == "S4"
    ]
    metrics["s4_minimum_selected_inertia_separation"] = (
        min(s4_separations) if s4_separations else None
    )
    checks["s4_inertia_guard_all_successes"] = bool(s4_separations) and all(
        value >= protocol["s4_inertia_selection"]["minimum_separation"]
        for value in s4_separations
    )
    passed = all(checks.values())
    report = {
        "schema_version": "etflow-e3f02-rare-targets-geometry-v3",
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
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
