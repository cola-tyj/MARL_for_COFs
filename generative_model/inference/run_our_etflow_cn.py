"""Batch our_ET_Flow v5 inference for the uploaded Cn C3 graph package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from generative_model.conformer.etflow_bridge import ETFlowRuntime
from generative_model.data.build_cof_package import save_deterministic_npz
from generative_model.data.cn_graph_dataset import CnGraphDataset
from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT, DEFAULT_CACHE, _atomic_json, _fingerprint, _sha256, _symbols, _write_xyz,
)
from generative_model.inference.generate_etflow_symmetric_xyz_v5 import (
    DEFAULT_PROTOCOL, run_prediction_v5,
)


DEFAULT_INPUT = ROOT / "generative_model/data/Cn"
DEFAULT_OUTPUT = ROOT / "generative_model/results/Cn"


def _load_frozen_protocol(protocol_path: Path, cache: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_fingerprint"] != _fingerprint(protocol):
        raise RuntimeError("our_ET_Flow v5 protocol fingerprint 不一致")
    for relative, expected in protocol["identity"]["source_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"our_ET_Flow v5 source hash 不一致: {relative}")
    checkpoint = cache / "drugs-o3.ckpt"
    if _sha256(checkpoint) != protocol["identity"]["checkpoint_sha256"]:
        raise RuntimeError("ET-Flow checkpoint SHA 不一致")
    if "C3" not in protocol["supported_target_pgs"]:
        raise RuntimeError("冻结 v5 protocol 不支持 C3")
    return protocol


def _record_directory(output_root: Path, sample: dict[str, Any]) -> Path:
    return output_root / sample["metadata"]["Source_Group"] / "records" / sample["molecule_id"]


def _write_record(
    sample: dict[str, Any], arrays: dict[str, np.ndarray], audit: dict[str, Any],
    *, seed: int, protocol_path: Path, protocol: dict[str, Any], cache: Path,
    output_root: Path,
) -> dict[str, Any]:
    record_dir = _record_directory(output_root, sample)
    record_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = record_dir / "coordinates.npz"
    save_deterministic_npz(coordinates_path, arrays)
    symbols = _symbols(arrays["atomic_numbers"])
    comment = (
        f"molecule_id={sample['molecule_id']} source=external_Cn "
        f"target_pg=C3 seed={seed}"
    )
    for stage in ("raw", "projected", "final"):
        _write_xyz(
            record_dir / f"{stage}.xyz", symbols, arrays[f"{stage}_positions"],
            f"{comment} stage={stage}",
        )
    final_xyz = (
        output_root / sample["metadata"]["Source_Group"] / "xyz"
        / f"{sample['molecule_id']}.xyz"
    )
    final_xyz.parent.mkdir(parents=True, exist_ok=True)
    _write_xyz(final_xyz, symbols, arrays["final_positions"], f"{comment} stage=final")
    selected = audit["selected"]
    report = {
        "schema_version": "our-etflow-cn-known-graph-inference-v1",
        "status": "PASS_OUR_ETFLOW_CN_XYZ_AWAIT_INDEPENDENT_POINT_GROUP_AUDIT",
        "passed_geometry_gate": True,
        "package_index": int(sample["package_index"]),
        "molecule_id": str(sample["molecule_id"]),
        "source_split": "external_Cn",
        "source_group": str(sample["metadata"]["Source_Group"]),
        "source_smiles": str(sample["metadata"]["SMILES"]),
        "requested_target_pg": "C3",
        "canonical_dataset_target_pg": "C3",
        "atom_count": len(arrays["atomic_numbers"]),
        "heavy_atom_count": int(np.sum(arrays["atomic_numbers"] != 1)),
        "explicit_h_count": int(np.sum(arrays["atomic_numbers"] == 1)),
        "seed": int(seed),
        "selected_candidate_id": int(selected["candidate_id"]),
        "selected": selected,
        "action_selection_audit": audit,
        "input_adapter_audit": sample["adapter_audit"],
        "source_identity": sample["source_identity"],
        "scope": {
            **protocol["scope"],
            "external_cn_input": True,
            "implicit_h_expanded_before_inference": True,
            "reference_3d_used": False,
        },
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "checkpoint_sha256": _sha256(cache / "drugs-o3.ckpt"),
            "coordinates_sha256": _sha256(coordinates_path),
            "final_xyz_sha256": _sha256(final_xyz),
        },
        "outputs": {
            "coordinates_npz": "coordinates.npz",
            "raw_xyz": "raw.xyz",
            "projected_xyz": "projected.xyz",
            "final_xyz": "final.xyz",
            "convenience_final_xyz": str(final_xyz.relative_to(output_root)),
        },
    }
    _atomic_json(record_dir / "report.json", report)
    return report


def run_batch(
    *, input_root: Path, output_root: Path, protocol_path: Path, cache: Path,
    device: str, seed_base: int, limit: int | None, resume: bool,
) -> dict[str, Any]:
    protocol = _load_frozen_protocol(protocol_path, cache)
    dataset = CnGraphDataset(input_root)
    selected = list(dataset.records if limit is None else dataset.records[:limit])
    if not selected:
        raise ValueError("Cn batch 不能为空")
    if seed_base < 0 or seed_base + len(dataset) >= 2**32:
        raise ValueError("seed_base 超出 NumPy/PyTorch 可复现范围")
    runtime = ETFlowRuntime(
        model_name=protocol["etflow"]["model_name"], device=device, cache=str(cache)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for completed, sample in enumerate(selected, start=1):
        seed = int(seed_base + sample["package_index"])
        record_dir = _record_directory(output_root, sample)
        report_path = record_dir / "report.json"
        if resume and report_path.is_file():
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            coordinates_path = record_dir / "coordinates.npz"
            if (
                existing.get("seed") == seed
                and existing.get("identity", {}).get("protocol_sha256") == _sha256(protocol_path)
                and coordinates_path.is_file()
                and existing.get("identity", {}).get("coordinates_sha256") == _sha256(coordinates_path)
            ):
                records.append(existing)
                print(json.dumps({
                    "completed": completed, "total": len(selected),
                    "molecule_id": sample["molecule_id"], "status": "RESUMED",
                }))
                continue
            raise RuntimeError(f"已有结果身份不匹配，拒绝静默覆盖: {record_dir}")
        try:
            audit, arrays = run_prediction_v5(sample, "C3", seed, runtime, protocol)
            report = _write_record(
                sample, arrays, audit, seed=seed, protocol_path=protocol_path,
                protocol=protocol, cache=cache, output_root=output_root,
            )
            records.append(report)
            print(json.dumps({
                "completed": completed, "total": len(selected),
                "molecule_id": sample["molecule_id"], "status": report["status"],
            }))
        except Exception as error:
            failure = {
                "package_index": int(sample["package_index"]),
                "molecule_id": str(sample["molecule_id"]),
                "source_group": str(sample["metadata"]["Source_Group"]),
                "seed": seed,
                "failure": f"{type(error).__name__}: {error}",
            }
            failures.append(failure)
            print(json.dumps({**failure, "status": "FAILED"}, ensure_ascii=False))

    rows = []
    for report in records:
        rows.append({
            "package_index": report["package_index"],
            "molecule_id": report["molecule_id"],
            "source_group": report["source_group"],
            "smiles": report["source_smiles"],
            "target_pg": report["requested_target_pg"],
            "atom_count": report["atom_count"],
            "heavy_atom_count": report["heavy_atom_count"],
            "explicit_h_count": report["explicit_h_count"],
            "seed": report["seed"],
            "minimum_pair_distance_angstrom": report["selected"]["final_minimum_pair_distance_angstrom"],
            "maximum_operation_error_angstrom": report["selected"]["final_maximum_operation_error_angstrom"],
            "final_xyz": report["outputs"]["convenience_final_xyz"],
            "final_xyz_sha256": report["identity"]["final_xyz_sha256"],
        })
    index_path = output_root / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) if rows else [
            "package_index", "molecule_id", "source_group", "smiles", "target_pg",
            "atom_count", "heavy_atom_count", "explicit_h_count", "seed",
            "minimum_pair_distance_angstrom", "maximum_operation_error_angstrom",
            "final_xyz", "final_xyz_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    batch = {
        "schema_version": "our-etflow-cn-batch-v1",
        "status": (
            "PASS_OUR_ETFLOW_CN_BATCH_AWAIT_INDEPENDENT_POINT_GROUP_AUDIT"
            if not failures and len(records) == len(selected)
            else "FAIL_OUR_ETFLOW_CN_BATCH_INCOMPLETE"
        ),
        "passed_generation": not failures and len(records) == len(selected),
        "requested_count": len(selected),
        "completed_count": len(records),
        "failure_count": len(failures),
        "group_counts": {
            group: sum(value["source_group"] == group for value in records)
            for group in ("C3k2", "C3k3")
        },
        "failures": failures,
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "checkpoint_sha256": _sha256(cache / "drugs-o3.ckpt"),
            "index_csv_sha256": _sha256(index_path),
        },
        "scope": {
            "input": "external Cn 2D heavy-atom graphs",
            "target_pg": "C3",
            "etflow_v5_reused_unchanged": True,
            "hard_projection_used": True,
            "f02_used": True,
            "reference_3d_used": False,
        },
    }
    _atomic_json(output_root / "batch_report.json", batch)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=2026082901)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_batch(
        input_root=args.input_root.resolve(), output_root=args.output_root.resolve(),
        protocol_path=args.protocol.resolve(), cache=args.cache.resolve(),
        device=args.device, seed_base=args.seed_base, limit=args.limit,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
