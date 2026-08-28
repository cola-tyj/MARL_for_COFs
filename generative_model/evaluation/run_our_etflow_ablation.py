"""Run the frozen paired four-route our_ET_Flow ablation panel.

This GPU-side runner intentionally does not import pymatgen.  It writes one
deterministic coordinate package for the independent ``env_cof`` audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from generative_model.conformer.etflow_bridge import ETFlowRuntime
from generative_model.data.build_cof_package import save_deterministic_npz
from generative_model.data.cof_graph_dataset import COFGraphDataset
from generative_model.evaluation.our_etflow_ablation import (
    ROUTES,
    route_geometry_metrics,
    run_etflow_raw_route,
    run_etkdg_route,
    run_full_f02_route,
    run_projection_route,
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
from generative_model.inference.generate_etflow_symmetric_xyz_v4 import (
    action_samples_v4,
)


DEFAULT_PROTOCOL = ROOT / "generative_model/evaluation/our_etflow_ablation_protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "generative_model/runs/our_etflow_ablation_v1"


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route in ROUTES:
        selected = [
            record["routes"][route]
            for record in records
            if record["routes"][route]["success"]
        ]
        result[route] = {
            "success_count": len(selected),
            "full_panel_count": len(records),
            "success_fraction": len(selected) / len(records) if records else 0.0,
        }
        for metric in (
            "collision_free_at_0p6",
            "bond_length_mae_to_canonical_angstrom",
            "kabsch_rmsd_to_canonical_angstrom",
            "pair_distance_mae_to_canonical_angstrom",
            "uff_single_point_energy_per_atom_kcal_mol",
            "runtime_seconds",
        ):
            values = [item[metric] for item in selected if item.get(metric) is not None]
            if metric == "collision_free_at_0p6":
                result[route]["collision_free_fraction_full_panel"] = (
                    sum(bool(value) for value in values) / len(records) if records else 0.0
                )
            elif values:
                result[route][f"mean_{metric}"] = float(np.mean(values))
                result[route][f"median_{metric}"] = float(np.median(values))
    return result


def _validate_protocol(protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_fingerprint") != _fingerprint(protocol):
        raise RuntimeError("our_ET_Flow ablation protocol fingerprint mismatch")
    for relative, expected in protocol["identity"]["source_sha256"].items():
        observed = _sha256(ROOT / relative)
        if observed != expected:
            raise RuntimeError(
                f"ablation source hash mismatch: {relative}: {observed} != {expected}"
            )
    v5_path = ROOT / "generative_model/inference/etflow_e3f02_protocol_v5.json"
    if _sha256(v5_path) != protocol["identity"]["v5_protocol_sha256"]:
        raise RuntimeError("frozen v5 protocol hash mismatch")
    checkpoint = ROOT / protocol["identity"]["checkpoint_path"]
    if _sha256(checkpoint) != protocol["identity"]["checkpoint_sha256"]:
        raise RuntimeError("official ET-Flow checkpoint hash mismatch")
    return protocol


def run(
    *,
    protocol_path: Path,
    package_path: Path,
    output_dir: Path,
    cache: Path,
    device: str,
    panel_limit: int | None = None,
) -> dict[str, Any]:
    protocol = _validate_protocol(protocol_path)
    v5_protocol = json.loads(
        (ROOT / "generative_model/inference/etflow_e3f02_protocol_v5.json")
        .read_text(encoding="utf-8")
    )
    dataset = COFGraphDataset(package_path)
    runtime = ETFlowRuntime(
        model_name=str(v5_protocol["etflow"]["model_name"]),
        device=device,
        cache=str(cache),
    )
    frozen_panel = protocol["panel"]["records"]
    if panel_limit is not None and not 1 <= panel_limit <= len(frozen_panel):
        raise ValueError("panel_limit must be within the frozen panel")
    panel = frozen_panel if panel_limit is None else frozen_panel[:panel_limit]
    atom_counts = np.asarray(
        [len(dataset[int(item["package_index"])]["atomic_numbers"]) for item in panel],
        dtype=np.int64,
    )
    atom_offsets = np.concatenate((np.asarray([0]), np.cumsum(atom_counts)))
    total_atoms = int(atom_offsets[-1])
    coordinates = {
        route: np.full((total_atoms, 3), np.nan, dtype="<f8") for route in ROUTES
    }
    canonical_positions = np.empty((total_atoms, 3), dtype="<f8")
    atomic_numbers = np.empty(total_atoms, dtype="<i8")
    success_masks = {
        route: np.zeros(len(panel), dtype=np.bool_) for route in ROUTES
    }
    records: list[dict[str, Any]] = []
    started_all = perf_counter()
    for panel_id, item in enumerate(panel):
        package_index = int(item["package_index"])
        target_pg = str(item["target_pg"])
        seed = int(item["seed"])
        canonical = dataset[package_index]
        if canonical["molecule_id"] != item["molecule_id"]:
            raise RuntimeError("panel molecule identity mismatch")
        if str(canonical["metadata"]["Target_PG"]) != target_pg:
            raise RuntimeError("panel Target_PG mismatch")
        start, end = map(int, atom_offsets[panel_id : panel_id + 2])
        canonical_positions[start:end] = canonical["positions"]
        atomic_numbers[start:end] = canonical["atomic_numbers"]
        record: dict[str, Any] = {
            "panel_id": panel_id,
            "package_index": package_index,
            "molecule_id": str(canonical["molecule_id"]),
            "target_pg": target_pg,
            "seed": seed,
            "atom_count": end - start,
            "split_iid": str(canonical["metadata"]["Split_IID"]),
            "split_core_ood": str(canonical["metadata"]["Split_Core_OOD"]),
            "routes": {},
        }
        try:
            samples = action_samples_v4(canonical, target_pg, v5_protocol)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            for route in ROUTES:
                record["routes"][route] = {"success": False, "failure": failure}
            records.append(record)
            continue

        route_outputs: dict[str, dict[str, Any]] = {}
        try:
            route_outputs[ROUTES[0]] = run_etkdg_route(
                samples[0],
                base_seed=seed,
                settings=protocol["routes"][ROUTES[0]],
            )
        except Exception as error:
            record["routes"][ROUTES[0]] = {
                "success": False,
                "failure": f"{type(error).__name__}: {error}",
            }

        raw_positions: list[np.ndarray] | None = None
        prior_seconds = 0.0
        try:
            prior_started = perf_counter()
            prediction = runtime.predict(
                samples[0],
                num_samples=int(v5_protocol["etflow"]["candidate_count"]),
                n_timesteps=int(v5_protocol["etflow"]["n_timesteps"]),
                seed=seed,
            )
            prior_seconds = perf_counter() - prior_started
            raw_positions = [
                np.asarray(value, dtype=np.float64) for value in prediction["positions"]
            ]
            if len(raw_positions) != int(v5_protocol["etflow"]["candidate_count"]):
                raise RuntimeError("ET-Flow returned an incomplete raw candidate set")
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            for route in ROUTES[1:]:
                record["routes"][route] = {
                    "success": False,
                    "failure": failure,
                }
        if raw_positions is not None:
            route_functions = {
                ROUTES[1]: lambda: run_etflow_raw_route(
                    samples[0], raw_positions, prior_seconds=prior_seconds
                ),
                ROUTES[2]: lambda: run_projection_route(
                    samples,
                    raw_positions,
                    target_pg,
                    v5_protocol,
                    prior_seconds=prior_seconds,
                ),
                ROUTES[3]: lambda: run_full_f02_route(
                    samples,
                    raw_positions,
                    target_pg,
                    seed,
                    v5_protocol,
                    prior_seconds=prior_seconds,
                ),
            }
            for route, function in route_functions.items():
                try:
                    route_outputs[route] = function()
                except Exception as error:
                    record["routes"][route] = {
                        "success": False,
                        "failure": f"{type(error).__name__}: {error}",
                    }

        for route, output in route_outputs.items():
            try:
                metrics = route_geometry_metrics(
                    samples[0], output["positions"], canonical["positions"]
                )
                coordinates[route][start:end] = output["positions"]
                success_masks[route][panel_id] = True
                record["routes"][route] = {
                    "success": True,
                    **strip_positions(output),
                    **metrics,
                }
            except Exception as error:
                record["routes"][route] = {
                    "success": False,
                    "failure": f"{type(error).__name__}: {error}",
                }
        for route in ROUTES:
            record["routes"].setdefault(
                route,
                {"success": False, "failure": "route produced no record"},
            )
        records.append(record)
        print(json.dumps({
            "completed": panel_id + 1,
            "total": len(panel),
            "package_index": package_index,
            "target_pg": target_pg,
            "success": {
                route: bool(record["routes"][route]["success"]) for route in ROUTES
            },
        }, ensure_ascii=False), flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    coordinate_path = output_dir / "coordinates.npz"
    arrays: dict[str, np.ndarray] = {
        "atom_offsets": atom_offsets.astype("<i8"),
        "atomic_numbers": atomic_numbers,
        "canonical_positions": canonical_positions,
    }
    for route in ROUTES:
        arrays[f"positions_{route}"] = coordinates[route]
        arrays[f"success_{route}"] = success_masks[route]
    save_deterministic_npz(coordinate_path, arrays)
    report = {
        "schema_version": "our-etflow-paired-ablation-run-v1",
        "status": (
            "PASS_EXECUTION_AWAIT_INDEPENDENT_POINT_GROUP_AUDIT"
            if panel_limit is None
            else "PASS_ENGINEERING_PREFLIGHT_NO_QUALITY_CLAIM"
        ),
        "quality_claim": False,
        "panel_count": len(panel),
        "frozen_panel_count": len(frozen_panel),
        "partial_panel_preflight": panel_limit is not None,
        "records": records,
        "summary_pre_point_group": _summary(records),
        "scope": protocol["scope"],
        "runtime": {
            "total_wall_seconds": perf_counter() - started_all,
            "timing_is_descriptive_not_deterministic": True,
            "etflow_prior_shared_across_three_etflow_routes": True,
        },
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "coordinates_sha256": _sha256(coordinate_path),
            "checkpoint_sha256": protocol["identity"]["checkpoint_sha256"],
        },
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--panel-limit", type=int)
    args = parser.parse_args()
    report = run(
        protocol_path=args.protocol.resolve(),
        package_path=args.package.resolve(),
        output_dir=args.output_dir.resolve(),
        cache=args.cache.resolve(),
        device=args.device,
        panel_limit=args.panel_limit,
    )
    print(json.dumps({
        "status": report["status"],
        "panel_count": report["panel_count"],
        "summary": report["summary_pre_point_group"],
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
