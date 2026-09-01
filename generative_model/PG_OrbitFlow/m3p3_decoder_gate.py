"""Run the frozen learned-IC Tier-32 decoder and original reconstruction Gate."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .graph_automorphism import build_graph_automorphism_contract
from .m0_oracle_reconstruction import build_oracle_targets, reconstruction_metrics
from .m2_torsion_training import _learned_decoder_targets
from .m2p1_phase_training import _prediction_checks, _reconstruction_checks
from .m3_tier32_training import _decoder_task, _model_setting
from .m3p1_automorphism_training import _matched_metrics, _prediction_record
from .m3p1_model import AutomorphismSetQuotientICPredictor
from .orbit_ic_model import build_orbit_ic_example
from .orbit_kinematics import build_orbit_parameterization
from .overfit import _panel_samples


SCHEMA_VERSION = "pg-orbitflow-m3p3-decoder-gate-v1"


def _dump(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-m3p3-decoder-protocol-v1":
        raise ValueError("unsupported M3.3 decoder protocol")
    if _sha256(Path(protocol["package_dir"]) / "manifest.json") != protocol[
        "canonical_manifest_sha256"
    ]:
        raise RuntimeError("canonical manifest changed")
    for item in protocol["predictor"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3.3 predictor identity changed")
    for item in protocol["implementation"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("M3.3 decoder implementation changed")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    import torch

    if args.prediction_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA prediction requested but unavailable")
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    examples = []
    for sample in _panel_samples(protocol, tier):
        geometry = build_geometry_contract(sample)
        graph, targets = build_orbit_ic_example(sample, geometry)
        automorphism = build_graph_automorphism_contract(sample, geometry)
        examples.append((sample, geometry, graph, targets, automorphism))

    model = AutomorphismSetQuotientICPredictor(**_model_setting(protocol))
    checkpoint = torch.load(
        protocol["predictor"]["checkpoint"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(args.prediction_device)
    model.eval()
    predictions = []
    prediction_records = []
    with torch.no_grad():
        for sample, _, graph, targets, automorphism in examples:
            prediction = {
                key: value.cpu().numpy()
                for key, value in model(
                    graph, automorphism, device=args.prediction_device
                ).items()
            }
            predictions.append(prediction)
            prediction_records.append(
                _prediction_record(sample, targets, prediction, automorphism)
            )
    prediction_values, prediction_checks, prediction_strata = _prediction_checks(
        prediction_records, protocol["prediction_gate"]
    )
    if not all(prediction_checks.values()):
        raise RuntimeError("frozen M3.3 predictor no longer passes its Gate")
    np.savez_compressed(
        output_dir / "predictions.npz",
        **{
            f"{sample.package_index}_{key}": value
            for (sample, _, _, _, _), prediction in zip(
                examples, predictions, strict=True
            )
            for key, value in prediction.items()
        },
    )
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    decoder_protocol = {
        key: protocol["decoder"][key]
        for key in ("initialization", "optimization", "energy_weights")
    }
    starts = int(protocol["decoder"]["initialization"]["starts_per_molecule"])
    stride = int(
        protocol["decoder"]["initialization"]["molecule_specific_seed_stride"]
    )
    tasks = []
    for (sample, geometry, _, _, _), prediction in zip(
        examples, predictions, strict=True
    ):
        decoder_targets = _learned_decoder_targets(prediction, geometry)
        specification = build_orbit_parameterization(sample)
        for start in range(starts):
            tasks.append(
                (
                    sample,
                    geometry,
                    decoder_targets,
                    specification,
                    decoder_protocol,
                    int(protocol["seed"]) + sample.package_index * stride + start,
                    start,
                )
            )

    collected = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(protocol["decoder"]["parallel_workers"]),
        mp_context=context,
    ) as executor:
        for package_index, start, candidate, coordinates in executor.map(
            _decoder_task, tasks, chunksize=1
        ):
            collected[(package_index, start)] = (candidate, coordinates)

    raw_records = []
    matched_records = []
    assignments = []
    coordinate_parts = []
    offsets = [0]
    for sample, geometry, _, _, automorphism in examples:
        candidates = [
            collected[(sample.package_index, start)][0] for start in range(starts)
        ]
        coordinate_candidates = [
            collected[(sample.package_index, start)][1] for start in range(starts)
        ]
        selected = min(
            range(starts), key=lambda index: candidates[index]["final_energy"]
        )
        coordinates = coordinate_candidates[selected]
        raw = reconstruction_metrics(
            sample,
            geometry,
            build_oracle_targets(sample, geometry),
            coordinates,
        )
        common = {
            "package_index": int(sample.package_index),
            "target_pg": sample.target_pg,
            "selected_start_index": selected,
            "selected_energy": float(candidates[selected]["final_energy"]),
            "finite_start_count": sum(
                candidate["all_values_finite"] for candidate in candidates
            ),
            "all_starts_finite": all(
                candidate["all_values_finite"] for candidate in candidates
            ),
        }
        raw_records.append({**common, **raw})
        matched, assignment = _matched_metrics(
            sample, geometry, automorphism, coordinates, raw
        )
        matched_records.append({**common, **matched})
        assignments.append({"package_index": int(sample.package_index), **assignment})
        coordinate_parts.append(coordinates.astype(np.float32))
        offsets.append(offsets[-1] + len(coordinates))
        print(
            f"reconstruct package_index={sample.package_index} PG={sample.target_pg} "
            f"torsion={matched['torsion_circular_mae_degrees']:.6g} "
            f"nonplanar={matched['nonplanar_torsion_circular_mae_degrees']:.6g} "
            f"rmsd={matched['kabsch_rmsd_angstrom']:.6g}A",
            flush=True,
        )

    raw_values, raw_checks, raw_strata = _reconstruction_checks(
        raw_records, protocol["reconstruction_gate"]
    )
    values, checks, strata = _reconstruction_checks(
        matched_records, protocol["reconstruction_gate"]
    )
    passed = all(checks.values())
    status = "PASS_M3_TIER32" if passed else "FAIL_M3P3_RECONSTRUCTION_GATE"
    np.savez_compressed(
        output_dir / "coordinates.npz",
        coordinates=np.concatenate(coordinate_parts),
        atom_offsets=np.asarray(offsets),
        package_indices=np.asarray(
            [sample.package_index for sample, *_ in examples]
        ),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "passed": passed,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "prediction_gate": {
            "passed": True,
            "checks": prediction_checks,
            "values": prediction_values,
            "strata": prediction_strata,
            "thresholds": protocol["prediction_gate"],
        },
        "raw_reconstruction_records": raw_records,
        "raw_reconstruction_gate": {
            "passed": all(raw_checks.values()),
            "checks": raw_checks,
            "values": raw_values,
            "strata": raw_strata,
            "thresholds": protocol["reconstruction_gate"],
        },
        "centralizer_automorphism_assignments": assignments,
        "reconstruction_records": matched_records,
        "reconstruction_gate": {
            "passed": passed,
            "checks": checks,
            "values": values,
            "strata": strata,
            "thresholds": protocol["reconstruction_gate"],
        },
        "scope": protocol["scope"],
    }
    _dump(output_dir / "report.json", report)
    print(f"{status} report={output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
