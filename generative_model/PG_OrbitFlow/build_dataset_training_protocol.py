"""Freeze dataset-scale PG-OrbitFlow smoke or full-training protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .c0_local_recovery import _sha256
from .data import PGOrbitFlowDataset
from .geometry import build_geometry_contract
from .factorized_cache import selection_fingerprint
from .global_shape_fingerprint_model import FingerprintGlobalShapePredictor
from .global_shape_model import build_global_shape_contract
from .m3_tier32_training import _model_setting
from .orbit_ic_model import build_orbit_ic_example


def _scan(package_dir, *, split: str, shape_setting: dict):
    dataset = PGOrbitFlowDataset(
        package_dir,
        split=split,
        split_scheme="iid",
        point_groups=("C2", "C3"),
    )
    records = []
    unsupported = []
    for position in range(len(dataset)):
        sample = dataset[position]
        try:
            geometry = build_geometry_contract(sample)
            build_orbit_ic_example(sample, geometry)
            build_global_shape_contract(
                sample,
                minimum_graph_distance=int(shape_setting["minimum_graph_distance"]),
                quantile_count=int(shape_setting["quantile_count"]),
            )
        except (ValueError, RuntimeError) as error:
            unsupported.append(
                {
                    "package_index": int(sample.package_index),
                    "molecule_id": sample.molecule_id,
                    "target_pg": sample.target_pg,
                    "num_atoms": len(sample.atom_types),
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
            continue
        records.append(
            {
                "package_index": int(sample.package_index),
                "molecule_id": sample.molecule_id,
                "target_pg": sample.target_pg,
                "num_atoms": len(sample.atom_types),
            }
        )
    return records, unsupported, len(dataset)


def _balanced_bounded(records, *, per_pg: int, max_atoms: int):
    result = []
    for point_group in ("C2", "C3"):
        candidates = sorted(
            (
                row
                for row in records
                if row["target_pg"] == point_group
                and int(row["num_atoms"]) <= max_atoms
            ),
            key=lambda row: int(row["package_index"]),
        )
        if len(candidates) < per_pg:
            raise RuntimeError(
                f"not enough {point_group} records for bounded dataset smoke"
            )
        result.extend(candidates[:per_pg])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--m3-protocol", type=Path, required=True)
    parser.add_argument("--m3-report", type=Path, required=True)
    parser.add_argument("--m4-report", type=Path, required=True)
    parser.add_argument("--factorized-audit", type=Path, required=True)
    parser.add_argument("--smoke-audit", type=Path)
    parser.add_argument("--factorized-cache-manifest", type=Path)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("generative_model/data/processed/v2"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path("generative_model/PG_OrbitFlow").resolve()
    m3_protocol_path = args.m3_protocol.resolve()
    m3_report_path = args.m3_report.resolve()
    m4_report_path = args.m4_report.resolve()
    factorized_path = args.factorized_audit.resolve()
    m3 = json.loads(m3_protocol_path.read_text(encoding="utf-8"))
    m3_report = json.loads(m3_report_path.read_text(encoding="utf-8"))
    m4_report = json.loads(m4_report_path.read_text(encoding="utf-8"))
    factorized = json.loads(factorized_path.read_text(encoding="utf-8"))
    if m3_report.get("status") != "PASS_M3_TIER32" or not m3_report.get("passed"):
        raise RuntimeError("dataset training requires passed M3")
    if m4_report.get("status") != "FAIL_M4_UNSEEN_PREDICTION_PROCEED_TO_DATASET_TRAINING":
        raise RuntimeError("dataset training requires the frozen M4 decision")
    if factorized.get("status") != "PASS_FACTORIZED_AUTOMORPHISM_ADVANCE_TO_DATASET_SMOKE":
        raise RuntimeError("factorized automorphism audit did not pass")
    smoke_audit_path = args.smoke_audit.resolve() if args.smoke_audit else None
    if args.mode == "full":
        if smoke_audit_path is None:
            raise RuntimeError("full training requires the passed dataset smoke audit")
        smoke_audit = json.loads(smoke_audit_path.read_text(encoding="utf-8"))
        if smoke_audit.get("status") != "PASS_DATASET_TRAINER_SMOKE_READY_FOR_FULL_RUN":
            raise RuntimeError("dataset trainer smoke audit did not pass")

    package_dir = args.package_dir.resolve()
    shape_setting = m3["global_shape_contract"]
    train_records, train_unsupported, train_total = _scan(
        package_dir, split="train", shape_setting=shape_setting
    )
    validation_records, validation_unsupported, validation_total = _scan(
        package_dir, split="val", shape_setting=shape_setting
    )
    if args.mode == "smoke":
        selected_train = _balanced_bounded(
            train_records, per_pg=32, max_atoms=30
        )
        selected_validation = _balanced_bounded(
            validation_records, per_pg=8, max_atoms=30
        )
        training = {
            "optimizer_steps": 256,
            "batch_size_molecules": 8,
            "point_group_composition_per_batch": {"C2": 4, "C3": 4},
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "checkpoint_every": 64,
        }
    else:
        selected_train = train_records
        selected_validation = validation_records
        training = {
            "optimizer_steps": 20_000,
            "batch_size_molecules": 8,
            "point_group_composition_per_batch": {"C2": 4, "C3": 4},
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "checkpoint_every": 1_000,
        }
    cache_spec = None
    factorized_train_unsupported = []
    factorized_validation_unsupported = []
    if args.factorized_cache_manifest is not None:
        cache_manifest_path = args.factorized_cache_manifest.resolve()
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        source_records = selected_train + selected_validation
        if cache_manifest.get("status") != "PASS_FACTORIZED_CACHE_COMPLETE_WITH_EXPLICIT_UNSUPPORTED":
            raise RuntimeError("factorized cache manifest did not pass")
        if cache_manifest.get("canonical_manifest_sha256") != _sha256(package_dir / "manifest.json"):
            raise RuntimeError("factorized cache canonical identity mismatch")
        if cache_manifest.get("source_selection_fingerprint") != selection_fingerprint(source_records):
            raise RuntimeError("factorized cache source selection mismatch")
        if int(cache_manifest.get("source_record_count", -1)) != len(source_records):
            raise RuntimeError("factorized cache source record count mismatch")
        cached_indices = {int(row["package_index"]) for row in cache_manifest["records"]}
        unsupported_indices = {
            int(row["package_index"]) for row in cache_manifest["unsupported_records"]
        }
        expected_indices = {int(row["package_index"]) for row in source_records}
        if cached_indices & unsupported_indices or cached_indices | unsupported_indices != expected_indices:
            raise RuntimeError("factorized cache partition is not exact")
        selected_train = [
            row for row in selected_train if int(row["package_index"]) in cached_indices
        ]
        selected_validation = [
            row for row in selected_validation if int(row["package_index"]) in cached_indices
        ]
        factorized_train_unsupported = [
            row for row in cache_manifest["unsupported_records"] if row["split"] == "train"
        ]
        factorized_validation_unsupported = [
            row for row in cache_manifest["unsupported_records"] if row["split"] == "val"
        ]
        final_records = selected_train + selected_validation
        if cache_manifest.get("cached_selection_fingerprint") != selection_fingerprint(final_records):
            raise RuntimeError("factorized cache final selection mismatch")
        cache_spec = {
            "directory": str(cache_manifest_path.parent),
            "manifest": str(cache_manifest_path),
            "manifest_sha256": _sha256(cache_manifest_path),
            "selection_fingerprint": cache_manifest["cached_selection_fingerprint"],
            "record_count": cache_manifest["cached_record_count"],
        }
    model_setting = dict(m3["model"])
    model = FingerprintGlobalShapePredictor(
        quantile_count=int(shape_setting["quantile_count"]),
        fingerprint_bits=int(model_setting["wl_fingerprint_bits"]),
        **_model_setting({"model": model_setting}),
    )
    isolation = model.parent.configure_trainable_parameters()
    model_setting["parameter_count_expected"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    model_setting["trainable_parameter_count_expected"] = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    sources = {
        "runner": root / "dataset_training.py",
        "model": root / "global_shape_fingerprint_model.py",
        "ic_model": root / "orbit_ic_model.py",
        "factorized_automorphism": root / "factorized_automorphism.py",
        "factorized_cache": root / "factorized_cache.py",
        "cache_builder": root / "precompute_factorized_cache.py",
        "shape_contract": root / "global_shape_model.py",
        "geometry": root / "geometry.py",
    }
    protocol = {
        "schema_version": "pg-orbitflow-dataset-training-protocol-v1",
        "mode": args.mode,
        "seed": 20263101,
        "package_dir": str(package_dir),
        "canonical_manifest_sha256": _sha256(package_dir / "manifest.json"),
        "data": {
            "split_scheme": "iid",
            "point_groups": ["C2", "C3"],
            "train_split": "train",
            "validation_split": "val",
            "train_total_c2_c3": train_total,
            "train_supported_count": len(train_records),
            "train_unsupported": train_unsupported,
            "validation_total_c2_c3": validation_total,
            "validation_supported_count": len(validation_records),
            "validation_unsupported": validation_unsupported,
            "train_factorized_unsupported": factorized_train_unsupported,
            "validation_factorized_unsupported": factorized_validation_unsupported,
            "train_records": selected_train,
            "validation_records": selected_validation,
            "selection": (
                "smoke: first 32/8 records per PG after package-index sorting, N<=30; "
                "full: every strictly supported record in the frozen split"
            ),
        },
        "model": model_setting,
        "model_initialization": {
            "type": "fresh deterministic random initialization",
            "m3_checkpoint_loaded": False,
            "legacy_local_rotor_head_frozen": True,
            "trainable_parameter_count": isolation["trainable_parameter_count"],
        },
        "global_shape_contract": shape_setting,
        "training": {
            **training,
            "loss_weights": {
                "bond_mse": 10.0,
                "angle_cosine_mse": 2.0,
                "torsion_circular": 2.0,
                "global_shape_mse": 1.0,
                "worst_molecule_torsion": 1.0,
                "worst_molecule_nonplanar": 1.0,
            },
            "torsion_assignment": (
                "one global action-augmented graph-isomorphism witness permutation; "
                "finite torsion-permutation closure capped at 10000"
            ),
        },
        "parent_evidence": {
            "m3_protocol": {"path": str(m3_protocol_path), "sha256": _sha256(m3_protocol_path)},
            "m3_report": {"path": str(m3_report_path), "sha256": _sha256(m3_report_path)},
            "m4_unseen_failure": {"path": str(m4_report_path), "sha256": _sha256(m4_report_path)},
            "factorized_audit": {"path": str(factorized_path), "sha256": _sha256(factorized_path)},
            **(
                {
                    "dataset_smoke_audit": {
                        "path": str(smoke_audit_path),
                        "sha256": _sha256(smoke_audit_path),
                    }
                }
                if smoke_audit_path is not None
                else {}
            ),
        },
        "implementation": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "smoke_gate": {
            "all_losses_and_gradients_finite": True,
            "training_loss_last32_over_first32_max": 1.0,
            "checkpoint_reload_prediction_max_abs_difference": 0.0,
            "train_validation_overlap_count": 0,
            "c2_c3_present_in_every_batch": True,
            "unsupported_records_explicit": True,
            "validation_used_for_gradient_updates": False,
        },
        "scope": {
            "iid_test_used": False,
            "core_ood_used": False,
            "validation_used_for_training_or_model_selection": False,
            "target_cartesian_model_input_used": False,
            "hard_projection_used": False,
            "f0p2_used": False,
            "quality_claim": False,
        },
    }
    if cache_spec is not None:
        protocol["factorized_cache"] = cache_spec
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"WROTE_DATASET_PROTOCOL mode={args.mode} train={len(selected_train)} "
        f"validation={len(selected_validation)} unsupported_train={len(train_unsupported)} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
