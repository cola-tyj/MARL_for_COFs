"""Precompute restartable factorized contracts for a frozen dataset protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .dataset_training import _load, _samples
from .factorized_automorphism import build_factorized_automorphism_contract
from .factorized_cache import cache_path, load_contract, save_contract, selection_fingerprint
from .geometry import build_geometry_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = _load(protocol_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = protocol["data"]["train_records"]
    validation_records = protocol["data"]["validation_records"]
    records = train_records + validation_records
    entries = [
        ("train", sample, record)
        for sample, record in zip(
            _samples(protocol, split="train", records=train_records),
            train_records,
            strict=True,
        )
    ]
    entries.extend(
        ("val", sample, record)
        for sample, record in zip(
            _samples(protocol, split="val", records=validation_records),
            validation_records,
            strict=True,
        )
    )
    samples = [sample for _split, sample, _record in entries]
    if len(samples) != len(records):
        raise RuntimeError("cache sample count mismatch")

    artifact_rows = []
    unsupported_rows = []
    for position, (split, sample, record) in enumerate(entries, start=1):
        path = cache_path(output_dir, sample.package_index)
        unsupported_path = output_dir / f"{int(sample.package_index):06d}.unsupported.json"
        if path.is_file():
            contract = load_contract(path)
        elif unsupported_path.is_file():
            failure = json.loads(unsupported_path.read_text(encoding="utf-8"))
            expected = {
                "package_index": int(sample.package_index),
                "molecule_id": sample.molecule_id,
                "target_pg": sample.target_pg,
                "num_atoms": len(sample.atom_types),
                "split": split,
            }
            if any(failure.get(key) != value for key, value in expected.items()):
                raise RuntimeError("cached unsupported identity changed")
            unsupported_rows.append(
                {**failure, "path": unsupported_path.name, "sha256": _sha256(unsupported_path)}
            )
            if position == 1 or position % 25 == 0 or position == len(samples):
                print(
                    json.dumps(
                        {
                            "completed": position,
                            "total": len(samples),
                            "package_index": int(sample.package_index),
                            "status": "explicit_unsupported",
                        }
                    ),
                    flush=True,
                )
            continue
        else:
            geometry = build_geometry_contract(sample)
            try:
                contract = build_factorized_automorphism_contract(sample, geometry)
            except (ValueError, RuntimeError) as error:
                failure = {
                    "package_index": int(sample.package_index),
                    "molecule_id": sample.molecule_id,
                    "target_pg": sample.target_pg,
                    "num_atoms": len(sample.atom_types),
                    "split": split,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
                unsupported_path.write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                unsupported_rows.append(
                    {**failure, "path": unsupported_path.name, "sha256": _sha256(unsupported_path)}
                )
                print(
                    json.dumps(
                        {
                            "completed": position,
                            "total": len(samples),
                            "package_index": int(sample.package_index),
                            "status": "explicit_unsupported",
                            "reason": str(error),
                        }
                    ),
                    flush=True,
                )
                continue
            save_contract(path, contract)
            contract = load_contract(path)
        geometry = build_geometry_contract(sample)
        expected_torsions = int(np.max(geometry.torsion_orbit_id)) + 1
        if contract.allowed_torsion_permutations.shape[1] != expected_torsions:
            raise RuntimeError("cached contract no longer matches geometry")
        artifact_rows.append(
            {
                "package_index": int(sample.package_index),
                "molecule_id": sample.molecule_id,
                "target_pg": sample.target_pg,
                "num_atoms": len(sample.atom_types),
                "split": split,
                "path": path.name,
                "sha256": _sha256(path),
                "allowed_torsion_permutation_count": int(
                    contract.allowed_torsion_permutation_count
                ),
            }
        )
        if position == 1 or position % 25 == 0 or position == len(samples):
            print(
                json.dumps(
                    {"completed": position, "total": len(samples), "package_index": int(sample.package_index)}
                ),
                flush=True,
            )

    manifest = {
        "schema_version": "pg-orbitflow-factorized-cache-manifest-v1",
        "status": "PASS_FACTORIZED_CACHE_COMPLETE_WITH_EXPLICIT_UNSUPPORTED",
        "passed": True,
        "source_protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "canonical_manifest_sha256": protocol["canonical_manifest_sha256"],
        "factorized_implementation": protocol["implementation"]["factorized_automorphism"],
        "source_record_count": len(records),
        "source_selection_fingerprint": selection_fingerprint(records),
        "cached_record_count": len(artifact_rows),
        "cached_selection_fingerprint": selection_fingerprint(artifact_rows),
        "unsupported_record_count": len(unsupported_rows),
        "records": artifact_rows,
        "unsupported_records": unsupported_rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PASS_FACTORIZED_CACHE_COMPLETE_WITH_EXPLICIT_UNSUPPORTED "
        f"cached={len(artifact_rows)} unsupported={len(unsupported_rows)} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
