"""Export reproducible known-graph/action -> XYZ examples from formal E2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from generative_model.data.v2_dataset import COFSymmetryDataset


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "generative_model/data/processed/v2"
RUN = ROOT / "generative_model/runs/etkdg_orbit_e2"
DEFAULT_OUTPUT = ROOT / "generative_model/visualization/e2_examples"
DEFAULT_EXAMPLES = (
    ("c2_exact", 45, "C2", "C2"),
    ("c2_compatible_supergroup", 92, "C2", "C2h"),
    ("c3_exact", 81, "C3", "C3"),
    ("c3_compatible_supergroup", 65, "C3", "C3h"),
    ("c2_sn_exact", 1684, "C2", "C2"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_xyz(
    path: Path,
    symbols: list[str],
    positions: np.ndarray,
    comment: str,
) -> None:
    if positions.shape != (len(symbols), 3) or not np.isfinite(positions).all():
        raise ValueError(f"XYZ shape/finite validation failed: {path}")
    lines = [str(len(symbols)), comment]
    lines.extend(
        f"{symbol:<2s} {x: .10f} {y: .10f} {z: .10f}"
        for symbol, (x, y, z) in zip(symbols, positions, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_examples(output_dir: Path) -> dict[str, Any]:
    geometry_path = RUN / "geometry_report.json"
    coordinates_path = RUN / "coordinates.npz"
    point_group_path = RUN / "point_group_report.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    point_group = json.loads(point_group_path.read_text(encoding="utf-8"))
    if geometry["status"] != "PASS_ETKDG_ORBIT_E2_GEOMETRY_AWAIT_POINT_GROUP_AUDIT":
        raise RuntimeError("formal E2 geometry gate is not passed")
    if point_group["status"] != "PASS_ETKDG_ORBIT_E2_DATASET_SCALE_C23":
        raise RuntimeError("formal E2 point-group gate is not passed")
    if geometry["coordinates_sha256"] != _sha256(coordinates_path):
        raise RuntimeError("E2 coordinate SHA does not match geometry report")

    dataset = COFSymmetryDataset(PACKAGE)
    point_group_lookup = {
        int(record["package_index"]): record for record in point_group["records"]
    }
    number_to_symbol = {
        int(number): symbol
        for symbol, number in dataset.vocab["atomic_numbers"].items()
    }
    with np.load(coordinates_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    coordinate_lookup = {
        int(package_index): position
        for position, package_index in enumerate(arrays["package_indices"])
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, Any]] = []
    for ordinal, (label, package_index, expected_target, expected_actual) in enumerate(
        DEFAULT_EXAMPLES, start=1
    ):
        sample = dataset[package_index]
        pg_record = point_group_lookup[package_index]
        if sample["target_pg"] != expected_target:
            raise RuntimeError(f"example {package_index} target PG changed")
        if pg_record["actual_pg"] != expected_actual or not pg_record["pg_compatible"]:
            raise RuntimeError(f"example {package_index} actual/compatible result changed")
        coordinate_position = coordinate_lookup[package_index]
        start, end = map(
            int, arrays["atom_offsets"][coordinate_position : coordinate_position + 2]
        )
        positions = np.asarray(arrays["positions"][start:end], dtype=np.float64)
        symbols = []
        for atomic_number in sample["atomic_numbers"]:
            if int(atomic_number) not in number_to_symbol:
                raise RuntimeError(f"no element fallback allowed: Z={int(atomic_number)}")
            symbols.append(number_to_symbol[int(atomic_number)])
        stem = f"{ordinal:02d}_{label}_{sample['molecule_id']}"
        xyz_path = output_dir / f"{stem}.xyz"
        metadata_path = output_dir / f"{stem}.json"
        molecule_record_path = RUN / "records" / f"package_{package_index:06d}.json"
        molecule_record = json.loads(molecule_record_path.read_text(encoding="utf-8"))
        comment = (
            f"E2 output molecule_id={sample['molecule_id']} package_index={package_index} "
            f"target_pg={sample['target_pg']} actual_pg={pg_record['actual_pg']} "
            f"compatible={pg_record['pg_compatible']} coordinates=centroid-centered_angstrom"
        )
        _write_xyz(xyz_path, symbols, positions, comment)
        target_action = {
            "operation_matrices": sample["target_symmetry"]["operation_matrices"],
            "permutation_index": sample["target_symmetry"]["permutation_index"],
            "orbit_id": sample["target_orbit_id"],
            "orbit_size": sample["target_orbit_size"],
        }
        metadata = _jsonable({
            "schema_version": "etkdg-orbit-e2-visual-example-v1",
            "label": label,
            "input": {
                "package_index": package_index,
                "molecule_id": sample["molecule_id"],
                "source_split": pg_record["source_split"],
                "smiles": sample["metadata"]["SMILES"],
                "known_graph": {
                    "atomic_numbers": sample["atomic_numbers"],
                    "formal_charges": sample["formal_charges"],
                    "radical_electrons": sample["radical_electrons"],
                    "bond_index": sample["bond_index"],
                    "bond_types": sample["bond_types"],
                },
                "known_target_pg_orbit_action": {
                    "target_pg": sample["target_pg"],
                    **target_action,
                },
            },
            "output": {
                "xyz_file": xyz_path.name,
                "xyz_sha256": _sha256(xyz_path),
                "coordinate_unit": "angstrom",
                "centroid_centered": True,
                "atom_count": len(symbols),
                "selected_seed": molecule_record["selected"]["seed"],
                "selected_orientation_id": molecule_record["selected"]["orientation_id"],
                "minimum_pair_distance_angstrom": molecule_record["selected"][
                    "minimum_pair_distance_angstrom"
                ],
                "bond_length_mae_to_reference_angstrom": molecule_record["selected"][
                    "bond_length_mae_to_reference_angstrom"
                ],
                "maximum_operation_error_angstrom": molecule_record["selected"][
                    "maximum_operation_error_angstrom"
                ],
                "analyzer_pg": pg_record["analyzer_pg"],
                "actual_pg": pg_record["actual_pg"],
                "pg_exact_match": pg_record["pg_exact_match"],
                "pg_compatible": pg_record["pg_compatible"],
                "contains_sn": pg_record["contains_sn"],
            },
            "identity": {
                "e2_molecule_record_sha256": _sha256(molecule_record_path),
                "e2_coordinates_sha256": _sha256(coordinates_path),
                "e2_geometry_report_sha256": _sha256(geometry_path),
                "e2_point_group_report_sha256": _sha256(point_group_path),
            },
        })
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_records.append({
            "label": label,
            "package_index": package_index,
            "molecule_id": sample["molecule_id"],
            "target_pg": sample["target_pg"],
            "actual_pg": pg_record["actual_pg"],
            "pg_exact_match": pg_record["pg_exact_match"],
            "pg_compatible": pg_record["pg_compatible"],
            "contains_sn": pg_record["contains_sn"],
            "atom_count": len(symbols),
            "xyz_file": xyz_path.name,
            "metadata_file": metadata_path.name,
            "xyz_sha256": _sha256(xyz_path),
            "metadata_sha256": _sha256(metadata_path),
        })
    manifest = {
        "schema_version": "etkdg-orbit-e2-visual-examples-v1",
        "description": "Representative formal E2 known-graph/action to XYZ outputs",
        "records": manifest_records,
        "identity": {
            "geometry_report_sha256": _sha256(geometry_path),
            "coordinates_sha256": _sha256(coordinates_path),
            "point_group_report_sha256": _sha256(point_group_path),
        },
    }
    manifest_path = output_dir / "examples.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"EXPORTED examples={len(manifest_records)} manifest={manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_examples(args.output_dir.resolve())


if __name__ == "__main__":
    main()
