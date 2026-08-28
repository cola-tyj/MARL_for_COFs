"""Run the frozen unconstrained GFN2-xTB relaxation panel in ``env_cof``."""

from __future__ import annotations

import argparse
from importlib.metadata import version as package_version
import json
import os
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from generative_model.data.build_cof_package import save_deterministic_npz
from generative_model.data.cof_graph_dataset import COFGraphDataset
from generative_model.evaluation.our_etflow_xtb_relaxation import (
    center_positions,
    electronic_state,
    maximum_force_norm,
    relaxation_geometry_metrics,
)
from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT,
    _atomic_json,
    _fingerprint,
    _sha256,
)


DEFAULT_PROTOCOL = ROOT / "generative_model/evaluation/our_etflow_xtb_protocol_v1.json"
DEFAULT_SOURCE = ROOT / "generative_model/runs/our_etflow_ablation_v1"
DEFAULT_OUTPUT = ROOT / "generative_model/runs/our_etflow_xtb_relaxation_v1"
DEFAULT_PACKAGE = ROOT / "generative_model/data/processed/v2"


def _validate_protocol(protocol_path: Path, source_dir: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_fingerprint") != _fingerprint(protocol):
        raise RuntimeError("xTB protocol fingerprint mismatch")
    identity = protocol["identity"]
    expected_inputs = {
        "parent_ablation_report_sha256": source_dir / "report.json",
        "source_coordinates_sha256": source_dir / "coordinates.npz",
        "source_point_group_report_sha256": source_dir / "point_group_report.json",
    }
    for key, path in expected_inputs.items():
        if _sha256(path) != identity[key]:
            raise RuntimeError(f"xTB frozen input hash mismatch: {path}")
    for relative, expected in identity["source_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"xTB source hash mismatch: {relative}")
    return protocol


def _make_calculator(settings: dict[str, Any], state: dict[str, int]):
    try:
        from tblite.ase import TBLite
    except ImportError as error:
        raise RuntimeError(
            "tblite is required; install it in env_cof before running this protocol"
        ) from error
    return TBLite(
        method=str(settings["method"]),
        charge=int(state["charge"]),
        multiplicity=int(state["multiplicity"]),
        accuracy=float(settings["accuracy"]),
        max_iterations=int(settings["maximum_scf_iterations"]),
        electronic_temperature=float(settings["electronic_temperature_kelvin"]),
        verbosity=int(settings["calculator_verbosity"]),
    )


def run(
    protocol_path: Path,
    source_dir: Path,
    package_dir: Path,
    output_dir: Path,
    panel_limit: int | None = None,
) -> dict[str, Any]:
    try:
        import ase
        import tblite
        from ase import Atoms
        from ase.optimize import LBFGS
    except ImportError as error:
        raise RuntimeError("ASE and tblite must both be installed in env_cof") from error

    protocol = _validate_protocol(protocol_path, source_dir)
    observed_tblite_version = package_version("tblite")
    if observed_tblite_version != str(
        protocol["relaxation"]["tblite_version"]
    ):
        raise RuntimeError("tblite version does not match the frozen protocol")
    if str(ase.__version__) != str(protocol["relaxation"]["ase_version"]):
        raise RuntimeError("ASE version does not match the frozen protocol")
    for name, expected in protocol["relaxation"]["thread_environment"].items():
        if os.environ.get(name) != str(expected):
            raise RuntimeError(f"{name} must equal {expected} for the frozen protocol")
    panel = protocol["panel"]["records"]
    if panel_limit is not None:
        if not 1 <= panel_limit <= len(panel):
            raise ValueError("panel_limit must be within the frozen panel")
        panel = panel[:panel_limit]
    dataset = COFGraphDataset(package_dir)
    with np.load(source_dir / "coordinates.npz", allow_pickle=False) as archive:
        source_arrays = {name: archive[name] for name in archive.files}
    source_offsets = np.asarray(source_arrays["atom_offsets"], dtype=np.int64)
    source_positions_all = np.asarray(
        source_arrays["positions_etflow_hard_projection_f02"], dtype=np.float64
    )
    atom_counts = np.asarray([int(item["atom_count"]) for item in panel], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(atom_counts))).astype("<i8")
    total_atoms = int(offsets[-1])
    input_positions = np.full((total_atoms, 3), np.nan, dtype="<f8")
    relaxed_positions = np.full((total_atoms, 3), np.nan, dtype="<f8")
    atomic_numbers = np.empty(total_atoms, dtype="<i8")
    success = np.zeros(len(panel), dtype=np.bool_)
    records: list[dict[str, Any]] = []
    settings = protocol["relaxation"]
    run_started = perf_counter()
    for local_id, item in enumerate(panel):
        panel_id = int(item["panel_id"])
        source_start, source_end = map(int, source_offsets[panel_id : panel_id + 2])
        out_start, out_end = map(int, offsets[local_id : local_id + 2])
        sample = dataset[int(item["package_index"])]
        if str(sample["molecule_id"]) != str(item["molecule_id"]):
            raise RuntimeError("xTB panel molecule identity mismatch")
        before = center_positions(source_positions_all[source_start:source_end])
        if len(before) != out_end - out_start:
            raise RuntimeError("xTB panel atom count mismatch")
        input_positions[out_start:out_end] = before
        atomic_numbers[out_start:out_end] = sample["atomic_numbers"]
        state = electronic_state(sample["formal_charges"], sample["radical_electrons"])
        record: dict[str, Any] = {
            "panel_id": panel_id,
            "package_index": int(item["package_index"]),
            "molecule_id": str(item["molecule_id"]),
            "target_pg": str(item["target_pg"]),
            "atom_count": len(before),
            "electronic_state": state,
            "success": False,
        }
        started = perf_counter()
        try:
            atoms = Atoms(
                numbers=np.asarray(sample["atomic_numbers"], dtype=np.int64),
                positions=before,
                pbc=False,
            )
            atoms.calc = _make_calculator(settings, state)
            energy_before = float(atoms.get_potential_energy())
            forces_before = np.asarray(atoms.get_forces(), dtype=np.float64)
            optimizer = LBFGS(atoms, logfile=None)
            converged = bool(optimizer.run(
                fmax=float(settings["fmax_ev_per_angstrom"]),
                steps=int(settings["maximum_geometry_steps"]),
            ))
            energy_after = float(atoms.get_potential_energy())
            forces_after = np.asarray(atoms.get_forces(), dtype=np.float64)
            after = center_positions(np.asarray(atoms.positions, dtype=np.float64))
            if not np.isfinite([energy_before, energy_after]).all():
                raise RuntimeError("GFN2-xTB returned a non-finite energy")
            metrics = relaxation_geometry_metrics(
                before, after, sample["positions"], sample["bond_index"]
            )
            relaxed_positions[out_start:out_end] = after
            success[local_id] = True
            record.update({
                "success": True,
                "optimizer_converged": converged,
                "optimizer_steps": int(optimizer.nsteps),
                "energy_before_ev": energy_before,
                "energy_after_ev": energy_after,
                "energy_change_ev": energy_after - energy_before,
                "energy_before_ev_per_atom": energy_before / len(before),
                "energy_after_ev_per_atom": energy_after / len(before),
                "energy_change_ev_per_atom": (energy_after - energy_before) / len(before),
                "maximum_force_before_ev_per_angstrom": maximum_force_norm(forces_before),
                "maximum_force_after_ev_per_angstrom": maximum_force_norm(forces_after),
                **metrics,
            })
        except Exception as error:
            record["failure"] = f"{type(error).__name__}: {error}"
        record["runtime_seconds"] = perf_counter() - started
        records.append(record)
        print(json.dumps({
            "completed": local_id + 1,
            "total": len(panel),
            "package_index": record["package_index"],
            "success": record["success"],
        }, ensure_ascii=False), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = output_dir / "relaxed_coordinates.npz"
    save_deterministic_npz(coordinates_path, {
        "atom_offsets": offsets,
        "atomic_numbers": atomic_numbers,
        "input_positions": input_positions,
        "relaxed_positions": relaxed_positions,
        "success": success,
    })
    successful = [record for record in records if record["success"]]
    finite = all(
        np.isfinite([
            record["energy_before_ev"],
            record["energy_after_ev"],
            record["maximum_force_before_ev_per_angstrom"],
            record["maximum_force_after_ev_per_angstrom"],
        ]).all()
        for record in successful
    )
    report = {
        "schema_version": "our-etflow-gfn2-xtb-relaxation-run-v1",
        "status": "PASS_EXECUTION_AWAITING_POINT_GROUP_AUDIT",
        "quality_claim": False,
        "panel_count": len(panel),
        "full_frozen_panel": len(panel) == int(protocol["panel"]["molecule_count"]),
        "records": records,
        "summary_pre_point_group": {
            "calculation_success_count": len(successful),
            "calculation_success_fraction_full_panel": len(successful) / len(panel),
            "optimizer_converged_fraction_full_panel": sum(
                bool(record.get("optimizer_converged")) for record in records
            ) / len(panel),
            "energy_nonincreasing_fraction_among_success": (
                sum(record["energy_change_ev"] <= 1e-8 for record in successful)
                / len(successful) if successful else 0.0
            ),
            "collision_free_after_fraction_full_panel": sum(
                bool(record.get("collision_free_after_at_0p6")) for record in records
            ) / len(panel),
            "median_kabsch_rmsd_pre_to_post_angstrom": (
                None if not successful else float(np.median([
                    record["kabsch_rmsd_pre_to_post_angstrom"] for record in successful
                ]))
            ),
            "all_successful_values_finite": finite,
        },
        "runtime": {"total_seconds": perf_counter() - run_started},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ase": ase.__version__,
            "tblite": observed_tblite_version,
        },
        "identity": {
            "protocol_sha256": _sha256(protocol_path),
            "source_coordinates_sha256": _sha256(source_dir / "coordinates.npz"),
            "relaxed_coordinates_sha256": _sha256(coordinates_path),
            "runner_source_sha256": _sha256(Path(__file__)),
        },
    }
    _atomic_json(output_dir / "relaxation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-limit", type=int)
    args = parser.parse_args()
    report = run(
        args.protocol.resolve(), args.source_dir.resolve(), args.package.resolve(),
        args.output_dir.resolve(), args.panel_limit,
    )
    print(f"{report['status']} report={args.output_dir / 'relaxation_report.json'}")


if __name__ == "__main__":
    main()
