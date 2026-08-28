"""Build deterministic presentation assets from frozen our_ET_Flow evaluations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generative_model.evaluation.our_etflow_ablation import ROUTES
from generative_model.data.cof_graph_dataset import COFGraphDataset
from generative_model.conformer.etflow_ef1_projection import kabsch_rmsd
from generative_model.inference.generate_etflow_symmetric_xyz import (
    ROOT,
    _atomic_json,
    _sha256,
)


RUN_ROOT = ROOT / "generative_model/runs"
OUTPUT = (
    ROOT
    / "docs/presentation/generative_model_progress_20260821/assets/our_etflow_eval_v1"
)
RUNS = {
    "paired_ablation": RUN_ROOT / "our_etflow_ablation_v1",
    "core_ood": RUN_ROOT / "our_etflow_core_ood_v1",
    "external": RUN_ROOT / "our_etflow_external_v1",
    "controllability": RUN_ROOT / "our_etflow_controllability_v1",
}
XTB_RUN = RUN_ROOT / "our_etflow_xtb_relaxation_v1"
LABELS = {
    "etkdg_v3_best_of_n": "ETKDGv3",
    "etflow_raw": "ET-Flow raw",
    "etflow_hard_projection": "+ hard projection",
    "etflow_hard_projection_f02": "+ projection + F0.2",
}
SHORT_LABELS = {
    "etkdg_v3_best_of_n": "ETKDGv3",
    "etflow_raw": "Raw",
    "etflow_hard_projection": "Hard",
    "etflow_hard_projection_f02": "Full F0.2",
}
COLORS = {
    "etkdg_v3_best_of_n": "#7f8c8d",
    "etflow_raw": "#377eb8",
    "etflow_hard_projection": "#e41a1c",
    "etflow_hard_projection_f02": "#4daf4a",
}
ELEMENTS = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si",
    15: "P", 16: "S", 17: "Cl", 35: "Br", 50: "Sn", 53: "I",
}
ELEMENT_COLORS = {
    "H": "#f2f2f2", "B": "#f4a460", "C": "#555555", "N": "#3050f8",
    "O": "#ff0d0d", "F": "#90e050", "Si": "#f0c8a0", "P": "#ff8000",
    "S": "#ffff30", "Cl": "#1ff01f", "Br": "#a62929", "Sn": "#668080",
    "I": "#940094",
}


def _load(run: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    point_group = json.loads((run / "point_group_report.json").read_text(encoding="utf-8"))
    with np.load(run / "coordinates.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return report, point_group, arrays


def _save_figure(figure: plt.Figure, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    figure.savefig(png, dpi=180, bbox_inches="tight", metadata={"Software": "matplotlib"})
    figure.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    return [png, svg]


def _route_summary(
    report: dict[str, Any], point_group: dict[str, Any], route: str
) -> dict[str, float]:
    pre = report["summary_pre_point_group"][route]
    pg = point_group["summary"][route]
    panel = float(pre["full_panel_count"])
    joint = 0
    for geometry_record, symmetry_record in zip(
        report["records"], point_group["records"], strict=True
    ):
        geometry = geometry_record["routes"][route]
        symmetry = symmetry_record["routes"][route]
        joint += bool(
            geometry.get("success")
            and geometry.get("collision_free_at_0p6")
            and symmetry.get("pg_compatible")
        )
    return {
        "panel_count": panel,
        "generation_fraction": float(pre["success_fraction"]),
        "pg_compatible_fraction": float(pg["pg_compatible_fraction_full_panel"]),
        "collision_free_fraction": float(pre["collision_free_fraction_full_panel"]),
        "joint_fraction": joint / panel,
        "mean_bond_mae_angstrom": float(
            pre["mean_bond_length_mae_to_canonical_angstrom"]
        ),
        "median_uff_energy_per_atom_kcal_mol": float(
            pre["median_uff_single_point_energy_per_atom_kcal_mol"]
        ),
        "median_runtime_seconds": float(pre["median_runtime_seconds"]),
    }


def _plot_ablation_overview(
    report: dict[str, Any], point_group: dict[str, Any]
) -> list[Path]:
    summaries = {route: _route_summary(report, point_group, route) for route in ROUTES}
    labels = [SHORT_LABELS[route] for route in ROUTES]
    colors = [COLORS[route] for route in ROUTES]
    x = np.arange(len(ROUTES))
    figure, axes = plt.subplots(2, 2, figsize=(12, 9.4), layout="constrained")
    width = 0.36
    axes[0, 0].bar(
        x - width / 2,
        [summaries[route]["pg_compatible_fraction"] for route in ROUTES],
        width, label="PG-compatible", color="#984ea3",
    )
    axes[0, 0].bar(
        x + width / 2,
        [summaries[route]["collision_free_fraction"] for route in ROUTES],
        width, label="Collision-free", color="#ff7f00",
    )
    axes[0, 0].set_ylim(0, 1.08)
    axes[0, 0].set_ylabel("Full-panel fraction")
    axes[0, 0].set_title("Symmetry and collision")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(
        x, [summaries[route]["mean_bond_mae_angstrom"] for route in ROUTES],
        color=colors,
    )
    axes[0, 1].set_ylabel("Mean bond-length MAE (Å)")
    axes[0, 1].set_title("Bond geometry")

    axes[1, 0].bar(
        x, [summaries[route]["median_uff_energy_per_atom_kcal_mol"] for route in ROUTES],
        color=colors,
    )
    axes[1, 0].set_yscale("symlog", linthresh=1.0)
    axes[1, 0].set_ylabel("Median UFF energy / atom (kcal mol$^{-1}$)")
    axes[1, 0].set_title("Reference-free candidate energy")

    axes[1, 1].bar(
        x, [summaries[route]["median_runtime_seconds"] for route in ROUTES],
        color=colors,
    )
    axes[1, 1].set_ylabel("Median wall runtime (s / molecule)")
    axes[1, 1].set_title("End-to-end runtime")
    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Paired four-route ablation (C2/C3/S4/D6h, n=32)", fontsize=14)
    return _save_figure(figure, OUTPUT / "paired_ablation_overview")


def _plot_pg_success(report: dict[str, Any], point_group: dict[str, Any]) -> list[Path]:
    targets = ("C2", "C3", "S4", "D6h")
    x = np.arange(len(targets))
    width = 0.19
    figure, axis = plt.subplots(figsize=(10, 5.4))
    for route_id, route in enumerate(ROUTES):
        values = [
            point_group["summary"][route]["by_target_pg"][target][
                "pg_compatible_fraction"
            ]
            for target in targets
        ]
        axis.bar(
            x + (route_id - 1.5) * width, values, width,
            label=LABELS[route], color=COLORS[route],
        )
    axis.set_xticks(x, targets)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("PG-compatible fraction")
    axis.set_title("Success by requested point group (full denominator)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    return _save_figure(figure, OUTPUT / "point_group_success_rates")


def _plot_energy_symmetry(report: dict[str, Any], point_group: dict[str, Any]) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8.5, 6))
    for route in ROUTES:
        xs, ys = [], []
        for geometry_record, symmetry_record in zip(
            report["records"], point_group["records"], strict=True
        ):
            geometry = geometry_record["routes"][route]
            symmetry = symmetry_record["routes"][route]
            if not geometry.get("success") or not symmetry.get("analyzer_success"):
                continue
            xs.append(max(float(symmetry["actual_mean_rms_error_angstrom"]), 1e-18))
            ys.append(float(geometry["uff_single_point_energy_per_atom_kcal_mol"]))
        axis.scatter(xs, ys, s=32, alpha=0.76, label=LABELS[route], color=COLORS[route])
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=1.0)
    axis.set_xlabel("Actual-group mean operation RMS error (Å)")
    axis.set_ylabel("UFF single-point energy / atom (kcal mol$^{-1}$)")
    axis.set_title("Symmetry precision does not imply physical quality")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    return _save_figure(figure, OUTPUT / "energy_vs_symmetry_error")


def _plot_runtime_success(report: dict[str, Any], point_group: dict[str, Any]) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for route in ROUTES:
        summary = _route_summary(report, point_group, route)
        axis.scatter(
            summary["median_runtime_seconds"], summary["joint_fraction"],
            s=170, color=COLORS[route], label=LABELS[route], edgecolor="black",
        )
        axis.annotate(
            LABELS[route],
            (summary["median_runtime_seconds"], summary["joint_fraction"]),
            xytext=(5, 6), textcoords="offset points", fontsize=9,
        )
    axis.set_ylim(0, 1.08)
    axis.set_xlabel("Median end-to-end runtime (s / molecule)")
    axis.set_ylabel("PG-compatible AND collision-free fraction")
    axis.set_title("Runtime–success trade-off (paired n=32)")
    axis.grid(alpha=0.2)
    return _save_figure(figure, OUTPUT / "runtime_vs_joint_success")


def _load_xtb() -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    report = json.loads((XTB_RUN / "relaxation_report.json").read_text(encoding="utf-8"))
    point_group = json.loads((XTB_RUN / "point_group_report.json").read_text(encoding="utf-8"))
    with np.load(XTB_RUN / "relaxed_coordinates.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return report, point_group, arrays


def _plot_xtb_overview(report: dict[str, Any], point_group: dict[str, Any]) -> list[Path]:
    targets = ("C2", "C3", "S4", "D6h")
    target_colors = {"C2": "#377eb8", "C3": "#4daf4a", "S4": "#984ea3", "D6h": "#ff7f00"}
    joined = list(zip(report["records"], point_group["records"], strict=True))
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for target_id, target in enumerate(targets):
        selected = [(record, audit) for record, audit in joined if record["target_pg"] == target]
        jitter = np.linspace(-0.12, 0.12, len(selected))
        axes[0, 0].scatter(
            target_id + jitter,
            [record["energy_change_ev_per_atom"] for record, _ in selected],
            color=target_colors[target], edgecolor="#333333", linewidth=0.3,
        )
        axes[0, 1].scatter(
            target_id + jitter,
            [record["kabsch_rmsd_pre_to_post_angstrom"] for record, _ in selected],
            color=target_colors[target], edgecolor="#333333", linewidth=0.3,
        )
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].set_ylabel("GFN2-xTB energy change / atom (eV)")
    axes[0, 0].set_title("Energy relaxation (all 32 decrease)")
    axes[0, 1].axhline(0.5, color="#e41a1c", ls="--", lw=1, label="Gate 0.5 Å")
    axes[0, 1].set_ylabel("Pre/post Kabsch RMSD (Å)")
    axes[0, 1].set_title("Geometry displacement")
    axes[0, 1].legend(frameon=False)

    x = np.arange(len(targets))
    before = [point_group["summary"]["by_target_pg"][target]["before_compatible_fraction"] for target in targets]
    after = [point_group["summary"]["by_target_pg"][target]["after_compatible_fraction"] for target in targets]
    axes[1, 0].bar(x - 0.18, before, 0.36, label="Before", color="#999999")
    axes[1, 0].bar(x + 0.18, after, 0.36, label="After", color="#4daf4a")
    axes[1, 0].set_ylim(0, 1.08)
    axes[1, 0].set_ylabel("PG-compatible fraction")
    axes[1, 0].set_title("Point-group retention")
    axes[1, 0].legend(frameon=False)

    before_forces = [record["maximum_force_before_ev_per_angstrom"] for record, _ in joined]
    after_forces = [record["maximum_force_after_ev_per_angstrom"] for record, _ in joined]
    for index, (left, right) in enumerate(zip(before_forces, after_forces, strict=True)):
        axes[1, 1].plot((0, 1), (left, right), color="#bbbbbb", alpha=0.45, lw=0.8)
    axes[1, 1].scatter(np.zeros(len(joined)), before_forces, color="#e41a1c", s=24, label="Before")
    axes[1, 1].scatter(np.ones(len(joined)), after_forces, color="#4daf4a", s=24, label="After")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks((0, 1), ("Before", "After"))
    axes[1, 1].set_ylabel("Maximum force (eV Å$^{-1}$)")
    axes[1, 1].set_title("Force reduction")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(x, targets)
        axis.grid(axis="y", alpha=0.2)
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.suptitle("Unconstrained GFN2-xTB physical-stability pilot (n=32)", fontsize=14)
    return _save_figure(figure, OUTPUT / "xtb_relaxation_overview")


def _plot_xtb_energy_symmetry(report: dict[str, Any], point_group: dict[str, Any]) -> list[Path]:
    target_colors = {"C2": "#377eb8", "C3": "#4daf4a", "S4": "#984ea3", "D6h": "#ff7f00"}
    figure, axis = plt.subplots(figsize=(8.5, 6))
    for record, audit in zip(report["records"], point_group["records"], strict=True):
        compatible = bool(audit["after"]["pg_compatible"])
        axis.scatter(
            max(float(audit["after"]["actual_mean_rms_error_angstrom"]), 1e-18),
            float(record["energy_change_ev_per_atom"]),
            color=target_colors[record["target_pg"]],
            marker="o" if compatible else "X",
            s=45 if compatible else 130,
            edgecolor="black", linewidth=0.4,
        )
        if not compatible:
            axis.annotate(
                f"{record['molecule_id']}\n{audit['before']['actual_pg']}→{audit['after']['actual_pg']}",
                (audit["after"]["actual_mean_rms_error_angstrom"], record["energy_change_ev_per_atom"]),
                xytext=(8, 6), textcoords="offset points", fontsize=9,
            )
    for target, color in target_colors.items():
        axis.scatter([], [], color=color, label=target)
    axis.set_xscale("log")
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xlabel("Post-relaxation actual-group mean operation RMS error (Å)")
    axis.set_ylabel("GFN2-xTB energy change / atom (eV)")
    axis.set_title("Energy relaxation versus retained symmetry precision")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=4)
    return _save_figure(figure, OUTPUT / "xtb_energy_change_vs_symmetry_error")


def _write_xyz(path: Path, numbers: np.ndarray, positions: np.ndarray, comment: str) -> None:
    symbols = []
    for number in np.asarray(numbers, dtype=np.int64):
        if int(number) not in ELEMENTS:
            raise ValueError(f"unsupported element Z={int(number)}; fallback forbidden")
        symbols.append(ELEMENTS[int(number)])
    values = np.asarray(positions, dtype=np.float64)
    if values.shape != (len(symbols), 3) or not np.isfinite(values).all():
        raise ValueError("XYZ coordinates must be finite [N,3]")
    lines = [str(len(symbols)), comment]
    lines.extend(
        f"{symbol:<2s} {x: .10f} {y: .10f} {z: .10f}"
        for symbol, (x, y, z) in zip(symbols, values, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0])
    rows = [line.split() for line in lines[2:]]
    if len(rows) != count:
        raise ValueError(f"XYZ atom count mismatch: {path}")
    return [row[0] for row in rows], np.asarray(
        [[float(value) for value in row[1:4]] for row in rows], dtype=np.float64
    )


def _plot_xyz_examples(examples: list[dict[str, Any]]) -> list[Path]:
    selected = [
        next(item for item in examples if item["role"] == role and item["target_pg"] == target)
        for role, target in (
            ("failure_collision_after_hard_projection", "C2"),
            ("success_same_graph_after_f02", "C2"),
            ("failure_c2_c3_collapse", "C2"),
            ("failure_c2_c3_collapse", "C3"),
            ("success_multi_pg_response", "C2"),
            ("success_multi_pg_response", "C3"),
        )
    ]
    dataset = COFGraphDataset(ROOT / "generative_model/data/processed/v2")
    loaded = [_read_xyz(ROOT / item["path"]) for item in selected]
    minimum_distances = []
    for _, positions in loaded[:2]:
        distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=-1
        )[np.triu_indices(len(positions), k=1)]
        minimum_distances.append(float(distances.min()))
    collapse_rmsd = kabsch_rmsd(loaded[2][1], loaded[3][1])
    responsive_rmsd = kabsch_rmsd(loaded[4][1], loaded[5][1])
    titles = (
        f"Hard projection: collision (dmin={minimum_distances[0]:.3f} Å)",
        f"Same graph: F0.2 repaired (dmin={minimum_distances[1]:.3f} Å)",
        f"Collapsed response: request C2 (pair Δ={collapse_rmsd:.4f} Å)",
        f"Collapsed response: request C3 (pair Δ={collapse_rmsd:.4f} Å)",
        f"Responsive graph: request C2 (pair Δ={responsive_rmsd:.3f} Å)",
        f"Responsive graph: request C3 (pair Δ={responsive_rmsd:.3f} Å)",
    )
    figure = plt.figure(figsize=(14, 8.8), layout="constrained")
    for panel_id, (item, title, loaded_item) in enumerate(
        zip(selected, titles, loaded, strict=True), start=1
    ):
        axis = figure.add_subplot(2, 3, panel_id, projection="3d")
        symbols, positions = loaded_item
        sample = dataset[int(item["package_index"])]
        for left, right in np.asarray(sample["bond_index"], dtype=np.int64).T:
            segment = positions[[int(left), int(right)]]
            axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#888888", lw=0.8)
        for symbol in sorted(set(symbols)):
            mask = np.asarray([value == symbol for value in symbols])
            axis.scatter(
                positions[mask, 0], positions[mask, 1], positions[mask, 2],
                s=28 if symbol != "H" else 14,
                color=ELEMENT_COLORS[symbol], edgecolor="#333333", linewidth=0.3,
                depthshade=True,
            )
        center = (positions.max(axis=0) + positions.min(axis=0)) / 2
        radius = max(float(np.ptp(positions, axis=0).max()) / 2, 1.0)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_axis_off()
        axis.view_init(elev=22, azim=35)
        axis.set_title(f"{title}\n{item['molecule_id']}", fontsize=10)
    figure.suptitle("Representative success and failure XYZ outputs", fontsize=14)
    return _save_figure(figure, OUTPUT / "xyz_examples_overview")


def _export_xyz_examples(
    report: dict[str, Any], arrays: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    output = OUTPUT / "xyz_examples"
    output.mkdir(parents=True, exist_ok=True)
    offsets = arrays["atom_offsets"]
    hard_failure = next(
        index for index, record in enumerate(report["records"])
        if record["routes"]["etflow_hard_projection"].get("success")
        and not record["routes"]["etflow_hard_projection"].get("collision_free_at_0p6")
        and record["routes"]["etflow_hard_projection_f02"].get("collision_free_at_0p6")
    )
    record = report["records"][hard_failure]
    start, end = map(int, offsets[hard_failure : hard_failure + 2])
    examples = []
    for route, role in (
        ("etflow_hard_projection", "failure_collision_after_hard_projection"),
        ("etflow_hard_projection_f02", "success_same_graph_after_f02"),
    ):
        path = output / f"{record['molecule_id']}_{route}.xyz"
        _write_xyz(
            path, arrays["atomic_numbers"][start:end], arrays[f"positions_{route}"][start:end],
            f"role={role}; package_index={record['package_index']}; target_pg={record['target_pg']}",
        )
        examples.append({
            "role": role,
            "package_index": int(record["package_index"]),
            "molecule_id": str(record["molecule_id"]),
            "target_pg": str(record["target_pg"]),
            "route": route,
            "path": path.relative_to(ROOT).as_posix(),
        })

    control_run = RUNS["controllability"]
    control_report = json.loads((control_run / "report.json").read_text(encoding="utf-8"))
    with np.load(control_run / "coordinates.npz", allow_pickle=False) as archive:
        control = {name: archive[name] for name in archive.files}
    for package_index, role_prefix in ((2529, "failure_c2_c3_collapse"), (861, "success_multi_pg_response")):
        local_id = next(
            index for index, item in enumerate(control_report["records"])
            if int(item["package_index"]) == package_index
        )
        item = control_report["records"][local_id]
        start, end = map(int, control["atom_offsets"][local_id : local_id + 2])
        for target in ("C2", "C3", "D6h"):
            path = output / f"{item['molecule_id']}_{target}.xyz"
            _write_xyz(
                path, control["atomic_numbers"][start:end],
                control[f"final_positions_{target}"][start:end],
                f"role={role_prefix}; package_index={package_index}; requested_pg={target}",
            )
            examples.append({
                "role": role_prefix,
                "package_index": package_index,
                "molecule_id": str(item["molecule_id"]),
                "target_pg": target,
                "route": "our_etflow_v5",
                "path": path.relative_to(ROOT).as_posix(),
            })
    return examples


def _export_xtb_examples(
    report: dict[str, Any], point_group: dict[str, Any], arrays: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    output = OUTPUT / "xyz_examples"
    failure_id = next(
        index for index, record in enumerate(point_group["records"])
        if not record["after"]["pg_compatible"]
    )
    stable_id = next(
        index for index, record in enumerate(point_group["records"])
        if record["target_pg"] == "D6h" and record["after"]["pg_compatible"]
    )
    examples = []
    for local_id, prefix in ((failure_id, "xtb_symmetry_breaking"), (stable_id, "xtb_symmetry_retained")):
        record = report["records"][local_id]
        audit = point_group["records"][local_id]
        start, end = map(int, arrays["atom_offsets"][local_id : local_id + 2])
        for stage, key in (("before", "input_positions"), ("after", "relaxed_positions")):
            path = output / f"{record['molecule_id']}_xtb_{stage}.xyz"
            _write_xyz(
                path, arrays["atomic_numbers"][start:end], arrays[key][start:end],
                (
                    f"role={prefix}; stage={stage}; package_index={record['package_index']}; "
                    f"target_pg={record['target_pg']}; actual_pg={audit[stage]['actual_pg']}"
                ),
            )
            examples.append({
                "role": prefix,
                "stage": stage,
                "package_index": int(record["package_index"]),
                "molecule_id": str(record["molecule_id"]),
                "target_pg": str(record["target_pg"]),
                "actual_pg": str(audit[stage]["actual_pg"]),
                "route": "our_etflow_v5_then_unconstrained_gfn2_xtb",
                "path": path.relative_to(ROOT).as_posix(),
            })
    return examples


def _plot_xtb_xyz_examples(examples: list[dict[str, Any]]) -> list[Path]:
    ordered = [
        next(item for item in examples if item["role"] == role and item["stage"] == stage)
        for role, stage in (
            ("xtb_symmetry_breaking", "before"),
            ("xtb_symmetry_breaking", "after"),
            ("xtb_symmetry_retained", "before"),
            ("xtb_symmetry_retained", "after"),
        )
    ]
    dataset = COFGraphDataset(ROOT / "generative_model/data/processed/v2")
    titles = (
        "Breaking case: before (D6h)", "Breaking case: after (D3h)",
        "Retained case: before (D6h)", "Retained case: after (D6h)",
    )
    figure = plt.figure(figsize=(15, 4.8), layout="constrained")
    for panel_id, (item, title) in enumerate(zip(ordered, titles, strict=True), start=1):
        axis = figure.add_subplot(1, 4, panel_id, projection="3d")
        symbols, positions = _read_xyz(ROOT / item["path"])
        sample = dataset[int(item["package_index"])]
        for left, right in np.asarray(sample["bond_index"], dtype=np.int64).T:
            segment = positions[[int(left), int(right)]]
            axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#888888", lw=0.7)
        for symbol in sorted(set(symbols)):
            mask = np.asarray([value == symbol for value in symbols])
            axis.scatter(
                positions[mask, 0], positions[mask, 1], positions[mask, 2],
                s=25 if symbol != "H" else 12, color=ELEMENT_COLORS[symbol],
                edgecolor="#333333", linewidth=0.25,
            )
        center = (positions.max(axis=0) + positions.min(axis=0)) / 2
        radius = max(float(np.ptp(positions, axis=0).max()) / 2, 1.0)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_axis_off()
        axis.view_init(elev=22, azim=35)
        axis.set_title(f"{title}\n{item['molecule_id']}", fontsize=10)
    figure.suptitle("GFN2-xTB relaxation: symmetry-breaking and retained D6h examples", fontsize=13)
    return _save_figure(figure, OUTPUT / "xtb_xyz_before_after")


def build() -> dict[str, Any]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "our-etflow-eval-v1",
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loaded = {name: _load(path) for name, path in RUNS.items() if name != "controllability"}
    report, point_group, arrays = loaded["paired_ablation"]
    generated = []
    generated.extend(_plot_ablation_overview(report, point_group))
    generated.extend(_plot_pg_success(report, point_group))
    generated.extend(_plot_energy_symmetry(report, point_group))
    generated.extend(_plot_runtime_success(report, point_group))
    xyz_examples = _export_xyz_examples(report, arrays)
    generated.extend(_plot_xyz_examples(xyz_examples))

    xtb_report, xtb_point_group, xtb_arrays = _load_xtb()
    generated.extend(_plot_xtb_overview(xtb_report, xtb_point_group))
    generated.extend(_plot_xtb_energy_symmetry(xtb_report, xtb_point_group))
    xtb_examples = _export_xtb_examples(xtb_report, xtb_point_group, xtb_arrays)
    xyz_examples.extend(xtb_examples)
    generated.extend(_plot_xtb_xyz_examples(xtb_examples))
    generated.extend(ROOT / item["path"] for item in xyz_examples)

    table_path = OUTPUT / "evaluation_summary.csv"
    fieldnames = [
        "panel", "route", "panel_count", "generation_fraction", "pg_compatible_fraction",
        "collision_free_fraction", "joint_fraction", "mean_bond_mae_angstrom",
        "median_uff_energy_per_atom_kcal_mol", "median_runtime_seconds",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for panel, (panel_report, panel_pg, _) in loaded.items():
            for route in ROUTES:
                writer.writerow({
                    "panel": panel,
                    "route": route,
                    **_route_summary(panel_report, panel_pg, route),
                })
    generated.append(table_path)
    xtb_table_path = OUTPUT / "xtb_relaxation_records.csv"
    xtb_fields = [
        "package_index", "molecule_id", "target_pg", "atom_count",
        "optimizer_converged", "optimizer_steps", "runtime_seconds",
        "energy_before_ev", "energy_after_ev", "energy_change_ev",
        "energy_change_ev_per_atom", "maximum_force_before_ev_per_angstrom",
        "maximum_force_after_ev_per_angstrom", "kabsch_rmsd_pre_to_post_angstrom",
        "bond_length_mae_pre_to_post_angstrom", "minimum_pair_distance_before_angstrom",
        "minimum_pair_distance_after_angstrom", "actual_pg_before", "actual_pg_after",
        "pg_compatible_after", "pg_exact_match_after",
    ]
    with xtb_table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=xtb_fields, lineterminator="\n")
        writer.writeheader()
        for record, audit in zip(
            xtb_report["records"], xtb_point_group["records"], strict=True
        ):
            writer.writerow({
                **{name: record[name] for name in xtb_fields if name in record},
                "actual_pg_before": audit["before"]["actual_pg"],
                "actual_pg_after": audit["after"]["actual_pg"],
                "pg_compatible_after": audit["after"]["pg_compatible"],
                "pg_exact_match_after": audit["after"]["pg_exact_match"],
            })
    generated.append(xtb_table_path)
    manifest_path = OUTPUT / "figure_manifest.json"
    input_files = []
    for name, run in RUNS.items():
        for filename in (
            "report.json", "coordinates.npz",
            "controllability_audit_v2.json" if name == "controllability" else "point_group_report.json",
        ):
            path = run / filename
            input_files.append({
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            })
    for filename in (
        "relaxation_report.json", "relaxed_coordinates.npz",
        "point_group_report.json", "summary.json",
    ):
        path = XTB_RUN / filename
        input_files.append({
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(path),
        })
    manifest = {
        "schema_version": "our-etflow-presentation-assets-v1",
        "status": "PASS_PRESENTATION_ASSETS_BUILT_FROM_FROZEN_RESULTS",
        "input_files": input_files,
        "xyz_examples": xyz_examples,
        "artifacts": [],
        "builder_source_sha256": _sha256(Path(__file__)),
    }
    for path in sorted(generated, key=lambda item: item.as_posix()):
        manifest["artifacts"].append({
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(path),
        })
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    result = build()
    print(json.dumps({
        "status": result["status"],
        "artifact_count": len(result["artifacts"]),
        "xyz_example_count": len(result["xyz_examples"]),
        "output": str(OUTPUT),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
