"""Render deterministic 2D graph / generated 3D comparison images for Cn."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
from textwrap import shorten
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from generative_model.data.cn_graph_dataset import CnGraphDataset
from generative_model.inference.generate_etflow_symmetric_xyz import ROOT, _atomic_json


DEFAULT_INPUT = ROOT / "generative_model/data/Cn"
DEFAULT_RESULTS = ROOT / "generative_model/results/Cn"
COLORS = {
    1: "#d9d9d9", 6: "#333333", 7: "#3050f8", 8: "#ff0d0d", 9: "#90e050",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _oriented(positions: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    centered = values - values.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    result = centered @ vectors[:, ::-1]
    for axis in range(3):
        index = int(np.argmax(np.abs(result[:, axis])))
        if result[index, axis] < 0:
            result[:, axis] *= -1
    return result


def _render_one(
    sample: dict[str, Any], positions: np.ndarray, validity: dict[str, str],
    point_group: dict[str, Any], output: Path,
) -> None:
    molecule = Chem.MolFromSmiles(sample["metadata"]["SMILES"])
    if molecule is None:
        raise ValueError(f"SMILES 无法绘图: {sample['molecule_id']}")
    AllChem.Compute2DCoords(molecule, canonOrient=True)
    image_2d = Draw.MolToImage(
        molecule, size=(620, 520), kekulize=True,
        legend="2D input graph (implicit H)",
    )
    coords = _oriented(positions)
    numbers = np.asarray(sample["atomic_numbers"], dtype=np.int64)
    fig = plt.figure(figsize=(12, 6), dpi=140)
    left = fig.add_subplot(1, 2, 1)
    left.imshow(image_2d)
    left.axis("off")
    right = fig.add_subplot(1, 2, 2, projection="3d")
    for begin, end in sample["bond_index"].T:
        pair = coords[[int(begin), int(end)]]
        right.plot(pair[:, 0], pair[:, 1], pair[:, 2], color="#777777", lw=2.2)
    for number in sorted(set(map(int, numbers))):
        mask = numbers == number
        size = 30 if number == 1 else 105
        right.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            s=size, color=COLORS[number], edgecolor="#111111", linewidth=0.45,
            depthshade=True, label=Chem.GetPeriodicTable().GetElementSymbol(number),
        )
    for index, (number, position) in enumerate(zip(numbers, coords, strict=True)):
        if int(number) != 1:
            symbol = Chem.GetPeriodicTable().GetElementSymbol(int(number))
            right.text(*position, f"{symbol}{index}", fontsize=6.5, color="#111111")
    span = max(float(np.ptp(coords[:, axis])) for axis in range(3))
    span = max(span, 1.0) * 0.62
    for setter in (right.set_xlim, right.set_ylim, right.set_zlim):
        setter(-span, span)
    right.set_box_aspect((1, 1, 1))
    right.view_init(elev=20, azim=35)
    right.set_axis_off()
    right.set_title("our_ET_Flow final 3D (explicit H)", fontsize=11)
    right.legend(loc="upper right", fontsize=7, frameon=False)
    passed = validity["strict_screen_pass"].lower() == "true"
    title = (
        f"{sample['molecule_id']}  |  requested C3 → actual {point_group['actual_pg']}  |  "
        f"screen={'PASS' if passed else 'WARNING'}\n"
        f"{shorten(sample['metadata']['SMILES'], width=100, placeholder='…')}  |  "
        f"dmin={float(validity['minimum_pair_distance_angstrom']):.3f} Å  |  "
        f"UFF ΔRMSD={float(validity['uff_relaxation_kabsch_rmsd_angstrom']):.3f} Å"
    )
    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _contact_sheets(images: list[Path], output_dir: Path, group: str) -> list[Path]:
    results = []
    per_page, columns = 20, 4
    thumb_size = (480, 240)
    for page_index in range(0, len(images), per_page):
        selected = images[page_index : page_index + per_page]
        rows = (len(selected) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), "white")
        for offset, path in enumerate(selected):
            with Image.open(path) as source:
                item = source.convert("RGB")
                item.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                x = (offset % columns) * thumb_size[0] + (thumb_size[0] - item.width) // 2
                y = (offset // columns) * thumb_size[1] + (thumb_size[1] - item.height) // 2
                canvas.paste(item, (x, y))
        output = output_dir / f"{group}_page_{page_index // per_page + 1:02d}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, optimize=True)
        results.append(output)
    return results


def render_batch(input_root: Path, results_root: Path) -> dict[str, Any]:
    validity_path = results_root / "validity_report.json"
    if not validity_path.is_file():
        raise FileNotFoundError("请先运行 Cn validity audit")
    validity_report = json.loads(validity_path.read_text(encoding="utf-8"))
    validity_csv = results_root / "validity_records.csv"
    with validity_csv.open(encoding="utf-8", newline="") as handle:
        validity = {row["molecule_id"]: row for row in csv.DictReader(handle)}
    dataset = CnGraphDataset(input_root)
    visual_root = results_root / "visualization"
    images: list[Path] = []
    image_records = []
    for sample in dataset.records:
        group = sample["metadata"]["Source_Group"]
        record_dir = results_root / group / "records" / sample["molecule_id"]
        with np.load(record_dir / "coordinates.npz", allow_pickle=False) as archive:
            positions = np.asarray(archive["final_positions"], dtype=np.float64)
        point_group = json.loads(
            (record_dir / "point_group_report.json").read_text(encoding="utf-8")
        )
        output = visual_root / group / f"{sample['molecule_id']}.png"
        _render_one(sample, positions, validity[sample["molecule_id"]], point_group, output)
        images.append(output)
        image_records.append({
            "molecule_id": sample["molecule_id"],
            "source_group": group,
            "smiles": sample["metadata"]["SMILES"],
            "actual_pg": point_group["actual_pg"],
            "strict_screen_pass": validity[sample["molecule_id"]]["strict_screen_pass"],
            "image": str(output.relative_to(results_root)),
            "sha256": _sha256(output),
        })
        print(json.dumps({
            "completed": len(images), "total": len(dataset),
            "molecule_id": sample["molecule_id"],
        }))
    contact_paths = []
    for group in ("C3k2", "C3k3"):
        contact_paths.extend(_contact_sheets(
            [path for path in images if path.parent.name == group],
            visual_root / "contact_sheets", group,
        ))
    cards = "\n".join(
        f'<figure><a href="{html.escape(record["image"].replace("visualization/", ""))}">'
        f'<img loading="lazy" src="{html.escape(record["image"].replace("visualization/", ""))}"></a>'
        f'<figcaption>{html.escape(record["molecule_id"])} · {html.escape(record["actual_pg"])} · '
        f'{"PASS" if record["strict_screen_pass"].lower() == "true" else "WARNING"}</figcaption></figure>'
        for record in image_records
    )
    html_path = visual_root / "index.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Cn 2D/3D visualization</title>"
        "<style>body{font-family:sans-serif;margin:20px}main{display:grid;grid-template-columns:"
        "repeat(auto-fill,minmax(360px,1fr));gap:14px}figure{margin:0;border:1px solid #ddd;"
        "padding:8px}img{width:100%;height:auto}figcaption{font-size:13px}</style>"
        "<h1>Cn: 2D graph / our_ET_Flow final 3D</h1><main>" + cards + "</main>\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "our-etflow-cn-visualization-v1",
        "image_count": len(images),
        "screen_pass_count": validity_report["strict_screen_pass_count"],
        "screen_warning_count": (
            validity_report["molecule_count"]
            - validity_report["strict_screen_pass_count"]
        ),
        "contact_sheet_count": len(contact_paths),
        "records": image_records,
        "contact_sheets": {
            str(path.relative_to(results_root)): _sha256(path) for path in contact_paths
        },
        "identity": {
            "validity_report_sha256": _sha256(validity_path),
            "validity_records_sha256": _sha256(validity_csv),
            "renderer_sha256": _sha256(Path(__file__)),
            "html_sha256": _sha256(html_path),
        },
    }
    _atomic_json(visual_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    manifest = render_batch(args.input_root.resolve(), args.results_root.resolve())
    print(json.dumps({
        "status": "PASS_OUR_ETFLOW_CN_VISUALIZATION",
        "image_count": manifest["image_count"],
        "contact_sheet_count": manifest["contact_sheet_count"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
