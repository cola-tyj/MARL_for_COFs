#!/usr/bin/env python
"""
COF Core visualization utilities.

Supports:
  - py3Dmol (interactive Jupyter 3D view) ← BEST
  - matplotlib 3D scatter (PNG)
  - XYZ export (VESTA/Avogadro/ASE)

Usage:
    from symmcd_diffusion.utils.visualize_core import view_core
    view_core(atom_idx, positions)  # opens interactive 3D in Jupyter
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Atom styling ──────────────────────────────────────────────────────────
ATOM_COLORS = {
    0:  "#FFFFFF",  # H - white
    1:  "#404040",  # C - dark grey
    2:  "#3050F0",  # N - blue
    3:  "#FF3030",  # O - red
    4:  "#90E050",  # F - light green
    5:  "#E0E050",  # S - yellow
    6:  "#FFB0B0",  # B - pink
    7:  "#C08040",  # Cu - bronze
    8:  "#8080B0",  # Zn - grey-blue
    9:  "#F08030",  # Co - orange
    10: "#FF0000",  # Q - bright red (connector point)
    11: "#0080FF",  # R - bright blue (FG attachment)
}
ATOM_SYMBOLS = ["H","C","N","O","F","S","B","Cu","Zn","Co","Q","R"]
ATOM_RADII = [0.25, 0.70, 0.65, 0.60, 0.50, 1.00, 0.85, 1.30, 1.25, 1.25, 0.70, 0.70]


def core_to_xyz(atom_idx: torch.Tensor, positions: torch.Tensor, filepath: str):
    """Export Core to XYZ format (viewable in VESTA/Avogadro/ASE)."""
    n = atom_idx.size(0)
    lines = [f"{n}", f"COF Core — {n} atoms"]
    for i in range(n):
        sym = ATOM_SYMBOLS[atom_idx[i].item()]
        x, y, z = positions[i].tolist()
        lines.append(f"{sym:3s} {x:10.6f} {y:10.6f} {z:10.6f}")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved XYZ: {filepath}")


def core_to_png(
    atom_idx: torch.Tensor,
    positions: torch.Tensor,
    filepath: str = "core.png",
    title: str = "COF Core",
    figsize: tuple = (10, 8),
):
    """Render Core as 3D scatter plot (PNG via matplotlib)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    pos = positions.cpu().numpy()
    idx = atom_idx.cpu().numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for atom_type in range(len(ATOM_SYMBOLS)):
        mask = idx == atom_type
        if not mask.any():
            continue
        ax.scatter(
            pos[mask, 0], pos[mask, 1], pos[mask, 2],
            c=ATOM_COLORS.get(atom_type, "#888888"),
            s=[ATOM_RADII[atom_type] * 300] * mask.sum(),
            label=ATOM_SYMBOLS[atom_type],
            edgecolors="black",
            linewidth=0.5,
            alpha=0.9,
        )

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved PNG: {filepath}")


def visualize_core(data, output_prefix: str = "core", output_dir: str = None):
    """Visualize a Core (PyG Data object). Saves to generated/visualizations/ by default."""
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "generated" / "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    atom_idx = data.x.argmax(dim=-1)
    xyz_path = os.path.join(output_dir, f"{output_prefix}.xyz")
    png_path = os.path.join(output_dir, f"{output_prefix}.png")
    core_to_xyz(atom_idx, data.positions, xyz_path)
    core_to_png(atom_idx, data.positions, png_path,
                title=f"Core: {data.symmetry} | Q={data.q_count} R={data.r_count}")


# ── py3Dmol interactive 3D ──────────────────────────────────────────────────

def view_core(atom_idx, positions, width=600, height=500):
    """
    Interactive 3D visualization using py3Dmol.

    Args:
        atom_idx: (N,) tensor of atom type indices
        positions: (N, 3) tensor of coordinates
        width, height: viewport dimensions

    Returns:
        py3Dmol view object (auto-displays in Jupyter)
    """
    import py3Dmol

    pos = positions.cpu().numpy() if isinstance(positions, torch.Tensor) else np.array(positions)
    idx = atom_idx.cpu().numpy() if isinstance(atom_idx, torch.Tensor) else np.array(atom_idx)

    # Build XYZ string
    n = len(idx)
    xyz_str = f"{n}\nCOF Core\n"
    for i in range(n):
        sym = ATOM_SYMBOLS[idx[i]]
        x, y, z = pos[i]
        xyz_str += f"{sym:3s} {x:10.6f} {y:10.6f} {z:10.6f}\n"

    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_str, "xyz")

    # Style: ball-and-stick with atom-specific colors
    view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.3}})

    # Color Q and R atoms specially
    view.addModel(xyz_str, "xyz")
    view.setStyle({"elem": "Q"}, {"sphere": {"color": "red", "scale": 0.7}})
    view.setStyle({"elem": "R"}, {"sphere": {"color": "blue", "scale": 0.7}})

    view.zoomTo()
    return view


def view_core_from_data(data, width=600, height=500):
    """Interactive 3D view from a PyG Data object."""
    atom_idx = data.x.argmax(dim=-1)
    return view_core(atom_idx, data.positions, width, height)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from symmcd_diffusion.data.core_dataset import CoreDataset

    ds = CoreDataset(augment=False)
    print(f"Dataset: {len(ds)} cores")

    # Visualize one core of each symmetry
    seen = set()
    for i, d in enumerate(ds):
        sym = d.symmetry
        if sym in seen:
            continue
        seen.add(sym)
        visualize_core(d, f"core_{sym}")
        if len(seen) >= 6:
            break

    print("\nDone! View .png files or open .xyz in VESTA/Avogadro.")
