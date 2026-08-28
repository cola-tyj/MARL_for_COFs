#!/usr/bin/env python3
"""将单个 XYZ 分子导出为可交互的 HTML 三维视图。

示例：
    python cof_symmetry_pipeline/visualization/visualize_xyz.py \
        cof_symmetry_pipeline/output/xyz/aug_000006.xyz

    python cof_symmetry_pipeline/visualization/visualize_xyz.py molecule.xyz \
        --output molecule.html --style stick --labels --open
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import webbrowser


STYLE_CONFIGS = {
    "ball-and-stick": {
        "stick": {"radius": 0.12},
        "sphere": {"scale": 0.28},
    },
    "stick": {"stick": {"radius": 0.14}},
    "sphere": {"sphere": {"scale": 0.75}},
    "line": {"line": {"linewidth": 1.5}},
}


def parse_xyz(path: Path) -> tuple[str, list[tuple[str, float, float, float]]]:
    """读取并校验标准 XYZ 文件，返回原文和原子坐标。"""
    if not path.is_file():
        raise ValueError(f"XYZ 文件不存在: {path}")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ 文件至少需要原子数行和注释行")

    try:
        expected_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("XYZ 第一行必须是原子数") from exc
    if expected_atoms <= 0:
        raise ValueError("XYZ 原子数必须大于 0")

    atom_lines = lines[2:2 + expected_atoms]
    if len(atom_lines) != expected_atoms:
        raise ValueError(
            f"XYZ 声明了 {expected_atoms} 个原子，但只有 {len(atom_lines)} 行坐标"
        )

    atoms = []
    for line_number, line in enumerate(atom_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"XYZ 第 {line_number} 行缺少元素或三维坐标")
        try:
            x, y, z = (float(value) for value in fields[1:4])
        except ValueError as exc:
            raise ValueError(f"XYZ 第 {line_number} 行包含无效坐标") from exc
        atoms.append((fields[0], x, y, z))

    return text, atoms


def build_viewer(
    xyz_text: str,
    atoms: list[tuple[str, float, float, float]],
    style: str,
    background: str,
    labels: bool,
    spin: bool,
):
    """创建 py3Dmol 视图；XYZ 中未记录的化学键由 3Dmol.js 按距离推断。"""
    try:
        import py3Dmol
    except ImportError as exc:
        raise RuntimeError(
            "缺少 py3Dmol，请运行: pip install py3Dmol"
        ) from exc

    viewer = py3Dmol.view(width=1000, height=750)
    viewer.addModel(xyz_text, "xyz")
    viewer.setStyle({}, STYLE_CONFIGS[style])
    viewer.setBackgroundColor(background)

    if labels:
        for symbol, x, y, z in atoms:
            if symbol == "H":
                continue
            viewer.addLabel(
                symbol,
                {
                    "position": {"x": x, "y": y, "z": z},
                    "fontSize": 10,
                    "fontColor": "black",
                    "backgroundColor": "white",
                    "backgroundOpacity": 0.65,
                    "borderThickness": 0,
                },
            )

    viewer.zoomTo()
    if spin:
        viewer.spin("y", 0.7)
    return viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 XYZ 分子导出为浏览器可交互的三维 HTML。"
    )
    parser.add_argument("xyz", type=Path, help="输入 XYZ 文件")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="输出 HTML；默认保存到本脚本目录下的 <XYZ文件名>.html",
    )
    parser.add_argument(
        "--style",
        choices=tuple(STYLE_CONFIGS),
        default="ball-and-stick",
        help="显示样式（默认: ball-and-stick）",
    )
    parser.add_argument(
        "--background",
        default="white",
        help="背景颜色名称或十六进制颜色（默认: white）",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="显示非氢原子的元素标签",
    )
    parser.add_argument(
        "--spin",
        action="store_true",
        help="打开页面后自动绕 y 轴旋转",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="生成后使用默认浏览器打开 HTML",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xyz_path = args.xyz.expanduser().resolve()
    output_path = args.output
    if output_path is None:
        output_path = Path(__file__).resolve().parent / f"{xyz_path.stem}.html"
    else:
        output_path = output_path.expanduser().resolve()
    if output_path.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("输出文件扩展名必须是 .html 或 .htm")

    xyz_text, atoms = parse_xyz(xyz_path)
    viewer = build_viewer(
        xyz_text=xyz_text,
        atoms=atoms,
        style=args.style,
        background=args.background,
        labels=args.labels,
        spin=args.spin,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    viewer.write_html(str(output_path), fullpage=True)

    composition = Counter(symbol for symbol, *_ in atoms)
    formula = " ".join(f"{symbol}:{count}" for symbol, count in sorted(composition.items()))
    print(f"输入: {xyz_path}")
    print(f"原子: {len(atoms)} ({formula})")
    print(f"输出: {output_path}")

    if args.open_browser:
        opened = webbrowser.open(output_path.as_uri())
        if not opened:
            print("未能自动打开浏览器，请手动打开输出 HTML。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
