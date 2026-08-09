#!/usr/bin/env python3
"""
对 2D 数据集中的分子跑 3D 对称性验证（Stage 2+3 精简版）。

输入: 包含 SMILES 列的 CSV 文件
输出:
  - <output>_screening.csv   完整筛查结果（含 PASS / FAIL，供调试）
  - <output>_pass.csv        仅 PASS 分子（供后续使用）

用法:
  python mining/validate_augmented.py                                    # 默认处理 augmented_dataset.csv
  python mining/validate_augmented.py --input output/xxx.csv --workers 8
"""

import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import EmbeddingConfig, RelaxConfig, SymmetryConfig
from relaxer import SymmetryScreener, write_xyz

# ── 子进程全局变量 ──────────────────────────────────────────────────────
_worker_screener = None
_worker_xyz_dir = ""


def _worker_init(xyz_dir: str):
    global _worker_screener, _worker_xyz_dir
    _worker_screener = SymmetryScreener(
        embed_config=EmbeddingConfig(num_conformer_attempts=4),
        sym_config=SymmetryConfig(),
        relax_config=RelaxConfig(backend="rdkit_mmff", max_steps=40, force_threshold=5.0),
    )
    _worker_xyz_dir = xyz_dir


def _screen_one(args):
    global _worker_screener, _worker_xyz_dir
    idx, smi, target_pg, core, arm = args

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {
            "idx": idx, "smiles": smi, "target_pg": target_pg,
            "core": core, "arm": arm,
            "status": "FAIL_INVALID_SMILES",
            "initial_pg": None, "final_pg": None,
            "symmetry_preserved": False,
            "energy": float("nan"), "max_force": float("inf"), "xyz_path": "",
            "error": "无法从 SMILES 重建分子",
        }

    mol.SetProp("target_pg", target_pg)
    mol.SetProp("core_name", core)
    mol.SetProp("arm_name", arm)

    try:
        result = _worker_screener.screen(mol)
    except Exception as exc:
        return {
            "idx": idx, "smiles": smi, "target_pg": target_pg,
            "core": core, "arm": arm,
            "status": f"FAIL_EXCEPTION",
            "initial_pg": None, "final_pg": None,
            "symmetry_preserved": False,
            "energy": float("nan"), "max_force": float("inf"), "xyz_path": "",
            "error": f"{type(exc).__name__}: {exc}",
        }

    # 写 XYZ（仅 PASS）
    xyz_path = ""
    if result["status"] == "PASS" and result.get("mol") is not None:
        os.makedirs(_worker_xyz_dir, exist_ok=True)
        xyz_path = os.path.join(_worker_xyz_dir, f"aug_{idx:06d}.xyz")
        try:
            write_xyz(
                result["mol"], xyz_path,
                comment=(
                    f"SMILES={result['smiles']} PG={result['final_pg']} "
                    f"E={result['energy']:.4f}kcal"
                ),
            )
        except Exception:
            xyz_path = ""

    return {
        "idx": idx,
        "smiles": smi,
        "target_pg": target_pg,
        "core": core,
        "arm": arm,
        "status": result["status"],
        "initial_pg": result.get("initial_pg"),
        "final_pg": result.get("final_pg"),
        "symmetry_preserved": result.get("symmetry_preserved", False),
        "energy": result.get("energy", float("nan")),
        "max_force": result.get("max_force", float("inf")),
        "xyz_path": xyz_path,
        "error": result.get("error", ""),
    }


# ── 主函数 ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="对 2D 分子数据集跑 3D 对称性验证"
    )
    parser.add_argument(
        "--input", type=str,
        default="cof_symmetry_pipeline/output/augmented_dataset.csv",
        help="输入 CSV 路径（需含 SMILES, Target_PG, Core, Arm 列）",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出文件前缀（默认与输入同名 + _3d）",
    )
    parser.add_argument(
        "--xyz-dir", type=str,
        default="cof_symmetry_pipeline/output/xyz",
        help="XYZ 坐标输出目录",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="并行进程数 (default: 4)",
    )
    parser.add_argument(
        "--batch", type=int, default=200,
        help="进度报告批次大小 (default: 200)",
    )
    args = parser.parse_args()

    # ── 加载输入 ──
    df = pd.read_csv(args.input)
    # 兼容不同列名
    col_smiles = "SMILES" if "SMILES" in df.columns else "smiles"
    col_pg = "Target_PG" if "Target_PG" in df.columns else "target_pg"
    col_core = "Core" if "Core" in df.columns else "core"
    col_arm = "Arm" if "Arm" in df.columns else "arm"

    smiles_list = df[col_smiles].tolist()
    target_pgs = df[col_pg].tolist() if col_pg in df.columns else ["C1"] * len(df)
    cores = df[col_core].tolist() if col_core in df.columns else ["?"] * len(df)
    arms = df[col_arm].tolist() if col_arm in df.columns else ["?"] * len(df)

    print(f"输入: {args.input}")
    print(f"待验证: {len(smiles_list)} 分子")
    print(f"并行: {args.workers} workers")
    print()

    # ── 确定输出路径 ──
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        out_full = f"{base}_3d_screening.csv"       # 全量筛查结果（含 FAIL）
        out_pass = f"{base}_3d_pass.csv"            # 仅 PASS
    else:
        out_full = f"{args.output}_screening.csv"
        out_pass = f"{args.output}_pass.csv"

    # ── 并行筛查 ──
    tasks = [
        (i, s, t, c, a)
        for i, (s, t, c, a) in enumerate(zip(smiles_list, target_pgs, cores, arms))
    ]
    results = [None] * len(tasks)
    n_tasks = len(tasks)

    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(args.xyz_dir,),
    ) as ex:
        futures = {}
        for task in tasks:
            futures[ex.submit(_screen_one, task)] = task[0]

            if len(futures) >= args.batch:
                for f in as_completed(futures):
                    r = f.result()
                    results[r["idx"]] = r
                futures = {}
                n_done = sum(1 for r in results if r is not None)
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 0.1)
                eta = (n_tasks - n_done) / max(rate, 0.01)
                print(
                    f"  [{n_done}/{n_tasks}] {rate:.1f} mol/s  "
                    f"ETA: {eta:.0f}s"
                )

        # 收尾
        for f in as_completed(futures):
            r = f.result()
            results[r["idx"]] = r

    elapsed = time.time() - t0
    print(f"\n完成: {elapsed:.0f}s ({n_tasks / elapsed:.1f} mol/s)")

    # ── 统计 ──
    status_counts = Counter(r["status"] for r in results)
    n_pass = status_counts.get("PASS", 0)

    print(f"\n=== 筛查结果 ===")
    print(f"  PASS:                     {n_pass:5d} ({100 * n_pass / n_tasks:.1f}%)")
    for status, count in sorted(status_counts.items()):
        if status != "PASS":
            print(f"  {status:25s}: {count:5d} ({100 * count / n_tasks:.1f}%)")

    pg_counts = Counter(
        r["final_pg"] for r in results
        if r.get("final_pg") and r["status"] == "PASS"
    )
    if pg_counts:
        print(f"\nPASS 分子点群分布:")
        for pg, n in pg_counts.most_common():
            print(f"  {pg:6s}: {n:4d}")

    # ── 保存 ──
    result_df = pd.DataFrame(results)

    # 全量筛查结果（含 FAIL，供调试）
    result_df.to_csv(out_full, index=False)
    print(f"\n全量筛查结果: {out_full} ({len(result_df)} 行)")

    # 仅 PASS
    pass_df = result_df[result_df["status"] == "PASS"].copy()
    pass_df.to_csv(out_pass, index=False)
    print(f"PASS 分子:     {out_pass} ({len(pass_df)} 行)")

    if n_pass > 0:
        n_xyz = pass_df["xyz_path"].apply(
            lambda p: bool(p) and os.path.exists(p)
        ).sum() if "xyz_path" in pass_df.columns else 0
        print(f"XYZ 文件:      {n_xyz} 个")


if __name__ == "__main__":
    main()
