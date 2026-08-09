#!/usr/bin/env python3
"""
cof_symmetry_pipeline.main — 端到端 COF 对称中间体核心数据集构建管线。

完整流程:
  1. 组合生成 → 2D 分子图（100% 拓扑对称）
  2. ETKDGv3 嵌入 → 3D 构象 + 初始点群分析
  3. MLIP 几何优化 → 应力张力筛查
  4. 对称破缺检测 → 淘汰非对称结构
  5. 输出 CSV + XYZ + PyTorch Dataset

用法:
    python main.py                                    # 默认全组合，本地串行
    python main.py --cores triazine_C3 porphyrin_C4   # 指定核心
    python main.py --arms CHO_ph NH2_ph               # 指定臂
    python main.py --workers 8                        # 8 进程并行
    python main.py --backend mace --max-steps 100     # MACE + 自定义步数
    python main.py --dataset-demo                     # 生成后运行 PyTorch Dataset 演示
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# 确保项目路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CORE_TEMPLATES,
    ARM_LIBRARY,
    ATOMIC_NUMBER_TO_SYMBOL,
    EmbeddingConfig,
    PipelineConfig,
    RelaxConfig,
    SymmetryConfig,
)
from generator import MoleculeGenerator, generate_dataset
from relaxer import Embedder, GeometryRelaxer, SymmetryAnalyzer, SymmetryScreener, write_xyz

# =============================================================================
# 日志配置
# =============================================================================


def setup_logging(config: PipelineConfig) -> logging.Logger:
    """配置文件和终端的双重日志输出。"""
    os.makedirs(os.path.dirname(config.log_file) or ".", exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    # 文件 handler
    fh = logging.FileHandler(config.log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # 终端 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return logging.getLogger("cof_pipeline")


# =============================================================================
# 单分子处理 Worker（供多进程调用）
# =============================================================================
# 设计要点：
#   - 每个子进程持有独立的 SymmetryScreener（含独立力场实例）
#   - 避免跨进程共享 RDKit Mol（属性经 pickle 会丢失）
#   - 改为传递 SMILES 字符串 + 元数据，子进程内重建 Mol

# 子进程全局变量（由 _worker_initializer 设置）
_worker_screener: Optional[SymmetryScreener] = None
_worker_xyz_dir: str = ""


def _worker_initializer(relax_config_dict: dict, xyz_dir: str):
    """
    子进程初始化：每个子进程在启动时调用一次。
    创建独立的 SymmetryScreener（含独立的 MMFF/MACE 力场实例）。
    """
    global _worker_screener, _worker_xyz_dir
    relax_config = RelaxConfig(**relax_config_dict)
    _worker_screener = SymmetryScreener(
        embed_config=EmbeddingConfig(),
        sym_config=SymmetryConfig(),
        relax_config=relax_config,
    )
    _worker_xyz_dir = xyz_dir


def _process_one(args: Tuple) -> Dict:
    """
    单个分子的 Stage 2–3 处理（供多进程调用）。

    Args:
        args: (mol_index, smiles, target_pg, core_name, arm_name)
              ——传 SMILES 而非 RDKit Mol，因 Mol 属性经 pickle 丢失
    """
    global _worker_screener, _worker_xyz_dir
    idx, smiles, target_pg, core_name, arm_name = args

    # 从 SMILES 重建 RDKit Mol
    mol_2d = Chem.MolFromSmiles(smiles)
    if mol_2d is None:
        return {
            "mol_index": idx, "smiles": smiles,
            "target_pg": target_pg, "core_name": core_name, "arm_name": arm_name,
            "initial_pg": None, "final_pg": None,
            "symmetry_preserved": False,
            "energy": float("nan"), "energy_per_atom": float("nan"),
            "max_force": float("inf"), "xyz_path": None,
            "status": "FAIL_EXCEPTION",
            "error": "无法从 SMILES 重建分子",
        }
    mol_2d.SetProp("target_pg", target_pg)
    mol_2d.SetProp("core_name", core_name)
    mol_2d.SetProp("arm_name", arm_name)

    # —— 层级核心优化：按分子大小减少构象尝试 ——
    orig_attempts = _worker_screener.embedder.config.num_conformer_attempts
    n_heavy = sum(1 for a in mol_2d.GetAtoms() if a.GetAtomicNum() > 1)
    if n_heavy > 75:
        _worker_screener.embedder.config.num_conformer_attempts = 1
    elif n_heavy > 50:
        _worker_screener.embedder.config.num_conformer_attempts = max(1, orig_attempts // 3)

    result = {
        "mol_index": idx,
        "smiles": smiles,
        "target_pg": target_pg,
        "core_name": core_name,
        "arm_name": arm_name,
        "initial_pg": None, "final_pg": None,
        "symmetry_preserved": False,
        "energy": float("nan"), "energy_per_atom": float("nan"),
        "max_force": float("inf"), "xyz_path": None,
        "status": "PENDING", "error": None,
    }

    try:
        screen_result = _worker_screener.screen(mol_2d)
    except Exception as exc:
        result["status"] = "FAIL_EXCEPTION"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    # 合并筛查结果
    result.update({k: v for k, v in screen_result.items() if k != "mol"})
    relaxed_mol = screen_result.get("mol")

    # 保存 XYZ 文件
    if relaxed_mol is not None and result["status"] == "PASS":
        xyz_filename = f"mol_{idx:06d}.xyz"
        xyz_path = os.path.join(_worker_xyz_dir, xyz_filename)
        comment = (
            f"SMILES={result['smiles']} "
            f"PG={result['final_pg']} "
            f"E={result['energy']:.6f}kcal"
        )
        try:
            write_xyz(relaxed_mol, xyz_path, comment=comment)
            result["xyz_path"] = xyz_path
        except Exception as exc:
            logger = logging.getLogger("cof_pipeline")
            logger.warning(f"XYZ 写入失败 mol_{idx}: {exc}")

    return result


def _safe_smiles(mol: Chem.Mol) -> str:
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return "N/A"


# =============================================================================
# 管线编排器
# =============================================================================

class COFPipeline:
    """端到端管线编排。"""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger("cof_pipeline")

        # 组件
        self.generator = MoleculeGenerator()
        self.screener = SymmetryScreener(
            embed_config=self.config.embedding,
            sym_config=self.config.symmetry,
            relax_config=self.config.relax,
        )

        # 输出路径
        self.output_dir = self.config.output_dir
        self.xyz_dir = os.path.join(self.output_dir, self.config.xyz_subdir)
        self.csv_path = os.path.join(self.output_dir, self.config.csv_filename)
        os.makedirs(self.xyz_dir, exist_ok=True)

    # ── 阶段 1：2D 生成 ──────────────────────────────────────────────────

    def stage1_generate(
        self,
        cores: Optional[List[str]] = None,
        arms: Optional[List[str]] = None,
    ) -> List[Chem.Mol]:
        """组合生成所有 2D 分子。若配置了 max_heavy_atoms，过滤超大分子。"""
        self.logger.info("=" * 60)
        self.logger.info("Stage 1: 2D 拓扑对称组合生成")
        self.logger.info("=" * 60)

        core_names = cores or list(self.generator.core_templates.keys())
        # —— 层级模式：默认只处理层级核心（含 __），跳过已完成的原始核心 ——
        if self.config.hierarchical_mode and cores is None:
            core_names = [c for c in core_names if '__' in c]
            self.logger.info(f"层级模式: 自动排除原始核心，选中 {len(core_names)} 个层级核心")
        arm_names = arms or list(self.generator.arm_library.keys())
        self.logger.info(f"核心数: {len(core_names)}, 臂数: {len(arm_names)}")
        self.logger.info(f"预期组合数: {len(core_names) * len(arm_names)}")

        mols = generate_dataset(cores=core_names, arms=arm_names)

        # —— 层级核心优化：重原子数预过滤 ——
        max_heavy = self.config.max_heavy_atoms
        if max_heavy > 0:
            before = len(mols)
            mols = [
                m for m in mols
                if sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1) <= max_heavy
            ]
            if before > len(mols):
                self.logger.info(
                    f"MW 预过滤: {before} → {len(mols)} "
                    f"(砍掉 {before-len(mols)} 个, 阈值 heavy≤{max_heavy})"
                )

        self.logger.info(f"Stage 1 完成: {len(mols)} 个有效 2D 分子")
        return mols

    # ── 阶段 2+3：嵌入 + 松弛 + 对称快筛 ─────────────────────────────────

    def stage2_3_screen(
        self,
        molecules_2d: List[Chem.Mol],
    ) -> List[Dict]:
        """
        对一批 2D 分子完成嵌入 / 松弛 / 对称破缺筛查。

        根据配置选择串行或多进程模式。
        层级核心模式下：按重原子数升序处理，大分子减少构象尝试。
        """
        n_total = len(molecules_2d)

        # —— 层级核心优化：按大小排序，小的先跑 ——
        if self.config.hierarchical_mode:
            molecules_2d = sorted(
                molecules_2d,
                key=lambda m: sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)
            )
            self.logger.info("层级模式: 分子已按重原子数升序排列")

        self.logger.info("=" * 60)
        self.logger.info(f"Stage 2+3: 3D 嵌入 + MLIP 松弛 + 对称快筛 ({n_total} 分子)")
        self.logger.info(f"后端: {self.config.relax.backend}, "
                         f"workers: {self.config.relax.num_workers}")
        self.logger.info("=" * 60)

        results: List[Dict] = []

        if self.config.relax.num_workers > 1:
            results = self._screen_parallel(molecules_2d)
        else:
            results = self._screen_sequential(molecules_2d)

        # 统计
        status_counts = {}
        for r in results:
            s = r.get("status", "UNKNOWN")
            status_counts[s] = status_counts.get(s, 0) + 1

        self.logger.info("─" * 40)
        self.logger.info("筛查统计:")
        for status, count in sorted(status_counts.items()):
            pct = 100 * count / max(n_total, 1)
            self.logger.info(f"  {status:25s}: {count:5d} ({pct:5.1f}%)")
        self.logger.info("─" * 40)

        return results

    def _screen_sequential(self, molecules_2d: List[Chem.Mol]) -> List[Dict]:
        """串行处理（调试/单机模式）。"""
        results = []
        t0 = time.time()
        for i, mol in enumerate(molecules_2d):
            if (i + 1) % max(1, len(molecules_2d) // 10) == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                self.logger.info(
                    f"  进度: {i+1}/{len(molecules_2d)} "
                    f"({rate:.1f} mol/s)"
                )

            try:
                result = self.screener.screen(mol)
            except Exception as exc:
                result = {
                    "mol_index": i,
                    "smiles": _safe_smiles(mol),
                    "target_pg": mol.GetProp("target_pg") if mol.HasProp("target_pg") else "C1",
                    "status": "FAIL_EXCEPTION",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            result["mol_index"] = i
            result["core_name"] = mol.GetProp("core_name") if mol.HasProp("core_name") else "?"
            result["arm_name"] = mol.GetProp("arm_name") if mol.HasProp("arm_name") else "?"
            results.append(result)

            # 保存 XYZ 文件（在串行模式下直接写入）
            if result["status"] == "PASS" and result.get("mol") is not None:
                xyz_filename = f"mol_{i:06d}.xyz"
                xyz_path = os.path.join(self.xyz_dir, xyz_filename)
                comment = (
                    f"SMILES={result['smiles']} "
                    f"PG={result['final_pg']} "
                    f"E={result['energy']:.6f}kcal"
                )
                try:
                    write_xyz(result["mol"], xyz_path, comment=comment)
                    result["xyz_path"] = xyz_path
                except Exception as exc:
                    self.logger.warning(f"XYZ 写入失败 mol_{i}: {exc}")

        return results

    def _screen_parallel(self, molecules_2d: List[Chem.Mol]) -> List[Dict]:
        """多进程并行处理。"""
        n_workers = self.config.relax.num_workers
        n_total = len(molecules_2d)

        # 准备参数
        relax_config_dict = {
            "backend": self.config.relax.backend,
            "max_steps": self.config.relax.max_steps,
            "fmax": self.config.relax.fmax,
            "force_threshold": self.config.relax.force_threshold,
            "energy_threshold": self.config.relax.energy_threshold,
            "optimizer": self.config.relax.optimizer,
            "mace_model": self.config.relax.mace_model,
            "mace_device": self.config.relax.mace_device,
            "xtb_method": self.config.relax.xtb_method,
        }

        # 传 SMILES+属性，避免 RDKit Mol 经 pickle 丢失属性
        tasks = [
            (i,
             _safe_smiles(mol),
             mol.GetProp("target_pg") if mol.HasProp("target_pg") else "C1",
             mol.GetProp("core_name") if mol.HasProp("core_name") else "?",
             mol.GetProp("arm_name") if mol.HasProp("arm_name") else "?")
            for i, mol in enumerate(molecules_2d)
        ]
        results_dict: Dict[int, Dict] = {}

        self.logger.info(f"启动 {n_workers} 个子进程...")
        t0 = time.time()

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_initializer,
            initargs=(relax_config_dict, self.xyz_dir),
        ) as executor:
            futures = {
                executor.submit(_process_one, task): task[0]
                for task in tasks
            }

            n_done = 0
            for future in as_completed(futures):
                n_done += 1
                try:
                    result = future.result(timeout=600)  # 10 min timeout per mol
                    results_dict[result["mol_index"]] = result
                except Exception as exc:
                    mol_idx = futures[future]
                    results_dict[mol_idx] = {
                        "mol_index": mol_idx,
                        "smiles": "N/A",
                        "status": "FAIL_TIMEOUT",
                        "error": str(exc),
                    }
                    self.logger.error(f"mol_{mol_idx} 超时/异常: {exc}")

                if n_done % max(1, n_total // 20) == 0:
                    elapsed = time.time() - t0
                    rate = n_done / elapsed
                    eta = (n_total - n_done) / max(rate, 1e-6)
                    self.logger.info(
                        f"  [{n_done}/{n_total}] {rate:.2f} mol/s, "
                        f"ETA: {eta:.0f}s"
                    )

        # 按原始顺序排序
        results = [results_dict[i] for i in range(n_total)]
        elapsed = time.time() - t0
        self.logger.info(f"并行处理完成: {n_total} 分子 / {elapsed:.1f}s "
                         f"({n_total/elapsed:.2f} mol/s)")
        return results

    # ── 阶段 4：输出 ──────────────────────────────────────────────────────

    def stage4_export(self, results: List[Dict]):
        """输出 CSV 数据集 + 统计摘要。"""
        self.logger.info("=" * 60)
        self.logger.info("Stage 4: 导出数据集")
        self.logger.info("=" * 60)

        # 构建 DataFrame
        rows = []
        for r in results:
            rows.append({
                "Index": r.get("mol_index", -1),
                "SMILES": r.get("smiles", "N/A"),
                "Core": r.get("core_name", "?"),
                "Arm": r.get("arm_name", "?"),
                "Point_Group": r.get("final_pg", r.get("initial_pg", "?")),
                "Target_PG": r.get("target_pg", "?"),
                "Initial_PG": r.get("initial_pg", "?"),
                "Symmetry_Preserved": r.get("symmetry_preserved", False),
                "Energy_kcal_mol": r.get("energy", float("nan")),
                "Energy_per_Atom": r.get("energy_per_atom", float("nan")),
                "Max_Displacement_A": r.get("max_force", float("inf")),
                "XYZ_File_Path": r.get("xyz_path", ""),
                "Status": r.get("status", "UNKNOWN"),
                "Error": r.get("error", ""),
            })

        df = pd.DataFrame(rows)

        # 保存 CSV
        df.to_csv(self.csv_path, index=False)
        self.logger.info(f"CSV 已保存: {self.csv_path} ({len(df)} 行)")

        # 统计摘要
        n_pass = (df["Status"] == "PASS").sum()
        n_total = len(df)
        self.logger.info(f"最终通过率: {n_pass}/{n_total} "
                         f"({100*n_pass/max(n_total,1):.1f}%)")

        # 按核心和臂的分布
        if n_pass > 0:
            df_pass = df[df["Status"] == "PASS"]
            self.logger.info("\n按核心分布 (PASS):")
            for core, grp in df_pass.groupby("Core"):
                self.logger.info(f"  {core}: {len(grp)}")
            self.logger.info("\n按臂分布 (PASS):")
            for arm, grp in df_pass.groupby("Arm"):
                self.logger.info(f"  {arm}: {len(grp)}")
            self.logger.info("\n按点群分布 (PASS):")
            for pg, grp in df_pass.groupby("Point_Group"):
                self.logger.info(f"  {pg}: {len(grp)}")

        # 保存统计 JSON
        stats_path = os.path.join(self.output_dir, "pipeline_stats.json")
        stats = {
            "timestamp": datetime.now().isoformat(),
            "n_total": n_total,
            "n_pass": int(n_pass),
            "pass_rate": float(100 * n_pass / max(n_total, 1)),
            "status_counts": df["Status"].value_counts().to_dict(),
            "config": {
                "cores": list(CORE_TEMPLATES.keys()),
                "arms": list(ARM_LIBRARY.keys()),
                "backend": self.config.relax.backend,
                "max_steps": self.config.relax.max_steps,
                "force_threshold": self.config.relax.force_threshold,
            },
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        self.logger.info(f"统计已保存: {stats_path}")

        return df

    # ── 构象展开（倍增数据集）───────────────────────────────────────────
    # 策略：对每个通过筛查的分子，用不同随机种子生成多个 3D 构象
    # 每个构象都是独立的数据点（同一分子，不同几何构型）
    # 理论倍增率 = n_expand × (构象通过率)
    # 实践中：182 基础分子 × 20 种子 → ~3000 个独立数据点

    def stage_expand_conformers(
        self, results: List[Dict], n_expand: int = 20
    ) -> List[Dict]:
        """对每个PASS分子生成多个构象，返回所有通过筛查的构象。"""
        # 提取所有通过第一轮筛查的基础分子
        pass_results = [r for r in results if r.get("status") == "PASS"]
        if not pass_results:
            self.logger.info("无 PASS 分子，跳过构象展开")
            return list(results)

        self.logger.info("=" * 60)
        self.logger.info(
            f"构象展开: {len(pass_results)} 基础分子 × {n_expand} 种子"
        )
        self.logger.info("=" * 60)

        all_results = list(results)  # 保留原始结果
        # 质数种子序列：保证不同种子间的伪随机独立性
        base_seeds = [
            137, 251, 383, 509, 631, 757, 883, 1009, 1151, 1297,
            1447, 1597, 1753, 1907, 2063, 2213, 2371, 2531, 2693, 2851,
            3011, 3169, 3329, 3491, 3653, 3817, 3983, 4153, 4327, 4441,
        ]

        t0 = time.time()
        n_new = 0  # 累计新增构象数
        for base_r in pass_results:
            # 从 SMILES 重建分子（保证每个循环有干净的 Mol 对象）
            mol_2d = Chem.MolFromSmiles(base_r["smiles"])
            if mol_2d is None:
                continue
            mol_2d.SetProp("target_pg", base_r.get("target_pg", "C1"))
            mol_2d.SetProp("core_name", base_r.get("core_name", "?"))
            mol_2d.SetProp("arm_name", base_r.get("arm_name", "?"))

            # 用不同种子生成构象
            for j, seed in enumerate(base_seeds[:n_expand]):
                self.screener.embedder.config.random_seed = seed
                try:
                    # 运行完整筛查（嵌入 + 松弛 + 对称检测）
                    result = self.screener.screen(mol_2d)
                except Exception:
                    continue  # 个别种子可能失败，跳过

                # 只保留通过筛查的构象
                if result["status"] == "PASS":
                    n_new += 1
                    result["mol_index"] = len(all_results)
                    result["core_name"] = base_r["core_name"]
                    result["arm_name"] = base_r["arm_name"]
                    result["conformer_seed"] = seed  # 记录种子来源

                    # 保存 XYZ 坐标文件（供后续训练直接加载）
                    relaxed_mol = result.get("mol")
                    if relaxed_mol is not None:
                        xyz_filename = (
                            f"mol_{result['mol_index']:06d}_s{seed:04d}.xyz"
                        )
                        xyz_path = os.path.join(self.xyz_dir, xyz_filename)
                        comment = (
                            f"SMILES={result['smiles']} "
                            f"PG={result['final_pg']} "
                            f"E={result['energy']:.6f}kcal seed={seed}"
                        )
                        try:
                            write_xyz(relaxed_mol, xyz_path, comment=comment)
                            result["xyz_path"] = xyz_path
                        except Exception:
                            pass
                    all_results.append(result)

            # 定期报告进度
            if (n_new + 1) % max(1, (len(pass_results) * n_expand) // 10) == 0:
                elapsed = time.time() - t0
                self.logger.info(
                    f"  新构象: {n_new} ({n_new / max(elapsed, 0.1):.1f}/s)"
                )

        elapsed = time.time() - t0
        self.logger.info(
            f"构象展开完成: +{n_new} 新构象, "
            f"总计 {len(all_results)} 条 ({elapsed:.1f}s)"
        )
        return all_results

    # ── 一键运行 ──────────────────────────────────────────────────────────

    def run(
        self,
        cores: Optional[List[str]] = None,
        arms: Optional[List[str]] = None,
        n_expand: int = 0,
    ) -> pd.DataFrame:
        """
        运行完整管线：生成 → 筛查 → 构象展开 → 导出。

        Returns:
            包含所有分子筛查结果的 DataFrame
        """
        t_start = time.time()
        self.logger.info("=" * 60)
        self.logger.info("COF 对称核心数据集管线 — 启动")
        self.logger.info(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
        self.logger.info(f"输出目录: {self.output_dir}")
        if self.config.hierarchical_mode:
            self.logger.info(
                f"层级模式: max_heavy={self.config.max_heavy_atoms}, "
                f"conformer_attempts={self.config.embedding.num_conformer_attempts}"
            )
        self.logger.info("=" * 60)

        # Stage 1
        mols_2d = self.stage1_generate(cores=cores, arms=arms)
        if not mols_2d:
            self.logger.error("Stage 1 未生成任何分子，终止")
            return pd.DataFrame()

        # —— 层级核心优化：对每个分子根据大小动态调整 conformer 尝试数 ——
        if self.config.hierarchical_mode:
            orig_attempts = self.screener.embedder.config.num_conformer_attempts
            # 对大分子减少尝试次数
            for mol in mols_2d:
                n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
                if n_heavy > 75:
                    # 大分子：1次尝试就够了
                    mol.SetIntProp("_hier_attempts", 1)
                elif n_heavy > 50:
                    mol.SetIntProp("_hier_attempts", max(1, orig_attempts // 3))
                else:
                    mol.SetIntProp("_hier_attempts", orig_attempts)

        # Stage 2+3
        results = self.stage2_3_screen(mols_2d)

        # 构象展开（仅当 n_expand > 0 时启用，默认关闭因 MMFF 不同种子收敛到同一极小点）
        if n_expand > 0:
            results = self.stage_expand_conformers(results, n_expand=n_expand)

        # Stage 4
        df = self.stage4_export(results)

        elapsed = time.time() - t_start
        n_pass = (df["Status"] == "PASS").sum() if "Status" in df.columns else 0
        self.logger.info("=" * 60)
        self.logger.info(
            f"管线完成! 用时: {elapsed:.1f}s, "
            f"通过: {n_pass}/{len(df)} "
            f"({100*n_pass/max(len(df),1):.1f}%)"
        )
        self.logger.info("=" * 60)

        return df


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="COF 对称中间体核心数据集构建管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                         # 全组合，串行
  python main.py --cores triazine_C3 biphenyl_C2          # 指定核心
  python main.py --arms CHO_ph NH2_ph BOH2_ph             # 指定臂
  python main.py --workers 8 --backend mace               # 8进程 + MACE
  python main.py --workers 4 --backend tblite --max-steps 50  # xTB 快速快筛
  python main.py --dataset-demo                           # 生成后演示 PyTorch Dataset
        """,
    )
    # 生成控制
    p.add_argument("--cores", type=str, nargs="*", default=None,
                   help="核心模板名称 (空格分隔)，默认全部")
    p.add_argument("--arms", type=str, nargs="*", default=None,
                   help="臂名称 (空格分隔)，默认全部")

    # MLIP 控制
    p.add_argument("--backend", type=str, default="rdkit_mmff",
                   choices=["rdkit_mmff", "rdkit_uff", "mace", "tblite"],
                   help="优化后端 (default: rdkit_mmff)")
    p.add_argument("--max-steps", type=int, default=100,
                   help="最大优化步数 (default: 100)")
    p.add_argument("--fmax", type=float, default=0.05,
                   help="力收敛阈值 eV/Å (default: 0.05)")
    p.add_argument("--force-threshold", type=float, default=3.0,
                   help="原子位移/力淘汰阈值 (MMFF: Å, ASE: eV/Å, default: 3.0)")

    # 并行控制
    p.add_argument("--workers", type=int, default=1,
                   help="并行进程数 (default: 1, 串行)")

    # 输出控制
    p.add_argument("--output-dir", type=str,
                   default="cof_symmetry_pipeline/output",
                   help="输出目录 (default: cof_symmetry_pipeline/output)")
    p.add_argument("--csv-name", type=str, default="pipeline_results.csv",
                   help="CSV 文件名")

    # 层级核心优化
    p.add_argument("--hierarchical", action="store_true",
                   help="层级核心模式：MW预过滤 + 减少构象尝试 + 按大小排序")
    p.add_argument("--max-heavy", type=int, default=0,
                   help="重原子数上限 (default: 0=不过滤, 层级模式建议 75)")

    # 其他
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--dataset-demo", action="store_true",
                   help="生成后运行 PyTorch Dataset 演示")
    return p.parse_args()


def run_dataset_demo(csv_path: str, xyz_dir: str):
    """运行 PyTorch Dataset 演示。"""
    print("\n" + "=" * 60)
    print("PyTorch Dataset 演示")
    print("=" * 60)

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("[跳过] PyTorch 未安装")
        return

    from dataset import COFSymmetryDataset, SimpleARTransformer

    if not os.path.exists(csv_path):
        print(f"[跳过] CSV 不存在: {csv_path}")
        return

    ds = COFSymmetryDataset(
        csv_path=csv_path,
        xyz_dir=xyz_dir,
        max_atoms=200,
    )
    print(f"数据集大小: {len(ds)} 分子")

    if len(ds) == 0:
        print("[跳过] 数据集中无 PASS 分子")
        return

    dl = DataLoader(
        ds, batch_size=4, shuffle=True,
        collate_fn=COFSymmetryDataset.collate_fn,
    )
    batch = next(iter(dl))

    print(f"\n批次结构:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s}: shape={tuple(v.shape)} dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"  {k:20s}: list[{len(v)}] e.g. {v[0][:50]}")
        else:
            print(f"  {k:20s}: {v}")

    # 模型演示
    model = SimpleARTransformer(
        vocab_size=12,
        d_model=128,
        nhead=4,
        num_layers=4,
        max_len=200,
    )
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    logits = model(
        batch["atom_tokens"],
        batch["coords"],
        sym_mask=batch["sym_attn_mask"],
        attn_mask=batch["attention_mask"],
    )
    print(f"输出 logits shape: {tuple(logits.shape)}")
    print("\nDataset + Model demo 成功! ✓")


# =============================================================================
# main
# =============================================================================

def main():
    args = parse_args()

    # 构建配置
    config = PipelineConfig(
        embedding=EmbeddingConfig(
            random_seed=args.seed,
            # 层级模式: 减少每个分子的构象尝试次数
            num_conformer_attempts=3 if args.hierarchical else 8,
        ),
        symmetry=SymmetryConfig(tolerance=0.3),
        relax=RelaxConfig(
            backend=args.backend,
            max_steps=args.max_steps,
            fmax=args.fmax,
            force_threshold=args.force_threshold,
            num_workers=args.workers,
        ),
        output_dir=args.output_dir,
        csv_filename=args.csv_name,
        log_file=os.path.join(args.output_dir, "pipeline.log"),
        log_level=args.log_level,
        seed=args.seed,
        # 层级核心优化
        max_heavy_atoms=args.max_heavy if args.max_heavy > 0 else (75 if args.hierarchical else 0),
        hierarchical_mode=args.hierarchical,
    )

    # 初始化日志
    logger = setup_logging(config)
    logger.info(f"配置: backend={args.backend}, workers={args.workers}, "
                f"max_steps={args.max_steps}")

    # 运行管线
    pipeline = COFPipeline(config)
    df = pipeline.run(cores=args.cores, arms=args.arms)

    # PyTorch Dataset 演示
    if args.dataset_demo:
        csv_path = os.path.join(args.output_dir, args.csv_name)
        xyz_dir = os.path.join(args.output_dir, config.xyz_subdir)
        run_dataset_demo(csv_path, xyz_dir)

    return df


if __name__ == "__main__":
    main()
