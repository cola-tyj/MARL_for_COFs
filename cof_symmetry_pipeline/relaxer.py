"""
cof_symmetry_pipeline.relaxer — 3D 嵌入、点群分析、几何松弛、对称破缺快筛。

管线：
  1. ETKDGv3 嵌入 → 3D 构象
  2. pymatgen PointGroupAnalyzer → 初始点群
  3. 几何优化 (RDKit MMFF94 / MACE / xTB) → 应力筛查
  4. 再次点群分析 → 对称破缺检测

后端优先级: rdkit_mmff (默认, 内置) > mace > xtb > rdkit_uff
"""

import logging
import math
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from config import (
    BROKEN_SYMMETRY_TAGS,
    PG_EQUIVALENCE,
    EmbeddingConfig,
    RelaxConfig,
    SymmetryConfig,
)

logger = logging.getLogger(__name__)

# =============================================================================
# 可选导入
# =============================================================================

try:
    from pymatgen.core import Molecule as PmgMolecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    logger.warning("pymatgen 未安装，点群分析将退回简易几何判定")

try:
    from ase import Atoms as AseAtoms
    from ase.optimize import BFGS as AseBFGS
    from ase.optimize import FIRE as AseFIRE
    from ase.optimize import LBFGS as AseLBFGS
    HAS_ASE = True
    ASE_OPTIMIZERS = {"BFGS": AseBFGS, "FIRE": AseFIRE, "LBFGS": AseLBFGS}
except ImportError:
    HAS_ASE = False
    ASE_OPTIMIZERS = {}
    logger.warning("ASE 未安装，仅使用 RDKit 内置力场")


# =============================================================================
# 3D 构象嵌入器
# =============================================================================

class Embedder:
    """
    使用 ETKDGv3 将 2D 分子嵌入 3D 坐标空间。

    ETKDGv3 = Experimental Torsion Knowledge Distance Geometry v3
    - 基于扭转角知识的实验性距离几何算法
    - 随机坐标初始化 → 距离边界平滑 → 度量矩阵嵌入 → 手性约束
    """

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()

    def embed(self, mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], bool, str]:
        """
        对分子进行 3D 嵌入。

        流程：加氢 → ETKDGv3 嵌入 → MMFF94 初步结构优化

        Args:
            mol: 无 H 的 RDKit Mol

        Returns:
            (含 3D 构象的 Mol（已加 H）, 成功与否, 状态信息)
        """
        # 添加显式氢原子（对力场和点群分析都很重要）
        mol_h = Chem.AddHs(mol)

        # 配置 ETKDGv3 参数
        params = AllChem.ETKDGv3()
        params.randomSeed = self.config.random_seed     # 随机种子（可调以获取不同构象）
        params.maxIterations = self.config.max_attempts  # 最大尝试次数
        params.useRandomCoords = self.config.use_random_coords  # 随机坐标初始化
        params.pruneRmsThresh = self.config.prune_rms_thresh    # 构象去重 RMSD 阈值

        # 执行距离几何嵌入
        try:
            result = AllChem.EmbedMolecule(mol_h, params)
        except Exception as exc:
            return None, False, f"ETKDGv3 抛异常: {exc}"

        # 返回码 0 = 成功，非 0 = 位阻过大或坐标无法满足距离约束
        if result != 0:
            return None, False, f"ETKDGv3 返回码={result}（位阻冲突）"

        # MMFF94 初步优化：修正不合理的键长和键角（不做完整松弛）
        # MMFF94 = Merck Molecular Force Field，专门为有机分子参数化
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass  # MMFF 失败不致命（某些官能团可能无参数）

        return mol_h, True, "OK"


# =============================================================================
# 点群对称分析器
# =============================================================================

class SymmetryAnalyzer:
    """基于 pymatgen 的点群分析，含回退方案。"""

    def __init__(self, config: SymmetryConfig = None):
        self.config = config or SymmetryConfig()

    def analyze(self, mol: Chem.Mol, conf_id: int = -1) -> Dict:
        """
        分析分子的 3D 点群对称性。

        Returns:
            {'sch_symbol': str, 'is_symmetric': bool,
             'n_rotations': int, 'error': str|None}
        """
        if not HAS_PYMATGEN:
            return self._fallback_analysis(mol, conf_id)

        try:
            conf = mol.GetConformer(conf_id)
            n_atoms = mol.GetNumAtoms()
            species = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]
            coords = [[
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ] for i in range(n_atoms)]

            mol_pmg = PmgMolecule(species, coords)
            analyzer = PointGroupAnalyzer(
                mol_pmg,
                tolerance=self.config.tolerance,
                eigen_tolerance=self.config.eigen_tolerance,
            )
            sch = analyzer.sch_symbol
            return {
                "sch_symbol": sch,
                "is_symmetric": sch not in BROKEN_SYMMETRY_TAGS,
                "n_rotations": self._count_rotations(sch),
                "error": None,
            }
        except Exception as exc:
            logger.warning(f"点群分析异常: {exc}")
            return self._fallback_analysis(mol, conf_id)

    @staticmethod
    def _count_rotations(sch: str) -> int:
        if sch in ("C1", "Cs", "Ci", "S2"):
            return 0
        import re
        nums = re.findall(r"\d+", sch)
        return max((int(n) for n in nums), default=0)

    def _fallback_analysis(self, mol: Chem.Mol, conf_id: int) -> Dict:
        """简易惯性张量回退。"""
        try:
            conf = mol.GetConformer(conf_id)
            n = mol.GetNumAtoms()
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n)])
            masses = np.array([atom.GetMass() for atom in mol.GetAtoms()])
            com = np.average(coords, axis=0, weights=masses)
            centered = coords - com

            I = np.zeros((3, 3))
            for i in range(n):
                m = masses[i]
                r = centered[i]
                I += m * (np.eye(3) * (r @ r) - np.outer(r, r))
            evals, _ = np.linalg.eigh(I)

            tol = self.config.eigen_tolerance
            unique_evals = []
            for ev in evals:
                denom = max(abs(ev), 1e-8)
                if not unique_evals or abs(ev - unique_evals[-1]) / denom > tol:
                    unique_evals.append(ev)

            n_degen = 3 - len(unique_evals)
            if n_degen >= 2:
                guess = "C3+"
            elif n_degen == 1:
                guess = "C2"
            else:
                guess = "C1"
            return {
                "sch_symbol": guess,
                "is_symmetric": guess not in BROKEN_SYMMETRY_TAGS,
                "n_rotations": int(guess[1]) if len(guess) > 1 and guess[1].isdigit() else 0,
                "error": "回退模式——安装 pymatgen 以获得精确点群",
            }
        except Exception:
            return {
                "sch_symbol": "?", "is_symmetric": False,
                "n_rotations": 0, "error": "回退分析失败",
            }


# =============================================================================
# 几何优化器
# =============================================================================

class GeometryRelaxer:
    """
    几何优化器——支持多种后端，默认使用 RDKit 内置力场。

    后端:
      - "rdkit_mmff": RDKit MMFF94 (有机分子优化效果最好，默认)
      - "rdkit_uff":  RDKit UFF (通用力场，覆盖元素更广)
      - "mace":       MACE-OFF23 机器学习势 (需 mace-torch)
      - "tblite":     GFN2-xTB 半经验量子化学 (需 tblite)
    """

    def __init__(self, config: RelaxConfig = None):
        self.config = config or RelaxConfig()

    def relax(
        self, mol: Chem.Mol, conf_id: int = -1
    ) -> Tuple[Chem.Mol, float, float, bool, str]:
        """
        对分子进行几何优化。

        Args:
            mol: 含 3D 构象的 RDKit Mol（含 H）

        Returns:
            (松弛后 Mol, 能量 kcal/mol, 最大位移 Å, 成功, 状态)
        """
        backend = self.config.backend

        # ── ASE 后端 (MACE / xTB) ──
        if backend in ("mace", "tblite"):
            return self._relax_ase(mol, conf_id)

        # ── RDKit 力场后端 ──
        if backend == "rdkit_uff":
            return self._relax_rdkit_uff(mol, conf_id)
        else:
            # 默认: MMFF94
            return self._relax_rdkit_mmff(mol, conf_id)

    # ── RDKit MMFF94 ────────────────────────────────────────────────────

    def _relax_rdkit_mmff(
        self, mol: Chem.Mol, conf_id: int = -1
    ) -> Tuple[Chem.Mol, float, float, bool, str]:
        """使用 RDKit MMFF94 力场优化。"""
        try:
            coords_before = self._get_coords(mol, conf_id)

            ff = AllChem.MMFFGetMoleculeForceField(
                mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id
            )
            if ff is None:
                # MMFF 参数不完整 → 回退到 UFF
                logger.debug("MMFF 不可用, 回退到 UFF")
                return self._relax_rdkit_uff(mol, conf_id)

            ff.Initialize()
            energy_before = ff.CalcEnergy()

            result = ff.Minimize(maxIts=self.config.max_steps)
            converged = (result == 0)

            energy_after = ff.CalcEnergy()
            coords_after = self._get_coords(mol, conf_id)
            max_disp = np.sqrt(((coords_after - coords_before) ** 2).sum(axis=1)).max()

            return mol, energy_after, float(max_disp), True, (
                "converged" if converged else f"max_steps({self.config.max_steps})"
            )
        except Exception as exc:
            logger.error(f"MMFF 优化失败: {exc}")
            return mol, float("nan"), float("inf"), False, str(exc)

    # ── RDKit UFF ────────────────────────────────────────────────────────

    def _relax_rdkit_uff(
        self, mol: Chem.Mol, conf_id: int = -1
    ) -> Tuple[Chem.Mol, float, float, bool, str]:
        """使用 RDKit UFF 力场优化。"""
        try:
            coords_before = self._get_coords(mol, conf_id)

            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            if ff is None:
                return mol, float("nan"), float("inf"), False, "UFF 不可用"

            ff.Initialize()
            energy_before = ff.CalcEnergy()

            result = ff.Minimize(maxIts=self.config.max_steps)
            converged = (result == 0)

            energy_after = ff.CalcEnergy()
            coords_after = self._get_coords(mol, conf_id)
            max_disp = np.sqrt(((coords_after - coords_before) ** 2).sum(axis=1)).max()

            return mol, energy_after, float(max_disp), True, (
                "converged" if converged else f"max_steps({self.config.max_steps})"
            )
        except Exception as exc:
            logger.error(f"UFF 优化失败: {exc}")
            return mol, float("nan"), float("inf"), False, str(exc)

    # ── ASE 后端 ─────────────────────────────────────────────────────────

    def _relax_ase(
        self, mol: Chem.Mol, conf_id: int = -1
    ) -> Tuple[Chem.Mol, float, float, bool, str]:
        """通过 ASE 使用 MACE 或 tblite 优化。"""
        if not HAS_ASE:
            return mol, float("nan"), float("inf"), False, "ASE 未安装"

        calc = self._make_ase_calculator()
        if calc is None:
            return mol, float("nan"), float("inf"), False, "无可用 ASE 计算器"

        try:
            conf = mol.GetConformer(conf_id)
            symbols = [mol.GetAtomWithIdx(i).GetSymbol()
                       for i in range(mol.GetNumAtoms())]
            positions = [[conf.GetAtomPosition(i).x,
                          conf.GetAtomPosition(i).y,
                          conf.GetAtomPosition(i).z]
                         for i in range(mol.GetNumAtoms())]

            atoms = AseAtoms(symbols=symbols, positions=positions)
            atoms.calc = calc

            opt_cls = ASE_OPTIMIZERS.get(self.config.optimizer, AseBFGS)
            opt = opt_cls(atoms)
            try:
                opt.run(fmax=self.config.fmax, steps=self.config.max_steps)
                converged = opt.converged()
            except Exception as exc:
                logger.warning(f"ASE 优化中断: {exc}")
                converged = False

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            max_force = float(np.sqrt((forces ** 2).sum(axis=1)).max())

            # 将坐标写回 RDKit Mol
            rw_mol = Chem.RWMol(mol)
            rw_conf = rw_mol.GetConformer(conf_id)
            for i, pos in enumerate(atoms.positions):
                rw_conf.SetAtomPosition(i, pos)

            status = "converged" if converged else f"max_steps({self.config.max_steps})"
            return rw_mol.GetMol(), energy, max_force, True, status

        except Exception as exc:
            logger.error(f"ASE 松弛失败: {exc}")
            return mol, float("nan"), float("inf"), False, str(exc)

    def _make_ase_calculator(self):
        """按后端创建 ASE 计算器。"""
        backend = self.config.backend

        if backend == "mace":
            try:
                from mace.calculators import MACECalculator
                calc = MACECalculator(
                    model_path=self.config.mace_model,
                    device=self.config.mace_device,
                )
                logger.info(f"✓ MACE: {self.config.mace_model}")
                return calc
            except ImportError:
                logger.warning("mace-torch 未安装")
            except Exception as exc:
                logger.warning(f"MACE 加载失败: {exc}")

        if backend in ("mace", "tblite"):
            try:
                from tblite.ase import TBLite
                calc = TBLite(method=self.config.xtb_method)
                logger.info(f"✓ tblite: {self.config.xtb_method}")
                return calc
            except ImportError:
                logger.warning("tblite 未安装")
            except Exception as exc:
                logger.warning(f"tblite 加载失败: {exc}")

        return None

    # ── 工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_coords(mol: Chem.Mol, conf_id: int = -1) -> np.ndarray:
        conf = mol.GetConformer(conf_id)
        return np.array([
            [conf.GetAtomPosition(i).x,
             conf.GetAtomPosition(i).y,
             conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ])


# =============================================================================
# 对称破缺快筛器
# =============================================================================

class SymmetryScreener:
    """
    组合嵌入→点群分析→松弛→对称破缺检测的完整快筛管线。
    """

    def __init__(
        self,
        embed_config: EmbeddingConfig = None,
        sym_config: SymmetryConfig = None,
        relax_config: RelaxConfig = None,
    ):
        self.embedder = Embedder(embed_config)
        self.sym_analyzer = SymmetryAnalyzer(sym_config)
        self.relaxer = GeometryRelaxer(relax_config)
        self.relax_config = relax_config or RelaxConfig()

    def screen(self, mol_2d: Chem.Mol) -> Dict:
        """
        对单个分子完成全流程筛查。

        支持多构象尝试：若嵌入失败或初始点群不匹配，
        使用不同随机种子重试，选最佳匹配继续松弛。

        Returns:
            {'smiles', 'mol', 'target_pg', 'initial_pg', 'final_pg',
             'symmetry_preserved', 'energy', 'energy_per_atom', 'max_force',
             'xyz_path', 'status', 'error'}
        """
        target_pg = mol_2d.GetProp("target_pg") if mol_2d.HasProp("target_pg") else "C1"
        n_attempts = self.embedder.config.num_conformer_attempts

        # Canonical SMILES
        try:
            smiles = Chem.MolToSmiles(mol_2d, canonical=True)
        except Exception:
            smiles = "N/A"

        # ── Stage 2: 多构象尝试（选最佳点群匹配）──
        # 核心思想：ETKDGv3 是随机算法，不同种子产生不同构象
        # 每个构象计算点群后按匹配度打分，选得分最高的进入松弛阶段
        # 乘 137（质数）保证各种子间的伪随机独立性
        best_mol_h = None          # 最佳构象（RDKit Mol）
        best_pg_score = -999.0     # 最佳点群匹配得分
        best_pg_result = None      # 最佳构象的点群分析结果
        best_embed_msg = ""        # 最后一次嵌入失败的错误信息

        for attempt in range(n_attempts):
            seed = self.embedder.config.random_seed + attempt * 137
            self.embedder.config.random_seed = seed

            # 嵌入 3D 构象（含 MMFF 初步优化）
            mol_h, ok, msg = self.embedder.embed(mol_2d)
            if not ok or mol_h is None:
                best_embed_msg = msg
                continue  # 本次嵌入失败，尝试下一个种子

            # 计算点群并打分
            pg_result = self.sym_analyzer.analyze(mol_h)
            pg = pg_result["sch_symbol"]
            score = self._pg_match_score(target_pg, pg)

            # 保留最佳构象
            if score > best_pg_score:
                best_pg_score = score
                best_mol_h = mol_h
                best_pg_result = pg_result

            # 精确匹配则无需继续尝试（已找到最优构象）
            if score >= 3.0:
                break

        # 所有尝试均嵌入失败
        if best_mol_h is None:
            return {
                "smiles": smiles, "mol": None,
                "target_pg": target_pg,
                "initial_pg": None, "final_pg": None,
                "symmetry_preserved": False,
                "energy": float("nan"), "energy_per_atom": float("nan"),
                "max_force": float("inf"), "xyz_path": None,
                "status": "FAIL_EMBED",
                "error": f"所有 {n_attempts} 次嵌入尝试失败: {best_embed_msg}",
            }

        n_atoms_3d = best_mol_h.GetNumAtoms()
        initial_pg = best_pg_result["sch_symbol"]

        # ── Stage 3: Geometry Relaxation ──
        t0 = time.time()
        relaxed_mol, energy, max_disp, relax_ok, relax_status = \
            self.relaxer.relax(best_mol_h)
        _elapsed = time.time() - t0

        # 能量/力淘汰
        if not relax_ok:
            return {
                "smiles": smiles, "mol": None,
                "target_pg": target_pg,
                "initial_pg": initial_pg, "final_pg": None,
                "symmetry_preserved": False,
                "energy": float("nan"), "energy_per_atom": float("nan"),
                "max_force": float("inf"), "xyz_path": None,
                "status": "FAIL_RELAX",
                "error": relax_status,
            }

        if abs(energy) > 1e8 or np.isnan(energy):
            return {
                "smiles": smiles, "mol": relaxed_mol,
                "target_pg": target_pg,
                "initial_pg": initial_pg, "final_pg": None,
                "symmetry_preserved": False,
                "energy": energy, "energy_per_atom": energy / max(n_atoms_3d, 1),
                "max_force": max_disp, "xyz_path": None,
                "status": "FAIL_ENERGY",
                "error": f"能量异常: {energy:.2f} kcal/mol",
            }

        if max_disp > self.relax_config.force_threshold:
            return {
                "smiles": smiles, "mol": relaxed_mol,
                "target_pg": target_pg,
                "initial_pg": initial_pg, "final_pg": None,
                "symmetry_preserved": False,
                "energy": energy,
                "energy_per_atom": energy / max(n_atoms_3d, 1),
                "max_force": max_disp, "xyz_path": None,
                "status": "FAIL_FORCE",
                "error": f"原子位移过大: {max_disp:.2f} Å > {self.relax_config.force_threshold}",
            }

        # ── Stage 3c: 对称破缺检测 ──
        pg_final = self.sym_analyzer.analyze(relaxed_mol)
        final_pg = pg_final["sch_symbol"]
        symmetry_preserved = self._check_symmetry(target_pg, final_pg)

        if not symmetry_preserved:
            return {
                "smiles": smiles, "mol": relaxed_mol,
                "target_pg": target_pg,
                "initial_pg": initial_pg, "final_pg": final_pg,
                "symmetry_preserved": False,
                "energy": energy,
                "energy_per_atom": energy / max(n_atoms_3d, 1),
                "max_force": max_disp, "xyz_path": None,
                "status": "FAIL_SYMMETRY_BREAK",
                "error": f"对称破缺: {target_pg} → {final_pg}",
            }

        # ── 全部通过 ──
        return {
            "smiles": smiles, "mol": relaxed_mol,
            "target_pg": target_pg,
            "initial_pg": initial_pg, "final_pg": final_pg,
            "symmetry_preserved": True,
            "energy": energy,
            "energy_per_atom": energy / max(n_atoms_3d, 1),
            "max_force": max_disp, "xyz_path": None,
            "status": "PASS",
            "error": None,
        }

    @staticmethod
    def _pg_match_score(target_pg: str, actual_pg: str) -> float:
        """
        点群匹配得分（用于多构象选优：在 N 个候选构象中选点群最接近目标者）。

        评分体系（按匹配质量降序）：
          3.0 : 精确匹配 (target == actual，如 C3 → C3)
          2.0 : 等价类内匹配 (如 C3v → C3，同一旋转轴的不同派生群)
          1.0 : 超群匹配 (如 D3h → C3，实际对称性比目标更高)
          0.5 : 数值匹配 (虽不在等价类但旋转轴阶数一致，如 C6 → C3 的目标)
          0.0 : 无匹配（点群完全不同但非破缺）
         -1.0 : 对称破缺 (C1, Cs, Ci — 无旋转轴，结构已塌陷)
        """
        # 对称破缺 — 最差情况
        if actual_pg in BROKEN_SYMMETRY_TAGS:
            return -1.0
        # 精确匹配 — 最优
        if actual_pg == target_pg:
            return 3.0
        # 目标等价类 — 良好
        allowed = PG_EQUIVALENCE.get(target_pg, {target_pg})
        if actual_pg in allowed:
            return 2.0
        # 实际点群的超群包含目标 — 可接受（更高对称性）
        allowed_higher = PG_EQUIVALENCE.get(actual_pg, set())
        if target_pg in allowed_higher:
            return 1.0
        # 弱匹配：提取 Schoenflies 符号中的数字，比较旋转轴阶数
        # 例如：C6 有 6 阶轴 → 对目标 C3（3 阶轴）有一定匹配度
        import re
        tgt_nums = set(re.findall(r"\d+", target_pg))
        act_nums = set(re.findall(r"\d+", actual_pg))
        if tgt_nums and act_nums and max(int(n) for n in act_nums) >= max(int(n) for n in tgt_nums):
            return 0.5
        return 0.0

    @staticmethod
    def _check_symmetry(target_pg: str, actual_pg: str) -> bool:
        """检查实际点群是否满足目标要求（松匹配）。"""
        if actual_pg in BROKEN_SYMMETRY_TAGS:
            return False
        if actual_pg == target_pg:
            return True
        allowed = PG_EQUIVALENCE.get(target_pg, {target_pg})
        if actual_pg in allowed:
            return True
        allowed_higher = PG_EQUIVALENCE.get(actual_pg, set())
        if target_pg in allowed_higher:
            return True
        return False


# =============================================================================
# XYZ 文件写入
# =============================================================================

def write_xyz(
    mol: Chem.Mol,
    filepath: str,
    comment: str = "",
    conf_id: int = -1,
):
    """将 RDKit Mol 的 3D 坐标写入 XYZ 格式。"""
    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(f"{n}\n")
        f.write(f"{comment}\n")
        for i in range(n):
            atom = mol.GetAtomWithIdx(i)
            sym = atom.GetSymbol()
            pos = conf.GetAtomPosition(i)
            f.write(f"{sym:3s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")
