"""
cof_symmetry_pipeline.generator — 2D 拓扑对称组合生成。

通过 RDKit 的 ReplaceSubstructs 将侧链端基等价替换到核心模板的
对称占位符上，保证生成的 2D 分子图 100% 具备指定拓扑对称性。

技术细节：
  - 核心和臂均使用 * (RDKit dummy atom, Z=0) 作为反应位点
  - ReplaceSubstructs 生成带桥接 * 原子的中间体
  - _cleanup_bridges() 移除桥接 * 原子并形成直接 C-C 键
"""

import logging
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from config import CORE_TEMPLATES, ARM_LIBRARY

logger = logging.getLogger(__name__)


# =============================================================================
# 桥接原子清理
# =============================================================================

def _cleanup_bridges(mol: Chem.Mol) -> Chem.Mol:
    """
    移除分子中充当桥接的 dummy (*) 原子，并将其两端邻居直接键连。

    ReplaceSubstructs 将核心的 * 替换为臂时，臂的 * 原子会成为
    连接核心与臂的"桥"（degree=2，两侧分别是核心锚点和臂的挂载碳）。
    本函数移除这些桥接 * 并形成直接 C-C 键。
    """
    # 找出所有桥接 dummy 原子（atomic_num=0 且 degree=2）
    bridges: List[Tuple[int, int, int]] = []  # (bridge_idx, neighbor1, neighbor2)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetDegree() == 2:
            nbrs = atom.GetNeighbors()
            bridges.append((atom.GetIdx(), nbrs[0].GetIdx(), nbrs[1].GetIdx()))

    if not bridges:
        return mol

    # 收集所有桥接原子索引（升序排列，用于后续计算偏移量）
    bridge_indices = sorted([b[0] for b in bridges])

    # 从高索引向低索引移除桥接原子，保证每次移除只影响索引更高的原子
    mol_rw = Chem.RWMol(mol)
    for bridge_idx, _, _ in sorted(bridges, reverse=True):
        mol_rw.RemoveAtom(bridge_idx)

    # 在桥接原子两端创建直连键
    # 邻居索引需要校正：原始索引减去在它之前被移除的桥接原子数量
    # 例如：原子索引 5，已移除索引 2,3 → 实际索引 = 5 - 2 = 3
    for _, n1, n2 in bridges:
        adj_n1 = n1 - sum(1 for bi in bridge_indices if bi < n1)
        adj_n2 = n2 - sum(1 for bi in bridge_indices if bi < n2)
        try:
            mol_rw.AddBond(adj_n1, adj_n2, Chem.BondType.SINGLE)
        except Exception:
            pass  # 键已存在则跳过（极少情况，如芳香环内已经成键）

    return mol_rw.GetMol()


# =============================================================================
# 核心生成器
# =============================================================================

class MoleculeGenerator:
    """组合合成引擎：核心 + 臂 → 完整分子"""

    def __init__(self, core_templates: dict = None, arm_library: dict = None):
        self.core_templates = core_templates or CORE_TEMPLATES
        self.arm_library = arm_library or ARM_LIBRARY
        self._dummy_query = Chem.MolFromSmiles("*")
        if self._dummy_query is None:
            raise RuntimeError("无法解析 dummy query '*'")

    # ── 单次生成 ────────────────────────────────────────────────────────

    def generate_one(
        self, core_name: str, arm_name: str
    ) -> Optional[Chem.Mol]:
        """
        将一个臂的所有对称位点替换到核心上，返回完整分子。

        核心算法（三步走）：
          1. ReplaceSubstructs → 核心的 * 被替换为臂分子（臂的 * 成为桥原子）
          2. _cleanup_bridges → 移除桥接 * 原子，形成核心-臂直连键
          3. SanitizeMol    → 芳香性感知 + 化合价校验

        Args:
            core_name: CORE_TEMPLATES 的键
            arm_name:  ARM_LIBRARY 的键

        Returns:
            RDKit Mol（已 sanitize，无 dummy 原子），失败返回 None
        """
        # 检查核心和臂是否存在于模板库中
        core_info = self.core_templates.get(core_name)
        arm_info = self.arm_library.get(arm_name)
        if core_info is None:
            logger.error(f"未知核心: {core_name}")
            return None
        if arm_info is None:
            logger.error(f"未知臂: {arm_name}")
            return None

        # 解析 SMILES → RDKit Mol 对象
        core_mol = Chem.MolFromSmiles(core_info["smiles"])
        arm_mol = Chem.MolFromSmiles(arm_info["smiles"])
        if core_mol is None:
            logger.error(f"核心 SMILES 解析失败: {core_name} → {core_info['smiles'][:60]}")
            return None
        if arm_mol is None:
            logger.error(f"臂 SMILES 解析失败: {arm_name} → {arm_info['smiles'][:60]}")
            return None

        # Step 1: 替换 —— 核心中所有 * 被等价替换为臂分子
        # replaceAll=True 确保核心的 N 个对称位点被同一个臂替换，保证拓扑对称性
        try:
            products = AllChem.ReplaceSubstructs(
                core_mol, self._dummy_query, arm_mol, replaceAll=True
            )
        except Exception as exc:
            logger.warning(f"ReplaceSubstructs 异常 ({core_name}+{arm_name}): {exc}")
            return None

        if not products or len(products) == 0:
            logger.warning(f"ReplaceSubstructs 无产物: {core_name}+{arm_name}")
            return None

        mol = products[0]
        if mol is None:
            return None

        # Step 2: 清理 —— 移除桥接 * 原子，形成核心-臂直连键
        # 臂的 * 在 ReplaceSubstructs 后成为 degree=2 的桥原子
        # _cleanup_bridges 检测并移除此类原子，两侧直接成键
        try:
            mol = _cleanup_bridges(mol)
        except Exception as exc:
            logger.warning(f"桥接清理失败 ({core_name}+{arm_name}): {exc}")
            return None

        # Step 3: 化学校验 —— Kekulé 化（芳香性感知）+ 化合价检查
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            logger.warning(f"Sanitize 失败 ({core_name}+{arm_name}): {exc}")
            return None

        # 确认无残留 dummy 原子（安全检查）
        if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            logger.warning(f"残留 dummy 原子: {core_name}+{arm_name}")
            return None

        # 在分子上附着元数据（用于后续阶段追踪来源）
        mol.SetProp("core_name", core_name)
        mol.SetProp("arm_name", arm_name)
        mol.SetProp("target_pg", core_info.get("target_pg", "C1"))

        return mol

    # ── 全组合生成 ────────────────────────────────────────────────────────

    def generate_all(
        self,
        cores: Optional[List[str]] = None,
        arms: Optional[List[str]] = None,
    ) -> List[Chem.Mol]:
        """
        遍历所有核心×臂的组合，生成分子列表。
        """
        core_names = cores or list(self.core_templates.keys())
        arm_names = arms or list(self.arm_library.keys())

        molecules: List[Chem.Mol] = []
        attempted, success, failed = 0, 0, 0

        for core_name in core_names:
            for arm_name in arm_names:
                attempted += 1
                mol = self.generate_one(core_name, arm_name)
                if mol is not None:
                    molecules.append(mol)
                    success += 1
                    logger.info(
                        f"  OK  {core_name:30s} + {arm_name:15s} -> "
                        f"{mol.GetNumAtoms()} atoms"
                    )
                else:
                    failed += 1
                    logger.warning(f"  FAIL {core_name:30s} + {arm_name:15s}")

        logger.info(
            f"生成完毕: {success}/{attempted} 成功 "
            f"({100*success/max(attempted,1):.1f}%)"
        )
        return molecules

    # ── 去重 ────────────────────────────────────────────────────────────

    @staticmethod
    def deduplicate(molecules: List[Chem.Mol]) -> List[Chem.Mol]:
        """基于 Canonical SMILES 和分子式去重，保留首次出现者。"""
        seen = set()
        unique = []
        for mol in molecules:
            try:
                smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            except Exception:
                continue
            formula = rdMolDescriptors.CalcMolFormula(mol)
            key = (smi, formula)
            if key not in seen:
                seen.add(key)
                unique.append(mol)
        n_dup = len(molecules) - len(unique)
        if n_dup > 0:
            logger.info(f"去重: 移除 {n_dup} 个重复分子, 保留 {len(unique)}")
        return unique

    # ── 质量验证 ─────────────────────────────────────────────────────────

    @staticmethod
    def validate_mol(mol: Chem.Mol) -> Tuple[bool, str]:
        """化合价检验 + 结构合理性检查。"""
        # 1. Sanitization
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            return False, f"Sanitize: {e}"

        # 2. 无 dummy 原子残留
        if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            return False, "残留 dummy 原子"

        # 3. 化合价检查
        try:
            issues = Chem.rdmolops.GetMolValenceIssues(mol)
            if issues:
                return False, f"化合价异常: {len(issues)} 处"
        except Exception:
            pass

        # 4. 原子数合理性
        n_atoms = mol.GetNumAtoms()
        if n_atoms < 5:
            return False, f"原子数过少: {n_atoms}"
        if n_atoms > 500:
            return False, f"原子数过多: {n_atoms}"

        # 5. 碳骨架检查
        if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
            return False, "无碳原子"

        return True, "OK"


# =============================================================================
# 便捷函数
# =============================================================================

def generate_dataset(
    cores: Optional[List[str]] = None,
    arms: Optional[List[str]] = None,
    dedup: bool = True,
    validate: bool = True,
) -> List[Chem.Mol]:
    """一键生成 + 去重 + 验证数据集。"""
    gen = MoleculeGenerator()
    mols = gen.generate_all(cores=cores, arms=arms)

    if validate:
        valid_mols = []
        for mol in mols:
            ok, reason = MoleculeGenerator.validate_mol(mol)
            if ok:
                valid_mols.append(mol)
            else:
                logger.warning(f"验证未通过: {reason}")
        mols = valid_mols
        logger.info(f"验证后保留 {len(mols)} 个分子")

    if dedup:
        mols = MoleculeGenerator.deduplicate(mols)

    return mols
