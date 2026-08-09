"""
cof_symmetry_pipeline.dataset — PyTorch Dataset 接口。

COFSymmetryDataset:
  从 CSV+XYZ 文件中读取分子数据，转换为自回归 Transformer 训练所需的
  Tensor 表征（不对称单元序列 + 对称注意力掩码提示）。

设计思路（参考 apart_diffusion）：
  - 分子 = 不对称单元(AU) × 对称操作
  - AU 原子序列 → token sequence for autoregressive LM
  - 对称注意力掩码：告知模型哪些位置在对称操作下等价
"""

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# =============================================================================
# 常量
# =============================================================================

# 原子类型词汇表（按常见度排序，0=pad, 1=H, 2=C, 3=N, 4=O, 5=F, 6=B, ...）
ATOM_VOCAB = {"<pad>": 0, "H": 1, "C": 2, "N": 3, "O": 4, "F": 5,
              "B": 6, "P": 7, "S": 8, "Cl": 9, "Br": 10, "I": 11}
ATOM_VOCAB_SIZE = len(ATOM_VOCAB)

# 键类型词汇表
BOND_VOCAB = {"<pad>": 0, "NONE": 1, "SINGLE": 2, "DOUBLE": 3,
              "TRIPLE": 4, "AROMATIC": 5}


# =============================================================================
# 辅助函数
# =============================================================================

def parse_xyz(filepath: str) -> Tuple[List[str], np.ndarray]:
    """
    解析 XYZ 文件。

    Returns:
        (原子符号列表, shape=(N,3) 坐标数组)
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    symbols = []
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    for i, line in enumerate(lines[2:2 + n_atoms]):
        parts = line.strip().split()
        if len(parts) >= 4:
            symbols.append(parts[0])
            coords[i] = [float(x) for x in parts[1:4]]
    return symbols, coords


def atoms_to_tensor(
    symbols: List[str], coords: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """原子符号 + 坐标 → token_ids + coord tensor"""
    token_ids = torch.tensor(
        [ATOM_VOCAB.get(s, ATOM_VOCAB["C"]) for s in symbols],
        dtype=torch.long,
    )
    coord_tensor = torch.from_numpy(coords).float()
    return token_ids, coord_tensor


def compute_symmetry_attention_mask(
    coords: np.ndarray,
    symbols: List[str],
    rotation_order: int,
    tolerance: float = 0.5,
) -> torch.Tensor:
    """
    为自回归 Transformer 构建对称注意力掩码。

    物理含义：
      分子有 C_n 旋转对称性 → 每个原子通过旋转 2π/n 映到等价原子
      → 模型在生成原子 i 时应能关注它的所有等价副本
      → mask[i, j] 控制 i→j 的注意力强度

    算法步骤：
      1. 平移分子至质心，计算惯性张量
      2. 取最大本征值方向 = 主旋转轴
      3. Rodrigues 旋转公式生成旋转矩阵 R(θ), θ=2π/n
      4. 旋转全分子坐标，找每个原子的等价原子（旋转后与原位 < tolerance）
      5. 等价类内 mask=1.0，跨类 mask=0.5（允许信息流通但不强化）

    Args:
        coords: (N, 3) 分子笛卡尔坐标
        symbols: 原子符号列表（用于区分元素类型）
        rotation_order: C_n 旋转阶数 n
        tolerance: 等价判定容差 (Å)，默认 0.5

    Returns:
        mask: (N, N) float tensor, mask[i,j]=1 表示 j 可被 i attend
    """
    n = len(symbols)

    # 无旋转对称 → 全通掩码（标准自回归）
    if rotation_order <= 1:
        return torch.ones(n, n)

    # ── 1. 平移至质心（消除平动自由度）──
    centered = coords - coords.mean(axis=0)

    # ── 2. 惯性张量 & 主旋转轴 ──
    # 惯性张量 I_ab = Σ m_i (|r|²δ_ab - r_a r_b)
    # 最大本征值对应 z 轴（分子最长轴），通常即旋转轴方向
    masses = np.array([
        1.008 if s == "H" else 12.01 if s == "C" else 14.01 if s == "N"
        else 16.00 if s == "O" else 10.81 if s == "B" else 12.01
        for s in symbols
    ])
    I = np.zeros((3, 3))
    for i in range(n):
        r = centered[i]
        I += masses[i] * (np.eye(3) * (r @ r) - np.outer(r, r))
    evals, evecs = np.linalg.eigh(I)
    axis = evecs[:, -1]  # 最大本征值方向 → 主旋转轴

    # ── 3. Rodrigues 旋转公式 ──
    # R = I + sin(θ) K + (1-cos(θ)) K²
    # 其中 K 是旋转轴 a 的叉积矩阵
    angle = 2 * math.pi / rotation_order
    ct, st = math.cos(angle), math.sin(angle)
    a = axis

    K = np.array([[0, -a[2], a[1]],
                   [a[2], 0, -a[0]],
                   [-a[1], a[0], 0]])
    R = np.eye(3) + st * K + (1 - ct) * (K @ K)

    rotated = (R @ centered.T).T  # 旋转后的坐标 (N, 3)

    # ── 4. 等价类映射 ──
    # 原子 i 的等价类 = { 所有可通过 C_n 操作与 i 重合的同元素原子 }
    equiv_map = {}  # atom_i → set of equivalent atom indices
    for i in range(n):
        equiv_map[i] = {i}  # 自身总是等价
        for j in range(n):
            if i == j:
                continue
            if symbols[i] != symbols[j]:
                continue  # 不同元素的原子不可能等价
            d2_orig = ((centered[i] - centered[j]) ** 2).sum()
            d2_rot = ((rotated[i] - centered[j]) ** 2).sum()
            if min(d2_orig, d2_rot) < tolerance ** 2:
                equiv_map[i].add(j)

    # ── 5. 构建注意力掩码 ──
    # 等价类内：完整注意力（mask=1.0）→ 强化对称一致性
    # 跨等价类：衰减注意力（mask=0.5）→ 允许长程交互但不主导
    mask = torch.ones(n, n)
    for i in range(n):
        for j in range(n):
            if j not in equiv_map[i]:
                mask[i, j] = 0.5

    return mask


def compute_asymmetric_unit_mask(
    coords: np.ndarray,
    symbols: List[str],
    rotation_order: int,
    tolerance: float = 0.5,
) -> Tuple[torch.Tensor, List[int]]:
    """
    识别不对称单元 (AU) 及其在全分子中的掩码。

    原理：从质心出发，选择旋转角 [0, 2π/n) 范围内的原子作为 AU。
    每个 AU 原子通过 C_n 操作生成 (n-1) 个等价副本。

    Returns:
        au_mask: (N,) bool tensor, True=该原子属于 AU
        au_map:   List[int], 每个 AU 原子在全分子中的索引
    """
    n = len(symbols)
    centered = coords - coords.mean(axis=0)

    # 惯性主轴
    I = np.zeros((3, 3))
    for i in range(n):
        r = centered[i]
        m = 12.0
        I += m * (np.eye(3) * (r @ r) - np.outer(r, r))
    evals, evecs = np.linalg.eigh(I)
    axis = evecs[:, -1]

    # 计算每个原子的方位角（绕主轴）
    azimuths = np.zeros(n)
    for i in range(n):
        r = centered[i]
        # 投影到垂直于主轴的平面
        r_perp = r - (r @ axis) * axis
        norm = np.linalg.norm(r_perp)
        if norm < 1e-6:
            azimuths[i] = 0.0  # 在轴上
        else:
            # 选择一个参考方向
            ref = np.array([1.0, 0.0, 0.0])
            ref_perp = ref - (ref @ axis) * axis
            ref_norm = np.linalg.norm(ref_perp)
            if ref_norm < 1e-6:
                ref = np.array([0.0, 1.0, 0.0])
                ref_perp = ref - (ref @ axis) * axis
                ref_norm = np.linalg.norm(ref_perp)
            cos_phi = (r_perp @ ref_perp) / (norm * ref_norm)
            cos_phi = np.clip(cos_phi, -1, 1)
            azimuths[i] = math.acos(cos_phi)

    # AU: 方位角在 [0, 2π/n) 范围内的原子
    sector = 2 * math.pi / rotation_order
    au_mask = azimuths < sector
    au_indices = np.where(au_mask)[0].tolist()

    return torch.tensor(au_mask, dtype=torch.bool), au_indices


# =============================================================================
# PyTorch Dataset
# =============================================================================

class COFSymmetryDataset(Dataset):
    """
    对称 COF 分子数据集。

    每个样本返回：
      - atom_tokens:  (max_len,) 原子类型 token IDs (padded)
      - coords:       (max_len, 3) 笛卡尔坐标 (padded)
      - attention_mask: (max_len,) 有效位置 mask
      - sym_attn_mask:  (max_len, max_len) 对称注意力掩码
      - au_mask:        (max_len,) 不对称单元 mask
      - point_group:    str 点群标签
      - smiles:         str SMILES 字符串
    """

    def __init__(
        self,
        csv_path: str,
        xyz_dir: str,
        max_atoms: int = 200,
        rotation_order: int = None,
    ):
        """
        Args:
            csv_path: 含 SMILES, Point_Group, XYZ_File_Path 的 CSV
            xyz_dir:  XYZ 文件目录
            max_atoms: 最大原子数（补齐长度）
            rotation_order: 若为 None，从 Point_Group 列自动推断
        """
        self.df = pd.read_csv(csv_path)
        # 只保留 PASS 状态的分子
        if "Status" in self.df.columns:
            self.df = self.df[self.df["Status"] == "PASS"].reset_index(drop=True)
        self.xyz_dir = xyz_dir
        self.max_atoms = max_atoms
        self.rotation_order = rotation_order
        logger.info(f"COFSymmetryDataset: {len(self.df)} 个通过筛选的分子")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        # 解析 XYZ
        xyz_file = row.get("XYZ_File_Path", None)
        if xyz_file is None or not os.path.exists(xyz_file):
            # 尝试从 xyz_dir 构建路径
            xyz_file = os.path.join(
                self.xyz_dir,
                os.path.basename(str(row.get("XYZ_File_Path", f"mol_{idx}.xyz"))),
            )

        symbols: List[str]
        coords: np.ndarray
        if os.path.exists(xyz_file):
            symbols, coords = parse_xyz(xyz_file)
        else:
            # 空占位
            symbols, coords = ["C"], np.zeros((1, 3), dtype=np.float32)
            logger.warning(f"XYZ 文件缺失: {xyz_file}")

        n = len(symbols)

        # Token 化
        token_ids, coord_tensor = atoms_to_tensor(symbols, coords)

        # 补齐至 max_atoms
        padded_tokens = torch.full(
            (self.max_atoms,), ATOM_VOCAB["<pad>"], dtype=torch.long
        )
        padded_coords = torch.zeros(self.max_atoms, 3)
        attn_mask = torch.zeros(self.max_atoms)
        actual_n = min(n, self.max_atoms)
        padded_tokens[:actual_n] = token_ids[:actual_n]
        padded_coords[:actual_n] = coord_tensor[:actual_n]
        attn_mask[:actual_n] = 1.0

        # 对称阶数推断
        pg = str(row.get("Point_Group", "C1"))
        rot_order = self.rotation_order
        if rot_order is None:
            rot_order = self._infer_rotation_order(pg)

        # 对称注意力掩码
        sym_mask = compute_symmetry_attention_mask(
            coords[:actual_n], symbols[:actual_n], rot_order
        )
        padded_sym_mask = torch.zeros(self.max_atoms, self.max_atoms)
        padded_sym_mask[:actual_n, :actual_n] = sym_mask

        # 不对称单元掩码
        au_mask_full, au_indices = compute_asymmetric_unit_mask(
            coords[:actual_n], symbols[:actual_n], rot_order
        )
        padded_au_mask = torch.zeros(self.max_atoms, dtype=torch.bool)
        padded_au_mask[:actual_n] = au_mask_full

        return {
            "atom_tokens": padded_tokens,
            "coords": padded_coords,
            "attention_mask": attn_mask,
            "sym_attn_mask": padded_sym_mask,
            "au_mask": padded_au_mask,
            "n_atoms": actual_n,
            "rotation_order": rot_order,
            "point_group": pg,
            "smiles": str(row.get("SMILES", "")),
        }

    @staticmethod
    def _infer_rotation_order(pg: str) -> int:
        """从点群符号推断旋转阶数"""
        import re
        numbers = re.findall(r"\d+", pg)
        if not numbers:
            return 1
        return max(int(n) for n in numbers)

    # ── 用于 collate_fn ──────────────────────────────────────────────────

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """PyTorch DataLoader 的批量整理函数"""
        keys = [
            "atom_tokens", "coords", "attention_mask",
            "sym_attn_mask", "au_mask", "n_atoms", "rotation_order",
        ]
        batched = {}
        for k in keys:
            vals = [item[k] for item in batch]
            if isinstance(vals[0], torch.Tensor):
                batched[k] = torch.stack(vals)
            else:
                batched[k] = torch.tensor(vals)
        # 字符串字段
        batched["point_group"] = [item["point_group"] for item in batch]
        batched["smiles"] = [item["smiles"] for item in batch]
        return batched


# =============================================================================
# 简单自回归表征演示
# =============================================================================

class SimpleARTransformer(torch.nn.Module):
    """
    演示用的原生 nn.Transformer 自回归模型。

    输入: atom_tokens + coords + sym_attn_mask
    输出: next-token prediction over atom vocabulary

    这是最小示例，展示如何使用 COFSymmetryDataset 的输出训练
    一个对称感知的自回归分子生成模型。
    """

    def __init__(
        self,
        vocab_size: int = ATOM_VOCAB_SIZE,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        max_len: int = 200,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embed = torch.nn.Embedding(vocab_size, d_model)
        self.coord_proj = torch.nn.Linear(3, d_model)
        self.pos_embed = torch.nn.Embedding(max_len, d_model)
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=0,  # decoder-only
            num_decoder_layers=num_layers,
            batch_first=True,
        )
        self.output_head = torch.nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,           # (B, L)
        coords: torch.Tensor,           # (B, L, 3)
        sym_mask: torch.Tensor = None,  # (B, L, L)
        attn_mask: torch.Tensor = None,  # (B, L)
    ) -> torch.Tensor:
        """
        Args:
            tokens:  输入 token IDs
            coords:  坐标
            sym_mask: 对称注意力掩码
            attn_mask: 填充掩码
        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = tokens.shape
        device = tokens.device

        # 嵌入
        tok_emb = self.token_embed(tokens)
        coord_emb = self.coord_proj(coords)
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos)
        x = tok_emb + coord_emb + pos_emb

        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(L, L, device=device) * float("-inf"), diagonal=1
        )

        # 融合对称掩码
        if sym_mask is not None:
            combined_mask = causal_mask + (1 - sym_mask) * float("-inf")
        else:
            combined_mask = causal_mask

        # key padding mask
        if attn_mask is not None:
            key_mask = (attn_mask == 0)
        else:
            key_mask = None

        # Decoder-only Transformer
        x = self.transformer(
            x, x,
            tgt_mask=combined_mask,
            tgt_key_padding_mask=key_mask,
        )
        return self.output_head(x)
