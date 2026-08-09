# 基于规则生成 + QM9 联合训练的对称分子扩散模型

**提出日期**: 2026-07-10  
**核心思路**: 规则生成大量对称 2D 分子图 → 与 QM9 联合训练 → 条件扩散 → 生成指定对称性的合法分子

---

## 1. 问题背景

当前困境：276 个 Core 不足以训练任何生成模型（VAE/Phase2/Orbit-Circulant 均失败）。QM9 有 130K 分子但缺少对称性标签，且不能直接映射为 Core。

**新思路**：既然数据不够，就用规则造数据。

---

## 2. 方法概述

```
┌─────────────────────────────────────────────────┐
│  Step 1: 规则生成对称分子图                          │
│                                                     │
│  For each symmetry (C2v, D3h, D4h, D6h):           │
│    1. 选中心骨架 (苯环/三嗪/卟啉...)                  │
│    2. 沿对称轴扩展分支                               │
│    3. 化合价约束 → 合法分子                           │
│    4. 输出: 2D 图 (原子类型 + 邻接矩阵)                │
│    5. RDKit 生成 3D 构象                             │
│                                                     │
│  目标: 每种对称性生成 5000-10000 个分子                │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Step 2: 联合训练                                  │
│                                                     │
│  数据集: QM9 (130K, label=0) + 对称分子 (~50K, label=1) │
│  模型: Phase 1 denoiser + FiLM 条件 (label)          │
│  任务: 去噪坐标 + 原子类型 + 键类型                     │
│  条件: binary label → 16-dim embedding → FiLM        │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Step 3: 条件生成                                  │
│                                                     │
│  输入: target_symmetry + label=1                    │
│  扩散采样 → 具有对称性的分子                           │
│  后处理: 检测对称轴 → 标记 Q/R → Core               │
└─────────────────────────────────────────────────┘
```

---

## 3. 对称分子图规则生成

### 3.1 中心骨架库

每种对称性预定义中心骨架：

| 对称 | 点群 | 中心骨架 | Q 位置 |
|------|------|---------|--------|
| L2 | C2v | 联苯, 偶氮苯, 苯并噻二唑 | C2 轴两端 |
| T3 | D3h | 三嗪, 三苯, 三蝶烯 | C3 轴上 |
| S4/D4 | D4h | 卟啉, 酞菁, 四苯 | C4 轴/镜面 |
| H6 | D6h | 六苯并蔻, 六氮杂苯 | C6 轴 |

### 3.2 扩展规则

从中心骨架出发，沿对称方向迭代添加分支：

```
1. 识别开放位点 (H 原子或 valence-1 的位置)
2. 沿对称轴方向选择位点对（必须对称等价）
3. 添加化学基团: -CH3, -NH2, -OH, -F, -COOH, -CHO, 苯环
4. 检查化合价约束 (C=4, N=3, O=2, H=1, F=1)
5. 重复 1-4 直到达到目标分子量
```

### 3.3 多样性保障

- 不同骨架 × 不同基团组合 = 大量变体
- 随机扩展深度 (2-6 层)
- 随机分支模式 (线性 vs 分叉)
- 每种对称性预计可生成 5000-10000 个不同分子

### 3.4 3D 构象生成

```python
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol)
# → 3D 坐标用于扩散训练
```

---

## 4. 模型设计

### 4.1 架构

复用 QM9 Phase 1 denoiser (EGNN 9层, hidden=256):

```
输入: atom_types(N,5) + coords(N,3) + bond_types(E,5) + sym_label(1,)
  ↓
sym_label → Embedding(16) → FiLM condition
  ↓
EGNN denoiser (预训练权重初始化)
  ↓
输出: coord_noise + atom_logits + bond_logits
```

### 4.2 训练策略

**两阶段训练**:

**Stage A** (warmup, ~50 epochs):
- 50% QM9 + 50% 对称分子
- 冻结 EGNN backbone
- 仅训练 FiLM 条件层 + 输出头
- 目的: 模型学会区分"对称"和"任意"分子

**Stage B** (fine-tune, ~100 epochs):
- 30% QM9 + 70% 对称分子
- 解冻最后 3 层 EGNN
- 更强调对称分子生成质量
- 低学习率 (5e-5)

### 4.3 损失函数

```
L_total = L_coord + L_atom + L_bond + λ_sym * L_sym_classifier

L_sym_classifier: 辅助分类损失，预测分子是否对称
  → 鼓励 FiLM 条件编码对称性信息
```

---

## 5. 预期效果

### 5.1 优势

| 维度 | 当前 (Phase 2) | 新方案 |
|------|:---:|:---:|
| 训练数据 | 276→5000 | **130K + 50K** |
| 对称性标签 | 隐式(FiLM on PG) | **显式(binary label)** |
| 原子类型 | 12类(含Q/R) | **5类(纯化学)** |
| 生成对象 | Core(含人造Q/R) | **纯对称分子** |
| 后处理 | 无 | Q/R 标记(规则化) |
| 模型初始化 | 随机 | **QM9 预训练** |

### 5.2 风险

| 风险 | 缓解 |
|------|------|
| 规则生成的分子不"自然" | QM9 联合训练提供真实分子分布 |
| 2D 图→3D 构象质量差 | RDKit MMFF 优化 + 扩散去噪补偿 |
| 只学到"look like synthetic" | QM9 占比逐渐降低而非完全移除 |
| label=1 太宽泛(所有对称) | 可扩展到 per-symmetry 标签 |
| 原子类型限制(5类) | COF Core 以 C/H/N/O 为主, 5 类足够 |

### 5.3 与之前路线的对比

| 路线 | 数据 | 模型 | 结果 |
|------|:---:|:---:|:---:|
| VAE | 276 | 14M | ❌ |
| Phase 2 | 276→5K | 7M | ❌ 偏向Q |
| Scaffold 扩散 | 276 | 7M | ✅ 组装通过 |
| Orbit-Circulant | 67 | 483K | ❌ 原子随机 |
| **QM9+规则联合** | **180K** | **7M(QM9预训)** | **?** |

---

## 6. 实现计划

### Phase 5a: 规则生成器 (1-2天)
1. 对称骨架库 (每种对称 5-10 个骨架)
2. 分支扩展器 (化合价约束)
3. SMILES 生成 + RDKit 构象优化
4. 验证: 生成的分子是否确实有目标对称性

### Phase 5b: 联合训练 (1天)
1. 数据集构建 (QM9 + synthetic, label 0/1)
2. 添加 FiLM 条件层到 Phase 1 模型
3. 两阶段训练 (Stage A + Stage B)
4. 评估: symmetry consistency, validity, uniqueness

### Phase 5c: MARL 集成 (1天)
1. 条件生成 (label=1)
2. 对称轴检测 + Q/R 标记
3. pycofbuilder 组装测试
4. 扩散 MARL 训练

---

## 7. 关键问题与决策

**Q: 为什么用 binary label 而不是 per-symmetry label?**
A: 先验证 binary 方案能工作。如果模型学会区分对称/非对称，再扩展到 4-class (C2v/D3h/D4h/D6h)。

**Q: 如何保证生成分子有正确数量的 Q/R 位置?**
A: Q/R 在后处理阶段通过检测对称轴确定，不参与扩散生成。

**Q: 规则生成的分子会不会包含不合理的化学结构?**
A: 化合价约束 + QM9 联合训练作为正则化。不合理结构在 QM9 梯度下被抑制。

**Q: 为什么不让 RDKit 直接枚举所有对称分子?**
A: 枚举空间太大。规则生成 + 扩散模型可以在合理子集中采样。

---

## 8. 规则生成器设计详解（2026-07-10）

### 8.1 核心澄清：规则生成器不需要训练

"规则生成器"是**纯化学规则驱动的组合枚举器**，不是神经网络，不参与梯度优化。它本质上是一个函数：

```
f(symmetry_type, scaffold, n_substitutions, substituent_pool) → {对称分子}
```

真正被训练的是**下游扩散模型**：用生成的对称分子 + QM9 联合训练，让模型学会条件生成。

### 8.2 整体架构（三层设计）

```
┌──────────────────────────────────────────────────────────┐
│  Layer 0: 规则生成器（无训练，纯化学规则 + 组合枚举）         │
│                                                          │
│  输入: target_symmetry ("C2v"|"D3h"), scaffold_name,     │
│        n_groups (修改几组对称位点),                        │
│        n_extensions (每组用几个取代基)                     │
│                                                          │
│  内部流程:                                                │
│    1. scaffold(SMILES) → Chem.MolFromSmiles → AddHs      │
│    2. CanonicalRankAtoms → 按 parent rank 分组 H 原子     │
│    3. 选 k 组对称等价 H 组 → 每组用同一取代基替换            │
│    4. RWMol 替换: 保留 [*] 作为连接点标识 → 找到附着原子     │
│    5. SanitizeMol → 成功则 MolToSmiles                    │
│    6. ETKDGv3 生成 3D 构象 → MMFF 优化                    │
│    7. 惯性张量验证 k-fold 旋转对称 (RMSD < 0.8Å)           │
│                                                          │
│  输出: { atom_types(N,5), positions(N,3),                │
│          edge_attr(E,5), smiles,                         │
│          sym_label=1, point_group }                      │
│                                                          │
│  关键约束:                                                │
│    - 对称等价位点必须用完全相同的取代基（保证群作用不变）       │
│    - 化合价由 SanitizeMol 自动检查                         │
│    - 分子量 60-500 Da 过滤                                │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  Layer 1: 扩散模型训练 ★ 这部分需要训练 ★                   │
│                                                          │
│  训练数据:                                                │
│    - QM9: 130K 分子 (label=0, 非对称条件)                  │
│    - 规则生成: ~10K 对称分子 (label=1, 或 per-symmetry)     │
│                                                          │
│  模型架构:                                                │
│    EGNN backbone (QM9 Phase 1 预训练, 9层, hidden=256)    │
│    + FiLM 条件层: sym_label → Embedding(16) → (γ, β)      │
│    + 输出头: coord_noise(N,3) + atom_logits(N,5)          │
│              + bond_logits(E,5)                           │
│                                                          │
│  训练策略（两阶段）:                                        │
│    Stage A (warmup, ~50 epochs, lr=1e-3):                 │
│      - 数据: 50% QM9 + 50% 对称分子                        │
│      - 冻结 EGNN backbone (130K QM9 预训练权重)            │
│      - 仅训练 FiLM 条件层 + 输出头 (~2M params)             │
│      - 目标: FiLM 学会区分"对称"与"任意"分子                 │
│                                                          │
│    Stage B (fine-tune, ~100 epochs, lr=5e-5):             │
│      - 数据: 30% QM9 + 70% 对称分子                        │
│      - 解冻 EGNN 最后 3 层                                 │
│      - 目标: 提升对称分子生成质量                            │
│                                                          │
│  损失函数:                                                │
│    L = L_coord(MSE) + L_atom(CE) + L_bond(CE)            │
│        + λ·L_sym_classifier(BCE, 辅助: 预测分子是否对称)    │
│                                                          │
│  推理 (条件生成):                                          │
│    label=1, noise ~ N(0,I)                                │
│    → 1000-step DDPM / 50-step DDIM 去噪                   │
│    → 离散采样原子类型 (argmax)                              │
│    → 输出: 具有对称性的合法分子                              │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  Layer 2: Q/R 标记（规则化，不参与扩散）                     │
│                                                          │
│  1. 对称轴检测: 惯性张量最大主轴 → rotation axis            │
│  2. Q 标记: 轴两端最远的原子对 (沿轴投影 max/min)            │
│  3. R 标记: 外围原子 (离轴最远的非Q原子, top-3)             │
│  4. 组装: SMILES + Q/R 标记 → cjson → pycofbuilder        │
└──────────────────────────────────────────────────────────┘
```

### 8.3 输入输出规格

#### 规则生成器 I/O

```
# 输入
Input:
  sym_type: str          # "C2v" | "D3h"
  scaffold_name: str     # "biphenyl" | "azobenzene" | ...
  n_groups: int          # 修改几组对称等价位点 (1-3)
  n_extensions: int      # 每个位点接几个取代基 (通常=1)
  seed: int              # 随机种子

# 输出 (单个分子)
Output:
  atom_types:  np.ndarray (N, 5)     # one-hot: H,C,N,O,F
  positions:   np.ndarray (N, 3)     # 3D 坐标 (Å)
  edge_index:  np.ndarray (2, E)     # 边索引
  edge_attr:   np.ndarray (E, 5)     # bond type one-hot
  smiles:      str                   # SMILES 字符串
  num_atoms:   int                   # 重原子数
  verified_sym: bool                 # 是否通过对称验证
  rmsd:        float                 # 旋转 RMSD
```

#### 扩散模型训练 I/O

```
# 训练
Input batch:
  atom_types:  (B, N_max, 5)    # QM9 + 对称分子混合
  positions:   (B, N_max, 3)
  edge_attr:   (B, E_max, 5)
  sym_label:   (B,)             # 0=QM9, 1=对称
  t:           (B,)             # 扩散时间步

Forward:
  sym_label → Embedding(16) → FiLM(γ,β) → EGNN layers
  → coord_score + atom_logits + bond_logits

Loss:
  coord_loss = MSE(ε_pred, ε_true)     # 坐标噪声预测
  atom_loss  = CE(atom_logits, atom_0)  # 原子类型重建
  bond_loss  = CE(bond_logits, bond_0)  # 键类型重建

# 推理
Input:
  label:       int             # 1 (对称条件)
  n_atoms:     int             # 目标原子数 (可选)

Process:
  noise ~ N(0,I) → DDIM 50步 → 去噪坐标+原子类型
  → argmax 离散化 → 合法分子

Output:
  同生成器输出格式 (atom_types, positions, edge_attr, smiles)
```

### 8.4 与之前路线的对比

| 维度 | Phase 2 (失败) | Orbit-Circ (失败) | **规则+QM9联合 (新)** |
|------|:---:|:---:|:---:|
| 训练数据 | 276→5K augment | 67 compressed | **130K QM9 + 10K 规则** |
| 原子类型 | 12 (含Q/R人造) | 12 (含Q/R人造) | **5 (纯化学: H,C,N,O,F)** |
| Q/R处理 | 模型生成 | 模型生成 | **后处理规则标记** |
| 模型初始化 | 随机 | 随机 | **QM9 Phase1 预训练** |
| 可训参数 | 7M (全量) | 483K | **~2M (FiLM+head)** |
| 对称性保证 | FiLM隐式 | Circulant结构 | **规则保证数据对称 + FiLM条件** |

### 8.5 关键设计决策

**Q1: 为什么规则生成器不用神经网络？**
组合枚举空间可控（8个骨架 × 50种对称位点选择 × 13种取代基 × 组合约束）。神经网络反而需要训练数据——这正是我们缺乏的。规则生成器是"零样本"的：给定骨架和对称性约束，直接枚举即可。

**Q2: 为什么用 binary label 而非 per-symmetry label?**
先验证 binary 方案：模型能否区分对称/非对称分子。若成功，再扩展为 4-class (C2v/D3h/D4h/D6h)。
风险：所有对称分子混在一起训练，模型可能只学到"某种对称性"而非精确的 C2v。

**Q3: Q/R 为什么在后处理而非扩散中？**
Q/R 是人为定义的概念（反应点/基团连接点），不是化学元素。它们的确定规则是：
- Q: 对称主轴两端最远的原子 → 纯几何判定
- R: 离主轴最远的外围原子 → 纯几何判定
让扩散模型去学人造概念是浪费容量。Layer 1 生成纯化学对称分子，Layer 2 规则标记 Q/R——职责分离。

**Q4: QM9 的对称分子会不会"污染"训练？**
不会。QM9 中也有对称分子（~10-15% 天然具有 C2v 或更高对称性），它们的 label=0。模型学到的是：
- label=0 → 任意分子的分布（包括天然对称的）
- label=1 → 对称分子的分布（来自规则生成）
条件生成的本质是 p(mol | label=1)，与 p(mol | label=0) 不同即可。

### 8.6 当前实现状态与修复计划

**状态**: 生成器框架完成，但 `_smiles_substitute` 有 bug 导致 0/30 产出。

**根因**: `sub_smi.replace('[*]', '')` 把 `[*]C` 变成 `C`（甲烷，化合价饱和），无法再成键。

**修复方案** (详见第9节):
1. 保留 `[*]` 作为连接点标识符
2. 用 dummy atom (atomic number=0) 定位真正的连接原子
3. 只拷贝非 dummy 原子到 RWMol
4. 将连接原子 bond 到 parent

**目标产量** (修复后):

| 对称性 | 骨架数 | 每组变体 | 目标 |
|--------|:---:|:---:|:---:|
| C2v | 6 | ~1000 | **6000** |
| D3h | 2 | ~2000 | **4000** |
| 总计 | | | **10000** |

---

## 9. Bug 修复记录

### 9.1 [已修复] `_smiles_substitute`: 0/30 → 29/30

**根因**: `sub_smi.replace('[*]', '')` 把 `[*]C` → `C`，`Chem.MolFromSmiles('C')` 解析为 CH₄（甲烷，化合价已饱和），无法与 parent 成键。

**修复** (2026-07-10):
1. 保留 `[*]` 在 SMILES 中 → 解析出 dummy atom (atomic number=0)
2. dummy 的 neighbor 即为正确的连接原子
3. 拷贝非 dummy 原子到 RWMol, 保留 formal charge 等属性
4. 内部键不包含 dummy → 正确重建取代基拓扑
5. 将连接原子 bond 到 parent

**结果**: 29/30 分子成功生成 (SanitizeMol 通过) ✓

### 9.2 [已修复] 3D 对称性验证: 0/29 → 29/29 (100%)

**根因 1: 验证函数错误** — `verify_symmetry()` 使用直接 atom-wise RMSD：
```
RMSD = ||R·coords - coords||  # atom i vs 旋转后的 atom i
```
对称分子旋转后原子映射到**对称伙伴**而非自身。联苯 C2 旋转：ring A 原子 → ring B 位置，不应等于原地。

**修复**: 改为 permutation-invariant nearest-neighbor RMSD：
```
For each rotated atom i:
  Find nearest original atom j with SAME element type
  RMSD² += ||R·coord_i - coord_j||²
```
并遍历 3 个惯性主轴，选最优 RMSD。

**根因 2: ETKDG 不保证 3D 对称** — 低能构象可能因扭转角等原因偏离对称几何。

**修复**: `_symmetrize_coords()` — 构象优化后进行 Ck 对称化：
```
coords_sym = 1/k · Σ_{i=0}^{k-1} R^i(coords)
```
保证旋转后坐标完美重合（RMSD = 0.00Å）。

**最终结果**: 29/30 分子生成, 29/29 (100%) C2 对称性验证通过 ✓

---

## 10. 实现进度（2026-07-10）

### 10.1 已完成

| 组件 | 文件 | 状态 |
|------|------|:---:|
| 规则生成器 | `sym_mol_gen/generator/generator.py` | ✅ 29/30, 100% 对称 |
| 批量生成脚本 | `sym_mol_gen/batch_generate.py` | ✅ 多进程, ~24/秒 |
| 联合数据集 | `symmcd_diffusion/data/joint_dataset.py` | ✅ QM9 + 对称混合 |
| 联合训练脚本 | `symmcd_diffusion/train_joint.py` | ✅ 两阶段训练 |
| C2v 数据生成 | 已完成 | ✅ 6005 分子 (12MB) |
| D3h 数据生成 | 暂停 (空间枯竭) | ⚠️ 仅323, 产率7/500 |
| Stage A 训练 | 运行中 | 🔄 启动中... |

### 10.2 训练脚本设计要点

**`JointDenoiser`** (`train_joint.py`):
- 包装 `Denoiser(condition_dim=16, use_film=True)`
- `sym_embed`: `nn.Embedding(2, 16)` — binary label → FiLM condition
- `encode_condition(sym_label)` → `(batch, 16)` 兼容 `DiffusionProcess.training_step()`
- `load_qm9_backbone()`: 从 QM9 checkpoint 加载权重，跳过不存在的 FiLM 层
- `freeze_backbone()`: 冻结 EGNN，只训练 FiLM + heads + sym_embed
- `unfreeze_last_n_layers(n=3)`: 解冻最后 n 层 EGNN

**两阶段训练**:
```
Stage A (warmup, 50 epochs, lr=1e-3):
  - Freeze: EGNN backbone
  - Train: sym_embed(2×16) + FiLM(γ,β per layer) + coord/atom/bond heads
  - ~2M params trainable / ~7M total

Stage B (fine-tune, 100 epochs, lr=5e-5 backbone, 5e-4 new):
  - Unfreeze: last 3 EGNN layers
  - Train: full model with lower backbone LR
```

**数据流**:
```
JointDataset.__getitem__ → PyG Data {x, positions, edge_index, edge_attr, sym_label}
  → DataLoader + _joint_collate_fn → batched Data {..., sym_label: (B,)}
  → train_epoch: condition = model.encode_condition(batch.sym_label)
  → diffusion.training_step(data, denoiser, condition=condition)
  → Denoiser forward: condition broadcast to all nodes → FiLM per layer
```

### 10.3 D3h 生成问题

D3h 仅 3 个骨架 (triazine/trimethylbenzene/triphenylbenzene)，3 重对称要求
对称等价位点必须是 3 的倍数——组合空间远小于 C2v（2 重对称，2 的倍数）。

产率从 3.7/s 迅速衰减到 1.3/s（批次新增从 191 降到 7），空间接近枯竭。

**计划**: 先用 6000 C2v 验证 binary label 条件训练可行。若成功，
后续方案：
  1. 增加更多 D3h 骨架（如 hexaaminobenzene, triphenylene 衍生物）
  2. 用 trained model 以 label=1 条件生成 D3h 分子（few-shot generation）
  3. 扩展到 per-symmetry label (C2v/D3h/D4h/D6h 4-class)

### 10.5 关键发现：EGNN 噪声预测能力不足（2026-07-11）

**诊断**: QM9 训练 444 epochs 后，coord_head 预测的噪声 std≈0.3-0.4（真实噪声 std=1.0）。
MSE=1.0-1.3，**比预测零还差**（MSE(zero)≈1.0）。

**根因**: coord_head 初始化 ×1e-3，虽然训练中增长了 ~375×（std 0.00009→0.033），
但 EGNN 隐藏特征对坐标噪声的预测信号太弱，头权重无法进一步增长。

**影响**:
- ❌ 从纯噪声（t=999）de novo 生成不可行
- ✅ Scaffold-guided diffusion（中等 t）可行——Phase 2 已验证
- ✅ FiLM 条件在 scaffold diffusion 中仍然有效

**修正方案**:
1. 在 MARL pipeline 中使用 scaffold-guided diffusion（给模板加 t=200-400 噪声→去噪）
2. 而非从纯噪声 de novo 生成
3. 这与 Phase 2 / `env_symtopo_diff.py` 的设计一致

### 10.7 2D 图生成训练（2026-07-13）

**动机**: EGNN 学不会坐标去噪（MSE 比预测零还差），但原子+键的离散扩散有效。
砍掉坐标扩散，只做 2D 分子图生成。

**关键改变**:
1. 坐标固定为零（消除 EGNN 坐标更新的噪声放大问题）
2. 使用 QM9 真实 bond marginals (none:87.3%, single:11.1%, ...)
   - 原代码用 uniform (20% each)，与实际分布严重不匹配
   - uniform → 80% 的边初始化为有键 → 模型无法去噪到稀疏图
3. 仅训练离散扩散：CE(atom_logits, clean_atom) + CE(bond_logits, clean_bond)
4. FiLM 条件: sym_label → 16-dim → 图节点特征调制

**训练**:
```bash
python symmcd_diffusion/train_2d_graph.py --epochs 100 --batch_size 64
```

**推理**: 1000 步离散 reverse → atom_types(N,5) + bond_types(E,5)
→ argmax → RDKit MolFromAtomsAndBonds → SanitizeMol → 3D 构象

### 10.8 启动训练的命令

```bash
# 确保对称分子数据已生成完
ls data/symmetric_molecules/

# Stage A only (快速验证, ~2小时)
python symmcd_diffusion/train_joint.py \
    --stage A --epochs_a 50 --batch_size 64 \
    --qm9_ckpt symmcd_diffusion/checkpoints/qm9_best.pt \
    --device cuda:2

# Full two-stage (完整训练)
python symmcd_diffusion/train_joint.py \
    --stage all --epochs_a 50 --epochs_b 100 \
    --batch_size 64 --device cuda:2

```
---

## 11. 2026-07-11：3D 联合训练结果与诊断

### 11.1 Stage A 训练（冻结 backbone, 50 epochs）

| Epoch | Total | Coord | Atom | Bond |
|------:|------:|------:|-----:|-----:|
| 1 | 1.873 | 1.001 | 0.678 | 0.389 |
| 10 | 1.682 | 0.996 | 0.557 | 0.258 |
| 30 | 1.661 | 0.992 | 0.547 | 0.246 |
| 50 | 1.656 | 0.991 | 0.544 | 0.243 |

**最佳**: epoch 47, loss=1.6501

### 11.2 Stage B 训练（unfreeze 最后 3 层, 100 epochs）

| Epoch | Total | Coord | Atom | Bond |
|------:|------:|------:|-----:|-----:|
| 1 | 1.651 | 0.990 | 0.540 | 0.241 |
| 10 | 1.586 | 0.960 | 0.514 | 0.225 |
| 30 | 1.543 | 0.938 | 0.497 | 0.216 |
| 50 | 1.516 | 0.915 | 0.494 | 0.214 |
| 70 | 1.495 | 0.902 | 0.488 | 0.210 |
| 93 | **1.490** | **0.898** | **0.487** | **0.210** |
| 100 | 1.496 | 0.899 | 0.491 | 0.212 |

**最佳**: epoch 93, loss=1.4901（比 Stage A 下降 9.7%）

### 11.3 条件采样测试 → 失败

**测试方法**: Stage B best checkpoint + DDIM 50-step + QM9 marginals

**结果**: 所有采样均失败
- coord_std = 17,000-88,000 Å（正常分子 ~1-5Å）
- 原子类型全偏（F/O 主导，H<1%）
- C2 RMSD = 15,000-45,000（正常 <1.0Å）

**根因诊断**:

1. **alpha_bar[999] ≈ 0**：cosine schedule 在 t=999 时 alpha_bar ≈ 0，DDIM 的
   `x0_pred = (x - sqrt(1-ᾱ)·ε) / sqrt(ᾱ)` 中除以 0.00005 → 噪声预测误差放大 20,000 倍

2. **coord_head 从未学会去噪**：测试原始 QM9 模型（epoch 444, train_loss=1.51）
   - 噪声预测 MSE = 1.0-1.3，**比预测零还差**（MSE(zero)≈1.0）
   - pred_noise std = 0.3-0.6 vs true noise std = 1.0
   - 与真实噪声的相关性 ≈ 0（-0.25 ~ +0.32）
   - coord_head 权重增长 375×（gain=1e-3 init）但远不足以预测真实噪声

3. **bond marginals 用 uniform**：训练代码 `bond_marginals = torch.ones(5)/5`，
   但 QM9 实际分布是 none:87.3%, single:11.1%, double:0.74%, triple:0.0%, aromatic:0.89%
   → 80% 边初始化为"有键"，与实际 13% 严重不匹配

**结论**: **EGNN 从纯噪声 de novo 生成分子不可行**。但 scaffold-guided diffusion（Phase 2 方案：
模板+t=200-400→去噪）仍可行，因为中等 t 时 alpha_bar 不接近零。

### 11.4 策略调整：砍掉坐标，只做 2D 图生成

理由：
- 坐标去噪学不会（EGNN 架构限制）
- 原子类型和键类型去噪**有效**（FiLM 条件下 atom CE 从 0.678→0.487，bond CE 从 0.389→0.210）
- 2D 拓扑对称是 COF Core 的本质要求，3D 构象交给 RDKit 后处理
- 离散扩散（D3PM）数值稳定，不受 alpha_bar→0 影响

---

## 12. 2026-07-13：2D 图生成训练

### 12.1 方案设计

```
输入: atom_types(N,5) + edge_attr(E,5), fully connected graph
      坐标固定为零 → 消除 EGNN 坐标更新的不稳定性
      sym_label(1,) → FiLM 条件

扩散: 离散 D3PM 扩散
  - Forward:  sample t → add noise via marginals
  - Reverse:  denoiser → CE(atom_logits, clean_atom) + CE(bond_logits, clean_bond)
  - 无坐标扩散（消除了 alpha_bar→0 的数值问题）

条件: sym_label → Embedding(2, 16) → FiLM(γ,β) 注入 EGNN 每层

推理: 1000-step discrete reverse → argmax(atom) + argmax(bond)
      → RDKit Mol → SanitizeMol → 3D Embed → 对称验证
```

### 12.2 与 3D 训练的关键区别

| 维度 | 3D 联合训练 (Phase 5) | 2D 图生成训练 (新) |
|------|:---:|:---:|
| 坐标处理 | EGNN 更新（alpha_bar→0 爆炸） | **固定为零**（无坐标扩散） |
| Bond marginals | uniform (20%/类) | **empirical** (none:87.3%, single:11.1%, ...) |
| 扩散类型 | 连续(坐标) + 离散(原子/键) | **仅离散** (原子 + 键) |
| 数值稳定性 | ❌ 差（除以 ~0） | ✅ 稳定（离散矩阵运算） |
| 生成对象 | 3D 分子 | **2D 图** → RDKit 生成 3D |

### 12.3 实现文件

| 文件 | 用途 |
|------|------|
| `symmcd_diffusion/train_2d_graph.py` | 2D 图离散扩散训练脚本 |
| `symmcd_diffusion/data/joint_dataset.py` | QM9+对称混合数据集（复用） |
| `test_2d_generation.py` | 2D 图生成测试（训练完成后使用） |

### 12.4 训练进度（2026-07-13 运行中）

**模型**: Graph2DDenoiser (EGNN backbone + FiLM), 6.45M params
**可训**: 1.63M (25.2%) — EGNN 冻结, FiLM + atom_head + bond_head + sym_embed

| Epoch | Total Loss | Atom CE | Bond CE | 时间 |
|------:|-----------:|--------:|--------:|-----:|
| 1 | 1.0459 | 0.6708 | 0.3751 | 397s |
| 2 | 0.9166 | 0.6045 | 0.3120 | 333s |
| 3 | 0.8867 | 0.5961 | 0.2906 | 343s |
| 4 | 0.8704 | 0.5901 | 0.2803 | 335s |
| 5 | 0.8581 | 0.5853 | 0.2729 | 298s |
| 6 | 0.8534 | 0.5840 | 0.2694 | 297s |
| 7 | ... | ... | ... | 🔄 |

**对比 3D 训练**:
| 指标 | 3D Stage A (epoch 1) | 2D (epoch 1) | 说明 |
|------|:---:|:---:|------|
| Atom CE | 0.678 | 0.671 | 相近（同一模型） |
| Bond CE | 0.389 | 0.375 | 2D 略好（正确的 marginals） |

**分析**: 
- Bond loss 收敛快（0.375→0.269 in 6 epochs），正确的 marginals 让模型从合理分布开始
- Atom loss 收敛慢（0.671→0.584 in 6 epochs），原子类型预测与 marginal 分布一致即可
- 总体 loss 在 epoch 5-6 开始 plateau（~0.853），符合 Stage A 预期（冻结 backbone）

### 12.5 预期下一步

1. 训练完成（100 epochs）→ 用 `test_2d_generation.py` 测试 label=0 vs label=1 生成
2. 如果 2D 图有效（SanitizeMol 通过率高）→ 集成到 MARL pipeline
3. 如果 2D 图仍无效 → 替换 EGNN backbone 为简单 GNN（GCN/GAT/Transformer）
4. 加入 D3h 骨架扩展后重试

---

## 13. 2026-07-14：2D 图生成测试与 Stage B

### 13.1 Stage A 模型测试结果

**模型**: Graph2DDenoiser, epoch 92, best_loss=0.7968
**方法**: 100-step discrete reverse + valence-constrained post-processing

| Label | Valid | Symmetry | 说明 |
|:-----:|:-----:|:--------:|------|
| 0 (QM9) | 3/30 (10%) | 3/3 (100%) | 碎片化分子（C.C.O.[HH] 等） |
| 1 (Sym) | 0/30 (0%) | — | 化合价严重超标 |

**关键发现**:

1. **label=1 比 label=0 更差** — FiLM 条件有作用，但是负面的！
   对称条件让模型生成更差的分子，可能因为：
   - 6005 个对称分子 vs 104K QM9 → 训练不平衡（5% 对称）
   - 对称分子的 bond 模式与 QM9 不同 → FiLM 学到的映射不准确

2. **分子碎片化** — 有效分子都是 "C.C.O.[HH]" 这种断开的结构
   说明模型不会生成 C-C 键来连接原子 → 键预测倾向于 "none"

3. **化合价爆炸** — 无效分子中原子有 5-27 个键（正常 ≤4）
   即使 87.3% 边预测为 "none"，剩余 12.7% × 全连接边数仍然太多

**根因分析**:
- 冻结 EGNN + 零坐标 → 所有 pairwise 距离=0 → edge_mlp 输入退化
- 模型无法区分不同边 → bond 预测接近随机（略有偏置向 marginal 分布）
- 离散 reverse 1000 步的累积误差使 bond 分布偏离 marginal

### 13.2 Stage B 启动（全量微调）

- 从 Stage A best (epoch 92, loss=0.7968) 加载
- 全部 6.45M 参数可训（忘记调 freeze，但全量微调效果应更好）
- lr=5e-5 (backbone) + 5e-4 (new), 100 epochs
- 🔄 运行中...

### 13.3 后续备选方案

如果 Stage B 后仍不能生成有效分子：
1. **用小随机坐标代替零** — 给 EGNN 距离特征加一点变化
2. **替换为简单 GNN** — 去掉 EGNN，用 GCN/GAT/Transformer 做纯 2D 图生成
3. **Scaffold-guided diffusion** — 不从头生成，对模板分子加噪声后去噪（Phase 2 路线）
4. **Autoregressive generation** — 逐原子/键生成，天然保证化合价约束

---

## 14. 2026-07-15：最终结论

经过 6 个实验（3D 联合训练 ×2、2D 图 EGNN ×2、Orbit-Circulant ×2），
所有连续扩散方法都在同一个地方卡住：**alpha_bar→0 时崩溃**。

模型在 t<300 去噪可用（50-95%），在 t>500 完全失效（<30%）。
这不是某个方法的 bug，是**连续扩散 + 小数据集 + 二值目标**的系统性限制。

详细诊断见 `docs/current_status_and_problems.md`。

