# 对称分子扩散模型——技术路线总结与卡壳点

**日期**: 2026-07-15  
**目的**: 记录当前所有尝试过的技术路线、失败根因、可用资产，为切换新方法做准备

---

## 一、目标

训练一个条件生成模型：给定对称类型 → 生成具有该对称性的合法 2D 分子图 → RDKit 给出 3D 构象 → Q/R 标记 → 用于 MARL COF 设计。

---

## 二、已尝试的路线

### 路线表

| # | 方法 | 扩散类型 | 架构 | 结果 |
|:--:|------|:--------:|------|:---:|
| 1 | **3D 联合训练** (QM9+Sym) | 混合连续-离散 | EGNN + FiLM | ❌ |
| 2 | **2D 图 EGNN** (离散扩散) | 离散 D3PM | EGNN + FiLM | ❌ |
| 3 | **Orbit-Circulant VP-SDE** | 连续 VP-SDE | Graph Transformer | ❌ |
| 4 | **Orbit-Circulant + Dequantization** | 连续 VP-SDE | Graph Transformer | ❌ |
| 5 | **Scaffold Diffusion** (Phase 2) | 混合连续-离散 | EGNN | ✅ |

---

## 三、每条路线的详细诊断

### 路线 1：3D 联合训练（QM9 + FiLM 条件）

**方法**: 复用 QM9 Phase 1 的 EGNN denoiser，加 FiLM 条件层（sym_label → 16-dim → γ,β），
在 QM9 (104K, label=0) + 规则生成 C2v 分子 (4.8K, label=1) 上训练。

**训练结果**:
- Stage A (冻结 backbone): best loss 1.6501
- Stage B (unfreeze 最后 3 层): best loss 1.4901
- Coord loss 几乎不动: 1.001 → 0.899

**卡壳点——根本性架构问题**:

1. **Coord_head 从未学会去噪坐标**
   - QM9 444 epochs 训练后，噪声预测 MSE = 1.0（**比预测零还差**）
   - 预测噪声与真实噪声相关性 ≈ 0（-0.25 ~ +0.32）
   - pred_noise std = 0.3-0.6 vs true noise std = 1.0
   
2. **ᾱ_999 ≈ 0 导致数值爆炸**
   - Cosine schedule: `x0_pred = (x - √(1-ᾱ)·ε_pred) / √(ᾱ)` 
   - 在 t=999: 除以 √(0.00005) ≈ 1/20000 → 噪声预测误差放大 20000 倍
   - 生成坐标 std = 17,000-88,000 Å（正常分子 1-5 Å）
   
3. **Bond marginals 不匹配**
   - 训练代码用 uniform (20%/类)，实际分布: none 87.3%
   - 80% 边初始化为"有键" → 离散 reverse 无法收敛到稀疏图

**根因**: **EGNN 的 equivariant coordinate update 机制不适合从纯噪声去噪坐标。**
这是架构级别的限制，不是调参能解决的。

**相关代码**:
- `symmcd_diffusion/train_joint.py`
- `symmcd_diffusion/data/joint_dataset.py`
- Checkpoint: `symmcd_diffusion/checkpoints/joint/stageB_best.pt`

---

### 路线 2：2D 图 EGNN（离散扩散，砍掉坐标）

**方法**: 砍掉坐标扩散，只用离散 D3PM 做 atom + bond 生成。
坐标固定为零 → EGNN 距离特征退化（所有 pairwise 距离=0），依赖 FiLM 条件 + 节点特征做预测。

**训练结果**:
- Stage A (冻结 backbone): best loss 0.7968
- Stage B (全量微调): best loss 0.7248
- Bond CE 从 0.375 → 0.216

**卡壳点——生成完全失败**:

1. **label=0 有效 10%，label=1 有效 0%**
2. 有效分子是碎片化的（"C.C.O.[HH]"——断开的多组分）
3. 无效分子原子化合价爆炸（C 有 5-27 个键）
4. **FiLM 条件有负面效果**——label=1 比 label=0 更差

**根因**:
- 零坐标 → 所有 pairwise 距离相同 → edge_mlp 无法区分边 → bond 预测接近随机
- 离散 reverse 1000 步 × 每步 ~20% 错误率 → bond 分布严重偏离
- 冻结 backbone 限制了 bond_head 学习能力；全量微调仍然不够

**相关代码**:
- `symmcd_diffusion/train_2d_graph.py`
- Checkpoint: `symmcd_diffusion/checkpoints/2d_graph/stageB_best.pt`

---

### 路线 3 & 4：Orbit-Circulant VP-SDE

**方法**: 基于 apart_diffusion 思路。检测分子旋转对称轴 → 原子划分 orbit → 
邻接矩阵压缩为块循环矩阵（circulant basis C ∈ R^(M×M×k)）→ VP-SDE 扩散去噪。

**数据**: 6005 个 C2v 分子 → 5404 个压缩训练样本 (M=13-32, k=2)

**训练结果**:
- 200 epochs: best loss 0.8478
- Dequantization 微调 50 epochs: best loss 0.8592（无改善）

**卡壳点——和路线 1 完全相同的模式**:

| t | x0_std | % in [0,1] |
|--:|-------:|:----------:|
| 0 | 0.25 | 100% ✅ |
| 100 | 0.29 | 95% ✅ |
| 300 | 0.54 | 51% ⚠️ |
| 500 | 1.01 | 33% ❌ |
| 700 | 1.88 | 22% ❌ |
| 900 | 5.82 | 11% ❌ |
| 999 | 780.90 | 0% ❌ |

模型在 t≤300 能有效去噪，在 t>500 完全崩溃。Dequantization（GDSS 的标准做法）无效。

**根因**: 
- Circulant basis 是二值的（0/1），VP-SDE 的高斯扩散在 t>500 时完全淹没二值结构
- 和路线 1 的坐标去噪是同一个问题——**连续扩散假设与二值/结构化数据不兼容**
- 训练 loss 0.85 是误导性的——主要来自低 t 的易预测部分

**这个路线证明了**: 即使完美利用对称性的数学结构（orbit-circulant 压缩），连续扩散仍然无法从纯噪声恢复二值图结构。

**相关代码**:
- `orbit_diffusion_v2/` (preprocessing, models, train.py, generate.py)
- Checkpoint: `orbit_diffusion_v2/checkpoints/best.pt`

---

### 路线 5：Scaffold Diffusion（Phase 2）——唯一能工作的方案

**方法**: 不做 de novo 生成。取一个模板分子，加中等噪声（t=200-400），模型去噪 → 结构微调。
FiLM 条件控制微调方向。

**为什么能工作**: 模型在 t≤300 时去噪有效（重建率 50-95%），scaffold diffusion 恰好只需要这个范围。

**相关代码**: `mappo/env_symtopo_diff.py`

---

## 四、所有路线的共性根因

```
连续扩散（DDPM/VP-SDE）+ 小数据集 + 结构化/二值目标
                    ↓
          alpha_bar → 0 at t > 500
                    ↓
           除以 sqrt(alpha_bar) ≈ 1/20000
                    ↓
       噪声预测微小误差被放大 20000 倍
                    ↓
              生成完全崩溃
```

**这不是任何一个方法的 bug**。这是**连续扩散模型在有限数据下的系统性限制**。
大模型（Stable Diffusion、DALL-E）能工作是因为：
1. 海量数据（数亿样本）
2. 模型容量极大（数亿参数）
3. 目标分布是"自然图像"（连续、平滑），而不是二值图/离散结构

---

## 五、当前资产

### 数据

| 数据集 | 位置 | 大小 | 用途 |
|--------|------|------|------|
| QM9 处理数据 | `symmcd_diffusion/data/qm9/processed/` | 325 MB | 化学合法性基础 |
| C2v 对称分子 (原始) | `data/symmetric_molecules/C2v_6005.npz` | 12 MB | 6005 个 SMILES+3D |
| C2v orbit 压缩 (train) | `orbit_diffusion_v2/data/c2v_train.npz` | 8.3 MB | 5404 个压缩样本 |
| C2v orbit 压缩 (val) | `orbit_diffusion_v2/data/c2v_val.npz` | 933 KB | 601 个压缩样本 |

### 模型 checkpoint

| 文件 | 用途 |
|------|------|
| `symmcd_diffusion/checkpoints/qm9_best.pt` | QM9 预训练 backbone（所有训练的起点） |
| `symmcd_diffusion/checkpoints/joint/stageB_best.pt` | 3D 联合训练最佳 |
| `symmcd_diffusion/checkpoints/2d_graph/stageB_best.pt` | 2D 图 EGNN 最佳 |
| `orbit_diffusion_v2/checkpoints/best.pt` | Orbit-circulant 最佳（t<300 可用） |

### 基础设施

| 组件 | 位置 | 功能 |
|------|------|------|
| 规则分子生成器 | `sym_mol_gen/` | Scaffold → 对称取代 → 3D → 验证 |
| Orbit 检测+压缩 | `orbit_diffusion_v2/preprocessing/` | 分子 → orbits → circulant basis |
| 联合数据集 | `symmcd_diffusion/data/joint_dataset.py` | QM9 + 对称分子混合 |
| Orbit 数据集 | `orbit_diffusion_v2/train.py` (OrbitDataset) | 压缩样本加载 |
| MARL 环境 | `mappo/env_symtopo*.py` | 对称+拓扑 MARL 环境 |

---

## 六、关键教训

1. **不要用连续扩散生成离散结构**——D3PM/离散扩散更适合分子图

2. **EGNN 不是通用去噪器**——它是为 3D 坐标 E(n) 等变设计的，不应强行用于 2D 图

3. **Symmetry 应该"编译"进表示层，不是靠条件"学习"**——orbit 压缩的方向是对的（数学保证对称），但扩散框架选错了

4. **不要只看训练 loss**——loss 0.85 看起来 OK，但模型只是学会了在低噪声区预测均值，从未真正学会去噪

5. **Bond marginals 必须匹配数据分布**——uniform (20%) vs 实际 (87% none) 的差距是致命的

6. **De novo 生成远难于 conditional modification**——scaffold diffusion 只需要 t<400 去噪，这是模型能做到的

7. **apart_diffusion 能工作 ≠ 简单复现就能工作**——师弟的数据规模、模型细节、训练技巧可能都是关键差异

---

## 七、建议的下一步方向

1. **离散扩散（D3PM）替代 VP-SDE**——对二值 circulant basis 使用离散扩散
2. **更大规模的 GDSS-style 训练**——更多数据 + 更强的 score network
3. **直接使用规则生成分子**——6005 个 C2v 分子作为 MARL 的 Core 池子，跳过扩散生成
4. **Scaffold diffusion 集成**——把 orbit-circulant 模型用于 scaffold-guided generation（t=200-400）
