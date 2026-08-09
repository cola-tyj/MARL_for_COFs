# De Novo 生成改进调研：数据稀缺下的对称性分子生成

> 2026年6月24日 | 针对 83 Core 数据不足时的模型改进方案

---

## 一、问题重述

**现状**：83 个独特 Core，12 种原子类型。离散扩散模型在 t>500 后原子类型预测崩溃（准确率 43-48%），无法从纯噪声 (t=999) 生成。
**根因**：12 类 × 1000 噪声级 × 83 样本 = 每个 (类型, 噪声级) 组合平均只有 ~7 个训练样本。

---

## 二、公开数据集调研

### 2.1 现有分子数据集

| 数据集 | 规模 | 原子类型 | 对称性标签 | COF 相关性 |
|------|:---:|:---:|:---:|:---:|
| **QM9** | 130K | H/C/N/O/F (5) | ❌ 无 | 🟡 小分子，无连接点 |
| **GEOM** | 37M conformers | H/C/N/O/F/S/Cl (7) | ❌ 无 | 🟡 药物分子，可计算点群 |
| **ZINC20** | 1.4B | H/C/N/O/F/S/Cl/Br/I/P (10) | ❌ 无 | 🟡 可筛选特定骨架 |
| **PubChem** | 110M | 全元素 | ❌ 无 | 🟡 需大量过滤 |
| **CSD** | 1.2M | 全元素 | ✅ 空间群 | 🔴 晶体，非分子 |
| **COD** | 480K | 全元素 | ✅ 空间群 | 🔴 晶体，非分子 |
| **ChEMBL** | 2.4M | 全元素 | ❌ 无 | 🟡 生物活性分子 |

### 2.2 评估结论

**没有现成的"对称性分子数据集"**。最接近的选择：

1. **GEOM**：37M 药物分子 conformers → 用我们的 `point_group.py` 计算点群 → 筛选特定对称性的分子 → 作为 Core 训练数据补充
2. **CSD/COD**：晶体数据有空间群 → 但晶体≠孤立分子，需提取非对称单元
3. **ZINC20**：1.4B 分子 → 用 RDKit 筛选含特定子结构的分子 → 计算点群

**推荐路径**：用 GEOM 子集（L2/T3/S4/H6 对称性分子）作为 Core 训练数据增强。

---

## 三、模型改进方案

### 3.1 方案 A：课程学习 + 渐进式原子类型

**核心思想**：从少类型到多类型，逐步扩展。

```
阶段 1: QM9 (5 类型) → 学习原子类型去噪 (130K 数据)
阶段 2: QM9 + 筛选的 GEOM 分子 (5→7 类型) → 增加 S, Cl
阶段 3: Core 数据 (7→12 类型) → 增加 Q, R, 金属
```

**优势**：利用大数据集打好基础，小数据集微调
**劣势**：需要处理 GEOM 数据（37M 太大，需筛选）

### 3.2 方案 B：坐标→原子类型 条件预测

**核心思想**：先扩散生成坐标（连续，易训练），再从坐标预测原子类型。

```
Stage 1 (Coord Diffusion):    噪声 → 3D 坐标 (模型擅长，MSE)
Stage 2 (Type Prediction):    3D 坐标 → 原子类型 (监督学习，非扩散)
```

**Stage 2 的几种实现**：

| 方法 | 说明 | 优势 |
|------|------|------|
| GNN 分类器 | 对每个原子做 12 类分类 | 简单，可复用 EGNN |
| 基于距离+元素半径 | 坐标→键长→元素推断 | 化学规则，无需训练 |
| Template matching | 最近邻 Core 模板匹配 | 利用 83 Core 模板 |

**优势**：分离连续和离散生成，各用所长
**劣势**：两阶段，原子类型可能不匹配坐标

### 3.3 方案 C：Self-Training / 自举

**核心思想**：用现有模型生成数据 → 筛选高质量生成 → 加入训练集。

```
1. 坐标精修法生成 1000 Core (100% atom acc)
2. 对每个 Core，用不同噪声级 (t=200~999) 加噪 → (noisy_input, clean_label) 对
3. 加入训练集 → 微调模型
4. 重复 3-5 轮
```

**优势**：不需要外部数据，自我改进
**劣势**：可能放大模型偏差

### 3.4 方案 D：对比学习 + 对称性条件增强

**核心思想**：同一对称性的 Core 应有相似的原子类型分布。

```
L_contrastive = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))

z_i, z_j: 同一对称性两个 Core 的嵌入
z_k: 不同对称性 Core 的嵌入
```

**优势**：利用对称性标签作为监督信号
**劣势**：需要设计对比学习架构

### 3.5 方案 E：GEOM 数据筛选 + 对称性标签

**具体步骤**：

```python
1. 从 GEOM 下载 drug-like 分子 (可选子集: ~100K)
2. 用 point_group.py 计算每个分子的点群
3. 筛选目标对称性的分子 (L2/C2v, T3/D3h, S4/D4h, H6/D6h)
4. 添加 Q/R "虚拟原子" 到连接点位置 (基于分子形状自动推断)
5. 构建增强训练集 (预期: 每个对称性 500-2000 个分子)
6. 重新训练 Phase 2
```

**预期效益**：83 → 2000+ 独特 Core，原子类型去噪应显著改善。

---

## 四、方案对比与推荐

| 方案 | 预期改善 | 实现难度 | 时间 | 推荐 |
|------|:---:|:---:|:---:|:---:|
| A: 课程学习 | ★★★ | 中 | 2天 | 🟡 |
| B: 坐标→类型 | ★★★★ | 中 | 1天 | 🔴 **首选** |
| C: Self-Training | ★★★ | 中 | 1天 | 🟡 |
| D: 对比学习 | ★★ | 高 | 3天 | 🟢 |
| E: GEOM 数据 | ★★★★★ | 高 | 3天 | 🟡 |

**推荐 B（坐标→类型预测）+ C（Self-Training）组合**：

1. **B 阶段 1**：扩散生成坐标（已验证有效）
2. **B 阶段 2**：训练 GNN 分类器 (坐标→原子类型)，用 83 Core + 增强数据
3. **C**：用 B 的输出作为伪标签 → 自举改进

## 五、参考文献

1. Hoogeboom, E., et al. (2022). "Equivariant diffusion for molecule generation in 3D." *ICML*. — EDM, 连续坐标扩散
2. Vignac, C., et al. (2023). "MiDi: Mixed graph and 3D denoising diffusion for molecule generation." *ICML*. — 混合连续-离散扩散
3. Xu, M., et al. (2022). "GeoDiff: A geometric diffusion model for molecular conformation generation." *ICLR*.
4. Axelrod, S., & Gomez-Bombarelli, R. (2022). "GEOM: Energy-annotated molecular conformations." *Scientific Data*.
5. Ramakrishnan, R., et al. (2014). "Quantum chemistry structures and properties of 134 kilo molecules." *Scientific Data*. — QM9
6. Irwin, J. J., et al. (2020). "ZINC20 — A free ultralarge-scale chemical database for ligand discovery." *JCIM*.
7. Groom, C. R., et al. (2016). "The Cambridge Structural Database." *Acta Cryst*. — CSD
8. Chen, T., et al. (2020). "A simple framework for contrastive learning of visual representations." *ICML*. — SimCLR
9. Xie, Q., et al. (2020). "Self-training with noisy student improves ImageNet classification." *NeurIPS*.
10. Lee, J., et al. (2013). "Pseudo-label: The simple and efficient semi-supervised learning method." *ICML Workshop*.
