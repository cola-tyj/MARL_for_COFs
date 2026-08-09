# 核心困难深度分析与解决方案调研

> 2026年6月25日

---

## 困难全景图

```
                    ┌─────────────────┐
                    │   MARL Agent     │
                    │ (探索 Core+Conn)  │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │ 困难1:       │ │ 困难2:       │ │ 困难3:       │
     │ De novo 不可行│ │ COF 组装率低  │ │ 实时评估太慢  │
     │ 83 Core 不足  │ │ 8-31%        │ │ 3-5s/COF     │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                │
            ▼                ▼                ▼
     ┌──────────────────────────────────────────────┐
     │              核心矛盾                        │
     │  MARL 需要大探索空间 → 但扩散无法生成新分子     │
     │  MARL 需要真实奖励   → 但预测器太慢不能用       │
     │  查表可加速         → 但限制了探索空间          │
     └──────────────────────────────────────────────┘
```

---

## 困难 1：扩散模型 De Novo 生成不可行

### 问题本质

83 个独特 Core 不足以训练离散扩散模型从纯噪声恢复 12 类原子类型。坐标去噪（MSE）容易，原子类型去噪（CrossEntropy, 12类）在 t>500 后崩溃。

### 实验证据

| 噪声级 | 原子预测准确率 | 说明 |
|:---:|:---:|------|
| t=200 | 100% | 完美恢复 |
| t=500 | 78% | 部分恢复 |
| t=800 | 43% | 接近随机 |
| t=999 | 48% | 坍缩为 H |

Uniform prior 重训（500 epochs）未能改善。

### 根因分析

离散扩散的反向过程 p(a_{t-1}|a_t) 是分类问题。12 类 × 1000 噪声级 × 83 Core = 每 (类别, 噪声级) 组合平均 ~7 个训练样本。对比：QM9 Phase 1 有 130K 样本 × 5 类 = 足够。

### 解决方案调研

| 方案 | 原理 | 可行性 | 时间 |
|------|------|:---:|:---:|
| **A: 数据扩充** | 外部数据集/合成数据 → 增加 Core 数量 | 中 | 2-5天 |
| **B: 监督解耦** | 坐标扩散 + 坐标→类型预测器 | **高** | 2天 |
| **C: 连续松弛** | 用连续嵌入替代离散类型，避免 D3PM | 低 | 1周+ |
| **D: 少样本学习** | Meta-learning / MAML 适配离散扩散 | 低 | 2周+ |

#### 方案 A 详析：数据扩充

**A1: GEOM 数据集筛选**
- GEOM: 37M 药物分子 conformers
- 计算点群 → 筛选目标对称性 → 添加 Q/R 虚拟原子
- 预期：数千~数万新 Core
- 难点：Q/R 原子放置需化学合理性

**A2: CSD (Cambridge Structural Database)**
- 1.2M 晶体结构，含空间群
- 提取非对称单元 → 分子片段
- 空间群 ↔ 分子点群映射
- 优势：天然有对称性标签

**A3: 合成数据生成 (pycofbuilder 反向)**
- pycofbuilder 可从 Core 生成 BB
- 逆过程：从 BB 提取 Core 骨架
- 但 BB 本就是 Core 生成的——循环论证
- 替代：用 pycofbuilder 的 Core 模板库 + 化学规则生成变体

#### 方案 B 详析：监督解耦（推荐）

```
阶段 1 (连续): 扩散生成 3D 坐标 (已有能力)
阶段 2 (离散): 监督模型从 3D 坐标预测原子类型

关键：阶段 2 不需要扩散——是标准分类问题，可用更多数据
```

**B 的实现要点**：
- 训练数据：83 Core + 坐标精修变体 (每个 Core 可生成 100+ 坐标变体)
- 模型：EGNN 分类器，输入 (坐标, 边) → 输出原子类型
- 83 × 100 = 8,300 训练样本 vs 83 原始 Core
- **预期：原子类型预测准确率从 43% → 80%+**

**B 的优势**：
- 阶段 1 已验证（坐标扩散从噪声可行）
- 阶段 2 是监督学习，不需要扩散
- 数据增强容易（坐标精修法已有）
- 与现有架构兼容

#### 方案 C, D：高难度，超出硕士范围

---

## 困难 2：COF 组装成功率低

### 问题本质

即使拓扑-对称性兼容（action masking 保证），pycofbuilder 的 COF 组装仍大量失败（键长冲突、连接性错误）。

### 各版本成功率

| 版本 | 动作空间 | 成功率 |
|:---:|------|:---:|
| v1 预验证 | 55 组合 | 72% |
| v5 conn+FG | 6×15×31 | 13% |
| v4b diversity | 6×6 | 8% |

### 根因

pycofbuilder 的 `from_name()` 组装 BB 时，并非所有 Core-Connector-FG 组合都产生预期连接性。某些 Core 含特殊结构（如 BENZ 是通用模板），与 COOH 组装后连接性不符。

### 解决方案

| 方案 | 做法 | 效果 |
|------|------|------|
| **过滤无效 Core** | valid_cores.json (78/83) | 轻微改善 |
| **过滤无效 conn+FG** | 测试 15×31 组合的 assembly 成功率 | 预期提升到 30-50% |
| **使用预组装 BB** | 跳过 from_name，直接用 pycofbuilder 的已组装 BB | 100% 但失去 Core 选择自由度 |
| **宽松组装参数** | Framework(dist_threshold=0.3) | 更多 COF 组装成功 |

---

## 困难 3：实时 N₂ 评估

### 问题本质

PMTransformer 预测需 3-5s/COF，RL 2000 episodes 需 3h。预计算查表限制探索空间，surrogate 模型特征不足。

### 解决方案

| 方案 | 原理 | 可行性 |
|------|------|:---:|
| **E: 导出图嵌入** | 获取 GCNN 中间层输出作为特征 | **高** |
| **F: 分批评估** | 训练 N 步，批量预测 N₂ | 中 |
| **G: 离线 RL** | 用历史数据训练，不需要实时奖励 | 高 |

#### 方案 E 详析（推荐）

PMTransformer 内部有 GCNN → 图嵌入 (768-dim)。如果导出这个嵌入：
- 70 个标注 COF 的图嵌入 + N₂ 值 → 训练 MLP regressor
- 新 COF → GCNN 提取嵌入 → MLP 预测 → <100ms
- 需要修改预测器代码以导出嵌入

#### 方案 G 详析

离线 RL (Batch RL / Offline RL)：
- 用 Agent 探索收集的 (state, action, reward) 数据
- 无需在线交互即可学习策略
- CQL (Conservative Q-Learning) 等算法专为此设计
- 适合我们的场景：COF 生成慢，但可批量生成大量数据

---

## 推荐路线图

### 短期（1-2天）— 立即可行

1. **方案 B 实施**：坐标→类型监督预测器
   - 坐标精修生成 8,300 个 (坐标, 原子类型) 训练对
   - 训练 EGNN 分类器
   - 测试两阶段 de novo 生成

2. **方案 E 实施**：导出图嵌入 + surrogate
   - 修改预测器导出 GCNN 嵌入
   - 训练 MLP regressor (768→64→32→1)
   - 验证预测精度和速度

### 中期（3-5天）— 论文完善

3. **方案 A1/A2**：外部数据扩充
4. **方案 G**：离线 RL 实验
5. **多目标 Pareto** 完整分析

### 长期 — 论文撰写

6. Phase 5 对比实验
7. 消融实验
8. 论文写作

---

## 参考文献

1. Vignac, C. et al. (2023). MiDi: Mixed graph and 3D denoising diffusion for molecule generation. *ICML*.
2. Hoogeboom, E. et al. (2022). Equivariant diffusion for molecule generation in 3D. *ICML*.
3. Axelrod, S. & Gomez-Bombarelli, R. (2022). GEOM: Energy-annotated molecular conformations. *Scientific Data*.
4. Kumar, A. et al. (2020). Conservative Q-learning for offline reinforcement learning. *NeurIPS*.
5. Levine, S. et al. (2020). Offline reinforcement learning: Tutorial, review, and perspectives on open problems. *arXiv*.
6. Ramakrishnan, R. et al. (2014). Quantum chemistry structures and properties of 134 kilo molecules. *Scientific Data*.
7. Groom, C.R. et al. (2016). The Cambridge Structural Database. *Acta Cryst*.
