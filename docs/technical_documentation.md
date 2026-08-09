# 技术文档：基于对称性条件扩散模型与 MARL 的 COF 材料设计

> 2026年6月22日 | 完整技术说明

---

## 目录

1. [系统概述](#1-系统概述)
2. [扩散模型：分子 Core 生成](#2-扩散模型分子-core-生成)
3. [强化学习：MARL COF 设计](#3-强化学习marl-cof-设计)
4. [数据流与格式转换](#4-数据流与格式转换)
5. [训练流程](#5-训练流程)
6. [评估指标](#6-评估指标)
7. [文件结构与权重](#7-文件结构与权重)

---

## 1. 系统概述

### 1.1 问题定义

**目标**：设计新型 COF（共价有机框架）材料以最大化 N₂ 气体吸附能力。

**输入**：无（系统自主探索设计空间）
**输出**：COF 晶体结构（CIF 格式），附 N₂ 吸附预测值

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MARL 强化学习层 (Phase 4)                       │
│                                                                         │
│  Agent 0: 拓扑+堆叠   Agent 1: Core-A对称性   Agent 2: Core-B对称性    │
│       │                    │                      │                    │
│       └────────────────────┼──────────────────────┘                    │
│                            │  (symmetry condition)                     │
│                            ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              扩散模型层 (Phase 1 + Phase 2)                       │  │
│  │                                                                  │  │
│  │  Phase 1: QM9 预训练 → 合法有机分子生成能力                       │  │
│  │  Phase 2: Core 微调   → 对称性条件 Core 生成                      │  │
│  │                                                                  │  │
│  │  输入: 对称性条件 (128-dim)                                       │  │
│  │  输出: Core 分子 (Q/R 连接点, 3D坐标, 原子类型)                    │  │
│  └─────────────────────────────┬────────────────────────────────────┘  │
│                                │                                       │
│  ┌─────────────────────────────▼────────────────────────────────────┐  │
│  │              pycofbuilder 化学组装层 (Phase 3)                     │  │
│  │                                                                  │  │
│  │  Core → +Connector → +FuncGroup → Building Block → COF 晶体       │  │
│  └─────────────────────────────┬────────────────────────────────────┘  │
│                                │                                       │
│  ┌─────────────────────────────▼────────────────────────────────────┐  │
│  │               评估层 (cof_predictor)                              │  │
│  │                                                                  │  │
│  │  COF CIF → PMTransformer → N₂ 吸附预测 → Reward                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 各层职责

| 层 | 解决的问题 | 输入 | 输出 | 方法 |
|------|------|------|------|------|
| 扩散模型 | 如何生成新分子？ | 对称性条件 + 噪声 | Core (Q/R, 坐标) | 混合连续-离散扩散 (DDPM + D3PM) |
| pycofbuilder | 如何组装完整构件？ | Core | Building Block (X 连接点) | 确定性化学规则 |
| Framework | 如何形成晶体？ | BB-A + BB-B + 拓扑 | COF 晶体 (.cif) | 网状化学组装 |
| 评估器 | COF 吸附多少 N₂？ | COF CIF | N₂ 吸附量 | PMTransformer (GCNN) |
| MARL | 如何选择最优设计？ | COF 属性 → Reward | 设计规格 (拓扑, 对称性) | PPO (近端策略优化) |

---

## 2. 扩散模型：分子 Core 生成

### 2.1 为什么用扩散模型

**核心问题**：COF 构件需要同时生成**连续变量**（3D 坐标）和**离散变量**（原子类型、键类型）。

| 候选方法 | 连续变量 | 离散变量 | 选择 |
|------|:---:|:---:|:---:|
| GAN | ✅ | ❌ 梯度无法通过离散采样 | ✗ |
| VAE | ✅ | 🟡 连续隐空间不适合化学约束 | ✗ |
| 自回归 | N/A | ✅ | ✗ 生成顺序固定，无法利用 3D 等变性 |
| **扩散模型** | ✅ DDPM | ✅ D3PM | **✓ 原生混合连续-离散** |

### 2.2 基础框架：MiDi (Vignac et al., 2023)

选择 MiDi 作为基础框架的原因：
1. **原生混合扩散**：DDPM（连续坐标）+ D3PM（离散原子/键类型）统一框架
2. **EGNN 骨干**：E(n) 等变图神经网络，天然保证旋转/平移/反射下预测不变
3. **FiLM 条件注入**：可在每层注入对称性条件
4. **QM9 预训练成熟**：有经过验证的训练方案

### 2.3 连续扩散：坐标生成 (DDPM)

**前向过程**（加噪）：
```
x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0, I)
```

**反向过程**（去噪）：
```
x_{t-1} = μ_t(x_t, x̂_0) + σ_t · z

其中：
  x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ) / √(ᾱ_t)        ← 预测干净坐标
  μ_t = √(ᾱ_{t-1})·β_t/(1-ᾱ_t) · x̂_0          ← DDPM 后验均值
      + √(α_t)·(1-ᾱ_{t-1})/(1-ᾱ_t) · x_t
```

**噪声调度**：Cosine schedule (Nichol & Dhariwal, 2021), T=1000 步

### 2.4 离散扩散：原子类型 + 键类型 (D3PM)

**为什么需要 D3PM**：原子类型（H/C/N/O/F/...）是离散类别，高斯噪声无意义。"C 原子加一点噪声"不能变成 "70% C + 30% N"。

**转移矩阵**：
```
Q̄_t = α_t · I + (1-α_t) · 1 · m^T

  m = 数据集的边缘分布 (e.g., H=51%, C=35%, ...)
  α_t = 余弦衰减 (1 → 0)
```

每个扩散步，类别概率按 Q̄_t 转移。当 t=T 时，分布收敛到边缘分布 m（而非纯噪声）。

**反向过程**（D3PM 后验）：
```
p(a_{t-1} | a_t, a₀) ∝ (a_t @ Q_t^T) ⊙ (a₀_pred @ Q̄_{t-1})
```

### 2.5 EGNN 骨干网络

**为什么用 EGNN 而非 Transformer**：分子的 3D 几何要求 E(3) 等变性（旋转/平移/反射下预测不变）。Transformer 的绝对位置编码破坏等变性。EGNN 仅使用原子间距离（不变）和相对坐标差（等变更新），天然保证 E(3) 等变性。

**架构**：

```
输入: (原子类型 one-hot, 3D坐标, 边索引, 时间步 t, 条件 c)
                    │
    ┌───────────────▼────────────────┐
    │  EGNNEmbedding                 │
    │  atom_embed(one-hot)           │  → h₀ (N, 256)
    │  + time_embed(t)               │
    │  + condition_embed(c)          │
    └───────────────┬────────────────┘
                    │
    ┌───────────────▼────────────────┐
    │  EGNN Layer × 9               │
    │                                │
    │  每条边 (i→j):                 │
    │    m_ij = edge_mlp(h_i, h_j,   │
    │            ||x_i-x_j||², e_ij) │
    │                                │
    │  坐标更新:                      │
    │    x_i' = x_i + Σ_j (x_i-x_j)  │
    │            · coord_mlp(m_ij)   │
    │                                │
    │  节点更新:                      │
    │    h_i' = h_i + node_mlp(      │
    │           h_i, Σ_j m_ij)       │
    │    h_i' = LayerNorm(h_i')      │
    │                                │
    │  条件注入 (FiLM):               │
    │    h' = γ(c)·h + β(c)          │
    └───────────────┬────────────────┘
                    │
    ┌───────────────▼────────────────┐
    │  输出头                        │
    │  coord_head: h → ε_pred (N,3) │
    │  atom_head:  h → a_logits (N,K)│
    │  bond_head:  [h_i,h_j] → b (E,K)│
    └────────────────────────────────┘
```

**关键设计决策**：
- Attention **禁用**：全局 softmax(dim=0) 对 ~20K 边归一化，SiLU 无界激活导致 float32 溢出
- 全 float32：EGNN + 输出头在 `autocast(enabled=False)` 中运行
- LayerNorm eps=1e-4：防止 float16 方差下溢

### 2.6 Phase 1：QM9 预训练

| 参数 | 值 |
|------|------|
| **训练数据** | QM9, 130,847 分子, H/C/N/O/F |
| **模型** | Denoiser, 5,184,778 参数 |
| **架构** | hidden_dim=256, 9 layers, attention=False |
| **扩散步数** | 1000, cosine schedule |
| **Batch size** | 64, grad_accum=2 |
| **学习率** | 1e-4, warmup 1000步 → cosine衰减 |
| **优化器** | AdamW |
| **混合精度** | AMP (autocast, EGNN float32) |
| **训练轮数** | 500 epochs |
| **最佳模型** | epoch 444, val_loss=1.462 |
| **GPU** | A6000 48GB, ~2.5天 |

**输入**：加噪分子 (x_t, a_t, e_t) + 时间步 t
**输出**：去噪预测 (ε_pred, a_logits, b_logits)
**损失**：
```
L_total = L_coord + L_atom + L_bond
L_coord = MSE(ε_pred, ε_true)
L_atom  = CrossEntropy(a_logits, a_true)
L_bond  = CrossEntropy(b_logits, b_true)
```

### 2.7 Phase 2：Core 对称性微调

**为什么微调而非从头训练**：
- QM9 已经教会模型生成合法有机分子的能力
- 微调只需适配 COF Core 的特定特征（更大分子、Q/R 点、对称性约束）

**为什么生成 Core 而非完整 BB**：

| 比较 | Core | 完整 Building Block |
|------|------|------|
| 原子数 | 12-40 | 20-168 |
| 原子类型 | 12 | 10 |
| 复杂度 | 刚性骨架 | 骨架+连接子+官能团 |
| 对称性 | **几何约束**（L2=2个Q沿180°） | 需要学习 |
| 条件维度 | 128 (仅对称性) | 384 (sym+conn+FG) |

**对称性编码**（SymmCD 适配）：

```
晶体空间群 (SymmCD): 15轴 × 26操作 = 390位
         ↓ 适配
分子点群 (本工作):  3主轴 × 13操作 = 39位

3主轴: 主轴(最高阶旋转轴), 副轴(垂直主轴), 第三轴(正交)
13操作: 1(恒等), -1(反演), 2(C2), 3(C3), 4(C4), 6(C6),
        m(镜面), -2(S2), -3(S3), -4(S4), -6(S6), 2/m, 4/m

编码流程: 16点群 → 39位二进制矩阵 → MLP(39→64→128) → 128-dim 嵌入
```

**FiLM 条件注入**：
```
h' = γ(condition) · h + β(condition)

γ_net: Linear(128→256) → SiLU → Linear(256→256)
β_net: Linear(128→256) → SiLU → Linear(256→256)

每层 EGNN 独立学习 γ 和 β → 不同抽象层级自适应调制
```

**训练配置**：

| 参数 | 值 |
|------|------|
| **训练数据** | 83 Core × 60x 增强 → 5,000 样本 |
| **模型** | SymmetryConditionedDenoiser, 7,049,890 参数 |
| **Core only** | 12 atom types, 128-dim FiLM, attention=False |
| **两阶段** | Stage A (1-50): 冻结 EGNN, 训练 FiLM+encoder |
|  | Stage B (50-500): 解冻末3层 EGNN |
| **辅助损失** | 对称性分类器, weight=0.1 |
| **训练轮数** | 500 epochs |

**输入**：对称性条件 (128-dim)
**输出**：Core (Q/R 连接点, 12 种原子类型, 3D 坐标)

---

## 3. 强化学习：MARL COF 设计

### 3.1 为什么用 RL

**核心问题**：从离散的设计选择（对称性对、Core 模板）中找到 N₂ 吸附最大的组合。

| 方法 | 适用性 | 问题 |
|------|:---:|------|
| 随机搜索 | 🟡 | 组合空间大 |
| 贝叶斯优化 | 🟡 | 离散空间 GP 不适用 |
| 遗传算法 | ✅ | 可行但无法利用梯度 |
| **RL (PPO)** | ✅ | **直接从奖励信号学习策略** |

**为什么 2-Agent 而非 3-Agent**：
- 拓扑类型**由对称性对自动决定**（如 T3+L2 → HCB_A/KGD 兼容），不需要单独的 Agent 选择
- 原先的 3-Agent 设计中 Agent 0 选拓扑会导致大量无效组合（拓扑-对称性不匹配，~28% 失败率）
- 精简为 2-Agent：Agent 1 选 Core-A 规格，Agent 2 选 Core-B 规格，拓扑自动推导

### 3.2 动作空间

| Agent | 动作 | 维度 | 示例 |
|:---:|------|:---:|------|
| Agent 1 | Core-A 对称性 + 模板 | 6 × N_cores | `(T3, TOTB)` |
| Agent 2 | Core-B 对称性 + 模板 | 6 × N_cores | `(L2, DPDA)` |

**对称性 (6种)**：L2(2连接点/C2v), T3(3/D3h), S4(4/D4h), H6(6/D6h), D4(4/D4h), R4(4/D4h)

**拓扑自动推导**（由连接点数对决定）：
```
(conn_a, conn_b) → 兼容拓扑
  T3(3) + L2(2)  → HCB_A, KGD, FXT_A
  S4(4) + L2(2)  → SQL_A, DIA_A, BOR
  L2(2) + L2(2)  → KGM_A
  H6(6) + L2(2)  → HXL_A, LON_A
```

### 3.3 状态空间

| Agent | 状态维度 | 内容 |
|:---:|:---:|------|
| Agent 1 | 8 | one-hot 对称性(6) + 归一化Reward(1) + 归一化连接数(1) |
| Agent 2 | 8 | 同上 |

### 3.4 奖励设计

```
R = N₂_adsorption × 10.0 + 5.0  (真实 N₂ 预测，40 个 COF 预计算)

或 (Mock 奖励):
R = base_adsorption(topology) + size_bonus + symmetry_bonus + novelty_bonus + noise

  assembly_failure: R = -1.0
```

### 3.5 MAPPO 训练

**当前实现**：3-Agent MAPPO，action masking

| 参数 | 值 |
|------|------|
| Agent 数量 | **3** |
| Agent 0 | 拓扑选择（9 options），Actor MLP(16→128→128→9) |
| Agent 1 | Core-A 对称性（6 options），**action masked**（只允许当前拓扑兼容的对称性） |
| Agent 2 | Core-B 对称性（6 options），**action masked** |
| Critic | 中心化 MLP(48→128→128→1)，接收联合状态 (s0,s1,s2) |
| 学习率 | 1e-4 |
| 剪辑范围 ε | 0.2 |
| 更新频率 | 每 20 episodes |
| 总 episodes | 2,000 |

**Action Masking 机制**：
- Agent 0 选择拓扑 → 查表获取兼容的 (conn_a, conn_b) 对
- Agent 1 的 logits 中不兼容的对称性被设为 -1e9 → softmax 后概率为 0
- Agent 2 同理
- **100% 的 episode 都是合法拓扑-对称性组合**
- 不合法的组合直接返回 -1 奖励，无需 COF 组装

**拓扑-对称性约束表**：

| 拓扑 | Agent 1 对称性 | Agent 2 对称性 |
|------|------|------|
| HCB_A | T3 | L2 |
| KGD | T3 | L2, T3 |
| FXT_A | T3 | L2 |
| SQL_A | S4, D4, R4 | L2 |
| DIA_A | S4, D4, R4 | L2, S4, D4, R4 |
| BOR | S4, D4, R4 | L2, S4, D4, R4 |
| HXL_A | H6 | L2 |
| LON_A | H6 | L2 |
| KGM_A | L2 | L2 |

**训练脚本**：`marl_interface/train_mappo_3agent.py`（环境） + `env_v3.py`

### 3.6 N₂ 吸附预测器

**模型**：PMTransformer（预训练 GCNN + N₂ 吸附微调头）

```
COF CIF → 图嵌入 (GCNN) → Transformer → N₂ 回归值
```

输入：COF 晶体结构的 CIF 文件
输出：N₂ 吸附量（regression logits，范围 -0.21 ~ 0.79）

---

## 4. 数据流与格式转换

```
Phase 1 训练:
  QM9 (.xyz) → PyG Data (x, positions, edge_index) → Denoiser → Loss

Phase 2 训练:
  Core (.cjson) → PyG Data (12 types) → CoreDataset → SymmetryConditionedDenoiser → Loss

Phase 3 生成:
  对称性条件 → Diffusion.sample() → atom_types + positions
    → Q→X 转换 → BuildingBlock (pycofbuilder)
    → Framework.from_building_blocks() → COF (.cif)

Phase 4 RL:
  Agent action → 对称性 → Core refinement (coord-only)
    → pycofbuilder BB → Framework COF → CIF
    → PMTransformer → N₂ uptake → Reward → PPO update
```

**关键格式**：

| 格式 | 用途 | 内容 |
|------|------|------|
| `.cjson` | Core 训练数据 | atom types (含 Q/R), 3D coords, properties |
| PyG Data | 模型输入 | x (one-hot atoms), positions, edge_index, edge_attr, symm_idx |
| `.xyz` | Core 可视化/导出 | atom symbols, 3D coords |
| `.cif` | COF 晶体 | 晶格参数, 原子坐标, 空间群 |
| `.pt` | 模型权重/PyG数据集 | torch.save 序列化 |

---

## 5. 训练流程

### Phase 1: QM9 预训练

```bash
CUDA_VISIBLE_DEVICES=1 python -u symmcd_diffusion/train_qm9.py \
    --epochs 500 --batch_size 64 --lr 1e-4 --grad_accum 2 --use_amp \
    --data_root ./data/qm9 --gpu_sleep 0.15
```

**训练曲线**：
- Epoch 1: loss=3.95
- Epoch 100: loss ~2.0
- Epoch 250: loss ~1.5
- Epoch 444: **best val_loss=1.462**
- Epoch 500: loss ~1.48

### Phase 2: Core 微调

```bash
CUDA_VISIBLE_DEVICES=2 python -u symmcd_diffusion/train_symmetry_conditioned.py \
    --qm9_ckpt symmcd_diffusion/checkpoints/qm9_best.pt \
    --epochs 500 --batch_size 32 --lr 5e-5 --grad_accum 2 --use_amp
```

**训练曲线**：
- Stage A (1-50): 冻结基座, loss 3.53 → 2.9
- Stage B (50-500): 解冻末3层, loss 2.9 → 1.78
- Symmetry loss: 2.09 → 0.004

### Phase 4: MARL 训练

```bash
python symmcd_diffusion/marl_interface/train_mappo_cof.py \
    --episodes 2000 --lr 1e-4 --update_every 20 --real_reward
```

**训练结果**：

| 指标 | Mock Reward | Real N₂ Reward |
|------|:---:|:---:|
| 组装成功率 | 72% | 72% |
| 平均 Reward | 14.4 | 14.5 |
| 最佳 Reward | 22.1 | 22.2 |
| 训练时间 | 2.2h | 2.2h |

---

## 6. 评估指标

### 扩散模型评估

| 指标 | 定义 | Phase 1 结果 | Phase 2 结果 |
|------|------|:---:|:---:|
| **RDKit 合法性** | SanitizeMol 通过率 | 97% | N/A (Q/R非真实元素) |
| **唯一性** | 合法分子中不重复的比例 | 100% | — |
| **原子类型保存** | 精修后与原模板一致的比例 | — | 100% |
| **Q/R 保存** | Q/R 计数保留比例 | — | 100% |
| **几何合法性** | 3层过滤器通过率 | — | 75% (500ep) |
| **对称性分类准确率** | 预测点群=条件点群 | — | 99.6% |

### MARL 评估

| 指标 | 定义 | 结果 |
|------|------|:---:|
| **组装成功率** | COF 组装成功的比例 | 72% |
| **平均 Reward** | 所有 episode 的平均奖励 | 14.5 |
| **最佳设计** | 训练中发现的最优 N₂ 吸附 | DBA1+2BPD (reward=22.2) |
| **N₂ 预测范围** | 40 个 COF 的预测值范围 | -0.21 ~ 0.79 |

---

## 7. 文件结构与权重

### 项目结构

```
MARL_for_COFs/
├── symmcd_diffusion/           # 扩散模型核心模块
│   ├── data/core/              # 83 Core cjson 训练数据
│   ├── models/                 # EGNN, Denoiser, Diffusion, ConditionalDenoiser
│   ├── symmetry/               # point_group (16点群), symmetry_encoder (39-bit)
│   ├── generation/             # CoreGenerator
│   ├── filters/                # 合法性过滤, 连接点验证
│   ├── marl_interface/         # Phase 4 环境 + PPO 训练脚本
│   ├── utils/visualize_core.py # py3Dmol/matplotlib/XYZ
│   ├── released_weights/       # 发布权重
│   └── generated/              # 生成输出 (CIF, XYZ, 查找表)
├── pycofbuilder/               # COF 组装框架
├── cof_predictor/              # N₂ 吸附预测器 (PMTransformer)
├── docs/                       # 文档
│   ├── implementation_plan.md  # 完整实施计划
│   ├── technical_documentation.md  # 本文档
│   ├── weekly_report_2026-06-18.md # 周报
│   ├── final_summary_2026-06-17.md # 最终总结
│   └── interaction_log.md      # 交互日志 (20条)
└── mappo/                      # 多智能体 PPO 框架
```

### 模型权重

| 文件 | 大小 | 内容 |
|------|:---:|------|
| `released_weights/qm9_best.pt` | 60MB | Phase 1 最佳 (epoch 444) |
| `released_weights/symmcd_500ep.pt` | 56MB | Phase 2 (500 epochs, 推荐) |
| `released_weights/symmcd_denoiser_finetuned.pt` | 27MB | Phase 2 (300 epochs) |
| `marl_interface/checkpoints/mappo_cof_policy.pt` | 101KB | PPO 策略网络 |

### 关键数据文件

| 文件 | 内容 |
|------|------|
| `generated/n2_lookup.json` | 40 COF 的真实 N₂ 预测值 |
| `generated/cof_batch_v2/` | 3 个扩散精修 COF (CIF) |
| `generated/cof_batch/` | 7 个 COF (pycofbuilder 模板) |
