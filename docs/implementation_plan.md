# 实施计划：基于对称性条件扩散模型与多智能体强化学习的共价有机框架设计

> **Implementation Plan: Symmetry-Conditioned Diffusion for MARL-Guided COF Design**
>
> 硕士论文课题 | 总周期：15 个月 | 创建日期：2026-06-08
>
> 英文原版：`C:\Users\l\.claude\plans\hashed-spinning-music.md`

---

## 一、项目背景与研究动机

### 1.1 当前系统现状

当前代码库（`D:\Demo\MARL_for_COFs`）使用 MAPPO（多智能体近端策略优化）算法，6 个 Agent 各自从一个包含 53 个离散构件（核心、连接臂、连接子、官能团）的固定词汇表中选择分子构件，通过 pycofbuilder 组装成 COF 晶体结构。目前仅支持 5 种拓扑类型（HCB_A, SQL, SQL_A, KGD, HXL_A），全部使用 AA 堆叠模式。

**核心局限**：系统本质上是**组合优化器**——只能重新组合已知的积木块，无法创造全新的分子构件。

### 1.2 研究目标

构建一个 **3-Agent MARL 系统**，其中：
- Agent 负责选择高层设计参数（拓扑类型、堆叠模式、对称性类型、官能团）
- **对称性条件扩散模型**（SymmCD 启发）根据这些参数生成全新的分子构件
- **Self-Play 闭环**持续将高质量构件反馈到扩散模型训练中

### 1.3 核心创新点

1. 将 SymmCD 的**对称性二进制编码**从晶体空间群（230 种）适配到分子点群（14 种）
2. 以**非对称单元生成**替代全分子生成：先生成对称性不等价原子，再通过点群操作复制到完整分子
3. 混合**连续扩散**（原子坐标）+ **离散扩散**（原子类型 + 键类型）
4. MARL 从离散选择（53 动作）升级为**连续生成**（扩散模型在对称性条件下采样）

### 1.4 基础框架选择：MiDi

基于 **MiDi**（Vignac et al., 2023）构建，原因：
1. 原生支持混合连续+离散扩散，与 COF 构件表示完美匹配
2. EGNN 去噪骨干可直接注入对称性条件
3. 有 QM9 预训练的成熟方案和文档
4. 代码库紧凑，适合单人开发

---

## 二、分阶段实施计划

---

### 第一阶段：扩散模型基础预训练（第 1-3 月）

**目标**：在 QM9 数据集（13 万分子）上预训练混合扩散模型，建立分子生成基线能力。

#### 新增目录结构

```
MARL_for_COFs/
  symmcd_diffusion/                   # 新增核心模块
    config/
      base_config.py                  # 基础配置（模型/训练/扩散参数）
    models/
      egnn.py                         # EGNN 等变图神经网络（FiLM + 注意力）
      denoiser.py                     # 混合扩散去噪网络
      noise_schedule.py               # 噪声调度器（cosine + 离散转移矩阵）
      diffusion_process.py            # 扩散过程（前向加噪 + 反向采样）
    data/
      qm9_dataset.py                  # QM9 数据加载器（PyG 格式）
    train_qm9.py                      # QM9 预训练脚本
```

#### 关键组件

| 文件 | 核心类/函数 | 功能 |
|------|-----------|------|
| `egnn.py` | `EGNNLayer`, `EGNN` | E(n) 等变消息传递，支持注意力和 FiLM 条件注入 |
| `denoiser.py` | `Denoiser` | 输入加噪数据 X/A/E + 时间步 t → 预测干净数据 |
| `noise_schedule.py` | `MixedNoiseScheduler` | 连续扩散（坐标，cosine schedule）+ 离散扩散（原子/键类型） |
| `diffusion_process.py` | `DiffusionProcess` | 前向扩散 q(x_t\|x_0) + 祖先采样 p(x_{t-1}\|x_t) |

#### QM9 训练配置

```python
hidden_dim: 256          # 隐藏层维度
num_layers: 9            # EGNN 层数
diffusion_steps: 1000    # 扩散步数
batch_size: 64           # 批大小
learning_rate: 1e-4      # 学习率
grad_accumulation: 2     # 梯度累积（适配 A6000 24GB 显存）
use_amp: True            # 混合精度训练
```

#### 验证标准

- RDKit 分子合法性 > 90%
- 生成分子唯一性 > 95%
- 原子类型分布与 QM9 训练集的 KL 散度 < 5%

#### 交付物

`qm9_denoiser_epoch500.pt` — 预训练好的去噪网络权重

---

### 第二阶段：对称性条件注入（第 3-5 月）

**目标**：将 SymmCD 的对称性编码适配到分子领域，训练条件扩散模型，并在增强后的 COF 数据集上微调。

#### 关键新增文件

| 文件 | 核心类/函数 | 功能 |
|------|-----------|------|
| `symmetry/point_group.py` | `compute_point_group()` | 惯性张量法计算 14 种分子点群（C1~D6h） |
| `symmetry/symmetry_encoder.py` | `SymmetryEncoder` | SymmCD 适配：3 主轴 × 13 对称操作 = 39 位二进制编码 |
| `models/conditional_denoiser.py` | `SymmetryConditionedDenoiser` | 条件去噪器：对称性(128) + 连接数(128) + 官能团(128) → 384 维条件向量 |
| `data/augmentation.py` | `COFAugmenter` | 数据增强：194 → ~2000 样本（旋转×5 + 噪声×2 + 骨架扩展×1.5） |
| `data/cof_dataset.py` | `COFBBDataModule` | COF 构件数据集加载（含对称性、连接数、官能团标签） |

#### 对称性编码方案

SymmCD 使用 **15 个晶体学轴 × 26 种对称操作 = 390 位**编码空间群。我们适配为：

```
分子点群编码: 3 主轴 × 13 对称操作 = 39 位二进制矩阵

3 主轴：主轴（最高阶旋转轴）、副轴（垂直主轴）、第三轴（正交于前两者）

13 对称操作：恒等(1)、反演(-1)、旋转(C2/C3/C4/C6)、镜面(m)、
            旋转反演(S2/S3/S4/S6)、组合(2/m、4/m)
```

#### 微调策略

- **阶段 A**（1-50 epoch）：冻结基础去噪器，仅训练 FiLM 层和条件嵌入
- **阶段 B**（50-100 epoch）：解冻最后 3 层 EGNN，全模型微调
- **辅助损失**：对称性分类器（从节点特征预测点群），权重 0.1

#### 验证标准

- 对称一致性 > 80%（生成分子的计算点群与条件匹配）
- 连接点正确率 > 85%（Q/X 连接点数量正确）
- RDKit 合法性 > 75%

#### 交付物

`symmcd_denoiser_finetuned.pt` — 对称性条件去噪模型

---

### 第三阶段：COF 构件生成器（第 5-7 月）

**目标**：构建端到端生成器，包含多层合法性过滤和 pycofbuilder 兼容的 cjson 导出。

#### 关键新增文件

| 文件 | 核心类/函数 | 功能 |
|------|-----------|------|
| `filters/legality_filter.py` | `LegalityFilter` | **5 层递进过滤**（见下图） |
| `filters/connectivity.py` | `ConnectionPointVerifier` | 连接点几何验证（L2:180°, T3:120°, S4:90°, H6:60°） |
| `data/cjson_io.py` | `CJSONExporter`, `CJSONImporter` | 扩散输出 ↔ cjson 格式双向转换 |
| `generation/generator.py` | `COFBBGenerator` | 端到端：条件采样 → 过滤 → cjson 导出 → 配对 |

#### 5 层合法性过滤 Pipeline

```
生成 1000 个候选构件
    │
    ▼
Layer 1: 原子级检查 ───────────────────────── 通过率 ~70%
  • 原子间距 ≥ 0.5Å（排除碰撞）
  • 化合价合理性
    │
    ▼ ~700 个
Layer 2: RDKit 解析 ────────────────────────── 通过率 ~60%
  • RDKit SanitizeMol（价键检查）
  • 芳香性检测
  • 排除断开碎片
    │
    ▼ ~420 个
Layer 3: 对称性验证 ───────────────────────── 通过率 ~70%
  • 计算实际点群 vs 目标点群
  • 容差 RMSD < 0.3Å
    │
    ▼ ~294 个
Layer 4: 连接性检查 ───────────────────────── 通过率 ~80%
  • 连接点数量匹配（L2=2, T3=3, S4=4, H6=6）
  • 连接向量几何正确（角度容差 ±15°）
    │
    ▼ ~235 个
Layer 5: COF 组装测试 ─────────────────────── 通过率 ~70%
  • 尝试 pycofbuilder 与互补构件配对组装
  • 验证 CIF 成功生成

    ▼ ~164 个合格构件

端到端通过率: ~16%（每 1000 个候选 → 164 个可用）
```

#### 交付物

可运行的 `COFBBGenerator`，与 pycofbuilder 完全集成

---

### 第四阶段：MARL 集成（第 7-10 月）

**目标**：将扩散生成器与 MARL 连接，从 6-Agent 离散选择改造为 3-Agent 设计规范系统。

#### 架构改造

```
改造前（固定词汇表）:              改造后（扩散生成）:
┌──────────────────────┐         ┌──────────────────────────┐
│ 6 个 Agent             │         │ 3 个 Agent                │
│ 每步选 1 个 token       │         │ 一次性指定完整设计          │
│ 从 53 个离散动作        │         │                            │
│ 组合优化               │         │ Agent 0: 拓扑(14) + 堆叠(8) │
│ 仅重组已知构件          │         │ Agent 1: 构件A规格          │
│                        │         │ Agent 2: 构件B规格          │
│                        │         │         ↓                  │
│                        │         │ 扩散模型生成实际分子构件     │
└──────────────────────┘         └──────────────────────────┘
```

#### 关键新增/修改文件

| 文件 | 核心类 | 功能 |
|------|--------|------|
| `mappo/env_v2.py` | `COFDesignEnvV2` | 3-Agent 环境：MultiDiscrete 动作空间 |
| `mappo/mappo_mpe_v2.py` | `Actor_MultiDiscrete`, `MAPPO_MPE_V2` | 多头 Actor + 中心化 Critic |
| `marl_interface/diffusion_env.py` | `DiffusionGeneratorWrapper` | 缓存层 + 预生成常用规格的构件池 |
| `marl_interface/reward_bridge.py` | `RewardBridge` | 扩散输出 → COF 组装 → 预测器 → 奖励 |

#### Agent 动作空间

| Agent | 选择内容 | 维度 | 示例 |
|:---:|------|:---:|------|
| Agent 0 | 拓扑类型 + 堆叠模式 | 11 + 7 | `(HCB_A, AA)` |
| Agent 1 | 对称性 + 官能团 | 4 + 8 | `(T3, CHO)` |
| Agent 2 | 对称性 + 官能团 | 4 + 8 | `(L2, NH2)` |

#### 奖励设计

```python
reward = N2_吸附量 × 0.1           # 主目标
       + 组装成功奖励              # +1.0（COF 成功组装）
       + 对称性兼容奖励            # +0.5（对称性-拓扑匹配）
       + 多样性奖励                # +0~0.5（探索未充分利用的拓扑）
       + RND 探索奖励              # 复用现有 RND 模块
```

#### Self-Play 闭环

```
扩散模型生成构件 → MARL 筛选 → 好的构件加入训练集 → 微调扩散模型
      ↑                                                      │
      └─────────────────── 循环 5 轮 ───────────────────────┘
每轮: 200 MARL episodes → 选 Top-50 构件 → 扩散模型微调 10 epochs
Self-play 数据上限: 总训练集的 20%（防止多样性崩溃）
```

#### 验证标准

- MARL + 扩散生成的 COF 平均 N₂ 吸附值 **超越固定词汇表基线**
- Self-Play 呈现改善趋势（第 5 轮 > 第 3 轮 > 第 1 轮）
- 拓扑多样性：至少覆盖 14 种拓扑中的 8 种

#### 交付物

端到端运行系统：`python mappo/run_v2.py --use-diffusion`

---

### 第五阶段：实验评估（第 10-13 月）

**目标**：全面的对比实验、消融研究和统计分析。

#### 基线方法

1. **固定词汇表 MAPPO**（现有系统）——主要对比对象
2. **随机基线**——随机选择拓扑 + 堆叠 + 构件
3. **纯扩散生成**——随机条件采样，无 MARL 优化
4. **遗传算法**——在扩散条件空间上做 GA 搜索
5. **贝叶斯优化**——GP 代理模型优化条件参数

#### 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **主要** | N₂/O₂ 吸附量 | 均值、最大值、Top-10 均值 |
| **质量** | 合法性、多样性、新颖性、可合成性 | Tanimoto 指纹距离、不在训练集中的比例 |
| **效率** | 达到最优的样本数、每个合法 COF 的时间 | 计算资源效率 |
| **对称性** | 对称一致性、拓扑覆盖率 | 生成与条件的一致性 |

#### 消融实验（8 维 × 5 随机种子 = 40 组实验）

1. 对称性条件：有 vs 无
2. 扩散步数：100 / 250 / 500 / 1000
3. 条件注入方式：FiLM vs 拼接
4. 合法性过滤层数：1~5
5. Self-Play 循环数：0 / 1 / 3 / 5
6. Agent 数量：3 vs 6
7. RND 探索奖励：开 vs 关
8. QM9 预训练：有 vs 无

#### 交付物

全部实验数据 + 论文级图表

---

### 第六阶段：论文撰写（第 13-15 月）

- 代码整理：添加 docstring、类型标注、README
- 可复现性：Dockerfile、精确依赖版本、预训练权重
- 补充材料：复现所有图表的脚本

---

## 三、甘特图

```
月份:   1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
阶段1:  [████ 基础预训练 ████]
阶段2:           [████ 对称性条件 ████]
阶段3:                    [████ 构件生成器 ████]
阶段4:                            [█████ MARL 集成 █████]
阶段5:                                         [█████ 实验 █████]
阶段6:                                                       [██ 论文 ██]
```

---

## 四、与 SymmCD 论文的对应关系

| SymmCD 概念 | 本课题适配 | 差异说明 |
|-------------|-----------|---------|
| 空间群 G（230 种） | 分子点群 P（14 种） | 分子无平移对称性，简化为点群 |
| 晶格参数 k（6 维） | 不需要 | 分子是有限体系，无周期性 |
| Wyckoff 位置 S | 分子点群 + 连接点几何 | 对称位置类型 → 构件对称性类型 |
| 非对称单元原子坐标 X | 轨道代表点坐标 X | 生成对称不等价原子，复制到全分子 |
| 原子类型 A | 原子类型 A + 键类型 B | 分子需要键拓扑 |
| 15 轴 × 26 操作 | 3 轴 × 13 操作 | 晶体学轴→分子主轴 |
| MP-20 数据集（4 万晶体） | QM9（13 万）+ COF 增强（2000） | 分子数据更丰富 |

---

## 五、关键技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 基础框架 | **MiDi** 而非 EDM | 原生混合连续+离散扩散，节省约 2 个月开发时间 |
| Agent 数量 | **3** 而非 6 | 扩散模型一次性生成完整构件，仅高层设计需要 RL |
| 条件注入 | **FiLM** 而非拼接 | 逐层自适应调制，SymmCD 验证有效 |
| 对称性编码 | **二进制矩阵** | 继承 SymmCD 框架，跨点群泛化能力强 |
| 采样策略 | **拒绝采样** 而非约束采样 | 更简单、可调试，早期过滤层快速淘汰无效样本 |
| A6000 策略 | AMP + 梯度累积 + batch=8 | 适配 24GB 显存限制 |

---

## 六、复用的现有代码

| 现有文件 | 复用内容 |
|----------|---------|
| `mappo/env.py` | Embedding_Layer, TransformerEncoder, vocab/mask 逻辑 |
| `mappo/mappo_mpe.py` | GAE 优势估计, PPO clip 目标, 价值归一化 |
| `mappo/reward.py` | RND 探索奖励类, predictor() 调用模式 |
| `pycofbuilder/framework.py` | Framework.from_building_blocks(), 14 拓扑 TOPOLOGY_DICT |
| `pycofbuilder/cjson.py` | ChemJSON 类 |
| `pycofbuilder/tools.py` | smiles_to_xsmiles() SMILES↔xSMILES 转换 |
| `cof_predictor/main.py` | doPredict() 吸附属性预测 |

---

## 七、验证计划

| 阶段 | 验证命令 | 通过标准 |
|:---:|------|:---:|
| 1 | `python symmcd_diffusion/train_qm9.py` | RDKit 合法性 > 90% |
| 2 | `python symmcd_diffusion/train_symmetry_conditioned.py` | 对称一致性 > 80% |
| 3 | `from symmcd_diffusion.generation import COFBBGenerator` | 端到端通过率 > 15% |
| 4 | `python mappo/run_v2.py --use-diffusion` | 超越固定词汇表基线 |
| 5 | `python experiments/full_comparison.py` | 统计显著性 p < 0.05 |

---

## 八、风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|:---:|------|
| 分子合法性过低 | 中 | 退回到片段级生成；RDKit 实时约束；RL 微调 |
| 训练数据不足 | 中低 | 化学规则枚举；PubChem 按点群爬取；Few-shot 方法 |
| A6000 显存溢出 | 低 | AMP + 梯度检查点 + 分批采样 |
| 扩散模型训练不稳定 | 中 | 使用成熟开源代码；降低学习率；渐进式条件注入 |
| 生成构件缺乏新颖性 | 中 | Novelty bonus；调整扩散温度；可量化报告 Novelty 指标 |

---

## Phase 1: Foundation — Molecular Diffusion Pre-training (Months 1-3)

### Goal
Set up MiDi, pre-train on QM9 (130K molecules), establish baseline molecular generation before adding symmetry conditioning.

### New Directory Structure
```
MARL_for_COFs/
  symmcd_diffusion/                   # New top-level module
    __init__.py
    config/
      base_config.py                  # Base configuration dataclass
      qm9_config.py                   # QM9 pre-training config
      cof_config.py                   # COF-specific config
    data/
      __init__.py
      qm9_dataset.py                  # QM9 loader (PyG)
      cof_dataset.py                  # COF building block dataset
      augmentation.py                 # Data augmentation (194 → 2000)
      cjson_io.py                     # cjson ↔ PyG Data conversion
    models/
      __init__.py
      egnn.py                         # EGNN backbone (from MiDi)
      denoiser.py                     # Mixed diffusion denoiser
      noise_schedule.py               # Noise schedules
      diffusion_process.py            # Forward/reverse diffusion
    symmetry/
      __init__.py
      point_group.py                  # Point group computation
      symmetry_encoder.py             # Binary symmetry encoding (SymmCD-adapted)
    filters/
      __init__.py
      legality_filter.py              # 5-layer legality filter
      connectivity.py                 # Connection point verification
    generation/
      __init__.py
      generator.py                    # End-to-end BB generator
      sampler.py                      # Sampling utilities
    marl_interface/
      __init__.py
      diffusion_env.py                # Diffusion wrapper for MARL
      reward_bridge.py                # Reward computation bridge
    train_qm9.py                      # QM9 pre-training script
    train_symmetry_conditioned.py     # Symmetry-conditioned fine-tuning
    train_self_play.py                # Self-play enhancement loop
```

### Key Files to Create

**`symmcd_diffusion/models/egnn.py`** — Port from MiDi:
- `EGNNLayer`: message function + coordinate update + node update
- `EGNN`: stacked layers with optional attention

**`symmcd_diffusion/models/denoiser.py`** — Core denoising network:
- `Denoiser(nn.Module)`: inputs noisy X/A/E + timestep t → predicts clean X/A/E
- X head: MLP from EGNN node features
- A head: linear from node features
- E head: MLP from edge features

**`symmcd_diffusion/models/diffusion_process.py`**:
- `DiffusionProcess`: `forward()` adds noise, `reverse()` ancestral sampling
- Continuous diffusion (cosine schedule) for coordinates
- Discrete diffusion (uniform transition matrix) for atom types + bond types

**`symmcd_diffusion/data/qm9_dataset.py`**:
- `QM9Dataset(Dataset)`: loads QM9 → PyG Data with `x`, `positions`, `edge_attr`

### QM9 Training Config
```python
hidden_dim: 256
num_layers: 9
diffusion_steps: 1000
batch_size: 64
learning_rate: 1e-4
grad_accumulation: 2  # A6000 24GB constraint
use_amp: True          # mixed precision
```

### Validation Criteria
- RDKit validity > 90%
- Uniqueness > 95%
- Atom type distribution within 5% KL divergence of QM9 training set

### Deliverable
`qm9_denoiser_epoch500.pt`

---

## Phase 2: Symmetry Conditioning (Months 3-5)

### Goal
Add symmetry conditioning to the diffusion model, compute point groups for QM9 molecules, train conditional denoiser, and fine-tune on augmented COF data.

### Key Files to Create

**`symmcd_diffusion/symmetry/point_group.py`**:
- `compute_point_group(atomic_numbers, positions) -> str`
- Algorithm: compute inertia tensor → diagonalize → get principal axes → test candidate point groups via symmetry operations → return best match (0.3Å RMSD tolerance)
- Target groups: C1, C2, C2v, C2h, C3, C3v, D3h, C4, D4h, C6, D6h, D2h, D2d, Td

**`symmcd_diffusion/symmetry/symmetry_encoder.py`**:
- `SymmetryEncoder(nn.Module)`: SymmCD-adapted binary encoding
- 3 axes × 13 operations = 39-bit binary matrix per point group
- MLP: 39 → 64 → 128 (encoding_dim)
- Unlike SymmCD's 15 axes for space groups, we use 3 principal axes (reduced from 15 since molecules lack translational symmetry)

**`symmcd_diffusion/models/conditional_denoiser.py`**:
- `SymmetryConditionedDenoiser`: wraps base denoiser + symmetry encoder + FiLM conditioning
- Condition sources: point_group (128-dim) + num_connectors (128-dim embedding) + func_group (128-dim embedding) → concatenated 384-dim
- FiLM layers in each EGNN layer: h' = γ(condition) * h + β(condition)

**`symmcd_diffusion/data/augmentation.py`**:
- `COFAugmenter`: 194 original → ~2000 augmented samples
- Strategies: random SO(3) rotation ×5, functional group perturbation ×2, scaffold extension ×1.5, position noise ×2

**`symmcd_diffusion/data/cof_dataset.py`**:
- `COFBBDataModule`: loads cjson files → PyG Data with `x`, `positions`, `edge_attr`, `symm_idx`, `num_connectors`, `func_group_type`
- Atom vocabulary: C, N, O, H, F, Cl, Br, S, Q (connector), X (placeholder)

### Fine-tuning Strategy
- Stage A (epochs 1-50): freeze base denoiser, train only FiLM layers + condition embeddings
- Stage B (epochs 50-100): unfreeze last 3 EGNN layers, full fine-tuning
- Auxiliary loss: symmetry classifier (predict point_group from node features), weight 0.1

### Validation Criteria
- Symmetry consistency > 80% (generated molecule's computed point group matches condition)
- Connectivity correctness > 85% (correct number of Q attachment points)
- RDKit validity > 75%

### Deliverable
`symmcd_denoiser_finetuned.pt`

---

## Phase 3: COF Building Block Generator (Months 5-7)

### Goal
Build the end-to-end generator with legality filtering and cjson export compatible with pycofbuilder.

### Key Files to Create

**`symmcd_diffusion/filters/legality_filter.py`** — 5 cascaded layers:
```
Layer 1: Atom-level checks (valence, charge neutrality) — pure geometry
Layer 2: RDKit parsing + SanitizeMol — chemical validity
Layer 3: Symmetry verification — computed point group matches target
Layer 4: Connectivity check — correct number/geometry of Q attachment points
Layer 5: COF assembly test — attempt pycofbuilder assembly with partner block
```
Expected cumulative pass rates: 70% → 60% → 70% → 80% → 70% ≈ 16% overall

**`symmcd_diffusion/filters/connectivity.py`**:
- `ConnectionPointVerifier`: checks Q/X positions match symmetry-expected geometry
  - L2: angle ~180°, T3: angles ~120°, S4: angles ~90°, H6: angles ~60°

**`symmcd_diffusion/data/cjson_io.py`**:
- `CJSONExporter`: diffusion output (atom types/positions/bonds) → ChemJSON → .cjson file
- Reuses `pycofbuilder.tools.smiles_to_xsmiles()` for xsmiles generation

**`symmcd_diffusion/generation/generator.py`**:
- `COFBBGenerator.generate(point_group, num_connectors, func_group, num_samples)`:
  1. Sample from diffusion model with condition
  2. Run 5-layer legality filter
  3. Export valid results to cjson
  4. Rejection sampling until num_samples valid or max_retries

### Validation Criteria
- > 50% of generated samples pass filters (per symmetry type)
- Generated BBs can be assembled into ≥ 3 different COF topologies

### Deliverable
Functioning `COFBBGenerator` integrated with pycofbuilder

---

## Phase 4: MARL Integration (Months 7-10)

### Goal
Connect diffusion generator to MARL, redesign from 6-agent token-selection to 3-agent design-specification, implement self-play improvement loop.

### Key Files to Create/Modify

**`mappo/env_v2.py`** (new, alongside existing `env.py`):
- `COFDesignEnv(gym.Env)` with 3 agents:
  - Agent 0: selects topology (14 types) + stacking (8 modes)
  - Agent 1: selects symmetry (5) + connector count (6) + functional group (10) for BB-A
  - Agent 2: same for BB-B
- `step()`: agents select → diffusion generates BBs → cjson export → COF assembly → predictor → rewards
- Reuses existing `TransformerEncoder` for 128-dim observations

**`mappo/mappo_mpe_v2.py`** (new):
- `Actor_MultiDiscrete`: multi-headed actor for heterogeneous action spaces
- `Critic_MLP`: same as existing, takes global state
- Per-agent action heads: Agent 0 (14+8 dims), Agents 1/2 (5+6+10 dims each)

**`mappo/reward_v2.py`** (new):
- `diffusion_cof_reward()`: bridges diffusion output to existing reward infrastructure
- Reward = N2_adsorption × 0.1 + symmetry_compatibility_bonus + validity_bonus + RND exploration
- Reuses existing `RND` class from `reward.py`

**`symmcd_diffusion/marl_interface/diffusion_env.py`**:
- `DiffusionGeneratorWrapper`: caching layer to avoid re-generating same (symm, conn, fg) combinations
- Pre-generation of candidate pools for common specifications

**`symmcd_diffusion/train_self_play.py`**:
- `SelfPlayLoop`: 5 cycles of (200 MARL episodes → select top-50 BBs by reward → add to diffusion training set → fine-tune 10 epochs)
- Self-play data capped at 20% of total training set to prevent diversity collapse

### Validation Criteria
- MARL + diffusion produces COFs with higher mean N2 adsorption than fixed-vocabulary baseline
- Self-play shows improving trend over cycles (cycle 5 > cycle 3 > cycle 1)
- Topology diversity: at least 8 of 14 topologies represented in generated designs

### Deliverable
End-to-end running system: `python run_v2.py --use-diffusion`

---

## Phase 5: Experiments (Months 10-13)

### Goal
Comprehensive evaluation against baselines, ablation studies, statistical analysis.

### New Directory
```
MARL_for_COFs/
  experiments/
    run_baselines.py      # All baseline implementations
    metrics.py            # Unified metric computation
    ablation.py           # Ablation study runner
    full_comparison.py    # Head-to-head comparison
    plot_results.py       # Publication-quality figures
```

### Baselines
1. **Fixed-vocabulary MAPPO** (existing) — primary comparison
2. **Random baseline** — random topology + stacking + BB selection
3. **Diffusion-only** — random condition sampling, no MARL optimization
4. **Genetic algorithm** — GA on diffusion condition space
5. **Bayesian optimization** — GP over condition space

### Metrics
- Primary: N2/O2 adsorption (mean, max, top-10)
- Quality: validity rate, diversity (Tanimoto), novelty (% not in training), synthesizability
- Efficiency: samples-to-best, time-per-valid-COF
- Symmetry: symmetry consistency, topology coverage

### Ablation Studies (8 dimensions, 5 seeds each)
1. Symmetry conditioning on/off
2. Diffusion steps: 100/250/500/1000
3. FiLM vs concatenation conditioning
4. Legality filter layers: 1-5
5. Self-play cycles: 0/1/3/5
6. Agent count: 3 vs 6
7. RND exploration on/off
8. QM9 pre-training on/off

### Deliverable
All experimental data + publication-quality figures

---

## Phase 6: Thesis Writing (Months 13-15)

- Clean codebase: docstrings, type hints, README
- Reproducibility: Dockerfile, exact dependency versions, pretrained weights
- Supplementary: scripts to reproduce all figures

---

## Gantt Summary
```
Months:  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
Phase 1: [███ Foundation ███]
Phase 2:          [███ Symm Cond ███]
Phase 3:                   [███ BB Gen ███]
Phase 4:                           [████ MARL Int ████]
Phase 5:                                        [████ Exp ████]
Phase 6:                                                    [██ Thesis ██]
```

## Critical Existing Files (Reuse)
- `mappo/env.py` → base for `env_v2.py` (Embedding_Layer, TransformerEncoder, vocab/mask logic)
- `mappo/mappo_mpe.py` → base for `mappo_mpe_v2.py` (GAE, PPO clip, value norm)
- `mappo/reward.py` → `RND` class, `predictor()` call pattern for `reward_v2.py`
- `pycofbuilder/framework.py` → `Framework.from_building_blocks()`, 14-topology `TOPOLOGY_DICT`
- `pycofbuilder/cjson.py` → `ChemJSON` class for cjson export
- `pycofbuilder/tools.py` → `smiles_to_xsmiles()` for SMILES↔cjson conversion
- `cof_predictor/main.py` → `doPredict()` for property evaluation

## Key Technical Decisions
1. **MiDi over EDM**: native mixed continuous+discrete diffusion avoids custom discrete diffusion module (~2 months saved)
2. **3 agents vs 6**: diffusion generates complete BBs in one pass; only high-level specs need RL
3. **FiLM conditioning**: per-layer adaptive modulation proven effective in SymmCD
4. **Binary symmetry encoding**: inherited from SymmCD, proven generalizability, reduced from 15→3 axes for molecules
5. **Rejection sampling over constrained sampling**: simpler, debugable, fast early filter layers reject most invalids cheaply
6. **A6000 strategy**: AMP + gradient accumulation + batch_size=8 for diffusion sampling

## Verification Plan
- **Phase 1**: `python symmcd_diffusion/train_qm9.py` → validity > 90%
- **Phase 2**: `python symmcd_diffusion/train_symmetry_conditioned.py` → symmetry consistency > 80%
- **Phase 3**: `python -c "from symmcd_diffusion.generation import COFBBGenerator; ..."` → > 50% pass rate
- **Phase 4**: `python mappo/run_v2.py --use-diffusion --episodes 100` → COFs generated with valid rewards
- **Phase 5**: `python experiments/full_comparison.py` → diffusion+MARL outperforms fixed-vocabulary baseline
