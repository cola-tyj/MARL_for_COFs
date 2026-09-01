# 2. 对称 COF 分子数据集构建

## 2.1 数据集目标

`cof_symmetry_pipeline` 的目标是程序化构造具有指定拓扑对称性和有效三维坐标的 COF 候选
构筑单元。历史管线采用：

```text
核心模板 + 臂模板
        ↓
2D 等价位点组合
        ↓
ETKDGv3 多初始构象嵌入
        ↓
MMFF94/UFF 几何优化
        ↓
pymatgen 点群检测与对称破缺筛选
        ↓
final_dataset.csv
```

唯一事实源为：

- `cof_symmetry_pipeline/output/final_dataset.csv`；
- 2,532 行，每行对应一个唯一 SMILES 和一个三维构象；
- `one_conformer_per_smiles=true`。

## 2.2 Stage 1：模板组合生成 2D 图

### 模板库

| 类型 | 来源 | 数量 |
|---|---|---:|
| 原始核心 | 文献/人工设计 | 41 |
| 层级核心 | 高连接节点 × C2 线性核心 | 414 |
| pyCOFBuilder 核心 | 外部模板导入 | 56 |
| **核心合计** |  | **511** |
| 原始臂 | 官能团与不同 spacer | 70 |
| pyCOFBuilder 臂 | 外部连接体导入 | 7 |
| **臂合计** |  | **77** |

核心中的 `*` 位于图自同构轨道的等价连接位置。`generator.py` 使用
`ReplaceSubstructs(..., replaceAll=True)` 将同一臂等价替换到所有连接点，再移除桥接 dummy
原子并建立真实连接键。最后执行 RDKit sanitize 和价态检查。

该阶段保证的是**二维拓扑对称性**，不能保证柔性臂在三维空间中采用完全对称的扭转角。

## 2.3 Stage 2：ETKDGv3 三维嵌入

流程为：

1. 从 SMILES 构造 RDKit 分子；
2. 添加显式 H；
3. 使用 ETKDGv3 生成多个初始构象；
4. 原始配置最多使用 8 个随机种子；
5. 大分子/层级核心减少尝试次数以控制成本；
6. 根据点群匹配和力场指标选择候选。

ETKDGv3 是知识增强的距离几何方法。它提供稳定的 2D→3D 基线，但也意味着当前坐标分布带有
ETKDG 的算法偏置，因此后续研究不能只在 ETKDG 构造的数据上证明“学到了独立的 3D 生成机制”。

## 2.4 Stage 3：力场松弛与点群筛选

历史构建主路径使用 RDKit MMFF94；部分参数不完整的分子可进入 UFF 支持路径。筛选失败包括：

- embedding 失败；
- 力场优化异常；
- 能量 NaN/发散；
- 原子位移过大；
- 松弛后点群不满足目标族。

历史 CSV 中的 `Point_Group` 来源于 pymatgen `PointGroupAnalyzer` 及当时的接纳规则。目标族和
实际点群并不要求字符串完全相同，例如目标 C3 的实际结构可以是 C3h 或 D3h。

## 2.5 Stage 4：最终数据整理

历史上游合并表共有 2,653 条：

| 处理 | 数量 | 结果 |
|---|---:|---|
| 缺少 `XYZ_Path` | 45 | 移至 `missing_xyz_path.csv` |
| 不满足新点群接纳规则 | 76 | 移至 `rejected_pg_rule.csv` |
| 最终保留 | 2,532 | 写入 `final_dataset.csv` |

最终来源：

| Source | 分子数 |
|---|---:|
| original_remaining | 1,175 |
| original_augmented | 86 |
| hierarchical_v1 | 113 |
| hierarchical_v2 | 48 |
| pycofbuilder | 1,110 |
| **合计** | **2,532** |

## 2.6 canonical v2：面向模型训练的严格重打包

原始 CSV/XYZ 随后被整理为模型无关的 `generative_model/data/processed/v2`：

```text
final_dataset.csv + 2,532 XYZ
        ├── cof_graphs.npz       canonical graph/atom/coordinate ground truth
        ├── symmetry_data.npz    target/actual operations, permutations, orbits
        ├── rdkit_features.npz   可复现的衍生特征
        ├── metadata.csv         分子元数据和 split
        └── manifest.json        schema、版本、环境、hash、数据指纹
```

canonical ground truth 包括：

- 原子类型和原子序数；
- 形式电荷与自由基电子；
- 基于 SMILES/RDKit 的真实化学键；
- 质心归零的 XYZ 坐标；
- sparse 无向 `bond_index/bond_types`；
- `atom_offsets/bond_offsets` 紧凑存储。

元素词表固定为 13 种：

```text
H, C, N, O, F, B, P, S, Cl, Br, I, Si, Sn
```

Si 和 Sn 使用独立 token；Sn 共有 2 个分子、4 个原子，绝不映射为 C 或 Si。

## 2.7 严格点群标注

v2 不直接把历史 `Point_Group` 当作最终真值，而是在统一协议下重新分析：

- `PointGroupAnalyzer` symmetry tolerance：0.3 Å；
- eigen tolerance：0.01；
- operation matrix tolerance：0.1；
- 原子匹配按元素分块执行 Hungarian assignment；
- 验证群操作闭包和 permutation composition。

每个分子区分保存：

| 字段 | 含义 |
|---|---|
| `target_pg` | 模板构造时请求的点群族 |
| `analyzer_pg` | pymatgen 初始候选标签 |
| `actual_pg` | 完整操作/置换/闭包验证后的标签 |
| `pg_exact_match` | target 与 actual 是否相同 |
| `pg_compatible` | target 是否为 actual 的子群 |

严格验证结果：

- `target_exact=1,456/2,532=57.50%`；
- `target_compatible=2,532/2,532=100%`；
- actual operations 共 8,827 个；
- target operation RMS 中位数约 `7.63×10^-6 Å`。

## 2.8 数据规模与分布

### 分子、原子和键

| 指标 | 数值 |
|---|---:|
| 分子 | 2,532 |
| 原子 | 91,113 |
| 无向化学键 | 96,555 |
| 原子数范围 | 9–105 |
| 原子数中位数 | 32 |
| 重原子数范围 | 8–77 |
| 重原子数中位数 | 22 |

### Target_PG

| Target_PG | 数量 | 占比 |
|---|---:|---:|
| C2 | 2,019 | 79.74% |
| C3 | 461 | 18.21% |
| S4 | 38 | 1.50% |
| D6h | 14 | 0.55% |

D6h 必须标记为 low-support/exploratory，不应作为主要统计结论。

### 元素计数

| H | C | N | O | F | B | P | S | Cl | Br | I | Si | Sn |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 28,038 | 47,128 | 5,231 | 6,253 | 1,727 | 74 | 109 | 1,503 | 604 | 181 | 187 | 74 | 4 |

### Split

| split | train | validation | test | 原则 |
|---|---:|---:|---:|---|
| IID | 2,026 | 253 | 253 | 按 Target_PG 分层固定划分 |
| Core-OOD | 2,024 | 250 | 258 | Core 分组隔离 |

Core-OOD 并不等于 fingerprint 距离很大。其 test-to-train 最大 Morgan/Tanimoto 均值为
`0.6760`，且有 57 条为 1.0，因此任何 OOD 结论都必须同时报告 scaffold/fingerprint 审计。

## 2.9 数据集优势

1. 具有 COF 构筑单元领域特异性；
2. 同时保存 2D 图、真实键、显式 H、3D 坐标和点群操作；
3. operations、permutations 和 atomic orbits 可以直接监督 symmetry-aware 模型；
4. 覆盖 Si/Sn 等通用小分子 benchmark 常缺少的元素；
5. 数据构建、split、环境和 SHA-256 可追溯；
6. canonical 数据与模型 adapter 分离，便于比较不同模型。

## 2.10 数据局限和后续改进

| 局限 | 对模型的影响 | 改进方案 |
|---|---|---|
| 只有 2,532 条 | 高容量模型容易记忆 | 通用 3D 数据预训练 + 对称数据微调 |
| 类别严重不平衡 | 模型忽略 S4/D6h | PG-balanced sampler/loss；扩充稀有类 |
| 一条 SMILES 一个构象 | 无法学习完整构象分布 | 同一图保存多个低能对称构象并按 graph 分 split |
| 坐标主要来自 ETKDG/MMFF | 可能学习算法偏差 | xTB/DFT 重优化子集；加入 GEOM/QMugs |
| template 组合偏置 | novel graph 泛化有限 | 外部 scaffold、quotient graph 和无模板候选 |
| 高对称群样本少 | 条件区分能力弱 | 分阶段先 C2/C3，再扩展 S4/D6h |
| compatible 容许超群 | 可能掩盖条件坍缩 | 同图 swapped-PG 反事实 Gate |

## 2.11 权威文件

- [数据集构建完整流程](../数据集构建完整流程.md)
- [pipeline README](../../cof_symmetry_pipeline/README.md)
- [final_dataset.csv](../../cof_symmetry_pipeline/output/final_dataset.csv)
- [v2 statistics.json](../../generative_model/data/processed/v2/statistics.json)
- [v2 validation_report.json](../../generative_model/data/processed/v2/validation_report.json)
- [v2 manifest.json](../../generative_model/data/processed/v2/manifest.json)

