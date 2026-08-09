# COF 对称核心数据集 — 输出文件说明

**最后更新**: 2026-08-08 | **数据集版本**: v2.0 (2,601 分子)

---

## 目录结构

```
output/
├── README.md                          ← 本文件
│
├── cof_dataset_3d_pass.csv           [主] 已验证数据集（2,653 分子，Dimension='validated'）
├── cof_dataset_full.csv              [主] 联合数据集（4,040 分子，Dimension: validated/topology）
│
├── augmented_dataset.csv             [源] 原始核心 2D 数据集（2,752 行）
├── augmented_3d_pass.csv             [源] 原始核心数据增强 PASS（135）
├── augmented_3d_screening.csv        [副] 原始核心 3D 筛查全量（1,226）
│
├── original_remaining_pass.csv       [源] 原始核心剩余筛查 PASS（1,213）
├── original_remaining_screening.csv  [副] 原始核心剩余筛查全量（1,526）
├── unscreened_original.csv           [临时] 待筛查分子列表
│
├── xyz/                              [3D坐标] PASS 分子 XYZ 文件
│
├── hierarchical/                     [源] 层级核心数据集 (max_heavy=75)
│   ├── hierarchical_cores_dataset.csv  125 PASS / 10,798 total
│   ├── pipeline_stats.json
│   └── xyz/
│
├── hierarchical_supplement/          [运行中] 层级核心补充 (max_heavy=100)
│
└── pycofbuilder/                     [源] pyCOFBuilder 核心数据集
    ├── pycofbuilder_cores_dataset.csv  1,159 PASS / 3,922 total
    ├── pipeline_stats.json
    └── xyz/
```

---

## 主数据集

### `cof_dataset_3d_pass.csv` （最终交付）

| 指标 | 值 |
|------|:--:|
| 独特分子 | 2,653 |
| 点群种类 | 17（C₂/C₂ₕ/C₂ᵥ/D₂/D₂ₕ/D₂d/C₃/C₃ₕ/C₃ᵥ/D₃/D₃ₕ/D₃d/S₄/D₆ₕ/C₆/C₆ₕ/C₁） |
| 平均能量 | 50.7 kcal/mol |
| 来源分布 | 原始(1,348) + 层级(177) + pyCOFBuilder(1,128) |
| 对称性保证 | 全部具有 ≥C₂ 旋转轴 |

列名: `SMILES`, `Core`, `Arm`, `Target_PG`, `Point_Group`, `Energy_kcal_mol`, `Source`, `Dimension`

### `cof_dataset_full.csv` （联合数据集）

| 指标 | 值 |
|------|:--:|
| 总分子 | 4,040 |
| validated | 2,653（已通过对称性验证，有 XYZ 坐标） |
| topology | 1,387（仅 SMILES 拓扑，尚未嵌入坐标） |

`Dimension` 列区分：`'validated'`=含点群+能量+XYZ，`'topology'`=仅 SMILES

### 来源文件

| 文件 | 行数 | 说明 |
|------|:--:|------|
| `augmented_dataset.csv` | 2,752 | 原始 41 核心 × 70 臂，仅 2D |
| `augmented_3d_pass.csv` | 135 | 原始核心数据增强阶段 PASS |
| `original_remaining_pass.csv` | 1,213 | 原始核心剩余分子 3D PASS |
| `hierarchical/hierarchical_cores_dataset.csv` | 10,798 | 层级核心全量筛查（125 PASS） |
| `pycofbuilder/pycofbuilder_cores_dataset.csv` | 3,922 | pyCOFBuilder 全量筛查（1,159 PASS） |

---

## 使用

```python
import pandas as pd

df = pd.read_csv("output/cof_dataset_3d_pass.csv")
# 按点群筛选
c3_mols = df[df['Point_Group'].str.contains('C3|D3')]
# 按来源筛选
pb_mols = df[df['Source'] == 'pycofbuilder']
```
