# COF 对称分子数据集——输出说明

**最后核对**：2026-08-09
**唯一数据事实源**：`final_dataset.csv`

## 当前交付文件

| 文件 | 行数 | 用途 |
|---|---:|---|
| `final_dataset.csv` | 2,532 | 当前可用于三维训练的数据；SMILES 唯一、XYZ 可访问且满足当前 PG 规则 |
| `missing_xyz_path.csv` | 45 | 从原 2,653 条整理结果中隔离的空 XYZ 路径记录，不进入训练 |
| `rejected_pg_rule.csv` | 76 | 具有 XYZ、但不满足当前 `PG_ALLOWED_ACTUAL` 的记录，不进入训练 |

`final_dataset.csv` 由历史上游合并表 `cof_dataset_3d_pass.csv` 整理得到。
分来源目录和中间 CSV 仅用于追溯构建过程，不作为训练入口。

## 字段

| 字段 | 类型 / 单位 | 含义 |
|---|---|---|
| `SMILES` | 字符串 | 分子的二维连接关系和化学组成；也是当前数据集的去重键，每一行均唯一。氢原子通常以隐式形式表示，实际三维文件中包含显式氢。 |
| `Core` | 字符串 | 生成该分子时采用的核心模板名称，对应 `config.py` 中的核心或导入核心标识，例如 `azobenzene_C2`、`pb_BDFN_L2`。它是模板 ID，不是重新计算得到的结构分类。 |
| `Arm` | 字符串 | 与核心连接位点组合的臂/官能团模板名称，例如 `CHO_phph`、`CONH2_direct`。它同样是生成阶段的模板 ID。 |
| `Target_PG` | Schoenflies 点群符号 | 核心模板设计时指定的目标对称族，例如 `C2`、`C3`、`C4`、`S4`、`D6h`；用于三维筛选，不一定与最终实际点群完全相同。 |
| `Point_Group` | Schoenflies 点群符号 | 分子经过三维嵌入和几何优化后，由点群分析得到的实际点群。该值必须满足 `PG_ALLOWED_ACTUAL[Target_PG]` 才会进入最终数据集。 |
| `Geometry` | 分类字符串 | 基于 `Plane_RMSD` 得到的整体平面性标签：`planar` 表示 `Plane_RMSD ≤ 0.5 Å`，`nonplanar` 表示 `Plane_RMSD > 0.5 Å`。它描述当前 XYZ 构象，不等同于点群标签。 |
| `Energy_kcal_mol` | 浮点数，kcal/mol | 三维构象经过力场几何优化后记录的总能量。管线默认使用 MMFF94，参数不完整时可回退到 UFF。该值主要用于构象筛选和质量检查，不是实验生成焓，也不宜当作不同分子间的绝对稳定性标尺。 |
| `XYZ_Path` | 字符串，文件路径 | 与该行分子对应的 XYZ 三维坐标文件。文件第一行是原子数，随后记录元素符号及 Å 单位的笛卡尔坐标；当前 CSV 保存的是本机绝对路径。 |
| `Source` | 分类字符串 | 数据来源/构建批次。当前取值为 `original_augmented`、`original_remaining`、`hierarchical_v1`、`hierarchical_v2`、`pycofbuilder`。 |
| `Plane_RMSD` | 浮点数，Å | XYZ 中全部原子到最佳拟合平面的垂直距离均方根；通过中心化坐标的 SVD 计算。越接近 0，整体越接近平面。该列是 `Geometry` 的直接分类依据。 |
| `Plane_MaxDist` | 浮点数，Å | XYZ 中所有原子到同一最佳拟合平面的最大绝对垂直距离，用于识别局部偏离平面最明显的原子；它不参与当前 `Geometry` 的阈值分类。 |

`Target_PG` 表示“设计目标”，`Point_Group` 表示“优化后实测结果”。例如目标为
`C2` 的分子实际得到 `D2`，只要该结果属于当前 C2 接纳集合，仍可保留。

`Source` 的具体含义如下：

| 取值 | 来源 |
|---|---|
| `original_augmented` | 原始核心数据中先行完成三维筛选的数据增强批次 |
| `original_remaining` | 原始核心数据的后续补充筛选批次 |
| `hierarchical_v1` | 层级核心构建流程第一批结果 |
| `hierarchical_v2` | 层级核心构建流程补充批次 |
| `pycofbuilder` | 从 pyCOFBuilder 核心和连接体模板导入并筛选的结果 |

文件不包含 `Dimension` 或 `Status` 字段；其中每一行都是经过三维筛选且当前具有
有效 XYZ 文件的记录。

## 当前统计

| 指标 | 值 |
|---|---:|
| 唯一分子 | 2,532 |
| Core / Arm | 98 / 75 |
| 实际点群 | 14 类 |
| planar / nonplanar | 1,202 / 1,330 |
| 平均能量 | 47.060 kcal/mol |
| 能量范围 | -622.632 ～ 760.772 kcal/mol |

来源分布：`original_remaining=1175`、`original_augmented=86`、
`hierarchical_v1=113`、`hierarchical_v2=48`、`pycofbuilder=1110`。

目标对称族分布：`C2=2019`、`C3=461`、`S4=38`、`D6h=14`；当前 C4 为 0。

## 使用

```python
import pandas as pd

df = pd.read_csv("cof_symmetry_pipeline/output/final_dataset.csv")
assert len(df) == 2532
assert df["SMILES"].is_unique
assert df["XYZ_Path"].notna().all()
```

注意：`dataset.py` 尚未作为当前训练入口。其原子词表已支持 Si、Sn，未知元素也会
显式报错；启用前仍需将路径字段由旧的 `XYZ_File_Path` 改为本数据集使用的
`XYZ_Path`。
