# COF 对称分子数据集 — 管线文档

**路径**: `cof_symmetry_pipeline/` | **分支**: `june_a6000` | **更新**: 2026-08-08

---

## 概述

COF（共价有机框架）对称中间体核心分子数据集构建管线。通过"核心模板 + 侧链臂"的组合合成策略，程序化生成具有指定点群对称性的有机分子，经 2D→3D→力场松弛→对称破缺检测四阶段筛选，输出可用于训练的 3D 分子数据集。

## 目录结构

```
cof_symmetry_pipeline/
├── config.py                配置中心：核心/臂模板库 + 运行参数
├── generator.py             Stage 1：2D 拓扑对称组合生成
├── relaxer.py               Stage 2-3：3D嵌入 + 点群分析 + 力场松弛 + 对称破缺检测
├── main.py                  CLI 入口 + 多进程编排
├── dataset.py               PyTorch Dataset / DataLoader 接口
├── rebuild_config.py        工具：从数据集反推 config.py
│
├── validate_augmented.py    工具：对 2D 数据集跑 3D 对称性验证
│
├── visualization/
│   └── visualize.py         8 张可视化图表（点群分布、热力图、结构画廊等）
│
├── output/                  数据集输出
│   ├── augmented_dataset.csv     2D 拓扑数据集（2,752 分子）
│   ├── augmented_3d_pass.csv     3D 验证 PASS（135 分子）
│   ├── augmented_3d_screening.csv 3D 筛查全量（含 FAIL）
│   ├── xyz/                      XYZ 坐标文件
│   ├── hierarchical/             层级核心数据集（125 PASS）
│   └── README.md                 输出文件详细说明
│
└── checkpoints/             模型检查点（预留）
```

## 快速开始

```bash
# 原始核心生成（41 核心 × 70 臂）
python cof_symmetry_pipeline/main.py --workers 8

# 层级核心生成（265 核心，MW 过滤 + 优化）
python cof_symmetry_pipeline/main.py --hierarchical --workers 8

# 自定义核心和臂
python cof_symmetry_pipeline/main.py --cores triazine_C3 biphenyl_C2 --arms CHO_ph NH2_ph

# 3D 验证已有 2D 数据集
python cof_symmetry_pipeline/validate_augmented.py --input output/augmented_dataset.csv --workers 8
```

## 四阶段管线

```
Stage 1 (generator.py)
  输入: CORE_TEMPLATES + ARM_LIBRARY (config.py)
  方法: ReplaceSubstructs → _cleanup_bridges → SanitizeMol
  输出: 2D 拓扑对称分子（SMILES）

Stage 2 (relaxer.py / Embedder)
  输入: 2D Mol（无 H）
  方法: ETKDGv3 距离几何嵌入（8 个随机种子选优）
  输出: 含 H 的 3D 构象 Mol

Stage 3 (relaxer.py / GeometryRelaxer + SymmetryAnalyzer)
  输入: 3D 构象 Mol
  方法: MMFF94 力场优化 → pymatgen PointGroupAnalyzer
  输出: 松弛后坐标 + Schoenflies 点群符号 + MMFF 能量
  淘汰: FAIL_EMBED / FAIL_RELAX / FAIL_ENERGY / FAIL_SYMMETRY_BREAK

Stage 4 (main.py)
  输出: CSV 数据集 + XYZ 坐标 + 统计 JSON
```

## 核心概念

### 模板库

- **核心模板** (`CORE_TEMPLATES`): 41 原始 + 414 层级 + 56 pyCOFBuilder = 511 个
  - `*` 标记对称反应位点，位于分子图自同构轨道的等效位置
  - 原始核心按臂数分：C₂(23) / C₃(13) / C₄(4) / H₆(1)
  - 层级核心 = C₃/C₄/H₆ 节点 + C₂ 扩展臂（18 × 23 = 414）
  - pyCOFBuilder 来源: 从 `pycofbuilder/data/core/` 导入，[Q]→*, [R*]→H 转换

- **臂模板** (`ARM_LIBRARY`): 75 原始 + 7 pyCOFBuilder = 82 个
  - 格式: `*─[spacer]─官能团`
  - 连接子类型: 直接挂载(49) / 苯基(10) / 联苯(5) / 乙炔桥(15)

### 对称性保证

1. **2D 拓扑**: `*` 位于图自同构轨道的等效位置，`ReplaceSubstructs` 等价替换 → 拓扑对称 = 核心对称
2. **3D 几何**: pymatgen `PointGroupAnalyzer` 计算 Schoenflies 符号，松匹配（C₃ₕ ∈ C₃ 等价类）
3. **对称破缺**: 柔性臂旋转 → 3D 构象破缺 → `FAIL_SYMMETRY_BREAK`

### 数据集规模

| 数据集 | 2D 分子 | 3D PASS | 来源 |
|------|:--:|:--:|------|
| 原始核心 | 2,752 | 1,348 | 41 原始核心 × 70 臂 |
| 层级核心 | 32,341 | 177 | 414 层级核心（两轮 MW 过滤） |
| pyCOFBuilder | 3,922 | 1,159 | 56 pb_ 核心 × 77 臂 |
| **合计** | **~39,000** | **2,653** | 去重后 (`cof_dataset_3d_pass.csv`) |

最终数据集: 
- `cof_dataset_3d_pass.csv` — 2,653 个 3D 验证分子（含能量、点群、`Dimension='3D'`）
- `cof_dataset_full.csv` — 4,040 个分子（2,653 3D + 1,387 仅2D），`Dimension` 列区分

## 配置说明

```python
# PipelineConfig 主要参数
embedding.num_conformer_attempts = 8   # 构象尝试数（层级模式 = 3）
relax.backend = "rdkit_mmff"           # 力场后端
relax.max_steps = 100                  # 最大优化步数
relax.force_threshold = 5.0            # 原子位移淘汰阈值
max_heavy_atoms = 0                    # 重原子数上限（0=不过滤，层级=75）
hierarchical_mode = False              # 层级核心模式
```

## 依赖

```
rdkit, pymatgen, numpy, pandas, matplotlib, torch (dataset.py 可选)
```

## 相关文档

- `docs/数据集构建完整流程.md` — 详细技术文档
- `docs/interaction_log.md` — 完整交互记录
- `docs/core_doc.md` — 41 个原始核心参考文献
- `output/README.md` — 输出文件说明
