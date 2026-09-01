# 数据格式说明

## 1. 文件总览

每个对称类一个子目录，共两组：

```
outputs/samples_c3_labels_v3/
├── C3k3/                              # C3 对称, 轨道数 k=3 → N = 3×3 = 9 原子
│   ├── graphs_C3k3.npz                # ★ 核心: 顶点特征 + 邻接矩阵 (74 个唯一分子)
│   ├── labels_C3k3.txt                # ★ 标签/索引文件 (SMILES + 对称参数)
│   ├── mol_001_C3k3.png ... mol_074_C3k3.png   # 逐分子结构图 (修正前/后对照)
│   ├── summary_C3k3_generated.png     # 汇总图: 修正前
│   └── summary_C3k3_corrected.png     # 汇总图: 修正后
└── C3k2/                              # C3 对称, 轨道数 k=2 → N = 3×2 = 6 原子
    ├── graphs_C3k2.npz                # (24 个唯一分子)
    ├── labels_C3k2.txt
    ├── mol_001_C3k2.png ... mol_024_C3k2.png
    ├── summary_C3k2_generated.png
    └── summary_C3k2_corrected.png
```

> 说明：`C3k3` 表示**对称阶 n=3**、**轨道数 k=3**。轨道划分为 k 条轨道，每条轨道 n=3 个等价原子，原子总数 $N = n \times k$：C3k3 → 9 原子，C3k2 → 6 原子。

**数量统计**（11 个随机种子 × 每种子 100 个原始样本 → 化学后处理 + 连通性筛选 → InChIKey 去重）：

| 类 | N | 累计连通保留 | **去重后（唯一）** |
|----|---|-------------|-------------------|
| C3k3 | 9 | 484 | **74** |
| C3k2 | 6 | 295 | **24** |
| 合计 | | 779 | **98** |

所有保留分子已用 nauty 复验：**98/98 均具有非平凡自同构**（满足 C3 对称）。部分分子实际对称性高于 C3（如苯环、环己烷为 C6），因为 C3 是 C6 的子群，投影算子保证的是**至少** C3 对称。

**去重方式**：用 RDKit `MolToInchiKey`（InChIKey）作为分子唯一性指纹，同一化学结构无论 SMILES 写法差异都归为同一 key；每个去重组保留第一个样本。C3k2 唯一分子较少是因为 6 原子 C3 对称分子的异构体空间有限（苯、环己烷、三取代环丙烷等为主）。

---

## 2. 核心文件：`graphs_*.npz`

使用 `numpy.load(path, allow_pickle=True)` 读取。包含以下字段：

| 字段 | 形状 | 类型 | 含义 |
|------|------|------|------|
| `x` | `[B, N, D]` | float32 | **顶点特征**（one-hot 连续值，来自采样输出） |
| `x_idx` | `[B, N]` | int32 | 顶点元素索引（`argmax(x, axis=-1)`，离散化） |
| `adj_gen` | `[B, N, N]` | int8 | **修正前**邻接矩阵（原始生成，键型 0/1/2/3/4） |
| `adj_corr` | `[B, N, N]` | int8 | **修正后**邻接矩阵（化学后处理，键型 0/1/2/3/4） |
| `n_atoms` | `[B]` | int32 | 每分子原子数（全连通筛选后 = N） |
| `smiles` | `[B]` | object(str) | 修正后 SMILES（推荐使用） |
| `smiles_gen` | `[B]` | object(str) | 修正前 SMILES |
| `order` | `[B]` | int32 | 对称阶 n（全为 3） |
| `orbit` | `[B]` | int32 | 轨道数 k（3 或 2） |
| `atom_vocab` | `[D]` | S4 (bytes) | 原子词汇表 `['C','N','O','F']` |

其中 `B` = 分子数（74 或 24），`N` = 原子数（9 或 6），`D` = 元素类型数（4：C/N/O/F）。

### 2.1 键型编码（adj_gen / adj_corr）

| 值 | 含义 |
|----|------|
| 0 | 无键 |
| 1 | 单键 |
| 2 | 芳香键 |
| 3 | 双键 |
| 4 | 三键 |

邻接矩阵为**对称矩阵**（`adj[i,j] == adj[j,i]`），对角为 0。

### 2.2 顶点特征（x / x_idx）

- `x[i]` 是分子 i 的顶点特征，`x[i, v, :]` 是第 v 个原子的特征向量。
- 由于来自扩散模型采样输出，`x` 是**连续软 one-hot**（各行之和 ≈ 1 但不严格等于 1）。
- 如需离散原子类型，直接用 `x_idx[i, v]` → `atom_vocab[x_idx[i, v]]`，例如 `atom_vocab[0]='C'`。

### 2.3 读取示例

```python
import numpy as np

d = np.load("outputs/samples_c3_labels_v3/C3k3/graphs_C3k3.npz", allow_pickle=True)
atom_vocab = [v.decode() for v in d["atom_vocab"]]

# 第 0 个分子
N   = int(d["n_atoms"][0])
n   = int(d["order"][0])     # 对称阶 = 3
k   = int(d["orbit"][0])     # 轨道数 = 3
x   = d["x"][0]              # [9, 4]  顶点特征
adj = d["adj_corr"][0]       # [9, 9]  修正后邻接
smi = d["smiles"][0]         # SMILES

print("元素:", [atom_vocab[i] for i in d["x_idx"][0]])
print("SMILES:", smi)
```

---

## 3. 标签文件：`labels_*.txt`

制表符分隔（TSV），每行一个分子：

```
# C3k3  N=9  C3 symmetry  (state_proj, multi-seed, InChIKey dedup)
# 累积 484, 去重后 74
idx_001  -  3  3  CN1CN(C)CN(C)C1
idx_002  -  3  3  C=C1CC(=C)CC(=C)C1
...
```

| 列 | 含义 |
|----|------|
| 1 | 去重后顺序编号 `idx_001`（与 `mol_*.png` 文件名对应） |
| 2 | 原种子内编号（多子集成因未保留，固定为 `-`） |
| 3 | 对称阶 n |
| 4 | 轨道数 k |
| 5 | 修正后 SMILES |

以 `#` 开头的是注释行（含类名、对称信息、生成/保留数量）。`labels_*.txt` 与 `graphs_*.npz` 的**行顺序一一对应**：labels 第 j 行 ↔ graphs 中第 j 个样本。

---

## 4. 图片文件

### 4.1 逐分子图 `mol_XXX_C3kX.png`（每分子一张）

每张图包含该分子的**修正前（Before）**与**修正后（After）**两个结构，顶部标注：
- 对称标签 `C3k3` / `C3k2`（含 N）
- 去重后编号
- 修正前/后的 SMILES

### 4.2 汇总图 `summary_*_generated.png` / `summary_*_corrected.png`

- 把所有去重后分子的**结构**按网格排列（不含邻接矩阵）。
- `_generated` = 修正前，`_corrected` = 修正后。
- 各一张，便于快速浏览整批结果。

---