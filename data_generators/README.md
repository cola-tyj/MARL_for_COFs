# data_generators — 分子图数据生成器

可扩展的环 + 取代基分子图生成器，输出纯数据格式（邻接矩阵 + 节点特征）。

## 快速开始

```bash
# 默认：C3 苯环 + 甲基/乙基/异丙基，5000 图
python generate.py

# 自定义参数
python generate.py --ring-size 6 --sym 3 --attach-pos 0,2,4 \
                   --subst methyl,ethyl,isopropyl --num 5000 --seed 42

# 只生成异丙基
python generate.py --subst isopropyl --num 1000

# 不可视化
python generate.py --vis-samples 0
```

## 输出

生成后 `output/` 目录下包含：

| 文件 | 说明 |
|------|------|
| `ring6_C3_met_eth_iso_5000.pkl` | `List[dict]`，每个 dict 含 `adj`, `x`, `n`, `k`, `subst_type` |
| `ring6_C3_met_eth_iso_5000.txt` | JSON 元数据（统计信息） |
| `ring6_C3_met_eth_iso_5000_samples.png` | 可视化样本 |

## 输出格式

```python
{
    'adj': np.ndarray (N, N),    # 二值邻接矩阵（int8）
    'x':   np.ndarray (N, 4),    # 节点特征 one-hot（第0列=碳）
    'n':   3,                    # C_n 对称阶数
    'k':   N // 3,               # 轨道数
    'subst_type': 'isopropyl',   # 取代基类型
}
```

## 模块说明

| 模块 | 职责 |
|------|------|
| `ring_builder.py` | `build_ring(n_nodes)` — 构建碳环 |
| `substituents.py` | 取代基配方注册表（分步挂甲基） |
| `assembler.py` | `assemble(ring, subst, positions)` — 拼接 |
| `utils.py` | 打乱、SMILES、可视化、格式转换 |
| `generate.py` | CLI 批量生成入口 |

## 添加新取代基

在 `substituents.py` 的 `SUBSTITUENT_RECIPES` 字典中添加条目：

```python
SUBSTITUENT_RECIPES = {
    ...
    'tert-butyl': [
        {'parent': 'ring',   'name': 'central_c'},
        {'parent': 'step_0', 'name': 'methyl_1'},
        {'parent': 'step_0', 'name': 'methyl_2'},
        {'parent': 'step_0', 'name': 'methyl_3'},
    ],
}
```

每个 step 描述一次「在每个父节点上挂一个甲基」操作：
- `parent: 'ring'` — 挂到环的挂载点
- `parent: 'step_N'` — 挂到第 N 步新增的节点

## 依赖

```
networkx
numpy
rdkit       # 可选，用于 SMILES 可视化
matplotlib  # 可选
```
