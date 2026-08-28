# COF 对称中间体核心数据集管线——设计与交付文档

> **历史设计（2026-07-18）**：本文记录 1,661/2,752 数据阶段及 JT-VAE 方案。当前数据与结论以 `docs/数据集构建完整流程.md` 和 `output/final_dataset.csv` 为准，JT-VAE 已停止。

**日期**: 2026-07-18（最终更新）  
**环境**: Linux, Python 3.10 (conda: `env_cof`)  
**路径**: `/home/tianyajun/MARL_for_COFs/cof_symmetry_pipeline/`

---

## 一、项目概述

### 目标

程序化自底向上合成指定点群对称性（C₂, C₃, C₄, H₆）的 COF 中间体核心数据集。四阶段管线：2D 拓扑组合生成 → 3D 构象嵌入 → 力场几何优化 → 对称破缺快筛 → PyTorch 数据集导出。

### 核心设计原则

1. **对称性由构造保证，而非事后验证**——所有 `*` 占位符被同一个臂等价替换，产物拓扑对称性 = 核心对称性
2. **物理真实性优先于数量**——对称破缺的分子（如甲酰基旋转导致 C₃ → Cₛ）必须淘汰
3. **模块化、可扩展**——新增核心/臂只需在 `config.py` 添加条目，无需改代码
4. **零外部依赖即可运行**——默认使用 RDKit 内置 MMFF94 力场

---

## 二、文件清单与作用

```
cof_symmetry_pipeline/
├── requirements.txt
├── config.py                 # 核心/臂模板库（57核心×70臂）
├── generator.py              # Stage 1: 2D 拓扑组合生成
├── relaxer.py                # Stage 2-3: 3D 嵌入 + 点群分析 + 几何优化
├── main.py                   # 管线编排器 + CLI + 多进程
├── dataset.py                # PyTorch Dataset + symmetry attention mask
├── visualization/            # 数据集可视化 (8 张图)
│   └── visualize.py
├── mining/                   # 外部数据库挖掘实验
│   ├── mine_moses.py         #   MOSES 1.58M 三阶段筛选
│   ├── screen_qm9_10k.py     #   QM9 10K 可行性验证
│   └── validate_augmented.py #   增强数据 3D 验证
├── jtvae/                    # JT-VAE 对称分子生成模型
│   ├── preprocess.py         #   联结树分解 + 词表构建
│   ├── model.py              #   变分自编码器
│   ├── train.py              #   训练脚本
│   ├── processed/            #   预处理数据
│   └── output/               #   模型权重
└── output/
    ├── final_combined_dataset.csv  # 最终数据集 (1661 分子, 3D 验证)
    ├── augmented_dataset.csv       # 增强数据集 (2752 分子, 2D 拓扑对称)
    ├── cof_cores_dataset.csv      # 原始 3D 验证 (3802 行)
    ├── xyz/                        # 1661 个 XYZ 文件
    └── mining/                     # 挖掘实验结果
```

---

## 三、各文件详细设计

### 3.1 `config.py`——模板库与配置中心

**路径**: `cof_symmetry_pipeline/config.py`

**核心数据结构**:

| 结构 | 类型 | 作用 |
|------|------|------|
| `CORE_TEMPLATES` | `Dict[str, dict]` | 7 个对称核心模板，每个含 `smiles`/`target_pg`/`n_arms`/`symmetry_family` |
| `ARM_LIBRARY` | `Dict[str, dict]` | 10 种侧链端基，每个含 `smiles`/`description`/`functional_group` |
| `EmbeddingConfig` | `@dataclass` | ETKDGv3 参数（最大尝试次数、RMSD 阈值） |
| `SymmetryConfig` | `@dataclass` | 点群分析容差（tolerance=0.3Å, eigen_tolerance=0.01） |
| `RelaxConfig` | `@dataclass` | 优化参数（后端、步数、力收敛阈值、淘汰阈值） |
| `PipelineConfig` | `@dataclass` | 总管线配置（输出路径、日志级别、随机种子） |
| `PG_EQUIVALENCE` | `Dict[str, set]` | 点群等价类——允许的高/低对称松匹配 |
| `BROKEN_SYMMETRY_TAGS` | `set` | 判定为对称破缺的低对称标记 `{C1, Cs, Ci}` |

**核心模板**（SMILES 中的 `*` 是 RDKit dummy atom, Z=0）：

| 键 | 对称族 | Arm 数 | 说明 |
|----|:------:|:------:|------|
| `triazine_C3` | C₃ | 3 | 1,3,5-三嗪，* 位于 C-2,4,6 位 |
| `triphenylbenzene_C3` | C₃ | 3 | 1,3,5-三苯基苯，* 位于各苯环对位 |
| `porphyrin_C4` | C₄ | 4 | 卟啉 meso-四取代 |
| `tetraphenylmethane_C4` | C₄ | 4 | 四苯基甲烷，* 位于各苯环对位 |
| `biphenyl_C2` | C₂ | 2 | 4,4'-联苯 |
| `pyrene_dimer_C2` | C₂ | 2 | 芘并二联 |
| `hexaphenylbenzene_H6` | D₆ₕ | 6 | 六苯基苯，* 位于各苯环对位 |

**臂模板**（按官能团分四族）：

| 族 | 变体 | 键示例 |
|----|------|--------|
| 醛基 (-CHO) | 直接 / 苯基 / 联苯基 | `CHO_direct`, `CHO_ph`, `CHO_phph` |
| 氨基 (-NH₂) | 直接 / 苯基 / 联苯基 | `NH2_direct`, `NH2_ph`, `NH2_phph` |
| 硼酸基 (-B(OH)₂) | 直接 / 苯基 | `BOH2_direct`, `BOH2_ph` |
| 乙炔基 (-C≡CH) | 苯基 / 联苯基 | `CCH_ph`, `CCH_phph` |

**点群等价类设计逻辑**:

```python
PG_EQUIVALENCE = {
    "C3":  {"C3", "C3v", "C3h", "D3", "D3h", "D3d"},   # 所有包含 C₃ 轴的点群
    "C4":  {"C4", "C4v", "C4h", "S4", "D2d", "D4", "D4h", "D4d"},
    "C2":  {"C2", "C2v", "C2h", "D2", "D2h", "D2d"},
    "D6h": {"D6h", "D6", "D3h", "D3d", "C6v", "C6h", "C6", "D6d"},
}
```

ETKDG 嵌入的数值偏差可能导致 Schoenflies 符号在等价类间漂移（如 C₃ → C₃ᵥ）。等价类松匹配容忍这种扰动。**关键是**：实际点群必须是目标点群的等价类或超群，接受更高对称性（如 D₃ₕ 包含 C₃ 轴），拒绝更低对称性（C₁, Cₛ, Cᵢ → `BROKEN_SYMMETRY_TAGS`）。

---

### 3.2 `generator.py`——2D 拓扑组合生成

**路径**: `cof_symmetry_pipeline/generator.py`

**类与函数**:

| 名称 | 类型 | 作用 |
|------|------|------|
| `_cleanup_bridges(mol)` | 函数 | 移除 ReplaceSubstructs 产生的桥接 `*` 原子 |
| `MoleculeGenerator` | 类 | 组合合成引擎 |
| `generate_dataset()` | 函数 | 一键生成 + 去重 + 验证 |

**`MoleculeGenerator.generate_one(core_name, arm_name)` 核心算法**:

```
输入: 核心 SMILES (*c1nc(*)nc(*)n1)  +  臂 SMILES (*c1ccc(C=O)cc1)

Step 1: AllChem.ReplaceSubstructs(core, query='*', replacement=arm, replaceAll=True)
        核心的 3 个 * 被替换为 3 个臂拷贝
        → 中间体含 3 个桥接 * 原子（degree=2，一侧连核心锚点、一侧连臂苯环）

Step 2: _cleanup_bridges(mol)
        遍历所有 atomic_num==0 且 degree==2 的原子
        → 降序移除桥接 * → 邻居索引校正 → 创建直连 C-C 键

Step 3: Chem.SanitizeMol(mol)
        芳香性感知 + 化合价校验

输出: 纯 2D 分子（无 dummy 原子），含元数据 (core_name, arm_name, target_pg)
```

**`_cleanup_bridges` 索引校正算法**:

```python
bridge_indices = sorted([b[0] for b in bridges])

# 降序移除
for bridge_idx in sorted(bridges, reverse=True):
    mol_rw.RemoveAtom(bridge_idx)

# 邻居索引 = 原始索引 - 已移除的、索引更小的桥接原子数
for _, n1, n2 in bridges:
    adj_n1 = n1 - sum(1 for bi in bridge_indices if bi < n1)
    adj_n2 = n2 - sum(1 for bi in bridge_indices if bi < n2)
    mol_rw.AddBond(adj_n1, adj_n2, SINGLE)
```

**为什么降序移除**：从高索引向低索引移除保证每次移除只影响索引更高的原子，而已经被处理的原子不再受影响。

**`MoleculeGenerator.deduplicate()`**：基于 `(CanonicalSMILES, MolecularFormula)` 二元组去重，不同核心-臂组合可能产生相同产物（如对称性更高的偶合）。

**`MoleculeGenerator.validate_mol()`**：四级验证——Sanitize → 无 dummy 残留 → 化合价检查 → 原子数合理性(5-500) → 碳骨架存在。

---

### 3.3 `relaxer.py`——3D 嵌入 + 点群分析 + 几何优化 + 对称破缺检测

**路径**: `cof_symmetry_pipeline/relaxer.py`

**四个核心类**:

```
Embedder ──────────→ SymmetryAnalyzer ──────────→ GeometryRelaxer
(ETKDGv3 嵌入)       (pymatgen 点群)              (MMFF/UFF/MACE/xTB)
                                │                          │
                                └──── SymmetryScreener ────┘
                                      (串联三者，统一输出)
```

#### Embedder

| 步骤 | 操作 |
|------|------|
| 1 | `Chem.AddHs(mol)` — 加氢 |
| 2 | `AllChem.ETKDGv3()` — 随机坐标 + 最多 100 次尝试 |
| 3 | `AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)` — 初步键长键角优化 |

返回 `(mol_h, success, status)`。嵌入失败（返回码≠0）= 位阻冲突严重 → `FAIL_EMBED`。

#### SymmetryAnalyzer

**主路径**（`HAS_PYMATGEN=True`）:

```python
# RDKit → pymatgen
species = [atom.GetSymbol() for atom in mol.GetAtoms()]
coords = [[pos.x, pos.y, pos.z] for pos in conf.GetAtomPosition(i)]
mol_pmg = PmgMolecule(species, coords)

# 分析（tolerance=0.3Å）
analyzer = PointGroupAnalyzer(mol_pmg, tolerance=0.3, eigen_tolerance=0.01)
sch = analyzer.sch_symbol  # Schoenflies 符号
```

**回退路径**（`HAS_PYMATGEN=False`）:

惯性张量简并度推测：
- 2 个简并本征值 → C₃⁺（存在 ≥3 重旋转轴）
- 1 个简并本征值 → C₂
- 无简并 → C₁

#### GeometryRelaxer

**后端优先级与实现**:

| 后端 | 来源 | 可用性 | 能量单位 | "力"度量 |
|------|------|:------:|:--------:|----------|
| `rdkit_mmff` | RDKit 内置 | ✅ 始终 | kcal/mol | 最大原子位移 (Å) |
| `rdkit_uff` | RDKit 内置 | ✅ 始终 | kcal/mol | 最大原子位移 (Å) |
| `mace` | `mace-torch` | ⚠️ 需安装 | eV | 残余力 (eV/Å) |
| `tblite` | `tblite` | ⚠️ 需安装 | eV | 残余力 (eV/Å) |

**RDKit MMFF94 优化流程**:

```python
coords_before = get_coords(mol)
ff = AllChem.MMFFGetMoleculeForceField(mol, 
      AllChem.MMFFGetMoleculeProperties(mol))
ff.Initialize()
ff.Minimize(maxIts=max_steps)          # 迭代优化
coords_after = get_coords(mol)
max_disp = max(|coords_after - coords_before|)  # 最大原子位移作为"力"的代理
energy = ff.CalcEnergy()
```

若 MMFF 参数化失败（某些官能团无参数），自动回退到 UFF。

**ASE 后端优化流程**（MACE/xTB 可用时）:

```python
# RDKit → ASE
atoms = AseAtoms(symbols=symbols, positions=positions)
atoms.calc = MACECalculator(model_path='MACE-OFF23')

# BFGS 优化
opt = BFGS(atoms)
opt.run(fmax=0.05 eV/Å, steps=max_steps)

# 力检查
forces = atoms.get_forces()
max_force = max(√(fx²+fy²+fz²))
```

#### SymmetryScreener

**`screen(mol_2d) → Dict` 完整判定流程**:

```
1. Embedder.embed()
   ├─ 失败 → FAIL_EMBED, return
   └─ 成功 → mol_h (含 H)

2. SymmetryAnalyzer.analyze(mol_h)
   └─ → initial_pg

3. GeometryRelaxer.relax(mol_h)
   ├─ 失败 → FAIL_RELAX, return
   └─ 成功 → relaxed_mol, energy, max_disp

4. 能量/力筛查
   ├─ |E| > 1e8 或 NaN → FAIL_ENERGY
   ├─ max_disp > force_threshold → FAIL_FORCE
   └─ 通过 → 继续

5. 对称破缺检测
   ├─ final_pg ∈ BROKEN_SYMMETRY_TAGS → FAIL_SYMMETRY_BREAK
   ├─ final_pg == target_pg → PASS
   ├─ final_pg ∈ PG_EQUIVALENCE[target_pg] → PASS
   ├─ target_pg ∈ PG_EQUIVALENCE[final_pg] → PASS
   └─ 否则 → FAIL_SYMMETRY_BREAK

6. PASS → result['mol'] = relaxed_mol
```

**返回 Dict 字段**: `smiles`, `mol`, `target_pg`, `initial_pg`, `final_pg`, `symmetry_preserved`, `energy`, `energy_per_atom`, `max_force`, `xyz_path`, `status`, `error`

---

### 3.4 `main.py`——管线编排器

**路径**: `cof_symmetry_pipeline/main.py`

**核心类**: `COFPipeline`

| 方法 | 阶段 | 作用 |
|------|:----:|------|
| `stage1_generate()` | 1 | 调 `generate_dataset()` 生成 2D 分子 |
| `stage2_3_screen()` | 2+3 | 根据 `num_workers` 选择串行/并行筛查 |
| `_screen_sequential()` | 2+3 | 单进程逐分子处理，PASS 时写入 XYZ |
| `_screen_parallel()` | 2+3 | `ProcessPoolExecutor` + `_worker_initializer` + `_process_one` |
| `stage4_export()` | 4 | 构建 DataFrame → CSV + JSON 统计 |

**多进程设计**:

```
主进程                               子进程 1..N
  │                                      │
  ├─ ProcessPoolExecutor(                ├─ _worker_initializer(config, xyz_dir)
  │     initializer=_worker_initializer, │   └─ 创建独立 SymmetryScreener
  │     max_workers=N                    │   └─ 加载独立的计算器实例
  │   )                                  │
  │                                      ├─ _process_one((idx, mol_2d))
  ├─ executor.submit(_process_one, t)    │   └─ screener.screen(mol_2d)
  │   for t in tasks                     │   └─ PASS → write_xyz()
  │                                      │   └─ return result dict
  ├─ as_completed(futures)               │
  │   └─ 收集结果                        │
  └─ 按 mol_index 排序                   │
```

每个子进程独立初始化 `SymmetryScreener`（含独立的 MMFF 力场），避免跨进程共享 RDKit 对象的序列化问题和力场实例的线程安全问题。`_process_one` 的 timeout 为 600 秒/分子。

**CLI 参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cores` | 全部 7 个 | 空格分隔的核心名 |
| `--arms` | 全部 10 个 | 空格分隔的臂名 |
| `--backend` | `rdkit_mmff` | 优化后端 |
| `--max-steps` | 100 | 最大优化步数 |
| `--fmax` | 0.05 | 力收敛阈值 |
| `--force-threshold` | 3.0 | 原子位移/力淘汰阈值 |
| `--workers` | 1 | 并行进程数（1=串行） |
| `--output-dir` | `output/` | 输出目录 |
| `--dataset-demo` | False | 运行 PyTorch Dataset 演示 |

---

### 3.5 `dataset.py`——PyTorch 数据集接口

**路径**: `cof_symmetry_pipeline/dataset.py`

#### COFSymmetryDataset

**`__getitem__` 返回的 Tensor 结构**:

| 键 | Shape | 含义 |
|----|-------|------|
| `atom_tokens` | `(max_len,)` | 原子类型 token IDs (pad=0, H=1, C=2, N=3, O=4, ...) |
| `coords` | `(max_len, 3)` | 3D 笛卡尔坐标 (Å)，补齐部分为 0 |
| `attention_mask` | `(max_len,)` | 有效原子=1, pad=0 |
| `sym_attn_mask` | `(max_len, max_len)` | 对称注意力掩码 |
| `au_mask` | `(max_len,)` | 不对称单元标记 (True=属于 AU) |
| `n_atoms` | scalar | 实际原子数 |
| `rotation_order` | scalar | 旋转轴阶数 n |
| `point_group` | str | Schoenflies 符号 |
| `smiles` | str | Canonical SMILES |

#### 对称注意力掩码 `compute_symmetry_attention_mask()`

**算法**:

```
1. 平移至质心，计算惯性张量
2. 取最大本征值方向 = 主旋转轴 z
3. Rodrigues 旋转: R = I + sin(θ)K + (1-cos(θ))K²
   将全分子绕 z 轴旋转 2π/n
4. 找每个原子的等价原子：旋转后与原位置 < tolerance(0.5Å)
5. 掩码:
   - 等价类内: mask[i,j] = 1.0
   - 跨等价类: mask[i,j] = 0.5
```

**物理含义**: 当 Transformer 生成不对称单元的第 k 个原子时，它可以通过 mask 看到它在 C_n 旋转下的所有等价位置，从而保持生成物的对称一致性。

#### 不对称单元识别 `compute_asymmetric_unit_mask()`

计算每个原子绕主轴的方位角，将方位角在 [0, 2π/n) 范围内的原子标记为 AU。AU 是分子的最小生成集——整个分子可通过将 AU 旋转 n-1 次重建。

#### SimpleARTransformer

decoder-only Transformer 演示模型：token_embed + coord_proj + pos_embed → `nn.Transformer` (decoder-only) → output_head。`sym_attn_mask` 与因果掩码 (`triu(-inf)`) 融合输入 decoder。

---

## 四、管线运行结果

### 最终运行（3990 组合 → 1526 独特分子，3D 验证）

```
Stage 1:  3990/3990 生成成功  (100%)   57核心 × 70臂
Stage 2+3: 1526/3802 初筛PASS  (40.1%)  多构象采样（8 seeds，选最优）
Stage 4:  CSV(3802行) + XYZ(1526个) + JSON统计

总用时: 56 分钟（4 workers 并行初筛，无构象展开）
```

### 数据增强：2D 拓扑对称数据集

对已验证过的 41 个核心，生成全部 70 种臂组合（2D 层面，对称性由构造保证）：

```
41 核心 × 70 臂 = 2870 组合 → 去重后 2752 独特分子
```

### 最终数据集双重结构

| 数据集 | 文件 | 规模 | 验证级别 |
|--------|------|:----:|:--------:|
| **3D 验证集** | `cof_cores_dataset.csv` | 1526 | 完全验证（XYZ + 点群 + 能量） |
| **2D 增强集** | `augmented_dataset.csv` | **2752** | 拓扑保证（SMILES + 对称标签） |

> 2D 增强集中，1526 个分子与 3D 验证集重叠（有完整 3D 数据），1226 个为纯 2D 新增（可直接用于图生成模型训练）。

### PASS 分布（3D 验证集）

**按对称族**:
| 族 | 核心数 | 分子数 | 点群种类 |
|----|:------:|:------:|:--------:|
| C₂ | 29 | 1273 | C₂, C₂ₕ, C₂ᵥ, D₂, D₂ₕ, D₂𝒹 |
| C₃ | 11 | 244 | C₃, C₃ₕ, C₃ᵥ, D₃, D₃ₕ, D₃𝒹 |
| C₄/S₄ | 6 | ? | S₄, D₂𝒹 |
| D₆ₕ | 1 | 9 | D₆ₕ |

**按核心 TOP 10（新核心标注 ★）**:
| 核心 | PASS | 族 | 说明 |
|------|:----:|:--:|------|
| `bifuran_C2` ★ | 66 | C₂ | 联呋喃，O杂环刚棒 |
| `bithiophene_C2` ★ | 64 | C₂ | 联噻吩，S杂环刚棒 |
| `oxamide_C2` ★ | 63 | C₂ | 草酰胺，最短二酰胺桥 |
| `bis_benzamide_C2` ★ | 62 | C₂ | 对苯二甲酰胺 |
| `bis_benzoate_C2` ★ | 62 | C₂ | 对苯二甲酸酯 |
| `pyrazine_phenyl_C2` ★ | 62 | C₂ | 吡嗪联苯，N杂环 |
| `phenylethynyl_pyridine_C2` ★ | 60 | C₂ | 苯乙炔基吡啶 |
| `bis_ethynyl_naphthalene_C2` ★ | 55 | C₂ | 二乙炔基萘，稠环 |
| `bis_ethynyl_anthracene_C2` ★ | 53 | C₂ | 二乙炔基蒽，大稠环 |
| `triazine_C3` | 55 | C₃ | 三嗪经典 |

**按点群（15 种）**:
| 点群 | 数量 | 占比 |
|------|:----:|:----:|
| C₂ | 647 | 42.4% |
| D₂ | 231 | 15.1% |
| C₂ₕ | 166 | 10.9% |
| C₃ | 102 | 6.7% |
| C₂ᵥ | 88 | 5.8% |
| S₄ | 85 | 5.6% |
| C₃ₕ | 69 | 4.5% |
| D₂ₕ | 35 | 2.3% |
| D₃ₕ | 31 | 2.0% |
| D₃ | 23 | 1.5% |
| D₂𝒹 | 21 | 1.4% |
| C₃ᵥ | 18 | 1.2% |
| D₆ₕ | 7 | 0.5% |
| D₃𝒹 | 2 | 0.1% |
| C₆ | 1 | <0.1% |

### 外部数据库挖掘实验

测试了"从类药物数据库中挖掘对称分子"的可行性：

| 来源 | 规模 | 对称率 | 结论 |
|------|:----:|:------:|------|
| QM9 (10K) | 134K | 0.7% | 分子太小（≤9重原子），不适合 |
| MOSES (1.58M) | 1.9M | 0.018% | 随机药物分子 99.4% 为 C₁ |

**结论**: 数据库挖掘路线的投入产出比（4.3 小时 / 90 分子）远低于模板生成（56 分钟 / 1526 分子），不推荐作为主要数据来源。

### 关键发现

1. **刚性棒状 C₂ 是最高效模式**: 乙炔桥、酰胺桥、杂环联芳烃均产出 50-70 分子/核心
2. **成功模板的共同特征**: 芳香环 + 刚性连接子 + 短臂直接挂载
3. **模板路线效率碾压挖掘路线**: 63× 更高产出速率
4. **数据增强有效**: 从 1526（3D验证）→ 2752（2D拓扑保证），增加 80%
5. **C₄/S₄ 和 D₆ₕ 族仍是瓶颈**: 自然界缺乏高对称刚性有机骨架

---

## 五、JT-VAE 对称分子生成模型

### 设计动机

数据集建设完成后，目标是用它训练一个**可以生成指定点群对称性分子**的生成模型。在评估三种方案后选择了 JT-VAE：

| 方案 | 架构 | 问题 | 
|------|------|------|
| GraphAF | 图自回归（逐原子） | 自回归天然破坏对称 |
| DiGress | 离散扩散（全局） | 1661 数据量偏少 |
| **JT-VAE** | 联结树 VAE | 树结构天然捕获对称 |

JT-VAE 的核心洞察：我们的分子都是"对称核心 + 相同臂"结构，在联结树分解中天然呈现为**根节点（核心）+ 相同子树的重复**。这比原子级图生成更适合保持对称性。

### 联结树分解

```
分子:  triazine ─ arm_ph_CHO ×3

树:    [triazine_ring] ─── [phenyl_CHO] ×3
          (core)            (identical subtrees)
```

子树完全相同 → 模型只需学会生成一个子树并重复 n 次 → 天然保持对称。

### 模型架构

```
Tree Encoder (MPNN)  ─→ μ, σ ─→ z ←── PG Embedding
                                    │
Tree Decoder (GRU auto-regressive) ←┘

输入:  树节点序列 + 邻接表 + 点群标签
输出:  重构的树节点序列
条件:  点群嵌入注入 latent space
```

| 组件 | 实现 |
|------|------|
| Encoder | 3 层 MPNN → 全局平均池化 → μ, σ |
| Latent | 64-dim, 点群嵌入注入 μ |
| Decoder | 2 层 GRU, 自回归生成节点序列 |
| 条件化 | PG Embedding → μ + pg_emb |
| KL 退火 | β: 0 → 1 (warmup 50 epochs) |

### 训练状态

| 项目 | 值 |
|------|:----:|
| 训练集 | 2,476 分子 |
| 验证集 | 276 分子 |
| 树词表 | 24 标签 (11 环 + 13 原子) |
| 点群 | 5 种 (C2/C3/S4/C4/D6h) |
| 参数量 | 671K |
| 设备 | CUDA |
| Epochs | 200 (进行中) |

**路径**: `cof_symmetry_pipeline/jtvae/`

### 生成流程（训练完成后）

```python
# 1. 指定点群
pg_id = pg_map['C3']

# 2. 从 prior 采样
z = torch.randn(1, 64) 

# 3. 解码树节点序列
tree_tokens = model.decode(z, max_len=50)

# 4. 树 → 分子图
mol = tree_to_molecule(tree_tokens, vocab)
```

---

## 六、使用方法

```bash
# 最小运行（全组合，串行，MMFF94 优化）
python cof_symmetry_pipeline/main.py

# 指定核心+臂
python cof_symmetry_pipeline/main.py \
    --cores triazine_C3 biphenyl_C2 \
    --arms CHO_ph NH2_ph

# 多进程加速
python cof_symmetry_pipeline/main.py --workers 8

# 切换后端（需安装对应包）
python cof_symmetry_pipeline/main.py --backend mace    # MACE-OFF23
python cof_symmetry_pipeline/main.py --backend tblite  # GFN2-xTB

# 自定义筛查参数
python cof_symmetry_pipeline/main.py \
    --force-threshold 2.0   `# 更严苛的位移控制` \
    --max-steps 200          `# 更充分的优化` \
    --seed 123

# 运行后演示 PyTorch Dataset
python cof_symmetry_pipeline/main.py --dataset-demo
```

---

## 七、扩展指南

### 新增核心

在 `config.py` 的 `CORE_TEMPLATES` 中添加条目：

```python
"my_new_core": {
    "smiles": "*c1cc(*)c(*)cc1*",  # * 标记对称反应位点
    "description": "自定义四臂核心",
    "target_pg": "C2",
    "n_arms": 4,
    "symmetry_family": "C2",
},
```

### 新增臂

在 `config.py` 的 `ARM_LIBRARY` 中添加条目：

```python
"nitro_ph": {
    "smiles": "*c1ccc([N+](=O)[O-])cc1",
    "description": "4-硝基苯",
    "functional_group": "nitro",
},
```

### 新增对称族

在 `PG_EQUIVALENCE` 中添加等价类：

```python
PG_EQUIVALENCE["C5"] = {"C5", "C5v", "C5h", "D5", "D5h", "D5d"}
```

---

## 八、已知局限

1. **~~卟啉模板~~** ✅ 已修复：替换为 meso-四苯基卟啉(TPP)单分子 SMILES，正常产出 4 个独特分子
2. **~~四苯基甲烷~~** ✅ 已修复：target_pg 改为 S₄，现产出 8 个独特分子（S₄, D₂𝒹 点群）
3. **~~通过率~~** ✅ 已大幅提升：从 15.7% → 40.1%（3990 组合，1526 PASS），增强至 2752 独特分子
4. **~~多进程 pickle~~** ✅ 已修复：改为传递 SMILES + 属性元组
5. **~~构象展开~~** ✅ 已确认无效并默认关闭
6. **~~卟啉模板~~** ✅ 已修复为 TPP 单分子 SMILES
7. **~~四苯基甲烷~~** ✅ target_pg 改为 S₄
6. **MMFF94 vs 量子化学**: MMFF 能量为 kcal/mol 量级，不能直接与 DFT 对比；若需要严格能量排序，应安装 MACE 或 xTB
7. **点群分析回退**: 无 pymatgen 时惯性张量法只能区分 C₁/C₂/C₃⁺，无法区分 C₃ᵥ 和 D₃ₕ
8. **C₄/S₄ 和 D₆ₕ 族稀缺**: C₄ 族占 14%，D₆ₕ 仅 1.6%——自然界缺乏高对称刚性有机骨架，需外部数据源补充
9. **7 个核心无产出**: spirobifluorene_C2、benzidine_C2、trispyrazolyl_C3、cyclobutane_tetracarboxyl_C4、adamantane_tetra_C4、hexaethynylbenzene_H6、pyrene_dimer_C2——均因 3D 构象无法保持对称被淘汰

---

## 九、与 `data_generators/` 的对比

| 维度 | `data_generators/` | `cof_symmetry_pipeline/` |
|------|-------------------|--------------------------|
| 对称性保证 | 构造（环+对称挂载） | 构造（等价替换） |
| 分子类型 | 纯碳氢 | COF 中间体（含 N/O/B/F/S） |
| 臂库 | 3种 | 70种（12 类官能团 × 直接/苯基/联苯基/乙炔桥/氧桥/氮桥） |
| 核心库 | 1 个环 | 57种（C₂×29 / C₃×11 / C₄×6 / D₆ₕ×1 + 各异构体） |
| 3D 嵌入 | ❌ | ✅ ETKDGv3 + 多构象采样（8 seeds） |
| 点群分析 | ❌ | ✅ pymatgen + 惯性张量回退 |
| 几何优化 | ❌ | ✅ MMFF/UFF/MACE/xTB |
| 对称破缺筛查 | ❌ | ✅ 三级淘汰 + 多构象选优 |
| 数据增强 | ❌ | ✅ 2D 拓扑保证 × 全部臂组合 = 2752 分子 |
| 外部挖掘 | ❌ | ✅ MOSES 1.58M 三阶段筛选实验 |
| 可视化 | ❌ | ✅ 8 张分析图 |
| PyTorch Dataset | ❌ | ✅ COFSymmetryDataset + symmetry attention mask |
| 输出格式 | `{adj, x, n, k}` | SMILES + XYZ + point_group + energy + symmetry_family |
| 数据集规模 | 5,000 拓扑图 | **2752 独特分子**（1661 含 3D 坐标 + XYZ） |
| 生成模型 | ❌ | ✅ JT-VAE (训练中) |

---

## 十、数据旅程总结

```
 11 → 30 → 182 → 751 → 1526 → 1661 → 2752
  │     │      │      │      │       │       │
  │     │      │      │      │       │       └─ 数据增强 (2D 拓扑对称, 全核心×全臂)
  │     │      │      │      │       │
  │     │      │      │      │       └─ 增强验证 (+135 新增 3D PASS)
  │     │      │      │      │
  │     │      │      │      └─ 大规模扩展 (57核×70臂, +775 独特分子)
  │     │      │      │
  │     │      │      └─ 臂扩展 (26核×24臂 → 26核×70臂)
  │     │      │
  │     │      └─ 核心扩展 + 多构象 (9核×10臂 → 16核×24臂)
  │     │
  │     └─ 卟啉/四苯基甲烷修复 + 卤素臂
  │
  └─ 初始: 9 核 × 10 臂, 模板合成
```

| 阶段 | 数据量 | 关键动作 |
|------|:------:|---------|
| v1 | 11 | 初始模板 (9核×10臂, 卟啉碎片/四苯基甲烷 C4→S4 修复) |
| v2 | 30 | 多构象采样 (8 seeds, PG_EQUIVALENCE 扩展) |
| v3 | 182 | 26核×24臂 (新增乙炔桥/杂环/卤素臂) |
| v4 | 751 | 26核×70臂 (新增 46 种臂, 构象展开验证无效) |
| v5 | 1526 | 57核×70臂 (新增 31 刚性核心, MOSES 挖掘实验) |
| v6 | 1661 | 增强集 3D 验证 (+135 PASS) |
| v7 | 2752 | 全核心×全臂 2D 拓扑保证 |
| v8 | 进行中 | JT-VAE 训练 (CUDA, 200 epochs) |
