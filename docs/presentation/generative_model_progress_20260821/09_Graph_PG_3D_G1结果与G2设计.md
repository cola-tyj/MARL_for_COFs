# Graph + Target_PG → 3D：G1 结果与 G2 设计

## 1. G1 测试的是什么

G1 不是学习模型，而是零参数 hard-symmetry baseline。它验证：已知真实 sparse graph、
`Target_PG`、目标 operation matrices 和 atom permutations 时，能否把一组受扰动坐标投影到
目标群不变子空间。

```text
真实全原子坐标
    ↓ 固定 Gaussian noise，σ=0.15 Å
noisy coordinates
    ↓ 一次 Reynolds group average
hard-symmetric coordinates
    ↓ 同一 pymatgen protocol 复验
actual PG / compatible / symmetry error / RMSD / collision
```

真实 atom/bond graph 全程只作条件，不预测、不修改。G1 面板在运行前从 IID-validation
冻结，严格嵌套：tier-1 为 1 条、tier-4 为 C2/C3/S4/D6h 各 1 条、tier-32 配额为
`24/6/1/1`。test/Core-OOD 未使用。

## 2. 冻结 Gate 与结果

| tier-32 指标 | 结果 | 冻结门槛 | 通过 |
|---|---:|---:|:---:|
| finite / centered / known graph unchanged | 32/32 | 全部 | 是 |
| coordinate RMSD 改善 | 32/32 | ≥28/32 | 是 |
| bond-length MAE 改善 | 32/32 | ≥24/32 | 是 |
| collision-free | 32/32 | 32/32 | 是 |
| mean RMSD improvement | 34.89% | ≥25% | 是 |
| mean projected RMSD | 0.16477 Å | ≤0.15 Å | 否 |
| max operation RMS error | 1.85e-7 Å | ≤1e-5 Å | 是 |
| max atom symmetry error | 3.45e-7 Å | ≤1e-4 Å | 是 |
| analyzer success / compatible | 31/32 | 32/32 | 否 |
| actual-PG exact | 25/32 | 描述性 | — |

所以冻结状态是 `FAIL_GRAPH_PG_3D_G1_HARD_PROJECTION`。不能因为多数指标很好而后验放宽
`0.15 Å` 或 32/32 analyzer 门槛。

## 3. 失败原因

### 3.1 hard projection 不是 denoiser

Reynolds average 只删除破坏目标群关系的坐标分量。Gaussian noise 中本来就在 target-group
invariant subspace 内的分量会被保留，因此所有分子都比 noisy input 更接近目标，但平均
RMSD 停在 `0.16477 Å`。对未加噪 target 直接 hard-project 的平均结构偏移只有
`0.00751 Å`，说明主要问题不是 annotation bias，而是缺少 graph-conditioned coordinate
prior/denoiser。

### 3.2 S4 analyzer 边界

唯一失败样本是 package index `1113`。投影坐标满足 stored S4 operations，最大 atom error
低于 `3.45e-7 Å`，但固定 analyzer tolerance `0.3 Å` 的候选路径识别为不含完整 S4 子群。
诊断性 tolerance sweep 在 `0.05–0.2 Å` 识别 S4，在 `0.3–0.4 Å` 失败，在 `0.5 Å`
识别为 compatible D2d，表现出非单调边界。

该 sweep 只用于解释，不替代冻结 `0.3 Å` protocol，也不把 G1 改判为通过。后续报告同时
保留两类指标：固定 analyzer 的 actual-PG 结果，以及对给定 target operations 的连续残差。

## 4. G1 最终结论

诊断状态为：

`HARD_PROJECTION_VALID_AS_CONSTRAINT_NOT_SUFFICIENT_AS_COORDINATE_GENERATOR`

即：hard projection 工程和群作用正确，可以作为 G2 的 constraint/postprocess；但它不能
独立从 known graph 产生高精度坐标，必须增加学习式 coordinate model。

## 5. G2 推荐设计

G2 先做 graph-conditioned coordinate denoising，再逐渐过渡到 diffusion sampling：

```text
atom/bond graph + noisy coordinates + noise level + Target_PG
                         ↓
             E(3)-equivariant graph denoiser
                         ↓
               predicted clean coordinates
                         ↓
              target-operation hard projection
                         ↓
      collision / RMSD / actual-PG / compatible validation
```

建议首个有界实现：

- 使用 sparse true bonds；不预测 bond class；
- atom、formal charge、radical、bond type 和 PG embedding 只作为 scalar features；
- 坐标更新只由相对向量乘 scalar message 构成，保持 E(3) equivariance；
- 训练目标包含 coordinate denoising、bond-length consistency 和 soft symmetry residual；
- inference 后使用 G1 已验证的 hard projection；
- 仍按 `1 → 4 → 32`，先证明 coordinate denoising 与 PG 复验，再进入 IID-train；
- G2 之前另行冻结 step、seed、noise schedule、参数量和质量 Gate。

G2 的成功不代表能生成新 graph；它只解决“给定合法 graph 和 PG，生成/恢复相应 3D
coordinates”。新 SMILES/graph generator 仍是后续独立阶段。
