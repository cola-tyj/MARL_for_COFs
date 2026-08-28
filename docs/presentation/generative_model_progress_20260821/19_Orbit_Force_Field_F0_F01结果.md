# Orbit-constrained Force Field：F0/F0.1 结果

## 1. 任务边界

输入仍是已知真实 molecular graph、`Target_PG`、目标群操作/permutation/orbit 和一个严格对称
的扰动构象；输出只是一组优化后的 3D 坐标。该实验没有训练神经网络、没有生成新 graph，
也没有使用 test/Core-OOD。每次能量评估只优化 atomic-orbit representative 坐标，再硬展开到
全部原子，所以对称性在优化全过程中由构造保证。

## 2. 力场覆盖率

严格从 canonical graph 重建 RDKit 分子，显式 H、原子序数、形式电荷、自由基和真实键均不
修改；禁止元素 fallback。

| 范围 | UFF setup | MMFF94s setup | 结论 |
|---|---:|---:|---|
| 全部数据 | 2,532/2,532 | 2,497/2,532 | UFF 可作为统一主力场 |
| 冻结 C2/C3 面板 | 40/40 | 40/40 | 两力场均可诊断 |
| Sn 分子 | 2/2 | 0/2 | Sn=50 保留；MMFF 不支持但 UFF 支持 |
| B 分子 | 33/33 | 0/33 | 不删除 B；统一走 UFF |

覆盖报告状态为 `PASS_STRICT_FORCE_FIELD_COVERAGE_AUDIT`。

## 3. F0 冻结实验

- Gate：8 个独立 IID-validation 分子（6 C2、2 C3）。
- 扰动：`σ=0.15/0.5/1.0 Å`，每档 2 个固定 seed，共 48 cases。
- 优化：orbit representative 空间中的 UFF + L-BFGS-B，最多 200 iterations。
- Gate：有限执行、能量不升、键长改善、碰撞、严格 operation error 和优化器收敛率。
- 坐标 RMSD、pair-distance MAE、MMFF 和实际点群重算只报告，不参与 Gate。

F0 的 9/10 checks 通过：键长 MAE 48/48 改善，mean ratio `0.02746`；collision-free
`44/48=0.91667`；coordinate RMSD ratio `0.61574`；pair-distance ratio `0.21561`；UFF 能量
48/48 下降；最大 operation error `2.94e-7 Å`。唯一失败项是 SciPy optimizer success
`14/48=0.29167 < 0.75`，34 cases 因达到 200 iterations 停止。因此正式状态仍记为
`FAIL_ORBIT_FORCE_FIELD_F0`，不能事后更改阈值。

## 4. 唯一一次 F0.1 收敛预算修复

F0.1 在运行前单独冻结，保留相同 48 cases、输入、UFF objective、所有 Gate 阈值和无 barrier
设计，只把 `maximum_iterations` 从 200 增至 500；F0 原失败报告永久保留。

| 指标 | F0.1 | Gate | 通过 |
|---|---:|---:|:---:|
| finite execution | 48/48 | 100% | 是 |
| UFF energy nonincrease | 48/48 | ≥95% | 是 |
| bond MAE improved | 48/48 | ≥75% | 是 |
| bond MAE ratio | 0.02743 | ≤0.85 | 是 |
| each-sigma bond ratio | 全部 ≤1.0 | ≤1.0 | 是 |
| collision-free | 44/48 = 0.91667 | ≥0.75 | 是 |
| optimizer success | 40/48 = 0.83333 | ≥0.75 | 是 |
| maximum operation error | `2.89e-7 Å` | ≤`1e-5 Å` | 是 |

正式状态：`PASS_ORBIT_FORCE_FIELD_F01_ADVANCE_TO_INITIAL_CONFORMER_INTERFACE`。

## 5. 正确解读

这是当前第一条通过预注册有界质量 Gate 的 `known graph + known group action + initial
conformer → strict-symmetry 3D` 后端。它证明多数固定案例中，graph、orbit mapping 和可用
初始构象已知时，严格 C2/C3 对称与良好局部键几何可以同时实现。

它尚未证明 `Target_PG → 新分子`，也尚未解决初始构象从何而来。4 个碰撞案例全部来自
package index 1490 的中/高噪声输入；总体阈值已通过，但应作为后续鲁棒性诊断保留。当前
后续使用 `env_cof` 完成了独立点群复核：44/48 compatible；4 个失败全部是 package 1490
的中/高噪声碰撞构象，actual PG 为 C1。因此 F0.1 Gate 原状态保留，但完整后端尚需一次
有界 collision repair。详见
[20_F01坐标归档与点群独立审计.md](20_F01坐标归档与点群独立审计.md)。

## 6. 下一步

先冻结一次只针对 4 个失败 case 的短程非键排斥修复，要求 4/4 无碰撞且 compatible。通过后
再冻结 initial-conformer interface：`SMILES/graph + Target_PG → 多个初始构象 → orbit mapping
→ UFF relaxation → analyzer`。不恢复 O1，也不立即训练 2,532 条模型。
