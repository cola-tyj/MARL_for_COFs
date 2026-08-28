# Fast-32 Flow 最终结果与 Orbit 硬对称接口

**日期**：2026-08-22  
**状态**：Fast-32 endpoint flow 正式停止；symmetry-by-construction 表示接口通过  
**任务边界**：已知 immutable molecular graph、目标群作用与 `Target_PG`，生成全原子 3D 坐标

## 1. Fast-32 Flow 做了什么

正式实验沿用冻结的 32/8 C2/C3 面板：32 条 IID-train、8 条独立 IID-validation；test 与
Core-OOD 未读取。Semla `EquiInvDynamics` 接收插值坐标、time、固定 graph 和 PG condition，
预测 clean endpoint；训练 1,024 steps 后，用固定 ODE 采样和 correct/swapped PG 对照验收。

该实验不是无条件新分子生成，也不改变原子或键。checkpoint 的输入是
`(x_t, time, atom/bond graph, Target_PG)`，输出是同一批原子的 endpoint coordinates。

## 2. 最终结果

正式状态为：

```text
FAIL_COORDINATE_FLOW_FAST32_STOP_ENDPOINT_BRANCH
```

| 指标 | 结果 | 判定 |
|---|---:|---|
| training loss tail/first window | 0.5494 | 通过 |
| recovery overall RMSD ratio | 0.8204 | 通过 |
| recovery improved cases | 79.17% | 通过 |
| `t=0.25` recovery ratio | 0.6947 | 通过 |
| `t=0.50` recovery ratio | 0.7294 | 通过 |
| `t=0.75` recovery ratio | 1.0370 | **失败** |
| prior sampling pair-distance MAE ratio | 0.7211 | 通过 |
| prior sampling collision-free | 1/16 = 6.25% | **失败** |
| correct PG 优于 swapped PG | 6/16 = 37.5% | **失败** |
| correct/swapped symmetry-error ratio | 1.0423 | **失败** |
| raw analyzer success / PG-compatible | 0/16 / 0/16 | **失败** |

模型的 loss、部分插值恢复和 PG sensitivity 说明存在学习信号，但从真正 Gaussian prior
积分时大量结构塌缩。`sampling_symmetry_improved_fraction=1.0` 不能单独解释为成功：塌缩坐标
也可能产生较低的操作误差；必须结合 collision-free、analyzer 和 correct-vs-swapped PG。

根据训练前冻结规则，`t=0.75` recovery 失败直接对应“停止 endpoint branch”，不允许用
更换积分器、降低阈值或继续训练来改写结论。模型 checkpoint 只保留为诊断证据，不进入
2,026 条正式训练。

## 3. 新的 symmetry-by-construction 表示

新实现不再要求网络独立输出每个原子的 3D 坐标。对于每个 atomic symmetry orbit：

1. 固定选择最小 atom index 作为 orbit representative；
2. 模型只产生每个 representative 的一个坐标；
3. 对 representative 的 stabilizer 做平均，确保轴上或镜面上的特殊位置严格成立；
4. 用保存的三维操作矩阵与 permutation index 展开该 orbit 的全部原子；
5. 最后执行一次 Reynolds projection，严格验证
   `X @ R_g.T == X[permutation_g]`。

实现同时提供 NumPy 审计路径和 Torch 可微路径；不创建 `N×N` permutation matrix，不改变
canonical atom/bond graph，不做元素 fallback。

## 4. 严格接口审计

审计只使用 IID-train 2,026 条和 IID-validation 253 条，共 2,279 条；test/Core-OOD 保持封存。

| 检查 | 结果 |
|---|---:|
| 成功处理 | 2,279/2,279 |
| strict failure | 0 |
| 重复运行字节确定性 | 2,279/2,279 |
| 最大 operation error | `6.11326e-7 Å` |
| collision-free | 2,279/2,279 |
| Sn-containing records | 2 |
| 平均 representative/atom 比例 | 0.46984 |
| C2 / C3 比例 | 0.50719 / 0.33402 |
| canonical→hard expansion RMSD 均值 | `0.00997 Å` |
| canonical→hard expansion bond MAE 均值 | `0.000294 Å` |

这里的 RMSD 是把已有近似对称 canonical coordinates 转为严格群作用坐标所需的扰动，
不是生成模型的采样误差。该 Gate 只证明“输出参数化能严格保证对称性且覆盖数据接口”，不能
宣称已经训练出生成器。

## 5. 单元测试与机器证据

6/6 单元测试通过，覆盖 C2、C3、非平凡 stabilizer、旋转协变、确定性 representative、
Torch 梯度，以及非法跨 orbit 群作用的严格失败。

- Fast-32 report SHA-256：`27c07670e14f254658aab42f1a93806444e7fa8db0841e332498956d5dce6c85`
- Fast-32 losses SHA-256：`5a7abadbb6f847f60ede643dd0a99bbb17b67bcb2a1beab38500550ae9bd411f`
- Fast-32 checkpoint SHA-256：`7cfe182faad7a8f9a1e987ad71c8e5c608c1b3cbe575f3af56f42d3204bbdf07`
- orbit audit SHA-256：`434b69b3e92a5c03b33d9d88dcf00dfdd0538bacca4e73e88677e361765adc0b`

机器报告：

- `generative_model/runs/graph_pg_3d/coordinate_flow_fast32_v1/report.json`
- `generative_model/smoke/reports/symmetry_by_construction_orbit_audit_v1.json`

## 6. 当前结论与下一步

Fast-32 flow 不是可用基座，学习式 raw PG endpoint 分支到此收尾。项目主线切到 R4：

```text
known graph + Target_PG + known group action/orbits
        ↓
orbit-representative coordinate model
        ↓ differentiable group expansion
exact-symmetry full-atom coordinates
        ↓
bond/collision/constrained-relaxation audit
```

后续 O0 已完成：等变 atom proposal 经 orbit pooling 后只保留 representative coordinates，
再严格展开全原子坐标；5/5 单元测试和真实 C2/C3/Sn 工程 Gate 全过，状态
`PASS_ORBIT_COORDINATE_O0_FREEZE_O1`。随后 512-step O1 已运行并因低噪声退化与碰撞失败，
不设计 learned orbit sampler；详见
[Orbit-coordinate O1 最终结果](18_Orbit_Coordinate_O1结果.md)。当前改做 constrained
force-field baseline；新 graph 的 automorphism/orbit 构造仍是后续独立接口。
