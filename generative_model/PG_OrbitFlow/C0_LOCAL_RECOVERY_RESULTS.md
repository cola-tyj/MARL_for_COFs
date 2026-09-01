# PG-OrbitFlow C0 局部化学几何恢复结果

## 1. 实验问题与冻结边界

H1–H3 表明模型能保持 C2/C3 群作用，但不能从 Cartesian harmonic prior 恢复合法键长、
键角和整体构象。C0 因此只回答一个更小的问题：同一个 Cartesian backbone 能否从真实终点
附近的群不变扰动恢复局部化学几何？

正式 C0-v2 保持 H1 的 4 个 IID-train 分子（2 C2 + 2 C3）、740,424 参数、随机初始化、
1,024 optimizer steps、batch size 4 和原 Tier-4 raw Gate。合计 4,096 次 molecule exposure；
每个点群独立按 `small×4 : medium×3 : large×2 : harmonic×1` 调度：

| PG | small | medium | large | harmonic | 合计 |
|---|---:|---:|---:|---:|---:|
| C2 | 820 | 615 | 409 | 204 | 2,048 |
| C3 | 820 | 615 | 409 | 204 | 2,048 |

没有使用 validation/test/Core-OOD、历史 checkpoint、外部预训练、输出 hard projection 或
F0.2。该实验不能形成泛化或新分子生成结论。

## 2. 输入、监督和新增目标

输入仍是 known graph、Target_PG、完整 `R_g/p_g` action、atomic orbits、时间 `t` 和群不变
Cartesian 坐标。局部样本通过给 canonical target 加 Reynolds-projected Gaussian noise 构造；
harmonic 样本继续使用 H1 graph-harmonic prior。模型仍只输出全原子 Cartesian velocity。

新增的 `GeometryContract` 严格从 canonical graph 派生 bond、bond angle、torsion、图距离不超过
3 的 local pair、ring bond 和 chirality candidate，并验证这些索引在所有群操作下闭合。训练在
端点外推及 4-step differentiable rollout 上同时施加 orbit-balanced bond/angle/torsion、local
pair、ring、chirality、overlap 和 symmetry loss；canonical graph 从不被修改。

## 3. v1 调度错误与修复

C0-v1 的全局十槽 cursor 与固定 C2/C3 交替发生相关，导致 409 个 harmonic exposure 全部分配
给 C3，C2 为 0。该 run 仅保留为执行错误审计，不能用于科学结论或作为 parent。

C0-v2 唯一执行修复是为 C2、C3 分别维护确定性 cursor；模型、样本、步数、loss 权重和 Gate
不变。证据见
[`c0_v1_sampling_correlation_diagnosis.json`](reports/c0_v1_sampling_correlation_diagnosis.json)。

## 4. 正式 C0-v2 结果

局部 Gate 在固定 `σ=0.10 Å` 下，每个分子使用 8 个 seed，共 32 个 case：

| 指标 | 阈值 | 结果 | 判定 |
|---|---:|---:|---|
| endpoint RMSD mean | ≤ 0.10 Å | 0.08568 Å | PASS |
| bond-length MAE mean | ≤ 0.03 Å | 0.04369 Å | **FAIL** |
| angle MAE mean | ≤ 5° | 3.731° | PASS |
| collision-free | = 1.0 | 1.0 | PASS |
| max action error | ≤ 1e-4 Å | 5.56e-7 Å | PASS |
| chirality preserved | = 1.0 | 1.0 | 不具判别力 |

手性项“不具判别力”是因为该平面 Tier-4 panel 的 active chirality case 数为 0，不能将数值 1.0
写成模型已经通过手性恢复验证。正式局部 Gate 仅因 bond MAE 超阈值而失败。

原 harmonic-prior raw Gate 完全未修改，结果仍失败：fixed endpoint RMSD `0.9492 Å`、raw
Kabsch `1.8328 Å`、pair-distance MAE `1.0993 Å`、bond MAE `0.6668 Å`；collision-free 和
action error 通过。说明网络已能进行有限的局部修复，但没有学会从宽 Cartesian prior 跨越到
化学构象流形，而且局部 curriculum 还牺牲了部分 H1 raw 性能。

同一正式 checkpoint 的 local 与 raw 评估各重复两次，metrics 和 NPZ 数组逐元素完全一致。
完整结论见
[`c0_local_recovery_v2_summary.json`](reports/c0_local_recovery_v2_summary.json) 和
[`c0_reproducibility_v1.json`](reports/c0_reproducibility_v1.json)。

## 5. 分支决定

正式状态为 `FAIL_C0_LOCAL_RECOVERY_STOP_CARTESIAN_BACKBONE`。按预注册规则：

- 不进入 full-prior curriculum、Tier-16/32 或全量训练；
- 不通过放宽 `0.03 Å` 键长阈值改写结论；
- 下一步实现 fixed-graph bond/angle/torsion internal-coordinate decoder，由内部坐标生成局部
  化学几何，再通过可微 kinematics、ring closure 与群 orbit coupling 恢复 Cartesian 3D。

## 6. 代码与证据索引

| 文件 | 功能 |
|---|---|
| `geometry.py` | canonical geometry contract、群闭合检查、可微局部几何 loss/metrics |
| `build_c0_protocol.py` | 冻结 C0-v2 配置及 v1 调度修复记录 |
| `c0_local_recovery.py` | 多噪声 curriculum、4-step rollout、训练和两套 Gate |
| `audit_c0_reproducibility.py` | formal checkpoint 的 local/raw 双重复数组审计 |
| `summarize_c0_experiment.py` | v1 错误诊断、v2 指标和分支决定归档 |
| `tests/test_c0_local_recovery.py` | geometry/action、调度、确定性、反向传播测试 |

