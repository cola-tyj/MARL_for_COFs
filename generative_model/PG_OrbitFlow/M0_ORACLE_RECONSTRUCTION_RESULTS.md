# PG-OrbitFlow M0 轨道空间 Oracle 重建结果

## 1. M0回答的问题

M0 不训练神经网络，只验证以下命题：给定真实的轨道级 bond/angle/torsion/local-pair/ring/
chirality 目标，能否从与目标 Cartesian 坐标无关的随机对称起点，通过 orbit-space 能量最小化
恢复完整 C2/C3 三维构象？

该实验用于把两类风险分开：

1. `R_g/p_g/orbit` exact lifting 与环闭合求解器是否正确；
2. 后续商图网络能否预测内部坐标。

只有第一项属于 M0。M0 通过不等于 learned generation、泛化或新分子生成通过。

## 2. 冻结协议

- 面板：原 Tier-4 的 4 个 IID-train 分子，2 C2 + 2 C3；
- 模型：无神经网络、无 checkpoint；
- 每个分子：16 个确定性 Gaussian orbit-parameter 起点；
- 优化：1,200-step Adam + 最多 200-step L-BFGS，float64 CPU；
- 选择：只按最低 oracle internal-coordinate energy 选择，不按 Kabsch/RMSD 选择；
- 未使用 target Cartesian 初始化、Cartesian coordinate loss、ETKDG、ET-Flow、MDS、NeRF、
  posthoc hard projection 或 F0.2；
- target XYZ 只用于提取 oracle orbit-level internal coordinates 和最终独立评估。

协议 SHA-256：`0ad72036c3e44576cd18997d069bd75f5a2c5169ae36cc7919160fe77c0c0024`。

## 3. Stabilizer-aware exact group lifting

每个 atom orbit 选择代表原子 (i)，计算其 stabilizer 的 fixed-subspace basis (B_i)，只优化
低维参数 (q_i)：

\[
x_i=B_iq_i.
\]

orbit 中其他原子由与 permutation 配对的群操作生成：

\[
x_{p_g(i)}=x_iR_g^T.
\]

所有可选 coset representative 必须产生相同坐标，否则严格失败。最终全坐标质心归零。真实
Tier-4 target 的 `encode → exact lift` 最大绝对往返误差为 `1.24e-7 Å`；合成 C2 轴上 fixed
atom 测试确认其自由度被正确压缩为一维。

## 4. 正式结果

Gate 使用四个分子中的最差值，而不是平均值：

| 指标 | 阈值 | 最差结果 | 判定 |
|---|---:|---:|---|
| exact-lift roundtrip max error | ≤ 1e-5 Å | 1.24e-7 Å | PASS |
| bond MAE | ≤ 0.001 Å | 1.15e-8 Å | PASS |
| angle MAE | ≤ 0.2° | 7.60e-7° | PASS |
| torsion circular MAE | ≤ 2° | 1.56e-6° | PASS |
| ring-closure MAE | ≤ 0.001 Å | 1.18e-8 Å | PASS |
| Kabsch RMSD | ≤ 0.10 Å | 5.55e-8 Å | PASS |
| collision-free fraction | = 1.0 | 1.0 | PASS |
| max group-action error | ≤ 1e-6 Å | 1.96e-7 Å | PASS |

正式状态：`PASS_M0_ORACLE_ADVANCE_TO_M1_ORBIT_IC_HEADS`。

## 5. 多起点诊断

| package index | PG | RMSD ≤ 0.1 Å | collision-free | 结论 |
|---:|---|---:|---:|---|
| 864 | C2 | 16/16 | 16/16 | 所有起点收敛 |
| 1045 | C2 | 16/16 | 16/16 | 所有起点收敛 |
| 1135 | C3 | 9/16 | 9/16 | 存在错误局部极小值 |
| 2391 | C3 | 9/16 | 9/16 | 存在错误局部极小值 |

最低 oracle energy 在四个分子上均选中正确解。因此当前确定性16起点策略足以通过 M0，但 C3
的非凸性明显高于 C2。后续 M1/M2 不能直接删掉 multi-start；若预测内部坐标存在噪声后选择
失效，再增加 symmetry-aware distance/stress initializer，而不是立即把 MDS 放进主解码器。

四个选中 seed 的独立重算均通过坐标与 collision 状态一致性审计。

## 6. M0证明了什么、没有证明什么

证明了：

- orbit representative + stabilizer basis 的参数化正确；
- exact group lifting 同时适用于当前 C2/C3；
- 不依赖生成树也能处理当前环/交联约束；
- 使用正确内部坐标时，能量求解不是当前 `0.03 Å` bond Gate 的瓶颈。

没有证明：

- 商图网络可以预测同等准确的内部坐标；
- noisy/inconsistent predicted constraints 仍能收敛；
- unseen molecule、IID-test、Core-OOD 或完整 2,532 分子的泛化；
- chirality、improper operations、S4/D6h 已覆盖。

下一步 M1 只学习 bond/angle orbit，暂时使用 oracle torsion，以最小变量检验商图预测是否能突破
`0.03 Å` bond Gate。

## 7. 代码和证据

| 文件 | 功能 |
|---|---|
| `orbit_kinematics.py` | stabilizer basis、orbit encode、differentiable exact lifting |
| `build_m0_protocol.py` | 冻结 M0 面板、优化器、Gate 和隔离边界 |
| `m0_oracle_reconstruction.py` | oracle orbit targets、能量、多起点求解、正式 Gate |
| `audit_m0_reproducibility.py` | 选中 seed 的独立重算审计 |
| `summarize_m0_experiment.py` | Gate、全部起点收敛率、hash 汇总 |
| `tests/test_m0_orbit_reconstruction.py` | fixed atom、exact lift、oracle energy、真实 panel 测试 |

结构化报告：

- [`m0_oracle_reconstruction_v1_summary.json`](reports/m0_oracle_reconstruction_v1_summary.json)
- [`m0_selected_start_reproducibility_v1.json`](reports/m0_selected_start_reproducibility_v1.json)

