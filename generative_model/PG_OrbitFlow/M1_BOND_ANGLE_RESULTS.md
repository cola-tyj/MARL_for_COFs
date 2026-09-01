# PG-OrbitFlow M1 商图 Bond/Angle Orbit Head 结果

## 1. 实验目标

M1 在通过的 M0 orbit-space decoder 上只增加 learned bond/angle orbit heads，检验商图网络能否
在原 Tier-4 四分子面板上记住精确局部几何。torsion 与有效 chirality 暂时来自 oracle，故本轮
不是完整 learned 2D→3D。

## 2. 输入输出与泄漏边界

模型输入仅包括：

- quotient atom nodes 的13元素、形式电荷、自由基、orbit size/stabilizer；
- Target_PG 与 group character features；
- quotient multiedges 的 multiplicity 与 canonical bond-type histogram；
- bond/angle orbit 的 quotient endpoint。

模型对象与监督对象使用不同 dataclass，forward 输入不含 XYZ、bond target 或 angle target。
网络输出：

- 每个 bond orbit 一个正键长；
- 每个 angle orbit 一个范围在 `[-1,1]` 的 angle cosine。

decoder 中 oracle local-pair 权重严格为0；ring bond 使用预测键长。只有 torsion 和当前无活跃
样本的 chirality 保持 oracle。

## 3. 模型与训练

- 4个 IID-train 分子：2 C2 + 2 C3；
- quotient message-passing network：266,402 参数、hidden 96、4层；
- 1,024 optimizer steps、batch size 4，即每一步同时使用四个分子；
- AdamW，learning rate `1e-3`；
- loss：`10 × bond-orbit MSE + 2 × angle-cosine MSE`；
- validation/test/Core-OOD、外部权重、MDS、NeRF、ETKDG、hard projection、F0.2 均未使用。

最终训练日志：平均 bond-orbit MAE `6.13e-7 Å`、angle MAE `6.49e-4°`。正式评估使用最终
step-1024 checkpoint，不做事后 checkpoint 选择。

首次在 CUDA 上执行 float64 小张量 L-BFGS decoder 时因工程耗时过高人工中止；训练已经完整
结束且 checkpoint 已写入。随后只将相同 checkpoint 的 decoder 评估迁移至 CPU，未重新训练，
能量、16起点和 Gate 均未改变。该执行修订及失败日志 hash 已写入正式报告。原 runner 已修复为
训练结束立即保存 loss，避免未来评估中断丢失完整 loss 数组；本次 v1 只保留5个预定 console
milestone 和全部 checkpoint。

## 4. 轨道预测 Gate

Gate 使用四个分子的最差值：

| 指标 | 阈值 | 最差结果 | 判定 |
|---|---:|---:|---|
| bond-orbit MAE | ≤ 0.03 Å | 1.07e-6 Å | PASS |
| angle-orbit MAE | ≤ 5° | 0.00228° | PASS |

## 5. 三维重建 Gate

模型预测的 bond/angle 广播回原图，oracle torsion 保留；每个分子使用16个确定性 orbit-space
起点并按最低内部坐标能量选择。相对于 canonical target 的最差结果：

| 指标 | 阈值 | 最差结果 | 判定 |
|---|---:|---:|---|
| bond MAE | ≤ 0.03 Å | 9.67e-7 Å | PASS |
| angle MAE | ≤ 5° | 0.00245° | PASS |
| torsion circular MAE | ≤ 2° | 2.39e-6° | PASS（oracle） |
| ring closure MAE | ≤ 0.03 Å | 1.70e-6 Å | PASS |
| Kabsch RMSD | ≤ 0.20 Å | 4.19e-5 Å | PASS |
| collision-free | = 1.0 | 1.0 | PASS |
| max group-action error | ≤ 1e-6 Å | 1.95e-7 Å | PASS |

预测数组逐元素重复一致，四个选中 CPU decoder seed 的独立重算也通过。正式状态：
`PASS_M1_ADVANCE_TO_M2_TORSION_HEAD`。

## 6. 结论边界与下一步

M1证明：

- quotient multigraph encoder 能在 Tier-4 上学习 bond/angle orbit；
- exact orbit broadcast 不会破坏 C2/C3 局部几何；
- 关闭 oracle local-pair 后，M0 solver 仍能根据预测 bond/angle 与 oracle torsion 重建3D。

M1没有证明：

- torsion 可以由模型预测；
- 四分子以外的泛化；
- novel graph 或无 action 输入的生成；
- noisy/inconsistent predicted IC 下的完整稳定性。

下一步 M2 在相同 Tier-4 面板上新增 torsion-orbit `sin/cos` head，并加入 ring/chirality 一致性
Gate。只有 M2 通过后，才可称为该面板上的完整 learned graph + Target_PG/action → 3D。

结构化证据：

- [`m1_bond_angle_tier4_v1_summary.json`](reports/m1_bond_angle_tier4_v1_summary.json)
- [`m1_reproducibility_v1.json`](reports/m1_reproducibility_v1.json)

