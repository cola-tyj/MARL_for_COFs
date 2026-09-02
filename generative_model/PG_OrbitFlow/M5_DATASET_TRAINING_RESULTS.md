# M5 数据级训练与独立 validation 结果

## 1. 结论

数据级训练按冻结协议完成20,000 steps，工程执行通过；但冻结 `last.pt` 在全部225个未参与训练的
IID-validation分子上未通过原M3/M4 prediction Gate。因此当前checkpoint不能进入3D decoder Gate，
也不能查看IID-test或Core-OOD。

- 训练状态：`PASS_DATASET_TRAINING_EXECUTION`；
- 独立验证状态：`FAIL_DATASET_VALIDATION_STOP_BEFORE_DECODER`；
- 审计执行：通过；
- unseen质量：失败；
- validation记录与训练结束时的descriptive-only记录逐项最大差异：`0`。

这说明训练流程、checkpoint恢复和验证复算是可靠的，但当前模型尚未学会跨分子泛化的torsion与
全局构象。失败不应解释为点群约束失效，也不应通过hard projection或F0.2掩盖。

## 2. 冻结训练设置

| 项目 | 数值 |
|---|---:|
| canonical/factorized cache记录 | 1,925 |
| 实际训练分子 | 1,700（C2 1,373；C3 327） |
| descriptive-only validation | 225（C2 182；C3 43） |
| optimizer steps | 20,000 |
| batch size | 8（每批4 C2 + 4 C3） |
| validation梯度更新 | 0 |
| IID-test/Core-OOD使用 | 否 |
| hard projection/F0.2使用 | 否 |

模型输入为known molecular graph、Target_PG/group action、orbit/factorized automorphism和WL graph
fingerprint；监督目标为bond/angle/torsion orbit及long-range shape quantiles。target Cartesian
coordinates不作为模型输入。

## 3. 训练执行结果

首32步与末32步平均总loss为 `25.694436 → 1.485556`，比值 `0.0578163`；checkpoint重载输出
逐元素差异为0。末1,000步平均loss为 `1.507896`，其中torsion loss `0.148637`、shape MSE
`0.0714940`。末1,000步斜率仍略为负，但距离冻结的2° torsion与0.03 Å shape Gate很远，不能
仅凭loss继续无限续训。

最终单步记录：loss `0.996816`、bond `0.000663902`、angle `0.000738783`、torsion
`0.133782`、shape `0.0752485`，gradient norm `4.31155`。

## 4. 独立 IID-validation Gate

审计器重新加载 `last.pt` 和full-v2 cache，对全部225条validation记录重新执行前向计算。所有
训练/数据/协议/hash检查通过，复算结果与训练report完全一致。

| 指标 | 冻结阈值 | C2最大值 | C3最大值 | 结果 |
|---|---:|---:|---:|---|
| bond orbit MAE | ≤0.03 Å | 0.039679 Å | 0.031554 Å | FAIL |
| angle orbit MAE | ≤5° | 2.62655° | 3.76703° | PASS |
| torsion orbit circular MAE | ≤2° | 44.7499° | 66.7120° | FAIL |
| nonplanar torsion circular MAE | ≤2° | 135.907° | 134.670° | FAIL |
| global-shape quantile MAE | ≤0.03 Å | 0.789961 Å | 1.353081 Å | FAIL |

自然validation分布的均值为：bond `0.0104166 Å`、angle `1.20008°`、torsion `16.8747°`、
nonplanar torsion `42.9032°`、shape `0.257680 Å`。局部bond/angle学习明显优于torsion和全局
shape；当前主要问题是跨图的相位/构象多模态与全局basin泛化，不是训练执行或对称lifting错误。

## 5. 决策

1. 不运行当前checkpoint的decoder Gate，因为其输入IC/shape prediction已失败；
2. 不查看IID-test/Core-OOD，不用validation反向挑checkpoint或放宽阈值；
3. 将本次20,000-step结果冻结为dataset-level baseline；
4. 后续结构改进应针对torsion多模态与global-shape条件表示，而不是简单延长同一训练；
5. 科学结论冻结后再进行代码目录重构，历史结果通过Git恢复点保留。

## 6. 文件与哈希

- full protocol：`c94a1841a290a15b52f1a2f2268ac3ef832bd6b8da10d8cee093b1177ed22470`
- factorized cache manifest：`b1800439535b5a8d873e203d6f3b7d57ae308bf3ba05b27022ad211f3e98ab63`
- training report：`86c2b012361ecc1f0cb927da32c56d5eadc7623dbeea649ad891bdc47b4e6cfb`
- losses：`4abeeaf35c1b3dcd22a227baf1292e9aefd1f28fdb9bd842c783f4b91a02a06a`
- final checkpoint：`3c164537a2e1bdd3215129079f0b072f838c88bd8bc56b582aa23e572b306a57`
- step-020000 checkpoint：`e1119a6900b3d63a95f25dc20d3fe61141441988a51857ce12bf30f5ec1444b8`
- independent validation report：`783f0ac64fb40342ceb51f4bf927e0305792851367a999f5751acea1caa23369`
- auditor：`845bf49a901a3670d323fb412416f7c493e9c4c310ffe764755560d3e72b2108`
