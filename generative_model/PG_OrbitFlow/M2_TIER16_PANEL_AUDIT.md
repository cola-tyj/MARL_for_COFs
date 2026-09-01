# M2 → Tier-16 面板与群相位特征审计

## 1. 为什么不能直接沿用旧 Tier-16

M2 虽然通过 Tier-4，但其中 33 个 torsion orbit 全部接近 0°/180°。旧 Cartesian-flow
Tier-16 的 C2 半区仍几乎全平面，不能验证新的 internal-coordinate 模型是否真正学会非平面
扭转与手性。因此本阶段只审计/冻结数据和表示，不启动训练。

对全部 C2/C3 IID-train 1,984 个分子扫描得到：

| 点群 | 分子数 | 含非平面 torsion | 含 active chirality |
|---|---:|---:|---:|
| C2 | 1,615 | 1,256 | 765 |
| C3 | 369 | 260 | 97 |

另有 22 个 C2 分子出现 torsion orbit circular mean 接近零，说明当前“每个 torsion orbit 一个
标量角”的合同对它们没有唯一定义；这些样本严格排除，不能静默选择任一角度。

## 2. 冻结面板

新 Tier-16 保留 M2 的四个样本，并各增加 6 个 C2/C3 IID-train 样本。所有 12 个新增样本都：

- 至少含一个距 0/180° 超过 5° 的 torsion orbit；
- 至少含一个绝对归一化三重积不小于 0.05 的 chirality candidate；
- circular torsion target 定义良好；
- 原子数不超过 21。

最终为 C2/C3 各 8 个，共 16 个；包含 78 个非平面 torsion orbit 和 47 个 active chirality
约束。冻结面板为 `configs/m2_tier16_panel_v2_phase_aware.json`，SHA-256
`55c3cc73bcba057bb08804708134c86f09b21e1d0009b65676049dec2357dfb2`。

## 3. 首次失败：quotient orbit aliasing

直接把 M2 head 用到上述面板时发现 34 处确定性冲突：不同 bond/angle/torsion orbit 在折叠后
具有相同 atom-orbit 路径与 bond feature，却对应不同真实内部坐标。尤其 C3 的不同 arm phase
在普通 quotient graph 中会折叠成同一路径。一个确定性 head 不可能从完全相同输入输出两个不同
目标，所以这不是增加 steps 或 loss 权重能解决的问题。

失败证据保留在 `reports/m2_tier16_panel_audit_v1.json`；没有通过换样本来回避该问题。

## 4. 修复：geometry group-phase feature

对 atom orbit 的 canonical representative (a_0)，只使用已存储的群作用 (R_g/P_g)，为
同一 orbit 内的原子 (a) 定义 coset-average matrix：

```math
A_a=\frac{1}{|\{g:P_g(a_0)=a\}|}\sum_{g:P_g(a_0)=a}R_g.
```

对几何路径相邻原子加入 (A_iA_{i+1}^{T}) 的展平矩阵：bond 为 9 维、angle 为 18 维、
torsion 为 27 维；正向与反向序列取确定性 canonical ordering。因此特征：

- 不读取 XYZ 或 target internal coordinate；
- 显式保留 C2/C3 群元素/coset 的相对 phase；
- 与无向 bond/angle/torsion reversal convention 一致；
- 不改变 canonical graph、action 或 M2 已冻结的默认模型。

加入后，冻结 Tier-16 的 phase-aware bond/angle/torsion exact-feature conflict 均为 0。正式审计
状态为 `PASS_M2_TIER16_PANEL_AUDIT`，报告 SHA-256
`46e690d356b00763c394b4639f78404f634e8f5f4e87bd597907d7f499c5d7bf`。

## 5. 下一步准入条件

下一步应冻结 M2.1 训练协议：从 M2 继承 quotient message-passing backbone，但因 head 输入维度
增加，重新初始化 bond/angle/torsion heads；使用冻结 Tier-16、PG-balanced mini-batch，并保持
oracle-free decoder。除 M2 原 Gate 外，必须新增：

- active chirality preserved fraction；
- non-planar torsion subset circular MAE；
- C2/C3 分层结果；
- phase-aware feature 必须启用且 conflict count 为 0。

在该协议和测试冻结前不启动 Tier-16 训练；Tier-16 未通过则不进入 Tier-32。
