# PG-OrbitFlow 构建设计与阶段 Gate

## 1. 最终模型拆分

PG-OrbitFlow 不把“新分子图生成”和“三维坐标生成”混成一个未经验证的大任务，而是拆成：

```text
Q：Core–Arm Quotient Generator
Target_PG → core + attachment orbits + representative arm(s) + graph action
                             ↓
                     完整对称 molecular graph
                             ↓
O：PG-Orbit Coordinate Flow
known graph + Target_PG/action/orbits → raw symmetric 3D
                             ↓
P：独立物理与化学评估
sanitize / valence / collision / actual-PG / xTB
```

当前实现的是 `O` 的第一个可训练版本，以及 `Q` 所需的 quotient graph 数据结构。

## 2. Q：Core–Arm Quotient Generator

### Q0：逐原子监督构建

数据构建时给 core 和 arm template 原子加稳定 atom-map，并把 map 传播到组合产物。每条数据
必须保存：

- `atom_role ∈ {core, arm, attachment}`；
- `arm_copy_id`；
- `attachment_orbit_id`；
- representative arm mask；
- core/arm local attachment frame；
- graph action 在 representative arm 与所有副本之间的原子映射。

无法得到唯一 mapping 的分子严格拒绝，不能通过最大公共子结构静默选择一个答案。

### Q1：条件 arm 生成

先固定/检索 core，只学习：

```text
core attachment environment + Target_PG + desired functional condition
                                  ↓
                      one representative arm per orbit
```

生成 arm 后通过完整 `O(3)` action 展开。S4/D6h 必须支持 improper operations；手性 arm 要
单独检查镜像是否改变 stereochemistry。

### Q2：core 与 attachment orbit 生成

生成 quotient core graph、连接位点及其 permutation action。必须验证：

```math
p_g p_h=p_{gh}, \qquad R_gR_h=R_{gh}
```

且 group action 是带元素和 bond topology 的图自同构。

### Q3：联合生成

只有 Q1/Q2 各自通过后，才联合训练 core、arm 和 action。否则失败来源无法归因。

## 3. O：PG-Orbit Coordinate Flow

### O0：数据与群表示 Gate——已通过

- C2/C3 IID train+validation 2,232/2,232 严格通过；
- 完整 `R_g/p_g` paired closure；
- atomic orbit 与 stabilizer 一致；
- quotient topology action mismatch 为 0；
- Sn 语义保留，无 fallback。

### O1：工程 Gate——已通过

- 2-step forward/backward；
- finite loss/gradient；
- E(3) 与 group-action 保持；
- raw ODE 不调用 post-hoc projection；
- checkpoint 和 exact model-state resume。

### O2：C2/C3 nested overfit raw-generation Gate——Tier-4 未通过

正式 Stage-1 前新增严格 4→16→32 nested Gate；每层必须通过才能初始化下一层。Tier-4 使用
2 C2 + 2 C3 IID-train 分子、740,424 参数、1,024 steps、batch 4、C2:C3 exposure 1:1，
不接触 validation/test/Core-OOD。阈值在运行前冻结，覆盖：

- raw operation RMS 和 worst-operation error；
- collision-free、bond MAE、pair-distance MAE；
- C2/C3 分层结果；
- fixed-time endpoint recovery 和训练 loss 收敛。

独立 transport 和 Cn phase-aligned transport 两轮均失败；后者显著改善几何且通过 loss、碰撞、
群作用检查，但 raw bond/Kabsch/pair-distance 仍失败。只读诊断证明最终 checkpoint 最好、
25–100 ODE steps 等价，残差集中在 `t≈0` 的早期向量场。故 Tier-16/32 与全量 Stage-1 均不
执行，不能通过放宽阈值改写结论。

随后按同一 Tier-4 面板和 Gate 完整执行了 H1–H3 单变量修复：graph-harmonic prior、
centralizer-averaged transport、endpoint auxiliary。三组均保持 raw action 和 collision，
但均未同时通过 loss-window、fixed endpoint、raw Kabsch、pair-distance 和 bond-length Gate。
因此没有合格的 Tier-16 parent，当前 Cartesian backbone 已按预注册规则停止。完整数字和代码
定位见 [H1_H3_RESULTS.md](H1_H3_RESULTS.md)。

为进一步定位能力边界，C0-v2 在同一 Tier-4 面板上加入 PG-balanced multi-noise curriculum、
直接 bond/angle/torsion/local-pair/ring/chirality 监督和 4-step differentiable rollout。模型从
`σ=0.10 Å` 的局部扰动恢复时通过 RMSD、angle、collision 和 group-action 检查，但 bond MAE
`0.04369 Å` 未过 `0.03 Å`；原 harmonic raw Gate 仍失败。故局部修复能力存在，但不足以把宽
Cartesian prior 映射到化学构象流形。结果和 v1 调度错误审计见
[C0_LOCAL_RECOVERY_RESULTS.md](C0_LOCAL_RECOVERY_RESULTS.md)。

下一条主线改为 fixed-graph internal-coordinate manifold：以 graph 决定 bond/angle/torsion
坐标及 ring-closure constraints，在 group orbit 内共享/变换内部坐标，再通过可微 kinematics
恢复 Cartesian 3D。该路线首先仍需通过同一个 Tier-4 panel；在通过前不进入 16/32 或全量训练。

该路线的第一项 M0 oracle 验证已经通过：真实 orbit-level internal coordinates、随机对称多起点、
stabilizer-aware exact lifting 和 orbit-space energy minimization 可将四个 Tier-4 分子全部重建到
近机器精度。C2 的16个起点全部收敛，C3 各有9/16收敛，最低 energy 可正确选解。该结果仅解除
运动学与求解器风险；下一项 M1 才开始学习 bond/angle orbit，并保持 oracle torsion 来隔离变量。
完整结果见 [M0_ORACLE_RECONSTRUCTION_RESULTS.md](M0_ORACLE_RECONSTRUCTION_RESULTS.md)。

M1 已在相同 Tier-4 面板上通过：quotient multigraph encoder 只读取 graph/group/orbit features，
预测 bond/angle orbit；oracle local-pair 已关闭，仅 torsion 暂时保持 oracle。最差轨道预测误差
为 `1.07e-6 Å/0.00228°`，重建 Kabsch 最差 `4.19e-5 Å`，4/4 collision/action Gate 通过。
该结果解除 bond/angle head 与 decoder 接口风险，但还不是完整 learned 2D→3D；M2 必须新增
torsion `sin/cos` head。详见 [M1_BOND_ANGLE_RESULTS.md](M1_BOND_ANGLE_RESULTS.md)。

M2 已完成并通过相同 Tier-4：在 M1 quotient encoder 上增加 reversal-invariant torsion-orbit
`(sin φ, cos φ)` head，1,024 steps 后同时关闭 decoder 的 oracle bond/angle/torsion/local-pair/
chirality。最差预测误差为 `6.76e-7 Å / 8.04e-5° / 0.001952°`，无 oracle 重建最差 Kabsch
为 `3.74e-6 Å`，4/4 无碰撞且 action error `1.94e-7 Å`；复现审计通过。由于该四分子面板的
33 个 torsion orbit 均接近 0/π，这只是平面扭转记忆与完整接口 Gate。进入 Tier-16 前必须先
审计并冻结非平面 torsion、有效 chirality、C2/C3 平衡及 feature-collision 支持，不能直接启动
全量训练或声明 unseen 泛化。完整证据见 [M2_TORSION_RESULTS.md](M2_TORSION_RESULTS.md)。

Tier-16 预审计进一步发现：普通 quotient 路径会丢失群元素/coset phase，在选定的非平面/手性
样本上产生 34 处“相同 head 输入、不同 IC target”的不可学习冲突。现已增加完全 graph/action-
only 的 relative group-phase matrices：对 atom-orbit representative 到具体 atom 的 operation
coset 做矩阵平均，并对相邻原子使用相对乘积；bond/angle/torsion 分别增加 9/18/27 维，正反路径
canonicalize。新 Tier-16（8 C2 + 8 C3）含78个非平面 torsion orbit、47个 active chirality，
phase-aware 冲突为0并通过面板审计。

M2.1 随后已按冻结协议完成：三个维度变化的 IC heads 重初始化，quotient backbone 从 M2 继承；
16条、2,048 steps、batch 4、C2/C3 每批2:2。bond/angle 通过，但 torsion/nonplanar torsion
最差 `50.9257°/84.3727°`，在 prediction Gate 后停止，未运行 decoder。只读诊断显示31个失败
torsion orbit 中30个在 torsion head 前已经具有完全相同的输入并发生重复输出塌缩，29个可直接
归因于 NH₂/CF₃/CH₃ 等同一 rotor 的等元素
末端置换。由此明确区分两类群结构：`R_g/P_g` 解决目标点群 action/coset phase；分子图自身的
local automorphism/stabilizer 决定可交换原子的无序 torsion slots。M2.2 必须使用
permutation-invariant circular set/rotor target，而非给网络加入任意 atom-index ID。详见
[M2P1_PHASE_TIER16_RESULTS.md](M2P1_PHASE_TIER16_RESULTS.md)。

M2.2 表示 Gate A 已通过：严格 graph-only 的 terminal-rotor contract 在同一面板识别12个
disjoint 2/3-slot sets，覆盖29/31个失败 orbit；可微 circular Hungarian assignment 已通过
permutation-invariance 与 finite-backward 测试。该阶段没有训练，也没有读取 XYZ/target 来
建组。下一单变量是 set-valued torsion head；在它通过同一 Tier-16 prediction Gate 前，不运行
3D decoder，也不扩大面板。

M2 v5 随后完成 operation-coupled graph-centralizer matching，Tier-16 的16/16分子与47/47
active chirality 均通过，并完成独立复现。基于这一 parent，M3 冻结 C2/C3 各16的 Tier-32
面板。普通平均 torsion loss 和 generalized automorphism head 仍遗留少数 hard molecules；
加入 Gate-aligned worst-molecule torsion/nonplanar loss 后，prediction Gate 通过。

多起点 decoder 的剩余问题不是局部 IC 精度，而是局部 IC energy 无法区分全局构象 basin。
最终单独增加 atom-order-invariant counted WL fingerprint 和16维 long-range distance quantile
head，只用于预测全局形状并选择 deterministic starts：

```text
score = learned IC final energy + 10 × predicted global-shape MSE
```

selector 不读取 target coordinates，decoder 仍关闭全部 oracle IC、hard projection 和 F0.2。
M3 最终32/32通过 reconstruction Gate，独立复现通过，状态 `PASS_M3_TIER32`。该结论只证明
固定训练面板的模型容量与完整接口，不证明 unseen graph 泛化。下一 Gate 是冻结 checkpoint 后
对 IID-validation 新图做只读评估；详见 [M3_TIER32_RESULTS.md](M3_TIER32_RESULTS.md)。

M4 只读评估已确认 M3 不具备 unseen graph 泛化，因此转入从随机初始化开始的数据级 orbit-IC +
shape 联合训练。训练使用1,963条严格支持的 IID-train，固定4 C2 + 4 C3 balanced batch；245条
IID-validation 只做描述性评估，IID-test/Core-OOD继续封存。graph-action-compatible
automorphism contract 先离线写成逐分子、pickle-free、可哈希缓存，避免重启重复 GraphMatcher；
缓存完成后才冻结20,000-step final 协议。模型仍不读 target Cartesian coordinates，raw 输出后也
不做 hard projection/F0.2。完整合同与命令见 [M4_DATASET_TRAINING.md](M4_DATASET_TRAINING.md)。

### O3：泛化 Gate——Cartesian 分支仍阻断，IC 分支已获准进入

原 Cartesian O2 仍然失败且永久停止；M3 的 internal-coordinate 替代路线已通过 Tier-32，
因此只允许使用冻结的 M3 parent 开始以下只读泛化评估。不得复活失败 Cartesian parent，也不
得在 validation/test 上继续训练或选择超参数。

- IID-test 一次性评估；
- Core-OOD；
- unseen core、unseen arm、unseen core–arm combination；
- 外部 graph；
- S4/D6h 作为独立扩展，D6h 仅 exploratory。

## 4. 当前 loss

```math
L=L_{flow}+0.25L_{bond}+0.1L_{overlap}+1.0L_{sym}
  +w_{endpoint}L_{endpoint}.
```

- `L_flow`：真实 conditional-flow velocity MSE；
- `L_bond`：从当前时间外推终点后的 canonical bond-length MSE；
- `L_overlap`：排除成键 pair 的元素半径碰撞 hinge；
- `L_sym`：所有 operation 的 endpoint MSE，同时报告 worst operation。
- `L_endpoint`：只在 H3 中令 `w_endpoint=1.0`，目标为 centralizer posterior mean；其他
  protocol 中该权重严格为 0。

虽然架构应使 `L_sym≈0`，仍保留它作为运行时断言和数值漂移监测，不能把接近零的 symmetry
loss 单独当作化学/生成质量成功。

## 5. Hard projection 的精确定义

历史 composite 路线为：

```text
ET-Flow raw coordinates → Reynolds hard projection → F0.2
```

当前 PG-OrbitFlow 为：

```text
group-invariant prior → equivariant ODE vector field → raw symmetric coordinates
```

模型输出后不做 projection。训练标签/先验的 Reynolds average 是坐标空间定义的一部分，类似
把模型直接定义在约束流形上，不应与推理后修复混写。后续消融必须至少比较：

1. ET-Flow raw；
2. ET-Flow raw + hard projection；
3. PG-OrbitFlow raw；
4. PG-OrbitFlow raw + F0.2（仅作安全后端）。

## 6. 风险

- 当前 v2 的 3D 标签来自 ETKDG/UFF 数据构建过程；需要外部构象和 xTB/DFT 防止学习构建器偏差；
- C3 样本少于 C2，训练 sampler 做 1:1 exposure，但 validation 必须报告自然分布与平衡面板；
- Kekulé bond type 在 101 条 C2/C3 train/validation 分子中不严格随 action 保持，当前仅在
  model feature 层做群平均，canonical bond graph不变；
- `PG-compatible` 可能由更高超群满足，必须同时报告 exact 和 swapped-PG controllability；
- quotient/core-arm 路线不天然覆盖笼状、稠合和多桥连分子，需要明确适用域。
