# PG-OrbitFlow

PG-OrbitFlow 是一条与现有 `our_ET_Flow` 完全隔离的新模型分支。它读取冻结的 canonical v2，
目标是逐步实现：

```text
Target_PG + quotient/core-arm molecular graph
                    ↓
point-group-aware orbit coordinate flow
                    ↓
raw symmetric 3D coordinates（模型前向不做 post-hoc hard projection）
                    ↓
独立 chemistry / collision / actual-PG / xTB Gate
```

当前已经完成两条有明确边界的验证：早期 Cartesian coordinate-flow backbone 在 Tier-4 失败并
按协议停止；替代的 orbit-level internal-coordinate 路线已通过 M2 Tier-16 和 M3 Tier-32
封闭训练面板 Gate。新 2D graph 的 core/arm generator 尚未实现，unseen graph 泛化也尚未
验证，因此不能把当前结果误写为“已生成新分子”或“已具备通用 2D→3D 能力”。

## 1. 与 our_ET_Flow 的隔离边界

- 不修改、不导入 `generative_model/inference/*our_etflow*`、历史 ET-Flow adapter 或 F0.2；
- 不加载任何历史 checkpoint；
- canonical v2 只读；
- 模型、配置、训练、测试、报告和运行目录全部位于本文件夹；
- 未来若需要 F0.2，只能作为独立评估分支调用，不得混入 raw model Gate。

## 2. 当前模型输入与输出

输入：

- 13 元素原子类型、形式电荷、自由基电子；
- immutable sparse canonical bond graph；
- `Target_PG`；
- 完整群操作 `R_g ∈ O(3)`，包含 `det(R_g)=-1` 的反射/反演类操作；
- 同元素原子置换 `p_g`；
- atomic orbit、orbit size、stabilizer；
- 群阶、det parity、operation trace moments、fixed-atom character 等群特征。

模型输出是全原子的时间相关坐标速度场 `v_θ(X_t,t,G,PG)`。训练目标为线性 conditional
flow matching：

```math
X_t=(1-t)Z_G+tX_G, \qquad v^*=X_G-Z_G.
```

其中 `Z_G` 是群不变先验，`X_G` 是 canonical XYZ 在目标群不变子空间中的严格训练标签。
坐标训练时除以固定的 `3.0 Å` scale，报告再换回 Å。

## 3. 对称性不是只靠 PG one-hot

当前显式使用以下群信息：

| 层级 | 特征 |
|---|---|
| 群级 | group order、proper/improper 比例、operation trace 的均值/方差/范围、fixed-atom character |
| orbit 级 | orbit size、stabilizer size、对应归一化与 log 特征 |
| action 级 | `R_g` 与 `p_g` 的配对闭包、同元素约束、orbit-preserving permutation |
| 图级 | atom orbit quotient、edge orbit、多重度、bond-type histogram |

模型采用 complete-pair E(3)-equivariant relative vector messages。只要输入坐标处于目标群不变
子空间，并且标量特征在原子置换下不变，预测向量场也留在同一子空间。因此 raw ODE 不需要在
每一步后追加 Reynolds projection。

这里仍使用 Reynolds average 来准备两类数据：

1. 将带有限数值误差的 canonical target 转换为严格监督标签；
2. 将随机先验初始化到对应 action 的 invariant subspace。

它们是训练数据/先验定义，不是模型输出后的 `hard projection → F0.2` 流程。

## 4. Quotient graph 与 core/arm 边界

v2 已有每个原子的 orbit，但只有分子级 `Core`/`Arm` 名称，没有逐原子 core/arm assignment。
当前实现不会根据名称猜原子标签，而是先构建严格可验证的 quotient molecular graph：

- 一个 quotient node 对应一个 atomic orbit；
- quotient edge 保存 endpoint orbit、多重度和 4 类 canonical bond histogram；
- bond topology 必须在所有 `p_g` 下严格不变；
- 由于 Kekulé 单/双键表示可在对称操作下交换，保留 mismatch 计数，并在模型侧使用群平均
  bond-order feature，canonical bond ground truth 不被修改。

下一阶段的 core/arm generator 必须先建立带 atom-map 的逐原子分解数据，不能把分子级模板名
直接当成原子级监督。详细路线见 [DESIGN.md](DESIGN.md)。

## 5. 已验证的数据规模

冻结 IID train + validation 的 C2/C3 合同审计：

| split | C2 | C3 | 合计 | 原子 | quotient nodes |
|---|---:|---:|---:|---:|---:|
| train | 1,615 | 369 | 1,984 | 71,306 | 33,502 |
| validation | 202 | 46 | 248 | 8,311 | 3,918 |

- 2,232/2,232 group action、同元素 permutation 和 bond topology 通过；
- 包含 2 个 Sn 分子，无元素 fallback；
- mean coordinate DOF ratio 为约 `0.47`；
- target projection RMSD 均值 `0.00694 Å`，最大 `0.09827 Å`；
- 101 个分子存在 Kekulé typed-action mismatch，但 topological mismatch 为 0。

完整证据见
[dataset_contract_c2_c3_v1.json](reports/dataset_contract_c2_c3_v1.json)。

## 6. 当前工程 smoke

两步 CPU smoke 使用 8 个有界 train 分子、4 个 validation 分子；PG-balanced sampler 实际
暴露 C2/C3 各 2 次。模型有 20,160 参数，2 optimizer steps、batch size 2。

- forward/backward、有限梯度、checkpoint、raw Heun ODE 均执行成功；
- validation raw operation 最大原子误差 `9.45e-7 Å`；
- raw ODE smoke 最大原子误差 `5.95e-7 Å`；
- 从 step 1 恢复后，step 2 的模型 state 与不中断训练逐 tensor 完全一致；
- 只有工程 Gate 通过，不形成模型质量结论。

本地运行结果在 `runs/smoke_v1/`，大文件默认不纳入 Git。

## 7. 冻结 Tier-4 质量 Gate

正式扩大到 16/32 分子或全量数据前，先执行了 `2 C2 + 2 C3` 的 IID-train-only 过拟合
Gate。模型为 740,424 参数，随机初始化，训练 1,024 optimizer steps、batch size 4，即
4,096 次 molecule exposure；C2/C3 暴露严格 1:1。validation、test、Core-OOD、外部权重、
输出 hard projection 和 F0.2 均未使用。

| transport | loss ratio | fixed endpoint RMSD | raw Kabsch RMSD | raw bond MAE | collision-free | action max error | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| independent | 0.6658 | 1.5633 Å | 1.8067 Å | 1.0081 Å | 1.000 | 2.33e-6 Å | FAIL |
| Cn phase-aligned | 0.4711 | 0.8867 Å | 1.3948 Å | 0.5745 Å | 0.875 | 1.37e-6 Å | FAIL |

相位对齐只改变 target/prior 在 Cn 主轴周围的 gauge 配对，不改变分子、模型、loss、训练预算或
Gate。它使几何指标改善约 23%–43%，证明 transport 修复方向有效；但仍未通过冻结阈值，因此
按协议停止，不运行 Tier-16/32 或全量 Stage-1。

只读诊断进一步排除了两个假设：最终 step 1,024 是固定输入表现最好的 checkpoint；25/50/100
步 Heun 的 raw Kabsch RMSD 分别为 1.39505/1.39479/1.39473 Å，故不是 checkpoint 选择或
ODE 步数不足。最大残差位于流前段：step 1,024 在 `t=0.1` 的端点 RMSD 为 1.4633 Å，
`t=0.9` 为 0.2226 Å。当前结论是：raw 群作用保持已经成功，瓶颈是早期向量场恢复化学几何的
能力。完整冻结结论见
[overfit_tier4_v2_failure_and_diagnosis.json](reports/overfit_tier4_v2_failure_and_diagnosis.json)。

### H1–H3 早期向量场单变量修复——全部完成，均未通过

在 phase-aligned Tier-4 后又冻结并完整执行三组累计消融：

1. H1：unnormalized graph-harmonic prior + single phase alignment；
2. H2：H1 + C2/C3 centralizer-averaged transport；
3. H3：H2 + endpoint auxiliary loss。

三组均保持相同 4 分子、1,024 steps、batch size 4、模型、seed、均匀时间、50-step Heun
和原 Gate，并从相同随机初始化独立训练。结果如下：

| variant | loss ratio | fixed RMSD | raw Kabsch | pair MAE | bond MAE | collision | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| H1 | 0.5583 | 0.7804 Å | 1.4537 Å | 0.8538 Å | 0.5432 Å | 1.000 | FAIL |
| H2 | 0.6266 | 1.4280 Å | 1.6555 Å | 0.9366 Å | 0.5957 Å | 0.875 | FAIL |
| H3 | 0.6006 | 1.4114 Å | 1.5527 Å | 0.8557 Å | 0.5742 Å | 0.875 | FAIL |

三组的 finite、collision 和 raw action checks 均通过，但几何、bond 和收敛 Gate 未通过。
只读诊断确认 final checkpoint 最好，25/50/100 Heun 不能改变结论；actual checkpoints 的
同进程 raw 双重复逐数组完全一致。故按预注册规则不运行 Tier-16/32，停止当前 Cartesian
backbone，转入 fixed-graph bond/angle/torsion manifold decomposition。详细设计、阈值、
输入输出和证据索引见 [H1_H3_RESULTS.md](H1_H3_RESULTS.md)。

### C0 局部化学几何恢复——接近但未通过

为区分“网络完全不会恢复局部几何”和“只是不足以跨越宽 Cartesian prior”，又执行了正式
C0-v2。仍使用相同 4 分子、740,424 参数、1,024 steps、batch 4；4,096 次 exposure 在 C2/C3
间严格平衡，并加入 small/medium/large/harmonic curriculum、完整 bond/angle/torsion/local-pair/
ring/chirality 监督和 4-step differentiable rollout。

固定 `σ=0.10 Å` 的 32-case 局部评估中，endpoint RMSD `0.08568 Å`、angle MAE `3.731°`、
collision `1.0`、action error `5.56e-7 Å` 均通过；bond MAE 为 `0.04369 Å`，未过冻结的
`0.03 Å` 阈值。手性候选数为 0，故手性数值不构成证据。原 harmonic raw Gate 仍失败（raw
Kabsch `1.8328 Å`、bond MAE `0.6668 Å`）。正式结论仍是停止该 Cartesian backbone，下一步
实现 internal-coordinate decoder。v1 的 PG-correlated curriculum 调度错误已单独归档，不能
作为结论。完整结果见 [C0_LOCAL_RECOVERY_RESULTS.md](C0_LOCAL_RECOVERY_RESULTS.md)。

### M0 orbit-space oracle reconstruction——通过

按 C0 停止规则，现已实现不含神经网络的 stabilizer-aware orbit parameterization、exact group
lifting 和内部坐标能量求解器。在相同 4 分子上，每个分子从16个与 target Cartesian 无关的
确定性随机对称起点优化；target 只以 oracle bond/angle/torsion/local-pair/ring/chirality orbit
进入能量。四个分子的最差 bond MAE `1.15e-8 Å`、angle MAE `7.60e-7°`、ring closure
`1.18e-8 Å`、Kabsch RMSD `5.55e-8 Å`、action error `1.96e-7 Å`，全部通过冻结 Gate。

C2 起点成功率为 16/16，C3 为 9/16；最低 energy 能选中正确解，说明 M0 solver 可用，但
C3 仍需要 multi-start 或后续全局初始化。M0 只证明解码器正确，尚未证明模型能预测内部坐标。
下一步为 M1 learned bond/angle orbit heads，oracle torsion 暂时保持不变。完整结果见
[M0_ORACLE_RECONSTRUCTION_RESULTS.md](M0_ORACLE_RECONSTRUCTION_RESULTS.md)。

### M1 learned bond/angle orbit heads——通过

在 M0 decoder 上新增 266,402 参数、hidden 96、4层的 quotient multigraph encoder，并在相同
4分子上训练1,024 steps。模型输入不含 XYZ；输出每个 bond orbit 的键长和每个 angle orbit 的
angle cosine。decoder 关闭 oracle local-pair，ring 使用预测键长，torsion 暂时保持 oracle。

最差 bond-orbit 预测 MAE `1.07e-6 Å`、angle-orbit MAE `0.00228°`；重建后最差 bond MAE
`9.67e-7 Å`、angle MAE `0.00245°`、Kabsch `4.19e-5 Å`、action error `1.95e-7 Å`，4/4
无碰撞，双 Gate 通过。CUDA 小张量 decoder 因工程耗时中止后，冻结 step-1024 checkpoint
仅迁移到 CPU 评估，执行修订已完整记录。M1 仍是四分子记忆实验且 torsion 为 oracle；下一步
M2 学习 torsion orbit。详见 [M1_BOND_ANGLE_RESULTS.md](M1_BOND_ANGLE_RESULTS.md)。

### M2 learned torsion orbit head——通过（严格限于平面 Tier-4）

M2 从 M1 step-1024 checkpoint 继承 quotient encoder 与 bond/angle heads，新增 torsion-orbit
`(sin φ, cos φ)` head。仍使用相同 2 C2 + 2 C3 分子、1,024 steps、batch 4；模型共
313,540 参数。decoder 同时关闭 oracle bond、angle、torsion、local-pair 和 chirality，只使用
模型预测的 orbit-level 内坐标、碰撞项和 exact group lifting。

轨道预测最差误差为 bond `6.76e-7 Å`、angle `8.04e-5°`、torsion `0.001952°`；重建最差
Kabsch `3.74e-6 Å`，4/4 collision-free，group action error `1.94e-7 Å`，双 Gate 与独立复现
审计均通过。该面板 33 个 torsion orbit 全是近 0/π 的平面构型，所以只证明固定面板上的全学习
IC 闭环，不证明非平面、手性或 unseen 泛化。下一步先审计并冻结含非平面/手性支持的 Tier-16
面板。详见 [M2_TORSION_RESULTS.md](M2_TORSION_RESULTS.md)。

### M2.1 phase-aware Tier-16——训练完成，torsion Gate 失败

全量 C2/C3 IID-train 扫描表明非平面与手性支持充足，但普通 quotient geometry feature 在候选
Tier-16 上出现 34 处 orbit alias：不同群相位被折叠为相同 head 输入却具有不同目标。项目没有
通过换样本规避，而是新增只从 `R_g/P_g` 构造的 relative coset/group-phase feature（bond 9维、
angle 18维、torsion 27维），将冲突降为 0。新面板严格嵌套 Tier-4、C2/C3 各8条，12个新增
样本全部包含非平面 torsion 和 active chirality，最大21原子。审计状态
`PASS_M2_TIER16_PANEL_AUDIT`。随后 M2.1 使用16条、2,048 steps、batch 4、C2/C3 每批2:2
完成正式训练。bond/angle 最差误差 `0.006317 Å/1.5452°` 通过，但 torsion/nonplanar torsion
最差 `50.9257°/84.3727°`，状态为 `FAIL_M2P1_PREDICTION_GATE_STOP_BEFORE_DECODER`。

只读诊断发现16条中10条全部通过；31个失败 torsion orbit 中30个在 torsion head 前已有完全
相同的输入并产生重复预测塌缩，29个明确属于 NH₂/CF₃/CH₃ 等同一 rotor 上的等元素末端置换。
当前主要矛盾是 atom-labelled torsion
监督没有对局部 graph automorphism 做 permutation-invariant 处理，而不是点群 coset feature
仍有精确 alias。下一步 M2.2 先修复 local rotor/set 表示；不延长 M2.1、不放宽 Gate，也不
运行被 prediction Gate 阻断的 decoder。详见
[M2P1_PHASE_TIER16_RESULTS.md](M2P1_PHASE_TIER16_RESULTS.md)。

M2.2 表示 Gate A 已完成：graph-only 合同识别12个 disjoint 2/3-slot local rotor sets，覆盖
29/31个失败 torsion orbit；可微 permutation-invariant circular matching 的3项测试通过。
状态 `PASS_M2P2_LOCAL_ROTOR_SET_REPRESENTATION_GATE_A` 只允许继续实现 set-valued head，
不代表已训练或已生成3D。

### M2 v5 与 M3 Tier-32——通过

M2 v5 已把 local terminal rotors 扩展为与 Target-PG operation 联动的 graph-centralizer
等价匹配：Tier-16 的16/16分子、47/47 active chirality 通过，并完成独立复现。

M3 随后在严格嵌套的32分子面板（C2/C3各16，157个非平面 torsion orbit、87个 active
chirality）完成全学习 IC 闭环。M3.3 使用 mean + worst-molecule torsion loss 后通过 prediction
Gate；M3.6 再用 atom-order-invariant WL graph fingerprint 预测16个全局形状分位数，为16-start
decoder 提供不读取 target coordinates 的 basin selection score。最终最差重建 bond/angle/
torsion/nonplanar torsion/Kabsch 分别为 `0.003716 Å / 1.5325° / 1.7653° / 1.7430° /
0.06666 Å`，87/87手性保持、32/32无碰撞、C2/C3分层和 group-action Gate 全过，独立复现也
通过。正式状态 `PASS_M3_TIER32`。

这一结果严格限于32个训练面板分子的容量/闭环验证，不是 unseen graph 泛化。下一阶段必须先
冻结模型，在未参与训练的 IID-validation graph 上做只读 Gate；详见
[M3_TIER32_RESULTS.md](M3_TIER32_RESULTS.md)。

### M4/M5 unseen 与数据级训练——执行通过，unseen质量失败

冻结 M3 在独立 IID-validation 新图上的只读预测没有通过，证明 Tier-32 结果主要是闭集容量验证，
正式决策为进入 dataset-level training，而不是继续微调 M3。数据级 trainer 的256-step smoke 已
通过修正版工程审计：64条训练、16条 descriptive-only validation、每批4 C2+4 C3，首末32步
loss 比为 `0.172870`，checkpoint 重载完全一致。全量合同为1,963条 train + 245条 validation，
20,000 steps。v2 factorized cache 已完成：2,208条中1,925条成功、283条严格unsupported；final
协议最终使用1,700条train和225条descriptive-only validation，逐文件hash、缓存回读及代表样本
forward/backward均通过。随后20,000-step训练完整结束，首末32步loss比为`0.0578163`，checkpoint
重载差异为0；但全部225条IID-validation的冻结复算未通过原Gate：最大torsion/nonplanar torsion/
shape误差为`66.712° / 135.907° / 1.35308 Å`。因此停止在decoder之前，不查看IID-test/Core-OOD，
当前结果冻结为dataset-level baseline。详见[M4_DATASET_TRAINING.md](M4_DATASET_TRAINING.md)和
[M5_DATASET_TRAINING_RESULTS.md](M5_DATASET_TRAINING_RESULTS.md)。

## 8. 文件索引

| 文件 | 功能 |
|---|---|
| `group.py` | O(3) operation + permutation 配对闭包、群特征、严格投影/误差 |
| `quotient.py` | atom/edge orbit quotient 和群平均 bond-order feature |
| `core_arm.py` | 未来逐原子 core/arm 监督的严格合同；不猜 substructure mapping |
| `data.py` | canonical v2 只读 adapter、PG-balanced sampler、对称先验 |
| `model.py` | 原生 E(3)-equivariant PG-OrbitFlow vector field |
| `flow.py` | flow-matching 样本和 raw Euler/Heun ODE |
| `losses.py` | flow、bond、collision、operation-level symmetry loss |
| `geometry.py` | C0 canonical bond/angle/torsion/local-pair/ring/chirality contract 与 loss |
| `orbit_kinematics.py` | stabilizer fixed-subspace 参数化与 differentiable exact group lifting |
| `train.py` | 训练、验证、checkpoint、exact resume、manifest |
| `overfit.py` | 冻结 4→16→32 nested overfit Gate；前一层失败即停止 |
| `diagnose_overfit.py` | 失败后只读 checkpoint/time/ODE-step 诊断，不改 Gate |
| `audit_h_protocols.py` | H0–H3 structured diff、冻结配置与 SHA-256 审计 |
| `audit_h_reproducibility.py` | actual H1–H3 checkpoint 的 raw 双重复审计 |
| `summarize_h_experiments.py` | H1–H3 结果、artifact hash 和分支决定汇总 |
| `build_c0_protocol.py` | C0 multi-noise curriculum、局部 Gate 和执行修订冻结 |
| `c0_local_recovery.py` | C0 训练、4-step rollout、local/original raw 双 Gate |
| `audit_c0_reproducibility.py` | C0 formal checkpoint 的 local/raw 数组级重复审计 |
| `summarize_c0_experiment.py` | C0-v1 调度错误与正式 C0-v2 结果汇总 |
| `build_m0_protocol.py` | M0 oracle 重建协议和 Gate 冻结 |
| `m0_oracle_reconstruction.py` | oracle orbit-IC energy、确定性多起点求解和评估 |
| `audit_m0_reproducibility.py` | M0 选中起点的独立重算审计 |
| `summarize_m0_experiment.py` | M0 Gate、起点收敛率和 artifact hash 汇总 |
| `orbit_ic_model.py` | quotient multigraph encoder 与 bond/angle/torsion orbit heads |
| `build_m1_protocol.py` | M1 模型、训练、预测/重建双 Gate 冻结 |
| `m1_bond_angle_training.py` | M1 训练及 learned-IC 多起点3D评估 |
| `evaluate_m1_checkpoint.py` | 冻结 checkpoint 的 CPU decoder 评估入口 |
| `audit_m1_reproducibility.py` | M1 prediction 与选中 decoder seed 重算审计 |
| `summarize_m1_experiment.py` | M1 训练证据、执行修订和结果汇总 |
| `build_m2_protocol.py` | M2 全学习 IC、oracle-free decoder 与双 Gate 冻结 |
| `m2_torsion_training.py` | M1 权重迁移、torsion head 训练及 all-learned IC 重建 |
| `audit_m2_reproducibility.py` | M2 prediction 与 selected decoder seed 独立复算 |
| `summarize_m2_experiment.py` | M2 Gate、复现和 artifact hash 汇总 |
| `audit_m2_tier16_panel.py` | 非平面/手性支持、quotient alias 与 phase-aware Tier-16 面板审计 |
| `build_m2p1_protocol.py` | phase-aware Tier-16 权重迁移、训练预算与双 Gate 冻结 |
| `m2p1_phase_training.py` | M2.1 PG-balanced 训练及 prediction-before-decoder Gate |
| `diagnose_m2p1_torsion.py` | M2.1 局部可交换末端 torsion 塌缩只读诊断 |
| `local_rotor.py` | graph-only local rotor set 合同和可微 circular assignment/loss |
| `audit_m2p2_rotor_sets.py` | M2.2 表示覆盖 Gate A |
| `m3p3_worst_molecule_refinement.py` | M3 Gate-aligned worst-molecule torsion refinement |
| `global_shape_fingerprint_model.py` | atom-order-invariant WL fingerprint + global-shape head |
| `m3p6_fingerprint_shape_training.py` | M3.6 shape-head-only 训练与冻结 Gate |
| `m3p6_shape_decoder_gate_v2.py` | 不读取 target 的多起点 shape-aware 正式 decoder Gate |
| `audit_m3_reproducibility.py` | M3 selected-start 与最终 Gate 独立复算 |
| `factorized_automorphism.py` | 可扩展的 action-compatible graph witness 与 torsion-permutation closure |
| `factorized_cache.py` | pickle-free factorized contract NPZ、selection fingerprint 与严格回读 |
| `precompute_factorized_cache.py` | 可中断续建的全数据合同缓存与 hash manifest |
| `build_dataset_training_protocol.py` | 数据级 smoke/full 数据、训练、证据与 cache identity 冻结 |
| `dataset_training.py` | PG-balanced 数据级训练、checkpoint history 和精确续训 |
| `audit_dataset_training_smoke.py` | 256-step 工程 smoke 的独立报告语义审计 |
| `audit_dataset.py` | C2/C3 数据合同审计 |
| `configs/` | 2-step smoke 与 Stage-1 配置 |
| `tests/` | 群闭包、strict failure、quotient、等变/对称、真实 v2 测试 |

## 9. 执行命令

工程测试：

```bash
cd /home/tianyajun/MARL_for_COFs
conda activate env_etflow

python -m unittest discover \
  -s generative_model/PG_OrbitFlow/tests -v

python -m generative_model.PG_OrbitFlow.audit_dataset \
  --package-dir generative_model/data/processed/v2 \
  --point-groups C2 C3 \
  --output generative_model/PG_OrbitFlow/reports/dataset_contract_c2_c3_v1.json
```

两步 smoke：

```bash
python -m generative_model.PG_OrbitFlow.train \
  --config generative_model/PG_OrbitFlow/configs/smoke_cpu.json \
  --output-dir generative_model/PG_OrbitFlow/runs/smoke_manual
```

下面的 Stage-1 C2/C3 配置属于已停止的 Cartesian coordinate-flow 历史分支；其 Tier-4 质量
Gate 未通过，当前不得启动。它不代表已经通过 M3 的 internal-coordinate 路线。

```bash
CUDA_VISIBLE_DEVICES=4 python -m generative_model.PG_OrbitFlow.train \
  --config generative_model/PG_OrbitFlow/configs/c2_c3_stage1.json \
  --output-dir generative_model/PG_OrbitFlow/runs/c2_c3_stage1_v1 \
  2>&1 | tee generative_model/PG_OrbitFlow/runs/c2_c3_stage1_v1.console.log
```

该配置原计划使用全部 1,984 个 C2/C3 IID-train 分子，PG-balanced 1:1 exposure，4,096
optimizer steps、batch size 4；目前只作为历史候选配置，不应绕过 Tier-4 stop rule。

已验证运行环境：Python `3.11.15`、PyTorch `2.5.0+cu121`、NumPy `1.26.2`、RDKit
`2024.03.5`。训练入口在第一次 CUDA 运算前固定
`CUBLAS_WORKSPACE_CONFIG=:4096:8`，并启用 PyTorch deterministic algorithms。
