# Coordinate Flow Matching 新计划与可行性评估

**计划版本**：v2026-08-22  
**状态**：F0 与工程 Gate 已通过；Fast-32 正式质量 Gate 失败，endpoint flow 分支已按预注册规则停止  
**历史保留**：G0–G2.2、Semla absolute-x0 Fast-32、EQGAT Fast-32 的计划与结论全部保留，
本文件只定义后续主线，不回写或删除旧实验。

## 1. 计划要解决的准确问题

当前主任务限定为：

> 已知合法、不可变的全原子 chemical graph 和 `Target_PG`，从中心化高斯坐标 prior 生成
> 化学几何合理、无严重碰撞、对目标点群有响应的 3D coordinates。

现阶段不解决：

- 从 `Target_PG` 直接生成新的原子、键或 SMILES；
- 为一个从未见过的新 graph 自动推导 symmetry permutation/orbits；
- S4/D6h 的主要定量结论；
- COF 晶体组装和周期结构生成。

因此，即使本计划全部通过，也只能证明 `graph + Target_PG → 3D` 子问题可用，不能直接宣称
已经完成 `Target_PG → 新分子`。

## 2. 为什么从 absolute-x0 改为 flow matching

Fast-32 已证明 Semla/EQGAT 的 E(3)、排列、梯度、13 元素和 PG condition 通路工作，但把
官方 flow backbone 当作单步 absolute clean-coordinate 回归器时，独立验证 RMSD ratio 为
`3.317/4.228`。这与上游原生用途不一致。

新计划按 SemlaFlow 的 endpoint flow matching 语义实现：

\[
x_t=(1-t)x_0+t x_1,\qquad t\in[0,1-\epsilon]
\]

其中：

- `x0`：按分子质心归零的高斯 prior；尺度按训练数据/原子数条件校准；
- `x1`：canonical v2 中质心归零的真实全原子坐标；
- 模型输入：`x_t`、显式 time、immutable graph、`Target_PG`；
- 第一实现预测 endpoint `x̂1`，最大限度复用官方 `Integrator` 的
  `(x̂1-x_t)/(1-t)` 速度定义；
- 采样：从 `x0` 出发做固定步数 ODE integration，得到最终坐标。

不继承 Semla absolute-x0 或 EQGAT Fast-32 权重；两者任务参数化不同，只复用 Semla
`EquiInvDynamics` 架构和已通过的适配/工程测试。

## 3. 分阶段执行计划

### F0：flow contract 与数学单元测试

先实现且只实现以下内容：

1. centered Gaussian prior，禁止 padding 原子参与质心/方差；
2. `x_t` 插值和 endpoint/velocity 公式；
3. 显式 time embedding，不能继续把 time 模糊地当作普通 sigma；
4. Euler/Heun 至少一种固定积分器；
5. fixed graph 与节点顺序全程不变；
6. 坐标 Å ↔ Semla GEOM normalization 的双向一致性；
7. E(3)、节点排列、finite backward、同 seed 重现和 checkpoint resume 测试。

F0 只判工程正确性，不用分子质量指标选择模型。

### F1：同一 Fast-32/8 面板训练

继续使用已经冻结且未发生泄漏的面板：

- IID-train：24 C2 + 8 C3；
- IID-validation：6 C2 + 2 C3；
- test/Core-OOD：不读取；
- 最大原子数：训练 63、验证 60；
- 面板没有 Sn，因此另加一个 Sn 样本的只读 forward/no-fallback 工程测试，但不把它混入
  Fast-32 模型质量统计。

训练需要在 protocol 冻结前明确：模型参数量、batch/pair budget、`t` 分布、prior scale、
loss 权重、积分步数、seed、checkpoint 间隔和停止条件。建议先使用 endpoint prediction，
不同时比较 endpoint/velocity 两套目标。

### F2：Fast Flow Gate

Gate 同时评估“插值恢复”和“从 prior 采样”，不能只看训练 loss：

| 类别 | 建议检查 |
|---|---|
| 工程 | 坐标/梯度有限、质心归零、graph fingerprint 不变、恢复训练一致 |
| 插值恢复 | 固定 `t=0.25/0.5/0.75` 下，endpoint prediction 相对 `x_t` 有稳定改善 |
| prior 采样 | 8 validation × 固定 seeds，从真正 Gaussian prior 完成多步积分 |
| 几何 | collision-free、bond-length MAE、angle/pair-distance 分布 |
| 对称性 | raw symmetry error、actual-PG exact/compatible、hard projection 前后变化 |
| 条件有效性 | correct PG 与 swapped/no-PG 对照；正确条件应降低目标 symmetry error |

正式阈值必须由 protocol 在训练前冻结。建议使用相对基线和比例指标，不恢复单构象
`RMSD≤0.05 Å` 的一票否决。hard projection 的收益必须单独报告，不能算成 flow 模型自身
学会了 symmetry。

### F3：2,026 条 IID-train 正式训练

只有 F2 通过后才启动：

- 训练：IID-train 2,026；
- 模型选择：IID-validation 253；
- test 253 与 Core-OOD 保持封存；
- batch 按总原子数或 dense-pair 数量动态分桶，避免 105 原子样本造成显存尖峰；
- C2/C3 作为主定量，S4/D6h 采用 class-balanced sampler 并标记探索性；
- 保存 EMA/普通权重、optimizer、protocol fingerprint、环境版本和 artifact hashes；
- 先跑固定短预算 profiling，再给出实际吞吐和总训练预算，不提前承诺耗时。

### F4：对称性增强，只允许一轮

若 F3 证明 geometry 可学但模型忽略 PG，只允许一次预注册增强：

- correct-PG vs shuffled-PG contrastive symmetry objective；
- PG condition dropout，用于建立 no-PG baseline；
- target operation/permutation 的 soft symmetry loss；
- hard projection 仍只作为独立 constraint/postprocess 对照。

若正确条件相对 shuffled/no-PG 仍无显著收益，停止纯学习式 PG conditioning，转向
symmetry-by-construction/template 路线，而不是继续增加 backbone。

### F5：与 2D graph 生成衔接

Flow 模型通过后，才能接收 2D/template 模块产生的新 graph。这里存在一个尚未解决的接口：
新 graph 没有 v2 中预存的 target atomic permutation/orbits。候选解决方案是：

1. 2D generator 同时输出 graph automorphism orbit/attachment orbit；或
2. 由 graph automorphism + 元素约束推导候选群作用；或
3. 使用 symmetry-by-construction 模板直接保证 node correspondence。

在该接口完成前，flow 模型只能对数据集中已有 graph 或具有明确 symmetry mapping 的新
template graph 工作。

## 4. 可行性结论

| 目标层级 | 可行性 | 依据 |
|---|---|---|
| F0 工程实现 | 高 | 官方 backbone/integrator 源码已在本地；环境、E(3)、排列和严格 adapter 已通过 |
| Fast-32 coordinate flow | 中高 | loss、symmetry、PG sensitivity 已显示学习信号；任务参数化可与上游对齐 |
| 2,026 条 graph+PG→3D | 中 | 数据量足以做域内条件训练，但类别失衡、单构象监督和最大 105 原子增加难度 |
| 对已有 graph 生成 PG-compatible 3D | 中 | hard projection/operations 已验证，但 raw PG 学习和 relaxation 稳定性尚未证明 |
| 仅给 PG 生成全新 3D 分子 | 中低 | 仍缺 2D graph generator、新 graph symmetry mapping 和端到端联合验证 |

综合判断：**作为当前子任务，计划可行，值得实现；作为最终完整目标，它是必要但不充分的
一段链路。** 最大的不确定性不是代码能否运行，而是 2,532 条单构象数据能否让模型学会
可泛化的 PG 条件效应，而不是只依赖 graph 本身。

## 5. 优点

- 与 SemlaFlow backbone 和官方积分语义一致，不再错误使用现有模型；
- 从 Gaussian prior 真正生成坐标，比“给真实构象加小噪声再恢复”更接近生成任务；
- 固定 graph 后不再承担 atom/bond validity，直接绕开 UAE/MiDi 的 N² bond bottleneck；
- 复用 v2 operations/permutations/orbits、统一 analyzer 和 hard projection；
- 训练、采样和 PG condition 可以分别做对照，结论更容易解释；
- 保留 13 元素和显式 H，Sn 不经过外部 vocabulary；
- 已经有固定 32/8 快验面板，可在投入 2,026 条长训前快速发现数学或采样错误。

## 6. 缺点与主要风险

- 训练目标比单步去噪复杂，需要正确处理 prior scale、time、endpoint 和积分器；
- endpoint velocity 在 `t→1` 时包含 `1/(1-t)`，数值稳定性需要 epsilon/step schedule；
- v2 是一条 SMILES 一个构象，模型可能把构象选择当作确定映射，低估多模态性；
- graph 本身已强烈暗示 C2/C3，模型可能忽略 PG token，必须做 shuffled/no-PG 对照；
- C2 占主导，S4/D6h 样本太少，不能期待同等质量；
- explicit H 和 dense pair attention 增加显存，105 原子样本需要动态 batching；
- hard projection 虽保证 stored group action，但可能恶化键长或制造碰撞，必须与 raw output
  分开报告；
- Fast-32 没有 Sn，只能证明接口支持，不能证明 Sn 的生成质量；
- 对新 graph 缺少 target permutations/orbits，限制了最终与 2D generator 的直接连接。

## 7. 停止条件与防止再次拖慢

1. 不增加第三个 backbone；主干固定为 Semla `EquiInvDynamics`。
2. F0 数学/工程测试未通过时禁止训练，不用调 learning rate 掩盖接口错误。
3. Fast Flow Gate 只允许一次 endpoint 主实验；若采样失败但 endpoint recovery 通过，只允许
   一次积分器修复。
4. 若 correct-PG 不优于 shuffled/no-PG，只允许 F4 的一轮条件增强。
5. F4 后仍无 PG 条件收益，停止学习式 PG conditioning，切换模板/hard-symmetry baseline。
6. 只有 Fast Flow Gate 通过才提供 2,026 条正式训练命令；test/Core-OOD 不用于救活模型。

## 8. 最终执行结果与路线切换

唯一一次 1,024-step Fast-32 已完成。尽管 loss 和低/中噪声 recovery 改善，`t=0.75`
recovery ratio 为 `1.0370`，prior sampling collision-free 仅 `1/16`，raw analyzer/compatible
均为 `0/16`，correct PG 仅在 `6/16` cases 优于 swapped PG。正式状态为
`FAIL_COORDINATE_FLOW_FAST32_STOP_ENDPOINT_BRANCH`。按第 7 节冻结规则，不能执行 integrator
repair，也不进入 2,026 条长训。

随后已实现 orbit-representative symmetry-by-construction 接口，并在 2,279 条 IID
train/validation 上零严格失败通过；最大操作误差 `6.11e-7 Å`，collision-free 100%，包含
2 条 Sn。当前主线由 F3/F4 切换为 R4 hard-symmetry baseline。完整结果见
[Fast-32 Flow 最终结果与 Orbit 硬对称接口](16_Fast32_Flow结果与Orbit硬对称接口.md)。

这里仍不能宣称已经获得可用生成器。后续 O0 工程 Gate 通过，但 512-step O1 因低噪声退化、
碰撞和总体 ratio 正式失败，不进入 learned sampler。hard-symmetry 表示层保留，下一步转向
orbit-constrained force-field relaxation。详见
[Orbit-coordinate O1 最终结果](18_Orbit_Coordinate_O1结果.md)。
