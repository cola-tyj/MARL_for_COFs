# PG-conditioned ET-Flow：EF2 实现与首轮训练协议

## 1. 创新主线

EF2 的核心研究对象不再是普通 COF domain adaptation，而是：

> 给官方 ET-Flow 增加 Target_PG、O(3) 群操作、原子 permutation/orbit 条件和 raw-endpoint
> symmetry loss，使模型在不执行 hard projection 的原始输出中响应目标点群。

普通无条件 COF fine-tune 保留为后续对照组，不再作为创新主线的前置阻塞。

## 2. 已实现模型

新增 [etflow_pg_conditioned.py](../../../generative_model/models/etflow_pg_conditioned.py)，在第三方
`etflow` 包之外包装官方 `BaseFlow`，不修改 canonical graph，也不覆盖官方 checkpoint。

```text
official ET-Flow velocity
          +
zero-initialized learned soft group-action restoring velocity
          ↓
PG-conditioned velocity field
```

条件输入为：

- `Target_PG` embedding（当前只支持 C2/C3）；
- 完整 `operation_matrices`；
- 长度为 N 的 `permutation_index`；
- 每个原子的 `orbit_size`。

软引导根据当前位置到 Reynolds invariant subspace 的残差构造等变方向，但只把它作为可学习
velocity correction；模型 forward 不用 projected coordinates 覆盖原坐标。因此它不同于 E3/F0.2
部署后端的 hard projection。

适配器最后一层严格零初始化。初始化时：

```text
conditioned_velocity == official_base_velocity
```

这样可从官方 `drugs-o3` 权重开始，同时把模型提升归因于新增条件模块。

### 2.1 当前模型如何实现并检验目标对称性

当前条件不是单独的 `Target_PG` 类别 token。对每个目标点群 (G)，接口同时提供完整的
三维群操作 (R_g\in O(3)) 和同元素、同图作用约束下的原子 permutation (p_g)。使用行向量
坐标约定时，目标结构应满足：

\[
XR_g^T\approx X[p_g],\qquad g\in G.
\]

推理阶段的 permutation 由 known graph 和 Target_PG 恢复，不读取 reference XYZ，也不通过
坐标 Hungarian assignment 猜测原子对应关系。C2/C3 action、atom permutation 和 orbit size
共同构成比普通 PG embedding 更强的结构条件。

对 official ET-Flow 在时刻 (t) 的输出 (v_{base})，A2 首先估计未经条件修正的端点：

\[
Y_{base}=X_t+(1-t)v_{base}.
\]

随后利用全部群操作计算 Reynolds average：

\[
P_G(Y_{base})=\frac{1}{|G|}\sum_{g\in G}\mathcal{T}_g(Y_{base}),
\]

其中 \(\mathcal{T}_g\) 同时执行 (R_g) 和 (p_g)。因此

\[
\Delta_{sym}=P_G(Y_{base})-Y_{base}
\]

是把 official 预测端点拉向目标点群 invariant subspace 的方向。零初始化、graph-shared learned
gate (a_\phi) 只把该方向作为 velocity correction：

\[
v=v_{base}+a_\phi
\frac{\Delta_{sym}}{\max(1-t,0.05)}.
\]

forward 不以 (P_G(Y)) 覆盖 raw coordinates，所以这是 learned soft restoring velocity，不是
生成后 hard projection。不同 Target_PG 会给出不同的 operation/permutation 集合，因而即使共享
同一 graph 和 prior，也会产生不同的三维轨迹。

训练和验证用 operation-level symmetry error 检查响应：

\[
e_g(X)=\sqrt{\frac{1}{N}\sum_i
\left\|x_iR_g^T-x_{p_g(i)}\right\|^2}.
\]

EF2-A2 fixed-endpoint 的 mean symmetry-error ratio 为 `0.005408`；EF2-B 完整 raw ODE 的
correct-action ratio 为 `0.014205`，32/32 均比 official base 改善；correct/alternate action 的
median Kabsch RMSD 为 `2.3777 Å`。这些结果证明模型响应群作用，而不是忽略 Target_PG。

这里的“保证”必须按两层表述：learned adapter 负责把 raw trajectory 强烈推向对称子空间；只有
hard projection 才提供数值精度内的严格群不变保证。raw operation error 很小也不等于已经通过
independent actual-PG analyzer，更不自动保证化学合理性；原子可以在满足群作用的同时发生非键
碰撞。EF2-B 正是因此只通过对称响应而未通过最终 raw geometry Gate。

## 3. 损失函数的数学修正

保留官方 linear-path flow matching：

\[
L_{flow}=\|v_\theta(x_t,t)-u_t\|^2.
\]

由预测速度估计原始终点：

\[
\hat X_1=X_t+(1-t)v_\theta(X_t,t),
\]

再计算：

\[
L_{sym}=\frac{1}{|G|}\sum_g
\|\hat X_1R_g^T-\hat X_1[p_g]\|^2.
\]

最终为 `L_flow + λ_sym L_sym`。代码明确禁止把任意 noisy-time velocity 本身强制为群不变量，
因为非对称噪声需要恢复速度主动消除；错误约束会妨碍 denoising。

## 4. 严格数据接口

新增 [etflow_pg_adapter.py](../../../generative_model/data/etflow_pg_adapter.py)：

- canonical explicit-H graph 通过既有 mapped-SMILES bridge 进入官方 featurizer；
- `model_to_canonical` 只按 atom map + heavy-parent graph lifting 得到；
- group action 通过解析公式搬运到 ET-Flow atom order，不使用坐标 Hungarian；
- Sn 始终为 Z=50，不允许映射成 C/Si；
- training/validation 可使用 v2 target symmetry annotation 作为监督；
- inference 必须使用 E3-A graph-only recovered action，禁止读取 stored action。

训练使用 symmetry annotation 不等于推理依赖 annotation；这与训练时读取标签、推理时预测/构造
条件的通常监督学习边界一致。

## 5. EF2-0 工程 Gate

固定面板为 package indices `45, 81, 1684, 2378`，覆盖 C2/C3 和全部两条 Sn。

v1 只失败 `zero_initialized_wrapper_exact`：审计错误地比较了两次独立 CUDA forward，GPU scatter
非确定归约带来最大 `2.18302e-6` 差异。该 FAIL 和 SHA 原样保留；它不是模型差异。

v2 仅把判据改成在同一次 wrapper forward 中比较 `velocity` 与 internal `base_velocity`，其余
panel/checkpoint/model/objective/decision 不变。结果 10/10 checks 通过：

- official checkpoint state fingerprint 前后相同；
- zero-init 最大误差 `0`；
- 4/4 conditioner gradient finite/nonzero，最小 gradient norm `5.85062e-4`；
- C2/C3 和 Sn 2/2 通过；
- 无元素 fallback、无坐标 assignment、无 hard projection。

正式状态：`PASS_ETFLOW_EF2_ENGINEERING_FREEZE_BOUNDED_TRAINING`。

## 6. EF2-A 有界训练协议

已冻结并完成一次运行：

- official `drugs-o3` base 全冻结，只训练 conditioner；
- 256 steps、batch size 1，覆盖固定 256 IID-train 一遍；
- 训练构成为 192 C2 + 64 C3，强制包含 Sn indices `1684/2378`；
- 固定 64 IID-validation：48 C2 + 16 C3；
- IID-test 完全不用于训练、调参或 checkpoint 选择；
- `lr=3e-4`、`λ_sym=1`、gradient clip 1、每 64 steps checkpoint；
- validation 只比较 raw estimated endpoint，不执行 projection/F0.2。

放行阈值：

| 指标 | 阈值 |
|---|---:|
| mean endpoint symmetry MSE ratio | ≤ 0.80 |
| symmetry improved fraction | ≥ 0.60 |
| C2/C3 各自 improved fraction | ≥ 0.50 |
| flow MSE ratio | ≤ 1.15 |

通过只允许进入 raw ODE sampling Gate；不等于已经证明 actual-PG 条件生成。

## 7. 用户在 tmux 中执行

```bash
cd /home/tianyajun/MARL_for_COFs
conda activate env_etflow
set -euo pipefail

OUT=generative_model/runs/etflow_ef2a_adapter_0256
mkdir -p "$OUT"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
python -m generative_model.smoke.train_etflow_ef2a_adapter \
  --device cuda \
  --cache generative_model/checkpoints/etflow \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

sha256sum "$OUT"/{report.json,losses.npz,last.pt} \
  "$OUT"/checkpoints/*.pt
```

脚本拒绝改变冻结的 `maximum_steps=256`，并在训练结束后自动输出 validation Gate。

## 8. 可重复身份

- EF2-0 v1 protocol：`20ca081e4308c34cd84b93ca935e25cee2673fe32d4733b9cc7feedab6fe36ab`
- EF2-0 v1 audit FAIL：`242a6327d0c233a44362e52c42ff5b256ab03bf54cb85985709077a25efff3fb`
- EF2-0 v2 protocol：`3c76ae4c709ee540eab15d7df3782fe34d35b88d689c42413e40eec74e7fa712`
- EF2-0 v2 engineering PASS：`d8ed3a110658ed79b7873df08b5e375b64acd3c2dc5aa1335c86d3051994bfd4`
- EF2-A training protocol：`a020d9005a3cee1e50448b49e847fcbc7e30376d7c116df4dff13e3f40876ddf`
- EF2-A final report：`fa0b5d02d0486e6958b9dd045e4a6c41c81953589d7e2eb17af809d36c26dc93`
- EF2-A losses：`a067eac88f818d419d370e33e095e2814ddca6b7dfe88e7838c5b33eb65e5028`
- EF2-A checkpoint audit：`0fe66f4aea531c45ee4b97c522f5d93abe8b2112d16283d9f0416c5f94fb0cdd`
- EF2-A failure diagnosis：`4aeee839deb6fd41c741436940609199b9dc6ff749e74f9a5811ac16e4bb3853`
- official checkpoint：`a24ae9a1fed2708696929308ed1dc10ab167fd66a2d51c44a4afb6c11badccb2`

## 9. EF2-A 运行结果

最终状态为 `FAIL_ETFLOW_EF2A_ADAPTER_STOP_AND_DIAGNOSE`。这表示冻结 Gate 未通过，
不是程序崩溃，也不是 official checkpoint 损坏。

| validation 指标 | 实际 | Gate | 判定 |
|---|---:|---:|---|
| mean endpoint symmetry MSE ratio | 0.91816 | ≤0.80 | FAIL |
| symmetry improved fraction | 0.51563 | ≥0.60 | FAIL |
| C2 improved fraction | 0.45833 | ≥0.50 | FAIL |
| C3 improved fraction | 0.68750 | ≥0.50 | PASS |
| mean flow MSE ratio | 0.97939 | ≤1.15 | PASS |

official base state 的训练前后 fingerprint 完全相同，全部 loss/gradient finite，两条 Sn 均被
训练面板实际访问。因此不能把本次失败解释为 base 被破坏、数值发散或元素 fallback。

### 9.1 四个 checkpoint 的只读审计

使用同一 64 条冻结 IID-validation、同一 seed/time 和原阈值重新评价 step 64/128/192/256；
未训练、未读取 IID-test、未改变阈值，也未执行 hard projection。

| step | endpoint ratio | overall improved | C2 improved | C3 improved | flow ratio | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 0.96751 | 0.48438 | 0.60417 | 0.12500 | 0.99184 | FAIL |
| 128 | 0.96734 | 0.48438 | 0.60417 | 0.12500 | 0.99177 | FAIL |
| 192 | 0.96546 | 0.45313 | 0.56250 | 0.12500 | 0.99179 | FAIL |
| 256 | 0.91816 | 0.51563 | 0.45833 | 0.68750 | 0.97939 | FAIL |

所以失败不能通过挑选较早 checkpoint 修复。

### 9.2 失败诊断

训练 panel 在文件中按 `192 C2 → 64 C3` 形成两个连续区段，而不是交错采样。step 192 前
C2 improved 尚为 56.25%，C3 只有 12.5%；完成最后 C3 区段后，C3 升到 68.75%，C2
降到 45.83%。这与顺序诱发的点群遗忘一致，但当前只作诊断证据，不宣称严格因果证明。

每一步又对应不同分子，且只访问一次。因此 console 中 step 64/128/192/256 的单样本 loss
不是同一对象的收敛轨迹；报告中的 first-32 与 last-32 也实际是 C2/C3 跨点群比较，不能用于
判定训练发散。

另一个更直接的信号是：按 official base 的 endpoint symmetry error 分四组，最低误差四分位
的 conditioned/base ratio 为 `1.9270`、改善率 `0/16`，而较高两个四分位的改善率为
`13/16` 和 `14/16`。当前 correction direction 由 noisy `x_t` 的 Reynolds residual 构造，
不是由 official base 已预测 endpoint 的 residual 构造；它能修复明显不对称的 endpoint，
却会扰动 base 已经较对称的输出。这是下一版必须单独验证的 conditioning-direction 假设。

## 10. 冻结结论与唯一下一步

EF2-A v1 正式停止：不延长步数、不解冻 official base、不放松 Gate，也不从失败 checkpoint
进入 raw ODE sampling 或 IID-test。

已实现并冻结一次 EF2-A2 因果修复：

1. soft restoring direction 改由 official base-predicted endpoint 构造，仍不覆盖 raw coordinates；
2. 256 条相同训练记录按固定 `C2,C2,C2,C3` 比例交错，消除连续点群区段；
3. 每个 package index 继续使用其在 A1 中对应的原始 step seed，避免交错同时改变训练噪声；
4. checkpoint、64 条 validation、训练样本预算、official base、损失权重和 Gate 阈值全部不变。

新方向定义为：

\[
Y_{base}=X_t+(1-t)v_{base},\qquad
\Delta=P_G(Y_{base})-Y_{base},
\]

\[
v=v_{base}+a_\phi(\mathrm{PG},t,G,\mathrm{orbit},\|\Delta\|)
\frac{\Delta}{\max(1-t,0.05)}.
\]

`a_φ` 是每个 graph 共享的零初始化可学习 gate。模型输出仍是 raw velocity；没有把坐标直接
替换成 `P_G(Y)`，因此不是 hard projection。

### 10.1 A2 工程 Gate

official checkpoint 工程审计已经通过：

- 12 个 A1+A2 定向单元测试全部通过；
- official state fingerprint 前后相同，zero-init 最大误差 `0`；
- 4 个 C2/C3/Sn stress 样本 gradient finite/nonzero，最小 norm `0.00559867`；
- 固定 64 条 validation 上 endpoint direction 为 64/64 改善；
- 诊断 contraction `α=0.25` 的 mean MSE ratio 为 `0.56250003`，理论值为 `0.5625`，
  最大偏差 `9.54e-7`；
- 无 hard projection、元素 fallback 或坐标 assignment。

正式状态：`PASS_ETFLOW_EF2A2_ENGINEERING_FREEZE_INTERLEAVED_TRAINING`。

### 10.2 A2 训练与 checkpoint 选择

训练仍为 256 条唯一记录各访问一次，保存 step 64/128/192/256。每个 checkpoint 都在同一
冻结 64 条 IID-validation 上评价；只有通过全部原 Gate 的 checkpoint 才 eligible，多个
eligible 时选择 endpoint symmetry ratio 最低者，再以较早 step 打破平局。IID-test 不参与。

```bash
cd /home/tianyajun/MARL_for_COFs
conda activate env_etflow
set -euo pipefail

GPU_ID=4
OUT=generative_model/runs/etflow_ef2a2_endpoint_interleaved_0256
mkdir -p "$OUT"

test "$(sha256sum generative_model/smoke/reports/etflow_ef2a2_training_protocol_v1.json | awk '{print $1}')" = \
  "eaf696764009ee46aa827dbd03a061f930c4d1c0825b73d79fc9d2a235ceb256"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -m generative_model.smoke.train_etflow_ef2a2_adapter \
  --device cuda \
  --cache generative_model/checkpoints/etflow \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

sha256sum "$OUT"/{report.json,losses.npz,last.pt} "$OUT"/checkpoints/*.pt
```

## 16. Learned raw 分支收尾后的可交付 E3/F0.2 路线

EF2-D4 已按冻结 Gate 正式停止后，项目没有继续修改阈值或堆训练步数，而是实现了明确分层的
推理接口：官方 ET-Flow 只生成 initial conformer；requested C2/C3 的 graph action 从 canonical
graph 独立恢复；E3 hard projection 保证群作用；F0.2 在 orbit subspace 内做 UFF+repulsion。
该接口不读取 reference XYZ、stored action/orbit，也不使用 ETKDG。

三个未见 IID-test 示例已由独立 env_cof/pymatgen 审计通过 requested-PG compatible 和无碰撞
Gate。该结果只建立 hard-projection composite route 的可交付性，不推翻本章关于 learned raw
分支的负结论，也不表述为“backbone 已学会无需 projection 的点群响应”。完整输入输出、哈希、
数值结果和失败反例见 [28_ETFlow_E3F02可交付推理接口.md](28_ETFlow_E3F02可交付推理接口.md)。

### 19.2 v2 训练结果：对称性有效，碰撞 Pareto Gate 失败

v2 完成全部 256 steps，训练/样本/父权重/无 hard projection 等结构检查全部通过。四个
checkpoint 的结果为：

| step | operation ratio | rollout ratio | C3 rollout ratio | collision ratio | Gate |
|---:|---:|---:|---:|---:|---|
| 64 | 0.970246 | 0.999836 | 0.999758 | 1.000955 | FAIL |
| 128 | 0.932633 | 0.999628 | 0.999448 | 1.002183 | FAIL |
| 192 | 0.874215 | 0.999298 | 0.998946 | 1.003792 | FAIL |
| 256 | 0.821101 | 0.998993 | 0.998475 | 1.005098 | FAIL |

所有 checkpoint 的唯一失败 check 都是冻结的 `nonbonded_collision_loss_ratio ≤ 1.0`。step 256
的 C2/C3 one-step operation 改善率均为 100%，flow ratio `0.998869`、bond ratio
`1.000358`，最大 endpoint correction `0.017286 Å`，说明训练稳定且条件方向有效，但随着
symmetry correction 增强，selection panel 的 collision loss 单调轻微恶化。

绝对量诊断显示，selection 的 mean collision loss 为
`0.0339690→0.0341422`，增加 `1.7317×10^-4`。32 条中 25 条前后均为零；主要增量来自 C2
package `2235`（`0.514885→0.521135`）。后半独立留出集得到 collision ratio `0.996634` 且
全部 checks 通过，但由于 selection 阶段没有任何 eligible checkpoint，这只能视为非晋级的
supporting evidence；不能事后使用 confirmation 覆盖正式 Gate。

正式状态：`FAIL_ETFLOW_EF2D2_OPERATION_ORBIT_TRAINING_STOP_BRANCH`。report SHA-256：
`400319212b5a69caefdbe4e75df608ccbeb224ab618d6cb71a7935858a4a6ae8`。

该结果暴露两个结构问题：

1. 当前 adapter 的可学习位移只能沿 operation-restoring residual 缩放；collision loss 能调 gate，
   但没有独立 repulsion/short-bond 修复方向，因此存在不可消除的 Pareto 冲突。
2. 对 C2 只有一个 nonidentity operation；对 C3 的两个非identity操作是互逆且 Frobenius error
   相等，所以本实验中 `worst-operation≈mean-operation`，不能约束“跨分子的碰撞长尾”。下一步
   更应使用 collision-active hard-example sampling 或 molecule-level CVaR/worst-case loss。

按冻结规则，该 checkpoint 不进入 raw ODE、IID-test 或 Core-OOD，也不通过降低 `1.0` 门槛
事后晋级。若继续 learned 路线，应新建 EF2-D3：从冻结 A2 重新训练，保留 operation/orbit
basis，同时加入独立的 Target_PG-compatible nonbonded/short-bond vector basis，并在 IID-train
内部预注册 collision-active 平衡采样和 molecule-level tail Gate。

## 18. EF2-D1：operation/orbit-aware 工程准入

C7-RS 说明 1:1 重采样能提高典型 C3，但不能控制完整 trajectory 的最差操作和长尾。因此新分支
不修改旧 A2/C7 权重，而是在冻结的 A2 step-256 上增加零增量 residual head：

- 对每个非 identity 操作计算按 permutation 对齐的逐原子 restoring residual；
- 所有操作共享 scorer，再沿 operation 维做 softmax，避免依赖 C3 操作枚举顺序；
- 对 operation error 做 orbit mean/max pooling 并广播回原子，orbit ID 重编号不改变结果；
- Target_PG、time、orbit size/fixed-atom 和 atom/orbit/graph error 控制 atom-wise gate；
- 每原子的 estimated-endpoint correction 使用 smooth cap，最大 `0.05 Å`；
- gate 末层严格零初始化，未训练时逐位复现 A2；禁止 hard projection。

损失不再只有整分子 scalar symmetry：

\[
L=L_{flow}+0.25L_{mean-op}+0.5L_{worst-op}+0.5L_{orbit}
  +0.25L_{bond}+0.25L_{nonbond}.
\]

工程 Gate 使用从 IID-train 单独留出的 C2 32 + C3 32；它们与 EF2-A2/C7-RS 训练集交集为
0。IID-validation、IID-test、Core-OOD 均未读取。7 项单元测试全部通过，真实样本结果为：

| 指标 | 结果 | 冻结门槛 | 判定 |
|---|---:|---:|---|
| direction mean-operation 改善率 | 1.0000 | ≥0.90 | PASS |
| direction worst-operation 改善率 | 1.0000 | ≥0.75 | PASS |
| 4-step rollout operation ratio | 0.996852 | ≤1.0 | PASS |
| 4-step rollout 改善覆盖率 | 0.9375 | ≥0.75 | PASS |
| rollout bond-loss ratio | 1.000189 | ≤1.25 | PASS |
| 最大 endpoint delta | 0.046488 Å | ≤0.050001 Å | PASS |
| zero-increment / Sn / parent frozen | 全部通过 | 必须通过 | PASS |

正式状态：`PASS_ETFLOW_EF2D1_OPERATION_ORBIT_ENGINEERING_FREEZE_TRAINING`。

- engineering protocol SHA-256：`092774634330d522399dee74b9f4d8b786beae12223ae130e24a6f9a54d33124`
- engineering report SHA-256：`8c6e2ff5d6c185005ea490e83012526aceb176c23bd063cb448e748dc524e785`

## 19. EF2-D2：已冻结的 paired C2/C3 训练

该阶段不是继续做“重采样-only”，而是训练 EF2-D1 已通过的显式 operation/orbit head：

- 256 个 optimizer steps；每个 step 严格计算一条 C2 和一条 C3，再对两者 loss 等权平均；
- 共 512 个唯一 IID-train 分子，C2/C3 各 256，包含 Sn 2/2；
- 全部 64 条 EF2-D1 development holdout 从训练中排除；
- 每 4 step 加一次两步 differentiable short rollout，并在 rollout 后估计 endpoint，施加同一组
  operation/orbit/bond/nonbond 结构目标；
- 每 64 step 保存 checkpoint。留出集前半 C2/C3 各 16 用于选择，后半各 16 只在选定后确认；
- official ET-Flow 和 A2 parent 全冻结，不读取 IID-test/Core-OOD，不做 hard projection/力场修复。

训练协议 SHA-256：
`1ee5f798230b2e8bc7eb4b2b1be515fe9d12a9da2b2efbf6ee33c9340143ee1d`。

当前仅能声称“operation/orbit head 已通过工程准入并冻结训练设计”。只有 EF2-D2 的 endpoint +
short-rollout Gate、之后的独立 raw ODE 和 point-group audit 都通过，才可声称 learned raw
Target_PG 响应得到验证。

下面的 v1 命令仅保留为首次启动记录，已知会在 NumPy seed 初始化失败，请勿再次执行；当前有效
命令见 19.1 的 v2 修复。

```bash
cd /home/tianyajun/MARL_for_COFs
conda activate env_etflow
set -euo pipefail

GPU_ID=4
OUT=generative_model/runs/etflow_ef2d2_operation_orbit_0256_v1
mkdir -p "$OUT"

test "$(sha256sum generative_model/smoke/reports/etflow_ef2d2_operation_orbit_training_protocol_v1.json | awk '{print $1}')" = \
  "1ee5f798230b2e8bc7eb4b2b1be515fe9d12a9da2b2efbf6ee33c9340143ee1d"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -m generative_model.smoke.train_etflow_ef2d2_operation_orbit \
  --device cuda \
  --cache generative_model/checkpoints/etflow \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

sha256sum "$OUT"/{report.json,losses.npz,last.pt} "$OUT"/checkpoints/*.pt
```

### 19.1 v1 启动失败与 v2 执行修复

v1 在训练 step 1 前执行 `np.random.seed(202608263300)` 时退出，因为 NumPy legacy
`RandomState` 只接受 `[0, 2^32-1]`。此时尚未构建模型/optimizer、读取训练样本或写 checkpoint；
失败目录仅保留 `console.log`，SHA-256 为
`eb84fd1c60c794a7d977561b64225ed106f4f6e2faf79ce19dee3c10ddfadfdb`。

v2 唯一改动是：NumPy 使用
`744800388 = 202608263300 mod 2^32`；Torch/CUDA 仍使用完整 seed `202608263300`。模型、512
条样本、C2/C3 pairing、loss、optimizer、步数、checkpoint cadence、留出集和 Gate 均不变。
v1 失败记录不覆盖，v2 从 step 1 写入新目录。v2 协议 SHA-256：
`a92e2393feac1d29f27dbd315d9afcf4fe36e243826483f05330b5b045caafbd`。

```bash
cd /home/tianyajun/MARL_for_COFs
conda activate env_etflow
set -euo pipefail

GPU_ID=4
OUT=generative_model/runs/etflow_ef2d2_operation_orbit_v2_0256
mkdir -p "$OUT"

test "$(sha256sum generative_model/smoke/reports/etflow_ef2d2_operation_orbit_training_protocol_v2.json | awk '{print $1}')" = \
  "a92e2393feac1d29f27dbd315d9afcf4fe36e243826483f05330b5b045caafbd"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -m generative_model.smoke.train_etflow_ef2d2_operation_orbit_v2 \
  --device cuda \
  --cache generative_model/checkpoints/etflow \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

sha256sum "$OUT"/{report.json,losses.npz,last.pt} "$OUT"/checkpoints/*.pt
```

### 19.2 EF2-D2 最终结果

EF2-D2 v2 已完成 256 steps，不存在执行、finite、数据泄漏或 checkpoint 结构问题。
step-256 将 selection panel 的 mean/worst operation ratio 降至 `0.821101`，C2/C3 的
operation-improved fraction 均为 `1.0`，flow ratio `0.998869`，short-rollout ratio
`0.998993`。但 nonbonded collision ratio 为 `1.005098 > 1.0`，且四个 checkpoint 都只在
这一项失败，因此不允许事后挑 checkpoint。碰撞绝对均值增量为 `1.7317e-4`，主要由
package `2235` 的 `0.514885→0.521135` 驱动。后半 confirmation panel 虽通过，但只是
非晋级诊断。正式状态：`FAIL_ETFLOW_EF2D2_OPERATION_ORBIT_TRAINING_STOP_BRANCH`。

### 19.3 EF2-D3 双基底有界收尾诊断

EF2-D3 不延长 D2，而是回到冻结 A2，将 operation/orbit restoring basis 与独立的
nonbonded/short-bond repulsion basis 合并；两个 gate 均为零初始化，合并 endpoint delta 共享
`0.05 Å` trust region，不使用 hard projection 或力场。

| 诊断 | 单步 collision ratio | 单步 bond-overlap ratio | 4-step collision ratio | 梯度 | 判定 |
|---|---:|---:|---:|---|---|
| D3-v1 endpoint overlap | 0.565250 | 0.290422 | 1.000501 | operation NaN | STOP |
| D3-v2 endpoint + trajectory overlap | 0.957837 | 0.895009 | 1.005786 | operation NaN | STOP |

D3-v2 已在与 D1/D2/D3-v1 零交集的全新 C2/C3 各 24 条 IID-train holdout 上运行，
13 项单元测全部通过。其失败不是 repulsion direction 错误：方向 panel 的活跃碰撞/短键
案例改善率都为 100%；问题是局部 endpoint direction 不能推导出多步 ODE 后的单调几何
改善，且 overlap 几何计算图仍可向 operation head 传播 NaN 梯度。因此不得降低 Gate、
扩大训练或将其称为可用模型。若再开 learned 分支，必须首先使 loss/geometry 的反向边界
全部数值安全，并对积分轨迹而不是只对 endpoint 定义 collision trust region/CVaR。

- D3-v1 protocol/report SHA-256：`5fb8291bbbbbd5d82ecaa82b96ca23c3a6c6e4316ac3b4dc76acda6940b80cc3` /
  `2c8906cc811fb877b1ca20cf7b520c7fce444f2a675b87cb2258a6a32edd7fc0`
- D3-v2 protocol/report SHA-256：`9b5d54f0440bcef27eedafad35e09fd6742f76f1c38eebb6fac610e62b4a1960` /
  `e05d59857adae4c41cd6b254d67251cbb8c28b136936492374858739659ed148`

### 19.4 EF2-D3-v3：gradient isolation 成功，trajectory Gate 仍失败

D3-v3 不修改 forward 数值，只在构造 analytic overlap vector field 时对 endpoint/trajectory
坐标做 stop-gradient，以阻断 inactive distance hinge 对 operation head 的 NaN 反向污染。两个
learned gate 的参数梯度保留，zero increment 语义不变。13 项测试全过，且在第三套
全新 C2/C3 各 24 条 holdout 上，之前的 operation-gradient 失败已消失。

唯一失败项仍是 4-step collision ratio：`1.001477 > 1.0`。其他检查全部通过，包括
operation、bond、archived package `2235`、端点 cap、parent frozen 和 no hard projection。这将问题
确定为 trajectory-level tail risk，而不再是梯度数值错误。正式状态：
`FAIL_ETFLOW_EF2D3_V3_ENGINEERING_STOP_BRANCH`。

### 19.5 EF2-D4：molecule-level trajectory CVaR 最终收尾

D4 作为新分支保留 D3-v3 failure，并将训练单位改为每步 4 C2 + 4 C3。共复用 D2
的 512 个唯一 IID-train 分子，对 endpoint 和每步 4-step differentiable rollout 分别计算
collision/short-bond 风险，并仅对每批最差 25% 的分子使用 CVaR 强化。selection 使用
IID-val C2/C3 各 32 条，confirmation 使用互斥的各 14 条；IID-test/Core-OOD 封存。

v1 完成 16 步后在首个 checkpoint 写入前因局部路径变量被 model output dict 覆盖而退出，
没有 checkpoint。v2 只修复变量名，科学协议不变，从 step 1 写新目录。完整结果：

| step | operation ratio | endpoint collision | rollout collision | rollout 恶化占比 | Gate |
|---:|---:|---:|---:|---:|---|
| 16 | 0.998327 | 0.999805 | 0.999977 | 17.19% | FAIL |
| 32 | 0.997194 | 0.999601 | 0.999952 | 17.19% | FAIL |
| 48 | 0.995933 | 0.999408 | 0.999929 | 12.50% | FAIL |
| 64 | 0.993881 | 0.999239 | 0.999960 | 15.625% | FAIL |

step 64 的 endpoint operation/collision/bond/flow 和总体 rollout collision 均通过，但 tail incidence
`15.625% > 10%`。step 48 是最佳非准入 checkpoint，仍同时在 operation 与 tail 失败；
对它的 confirmation 仅作诊断，不能倒推晋级。因此没有 selected checkpoint，不得进入 raw ODE
或 IID-test。正式状态：`FAIL_ETFLOW_EF2D4_TRAJECTORY_CVAR_STOP_LEARNED_DUAL_BASIS`。

- D3-v3 protocol/report SHA-256：`8c03e1b41ffe70c73cc5be06c5e215365441c3c94a6e29ee1d6239abdfaf9483` /
  `6fa378818a4e22a18a64cc6cb736ab38d122272b9bd6bd74a5caf87cbf2c772a`
- D4-v2 protocol/report SHA-256：`9792523fa85ea666760024902e1bee0382a74bdb00448aa654832a566d69a614` /
  `7763392e52750bbe80047456ed66be446fe8350fc0edafd23a7e07f818ea2ac9`

实用路线因此转向已通过 EF1 证据的 `ET-Flow learned conformer → E3 graph action/hard
symmetry projection → F0.2 orbit-constrained UFF + repulsion`。该路线不使用 ETKDG 生成，但必须
和“raw learned Target_PG generation”分开报告。

## 16. C7-RS：C2/C3 平衡重采样单变量实验

EF2-C6 在全新 64 条 IID-validation 上的唯一正式失败项是 C3 compatible：`12/16`，而 C2
为 `47/48`。因此 C7-RS 不修改 backbone、conditioner、loss 或 Gate，先只检验“C3 训练暴露
不足”这一条假设。

### 16.1 预冻结设计

- 总预算仍为 256 个唯一 IID-train 分子、batch size 1、每条只访问一次；
- C2/C3 从 `192:64` 改为 `128:128`，顺序固定为重复的 `C2,C3,C2,C3`；
- 最大化与 A2 的可比性：保留 128 个 A2 C2 和全部 64 个 A2 C3，只用 64 个新的 IID-train
  C3 替换 64 个 C2，因此 parent overlap 为 192/256；
- 126 个普通 C2 用固定 seed 的 SHA-256 排序选取，另强制保留两个 Sn 分子；不允许重复、
  不读取 IID-test/Core-OOD；
- parent 中保留的分子继续使用原 per-molecule noise seed；64 个新 C3 确定性继承被替换
  C2 释放的 seed offsets；
- official `drugs-o3` base、A2 endpoint conditioner 架构、zero initialization、optimizer、
  learning rate、symmetry weight、gradient clipping、64 条 validation、paired seeds、
  checkpoint cadence 和所有 endpoint Gate 阈值均不变；
- 本阶段不重训 overlap head，也不执行 hard projection。

这是一项“训练采样分布”的单变量实验，不是新模型结构实验。协议 SHA-256 为
`5c7b298202ccecce1e4c4a0debe6a9b2894296703340370e2bfa6ada1d26895c`。

### 16.2 训练结果

四个 checkpoint 均通过原 A2 endpoint Gate：

| step | validation endpoint ratio | C3 endpoint ratio | flow ratio | Gate |
|---:|---:|---:|---:|---|
| 64 | 0.651399 | 0.621385 | 0.894528 | PASS |
| 128 | 0.065589 | 0.033070 | 0.717117 | PASS |
| 192 | 0.016451 | 0.007491 | 0.702256 | PASS |
| 256 | 0.005375 | 0.003010 | 0.698924 | PASS |

按预冻结规则选择 step 256。训练 256/256 唯一、C2/C3 各 128、Sn 2/2、parent overlap
192、seed 映射、base state 不变、finite gradients 和 no-hard-projection 全部通过。相对原 A2
selected step 256：

- C3 endpoint symmetry ratio：`0.012847 → 0.003010`，candidate/parent=`0.23430`；
- overall flow ratio：`0.699048 → 0.698924`，没有可见退化；
- C2 endpoint ratio 为 `0.008489`，C2/C3 validation 改善率仍均为 100%。

正式状态为 `PASS_ETFLOW_C7RS_BALANCED_A2_ENDPOINT_AWAIT_RAW_ODE`。这支持“C3 重采样能增强
固定时刻 endpoint 收缩”，但不能据此宣称 raw 3D 的 C3 actual-PG 已改善；A2 endpoint 原本
已经很强，而 C6 失败发生在完整 ODE 与独立离散 point-group analyzer。因此下一步必须冻结
一个与 C6/旧小面板不重叠、C3 占比更高的 raw ODE confirmation。只有该 Gate 通过，才值得
以 C7-RS A2 为 parent 重新训练 collision-aware overlap head。

可重复身份：

- protocol：`5c7b298202ccecce1e4c4a0debe6a9b2894296703340370e2bfa6ada1d26895c`
- report：`c05de1576361b03c2a59fc15937961777c8b3bd4b6d8cf05a518ed24e8fb1049`
- losses：`d5268ff561d3277417d22b6b9f335115cb960ac525bc1d0a969a5395fb0f9d87`
- selected step-256 checkpoint：`49717d3c43257911f685979c6c9ade8c813295e9433ffa25564e1f575229bcf3`

### 16.3 不重叠 raw ODE 与独立 point-group 结果

在看到新坐标前，C7-RS raw Gate 固定使用所有剩余未用于旧验证的 17 个 C3，并用 SHA-256
确定性匹配 17 个 C2。34 个分子与 A2 validation、EF2-C3 raw panel 和 EF2-C6 panel 的并集
零重叠。旧 A2 与 C7-RS 从相同 prior、相同 graph-recovered action 和相同 50-step ODE
设置运行；不使用 overlap head、stored action、hard projection 或 force field。

冻结的关键门槛包括：candidate C3 compatible 至少 15/17、C2 至少 16/17、总体 compatible
至少 90%，C3 regression 最多 1 条；连续几何还要求 C3 mean operation RMS ratio to parent
不高于 0.90。collision-free 只设 70% 安全下限，因为本阶段明确尚未训练 overlap head。

结果为 formal FAIL：

| 指标 | 旧 A2 parent | C7-RS candidate | Gate/结论 |
|---|---:|---:|---|
| C3 operation RMS 改善 | — | 14/17 | 典型样本有正向信号 |
| C3 median operation ratio | — | 0.77909 | 中位改善 |
| C3 mean operation RMS | 0.06783 Å | 1.18233 Å | ratio 17.43，FAIL |
| C3 compatible | 12/17 | 13/17 | 要求 15/17，FAIL |
| C2 operation RMS 改善 | — | 0/17 | 全部轻重不等退化 |
| C2 compatible | 15/17 | 14/17 | 要求 16/17，FAIL |
| overall compatible | 27/34 | 27/34 | 79.41%，FAIL |
| collision-free | 27/34 | 29/34 | 安全下限通过 |
| bond quality | — | 27/34 | 79.41% <80%，FAIL |

C3 compatible 有 1 条修复、0 条回退，但没有达到绝对泛化门槛。失败由长尾主导：

- package `2461`：candidate operation RMS `18.84380 Å`、bond MAE `27.80284 Å`，相对 parent
  operation ratio `57.17`；
- package `2465`：operation ratio `6.28`，最短原子间距 `0.29287 Å`；
- package `1296`：operation ratio `2.11`，最短原子间距 `0.29093 Å`；
- package `114` 是 C2 回退：parent compatible，candidate analyzer/compatible 失败。

因此准确结论是：1:1 重采样改善了多数 C3 的典型连续响应，但 graph-shared scalar gate 的
fixed-time endpoint 训练没有约束完整 rollout 的最坏情况，并引入 C2 权衡。不能选择性删除
outlier、降低门槛或把 endpoint PASS 当作 raw PASS。正式状态为
`FAIL_ETFLOW_C7RS_RAW_C3_CONFIRMATION_STOP_RESAMPLING_ONLY`，不重训 overlap head。

下一分支应同时引入：per-operation 与 orbit pooling/broadcast 条件、non-identity/
worst-operation/orbit-balanced symmetry loss、multi-time 或 short-rollout loss，以及显式有界的
correction magnitude。C2/C3=1:1 继续保留为支持措施，但不再被视为充分修复。

可重复身份：

- raw protocol：`cfbb99164725d7d56ba8f43e5e441926ed103391ed6e6d05cab7ff5c7c2a21d9`
- geometry report：`06e9a55c06094c27bc064ecf537a7b4311978708e8936ef4fcfb408b0e6f4634`
- coordinates：`3f5a1050cc309346867ea49fa05b875cfeb0ee2c56a668572ec585d39d891fcf`
- point-group report：`0a3717e7a58f619d994414d15331f70d403b1cebc4f4b7092d572526472247f9`
- failure diagnosis：`06452ade66fa53eafe767ba785b276785e2cc92ee53c1484c87566d96fd1480a`

### 15.3 v2 训练结果与报告逻辑审计

v2 已完成 256 steps 和全部四个 checkpoint。原 `report.json` 显示 7/7 final Gate 为 true，
其余正向 structural checks 也为 true，但最终误报
`FAIL_ETFLOW_EF2C2_TRAINING_STOP_BRANCH`。原因是代码把两个负语义 provenance 字段直接放入
`all(structural_checks.values())`：

```json
{
  "iid_test_used": false,
  "v1_failed_checkpoint_reused": false
}
```

这两个 `false` 实际都表示合规。原报告和 checkpoint 均未覆盖；冻结只读审计将其规范化为
`iid_test_not_used=true` 和 `v1_failed_checkpoint_not_reused=true`，同时验证 256 个 finite
loss steps、64/128/192/256 checkpoint、step-256/last conditioner state 相同以及 A2 parent
fingerprint 不变。修正后的正式状态为：

`PASS_ETFLOW_EF2C2_TRAINING_ADVANCE_TO_RAW_ODE_GATE_AFTER_REPORT_LOGIC_AUDIT`。

| fixed-endpoint validation 指标 | A2 parent | EF2-C2 | 解释 |
|---|---:|---:|---|
| all-pair collision-free | 50/64 | 57/64 | 增加 7 条，但仍需 full ODE Gate |
| mean overlap loss | 0.036892 | 0.007298 | ratio `0.197811` |
| flow MSE ratio to A2 | — | 0.988969 | 未退化 |
| symmetry MSE ratio to A2 | — | 1.000000 | 基本不变 |
| active-case positive gate | — | 15/15 | collision head 响应方向正确 |

- 原 report SHA-256：`415f8354d3a34f0e4a71f7fe0596d0eac859152f1394feed49eb90c69c80a1c4`
- result-audit protocol SHA-256：`7bb79ba9a71da0717cf9b9b86263fea6422ffe2e2c4860070263a5d3a0bce2f0`
- corrected result audit SHA-256：`4809f3a8209be25b4415544221a56407c121dd3c8a96061060d72598ad5053c9`

该 PASS 只放行 paired A2-versus-EF2-C2 raw ODE validation；尚未放行 actual-PG 或 IID-test。

### 15.4 EF2-C3 full raw ODE Gate

EF2-C3 已冻结，复用 EF2-B 完全相同的 16 条 IID-validation、每分子 2 个 seed 和 50-step
Euler ODE。每个 candidate 从完全相同的 prior 分别运行 frozen A2 parent 与 trained EF2-C2，
并核对 A2 分支与 EF2-B 归档坐标；graph action 仍由 graph-only 接口恢复。Gate 要求：

- candidate all-pair collision-free fraction `≥0.90`，且相对 A2 至少增加 `0.05`；
- mean overlap loss ratio `≤0.50`；
- operation-RMS ratio to A2 `≤1.25`；
- bond-quality fraction `≥0.90`；
- 无 stored action、hard projection、force field 或 reference-coordinate sampling。

protocol SHA-256：`7fefb7d59ddc28dbd22302d9f8c8224b7a6be696eadeed75cd42238a0752180d`。

```bash
cd /home/tianyajun/MARL_for_COFs
conda activate env_etflow
set -euo pipefail

GPU_ID=4
OUT=generative_model/runs/etflow_ef2c3_raw_sampling_v1
mkdir -p "$OUT"

test "$(sha256sum generative_model/smoke/reports/etflow_ef2c3_raw_sampling_protocol_v1.json | awk '{print $1}')" = \
  "7fefb7d59ddc28dbd22302d9f8c8224b7a6be696eadeed75cd42238a0752180d"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -m generative_model.smoke.run_etflow_ef2c3_raw_sampling \
  --device cuda \
  --cache generative_model/checkpoints/etflow \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

sha256sum "$OUT"/{geometry_report.json,coordinates.npz,console.log}
```

只有 geometry PASS 后才运行独立 point-group audit；失败则停止 overlap branch，不切 checkpoint
或修改阈值。

### 15.5 EF2-C3 运行结果

EF2-C3 的核心几何指标全部达到预冻结门槛：

| 指标 | A2 parent | EF2-C2 candidate | Gate |
|---|---:|---:|---:|
| all-pair collision-free | 26/32 | 30/32 | candidate ≥90%，gain ≥0.05：PASS |
| mean overlap loss | 0.011656 | 0.000787 | ratio `0.067558`：PASS |
| mean operation RMS | 0.034248 Å | 0.033132 Å | ratio `0.967415`：PASS |
| bond-quality | — | 32/32 | PASS |

剩余 2 个失败均为 canonical short bond，而不是非键碰撞：package `2265` seed
`2026185165` 为 `0.50955 Å`，package `1522` seed `2026084422` 为 `0.57431 Å`。

但 formal status 仍为 `FAIL_ETFLOW_EF2C3_RAW_GEOMETRY_STOP_OVERLAP_BRANCH`，唯一失败项是
`parent_reproduction`：重算 A2 与 EF2-B 归档坐标的最大逐元素差为 `0.008783 Å`，超过冻结的
`1e-4 Å`。结构不变量差异远小于该逐坐标值：最大 Kabsch RMSD `0.005181 Å`、最大
pair-distance MAE `0.001923 Å`，且 A2 collision-free 仍为相同的 26/32。这与 50-step CUDA
scatter/ODE 数值差异一致，但不同于前一阶段的 boolean polarity bug；不能事后降低阈值并把
本 Gate 改写为 PASS。

因此当前准确结论是：trained overlap head 在同一次 paired run 中把 collision-free 从 81.25%
提高到 93.75%，overlap loss 降低 93.24%，且没有损害 bond/symmetry；但 control 的归档坐标
逐元素复现检查未过。按冻结边界暂不运行 actual-PG/IID-test，也不重训/换 checkpoint。下一步
若继续，应先冻结一次只诊断 CUDA ODE 数值复现性的重复实验，再决定能否建立新的 independent
PG Gate；该诊断不能覆盖 EF2-C3 原 FAIL。

EF2-C3 artifacts：

- geometry report SHA-256：`362173ccf1384747d585a5b5831aef9b875c57cdc2ad64a47feeb8816c4c7a40`
- coordinates SHA-256：`2adbb61332c031a5a1623a2edd6870adf73e1296447224d9cba2471ea6a7c0c9`
- console SHA-256：`fa58b928c3c2cdbbf8cca430716480f84bac6bac9cc520ff1d1bf2a8105fb7cc`

### 15.6 EF2-C4 数值复现诊断（已冻结，待 repeat-1）

EF2-C4 不重训、不换 checkpoint，也不把 EF2-C3 的 `parent_reproduction ≤1e-4 Å` 事后改成
较宽门槛。它使用完全相同的 runner、16×2 panel、seed、shared prior、A2/EF2-C2 checkpoint
和 50-step Euler ODE，在新目录生成 repeat-1，再只读比较 repeat-0/repeat-1。

诊断同时约束两类证据：

1. 每次运行自身的科学指标必须继续满足 collision、gain、overlap、symmetry 和 bond-quality
   原有量级；
2. 两次坐标以 Kabsch RMSD 与 pair-distance MAE 比较，避免只看易受整体刚体运动影响的逐元素
   坐标差；collision-free 比例和失败样本集合也必须稳定。

预冻结阈值为：parent/candidate 跨运行最大 Kabsch RMSD `≤0.02 Å`、最大 pair-distance MAE
`≤0.005 Å`；candidate collision-free 两次均 `≥0.90` 且相差不超过 `1/32`，失败集合对称差
不超过 2 条。协议 SHA-256 为
`bf56f5c38eb6f8ee095c01912b5d83ea49252bd7df24ef460848b398c580d377`。

若通过，状态只记为
`PASS_ETFLOW_EF2C4_NUMERICAL_REPLICATION_ADVANCE_TO_DUAL_POINT_GROUP_AUDIT`，含义是允许对两份
candidate 坐标做 independent actual-PG/compatible 审计；EF2-C3 原 formal FAIL 继续保留。
若失败，则停止 learned overlap branch，不在同一 panel 上调阈值。

### 15.7 EF2-C4 结果

EF2-C4 正式通过。candidate 两次生成的最大 Kabsch RMSD 仅 `9.32884e-5 Å`，最大
pair-distance MAE `2.17401e-5 Å`；collision-free 均为 30/32，两个失败样本及 seed 完全一致。
A2 parent 的跨运行最大 Kabsch RMSD 为 `0.00480376 Å`、pair-distance MAE 为
`0.00179267 Å`，与 EF2-C3 对旧归档的漂移量一致。这确认 candidate 改善具有可重复性，同时
说明 EF2-C3 的 sole reproduction failure 来自 CUDA ODE parent 数值漂移，而不是 learned
overlap head 随机失稳。

- result SHA-256：`a814d054de683a86c529a065b87564b34dfde100ed35683ff08517070a6e40dd`
- repeat-1 geometry SHA-256：`34950018e7e92ce72b0fece8ab9eb699c60df3e7efdee07b608b5213f77768fa`
- repeat-1 coordinates SHA-256：`a961d9ef491f8ae83f4e82607321b8b3edf0dbf1ed3b0fd10ca6a24180849a0b`

### 15.8 EF2-C5 双归档 point-group audit

EF2-C5 在 `env_cof` 中使用 v2 manifest 原始 pymatgen 协议，同时审计 repeat-0/repeat-1 的
A2 parent 与 EF2-C2 candidate。v1 首次执行完成 32 条 analyzer 后，汇总代码误从
`gate_thresholds.candidate_count_each` 读取不存在的键而抛出 `KeyError`，没有写结果；v2 仅把
期望数读取位置改为已冻结的 `panel.candidate_count_each_archive`，点群参数和 Gate 均未改变。

结果如下：

| 指标 | A2 parent | EF2-C2 candidate |
|---|---:|---:|
| analyzer success / compatible（两次相同） | 28/32 | 31/32 |
| exact match（两次相同） | 22/32 | 26/32 |
| C2 compatible | — | 23/24 |
| C3 compatible | — | 8/8 |
| candidate actual-PG 跨运行一致 | — | 31/32 |
| compatible label 跨运行一致 | — | 32/32 |

candidate 修复了 parent 的 3 条 PG failure：package/seed `1611/2026184511`、
`1522/2026084422`、`1552/2026084452`；没有造成任何 parent→candidate regression。唯一仍失败的
candidate 是 `296/2026083196`。因此 candidate 自身的全部预冻结 PG checks 均通过，compatible
相对 parent 提高 `0.09375`，exact 提高 `0.125`。

但是 EF2-C5 formal status 必须保留为
`FAIL_ETFLOW_EF2C5_DUAL_POINT_GROUP_AUDIT_STOP_LEARNED_OVERLAP_BRANCH`，因为协议还预先要求
每个 parent branch analyzer success `≥0.90`，而实际为 `0.875`。这不是 boolean polarity 或
代码计算错误，不能用 candidate 的正向结果事后覆盖。准确表述是“candidate 的小面板 PG 证据
通过，但整个预注册 Gate 因 control-only 条件失败”。

- v2 protocol SHA-256：`171ac076a6091956652834bab6c44cb0645fc0306c24ff1b4d4447a4b71a54ba`
- v2 result SHA-256：`ba9e9c90999239d78607cc215d0f41734d2974ed5e51cbee18cbde135c735b78`
- failure diagnosis SHA-256：`8ebfb7b8b05973d3d6debfed21cb93ddee39ca328de12729ca4e14ab6a51bb1d`

下一步不能在同一 32-case panel 上删除 parent 门槛并重判。合规选择是停止该分支；若明确决定
继续，只能新建一个与当前 16 分子完全不重叠的 IID-validation confirmatory panel，在看到新
坐标前预注册 candidate-only PG/geometry 门槛，并把 parent 保留为 diagnostic baseline。

### 15.9 EF2-C6 不重叠 IID-validation 独立确认

用户明确授权后，C6 在产生新坐标前从剩余 232 个未使用 IID-validation C2/C3 分子中，以
`SHA256("20260826:package_index")` 分层固定选择 48 个 C2 + 16 个 C3。该 64 分子与
EF2-B/C3/C4/C5 panel 零重叠；每分子使用一个新 seed 和 50-step raw Euler ODE。A2 parent
仍从相同 prior 生成，但只作 diagnostic；所有正式门槛只评价 frozen EF2-C2 candidate。
该 validation 子集中没有 Sn，因此不作 Sn 泛化结论。

预冻结 candidate Gate：collision-free、bond-quality、analyzer/compatible 均 `≥0.90`；
C2 compatible `≥0.90`、C3 compatible `≥0.875`，joint geometry+PG `≥0.85`；mean operation
RMS `≤0.10 Å`、mean overlap loss `≤0.01`。协议 SHA-256：
`01fb5c7cf8150514e852f1da61a0c0cbafaede0b1771cee03ca03eebc255a619`。

几何 Gate 正式 PASS：

| 指标 | A2 diagnostic | EF2-C2 candidate | Candidate Gate |
|---|---:|---:|---:|
| collision-free | 61/64 | 62/64 | ≥57.6：PASS |
| bond-quality | — | 62/64 | ≥57.6：PASS |
| mean operation RMS | 0.030412 Å | 0.032493 Å | ≤0.10 Å：PASS |
| mean overlap loss | 0.005265 | 0.001005 | ≤0.01：PASS |

独立 pymatgen point-group audit 的总体、C2 和 joint 指标也通过，但 C3 未通过：

| 指标 | 结果 | Gate |
|---|---:|---:|
| analyzer / compatible | 59/64 = 92.19% | ≥90%：PASS |
| exact match（描述性） | 43/64 = 67.19% | 不作否决 |
| C2 compatible | 47/48 = 97.92% | ≥90%：PASS |
| C3 compatible | 12/16 = 75.00% | ≥87.5%：FAIL |
| joint geometry + compatible PG | 58/64 = 90.63% | ≥85%：PASS |

因此唯一失败 check 是 `candidate_c3_compatible`，formal status 为
`FAIL_ETFLOW_EF2C6_DISJOINT_CONFIRMATION_STOP_LEARNED_OVERLAP_BRANCH`。这次不能归因于 control
或 Gate polarity：它是在全新分子上真实出现的 C3 泛化不足。candidate 相对 parent 仍修复 2 条
PG failure 且零回退，但不足以满足预注册 C3 门槛。

6 个 joint failure 中，4 个是 C3 PG failure（package `2429/81/1377/1407`），1 个是 C2 PG
failure（`418`），1 个是仅有 short-bond collision、但 C2 compatible 的 `1790`。其中 `81`
同时是严重 C3 几何异常（最短距离 `0.43974 Å`、bond MAE `1.65781 Å`）；`2429` 的 bond MAE
为 `0.27241 Å`。其余 3 个 PG failure 的 collision/bond geometry 达标。

- geometry report SHA-256：`9f7cb8cbf161d93063641759dc6141a63966855af9724e35f82c3b39fe996c11`
- coordinates SHA-256：`d817be31bc5dfbec884cb96fa6a8b4b97a049ccf92c658c12742b8774129d0ca`
- point-group report SHA-256：`47a1a28a7f7afe9daac522ee1122cd1bcbcb52b75c15af67b95aca0349ffcc66`
- failure diagnosis SHA-256：`0fdd299133cd59d58889c27470f9e0cfbf4e7155b832b1fcac394c17dfa4d122`

按冻结决策，当前 checkpoint 不进入 IID-test，也不允许通过降低 C3 门槛或继续训练步数补救。
learned overlap checkpoint 到此收尾。若未来继续 learned 路线，必须作为新的模型改动分支解决
C3 conditioning/generalization，而不是在 C6 panel 上调参；ETKDG+E3/F0.2 继续作为可靠的
symmetry-by-construction baseline。

A2 若无任一 checkpoint 过 Gate，就停止当前 scalar residual conditioner；不再延长、解冻 base、
改阈值或访问 IID-test。实际运行中四个 checkpoint 全部通过，结果见下一节。

该修复若仍失败，则停止当前 scalar residual conditioner，不继续用训练轮数掩盖架构不足；保留
ET-Flow 作为 learned initial-conformer prior，并回到 E3 action/projection 后端或设计更强的
PG-conditioned equivariant adapter。

## 11. 尚未完成

- EF2-A2 engineering 与 256-record 训练 Gate 已通过；
- raw ODE sampler 与 correct/alternate/action-ablation 已实现并完成，但 geometry Gate 未通过；
- 尚未在 IID-test 或 Core-OOD 上评价；
- 尚未证明 raw actual-PG compatible/exact 提升。

因此目前可以说“PG-conditioned ET-Flow 的 A2 conditioner 已通过 fixed-time raw-endpoint
validation，获得完整 raw ODE sampling 准入”，不能说“actual-PG 条件生成已经证明”。

## 12. A2 可重复身份

- A2 engineering protocol：`ccabc98ff9cb342ccccab0bfee31f849a8042aae192af9c8a43e778b4dff4570`
- A2 engineering PASS：`6c6fadf210177af5f363376b3d1fb9357e1c3eb2437df32526a12267984b30b0`
- A2 training protocol：`eaf696764009ee46aa827dbd03a061f930c4d1c0825b73d79fc9d2a235ceb256`
- A2 final report：`e293895f5047239b5661e3ac554601acf6f0a343ed271a00b5eeef9bf9f5d2e2`
- A2 losses：`8f8c1ee12617f811979f379003b137f35ffdac6bd623da9d96f7cb56a48d4228`
- selected step-256 checkpoint：`bee44230ab167def820d6147fad4140487de923c57a7eddf7cafc7703ee9f335`

## 13. A2 训练结果

正式状态：`PASS_ETFLOW_EF2A2_ADVANCE_TO_RAW_SAMPLING_GATE`。训练快速完成是预期行为：
official base 全冻结，只更新小型 conditioner，且 256 条分子各访问一次。

| checkpoint | endpoint MSE ratio | improved | C2 improved | C3 improved | flow MSE ratio | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 0.667973 | 64/64 | 48/48 | 16/16 | 0.899671 | PASS |
| 128 | 0.073569 | 64/64 | 48/48 | 16/16 | 0.719909 | PASS |
| 192 | 0.014558 | 64/64 | 48/48 | 16/16 | 0.701879 | PASS |
| 256 | 0.005408 | 64/64 | 48/48 | 16/16 | 0.699048 | PASS |

四个 checkpoint 均过冻结 Gate。按预注册的“eligible 中 endpoint ratio 最低、再取较早 step”规则，
选择 step 256。其逐分子 endpoint ratio 均值为 C2 `0.002945`、C3 `0.012847`；两类改善率
均为 100%。official base state 前后相同，训练记录 256/256 唯一、Sn 2/2、交错顺序、A1
per-sample noise seed 映射和 no-hard-projection 全部通过。

这说明 A1 的主要失败确实来自 restoring direction/训练组织，而不是 ET-Flow base 无法承载
条件 adapter。不过该结果仍只评价固定 `t=0.5`、相同 `x0` 的 estimated endpoint，并在
training/validation 使用已知 target action。它没有评价：

- 从 prior 出发的完整 ODE trajectory 与 raw final coordinates；
- inference 时的 graph-only recovered action；
- pymatgen actual-PG compatible/exact；
- correct Target_PG 相对 swapped/action-ablated 条件是否显著更优；
- collision、bond geometry 和构象多样性；
- IID-test 或 Core-OOD 泛化。

所以下一步必须冻结 raw sampling Gate，而不是直接宣布模型已实现指定点群生成。该 Gate 应锁定
step-256 checkpoint，使用同 seed 的 official-base/conditioned 成对 ODE 采样，全程不执行 E3
hard projection；先在 IID-validation 上评价 raw symmetry、actual PG、化学几何以及
correct/swapped/action-ablated 响应，通过后才允许一次 IID-test。

## 14. EF2-B raw ODE 结果

EF2-B 锁定 selected step-256，在 16 条 IID-validation（12 C2/4 C3）上使用 2 个共同 prior
seed。每个 candidate 都成对运行 official-base/action-ablation 与 correct graph-recovered
action；8 个 graph 还运行另一合法点群 action，共得到 32 对 base/correct 和 16 条 alternate
raw 50-step Euler ODE。reference XYZ 只用于事后 bond/构象指标，未参与采样；无 stored action
或 hard projection。

| 指标 | 结果 | Gate | 判定 |
|---|---:|---:|---|
| correct operation-RMS ratio | 0.01421 | ≤0.25 | PASS |
| correct symmetry improved | 32/32 | ≥90% | PASS |
| alternate operation-RMS ratio | 0.01687 | ≤0.50 | PASS |
| alternate symmetry improved | 16/16 | ≥75% | PASS |
| correct/alternate median Kabsch | 2.3777 Å | ≥0.05 Å | PASS |
| raw bond-quality | 32/32 | ≥90% | PASS |
| raw collision-free | 26/32 | ≥90% | **FAIL** |

正式状态：`FAIL_ETFLOW_EF2B_RAW_SAMPLING_STOP_BRANCH`。对称响应与条件敏感性均非常强，
但 raw 收缩导致 6/32 candidate 的最短非键距离低于 `0.6 Å`，最差 `0.22399 Å`。失败集中于
4 个 C2 graph（package `1611/2265/1522/2359`），不是随机单一 seed；同时其 bond MAE 仍全部
低于 `0.15 Å`，说明主要是非键碰撞而非键图整体崩坏。

按预冻结决策，geometry FAIL 后不运行 independent actual-PG Gate，不读取 IID-test，不降低
collision 阈值，也不事后挑选较早 checkpoint/guidance。当前 scalar endpoint conditioner
到此停止。它证明“learned raw trajectory 可以强烈响应 graph-recovered PG action”，但没有达到
“化学几何可接受的 raw 指定点群生成”。后续若继续 learned 路线，必须新建带 nonbonded
repulsion/collision-aware objective 的架构分支；若优先获得可用结构，则继续使用已经通过的
E3 action + F0.2 hard-symmetry 后端。

EF2-B identities：

- protocol：`6572184cb1ed0192611eceee1a421eafd91d096e0f1c1da956fc4a00c9d33b2f`
- geometry report：`50b02e46636883d19e2d3a6be3cc9e769b9b2c6310da80fb0b804ae305403ded`
- coordinates：`e579589c8f6669f7e466683c65eb0174c4af8d6d753555e2b36ffe059425b94b`

## 15. EF2-C/C2：碰撞分型与新 overlap-aware 分支

EF2-B 原 Gate 使用所有原子对的最短距离，因此“6 个 collision”不能直接等同为 6 个非键
碰撞。冻结坐标的只读分型结果是：

| 失败类型 | 数量 | 说明 |
|---|---:|---|
| nonbonded collision | 3 | package 1611 两个 seed、package 2359 一个 seed |
| canonical bond < 0.6 Å | 3 | package 2265 两个 seed、package 1522 一个 seed |

因此先建立的 nonbonded-only EF2-C v1 虽然让单步后的 32/32 非键距离和 bond-quality 达标，
但只对 3/6 原失败案例产生严格改善，按冻结 Gate 状态为
`FAIL_ETFLOW_EF2C_DIRECTION_REDESIGN_BEFORE_TRAINING`。该失败版本和报告保留，不启动训练。

重新设计的 EF2-C2 严格区分两个可微场：

\[
F_{nb}(X)=\sum_{i<j,(i,j)\notin E}
\max(0,0.8-d_{ij})\frac{x_i-x_j}{d_{ij}},
\]

\[
F_{bond}(X)=\sum_{i<j,(i,j)\in E}
\max(0,0.6-d_{ij})\frac{x_i-x_j}{d_{ij}}.
\]

两者只使用 immutable canonical bond graph；不推断新键、不做元素 fallback。合并场在进入
velocity head 前经过目标群的 Reynolds average：

\[
F_G=P_G(F_{nb}+F_{bond}),
\qquad
v_{C2}=v_{A2}+b_\psi\frac{F_G}{\max(1-t,0.05)}.
\]

`b_ψ` 是新的零初始化 graph-shared gate。official ET-Flow 与 selected A2 step-256 全冻结；
forward 不执行 hard projection、UFF 或其他生成后修复。

EF2-C2 的冻结只读方向 Gate 已通过：

- 6/6 原 all-pair 失败的最短距离严格改善；
- 32/32 单步结果的所有原子对距离均 `≥0.6 Å`，最小值 `0.641332 Å`；
- 32/32 保持 `bond MAE ≤0.15 Å`；
- symmetry error 最大增加仅 `6.94e-18 Å`；
- 未使用 hard projection 或 force field。

正式状态为 `PASS_ETFLOW_EF2C2_DIRECTION_FREEZE_TRAINING_BRANCH`。它证明新方向在冻结 raw
端点上几何正确，但不等于 trained full-ODE 已通过。

### 15.1 已冻结的 256-step 训练

新训练只更新 overlap head：

\[
L=L_{flow}+L_{sym}+4L_{nonbond}+4L_{short-bond}+0.25L_{direction}.
\]

复用 A2 的 256 条 `C2,C2,C2,C3` IID-train 顺序和每分子 noise assignment，每 64 step
保存诊断 checkpoint，但固定选择 final step-256，避免根据 validation 事后挑权重。64 条
IID-validation 仅允许检查：flow/symmetry 不明显退化、overlap loss 不升高、collision-free
fraction 不低于 A2，以及 active case gate 为正。通过后只允许一次冻结的 A2-versus-C2 raw
ODE Gate；IID-test/Core-OOD 仍封存。

训练协议 SHA-256：
`d51bc67de5fdfeff99a024c6aee9d68147a6fa9bc38436e1caaaca54f59919a9`。

```bash
cd /home/tianyajun/MARL_for_COFs
conda activate env_etflow
set -euo pipefail

GPU_ID=4
OUT=generative_model/runs/etflow_ef2c2_overlap_0256
mkdir -p "$OUT"

test "$(sha256sum generative_model/smoke/reports/etflow_ef2c2_training_protocol_v1.json | awk '{print $1}')" = \
  "d51bc67de5fdfeff99a024c6aee9d68147a6fa9bc38436e1caaaca54f59919a9"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -m generative_model.smoke.train_etflow_ef2c2_overlap_adapter \
  --device cuda \
  --cache generative_model/checkpoints/etflow \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

sha256sum "$OUT"/{report.json,losses.npz,last.pt} "$OUT"/checkpoints/*.pt
```

截至首次启动前，结论是“collision/short-bond 方向已通过，EF2-C2 bounded training 已准入”，
不能提前写成“raw collision 问题已由训练模型解决”。首次启动的执行情况见下一节。

### 15.2 首次启动的工程失败与 v2 重启

首次 v1 启动在完成前 64 个训练 step、准备保存首个 checkpoint 时退出。原因是训练脚本把
输出目录变量 `output` 重新赋值为模型返回的 `dict`，随后执行路径拼接触发：

```text
TypeError: unsupported operand type(s) for /: 'dict' and 'str'
```

这是纯 Python 局部变量遮蔽，不是 CUDA/OOM、数据或 loss 失败。退出前没有写出 checkpoint，
所以不能恢复；v1 console 和协议原样保留。v2 仅改用 `model_output` 与 `output_dir` 两个不同
变量，科学协议、样本、seed、loss、步数和 Gate 完全不变，并从 step 1 在新目录确定性重启。

- v1 failure console SHA-256：`ecfa4558f76b0400bd2fdda9b6c4f3de3f2cff76616a97478652b6302675c718`
- v2 protocol SHA-256：`075352be70bc4209e9fb19f6aaa95eeea2330ddfd94587b00b41997d12834805`

当前有效命令：

```bash
cd /home/tianyajun/MARL_for_COFs
conda activate env_etflow
set -euo pipefail

GPU_ID=4
OUT=generative_model/runs/etflow_ef2c2_overlap_v2_0256
mkdir -p "$OUT"

test "$(sha256sum generative_model/smoke/reports/etflow_ef2c2_training_protocol_v2.json | awk '{print $1}')" = \
  "075352be70bc4209e9fb19f6aaa95eeea2330ddfd94587b00b41997d12834805"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -m generative_model.smoke.train_etflow_ef2c2_overlap_adapter_v2 \
  --device cuda \
  --cache generative_model/checkpoints/etflow \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

sha256sum "$OUT"/{report.json,losses.npz,last.pt} "$OUT"/checkpoints/*.pt
```
