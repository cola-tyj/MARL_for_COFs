# our_ET_Flow：模型、训练、对称投影与代码维护说明

## 0. 先看这一页：到底训练了什么

### 0.1 一句话结论

当前可交付的 `our_ET_Flow v5` **不是用 2,532 条 COF 数据重新训练出的完整 ET-Flow**。
它只加载官方在 GEOM-DRUGS 上预训练的 `drugs-o3.ckpt`，再使用本项目的确定性
graph action、hard projection 和 F0.2 几何优化得到目标点群坐标。

项目确实训练过若干很小的 Target-PG adapter，但这些 adapter 都没有同时通过 raw ODE、碰撞和
泛化准入，因此 **v5 不加载 `generative_model/runs/` 中的任何训练权重**。

### 0.2 训练数据、batch 和 step 总表

下表中的“batch”指一次 optimizer update 使用的分子数；“step”指 optimizer update 次数；
“分子暴露数”是 `batch size × step`。同一分子是否重复由各冻结协议单独规定。

| 层次/实验 | 实际训练数据 | batch size | optimizer steps / batches | 分子暴露数 | 结果 | 最终 v5 使用 |
|---|---|---:|---:|---:|---|:---:|
| 官方 `drugs-o3` backbone | GEOM-DRUGS；由官方训练，本项目未重新下载完整 train split | 64/GPU | 250 epochs × 5,000 batches/epoch；8×A100（论文设置） | 本地无法从 checkpoint 独立还原唯一分子数 | 官方预训练权重 | 是，冻结 |
| EF2-A v1 | COF IID-train，192 C2 + 64 C3，一次遍历 | 1 | 256 | 256 | endpoint/raw 准入失败 | 否 |
| EF2-A2 | 同一 256 条，改为 `C2,C2,C2,C3` 交错 | 1 | 256 | 256 | endpoint 通过，raw ODE 碰撞准入失败 | 否 |
| EF2-C2 overlap head | 与 A2 相同的 256 条；A2 与官方 backbone 冻结 | 1 | 256 | 256 | overlap 改善，但独立 raw/PG 准入失败 | 否 |
| C7RS balanced A2 | COF IID-train，128 C2 + 128 C3，一次遍历 | 1 | 256 | 256 | endpoint 通过，C3 raw trajectory tail 失败 | 否 |
| EF2-D2 operation/orbit | COF IID-train，256 C2 + 256 C3，全部唯一；每 batch 各 1 条 | 2 | 256 | 512 | operation/orbit gate 失败 | 否 |
| EF2-D4 rollout/CVaR | 与 D2 相同的 512 条唯一记录；每 batch 为 4 C2 + 4 C3 | 8 | 64 | 512 | trajectory-tail gate 失败 | 否 |
| 最终 v5 | **无 COF optimizer training** | — | **0** | **0** | 固定 116 条审计全部 compatible | — |

这里的 256/512 条不是“从 2,532 条中训练多个 epoch”，而是用于低成本准入诊断的固定小面板，
每条记录只暴露一次。它们只能证明某个 adapter 设计在短程优化中有没有正确方向，不能当作完整微调、
收敛训练或最终模型质量实验。只有某个设计通过这些 Gate，才应扩大到完整 IID-train、多 epoch 和
Core-OOD；当前没有 adapter 到达这一步。

因此，如果问题是“我们自己的最终模型训练了多少批次”，严格答案是：

- 官方 backbone：沿用作者 checkpoint，本项目训练批次为 0；
- 自研 adapter 诊断：各分支只训练 64 或 256 个 optimizer batches，均未进入最终模型；
- 最终 v5：不进行 adapter 优化，直接执行冻结 backbone + 确定性约束后端。

官方论文中的 `5,000 batches/epoch/GPU` 是分布式训练口径。可以据此得到 250 个 epoch 共
1,250,000 个同步 update iterations；若每张 GPU 的 batch 都为 64，则一次同步 update 的全局
batch 为 512。这里不把由此计算的“样本呈现次数”误写成“唯一训练分子数”。

### 0.3 最终到底加载哪个 checkpoint

最终只加载：

```text
generative_model/checkpoints/etflow/drugs-o3.ckpt
SHA-256: a24ae9a1fed2708696929308ed1dc10ab167fd66a2d51c44a4afb6c11badccb2
```

不会加载：

```text
generative_model/runs/etflow_*/last.pt
generative_model/runs/etflow_*/checkpoints/*.pt
```

这些 `runs` 权重只用于回答“某种 adapter 设计是否值得继续”，不是发布模型组成部分。

### 0.4 训练、验证和最终交付不要混为一谈

| 动作 | 是否更新参数 | 使用什么数据 | 目的 |
|---|:---:|---|---|
| 官方 ET-Flow 预训练 | 是，官方完成 | GEOM-DRUGS | 学习一般 2D graph → raw 3D conformer prior |
| 本项目 adapter 小试 | 是 | IID-train 的 256/512 条小面板 | 诊断 Target-PG 是否能由 learned residual 响应 |
| checkpoint selection / raw gate | 否 | IID-validation 等固定面板 | 判断 adapter 能否进入下一阶段 |
| v5 独立审计 | 否 | C2 48 + C3 16 + S4 38 + D6h 14，共 116 条 | 验证最终确定性流水线的 geometry/PG compatibility |

下面章节保存实现细节和失败分支的负证据；日常了解项目进度只需先读本节、§1、§2 和最终结果表。

`runs` 的保留/删除标准见
[`generative_model/RUNS_RETENTION.md`](../generative_model/RUNS_RETENTION.md)。

## 1. 文档定位与结论

本文是当前 `our_ET_Flow` 路线的集中技术说明，回答以下问题：

1. 官方 ET-Flow backbone 原本是什么、如何训练和采样；
2. 本项目在 backbone 外训练过哪些 Target-PG adapter；
3. 哪些训练分支通过或失败，哪些权重实际进入最终推理；
4. `hard projection`、graph action、orbit-constrained F0.2 分别做什么；
5. 输入、输出、Gate、限制和代码文件在哪里。

当前可交付的 `our_ET_Flow` 定义为：

```text
known canonical graph + requested Target_PG
  -> frozen official ET-Flow drugs-o3: 4 raw conformers
  -> graph-only symmetry action recovery
  -> PCA/orientation search + Reynolds hard projection
  -> orbit-constrained UFF + short-range repulsion (F0.2)
  -> final.xyz + coordinates.npz + audit JSON
```

支持 `C2 / C3 / S4 / D6h`。完成审计的固定面板共 116 条：C2 48、C3 16、S4 38、D6h 14；
每一类的 geometry success 和独立 pymatgen `PG_Compatible` 均为 100%。

最重要的科学边界是：

> 最终交付路线没有使用训练后的 PG adapter。ET-Flow 只提供 learned initial conformer；目标
> 对称性由 graph-only action、Reynolds hard projection 和 orbit-constrained F0.2 保证。

因此可以声明“known graph + requested PG → compatible XYZ”，不能声明“官方 ET-Flow backbone
已经通过微调学会不依赖 hard projection 的 Target-PG 条件生成”。

---

## 2. 三个名称必须分开

| 名称 | 含义 | 是否进入最终 v5 推理 |
|---|---|:---:|
| 官方 ET-Flow | `etflow==0.1.2` 的 `BaseFlow/TorchMDDynamics` 与官方 `drugs-o3.ckpt` | 是，冻结使用 |
| learned PG adapter 实验 | EF2-A2、C2/C6、D2/D4 等 residual adapter 训练分支 | 否，保留为实验记录 |
| `our_ET_Flow` 交付模型 | 官方冻结 backbone + 本项目 graph action + hard projection + F0.2 + audit | 是 |

这一划分也解释了为什么“做过模型训练”和“最终模型使用官方 checkpoint”可以同时成立：我们确实
训练并评测过 adapter，但它们没有通过 raw ODE/碰撞/泛化 Gate，因此生产推理没有加载这些
adapter checkpoint。

---

## 3. 官方 ET-Flow backbone

### 3.1 任务形式

官方 `drugs-o3` 是已知分子图条件下的三维构象生成模型，不生成原子或化学键。输入包括：

- 原子序数 `z`；
- 官方 10 维原子特征 `node_attr`；
- 已知化学键 `bond_index` 和边特征；
- 当前坐标 `x_t` 与时间 `t`。

网络输出坐标速度场：

```text
v_theta(x_t, t, graph) in R^(N x 3)
```

官方 `BaseFlow.forward()` 会先把坐标按分子质心归零，再将真实键与截断半径内的邻接边合并，送入
`TorchMDDynamics`。

### 3.2 当前锁定的官方配置

项目实际加载的 `drugs-o3` 配置如下：

| 配置 | 值 |
|---|---:|
| network | `TorchMDDynamics` |
| hidden channels | 160 |
| layers | 20 |
| radial basis | 64 个 trainable `expnorm` RBF |
| attention heads | 8 |
| cutoff | 10 Å |
| maximum atomic number | `Z < 100` |
| node/edge feature dims | 10 / 1 |
| activation | SiLU |
| prior | harmonic |
| interpolation | linear |
| path noise | `sigma=0.1` |
| optimizer metadata in installed checkpoint config | AdamW, base LR `8e-4`, weight decay `1e-8` |
| LR scheduler | cosine annealing warm restarts |
| parity handling | `post_hoc` |

官方配置中 `so3_equivariant=false` 与 `parity_switch=post_hoc` 是包内实际值；本文不把它重新解释为
本项目训练出的 O(3) 条件网络。

官方 checkpoint：

```text
generative_model/checkpoints/etflow/drugs-o3.ckpt
SHA-256: a24ae9a1fed2708696929308ed1dc10ab167fd66a2d51c44a4afb6c11badccb2
```

### 3.3 官方 checkpoint 的训练来源

根据 [ET-Flow 官方仓库](https://github.com/shenoynikhil/ETFlow) 和
[NeurIPS 2024 论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/e8bd617e7dd0394ceadf37b4a7773179-Paper-Conference.pdf)：

- `drugs-o3` 对应 GEOM-DRUGS；
- 论文版 GEOM-DRUGS 训练 250 epochs；
- batch size 64，每个 epoch 每张 GPU 5,000 个 training batches，使用 8 张 A100；
- 论文报告使用 Adam、cosine annealing，学习率从 `1e-3` 降至 `1e-7`，weight decay `1e-10`；
- 按最低 validation error 选择 checkpoint；
- 论文表格给出的主体规模约 8.3M parameters、20 layers、160 hidden channels、8 heads。

论文训练超参数与本项目安装的 `etflow==0.1.2` checkpoint loader metadata 并非每一项字面相同：
例如本地运行配置是 10 维 `node_attr`、AdamW metadata 和 `8e-4/1e-8`。当前复现实验以本地实际
加载配置和 checkpoint SHA 为准；论文数字用于说明官方预训练来源。本项目没有下载官方完整
GEOM-DRUGS train split 来重新执行 8-GPU/250-epoch 训练，也不声称复现了官方预训练过程。

### 3.4 官方 flow-matching 训练目标

令真实构象为 `x_1`，harmonic graph prior 样本为 `x_0`。线性路径均值为：

```math
\mu_t = t x_1 + (1-t)x_0.
```

官方实现还加入端点为零的路径噪声：

```math
x_t = \mu_t + \sigma\sqrt{t(1-t)}\,\epsilon,
\qquad \sigma=0.1.
```

监督速度 `u_t` 是上述随机路径对时间的导数。网络用 molecule-balanced L2 flow-matching loss：

```math
\mathcal L_{flow}=\mathbb E\|v_\theta(x_t,t,G)-u_t\|_2^2.
```

该训练由官方完成；本项目没有重新训练 20 层 backbone，也没有改变官方 checkpoint tensor。

### 3.5 当前采样方式

v5 每个输入固定：

- 生成 4 个 raw conformer；
- 每个使用 50 个 ODE Euler steps；
- seed 由协议或命令显式给出；
- 初始坐标来自 official harmonic prior；
- 输出重新排列回 canonical atom order。

离散 ODE 更新可写为：

```math
x_{k+1}=x_k+\Delta t\,v_\theta(x_k,t_k,G).
```

这些 raw coordinates 尚未接收 `Target_PG`，因此 raw ET-Flow 输出本身不保证目标对称性。

---

## 4. canonical graph 到官方 ET-Flow 的严格 bridge

bridge 负责接口适配，不修改模型结构。其关键约束为：

1. 从 canonical 原子、charge、radical、typed sparse bonds 构造 explicit-H mapped SMILES；
2. 重原子的 atom map 固定为 `canonical_index + 1`；
3. RDKit 重新生成的 H 只按 typed-graph 邻接恢复索引，不用坐标匹配；
4. 原子数、元素、形式电荷、自由基电子、键类型必须逐项往返一致；
5. Sn 保持 `Z=50`，不映射到 C 或 Si；
6. 任何超出 ET-Flow `Z<100`、图变异或未知特征 schema 都严格失败。

官方 10 维特征中落入 `misc` bucket 的数量会写入审计，不会被静默隐藏。

实现位置：

- `generative_model/conformer/etflow_bridge.py`
  - `build_mapped_explicit_h_smiles()`
  - `audit_model_molecule()`
  - `reorder_positions_to_canonical()`
  - `ETFlowRuntime.predict()`

---

## 5. 在官方 backbone 外做过的 learned 修改

### 5.1 共同设计原则

所有 EF2 adapter 都位于第三方 `etflow` 包之外：

- 官方 backbone 冻结；
- adapter 最后一层零初始化，初始输出严格等于官方模型；
- 只训练 residual conditioner；
- canonical graph 不变；
- forward 中不使用 hard projection；
- 训练 loss 可以读 COF v2 的 Target-PG operation/permutation/orbit supervision；
- checkpoint 记录官方 base state 的前后哈希，验证 base 未改变。

### 5.2 EF2-A2：endpoint PG residual adapter

原始 backbone 给出 `v_base`，先估计线性路径终点：

```math
\hat x_1=x_t+(1-t)v_{base}.
```

对 `\hat x_1` 计算 soft Reynolds target `\Pi_G(\hat x_1)`，得到恢复方向：

```math
d_G=\Pi_G(\hat x_1)-\mathrm{center}(\hat x_1).
```

adapter 输入包括：

- `Target_PG` embedding；
- 时间及 `sin/cos` 时间特征；
- operation 数量；
- fixed-orbit 比例与平均 orbit size；
- symmetry residual 的 RMS/max norm。

MLP 输出每个分子共享的标量 gate，再加入速度：

```math
v=v_{base}+\tanh(a_\phi)\frac{d_G}{\max(1-t,0.05)}.
```

训练配置：

| 项目 | 值 |
|---|---:|
| steps / unique records | 256 / 256 |
| target sequence | 重复 `C2,C2,C2,C3` |
| batch size | 1 |
| optimizer | AdamW，仅 adapter |
| learning rate | `3e-4` |
| weight decay | 0 |
| gradient clip | 1.0 |
| loss | flow matching + `1.0 × endpoint symmetry MSE` |
| checkpoint cadence | 64 steps |

step 256 的 deterministic endpoint Gate 通过：mean endpoint symmetry ratio `0.005408`，
mean flow-MSE ratio `0.69905`，C2/C3 改善率均为 100%。但是随后 raw ODE 的 collision-free
只有 26/32，未达到冻结阈值，因此不能直接作为最终生成模型。

该分支的可执行模型和训练脚本已在 2026-08-28 清理，不再作为维护代码。冻结协议与结论仍保留在
`generative_model/smoke/reports/etflow_ef2a2_training_protocol_v1.json` 及本节中；删除清单见
`generative_model/maintenance/smoke_etflow_cleanup_manifest_20260828.json`。

### 5.3 collision-aware 与 operation/orbit-aware adapter

后续分支尝试解决 A2 的碰撞和 C3 泛化：

| 分支 | 相对 A2 的修改 | 训练设置/结果 | 是否进入 v5 |
|---|---|---|:---:|
| EF2-C2 | 增加 symmetry-compatible 非键/键 overlap repulsion vector field | 256 条、batch 1、256 steps、lr `3e-4`；overlap loss 明显下降、collision-free 提高，但独立确认的 C3 compatible 仅 12/16 | 否 |
| C7RS | 保持 A2 架构，把训练面板改为 C2:C3 = 128:128，并按 `C2,C3` 交错 | 256 条、batch 1、256 steps、lr `3e-4`；endpoint 通过，但 C3 raw trajectory tail 失败 | 否 |
| EF2-D2 | 非 identity operation softmax、atom-wise gate、orbit mean/max 特征；endpoint displacement 限制 0.05 Å | 512 条唯一记录、batch 2、256 steps、lr `5e-5`；operation ratio 到 `0.821101`，但 nonbond collision ratio `1.005098`，无 eligible checkpoint | 否 |
| EF2-D3 | operation basis 与 overlap basis 分离，safe norm/stop-gradient | 消除部分梯度 NaN，但短 rollout collision tail 仍失败 | 否 |
| EF2-D4 | 每步 4 C2 + 4 C3；endpoint/4-step rollout collision CVaR | 512 条唯一记录、batch 8、64 steps、lr `3e-5`；operation 改善，但碰撞恶化分子占 15.625% > 10% | 否 |

D2 的训练 loss 是：

```math
\mathcal L =
1.0\mathcal L_{flow}
+0.25\mathcal L_{mean\ op}
+0.50\mathcal L_{worst\ op}
+0.50\mathcal L_{orbit}
+0.25\mathcal L_{bond}
+0.25\mathcal L_{collision}.
```

这些实验分支不是当前部署或消融依赖，其模型、训练和运行脚本已在 2026-08-28 删除。协议、
最终 JSON 结果及本节的训练设置/结论继续保留，因而仍可审计“已测试的 soft learned
conditioner 没有达到准入”；这不等于 ET-Flow 或未来其他 PG-conditioned 架构原则上不可行。

---

## 6. graph-only symmetry action

### 6.1 action 的数学约定

对每个群元素 `g` 保存：

- 三维正交矩阵 `R_g in O(3)`；
- 长度为 `N` 的原子置换 `p_g`；
- 由所有 `p_g` 导出的 `orbit_id/orbit_size`。

行向量坐标的约定是：

```math
X R_g^T = X[p_g].
```

也就是操作 `R_g` 作用于原子 `i` 的坐标后，应落到同元素原子 `p_g(i)` 的坐标上。

### 6.2 action 如何从 graph 恢复

恢复过程只读取：原子序数、形式电荷、自由基、芳香性、typed bonds 和显式 H 邻接，不读取
reference XYZ 或 v2 保存的 permutation。

- C2：寻找 exact-order-2 typed-graph automorphism；
- C3：寻找 exact-order-3 typed-graph automorphism；
- S4：寻找 exact-order-4 graph automorphism，并配对 improper generator
  `S4 = C4 × horizontal reflection`；
- D6h：寻找满足 `r^6=e, s^2=e, srs=r^-1` 的 D6 graph action，再加入 horizontal reflection；
  当前 D6h contract 中 horizontal reflection 的 atom permutation 是 identity，因此最终坐标被约束在
  分子平面上。

重原子 automorphism 找到后，H 按其唯一重原子邻居确定性提升。terminal Lewis resonance 允许在
预定义的等价末端原子多重集中归一化表示，但 connectivity 和元素绝不放松。

高对称 graph 可能有多个合法 action。S4/D6h 最多确定性枚举 64 个 action；这不是数据增强，
而是在多个合法 graph-subgroup embedding 中寻找与 raw conformer 相位更相容的一个。

代码：

- C2/C3：`generative_model/symmetry/graph_action.py`
- S4/D6h action：`generative_model/symmetry/graph_action_extended.py`
- S4/D6h 多 action：`generative_model/symmetry/graph_action_candidates_v2.py`
- resonance 验证：`generative_model/optimization/etkdg_orbit_resonance.py`

若 graph 不存在请求群所需的 automorphism，接口严格失败；不会通过移动不同元素、修改键或调用
reference coordinates 伪造对称性。

---

## 7. hard projection 到底是什么

### 7.1 Reynolds projection

给定任意质心归零坐标 `X`，代码对每个 `(R_g,p_g)` 执行：

```python
accumulator[p_g] += X @ R_g.T
Y = center(accumulator / number_of_operations)
```

逐原子写成：

```math
Y_{p_g(i)}\;{+}{=}\;\frac{1}{|G|}(X_iR_g^T).
```

这就是有限群表示上的 Reynolds average。若 matrices/permutations 构成一致的有限群作用，一次平均
就是到群不变坐标子空间的线性正交投影，并满足：

```math
Y R_g^T=Y[p_g],\qquad \Pi_G(\Pi_G(X))=\Pi_G(X).
```

“hard”的含义是输出坐标被直接替换为 `Pi_G(X)`；它不是一个可违反的 loss，也不依赖网络是否
学会对称性。soft symmetry loss 只能鼓励误差变小，hard projection 在数值精度内把坐标放入
目标 action 的不变子空间。

核心实现：`generative_model/models/graph_pg_3d_projection.py::project_reynolds()`。

### 7.2 为什么 projection 前要做 orientation search

ET-Flow raw conformer 的整体朝向任意，而 operation matrices 使用固定标准坐标系。例如 C3 主轴
默认沿 z 轴。直接投影一个主轴未对齐的 raw conformer，可能把大量几何分量相互抵消，甚至造成
原子重合。

当前流程：

1. raw coordinates 质心归零；
2. 计算 covariance 的 PCA 主轴；
3. 枚举 24 个 determinant `+1` 的 signed axis permutations；
4. 对每个方向做一次 Reynolds projection；
5. 可选地缩放回 raw radius of gyration；
6. 选 projection RMSD 最小的方向。

实现：`generative_model/optimization/etkdg_orbit_initializer.py::orient_and_project_to_orbits()`；
ET-Flow wrapper 位于 `generative_model/conformer/etflow_ef1_projection.py`。

### 7.3 hard projection 能与不能保证什么

能保证：

- 对指定 matrices/permutations 的 operation error 接近机器精度；
- 同一 orbit 中原子的坐标满足群作用；
- 后续若只在相同 orbit subspace 中优化，对称性不会被破坏。

不能单独保证：

- graph 本身存在合法目标 action；
- 投影后无碰撞；
- 键长、价态或构象能量合理；
- pymatgen 一定返回 exact target label。实际分子可能生成目标群的超群，因此主要 Gate 使用
  `target group <= actual group` 的 compatible 定义。

因此 hard projection 后还必须执行 F0.2 和独立 point-group audit。

---

## 8. F0.2：保持对称的几何优化与碰撞修复

### 8.1 orbit-subspace 参数化

设 orbit representative coordinates 为 `q`，固定 expansion matrix 为 `A`：

```math
X=Aq.
```

优化变量只有 `q`，完整原子坐标每次由 action 展开。这样优化器不可能独立移动同一 orbit 中的
某个原子，因此优化过程中始终保持 hard symmetry。

### 8.2 优化目标

F0.2 使用：

```math
E(q)=E_{UFF}(Aq)+E_{repulsion}(Aq).
```

对每个非键原子对、距离 `r < r_c`：

```math
E_{repulsion}(r)=\frac12 k(r_c-r)^2,
```

固定参数：

| 参数 | 值 |
|---|---:|
| primary force field | RDKit UFF |
| repulsion cutoff `r_c` | 0.8 Å |
| force constant `k` | 2000 kcal mol⁻¹ Å⁻² |
| optimizer | L-BFGS-B |
| maximum iterations | 500 |
| `ftol` | `1e-9` |
| `gtol` | `1e-5` |

若 UFF 参数不完整、初始坐标不在 orbit subspace、出现 NaN/Inf 或完全重合导致梯度方向未定义，
均严格失败，不切换到其他 force field。

实现：

- `generative_model/optimization/orbit_force_field_repulsion.py`
- orbit expansion：`generative_model/optimization/orbit_force_field.py`
- 严格 RDKit graph/UFF 支持：`generative_model/optimization/force_field_support.py`

---

## 9. v5 完整候选选择

### 9.1 普通 C2/C3

每个输入生成 4 个 raw ET-Flow conformer。每个 raw conformer经过：orientation search、hard
projection、F0.2。只在严格 geometry Gate 通过的候选中，选择最终 `UFF + repulsion` objective
最低者。

### 9.2 S4/D6h 多 action

最多 64 个 graph-only actions 与 4 个 raw conformers 形成最多 256 个组合。先按以下
reference-free 指标排序：

1. `distance < 0.6 Å` 的 pair 数升序；
2. `distance < 0.8 Å` 的 pair 数升序；
3. minimum pair distance 降序；
4. projection RMSD 升序；
5. raw/action candidate id 稳定 tie-break。

只对 top 8 组合执行较贵的 F0.2，再从严格通过者中按最终 objective 选择。

### 9.3 S4 inertia guard

S4 v2 虽然 operation error 很小，但 9 条构象落在 pymatgen 的近球形惯量退化分类分支。v5 对
通过 geometry Gate 的 S4 候选计算 mass-weighted principal moments：

```math
s_I=\frac{\max(I_2-I_1,I_3-I_2)}{I_3}.
```

只保留 `s_I >= 0.012`，再继续使用原有最低 F0.2 objective 规则。该指标平移、旋转、尺度不变，
不调用 pymatgen，也不读取 reference XYZ。它避免 analyzer 的近球形歧义，不是修改坐标或放宽
point-group tolerance。

实现：`generative_model/inference/s4_inertia_selection.py`。

---

## 10. 严格 Gate 与独立审计

每个最终候选必须满足：

| Gate | 阈值 |
|---|---:|
| optimizer | `success`，或 projected gradient norm `<=1.0` |
| minimum all-pair distance | `>=0.6 Å` |
| maximum operation error | `<=1e-5 Å` |

随后在独立 `env_cof` 中使用 pymatgen：

- `PointGroupAnalyzer`；
- tolerance `0.3 Å`；
- eigen tolerance `0.01`；
- matrix tolerance `0.1`；
- 同元素块 Hungarian assignment；
- `PG_Compatible` 定义为 requested target group 是 actual group 的子群。

例如请求 S4、actual D2d 时，S4 是 D2d 子群，因此 compatible 为真，但 exact 为假。项目不会把
compatible 误写成 exact。

最终证据：

| 目标群 | 固定面板 | geometry | analyzer | compatible |
|---|---:|---:|---:|---:|
| C2 | 48 | 48/48 | 48/48 | 48/48 |
| C3 | 16 | 16/16 | 16/16 | 16/16 |
| S4 | 38，canonical 全部 | 38/38 | 38/38 | 38/38 |
| D6h | 14，canonical 全部 | 14/14 | 14/14 | 14/14 |

完成状态：

```text
PASS_ETFLOW_E3F02_KNOWN_GRAPH_TARGET_PG_TO_XYZ_DELIVERABLE
```

完成报告：`generative_model/smoke/reports/etflow_e3f02_completion_v1.json`。

---

## 11. 当前推理输入与输出

### 11.1 输入

当前 CLI 输入的是 canonical v2 数据包中的 `package_index` 和 requested PG。模型实际读取：

- atomic numbers；
- formal charges；
- radical electrons；
- typed sparse bonds；
- explicit H；
- requested `C2/C3/S4/D6h`；
- sampling seed。

当前 CLI 不是任意外部 SMILES 的公共 API。若接入新 graph/SMILES，必须先转换为相同 canonical
schema，并通过 bridge、连通性、typed graph automorphism 与元素严格审计。

### 11.2 输出目录

| 文件 | 含义 |
|---|---|
| `raw.xyz` | 官方 ET-Flow 的原始构象，未施加 Target-PG |
| `projected.xyz` | orientation search + hard projection 后坐标 |
| `final.xyz` | orbit-constrained F0.2 后交付坐标 |
| `coordinates.npz` | 原子、键、operation、permutation、orbit 和三阶段坐标 |
| `report.json` | bridge、候选、优化、Gate、协议与哈希 |
| `point_group_report.json` | 独立 pymatgen actual/exact/compatible 审计 |

---

## 12. 运行命令

### 12.1 生成一个分子

```bash
cd /home/tianyajun/MARL_for_COFs
conda activate env_etflow

CUDA_VISIBLE_DEVICES=4 python -m \
  generative_model.inference.generate_etflow_symmetric_xyz_v5 \
  --package-index 1102 \
  --target-pg S4 \
  --seed 2026084202 \
  --protocol generative_model/inference/etflow_e3f02_protocol_v5.json \
  --output-dir generative_model/runs/etflow_e3f02_examples/s4_001102 \
  --device cuda
```

### 12.2 独立 point-group 审计

```bash
conda activate env_cof

python -m generative_model.inference.audit_etflow_e3f02_point_group \
  --run-dir generative_model/runs/etflow_e3f02_v5_examples/s4_001102 \
  --protocol generative_model/inference/etflow_e3f02_protocol_v5.json
```

### 12.3 复核四点群完成报告

```bash
conda activate env_cof
python -m generative_model.inference.audit_etflow_e3f02_completion
```

---

## 13. 代码文件地图

| 功能 | 代码/协议 |
|---|---|
| canonical v2 graph loader | `generative_model/data/cof_graph_dataset.py` |
| official ET-Flow strict bridge/runtime | `generative_model/conformer/etflow_bridge.py` |
| official package BaseFlow | `env_etflow/.../site-packages/etflow/models/model.py` |
| official package TorchMD dynamics | `env_etflow/.../site-packages/etflow/networks/torchmd_net/model_dynamics.py` |
| 历史 learned-PG adapter 证据 | `generative_model/smoke/reports/etflow_*.json`；可执行旧分支已按 cleanup manifest 删除 |
| C2/C3 graph action | `generative_model/symmetry/graph_action.py` |
| S4/D6h graph action | `generative_model/symmetry/graph_action_extended.py` |
| S4/D6h multi-action enumeration | `generative_model/symmetry/graph_action_candidates_v2.py` |
| Reynolds projection/error | `generative_model/models/graph_pg_3d_projection.py` |
| PCA×24 orientation + projection | `generative_model/optimization/etkdg_orbit_initializer.py` |
| ET-Flow projection wrapper | `generative_model/conformer/etflow_ef1_projection.py` |
| orbit-constrained UFF+repulsion | `generative_model/optimization/orbit_force_field_repulsion.py` |
| S4 inertia guard | `generative_model/inference/s4_inertia_selection.py` |
| current v5 CLI/orchestration | `generative_model/inference/generate_etflow_symmetric_xyz_v5.py` |
| current frozen v5 protocol | `generative_model/inference/etflow_e3f02_protocol_v5.json` |
| single-run independent PG audit | `generative_model/inference/audit_etflow_e3f02_point_group.py` |
| C2/C3 batch runner/audit | `generative_model/inference/run_etflow_e3f02_iidtest.py`、`audit_etflow_e3f02_iidtest.py` |
| S4/D6h batch runner/audit | `generative_model/inference/run_etflow_e3f02_rare_targets_v3.py`、`audit_etflow_e3f02_rare_targets.py` |
| final completion audit | `generative_model/inference/audit_etflow_e3f02_completion.py` |
| paired ablation/Core-OOD/external evaluators | `generative_model/evaluation/our_etflow_ablation.py`、`run_our_etflow_*.py`、`audit_our_etflow_*.py` |
| same-graph multi-PG controllability | `generative_model/evaluation/run_our_etflow_controllability.py`、`audit_our_etflow_controllability.py` |
| GFN2-xTB relaxation protocol/runner | `generative_model/evaluation/our_etflow_xtb_protocol_v1.json`、`run_our_etflow_xtb_relaxation.py` |
| GFN2-xTB point-group audit/summary | `generative_model/evaluation/audit_our_etflow_xtb_point_group.py`、`summarize_our_etflow_xtb.py` |
| 清理策略与精确删除清单 | `generative_model/maintenance/prune_smoke_etflow_history.py`、`smoke_etflow_cleanup_manifest_20260828.json` |

说明：v5 并非可以孤立保留的单文件。其冻结协议将基础实现及
`generate_etflow_symmetric_xyz_v2.py`、`v3.py`、`v4.py`、`v5.py` 一并纳入源码 SHA-256，
最终消融与可控性实验也调用 v3/v4 的 action enumeration 和候选排序。因此这些版本文件属于
**v5/消融的运行依赖**，不是待删除的旧训练分支；擅自合并或删除会使冻结协议和现有结果失去
字节级可复现性。

---

## 14. 2026-08-27：四路线消融与 Core-OOD

### 14.1 冻结的 paired ablation

新增 EVAL_v1，使用相同 canonical graph、base seed 和 requested `Target_PG`，且三个 ET-Flow
分支共享同一批 4 个 raw samples。reference XYZ 只用于事后 bond/RMSD 指标，不参与生成或
候选选择。

| route | 定义 | Target_PG 是否进入坐标生成 |
|---|---|---:|
| ETKDGv3 best-of-4 | 4 个 ETKDGv3，按严格 UFF 单点能最低选样 | 否 |
| ET-Flow raw | 官方 checkpoint、50 ODE steps、4 samples，按 UFF 单点能选样 | 否 |
| ET-Flow + hard projection | raw + graph action + PCA×24 + Reynolds projection | 是 |
| ET-Flow + projection + F0.2 | 冻结 v5 完整候选选择与 orbit UFF+repulsion | 是 |

32 条面板为 C2/C3/S4/D6h 各 8。完整分母结果：

| route | 生成 | PG-compatible | collision-free | mean bond MAE (Å) | median UFF/atom (kcal/mol) | median runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| ETKDGv3 best-of-4 | 31/32 | 9/32 | 29/32 | 0.02575 | 2.8749 | 0.267 |
| ET-Flow raw | 32/32 | 6/32 | 32/32 | 0.02349 | 3.5696 | 6.905 |
| + hard projection | 32/32 | 30/32 | 27/32 | 0.11777 | 13.4112 | 7.410 |
| + F0.2 | 32/32 | 32/32 | 32/32 | 0.01770 | 1.6460 | 10.073 |

结论不是“hard projection 单独足够”：它把 compatible 从 raw 的 `18.75%` 提高到 `93.75%`，
但 collision-free 降到 `84.375%`，且能量/键长恶化；F0.2 才把 compatible 和 collision-free
同时恢复到 `100%`，并把 mean bond MAE 降到 `0.01770 Å`。完整路线相对 ETKDG 的
PG-compatible 提升 `71.875` 个百分点，代价是中位运行时间增加约 `9.81 s/分子`。

候选多样性没有被完全消除：完整路线的 mean within-graph pairwise Kabsch diversity 为
`1.51169 Å`，ETKDG 为 `1.59255 Å`，ET-Flow raw 为 `2.38643 Å`。

### 14.2 Core-OOD unseen graph

另冻结 16 C2 + 16 C3 Core-OOD test 面板：与上述消融及旧 v5 固定面板零重叠；相对
Core-OOD train 的 canonical SMILES、Core 和 exact Morgan fingerprint 均零重叠。筛选时有
56 个候选因 Morgan/Tanimoto `=1.0` 被严格排除。最终 32 条最近训练 Tanimoto 为
`0.1667–0.8235`，均值 `0.5795`，覆盖 10 个 unseen Core。

| route | PG-compatible | collision-free | mean bond MAE (Å) | median UFF/atom |
|---|---:|---:|---:|---:|
| ETKDGv3 best-of-4 | 4/32 | 31/32 | 0.02640 | 2.7220 |
| ET-Flow raw | 6/32 | 32/32 | 0.01694 | 2.1147 |
| + hard projection | 32/32 | 25/32 | 0.09328 | 14.6276 |
| + F0.2 | 32/32 | 32/32 | 0.01305 | 1.1679 |

完整路线在 Core-OOD 上 C2 与 C3 都是 `16/16 compatible`。这证明当前 composite backend
可迁移到未见 Core 的新 graph；它仍不证明 raw ET-Flow 学会了 PG 条件，因为 raw 只有
`6/32 compatible`，目标响应来自 graph action/projection/F0.2。

主要文件：

- protocol：`generative_model/evaluation/our_etflow_ablation_protocol_v1.json`、
  `our_etflow_core_ood_protocol_v1.json`；
- runner/auditor：`run_our_etflow_ablation.py`、
  `audit_our_etflow_ablation_point_group.py`；
- 汇总：`summarize_our_etflow_ablation.py`；
- 结果：`generative_model/runs/our_etflow_ablation_v1/`、
  `generative_model/runs/our_etflow_core_ood_v1/`。

### 14.3 外部于 COF v2 的图

另构建 12 条小型外部分子面板，并严格保证它们与 2,532 条 COF v2 在 canonical SMILES、
Bemis–Murcko scaffold 和 exact Morgan fingerprint 上均无重叠。这里的“外部”只表示未进入
COF v2；由于没有官方 GEOM-DRUGS 训练清单，不能声称它们未被官方 ET-Flow 预训练见过。

| route | 生成 | analyzer | PG-compatible | collision-free |
|---|---:|---:|---:|---:|
| ETKDGv3 best-of-4 | 12/12 | 11/12 | 3/12 | 12/12 |
| ET-Flow raw | 12/12 | 11/12 | 4/12 | 11/12 |
| + hard projection | 12/12 | 11/12 | 6/12 | 5/12 |
| + F0.2 | 10/12 | 10/12 | 10/12 | 10/12 |

完整路线的 C2、C3、D6h 分别为 `4/4、4/4、2/2 compatible`。两个 S4 失败不是 ET-Flow
或 graph action 不支持，而是 frozen v5 的 S4 inertia guard 拒绝 cubane/adamantane：这类
Td/Oh 高对称结构天然具有近简并惯量，现有防退化 guard 过于保守。该负结果保留为未来 v6
需要解决的域外 S4 限制，不在当前结果上放宽阈值。

代码和结果：`external_symmetric_candidates_v1.json`、
`build_our_etflow_external_package_v1.py`、`run_our_etflow_external.py` 以及
`generative_model/runs/our_etflow_external_v1/`。

### 14.4 同一 graph 的 Target_PG 可控性

从 canonical D6h 图中固定 8 条分子；对每条分子只生成一次相同的 4 个 raw ET-Flow
coordinates，然后分别请求 C2、C3、D6h。这样 graph、seed 和 raw tensor 完全相同，唯一变化是
requested Target_PG 及其 graph action/projection/F0.2。

- 三个 target 均生成 `8/8`，且独立 pymatgen compatible 均为 `8/8`；
- C2↔D6h、C3↔D6h 的结构响应率均为 `8/8`；
- C2↔C3 只有 `5/8` 的 Kabsch RMSD 大于预冻结 `0.05 Å`，低于 `6/8` 门槛；
- C2/C3 输出常落到共同超群 D3d 或 D6，因此 compatible 并不等于 target-specific response。

未响应的三条为 package `2529、860、2528`，C2↔C3 Kabsch RMSD 分别仅
`0.000283、0.000904、0.003053 Å`；其余 5 条为 `0.480–2.234 Å`。这说明失败集中在
“不同子群请求坍缩到同一高对称解”，不是所有样本都只有微弱响应。

正式状态为 `FAIL_OUR_ETFLOW_TARGET_PG_CONTROLLABILITY_DIAGNOSE`。它说明当前 composite
后端能够保证“至少包含所请求子群”，但对高对称 graph 尚不能稳定区分 C2 与 C3 的特异构象。
这项失败不能用 100% compatible 覆盖。

代码和结果：`run_our_etflow_controllability.py`、
`audit_our_etflow_controllability.py`、
`generative_model/runs/our_etflow_controllability_v1/controllability_audit_v2.json`。

### 14.5 GFN2-xTB 无约束物理松弛确认（已完成）

该实验直接使用 §14.1 中 32 个完整 F0.2 坐标，执行无约束 GFN2-xTB/LBFGS 松弛：

- C2/C3/S4/D6h 各 8；失败仍计入 32 个总分母；
- `tblite==0.7.0`、`ASE==3.24.0`、GFN2-xTB、`fmax=0.1 eV/Å`、最多 100 步；
- charge 和 multiplicity 严格来自 canonical formal charge/radical；
- 松弛期间不调用 hard projection、不施加 orbit constraint、不使用 reference XYZ；
- 比较能量、力、碰撞、键长、Kabsch RMSD，并独立重算松弛前后 actual/exact/compatible PG。

冻结协议为 `generative_model/evaluation/our_etflow_xtb_protocol_v1.json`，SHA-256
`33fd9d65ee2cdbc88860bb8ed639011151a7db6cd63ca3c165113e2e9573fde9`；4 项协议/几何测试已通过。

正式结果：

| 指标 | 结果 |
|---|---:|
| GFN2-xTB 计算成功 / LBFGS 收敛 | 32/32 / 32/32 |
| 能量下降 | 32/32；中位总变化 `-0.71371 eV` |
| 松弛后 collision-free | 32/32 |
| pre/post Kabsch RMSD | median `0.07155 Å`；max `0.85489 Å` |
| pre/post bond-length MAE | median `0.01852 Å`；max `0.29403 Å` |
| 松弛后 analyzer success | 32/32 |
| 松弛后 PG-compatible / exact | 31/32 / 20/32 |

分点群 compatible 为 C2 `8/8`、C3 `8/8`、S4 `8/8`、D6h `7/8`。唯一失稳样本
`cof_000858` 从 D6h 松弛为 D3h，能量变化 `-65.6451 eV`（`-0.91174 eV/atom`）、
RMSD `0.85489 Å`、bond MAE `0.29403 Å`；它没有碰撞，而是发生真实的低能降对称。
因此正式状态为 `PASS_OUR_ETFLOW_GFN2_XTB_PHYSICAL_STABILITY_PILOT`，同时必须保留
“D6h 不是逐分子 100% 稳定”的边界。

### 14.6 最终统计与可视化资产

已从冻结结果确定性生成 30 个汇报资产：四路线总览、各点群成功率、能量—actual symmetry
error、运行时间—联合成功率、xTB 松弛总览和能量—松弛后误差图、两组 XYZ 对照图、12 个
独立 XYZ 与两份 CSV。所有成功率使用完整面板分母；典型失败包括 hard-projection 碰撞、
C2/C3 条件坍缩和 D6h→D3h 物理降对称。

资产目录：`docs/presentation/generative_model_progress_20260821/assets/our_etflow_eval_v1/`；
manifest SHA-256 为 `104f679c45cc7d08c163f26cb513821da4f888e96dd4fac5c6f896bc17d27145`。
生成脚本为 `generative_model/evaluation/build_our_etflow_figures.py`。

### 14.7 补充实验总完成审计

为防止把“文件存在”或单个 PASS 当作全部目标完成，新增机器可读的逐项审计：
`generative_model/evaluation/audit_our_etflow_supplement_completion.py`。它分别核验四路线
paired contract、Core-OOD 与外部集非重叠、同图多点群条件实验、无约束 GFN2-xTB 四点群
面板，以及最终 CSV/PNG/SVG/XYZ 的输入与产物哈希。

正式审计结果为 `PASS_OUR_ETFLOW_SUPPLEMENT_REQUIREMENTS_COMPLETED`，六类要求均无缺失检查；
连续执行两次得到相同报告 SHA-256
`985d238ef73a329ff6437c018bd6709b4e1f7a3230ed7b2a792a7c0eb2dc0752`。审计明确区分
“实验完成”与“科学 Gate 通过”：C2↔C3 可控性 `5/8`、外部 S4 `0/2`、xTB 后 D6h
`7/8` 等负结果均原样保留，没有被总完成状态覆盖。

## 15. 当前限制与下一阶段

1. 当前解决的是 known graph → symmetric 3D，不生成 novel graph/SMILES；
2. graph 必须存在 requested PG 所要求的合法 graph automorphism；
3. 最终对称性依赖 hard projection，不是 raw learned PG response；
4. C2/C3 的最终 v5 证据是固定 IID-test 与 fresh Core-OOD 面板，不是全部 2,480 条；S4/D6h
   已在 canonical 数据集全覆盖；
5. D6h 只有 14 条 canonical 样本，应标注 low-support；
6. 外部 COF-v2 面板已完成，但两个高对称 S4 分子暴露了 inertia guard 的域外限制；
7. 同图多 PG 的 compatible 为 100%，但 C2↔C3 可区分响应只有 5/8，尚未证明强条件可控性；
8. GFN2-xTB pilot 已通过总体 Gate，但 D6h 只有 `7/8` 保持 compatible；不能据此声明所有
   生成结构都处于保持目标点群的量子化学局部极小值。

下一主线应是：

```text
Target_PG -> novel chemically valid graph/SMILES
          -> canonical graph validation
          -> current our_ET_Flow v5 3D backend
          -> independent PG and chemistry audit
```

若未来重新研究“不依赖 hard projection 的 learned PG-conditioned ET-Flow”，必须新建版本化
架构和预注册 Gate；不能把已失败的 EF2 adapter checkpoint 悄悄加入当前 v5，也不能用当前
composite route 的 100% compatible 结果反向宣称 soft adapter 训练成功。

---

## 16. 维护规则

- 官方 checkpoint、v5 protocol 和已归档报告用 SHA-256 固定；修改算法时新建 v6，不覆盖 v5；
- learned adapter 和 production composite route 的结论必须分栏记录；
- 任何新目标群都先实现 graph action closure/automorphism 测试，再接 projection；
- 任何新数据入口都必须保留 explicit H、Si/Sn、charge/radical 和 typed bonds，禁止 fallback；
- point-group 结果同时报告 target、actual、exact、compatible；
- geometry success 必须包含完整失败分母，不删除失败样本；
- 独立 analyzer 必须在 `env_cof` 运行，不能只报告 stored-operation error；
- 重大动态变化同步更新 `docs/项目动态状态.md` 和本文。
