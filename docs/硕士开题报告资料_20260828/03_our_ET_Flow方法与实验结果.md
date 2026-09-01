# 3. our_ET_Flow 方法与实验结果

## 3.1 任务与能力边界

our_ET_Flow 当前解决：

```text
known canonical graph + requested Target_PG
        → PG-compatible 3D coordinates
```

它不生成新的原子、化学键或 SMILES。最终交付路线也没有使用训练后的 PG adapter：官方
ET-Flow 只提供 learned initial conformer，目标对称性由 graph action、hard projection 和 F0.2
保证。

## 3.2 完整流程

```text
known graph ───────────────→ frozen ET-Flow drugs-o3
    │                              ↓ 4 raw conformers
    │
    └─ Target_PG → graph action: {R_g, p_g, orbit}
                                   ↓
                         PCA×24 orientation search
                                   ↓
                         Reynolds hard projection
                                   ↓
                    orbit-constrained UFF + repulsion
                                   ↓
                         strict geometry Gate
                                   ↓
                    independent actual-PG audit
                                   ↓
                               final.xyz
```

## 3.3 冻结 ET-Flow 构象先验

官方 `drugs-o3.ckpt` 是已知分子图条件下的构象生成模型。输入原子特征、真实键、当前坐标和时间，
输出坐标速度场：

```math
v_\theta(x_t,t,G)\in\mathbb R^{N\times3}.
```

当前每个输入固定生成 4 个 raw conformer，每个使用 50 个 ODE Euler steps：

```math
x_{k+1}=x_k+\Delta t\,v_\theta(x_k,t_k,G).
```

raw coordinates 未读取 Target_PG，因此不能把 raw ET-Flow 的结果称为 PG-conditioned generation。

## 3.4 从图恢复有限群作用

对于每个群元素保存：

- `R_g∈O(3)`：空间旋转/反射矩阵；
- `p_g`：长度为 N 的同元素原子置换；
- `orbit_id/orbit_size`：由所有置换导出的原子轨道。

约定：

```math
XR_g^T=X[p_g].
```

C2/C3 通过 typed graph automorphism 恢复 exact-order-2/3 的作用；H 按唯一重原子邻居提升。
S4/D6h 还需要恢复 improper/dihedral action，并对多个合法 graph embedding 做确定性枚举。若图
不存在目标点群要求的 automorphism，严格失败。

## 3.5 Reynolds hard projection

给定质心归零的 raw 坐标，执行：

```math
Y_{p_g(i)}\;{+}{=}\;\frac1{|G|}X_iR_g^T.
```

得到：

```math
YR_g^T=Y[p_g],\qquad \Pi_G(\Pi_G(X))=\Pi_G(X).
```

ET-Flow 的整体朝向任意，而标准群操作使用固定坐标系。因此投影前先进行 PCA，并枚举 24 个
determinant `+1` 的 signed axis permutations，选择 projection RMSD 最小的方向。

hard projection 的优点是数值上严格满足群作用；缺点是它不考虑化学能量，可能制造原子碰撞、
扭曲键长或把结构压扁。

## 3.6 F0.2：保持对称的几何修复

设 orbit representative coordinates 为 `q`，固定展开矩阵为 `A`：

```math
X=Aq.
```

F0.2 只优化 `q`，完整原子坐标每一步由群作用展开，因此不会单独移动同一 orbit 中的一个原子。
目标函数为：

```math
E(q)=E_{UFF}(Aq)+E_{repulsion}(Aq),
```

非键原子对在 `r<0.8 Å` 时使用：

```math
E_{repulsion}(r)=\frac12\times2000\times(0.8-r)^2.
```

优化器为 L-BFGS-B，最多 500 iterations。UFF 参数缺失、NaN/Inf、完全重合或无法在 orbit
subspace 中优化时严格失败，不静默切换方法。

## 3.7 候选选择与 Gate

C2/C3 的 4 个 raw conformer 分别执行 orientation、projection 和 F0.2。候选必须满足：

| Gate | 阈值 |
|---|---:|
| optimizer | success，或 projected gradient norm ≤ 1.0 |
| minimum all-pair distance | ≥ 0.6 Å |
| maximum operation error | ≤ 1e-5 Å |

通过者中选择 `UFF + repulsion` objective 最低者。随后在独立 `env_cof` 中使用 pymatgen 重新
计算 `actual_pg/exact/compatible`，不能仅凭 stored operation error 宣称点群通过。

## 3.8 四路线 paired ablation

冻结 32 条面板，C2/C3/S4/D6h 各 8，所有 ET-Flow 分支共享相同 raw samples：

| route | 生成 | PG-compatible | collision-free | mean bond MAE | median UFF/atom | median runtime |
|---|---:|---:|---:|---:|---:|---:|
| ETKDGv3 best-of-4 | 31/32 | 9/32 | 29/32 | 0.02575 Å | 2.8749 | 0.267 s |
| ET-Flow raw | 32/32 | 6/32 | 32/32 | 0.02349 Å | 3.5696 | 6.905 s |
| + hard projection | 32/32 | 30/32 | 27/32 | 0.11777 Å | 13.4112 | 7.410 s |
| + F0.2 | 32/32 | 32/32 | 32/32 | 0.01770 Å | 1.6460 | 10.073 s |

核心结论：

1. raw ET-Flow 几何较合理，但不响应 Target_PG；
2. projection 将 compatible 从 18.75% 提升到 93.75%，但碰撞和能量变差；
3. F0.2 才使 compatible、collision-free 和局部几何同时恢复；
4. 因此完整结果不能归因于 ET-Flow adapter 学习成功。

## 3.9 泛化证据

### Core-OOD

fresh 16 C2 + 16 C3 面板与旧面板零重叠，并排除 canonical SMILES、Core 和 exact Morgan
fingerprint 重叠。完整路线：

- `32/32 PG-compatible`；
- `32/32 collision-free`；
- mean bond MAE `0.01305 Å`；
- raw ET-Flow 仅 `6/32 compatible`。

这证明 composite backend 可以处理 unseen Core graph，但不证明官方 ET-Flow 未见过所有分子，
也不证明 raw PG conditioning。

### 外部 Cn 2D 图

新上传的 C3k2/C3k3 共 98 个外部 2D graph 不提供参考三维坐标。严格恢复显式 H 和 C3 graph
action 后：

- 98/98 生成成功；
- 98/98 独立 C3-compatible；
- 32/98 exact C3，其余多为 C3v/C3h/D3/D3h 等超群；
- 最小原子间距 `0.76144 Å`；
- 94 条通过保守综合筛查，4 条为需 xTB/DFT 复核的 WARNING。

该实验直接说明 current backend 可把外部 2D 图转为 C3-compatible XYZ，但最终对称性仍来自
显式 graph action/projection/F0.2。

## 3.10 物理稳定性 pilot

对 paired ablation 的 32 个 F0.2 坐标执行无约束 GFN2-xTB/LBFGS 松弛：

| 指标 | 结果 |
|---|---:|
| xTB 计算 / 收敛 | 32/32 / 32/32 |
| 能量下降 | 32/32 |
| 松弛后 collision-free | 32/32 |
| pre/post median RMSD | 0.07155 Å |
| 松弛后 PG-compatible | 31/32 |

唯一失败为 D6h→D3h 的真实低能降对称。这说明 F0.2 坐标整体接近可接受局部几何，但不能声明
所有目标点群都是量子化学局部极小值。

## 3.11 条件可控性不足

对 8 个相同 D6h graph，固定相同 raw tensors，分别请求 C2/C3/D6h：

- 三种 target 均为 8/8 compatible；
- C2↔D6h、C3↔D6h 强响应 8/8；
- C2↔C3 只有 5/8 超过 0.05 Å 响应门槛，低于 6/8 Gate。

原因是某些输出坍缩到同时包含 C2 和 C3 的共同超群。这个结果说明：

> compatible rate 衡量“至少满足目标子群”，不能单独证明 target-specific controllability。

## 3.12 当前贡献和限制

### 可作为论文工作的贡献

1. 建立从 graph automorphism 到 3D operation/permutation/orbit 的严格接口；
2. 验证 Reynolds projection 与 orbit-constrained force-field 的互补关系；
3. 建立 raw、projection、F0.2 和 independent actual-PG 的分层评测；
4. 在 COF、Core-OOD、外部 C3 图和 xTB 松弛上给出完整正负结果；
5. 为下一代 symmetry-by-construction generator 提供工程基线和诊断工具。

### 不能越界的声明

- 不能称为“ET-Flow 已学会指定点群生成”；
- 不能称为“Target_PG → novel molecule”；
- 不能用 compatible 100% 覆盖 C2/C3 可控性失败；
- 不能把 UFF/xTB pilot 等同于实验可合成性；
- D6h 只有 14 条，必须作为 exploratory 结果。

## 3.13 代码入口

| 功能 | 文件 |
|---|---|
| ET-Flow bridge | `generative_model/conformer/etflow_bridge.py` |
| C2/C3 graph action | `generative_model/symmetry/graph_action.py` |
| S4/D6h graph action | `generative_model/symmetry/graph_action_extended.py` |
| Reynolds projection | `generative_model/models/graph_pg_3d_projection.py` |
| PCA×24 orientation | `generative_model/optimization/etkdg_orbit_initializer.py` |
| F0.2 | `generative_model/optimization/orbit_force_field_repulsion.py` |
| v5 inference | `generative_model/inference/generate_etflow_symmetric_xyz_v5.py` |
| 独立 PG audit | `generative_model/inference/audit_etflow_e3f02_point_group.py` |

完整技术文档见 [our_ET_Flow.md](../our_ET_Flow.md)。

