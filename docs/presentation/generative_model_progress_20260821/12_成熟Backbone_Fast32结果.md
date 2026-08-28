# 成熟等变 Backbone：C2/C3 Fast-32 结果

## 1. 为什么做这一阶段

G2.2 证明手写三通道 E(3) 网络有学习信号，但单分子 step-1,024 仍未过绝对 RMSD Gate。
为加速，不再开发 G2.3/G2.4，而是直接复用锁定 SemlaFlow 源码中的成熟等变 backbone：

- 主模型：官方 `EquiInvDynamics`；
- 唯一备选：官方 `EqgatDynamics`；
- 上游 commit：`3f43103d3af138b86dbe9f29fe8085e83f9a6283`；
- 上游源码不修改，只增加本项目的 13 元素、真实键、sigma 和 `Target_PG` 适配层。

此前 SemlaFlow 的联合 atom/bond/XYZ 生成失败不等于该 backbone 不能做固定图坐标任务；本阶段
只预测坐标，不经过官方元素词表，Sn 不会映射成 C/Si。

## 2. 实验设计

固定面板通过 IID split 内 SHA-256 排序产生，与模型性能无关：

| 面板 | C2 | C3 | 总数 | 最大原子数 |
|---|---:|---:|---:|---:|
| IID-train | 24 | 8 | 32 | 63 |
| IID-validation | 6 | 2 | 8 | 60 |

训练使用六档坐标噪声 `0.05/0.1/0.2/0.4/0.8/1.2 Å`、batch 4、512 steps。验证为
8 molecules × 2 noise levels × 2 seeds = 32 cases。test/Core-OOD 未使用。

输入/输出为：

```text
immutable atom/bond graph + noisy XYZ + sigma + Target_PG
        ↓ official equivariant backbone + project adapter
centered clean-coordinate x0 prediction
```

Gate 不再要求单构象 RMSD ≤0.05 Å，而要求跨分子去噪改善、各 noise level 相对改善、无碰撞、
raw symmetry 改善、projected analyzer/compatible 和非零 PG-condition sensitivity。

## 3. 工程验收

两个 adapter 均通过：

- finite forward/backward；
- E(3) 旋转/平移/反射；
- 节点排列等变；
- 13 元素/电荷/自由基/键/PG 严格范围检查，无 fallback；
- canonical graph 不被修改。

Semla 首轮启动遗漏了官方 GEOM 坐标标准化：官方会以
`2.407038688659668 Å` 缩放坐标。该轮保留为接口失效证据，状态仅为
`INVALID_ADAPTER_COORDINATE_SCALE_MISSING`，不用于 backbone 判定。v2 修复后才是正式结果。

## 4. 正式结果

| 指标 | Semla EquiInv v2 | EQGAT | Gate |
|---|---:|---:|---:|
| 参数量 | 768,544 | 387,660 | 描述性 |
| last/first 64-step loss | 0.3044 | 0.7500 | ≤0.8 |
| mean predicted/noisy RMSD ratio | 3.317 | 4.228 | ≤0.9 |
| RMSD improved | 7/32 | 5/32 | ≥24/32 |
| collision-free | 13/32 | 8/32 | ≥90% |
| raw symmetry improved | 31/32 | 32/32 | ≥50% |
| projected analyzer/compatible | 19/32 | 10/32 | ≥90% |
| PG sensitivity RMSD | 0.183 Å | 0.102 Å | ≥1e-4 Å |

Semla v2 状态：

`FAIL_SEMLA_BACKBONE_FAST32_V2_SWITCH_ONCE_TO_EQGAT`

EQGAT 状态：

`FAIL_EQGAT_FAST32_STOP_BACKBONE_SCREEN`

两者 loss、raw symmetry 和 PG sensitivity 证明训练/条件通路不是完全失效；但绝对 `x0`
输出在独立分子上远差于 noisy input。Semla 明显优于 EQGAT，因此保留 Semla backbone，停止
横向增加模型。

## 5. 当前解释与下一步

当前失败不能继续简化为“backbone 不够强”。两个官方 backbone 原本服务于 flow matching，
本阶段却把它们用作单步 absolute clean-coordinate 回归器。下一步只改变任务参数化：

```text
centered Gaussian prior x0 + target coordinate x1
        ↓ sample t and interpolate x_t
Semla EquiInvDynamics(graph, PG, x_t, t)
        ↓ velocity or endpoint-consistent prediction
multi-step ODE integration → generated XYZ
```

仍复用同一 32/8 面板；若 flow fast Gate 通过，立即准备 2,026 条 IID-train 正式训练，不再
做第三个 backbone 或单分子 Gate。

## 6. 机器证据

- Semla v2 protocol：`adb08827627001856845ae1617aa7e65c2451366f540e565196a4e96c5d925a6`
- Semla v2 report：`925967e2786fdd7316ece84ecd248f56716ddab887321d14db03c41fafb00d83`
- Semla v2 checkpoint：`6748fe1ba49c49877fd4ef31feab7949d0d0b60190f2dd8b41314e8491b150a6`
- EQGAT protocol：`f42545ed81679c76af1d9f9312a17e18ab724789c2985262f5c80abcde8bc6fb`
- EQGAT report：`af7e377b182b9fa17d1698a38355d310bfdfeb0de65cf0f6e34236d5647f503e`
- EQGAT checkpoint：`9e498f337f4720ed6a21b0496d42ac180bd55b12b3935c81414b36ab9309c5ed`
