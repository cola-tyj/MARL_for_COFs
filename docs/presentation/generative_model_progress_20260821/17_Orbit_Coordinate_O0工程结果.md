# Orbit-coordinate O0 工程结果

**日期**：2026-08-22  
**状态**：`PASS_ORBIT_COORDINATE_O0_FREEZE_O1`  
**结论边界**：hard-symmetry denoiser 的工程接口通过；尚未训练，不能宣称生成质量通过

## 1. 为什么不是普通 graph→xyz MLP

冻结的 Fast-32 面板包含 24 C2 + 8 C3 IID-train 和 6 C2 + 2 C3 IID-validation。只读统计
显示不同分子的旋转主轴并不统一在 z 轴：轴向量绝对值的三个分量都覆盖较宽范围。因此，
直接在固定全局坐标系中用普通 MLP 回归 xyz 会把任意分子取向当作监督信号。

O0 继续使用 E(3)-equivariant G2.2 三通道图网络产生 atom-wise coordinate proposals，再把
所有同 orbit proposal 经 Reynolds pooling 合并为一个 representative，最后用目标群操作和
permutation index 展开全原子坐标。这样：

- graph encoder 对整体旋转/平移保持正确变换；
- 每个 orbit 最终只有一个独立代表坐标；
- 所有 atom proposal 都通过 pooling 获得梯度，不因只选择最小 atom index 而破坏节点置换；
- 输出严格满足保存的群作用，不依赖 soft symmetry loss；
- canonical atom/bond graph、显式 H 和 13 元素词表不变。

## 2. O0 输入、目标与输出

```text
immutable sparse graph + Target_PG
+ exact-symmetry noisy coordinates
+ target operations/permutations/orbits
        ↓ E(3) G2.2 atom proposals
Reynolds orbit pooling
        ↓ one coordinate / orbit
differentiable hard expansion
        ↓
centroid-zero exact-symmetry full-atom coordinates
```

- 输入噪声：只对 orbit representatives 加噪，再展开为严格对称的全原子输入；
- 训练目标：canonical v2 坐标经过同一 hard expansion 得到的最近严格对称坐标；
- 输出：原子数和原子顺序不变，只预测坐标；
- O0 只验工程路径，不从独立 Gaussian prior 采样，也不生成新 graph。

## 3. 冻结面板与模型

- train panel：32 条 IID-train，24 C2 + 8 C3，12–63 atoms，6–31 orbits；
- validation panel：8 条 IID-validation，6 C2 + 2 C3，15–60 atoms，5–20 orbits；
- 平均 coordinate DOF ratio：train `0.46230`，validation `0.45833`；
- Sn：固定 IID-train 样本 index `1684`，仅作 strict forward，不加入质量面板；
- test/Core-OOD：未读取。

模型复用 G2.2 的 bonded/local/global 三通道等变图编码器，hidden dimension 96、6 层，共
`1,145,970` 参数。这里复用的是已有架构和消息通路，不恢复失败 checkpoint，也不重启
endpoint flow。

## 4. 工程 Gate 结果

| 检查 | 结果 |
|---|---:|
| 单元测试 | 5/5 通过，包含节点置换协变 |
| 真实 C2+C3 前后向与梯度 | finite，通过 |
| 最大 operation error | `6.03464e-7 Å` |
| 相同输入重复输出 | bit-exact |
| Sn strict forward | 通过 |
| 非法元素/群作用 | strict failure，通过 |
| canonical graph | 未修改 |
| test/Core-OOD | 未使用 |

正式状态：

```text
PASS_ORBIT_COORDINATE_O0_FREEZE_O1
```

工程报告连续两次生成 SHA-256 一致，证明机器报告字节稳定。

## 5. 机器证据

- O0 protocol：`generative_model/smoke/reports/orbit_coordinate_o0_protocol_v1.json`
- protocol SHA-256：`fd1916dd9e5ceb415e790c53be0a0db1606a48d963b3d04fee4e6fc958217816`
- O0 engineering report：`generative_model/smoke/reports/orbit_coordinate_o0_engineering_v1.json`
- report SHA-256：`12c58d54562b0abdacd5834ab2156aee8dd3b3464d703158b69d0550480c0b2e`

## 6. 下一步 O1

O0 随后只授权了下一项：实现训练器并单独冻结其源码 hash 和最终质量阈值，然后运行唯一一次
512-step C2/C3 denoising Gate。该 O1 已完成并正式失败，结果见
[Orbit-coordinate O1 最终结果](18_Orbit_Coordinate_O1结果.md)。冻结配置为：

- noise sigma：`0.15/0.5/1.0 Å`；
- batch size：4 molecules；
- evaluate steps：0/128/256/512；
- validation 重点：recovery improvement、coordinate/bond ratio、collision-free、correct PG
  vs swapped PG，以及 hard operation error `≤1e-5 Å`。

O1 只检验“从严格对称扰动恢复构象”，不等于从 prior 生成。其最终状态为
`FAIL_ORBIT_COORDINATE_O1_STOP_DENOISER`，因此不设计 learned orbit sampler、不降低 Gate、
不延长训练；hard-symmetry 表示保留给下一条 constrained force-field baseline。
