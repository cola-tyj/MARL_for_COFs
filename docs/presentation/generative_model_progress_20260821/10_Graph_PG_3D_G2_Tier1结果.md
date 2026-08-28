# Graph + Target_PG → 3D：G2 tier-1 结果

## 1. 本阶段到底训练了什么

G2 首轮不是无条件生成，也不是从 `Target_PG` 直接生成新分子。它是一个单分子、已知图的
坐标去噪 Gate：

```text
输入 checkpoint
  canonical atom types / formal charges / radicals
  + immutable sparse true bonds and bond types
  + noisy centered all-atom coordinates
  + noise sigma
  + Target_PG index
        ↓
4-layer E(3)-equivariant sparse-bond denoiser（146,884 参数）
        ↓
输出 checkpoint
  centered clean-coordinate prediction only
        ↓（只在评估阶段）
G1 Reynolds hard projection → fixed point-group analyzer
```

模型不预测 atom、bond、SMILES 或分子大小。真实 graph 全程不变，显式 H 保留；所以本阶段
衡量的是“给定合法 graph 后能否恢复 3D”，不能称为新分子生成。

## 2. 工程测试

- finite forward/backward：通过；
- 旋转等变：通过；输入整体平移被质心归零消除；
- 原子重排等变：通过；
- continuous training 与 checkpoint-resume 参数/optimizer 完全一致：通过；
- 训练和评估期间 known-graph fingerprint 不变：通过。

模型使用 4 层 sparse true-bond message passing，坐标更新为相对向量乘学习得到的不变量标量。
训练目标为 coordinate MSE + `0.25 ×` true-bond length MSE + `0.1 ×` soft symmetry MSE。
训练噪声 `σ∈{0.05,0.10,0.15,0.20,0.30} Å` 循环；固定验证使用 8 个预注册 seed 和
`σ=0.15 Å`。test/Core-OOD 未使用。

## 3. 冻结 tier-1 Gate

样本固定为 IID-validation package index `2162`、`C2`、31 原子/33 键；图直径为 14。
质量 Gate 在训练前冻结：

| 检查 | step-512 门槛 |
|---|---:|
| 8 个 raw RMSD 全改善 | 8/8 |
| 8 个 projected RMSD 全改善 | 8/8 |
| projected collision-free | 8/8 |
| fixed analyzer compatible | 8/8 |
| mean raw RMSD | ≤0.08 Å |
| mean projected RMSD | ≤0.05 Å |
| max projected operation RMS | ≤1e-5 Å |
| max projected atom symmetry error | ≤1e-4 Å |

任何必要项失败都不能进入 tier-4。`PG exact` 仅描述，不把 actual supergroup（这里为 C2v）
误判为失败；正式条件是 target C2 为 actual PG 的 subgroup，即 compatible。

## 4. step-512 正式结果

| 指标 | step 0 | step 512 | Gate |
|---|---:|---:|---:|
| mean raw RMSD | 0.24781 Å | 0.21130 Å | ≤0.08 Å，失败 |
| mean projected RMSD | 0.16666 Å | 0.14660 Å | ≤0.05 Å，失败 |
| raw / projected improved | — | 8/8 / 8/8 | 通过 |
| collision-free / compatible | — | 8/8 / 8/8 | 通过 |
| max operation / atom symmetry error | — | 8.29e-8 / 1.34e-7 Å | 通过 |

正式状态：`FAIL_GRAPH_PG_3D_G2_TIER1_STOP_AND_DIAGNOSE`。只有两个绝对 RMSD 检查失败；
这是正向学习信号，但不是质量通过。

## 5. 唯一一次预注册续训

为区分“512 steps 不够”和“架构感受野不足”，在运行前绑定 step-512 checkpoint SHA，
保持模型、loss、noise schedule、验证 seed、analyzer 和 `0.08/0.05 Å` Gate 全部不变，只把
总步数扩到 2,048。预注册停止规则是：若仍失败，不再延长此 4 层架构。

| step | mean raw RMSD | mean projected RMSD |
|---:|---:|---:|
| 512 | 0.21130 Å | 0.14660 Å |
| 768 | 0.21264 Å | 0.14438 Å |
| 1,024 | 0.20373 Å | 0.13823 Å |
| 1,280 | 0.19903 Å | 0.13537 Å |
| 1,536 | 0.20425 Å | 0.13628 Å |
| 1,792 | 0.19445 Å | 0.13015 Å |
| 2,048 | 0.19252 Å | 0.12663 Å |

step 2,048 仍为 8/8 raw/projected 改善、8/8 collision-free/compatible，symmetry error 继续
约为 `1e-7 Å`；但两个 RMSD 阈值仍失败。正式状态：

`FAIL_GRAPH_PG_3D_G2_TIER1_CONTINUATION_STOP_ARCHITECTURE`

## 6. 结论与下一模型边界

不能把当前失败归因于“再多训练一点就会过”。2,048 steps 后仍有明显平台，而 4 层 sparse
message passing 面对直径 14 的图无法让远距离原子充分交换几何信息。下一模型必须改变
感受野，而不是继续同架构：

- canonical sparse graph 仍是不可变 ground truth；
- adapter 可动态增加 non-bonded/full-pair 或受控 radius/k-hop 几何消息；
- 增加 all-pair distance/局部 angle 等几何目标，但不重新预测键；
- 先重新通过 E(3)/排列/checkpoint 工程测试，再注册新的 tier-1 Gate；
- 只有新模型 tier-4、tier-32 validation 通过，才进入 IID-train dataset-level 训练。

因此当前仍未获得“指定点群 3D 生成模型”，但已经获得一个可复用的严格数据接口、hard
symmetry constraint，以及一个明确排除的 4 层 sparse-only baseline。

## 7. 证据与哈希

- 首轮 protocol：`3f5971e3336b313a15dd2967fa005738c233084a1468a3862aa9481508092aea`
- step-512 report：`37b7847c37760b12bef04280fcf31c5a06bcfc433290fbb600db0e8589dc1eab`
- 失败诊断：`8ce05e16aba7a66e117322ebe475fbf1e349313543efee2d79ff395edab6d7ba`
- 续训 protocol：`93e67a2e27b0f146971b973110b2979b52a0982ed36415c64a4497d6c6a40319`
- step-2048 report：`8d064ce0668598e60808694ac87210e15f4e83898a38afa480c733aedfd0d140`
- step-2048 checkpoint：`7b7105631eeefb1d9e41ad49b971567f8bcbd0f4c29a3c67387cd1ad05593768`

## 8. G2.1 动态 full-pair 单变量结果

随后仅改变消息边：参数量、4 层、loss、noise、验证 seed 和 Gate 均不变；模型内部从 66 条
有向真实键消息扩展为 31 原子的 930 条有向非自环 pair，真实键保留 1–4 类型，其余严格
标为 `none=0`。canonical sparse graph 没有被改写。

G2.1 的 finite backward、E(3)、节点排列和 exact-resume 工程测试全部通过，但 step 1,024
的 raw/projected mean RMSD 为 `0.22347/0.15233 Å`；同一步 sparse G2 已达到
`0.20373/0.13823 Å`。因此朴素 full-pair 等权聚合不但未修复，反而稀释了 bonded/local
geometry signal。正式状态为 `FAIL_GRAPH_PG_3D_G21_TIER1_STOP_AND_DIAGNOSE`，不续训。

下一版本不能只“加更多边”，而应分离 bonded/local/nonbonded 通道，采用分通道归一化或
层级聚合，并加入 all-pair distance/angle 等显式几何目标。G2.1 protocol/report SHA-256：
`f07133363e177ecd2c9b9517199d7f20d7215f53f44bf7c4814d336abf411c8b` /
`081000d3289de52497045aec838d2b0fa81a31b7461608057d5909bd2328e779`。

## 9. G2.2 三通道 C2 tier-1 结果

G2.2 落实了上述结构性修改，但仍只回答“已知 graph + C2 能否恢复 3D”，不是无条件生成：

```text
immutable canonical graph + noisy XYZ + sigma + Target_PG
  ├─ bonded channel
  ├─ local nonbonded channel（当前距离 ≤4.5 Å）
  └─ global nonbonded channel
       ↓ 三通道独立聚合/归一化，6 层、1,145,970 参数
centered XYZ prediction
       ↓ evaluation-only Reynolds projection + fixed analyzer
```

训练目标为 coordinate MSE + `0.25×` bond-length MSE + `0.25×` all-pair-distance MSE +
`0.1×` local bonded-angle cosine MSE + `0.1×` soft-symmetry MSE。模型与 trainer source hash、
1024 steps、固定 seed、checkpoint 间隔和原绝对 Gate 均在运行前冻结。finite backward、E(3)
旋转/平移、节点排列和 checkpoint exact-resume 全部通过。

| 架构（同一 C2、step 1,024） | 参数 | raw RMSD | projected RMSD |
|---|---:|---:|---:|
| G2 sparse-only | 146,884 | 0.20373 Å | 0.13823 Å |
| G2.1 naive full-pair | 146,884 | 0.22347 Å | 0.15233 Å |
| G2.2 three-channel | 1,145,970 | **0.18571 Å** | **0.12376 Å** |

G2.2 相对两个旧架构有确定改善；8/8 raw/projected RMSD 改善、8/8 collision-free、8/8
C2-compatible，projected symmetry error 约 `1e-7 Å`。但 `0.18571/0.12376 Å` 仍没有达到
预冻结 `0.08/0.05 Å`，正式状态为：

`FAIL_GRAPH_PG_3D_G22_C2_TIER1_STOP_AND_DIAGNOSE`

因此“分通道 + 显式几何监督”是正向架构证据，不是模型准入。C2/C3 tier-4 没有运行；也不
允许通过降低阈值、挑 seed 或直接延长训练把本次失败改写为通过。下一步只能先做有界诊断，
再预注册新的多尺度/attention 或多步 diffusion denoising 方案。

- G2.2 protocol SHA-256：`96f38e773ca91d95c2f01b1c594b253c376f70fb8f044ea43ad21c659abc14c4`
- G2.2 report SHA-256：`9866cd01ccaa5e3c82a23f1f5935768eab2e85e1fa3cee88ab9d73d769a866c0`
- last checkpoint SHA-256：`0815f055a5dbd7b35df1dbb273c0678fa519273a1fecc11e5b3f61954f2f83cf`
