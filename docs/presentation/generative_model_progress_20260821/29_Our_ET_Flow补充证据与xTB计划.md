# Our ET-Flow 补充证据与 GFN2-xTB 计划

## 1. 为什么需要补充实验

116 条旧审计证明了冻结 composite route 能把 canonical graph 转成目标点群 compatible XYZ，
但没有拆开 ET-Flow、hard projection 与 F0.2 的贡献，也没有充分回答未见 graph、同图多条件响应
和离开硬约束后的物理稳定性。因此追加四类证据，并保持所有失败样本在完整分母中。

## 2. 四路线 paired ablation

相同 graph、seed、Target_PG；三个 ET-Flow 分支共享完全相同的 4 个 raw coordinates。

| route | PG-compatible | collision-free | mean bond MAE (Å) | median UFF/atom |
|---|---:|---:|---:|---:|
| ETKDGv3 | 9/32 | 29/32 | 0.02575 | 2.8749 |
| ET-Flow raw | 6/32 | 32/32 | 0.02349 | 3.5696 |
| raw + hard projection | 30/32 | 27/32 | 0.11777 | 13.4112 |
| raw + projection + F0.2 | 32/32 | 32/32 | 0.01770 | 1.6460 |

结论：ET-Flow 提供 learned conformer prior；hard projection 提供目标对称，但会制造碰撞和
高能键长；F0.2 负责在对称子空间内恢复几何质量。三个模块缺一不可，不能把完整路线的结果
归因于 raw learned PG conditioning。

## 3. 未见图证据

### Core-OOD

fresh 16 C2 + 16 C3，与训练集 canonical SMILES/Core/exact Morgan fingerprint 零重叠，最近
训练 Tanimoto 为 `0.1667–0.8235`。完整路线 `32/32 compatible、32/32 collision-free`，
mean bond MAE `0.01305 Å`；raw 仅 `6/32 compatible`。

### 外部 COF v2

12 条分子与 COF v2 在 canonical SMILES、Bemis–Murcko scaffold、exact Morgan fingerprint
上零重叠。完整路线在 C2/C3/D6h 为 `10/10 compatible`；cubane、adamantane 两个 S4 被
现有 inertia guard 严格拒绝。该结果说明 v5 可以迁移到部分外部 graph，也明确暴露高对称
Td/Oh 结构的 S4 guard 泛化问题。由于缺少官方 GEOM train 清单，不声称这些分子未被官方
checkpoint 见过。

## 4. 同图多 Target_PG

8 个 D6h graph 共享同一 raw tensor，分别请求 C2/C3/D6h。三类均为 `8/8 compatible`，但
C2↔C3 仅 `5/8` 达到 `Kabsch RMSD > 0.05 Å`；3 个 graph 的两种请求收敛到共同 D3d/D6
超群。失败 package 为 `2529、860、2528`，两种输出 RMSD 仅
`0.000283、0.000904、0.003053 Å`。正式 Gate 失败，结论是“子群 compatibility 可控”，
不是“强 target-specific 形状可控”。

## 5. GFN2-xTB 物理稳定性实验

冻结 32 个 F0.2 输出，C2/C3/S4/D6h 各 8；用 `tblite==0.7.0`、GFN2-xTB、ASE LBFGS 做
无约束松弛，`fmax=0.1 eV/Å`、最多 100 步。松弛期间不施加 hard projection/orbit constraint。

协议 SHA-256：`33fd9d65ee2cdbc88860bb8ed639011151a7db6cd63ca3c165113e2e9573fde9`。

正式结果：计算和 LBFGS 收敛均 `32/32`，32 条能量全部下降，松弛后无碰撞 `32/32`；
中位 pre/post RMSD `0.07155 Å`，松弛后 compatible `31/32`、exact `20/32`。分组 compatible：
C2 `8/8`、C3 `8/8`、S4 `8/8`、D6h `7/8`。

唯一失败 `cof_000858` 为 D6h→D3h：能量下降 `65.6451 eV`、RMSD `0.85489 Å`、bond MAE
`0.29403 Å`，且没有碰撞。这表明它不是数值坏点，而是无约束量子化学松弛偏向更低对称结构。
总体 Gate 正式通过，但 D6h 稳定性必须报告为 `87.5%`，不能写成逐分子 100%。

## 6. 当前可声明与不可声明

可以声明：冻结 composite backend 在 32 条 paired ablation 和 32 条 Core-OOD 上达到 100%
PG-compatible/collision-free，并且贡献可由消融分解。

不可声明：raw ET-Flow 已学会 Target_PG conditioning；所有外部分子均未见于 GEOM；同图 C2/C3
一定生成不同形状；所有 F0.2 输出都能在无约束量子化学松弛后保持原目标点群。

## 7. 已生成的 PPT/可视化资产

以下资产均由冻结 JSON/NPZ 重建，成功率保留完整失败分母：

- [四路线消融总览](assets/our_etflow_eval_v1/paired_ablation_overview.png)
- [各目标点群成功率](assets/our_etflow_eval_v1/point_group_success_rates.png)
- [能量—actual symmetry error](assets/our_etflow_eval_v1/energy_vs_symmetry_error.png)
- [运行时间—联合成功率](assets/our_etflow_eval_v1/runtime_vs_joint_success.png)
- [典型成功/失败 XYZ 六联图](assets/our_etflow_eval_v1/xyz_examples_overview.png)
- [综合统计 CSV](assets/our_etflow_eval_v1/evaluation_summary.csv)
- [12 个独立 XYZ 与完整哈希清单](assets/our_etflow_eval_v1/figure_manifest.json)
- [GFN2-xTB 松弛总览](assets/our_etflow_eval_v1/xtb_relaxation_overview.png)
- [xTB 能量变化—对称误差](assets/our_etflow_eval_v1/xtb_energy_change_vs_symmetry_error.png)
- [D6h 稳定/降对称 XYZ 对照](assets/our_etflow_eval_v1/xtb_xyz_before_after.png)
- [32 条 xTB 明细 CSV](assets/our_etflow_eval_v1/xtb_relaxation_records.csv)

manifest SHA-256：`104f679c45cc7d08c163f26cb513821da4f888e96dd4fac5c6f896bc17d27145`。
PNG 供直接插入 PPT，SVG 供矢量排版。

## 8. 总完成审计

机器可读审计位于
`generative_model/evaluation/our_etflow_supplement_completion_v1.json`，SHA-256 为
`985d238ef73a329ff6437c018bd6709b4e1f7a3230ed7b2a792a7c0eb2dc0752`。其六项完成检查均为
true，并保留三个重要负结论：C2↔C3 条件 Gate 失败、外部 S4 0/2、GFN2-xTB 后一条
D6h→D3h。这里的总 PASS 表示“用户要求的实验和交付物已全部完成且证据可复核”，不表示
所有科学指标均达到 100%。
