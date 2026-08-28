# PPT 页纲（建议 16 页）

## 1. 研究目标

标题：面向 COF 构筑单元的指定对称性 3D 分子生成

- 输入：目标点群 `C2 / C3 / S4 / D6h`。
- 输出：化学有效、三维无碰撞、满足目标点群 exact 或 compatible 的新分子。
- 长期用途：作为后续 COF 组装与多智能体优化的候选构筑单元。

建议图：`Target_PG → molecular graph → 3D coordinates → symmetry validation`。

## 2. 数据集构建已完成

- 2,532 个分子，2,532 个唯一 canonical isomeric SMILES。
- 91,113 个原子、96,555 条无向真实化学键。
- 显式 H；13 元素：H/C/N/O/F/B/P/S/Cl/Br/I/Si/Sn。
- 原子数 9–105，中位数 32；重原子数中位数 22。
- canonical v2、哈希 manifest、固定 IID/Core-OOD split 均已冻结。

建议图：元素分布、原子数箱线图、数据处理流程。

## 3. 对称性标注资产

- target 分布：C2 2,019；C3 461；S4 38；D6h 14。
- 3D 重新分析后 target exact：1,456/2,532（57.50%）。
- target compatible：2,532/2,532（100%）。
- 已保存 operation matrices、原子 permutation、orbit_id/orbit_size 和 symmetry error。
- target-operation RMS error：median `7.63e-6 Å`，p95 `0.0656 Å`。

建议图：target/actual PG Sankey 或堆叠条形图。

## 4. 模型无关数据与评测架构

- `cof_graphs.npz`：原子、键、坐标 canonical ground truth。
- `symmetry_data.npz`：与 molecule index 一一对应的点群操作和 orbit。
- sparse graph 供 EQGAT 类模型；full-pair adapter 供 UAE/MiDi。
- 统一评测：validity、connected、collision、bond exact、RMSD、novelty、actual PG。

## 5. 候选模型与筛选策略

- SemlaFlow：直接 3D flow matching。
- MiDi：原子/键/坐标联合离散-连续 diffusion。
- UAE-3D/UDM-3D：先近无损 autoencoder，再在 latent 中生成。
- 采用 `1 → 4 → 32 → dataset` 的诊断 Gate，避免直接投入长训。
- tier-1 是单分子记忆/接口诊断，不是最终训练集；三模型具体输入输出、checkpoint 和
  判定规则见 [08_三模型实验设计与判定口径.md](08_三模型实验设计与判定口径.md)。
- 既往三个模型都未输入 `Target_PG`：验证的是通用 3D 建模基座，不是指定点群生成。

## 6. SemlaFlow 结果

- 正面：官方 GEOM checkpoint 原域 32/32 sanitize、32/32 connected。
- COF tier-1 曾出现 2/5 raw 样本命中训练 SMILES。
- 失败：四阶段 checkpoint 合计仅 5/20 raw valid，键缺失、过价和芳香性错误不稳定。
- 结论：`FAIL_GATE2_SWITCH_TO_MIDI`，保留为历史基线。

## 7. MiDi 结果

- 官方 checkpoint + 近似 GEOM statistics：100 样本 sanitize 74、connected 94、collision 0。
- COF deterministic validation 平均 loss ratio `0.98482`，仅改善 14/32 cases、4/8 timesteps。
- 未达到预冻结 60% case / 75% timestep coverage。
- 结论：停止 loss-weight 搜索，不进入 tier-4。

## 8. UAE-3D：已经证明的能力

- 官方三头兼容 subset：2,242 条；UAE 官方词表不包含 Sn，且没有 charge/radical heads。
- tier-1 最终 atom/bond/chemistry exact，RMSD `0.04813 Å`，通过。
- tier-4 到 step 3,072：atom 4/4，bond/sanitize/connected 2/4，RMSD max `0.14220 Å`。
- post-GELU 特征的 existence ROC-AUC 约 `0.888`，说明特征并非完全失效。
- 这里的 checkpoint 输入是已知真实 graph+3D，输出是 atom/bond/coordinate reconstruction；
  它不是 `Target_PG → 新分子` 的生成 checkpoint。

## 9. UAE-3D：当前正式失败

- 新 explicit existence/type head 工程链路可复现，但 64-step 正式 pilot 失败。
- atom 4/4、conditional type `95.95%`、RMSD max `0.144109 Å`。
- none accuracy `61.39%`、present recall `70.95%`、wrong unique pairs 1,464。
- bond exact/sanitize/connected 均为 0/4。
- 结论：关闭当前 325 参数 head-only 分支；不延长、不调 threshold、不进入 tier-32。

## 10. 为什么会卡住

- 数据中的真实键非常稀疏，而 UAE/MiDi 使用 N² full-pair 分类；none 类占绝对多数。
- atom/coordinate 和 bond objective 竞争；连续 loss 改善不等于整分子 exact 改善。
- 4 分子 Gate 对“每一条 pair 都正确”极严格，且大分子一个错误就整分子失败。
- 2,532 条数据不足以从头学习通用化学，又存在 C2 主导和 S4/D6h 低支持。
- 当前瓶颈是任务建模，不是单纯缺训练步数。

## 11. 什么时候能真正微调

- 诊断性微调已经做过；2,242/2,532 条 UAE dataset-level fine-tuning 不再启动。
- 最终 Gate A 非线性 transfer AUC `0.88977 < 0.90`、macro `0.80934 < 0.82`，且
  tier-4 existence exact 仍为 0/4。
- 因此按预冻结规则停止 UAE，不执行 Gate B，也不启动 UDM。
- 新路线的微调准入改为 coordinate-only：known graph + Target_PG 的 32 分子 validation
  Gate 通过后，再进入 IID-train。
- UAE 的 C2/C3 特例也已排除：冻结 step-3008 在 24 C2 + 6 C3 IID-val 上 joint 为 0/30、
  mean RMSD 2.033 Å；tier-4 C2/C3 2/2 是训练面板记忆。

## 12. 推荐的新主线：2D → 3D → symmetry

1. 目标点群驱动拓扑模板/2D graph 生成，优先保证化学有效性。
2. graph-conditioned 3D conformer model 只预测坐标，不重新猜键。
3. 使用 operation/permutation/orbit 施加 hard symmetry 或 symmetry loss。
4. 对称约束 relaxation，最后统一 point-group analyzer 复验。

建议图：见 [02_数据资产与技术架构.md](02_数据资产与技术架构.md) 的流程图。

## 13. 外部数据如何使用

- GEOM：约 45 万分子、3,700 万构象，适合 3D conformer 预训练。
- QMugs：约 66.5 万分子、约 200 万构象，适合较大药物样分子域。
- QM9：约 13 万小分子，只适合工程 smoke/小分子预训练，元素和尺寸偏离明显。
- 外部数据没有本项目 target_pg 标签：只能先做通用预训练，再用 COF 数据条件微调。
- 不直接混合并伪造 PG 标签；所有外部高对称样本必须用同一 protocol 重算。

## 14. 决策与近期里程碑

- 决策 A：UAE Gate A 已失败；不做 Gate B、不再延长 decoder 分支。
- 决策 B：`known graph + Target_PG → 3D` 已成为执行主线；G0 严格数据接口已通过全
  2,532 条审计；G1 hard projection 已完成，证明其适合作为约束层但不能替代坐标模型。
- 决策 C：该坐标模型通过后，再接 2D graph/template 生成，不同时重猜键和坐标。
- G2 首个 4 层 sparse-bond EGNN 已完成：8/8 固定实例改善且 symmetry/collision/compatible
  全过，但 step 2,048 raw/projected RMSD `0.19252/0.12663 Å` 未过 `0.08/0.05 Å`。
- G2.1 朴素 dynamic full-pair 在 step 1,024 为 `0.22347/0.15233 Å`，弱于 sparse 同 step 结果；
  说明等权加入所有非键 pair 会稀释局部真实键消息。
- G2.2 已分离 bonded/local/global 通道并加入 distance/angle objective；step 1,024 达到
  `0.18571/0.12376 Å`，优于前两架构但仍未过 `0.08/0.05 Å`，C2/C3 tier-4 未放行。
- 成熟 backbone fast32 已完成：Semla/EQGAT 在独立 C2/C3 validation 仅改善 7/32、5/32；
  Semla 更好但二者均不适合作为单步 absolute-x0 回归器，停止继续筛 backbone。
- Fast-32 flow 最终失败：高噪声 recovery ratio `1.037`、collision-free `1/16`、raw analyzer
  `0/16`、correct PG 优于 swapped `6/16`；按预注册规则停止 endpoint branch。
- 新里程碑：orbit-representative 硬对称接口已覆盖 2,279 条 IID train/validation，零严格
  失败，最大 operation error `6.11e-7 Å`，平均只需预测 46.98% 的原子坐标。
- orbit-coordinate O0 已通过：等变 atom proposals → orbit pooling → hard expansion；真实
  C2/C3/Sn 工程 Gate 全过，最大 operation error `6.03e-7 Å`。
- O1 512-step 已失败：overall/low-noise ratio `0.85997/1.01871`，collision-free `0.64583`；
  虽然 bond ratio `0.67691`、hard symmetry 保持，但不进入 learned sampler。
- orbit force-field 已取得首个后端质量通过：UFF 全数据覆盖 `2532/2532`；唯一 F0.1 在
  48 cases 上 bond ratio `0.02743`、collision-free `44/48`、operation error `2.89e-7 Å`，
  10/10 Gate checks 通过。
- 独立 pymatgen audit 为 44/48 compatible；4 个 package-1490 碰撞 case 为 C1。冻结 F0.2
  非键短程排斥已将这 4 个案例修复为无碰撞且 4/4 C2 exact/compatible，现已放行 ETKDG
  initial-conformer/atom-index/orbit-mapping interface。
- ETKDG interface 已通过 E1 和 dataset-scale E2：E2 为 2,232/2,232 geometry、
  8,928/8,928 candidates、collision/bond-quality 100%；独立 analyzer/compatible
  2,223/2,232（99.60%），两条 Sn 均通过。
- E3-A 已从 graph-only 输入为 2,232/2,232 分子恢复合法 C2/C3 action；E3-B recovered-action
  stress handoff 为 16/16 independent compatible。known graph 已不再依赖预存 action。
- ET-Flow + E3/F0.2 四点群交付已完成：C2/C3 固定 IID-test 64/64，canonical 全部
  S4/D6h 52/52，geometry 与独立 compatible 均 100%；单分子 CLI 输出 `final.xyz`。
- 稀有点群 v1/v2/v3 失败史保留：v1 单 action 为 21/52；v2 geometry 52/52 但独立
  compatible 43/52；v3 仅增加 reference-free S4 惯性 guard 后达到 52/52。
- E3-C `Target_PG → novel graph` 现在是最终主缺口。ET-Flow 在交付路线中只替换 ETKDG
  作为 learned initial-conformer prior，最终对称性由 hard projection/F0.2 保证，不能与
  raw learned PG conditioning 混写。
- 当前不能把 F0.1 表述为从零生成新分子，也不能用 stored-operation error 单独替代 actual PG。
- 最终 `Target_PG→新 graph+3D` 仍缺 2D generator；已有 graph 的 automorphism/orbit mapping
  已由 E3-A 关闭。
- “已知 SMILES/graph 上生成满足目标 PG 的 3D”近期标准已经完成；下一阶段开放新
  SMILES/graph 生成并接入冻结 3D 后端。
- 主报告以 C2/C3 固定测试和 S4/D6h canonical 全覆盖共同构成四目标点群证据；D6h 的
  quantitative conclusion 仍需标注 low-support（仅 14 个 canonical 样本）。

## 15. Our ET-Flow 的贡献分解与未见图

- 四路线 paired ablation：ETKDG/raw/hard/full 的 compatible 为 `9/32、6/32、30/32、32/32`。
- hard projection 单独使 collision-free 从 raw `32/32` 降为 `27/32`，F0.2 恢复到 `32/32`。
- fresh Core-OOD 完整路线 `32/32 compatible、32/32 collision-free`，train graph/Core/exact
  fingerprint 均无泄漏。
- 外部 COF-v2 面板 C2/C3/D6h 为 `10/10 compatible`；两个 S4 高对称图暴露 inertia guard 限制。

建议图：四路线 compatible/collision 双柱状图；Core-OOD 最近邻 Tanimoto 分布。
现成图：[paired_ablation_overview.png](assets/our_etflow_eval_v1/paired_ablation_overview.png)、
[point_group_success_rates.png](assets/our_etflow_eval_v1/point_group_success_rates.png)。

## 16. 条件可控性与物理稳定性

- 相同 D6h graph、seed、raw tensor 分别请求 C2/C3/D6h，三类均 `8/8 compatible`。
- C2↔C3 强响应只有 `5/8`，3 条坍缩到共同高对称超群；正式 Gate 失败。
- 32 条 GFN2-xTB 无约束松弛已完成：计算/收敛、能量下降、collision-free 均 `32/32`，
  中位 RMSD `0.07155 Å`，松弛后 compatible `31/32`。
- C2/C3/S4 均保持 `8/8`；D6h 为 `7/8`，`cof_000858` 由 D6h 降为 D3h。
- 当前结论：known graph 的 compatible 生成和总体物理稳定性 pilot 已成立；强 C2/C3
  target-specific 控制与逐分子量子化学对称稳定仍未完全解决。

建议图：同图 C2/C3/D6h 三构象示例；xTB 松弛前后能量—PG 保持散点图。
现成图：[xyz_examples_overview.png](assets/our_etflow_eval_v1/xyz_examples_overview.png)、
[xtb_relaxation_overview.png](assets/our_etflow_eval_v1/xtb_relaxation_overview.png)、
[xtb_xyz_before_after.png](assets/our_etflow_eval_v1/xtb_xyz_before_after.png)。
