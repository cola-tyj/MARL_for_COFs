# 生成模型阶段汇报材料（2026-08-21）

本目录用于制作阶段性 presentation/PPT。内容从项目机器报告和当前正式文档提炼，不替代
原始实验报告，也不把 `PASS_EXECUTION` 当作模型质量通过。

## 一句话结论

数据资产已经完成并可复现；SemlaFlow、MiDi、UAE-3D head-only 分支以及最终 UAE Gate A
均未通过 COF 迁移准入。项目不是卡在环境或代码，而是卡在“同时正确生成稀疏化学键和
3D 坐标”的任务耦合。UAE Gate B 已取消，执行主线已切换为 **已知 graph + Target_PG →
3D → orbit-based 对称性约束**，随后再接 2D/拓扑生成。最新 Fast-32 flow 质量 Gate 已失败；
orbit-representative 硬对称表示接口已在 2,279 条 IID train/validation 上通过；但后续 O1
learned denoiser 因低噪声退化和碰撞正式失败。项目随后转入 orbit-constrained force-field
baseline，不再延长 learned denoiser。该 baseline 的 F0.1 已通过；独立点群复核定位的 4 个
碰撞/C1 case 也已由 F0.2 有界非键排斥修复。ETKDG initial-conformer interface 已继续通过
E0 4 分子和 E1 32 分子 Gate；E1 为 128/128 embedding、32/32 无碰撞且 independent
compatible。dataset-scale C2/C3 E2 也已正式通过：2,232/2,232 几何成功，独立 analyzer/
compatible 为 2,223/2,232（99.60%），两条 Sn 均通过。E3-A 随后已从 graph-only 输入为
2,232/2,232 分子恢复合法 C2/C3 action；E3-B 将 recovered action 接入 E2，在 16 分子
stress panel 上达到 16/16 independent PG-compatible。当前 known-graph 链路已经不再依赖
预存 orbit action；下一瓶颈明确为 `Target_PG → novel valid 2D graph`。作为独立的构象先验
对照，新建 EF0/EF1 ET-Flow 路线；EF0-A 离线严格桥接和 EF0-B 官方 checkpoint 预检均已
通过。32 条 IID-test EF1 也已通过零样本 Gate：ET-Flow/ETKDG final compatible 均为
32/32，exact 分别为 20/32 和 21/32，经相同 F0.2 后 final 几何几乎一致。因此
ET-Flow 获得有界微调评估资格，但未证明显著优于 ETKDG，也不改变 E3-C 主缺口。
PG-conditioned ET-Flow 已随后实现：Target_PG embedding、operation/permutation/orbit
条件、soft restoring velocity 和 raw-endpoint symmetry loss 均已接入；官方 checkpoint
EF2-0 工程 Gate 10/10 通过。当前是「训练器已就绪」，不是「创新模型已训练成功」。

## 文件索引

- [01_PPT页纲.md](01_PPT页纲.md)：建议 14 页主汇报结构和每页要点。
- [02_数据资产与技术架构.md](02_数据资产与技术架构.md)：数据、接口、评测和目标架构。
- [03_模型进展与结果总表.md](03_模型进展与结果总表.md)：SemlaFlow、MiDi、UAE-3D 的结论。
- [04_当前瓶颈与微调准入.md](04_当前瓶颈与微调准入.md)：为什么卡住、何时可以开始数据集级微调。
- [05_替代路线与推荐方案.md](05_替代路线与推荐方案.md)：联合大数据训练和 2D→3D 方案。
- [06_外部数据与待执行命令.md](06_外部数据与待执行命令.md)：GEOM/QMugs/QM9 的定位与仅供用户执行的命令。
- [07_参考资料.md](07_参考资料.md)：论文、官方代码和数据来源。
- [08_三模型实验设计与判定口径.md](08_三模型实验设计与判定口径.md)：逐模型说明
  tier-1/4/32、checkpoint 输入输出、训练目标以及 passed/failed 的计算依据。
- [09_Graph_PG_3D_G1结果与G2设计.md](09_Graph_PG_3D_G1结果与G2设计.md)：新主线 G1 的
  冻结 hard-projection Gate、失败原因和 G2 coordinate denoiser 设计。
- [10_Graph_PG_3D_G2_Tier1结果.md](10_Graph_PG_3D_G2_Tier1结果.md)：G2 首个 4 层
  sparse-bond EGNN、G2.1 full-pair 和 G2.2 three-channel 的冻结 Gate、结果与停止结论。
- [11_UAE_C2_C3只读验证.md](11_UAE_C2_C3只读验证.md)：冻结 step-3008 在独立
  IID-validation C2/C3 面板上的重构、尺寸分层、点群复算与最终反证。
- [12_成熟Backbone_Fast32结果.md](12_成熟Backbone_Fast32结果.md)：Semla EquiInv 与
  EQGAT 在固定 32/8 C2/C3 面板上的数据集级快验、正式失败和 flow-matching 转向。
- [13_Coordinate_Flow_Matching计划与可行性.md](13_Coordinate_Flow_Matching计划与可行性.md)：
  当前版本化主计划、数学接口、阶段 Gate、可行性、优缺点与停止条件。
- [14_Coordinate_Flow_F0结果.md](14_Coordinate_Flow_F0结果.md)：flow contract、显式时间
  Semla adapter、13 项工程测试、机器报告与 F1 准入边界。
- [15_Coordinate_Flow_Fast32冻结协议.md](15_Coordinate_Flow_Fast32冻结协议.md)：固定 32/8
  面板、flow/training 配置、双层评估、预注册 Gate、环境与三种停止结论。
- [16_Fast32_Flow结果与Orbit硬对称接口.md](16_Fast32_Flow结果与Orbit硬对称接口.md)：正式
  Fast-32 flow 失败、停止理由，以及覆盖 2,279 条 IID train/validation 的可微 orbit
  representative 硬对称接口与下一阶段边界。
- [17_Orbit_Coordinate_O0工程结果.md](17_Orbit_Coordinate_O0工程结果.md)：群轴审计、等变
  atom proposal → orbit pooling → hard expansion 架构、冻结 O0 工程 Gate 与 O1 准入边界。
- [18_Orbit_Coordinate_O1结果.md](18_Orbit_Coordinate_O1结果.md)：唯一一次 512-step C2/C3
  hard-symmetry denoising、分噪声/碰撞结果、冻结失败结论和 force-field 路线切换。
- [19_Orbit_Force_Field_F0_F01结果.md](19_Orbit_Force_Field_F0_F01结果.md)：全数据
  UFF/MMFF 覆盖、F0 数值收敛失败、唯一 F0.1 修复及首个通过的严格对称 3D 后端 Gate。
- [20_F01坐标归档与点群独立审计.md](20_F01坐标归档与点群独立审计.md)：确定性 NPZ、
  env_cof/pymatgen actual-PG 复核、4 个 package-1490 碰撞/C1 失败及 ETKDG 前修复边界。
- [21_F02碰撞修复结果.md](21_F02碰撞修复结果.md)：冻结非键短程排斥、4 个失败案例的几何
  修复、独立 C2 exact/compatible 复验及 ETKDG 准入结论。
- [22_ETKDG_Orbit初始构象接口结果.md](22_ETKDG_Orbit初始构象接口结果.md)：reference-free
  ETKDG 多初始构象、严格 atom/orbit action、显式 Lewis 共振语义、E0/E1 结果与全量准入。
- [23_ETKDG_Orbit_E2运行协议.md](23_ETKDG_Orbit_E2运行协议.md)：2,232 分子 E2 面板、可断点
  dataset runner、独立 point-group auditor、冻结 Gate、正式 tmux 命令与输出语义。
- [24_Graph_Action_E3A_E3B结果.md](24_Graph_Action_E3A_E3B结果.md)：graph-only C2/C3
  automorphism/action recovery、2,232 条全量审计、recovered-action→3D stress Gate 与 E3-C
  新 2D graph 边界。
- [25_ETFlow_ZeroShot与Symmetry_Projection.md](25_ETFlow_ZeroShot与Symmetry_Projection.md)：
  ET-Flow 独立构象先验、strict explicit-H/Sn bridge、EF0-A 结果、EF0-B 命令和 EF1 对照设计。
- [26_ETFlow_EF1结果与微调评估.md](26_ETFlow_EF1结果与微调评估.md)：EF1 零样本对照结果、
  actual-PG 独立审计、可重复 hash、有界微调准入和 EF2 分段边界。
- [27_PG_Conditioned_ETFlow_EF2实现.md](27_PG_Conditioned_ETFlow_EF2实现.md)：创新模型数学接口、
  strict data adapter、EF2-0 工程 Gate、v1/v2 审计边界和已冻结的 256-step EF2-A 命令。
- [28_ETFlow_E3F02可交付推理接口.md](28_ETFlow_E3F02可交付推理接口.md)：不用 ETKDG/reference
  XYZ 的 known-graph + Target-PG → 3D 交付接口与四目标点群审计。
- [29_Our_ET_Flow补充证据与xTB计划.md](29_Our_ET_Flow补充证据与xTB计划.md)：四路线消融、
  Core-OOD、外部分子、同图多 PG 可控性，以及 GFN2-xTB 无约束松弛协议。
- [model_results.csv](model_results.csv)：可直接导入 PPT/Excel 的模型结果表。
- [uae3d_milestones.csv](uae3d_milestones.csv)：UAE-3D 关键 checkpoint 轨迹。

## 汇报时必须保留的边界

- 结论只针对本项目 COF 数据迁移，不否定模型在 QM9/GEOM 官方域的能力。
- `target_pg` 是设计目标；`actual_pg` 是 XYZ 重算结果；二者不能混写。
- `pg_compatible=true` 表示目标点群是实际点群的 subgroup，不等于 exact match。
- D6h 只有 14 条，S4 只有 38 条，只作为探索性结果。
- 当前没有任何模型获准宣称“能够从目标点群生成新的 2D graph/3D 分子”；但已知 graph +
  Target_PG 的确定性 C2/C3 action→3D baseline 已通过有界 Gate。
- 现有数据和结果不支持用 SMILES 单独作为点群真值；点群必须在 3D 坐标上复算。
