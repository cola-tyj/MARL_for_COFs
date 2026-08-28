# 项目文档索引

## 当前有效文档

- [项目静态上下文.md](项目静态上下文.md)：后续任务默认复用的项目背景、固定数据契约、
  技术分层、编码规范和测试规则；无需每次重读全部长文档。
- [项目动态状态.md](项目动态状态.md)：当前模型状态、下一动作和最新证据入口；发生代码、
  报错、测试、训练或环境变化时更新。
- [our_ET_Flow.md](our_ET_Flow.md)：当前 ET-Flow backbone、adapter 训练历史、实际启用的
  graph action / hard projection / F0.2、Gate、输入输出和代码文件地图。
- [UAE3D_Gate与指标定义.md](UAE3D_Gate与指标定义.md)：UAE-3D 确定性重构指标的
  计算口径、通用严格 Gate、分阶段 Gate 与状态字段语义。
- [数据集构建完整流程.md](数据集构建完整流程.md)：当前数据构建方法与依据 `final_dataset.csv` 重算的统计。
- [生成模型数据接口_v1_v2.md](生成模型数据接口_v1_v2.md)：冻结的 v1 canonical 数据、v2 symmetry annotation、
  模型 adapters、验证结果、已知限制和下一步工作。
- [生成模型训练规划.md](生成模型训练规划.md)：数据冻结后的生成模型选型、SemlaFlow/MiDi
  收尾结论、UAE-3D/UDM-3D 历史路线、graph+PG→3D 训练阶段与停止条件。当前版本化主线见
  [Coordinate Flow Matching 新计划与可行性评估](presentation/generative_model_progress_20260821/13_Coordinate_Flow_Matching计划与可行性.md)；UAE 的历史执行 Gate 见
  [generative_model/models/README.md](../generative_model/models/README.md)。
- [生成模型阶段汇报材料](presentation/generative_model_progress_20260821/README.md)：用于制作
  presentation/PPT 的数据摘要、模型结果、UAE 时间线、当前瓶颈、替代路线和外部数据命令。
- [core_doc.md](core_doc.md)：41 个原始 Core 的结构与文献资料。
- [sparse_reward_survey.md](sparse_reward_survey.md)：强化学习稀疏奖励的背景调研，不代表已实现结果。

当前唯一数据事实源是：

`cof_symmetry_pipeline/output/final_dataset.csv`

该文件当前包含 2,532 条唯一分子，全部具有可访问的 `XYZ_Path` 并满足当前点群接纳规则。45 条空路径记录保存在 `cof_symmetry_pipeline/output/missing_xyz_path.csv`，76 条点群规则不匹配记录保存在 `cof_symmetry_pipeline/output/rejected_pg_rule.csv`。

## 历史材料

旧生成模型、旧数据版本、阶段性计划和实验日志统一存放在 `history/`。这些文件保留研究过程，但文件中的“当前”“最终”和性能数字均不能直接作为项目现状引用。

分类和废弃文档清单见 [history/README.md](history/README.md)。
