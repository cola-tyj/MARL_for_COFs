# 历史材料分类

本目录只用于保存研究过程，不定义当前项目状态。

## 保留历史记录

| 文档 | 保留原因 |
|---|---|
| `interaction_log.md` | 完整的数据构建与决策时间线 |
| `training_log.md` | 各 MAPPO/生成实验的原始结果记录 |
| `current_status_and_problems.md` | 旧生成模型失败原因复盘 |
| `symmetric_graph_generation_7_10.md` | 规则生成和联合训练路线的完整失败过程 |
| `cof_symmetry_pipeline_design.md` | 1,661/2,752 数据阶段与 JT-VAE 设计记录 |
| `challenge_analysis_2026-06-25.md` | 当时的困难分析与预期方案 |
| `gap_analysis_2026-06-24.md` | 当时的阶段差距快照 |
| `de_novo_improvement_research.md` | 旧数据规模下的生成模型调研 |
| `implementation_plan.md` | 已停止的 diffusion + MARL 实施计划 |

## 删除过期结论

以下文档已从当前仓库删除，但仍可从 Git 清理前的 bundle 备份恢复：

| 文档 | 删除原因 |
|---|---|
| `paper_draft_zh.md` | 将失败的 symmetry-conditioned diffusion 写成已验证成果，并引用不可复现的准确率、合法性和 N₂ 结果 |
| `technical_documentation.md` | 同时把 2-Agent 与 3-Agent 称为当前实现，并将旧 checkpoint 和失败路线列为当前能力 |

## 引用规则

1. 数据规模、点群、来源和能量统计只引用 `../数据集构建完整流程.md`。
2. 训练数据只读取 `../../cof_symmetry_pipeline/output/final_dataset.csv`。
3. symmetry-conditioned diffusion、JT-VAE、Orbit-Circulant 和旧联合训练仅作为失败经验引用。
4. 历史 MAPPO 成功率必须同时注明实验 ID、环境、奖励来源以及是否使用 Mock predictor。
