# Legacy experiment entry points

本目录保存M0--M4、H1--H3及早期Cartesian/IC消融的独立builder、audit、diagnose和summarize入口。
这些文件不属于当前dataset-level训练或validation运行时，但为了复现实验过程而保留。

## 使用约定

- 模块路径由`generative_model.PG_OrbitFlow.<name>`改为
  `generative_model.PG_OrbitFlow.legacy.<name>`；
- 历史JSON/report绑定的6个入口仍留在上级目录，避免破坏已冻结的路径和SHA-256；
- 2026-09-02重构前的完整目录可通过Git标签
  `pg-orbitflow-m5-baseline-20260902`恢复；
- 不应从当前正式trainer导入本目录模块；legacy入口可以导入上级稳定模块。

## 分类

- `build_*.py`：冻结历史protocol JSON的一次性构建器；
- `audit_*.py`：历史checkpoint、panel、representation与复现审计；
- `diagnose_*.py`：失败分支的只读诊断；
- `summarize_*.py`：历史实验汇总入口；
- `evaluate_m1_checkpoint.py`：M1阶段的旧checkpoint评估入口。

本目录的目标是降低顶层入口噪声，不代表删除或否定旧实验结论。
