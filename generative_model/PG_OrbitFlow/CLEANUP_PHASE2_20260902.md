# PG-OrbitFlow 第二阶段清理与目录重构记录

日期：2026-09-02

## 恢复点

- 全量训练结果提交：`49fb0a4`；
- Git标签：`pg-orbitflow-m5-baseline-20260902`；
- 冻结validation审计：`FAIL_DATASET_VALIDATION_STOP_BEFORE_DECODER`；
- 重构前PG-OrbitFlow测试：61/61通过。

## Python目录重构

将42个不属于current dataset toolchain、M3复现闭包、现有测试闭包，且不被任何冻结JSON直接绑定的
历史入口移入`legacy/`。文件没有删除；相对导入已修正为新的package层级。

- 顶层保留：53个正式运行时、当前审计、测试依赖和JSON绑定兼容模块；
- legacy：42个历史builder/audit/diagnose/summarize入口，另加`__init__.py`；
- legacy模块导入检查：42/42通过；
- 冻结JSON潜在失效源码引用：0；
- 完整回归：61/61通过。

重构前完整目录可通过Git标签恢复。历史命令的新模块前缀为
`generative_model.PG_OrbitFlow.legacy`。

## 全量训练中间checkpoint清理

保留以下正式产物：

- `last.pt`：独立validation报告直接引用的最终checkpoint；
- `step-020000.pt`：最终训练步的独立证据副本；
- `report.json`、`losses.npz`；
- `dataset_training_full_v1.console.log`；
- frozen protocol、cache manifest及独立validation report。

删除`step-000001.pt`及`step-001000.pt`至`step-019000.pt`，共20个中间checkpoint、
166,814,472 bytes（约159.1 MiB）。这些文件受`.gitignore`管理，不属于Git；删除后如需恢复只能按
冻结协议重新训练，但不会影响最终checkpoint推理、验证或结果复现。
