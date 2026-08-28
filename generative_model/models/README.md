# 历史生成模型：最小可用保留集

本目录只保留 SemlaFlow、MiDi 和 UAE-3D 三条已关闭路线的最终可用接口，以及 Our ET-Flow
v5 直接依赖的通用 projection/symmetry 模块。历史调参脚本、重复 checkpoint、中间采样文件
和失败分支已于 2026-08-28 清理；科学结论继续保存在
`generative_model/smoke/reports/`、本目录 `reports/` 和 `docs/` 中。

当前生产主线不是这三个模型，而是
[Our ET-Flow](../../docs/our_ET_Flow.md)。canonical v2 数据、Our ET-Flow、最终消融、
Core-OOD、外部集、可控性和 GFN2-xTB 产物均未参与本次清理。

## 最终结论

| 模型 | 最终状态 | 保留原因 |
|---|---|---|
| SemlaFlow | `FAIL_GATE2_SWITCH_TO_MIDI` | 保留官方接口与随机 forward smoke，作为联合 flow baseline |
| MiDi | `FAIL_BOUNDED_SNR_DETERMINISTIC_GATE_STOP_BEFORE_SAMPLING` | 保留数据桥接与随机 forward smoke，作为 graph diffusion baseline |
| UAE-3D | `FAIL_GATE_A_STOP_UAE_BRANCH` | 保留最终 Gate A/C2-C3 只读诊断及其唯一 step-3008 权重 |

这些 FAIL 只针对本项目的 COF 迁移门槛，不表示模型在官方原始数据域无效。三模型的最终数字
和证据 hash 见 [model_evaluation_closeout_20260814.json](reports/model_evaluation_closeout_20260814.json)、
[生成模型训练规划](../../docs/生成模型训练规划.md)和
[三模型实验设计与判定口径](../../docs/presentation/generative_model_progress_20260821/08_三模型实验设计与判定口径.md)。

## 保留的可用入口

### SemlaFlow

- `semlaflow_bridge.py`：官方源码、元素/电荷词表与 COF adapter 的严格边界；
- `../smoke/run_semlaflow_forward.py`：随机权重单 batch 前向/反向 smoke；
- `../smoke/splits/semlaflow_v1.json`：冻结 smoke split；
- `../external/semla-flow/`：锁定的官方源码 checkout。

SemlaFlow `.ckpt` 已删除。如需重新执行官方 checkpoint 实验，应按
[环境说明](../environment/README.md)重新下载权重，不应恢复旧调参分支。

### MiDi

- `midi_bridge.py`：官方 MiDi graph/feature 严格映射；
- `midi_cof_runtime.py`：COF 统计量与 MiDi runtime 适配；
- `../smoke/run_midi_forward.py`：随机权重前向/反向 smoke；
- `../smoke/splits/midi_v1.json`：冻结 smoke split；
- `../external/midi/`：锁定的官方源码 checkout。

MiDi `.ckpt` 和 1.7 GB TABASCO 镜像已删除。最终 100 样本近似 GEOM 与 deterministic-loss
结论仍在 `../smoke/reports/midi_*.json`；如需复现实验，应重新下载输入并新建版本化 run。

### UAE-3D

保留的模型模块：

- `uae3d_bridge.py`、`uae3d_reconstruction.py`；
- `uae3d_bond_calibration.py`、`uae3d_bond_capacity.py`、`uae3d_gate_a.py`；
- 最终只读审计依赖的 `uae3d_hierarchical_bonds.py`、`uae3d_objectives.py`。

保留的最终入口：

- `../smoke/run_uae3d_forward.py`；
- `../smoke/build_uae3d_gate_a_protocol.py`、`audit_uae3d_gate_a.py`；
- `../smoke/build_uae3d_c23_readonly_protocol.py`、`audit_uae3d_c23_reconstruction.py`、
  `finalize_uae3d_c23_readonly_audit.py`；
- `../smoke/train_uae3d_reconstruction.py` 仅作为最终 checkpoint schema/加载依赖，不再用于续训。

唯一保留权重：

```text
generative_model/runs/uae3d_overfit4/
  from_tier1_w10_lr3em5_0512/checkpoints/step-003008.pt
SHA-256: 575dfb9de9ce1fb7ce7825bb947f105dc8dca7cb8c01909a57e662585c13a11a
```

它只允许用于只读 Gate A 和 24 C2 + 6 C3 重构复核，不能被描述为已准入的生成模型，也不应
恢复历史 tier-4、explicit-head、hard-pair 或 hierarchical 训练分支。

## 最小验证

在相应环境中可运行：

```bash
cd /home/tianyajun/MARL_for_COFs

# 不加载已删除的官方 checkpoint，只验证本地 bridge/Gate 模块
conda run -n env_cof python -m unittest \
  generative_model.tests.test_midi_bridge \
  generative_model.tests.test_uae3d_bridge \
  generative_model.tests.test_uae3d_bond_capacity \
  generative_model.tests.test_uae3d_gate_a -v
```

SemlaFlow/MiDi/UAE 官方源码 import 分别依赖 `env_gen`、`env_midi`、`env_uae3d`；不要在
`env_cof` 中把缺少模型专用依赖误判为源码损坏。

## 清理记录与恢复边界

清理工具：`generative_model/maintenance/prune_legacy_models.py`。正式 manifest：
`generative_model/maintenance/legacy_cleanup_manifest_20260828.json`。

- 首轮删除 159 个精确目标，断链扫描再删除 5 个只引用旧路径的脚本/测试；首轮回收
  `3,570,259,259 bytes`（约 `3.33 GiB`）；
- `generative_model/` 从约 `3.8 GB` 降至约 `446 MB`；
- MiDi/SemlaFlow 权重和外部数据只能通过重新下载恢复；
- 被删 runs 是未跟踪实验产物，不能从主仓库 Git 恢复；
- frozen JSON 报告、结论文档、官方源码 checkout 与 UAE 最终权重均保留。

任何后续旧模型实验都应新建版本化目录，禁止重新堆叠到已清理的历史路径中。

第二轮清理工具为 `generative_model/maintenance/prune_smoke_etflow_history.py`，manifest 为
`generative_model/maintenance/smoke_etflow_cleanup_manifest_20260828.json`。该轮删除 197 个
旧 smoke/ET-Flow/coordinate-flow/ETKDG 实验目标，回收 `309,297,989 bytes`；保留 v5 的
完整可执行依赖闭包、官方 ET-Flow checkpoint、四路线消融及 Core-OOD/external/
controllability/xTB 结果。`smoke/reports/` 未参与删除。
