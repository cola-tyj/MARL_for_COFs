# `generative_model/runs` 保留与清理规则

更新日期：2026-08-28。

## 1. 目录定位

`runs/` 是本地实验产物，不是 canonical dataset 或模型源码。当前 Our ET-Flow v5 只加载：

```text
generative_model/checkpoints/etflow/drugs-o3.ckpt
```

最终实现、消融和补充实验分别位于 `inference/`、`evaluation/` 及下列 A 级 run；不要把
`smoke/` 当成当前 ET-Flow 训练目录。

## 2. 已执行清理

### 2026-08-27 checkpoint 清理

删除 MiDi 与 SemlaFlow 的 72 个 `.pt/.ckpt`，释放 `27,637,231,393 bytes`。精确清单见
`checkpoint_cleanup_manifest_20260827.json`。

### 2026-08-28 旧模型最小化

删除 SemlaFlow/MiDi/UAE-3D 的冗余脚本、外部 MiDi 数据镜像、重复 checkpoint 和历史 runs，
首轮释放 `3,570,259,259 bytes`。UAE-3D 只保留 step-3008 权重；精确清单见
`maintenance/legacy_cleanup_manifest_20260828.json`。

### 2026-08-28 smoke 与 ET-Flow 历史分支清理

删除 197 个旧 smoke/EF2 learned-adapter/coordinate-flow/ETKDG 实验目标，释放
`309,297,989 bytes`。保留 `smoke/reports/`、v5 完整依赖、最终消融和全部补充实验；精确清单见
`maintenance/smoke_etflow_cleanup_manifest_20260828.json`。

清理后整个 `generative_model/` 约 139 MiB，`runs/` 只剩下列 A/B/C 级目录。

## 3. 当前保留等级

### A：当前结果，必须保留

- `etflow_e3f02_iidtest_v1`：C2/C3 固定批量结果；
- `etflow_e3f02_rare_targets_v3`：S4/D6h 最终 v5 批量结果；
- `etflow_e3f02_v5_examples`：最终可视化/独立审计示例；
- `our_etflow_ablation_v1`：四路线 paired ablation；
- `our_etflow_core_ood_v1`、`our_etflow_external_v1`；
- `our_etflow_controllability_v1`；
- `our_etflow_xtb_relaxation_v1`。

上述目录中的 JSON、NPZ、XYZ、日志和图表共同构成当前论文/汇报证据，不按扩展名单独删除。

### B：v5 冻结构建证据，必须保留三个文件

`etflow_e3f02_rare_targets_v2/` 只保留：

- `geometry_report.json`；
- `point_group_report.json`；
- `s4_inertia_boundary_diagnosis.json`。

它们是 v5 protocol builder 的显式输入，说明 S4 inertia guard 的来源；其 molecules、NPZ 和日志
已删除。

### C：历史模型最小诊断

- `uae3d_c23_readonly`；
- `uae3d_overfit4` 中唯一保留的 step-3008 checkpoint。

MiDi/SemlaFlow 的最终 JSON 结论位于 `smoke/reports/` 和 `models/reports/`，不再保留 run。

## 4. 源码保留边界

Our ET-Flow v5 并非单个 `*_v5.py`。冻结协议对基础文件及 v2–v5 源码逐一记录 SHA-256，消融和
controllability 还直接使用 v3/v4 的多 action/候选排序。因此 v2–v4 属于 v5 可执行闭包，不能
删除或机械合并。旧 EF2 soft learned-conditioner 的训练、模型和采样脚本已删除，只保留 JSON
证据与文档结论。

后续清理必须先运行 dry-run，检查协议 `identity.source_sha256` 零冲突，再执行并重新跑当前
消融/xTB 测试与 supplement completion audit。禁止删除 `checkpoints/etflow/drugs-o3.ckpt`。

本轮清理后的本地 import 扫描为 0 条断链，`env_cof` 完整测试 104/104 通过，supplement
completion audit 继续 PASS。初次清单外的 3 个纯历史文件删除和测试断言收口记录在
`maintenance/smoke_etflow_cleanup_followup_20260828.json`。
