# PG-OrbitFlow 第一阶段清理记录

日期：2026-09-01

## 清理前恢复点

- Git 分支：`june_a6000`
- 清理前提交：`05fc327`（`feat: add PG-OrbitFlow dataset training pipeline`）
- 远端：`origin/june_a6000`
- 回归测试：PG-OrbitFlow 61/61、Cn 接口 4/4 通过。

## 删除规则

历史 checkpoint 仅在同时满足以下条件时删除：

1. 位于 `generative_model/PG_OrbitFlow/runs/`；
2. 扩展名为 `.pt` 或 `.ckpt`；
3. 未被 `configs/*.json` 或 `reports/*.json` 以运行相对路径引用；
4. 不位于活动训练目录 `runs/dataset_training_full_v1/`。

按该规则识别出 114 个冗余权重，共 496,363,436 bytes（约 473.4 MiB）。删除前已逐文件计算
SHA-256，并确认文件可读；这些 checkpoint 受仓库 `.gitignore` 管理，不属于 Git 快照。

同时删除：

- Python `__pycache__`；
- 已被 v2 取代的 `cache/factorized_full_v1/` 及其 console log；
- 已被 v2 取代的 `cache/factorized_smoke_v1/`。

## 明确保留

- 活动全量训练：`runs/dataset_training_full_v1/`；
- 最终 full-v2 cache：`cache/factorized_full_v2/`；
- 当前 smoke-v2 cache：`cache/factorized_smoke_v2/`；
- 所有 JSON、日志、loss、NPZ、结论文档和消融代码；
- 所有被 protocol/report 直接引用的 checkpoint；
- 最终 M3 parent：`runs/m3p6_fingerprint_shape_v1/last.pt`。

## 历史权重清理分组

| 运行目录 | 文件数 | bytes |
|---|---:|---:|
| `c0_local_recovery_v1` | 6 | 51,887,107 |
| `c0_local_recovery_v2_pg_balanced` | 6 | 51,887,107 |
| `dataset_training_smoke_v2` | 6 | 46,321,132 |
| `m1_bond_angle_tier4_v1` | 5 | 16,284,766 |
| `m2_torsion_tier4_v1` | 5 | 19,147,850 |
| `m2p1_phase_tier16_v1` | 5 | 19,465,610 |
| `m2p2_coupled_terminal_tier16_v4` | 5 | 9,524,710 |
| `m2p2_local_rotor_set_tier16_v1` | 1 | 1,904,942 |
| `m2p2_local_rotor_set_tier16_v2` | 6 | 11,425,152 |
| `m2p2_local_rotor_set_tier16_v3` | 6 | 11,425,152 |
| `m2p2_operation_coupled_tier16_v5` | 5 | 9,524,710 |
| `m3_tier32_v2` | 9 | 40,497,138 |
| `m3p1_automorphism_tier32_v2` | 9 | 42,387,642 |
| `m3p2_torsion_refinement_v1` | 5 | 12,569,910 |
| `m3p3_worst_molecule_refinement_v1` | 5 | 12,569,910 |
| `m3p4_global_pair_v1` | 6 | 12,342,520 |
| `m3p5_global_shape_v1` | 5 | 9,749,070 |
| `m3p5b_shape_lbfgs_v1` | 1 | 6,628,607 |
| `m3p6_fingerprint_shape_v1` | 5 | 23,579,510 |
| `overfit_tier4_h1_harmonic` | 1 | 8,626,131 |
| `overfit_tier4_h2_centralizer` | 1 | 8,626,131 |
| `overfit_tier4_h3_endpoint` | 1 | 8,626,131 |
| `overfit_tier4_v1` | 6 | 51,887,171 |
| `overfit_tier4_v2_phase_aligned` | 1 | 8,626,131 |
| `smoke_v1` | 3 | 849,196 |
| **合计** | **114** | **496,363,436** |

## 执行后验证

- 实际删除 checkpoint：114 个，496,363,436 bytes；
- 实际删除旧缓存：106 个已跟踪文件，267,051 bytes；
- 删除 `__pycache__`：2 个目录；
- 清理后 PG-OrbitFlow 目录大小：291,970,468 bytes；
- 剩余 checkpoint：40 个，共 280,508,796 bytes；
- protocol/report 中 checkpoint 引用失效数：0；
- 活动训练进程未中断，并在清理期间继续写出 `step-008000.pt`。
