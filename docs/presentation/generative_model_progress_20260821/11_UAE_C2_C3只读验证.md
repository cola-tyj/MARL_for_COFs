# UAE-3D C2/C3 只读验证

## 1. 为什么需要验证

UAE tier-4 最佳 checkpoint 在四个训练分子上达到 2/4 完整重构，成功的恰好是 C2、C3：

| 分子 | Target_PG | 原子数 | bond exact | RMSD |
|---|---:|---:|:---:|---:|
| `cof_000864` | C2 | 10 | 是 | 0.04368 Å |
| `cof_001135` | C3 | 9 | 是 | 0.05812 Å |
| `cof_001114` | S4 | 49 | 否 | 0.14220 Å |
| `cof_000862` | D6h | 72 | 否 | 0.12873 Å |

这里同时混杂了点群、原子数和训练集记忆，不能直接得出“UAE 更适合 C2/C3”。因此新增
一次冻结权重、独立 IID-validation 的只读审计。

## 2. 冻结协议

- checkpoint：tier-4 step 3,008，SHA-256
  `575dfb9de9ce1fb7ce7825bb947f105dc8dca7cb8c01909a57e662585c13a11a`；
- 权重不更新，使用 official encoder `z_mean` 和原六分类 argmax decoder；
- 面板复用 Gate A 的 IID-validation：C2 24 条、C3 6 条；
- 不在 validation 上拟合 threshold，不使用 test/Core-OOD；
- 同时按 `≤16`、`17–40`、`>40` 原子分层；
- 重构坐标仅减去整体质心后使用固定 symmetry analyzer，结构本身不修复；
- UAE 不输入 `Target_PG`，因此该实验只测重构，不是条件生成。

联合成功要求 atom/bond exact、sanitize、connected、RMSD `≤0.15 Å`、最小距离 `≥0.6 Å`
和 reconstructed PG compatible 同时成立。无论结果如何，都不改写既有
`FAIL_GATE_A_STOP_UAE_BRANCH`。

## 3. 结果

| 指标 | C2 | C3 | 合计 |
|---|---:|---:|---:|
| 分子数 | 24 | 6 | 30 |
| atom exact | 0 | 0 | 0 |
| bond exact | 0 | 0 | 0 |
| sanitize valid | 2 | 3 | 5 |
| connected | 0 | 0 | 0 |
| collision-free | 7 | 1 | 8 |
| analyzer compatible | 0 | 0 | 0 |
| joint success | 0 | 0 | 0 |
| mean coordinate RMSD | 1.982 Å | 2.236 Å | 2.033 Å |
| max coordinate RMSD | 5.654 Å | 3.751 Å | 5.654 Å |

按尺寸分层同样没有发现可迁移的小分子优势：`≤16` 原子只有 3 条，atom/bond/connected/
compatible/joint 均为 0/3，平均 RMSD `1.477 Å`。中型与大型分别也是 0 条联合成功。

所有重构坐标在固定 analyzer 下均不存在完整目标 C2/C3 子群；这不是 analyzer 工程失败，
而是输出坐标没有满足目标 subgroup。最小原子间距最低为 `0.0493 Å`，存在严重碰撞。

## 4. 结论

正式状态：

`UAE_C23_READONLY_WEAK_RECONSTRUCTION_EVIDENCE_KEEP_BRANCH_CLOSED`

tier-4 的 C2/C3 2/2 成功是已训练四分子面板上的记忆证据，不能迁移到独立 C2/C3
validation。限制到简单点群不会使 UAE Gate 通过，也不支持重开 Gate B、tier-32、UDM 或
dataset-level fine-tuning。

这不否定“把 C2/C3 作为新 graph→3D 路线的主要目标”：后者固定真实 graph，只预测坐标，
不承担 UAE 失败的 full-pair graph reconstruction。UAE 现阶段只保留历史重构基线，不作为
G2.2 初始化的正证据。

## 5. 机器证据

- protocol SHA-256：`d082cd5884e6409ae361782c56ac136de19d462b2b1ee7d70ae3e7f4dedfd3f7`
- final report SHA-256：`d8f6be65cf60c59291981deddac9fbe83f26ed6a70be0319192c43c43cfeb2d1`
- reconstruction intermediate SHA-256：`81803f4b59a43ac064f70910d1197afabc2c17fccf16fff1d960eb24924d7ec9`
- coordinate artifact SHA-256：`731f7c2d6d7cf0fdbf7845f6712648fa624d6f59bfc198d18b167f1fe182b638`
