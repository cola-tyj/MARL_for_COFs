# ETKDG-Orbit E2：全数据规模运行协议

## 1. 当前状态

E2 的两个执行模块和 **2,232 分子正式任务均已完成**：

1. `run_etkdg_orbit_e2.py`：在 `env_uae3d` 中生成并筛选初始 3D，逐分子原子化落盘，支持
   同一输出目录断点续跑；
2. `audit_etkdg_orbit_e2_point_groups.py`：在 `env_cof` 中独立读取最终坐标，用 pymatgen
   protocol 重算 actual point group 和 compatible。

这不是模型训练，不使用 GPU，也不产生 checkpoint。它回答的是：E1 已通过的
`known graph + known target group action → 3D` 链条能否扩展到全部开发域 C2/C3 数据。
最终答案为通过：`PASS_ETKDG_ORBIT_E2_DATASET_SCALE_C23`。

正式 v2 protocol：

- 文件：`generative_model/smoke/reports/etkdg_orbit_e2_protocol_v2.json`；
- SHA-256：`d2480ea341815764b7e6e5f091703948748da0724e49d1be3a32075573ce5de6`；
- v1 只用于 2 分子工程交接测试，未运行正式全量；v2 将 optimizer acceptable termination
  比例恢复为 F0.2/E1 继承值 `0.75`，其余科学边界不变。

## 2. 冻结范围

| split / Target_PG | 数量 |
|---|---:|
| IID train / C2 | 1,615 |
| IID train / C3 | 369 |
| IID validation / C2 | 202 |
| IID validation / C3 | 46 |
| 合计 | 2,232 |

共 79,617 个原子，最大 87 原子；含 Sn 的 package index 为 `1684`、`2378`。每个分子固定
4 个 ETKDGv3 seed，共 8,928 个候选。IID test 与 Core-OOD 不读取、不参与阈值选择。

## 3. Runner 的输入、输出和恢复语义

输入是 canonical v2 graph、Target_PG 对应的已知 operation/permutation/orbit，以及冻结的
4 个随机种子。reference XYZ 不参与初始化或候选选择，只用于事后 bond/coordinate 指标。

每个分子先写：

- `records/package_XXXXXX.json`：严格 graph action、4 个候选、选中 seed、失败原因和指标；
- `coordinates/package_XXXXXX.npz`：成功分子的 canonical atom-order 坐标。

单分子文件通过临时文件加 `os.replace` 原子化提交。重新运行同一命令时，runner 会校验
protocol SHA、panel SHA、record/coordinate SHA，校验通过的分子直接恢复；不删除已有结果，
也不重复计算。全部完成后聚合：

- `geometry_report.json`；
- `coordinates.npz`；
- `completion_status.json`。

`--max-molecules` 只允许工程前缀测试；正式任务必须省略。

## 4. 两层 Gate

几何层主要阈值：分子成功率 `≥0.98`、8,928 个冻结候选总体成功率 `≥0.95`、无
`<0.6 Å` 碰撞比例 `≥0.98`、bond MAE `≤0.15 Å` 比例 `≥0.98`、最大 operation error
`≤1e-5 Å`、acceptable optimizer termination 比例 `≥0.75`，两条 Sn 均须成功。

独立点群层主要阈值：全 panel analyzer 成功率 `≥0.98`、Target_PG compatible 比例
`≥0.95`，两条 Sn 的 compatible 比例必须为 `1.0`。exact match 和 actual-PG 分布照常报告，
但不把 exact 当作 compatible 的替代。

只有独立审计最终给出
`PASS_ETKDG_ORBIT_E2_DATASET_SCALE_C23`，才能宣称 dataset-scale known-graph C2/C3
baseline 通过。模块建成或 runner 正常结束都不等于科学 Gate 通过。

## 5. 正式运行命令

在 tmux 中先运行几何阶段：

```bash
conda activate env_uae3d
cd /home/tianyajun/MARL_for_COFs
set -euo pipefail

OUT=generative_model/runs/etkdg_orbit_e2
mkdir -p "$OUT"

test "$(sha256sum generative_model/smoke/reports/etkdg_orbit_e2_protocol_v2.json | awk '{print $1}')" = \
  "d2480ea341815764b7e6e5f091703948748da0724e49d1be3a32075573ce5de6"

/usr/bin/time -v python -m generative_model.smoke.run_etkdg_orbit_e2 \
  2>&1 | tee -a "$OUT/console.log"
```

断线后重新进入 tmux；若 Python 任务本身被中止，原样重跑上面最后一条命令即可续跑。不要
更换 `OUT`，不要加 `--max-molecules`。

几何阶段完成并生成 `geometry_report.json` 后，再运行独立点群审计：

```bash
conda activate env_cof
cd /home/tianyajun/MARL_for_COFs
set -euo pipefail

OUT=generative_model/runs/etkdg_orbit_e2
python -m generative_model.smoke.audit_etkdg_orbit_e2_point_groups \
  2>&1 | tee -a "$OUT/point_group_console.log"

sha256sum "$OUT"/{geometry_report.json,coordinates.npz,completion_status.json,point_group_report.json}
```

查看进度：

```bash
find generative_model/runs/etkdg_orbit_e2/records -name 'package_*.json' | wc -l
tail -f generative_model/runs/etkdg_orbit_e2/console.log
```

`tail -f` 长时间停在最后一行通常只是等待新输出；按 `Ctrl-C` 只退出 tail，不会停止 tmux
中的 runner。

## 6. 工程验证

- 普通分子与 Sn 分子的 runner→NPZ→独立 analyzer 交接已贯通；
- 同一输出目录从 2 条前缀恢复到 4 条前缀，已有 record 被识别为 `resumed=true`；
- `env_uae3d` 与 `env_cof` 各运行 16 项 E0/E1/E2/F0.2 联合回归，均为 16/16 通过；
- v2 自动化测试覆盖 protocol fingerprint、源码 SHA、panel 边界、Sn、候选预算、质量阈值
  和 resume-state 绑定。

这些工程测试在正式任务之前证明了执行模块和协议可用；科学结论由下节正式输出给出。

## 7. 正式结果

### 7.1 几何层

| 指标 | 结果 | 结论 |
|---|---:|:---:|
| 分子成功 | 2,232/2,232 | 通过 |
| ETKDG 候选成功 | 8,928/8,928 | 通过 |
| collision-free | 100% | 通过 |
| bond-quality | 100% | 通过 |
| 最短原子间距 | 0.749019 Å | 通过 |
| 最大 bond MAE | 0.079318 Å | 通过 |
| 最大 operation error | 6.87793e-7 Å | 通过 |
| acceptable optimizer termination | 98.4767% | 通过 |
| Sn 成功 | 2/2 | 通过 |
| terminal-resonance normalization | 101 个分子 | 报告项 |

几何状态为 `PASS_ETKDG_ORBIT_E2_GEOMETRY_AWAIT_POINT_GROUP_AUDIT`。全任务 wall time
为 1:05:40。

### 7.2 独立点群层

| 指标 | 结果 | 冻结阈值 | 结论 |
|---|---:|---:|:---:|
| analyzer success | 2,223/2,232 = 99.5968% | ≥98% | 通过 |
| Target_PG compatible | 2,223/2,232 = 99.5968% | ≥95% | 通过 |
| exact match | 1,280/2,232 = 57.3477% | 仅报告 | — |
| Sn compatible | 2/2 = 100% | 100% | 通过 |

成功重算的 2,223 个分子全部 compatible。actual-PG 分布为：C2 1,121、C2h 490、C2v
102、D2 46、D2h 49、C3 159、C3h 151、C3v 8、D3 51、D3h 46。exact 低于 compatible
是因为大量结构实际具有目标群的超群，不是对称性失败。

9 个失败均位于 IID-train/C2，package index 为 `17, 121, 184, 287, 408, 591, 1691,
1866, 1869`；统一异常为“候选操作中不存在完整 C2 目标子群”。它们的 geometry 均成功，
在全 panel 口径中保守计为 analyzer/compatible 失败，没有静默修复，也未触碰 IID-test 或
Core-OOD。

最终状态：`PASS_ETKDG_ORBIT_E2_DATASET_SCALE_C23`。这证明当前方法是可靠的
**known graph + known target orbit action → C2/C3 3D baseline**；它仍不代表可以从
Target_PG 独立生成新 SMILES/graph。下一主问题转为新 graph 的 automorphism/orbit 推导，
或先做 Target_PG-conditioned 2D graph generation 再接此 3D 后端。

最终哈希：

- geometry report：`11c7dc116f03446513b7d4d0e5a6071b637eb2b7dcd18976c19e3bfdf9dbdf18`；
- coordinates：`07689da73971b241df1b559ac51f433acb9e077bc4b0764f1703ece2ac000a07`；
- point-group report：`11b68bab136cae8689661202e4e9d9cf42923007812ce7c04a28ec9b60c92b97`。

## 8. 代表性 XYZ 与交互可视化

已从正式 E2 坐标确定性导出五个代表样本：

| 类型 | molecule | Target → Actual | 说明 |
|---|---|---|---|
| C2 exact | cof_000045 | C2 → C2 | IID-validation |
| C2 compatible supergroup | cof_000092 | C2 → C2h | exact=false、compatible=true |
| C3 exact | cof_000081 | C3 → C3 | IID-validation |
| C3 compatible supergroup | cof_000065 | C3 → C3h | exact=false、compatible=true |
| C2 Sn exact | cof_001684 | C2 → C2 | 严格保留 Sn |

示例目录为 `generative_model/visualization/e2_examples/`。每个样本同时包含输出 `.xyz` 和
输入/审计 `.json`；JSON 保存 canonical graph、Target_PG operation matrices、原子
permutation/orbit、选中 seed、几何指标和 independent actual-PG。示例 manifest SHA-256 为
`b6436f714ee824a09d2a84e847cb5ee97d155391039631cd517ee3515ab3e1e5`，重复导出字节一致。

在 VS Code/Jupyter 中打开
`generative_model/visualization/visualize_etkdg_orbit_e2_examples.ipynb`，使用 `env_cof`
kernel，即可下拉选择并在单元格内旋转、缩放。Notebook 已完成一次无错误执行验证。
