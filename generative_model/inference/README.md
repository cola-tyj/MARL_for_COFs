# Our ET-Flow v5 推理目录

本目录只负责 `known graph + requested Target_PG -> 3D XYZ` 的冻结推理与批量审计，不负责
训练 learned PG adapter。公共 Python 入口是：

```python
from generative_model.inference import generate_v5
```

CLI 入口为：

```bash
python -m generative_model.inference.generate_etflow_symmetric_xyz_v5 --help
```

外部 Cn 2D 图的批量入口为 `run_our_etflow_cn.py`，它只增加重原子图→显式 H 的严格适配层，
随后直接调用冻结的 `run_prediction_v5`。独立点群审计入口为
`audit_our_etflow_cn.py`，结果保存在 `generative_model/results/Cn`。

## 文件分组

| 分组 | 文件 | 作用 |
|---|---|---|
| 当前单分子推理 | `generate_etflow_symmetric_xyz_v5.py` | v5 orchestration 与严格源码/checkpoint 校验 |
| v5 依赖闭包 | `generate_etflow_symmetric_xyz.py`、`_v2.py`、`_v3.py`、`_v4.py` | graph action、候选枚举/排序、S4/D6h 路径；不是可删旧版本 |
| 几何 guard | `s4_inertia_selection.py` | reference-free S4 inertia boundary guard |
| 当前协议 | `etflow_e3f02_protocol_v5.json` | 单分子冻结协议 |
| C2/C3 批量 | `run_etflow_e3f02_iidtest.py`、`audit_etflow_e3f02_iidtest.py` | 固定 IID panel 与独立审计 |
| S4/D6h 批量 | `run_etflow_e3f02_rare_targets_v3.py`、`audit_etflow_e3f02_rare_targets.py` | rare-target 全量与独立审计 |
| 总完成审计 | `audit_etflow_e3f02_completion.py` | 合并四点群最终结论 |
| 冻结构建器 | `build_*` | 只用于重建协议，不是日常推理入口 |

`etflow_e3f02_protocol_v1-v4.json` 和 rare-target v1-v2 是 v5 的冻结 provenance chain；
`build_etflow_e3f02_protocol_v3/v4.py` 还被 v5 的 `identity.source_sha256` 显式校验。不要为了
文件名整齐而移动、合并或重命名这些文件，否则现有 v5 report 将不能通过源码身份检查。

训练期失败的 EF2 soft conditioner、collision-aware、dual-basis 和 trajectory-risk 脚本已经
删除；其 JSON 证据在 `../smoke/reports/`，结论见 `../../docs/our_ET_Flow.md`。
