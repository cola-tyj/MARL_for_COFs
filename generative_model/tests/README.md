# 测试目录职责

清理后本目录仅保留当前数据、最小历史模型接口和 Our ET-Flow v5/消融依赖的回归测试。

| 类别 | 代表文件 |
|---|---|
| canonical v1/v2 与严格元素映射 | `test_data_package.py`、`test_v2_package.py` |
| ET-Flow bridge 与 v5 继承链 | `test_etflow_bridge.py`、`test_etflow_e3f02_inference*.py` |
| graph action / hard projection / S4 guard | `test_graph_action*.py`、`test_graph_pg_3d_projection.py`、`test_s4_inertia_selection.py` |
| F0.2/消融依赖 | `test_etkdg_orbit_initializer.py`、`test_our_etflow_ablation.py` |
| xTB 补充实验 | `test_our_etflow_xtb_relaxation.py` |
| 历史模型最小可用接口 | `test_midi_bridge.py`、`test_uae3d_*.py` |
| 通用评测 | `test_evaluation.py`、`test_chemical_decoder.py`、`test_probability_audit.py` |

完整测试：

```bash
conda run -n env_cof python -m unittest discover -s generative_model/tests -q
```

部分 `test_etflow_e3f02_inference_v3/v4.py`、`test_graph_action_candidates_v2.py` 和
`test_s4_inertia_selection.py` 被 v5 protocol 直接记录源码 SHA-256，不能随意移动或合并。
