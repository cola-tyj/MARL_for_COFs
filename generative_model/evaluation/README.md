# 统一生成分子评测接口

该目录提供真实数据与模型生成结果共用的评测契约。评测层只读取数据，不修改
`cof_graphs.npz`、`symmetry_data.npz` 或 split。

## 目录职责

本目录包含两层，文件名相近但职责不同：

- 通用评测层：`schema.py`、`geometry.py`、`chemistry.py`、`evaluator.py`、
  `evaluate_generated_npz.py` 和 `reference.py`；
- Our ET-Flow 补充实验层：`our_etflow_*.py/json` 及对应的 `build_`、`run_`、`audit_`、
  `summarize_` 文件。

补充实验有意保持四阶段分离：builder 冻结样本/阈值，runner 生成结果，auditor 在独立环境重算
点群，summarizer 只聚合已冻结结果。它们不是应合并成单一脚本的重复代码；合并会让生成阶段
能够影响独立审计，并改变协议记录的源码 SHA-256。当前最终总审计入口为：

```bash
conda run -n env_cof python -m generative_model.evaluation.audit_our_etflow_supplement_completion
```

## 输入契约

`evaluate()` 接受 `EvaluationSample` 或包含下列字段的 mapping：

| 字段 | 形状 | 约定 |
|---|---|---|
| `molecule_id` | scalar | 非空分子标识 |
| `atomic_numbers` | `[N]` | 正整数原子序数；Sn 必须为 50 |
| `formal_charges` | `[N]` | 整数形式电荷 |
| `positions` | `[N,3]` | Å；非有限值会被计入数值失败，不会被修复 |
| `bond_index` | `[2,M]` | 稀疏无向键，每条只存一次且 `u < v` |
| `bond_types` | `[M]` | `1/2/3/4 = single/double/triple/aromatic` |
| `target_pg` | optional | 目标点群 |
| `actual_pg` | optional | 实际点群 |
| `pg_compatible` | optional | 目标群是否为实际群的子群 |

缺字段、错误 shape/dtype、越界键、self-bond、重复键以及 full-pair 的 `none=0` 进入稳定的
`FailureReason` 枚举。评测器只捕获这些预期输入错误，意外程序异常直接抛出。

## Python API

```python
from generative_model.evaluation import evaluate

report = evaluate(
    generated_samples,
    provenance={
        "source_type": "model_generated",
        "source_fingerprint": generation_run_sha256,
        "coordinate_unit": "angstrom",
        "model_name": "example",
        "checkpoint_sha256": checkpoint_sha256,
    },
    report_type="model_generation",
)
```

模型导出的 compact canonical NPZ 可直接评测：

```bash
conda run -n env_cof python -m generative_model.evaluation.evaluate_generated_npz \
  --input path/to/samples_canonical.npz \
  --output path/to/evaluation_report.json \
  --model-name MODEL_NAME \
  --checkpoint-sha256 CHECKPOINT_SHA256 \
  --sampling-report path/to/generation_report.json
```

NPZ 必须包含 `atomic_numbers`、`formal_charges`、`positions`、`bond_index`、`bond_types`、
`atom_offsets` 和 `bond_offsets`；所有 offset 和数组长度会严格检查，不做静默修复。

`provenance` 至少需要 `source_type`、`source_fingerprint` 和 `coordinate_unit`。模型报告还应
加入 checkpoint hash、采样 seed、采样配置和 conditioning 信息。

## 冻结数据参考报告

在仓库根目录运行：

```bash
conda run -n env_cof python -m generative_model.evaluation.reference
```

默认生成：

- `reports/v2/iid/reference_test.json`；
- `reports/v2/core_ood/reference_test.json`。

也可通过 `--package`、`--output-dir`、`--split` 和 `--split-schemes` 指定输入。报告不写入
时间戳和绝对仓库路径，JSON key/缩进/换行固定；相同数据和环境重复生成时字节一致。

## 当前范围

- report schema：`evaluation_report.schema.json`，版本 `1.1`；
- metric suite：`p1.0`；
- 输入/数值：schema 完整性、NaN/Inf、质心、半径、原子碰撞和图连通性；
- 化学/2D：严格显式 H 的 RDKit 重建与 sanitize、元素/键/环分布、uniqueness、相对 train
  的 novelty 和 nearest Morgan/Tanimoto；
- 3D：分键型键长、键角、非键碰撞、MMFF94s 参数覆盖、收敛率、能量变化和质心归零后的
  RMS 松弛位移；
- symmetry：对输入坐标重新运行 v2 的 pymatgen + Hungarian + 完整有限群协议，输出 actual
  PG、target exact/compatible、operation RMS、orbit-size 及分 `target_pg` 统计。

Morgan 指纹固定为 radius=2、2,048 bits、无显式 H、`include_chirality=false`。这是面向当前
图生成接口的 graph-only 指标；v2 Core-OOD 审计中保留的 `include_chirality=true` 是另一项
已有 split 审计，报告会分别记录配置，不能混用数值。

MMFF 的 `unsupported_parameters` 不计为优化不收敛；收敛率以 `parameterized_molecules` 为
分母。Sn 始终保持原子序数 50，RDKit sanitize 可以通过，但 MMFF 会如实记录为无参数。

`summary.status=PASS` 表示所有输入均按 schema 完成评测，不代表模型已经达到质量阈值；模型
质量必须读取 `metrics`。
