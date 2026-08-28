# ET-Flow 零样本构象路线与 symmetry projection

## 1. 与现有路线的边界

本路线只替换 `known graph → initial 3D conformer`，不能与 E2/E3 或 E3-C 混写。

| 编号 | 输入 → 输出 | 是否学习模型 | 当前状态 |
|---|---|---:|---|
| E2 | known graph + known action → ETKDG/orbit 3D | 否 | dataset-scale C2/C3 已通过 |
| E3-A/B | known graph + Target_PG → recovered action → E2 3D | 否 | 已通过 |
| **EF0/EF1** | known graph → ET-Flow conformer → recovered action/projection → 3D | 官方预训练模型，先 zero-shot | EF0-A 已通过；EF0-B 待用户运行 |
| E3-C | Target_PG → novel valid 2D graph | 尚未选择 | 未开始；不是 ET-Flow |

ET-Flow 是 graph-conditioned conformer generator，不接收 `Target_PG`。指定对称性来自 E3-A
恢复的 graph action 和后处理 hard projection；因此即使 EF1 通过，也只能表述为“ET-Flow
可以作为初始构象先验”，不能表述为“ET-Flow 自身学会了指定点群”。

## 2. symmetry projection 的固定设计

对 ET-Flow 输出的 canonical-order 坐标 `X` 执行：

1. 质心归零；
2. PCA 主轴对齐，并枚举 24 个 proper signed-axis orientations；
3. 对每个候选执行 Reynolds average：
   `X_sym = |G|^-1 Σ_g P_g^-1 X R_g^T`；
4. 保持输入 radius of gyration，避免投影任意缩放；
5. 选择 projection displacement 最小的 orientation；
6. 接同一 orbit-constrained UFF + short-range repulsion 后端；
7. 在 `env_cof` 用固定 pymatgen protocol 独立重算 actual-PG/compatible。

其中 `(R_g, P_g)` 只由 canonical graph + `Target_PG` 的 E3-A 恢复得到；不读取 reference XYZ，
也不读取 v2 中预存的 target action。公式约定保持
`X @ R_g.T ≈ X[permutation_g]`。

## 3. EF0-A：离线严格桥接（已完成）

EF0-A 不安装 ET-Flow、不下载 checkpoint，也不评价构象质量。固定 4 个 C2/C3 graph-only
样本 `45/81/1684/2378`，其中 `1684/2378` 是两条 Sn 分子。

严格接口包括：

- canonical 显式 H graph 生成 atom-mapped SMILES；
- 重原子由 atom map 恢复；官方 RDKit `MolFromSmiles → AddHs` 会重建 H，因此 H 只按
  canonical heavy-parent neighborhood 和稳定原子索引提升，绝不使用坐标 Hungarian；
- atomic number、formal charge、radical electron、typed sparse bonds 全量逐项核对；
- ET-Flow `max_z=100` 按 exclusive upper bound 审计，Sn 的 Z=50 原样保留；
- 官方辅助 node feature 的 `misc` bucket 必须单独报告，禁止静默掩盖；
- 模型坐标按 graph-only mapping 无损还原 canonical atom order；
- 后接 E3 projection 时不得读取 reference XYZ 或 ETKDG coordinates。

结果：8/8 Gate checks 通过；4/4 atom/graph roundtrip exact，Sn 2/2，最大 operation error
`1.8841e-15 Å`。正式状态：
`PASS_ETFLOW_EF0A_OFFLINE_BRIDGE_AWAIT_OFFICIAL_PREFLIGHT`。

- protocol SHA-256：`daed499dea346665ef96d8422d0040f4f009df3d602a079325b8196e8bc0082d`
- report SHA-256：`e5f71cc1ebd48b48b8966d8d0f3f7646765100c4171b5963a1d306a797df5020`

该结果只证明本地 adapter 与 symmetry layer 正确，不代表官方 checkpoint 能处理 COF 图。

## 4. EF0-B：官方包与 checkpoint 预检（已通过）

EF0-B 使用官方 `drugs-o3`、1 sample/molecule、10 ODE steps，只做 4 分子 forward。结果：

- official package 能导入，checkpoint 能加载并记录 SHA-256；
- 4/4 forward finite；
- canonical atom/graph roundtrip 不变；
- 两条 Sn 均不 fallback 且 forward 成功；
- upstream auxiliary `misc` 统计已落盘；
- hard projection operation error `≤1e-5 Å`。

7/7 checks 通过，正式状态
`PASS_ETFLOW_EF0B_OFFICIAL_PREFLIGHT_FREEZE_EF1`。实际身份为 Python 3.11.15、ET-Flow
0.1.2、`drugs-o3` checkpoint SHA-256
`a24ae9a1fed2708696929308ed1dc10ab167fd66a2d51c44a4afb6c11badccb2`。4/4 CUDA forward
finite、Sn 2/2，无元素 fallback，最大 operation error `4.09e-15 Å`。

EF0-B 不设置 RMSD、碰撞或 actual-PG 质量门槛；这些属于 EF1。质量预警是 raw 最短原子间距
约 `1.00–1.05 Å`，但 hard projection 后 3/4 降到 `0.12–0.21 Å`。因此 EF1 必须保留 raw、
projected 和相同 F0.2 修复后的 final 三层指标，不能只用接近零的 operation error 宣称成功。

EF0-B 已在安装/下载前冻结，protocol SHA-256 为
`07f93fa075f74ce02101ff586c91f947616efc0196cbe2810c9f02e4612b07f3`。

## 5. EF1：32 条 IID-test 零样本对照（已通过）

EF1 不训练、不微调。固定 IID-test 的 24 C2 + 8 C3，任何 checkpoint 或 threshold 均不得
根据结果更换。对每个 graph 使用相同 candidate 数量，比较两条支路：

```text
A: ETKDGv3 multi-seed → E3 projection → frozen F0.2 backend
B: ET-Flow drugs-o3 zero-shot → E3 projection → frozen F0.2 backend
```

raw 层报告 Kabsch RMSD、pair-distance MAE、bond-length MAE、minimum nonbonded distance 和
collision；final 层再报告 UFF objective、上述几何指标、actual-PG exact/compatible。主判据是
paired molecule-level ratio/win fraction，而不是只看全体均值。EF1 的价值判断为：

- 若 ET-Flow raw geometry 显著优于 ETKDG，且相同 projection/backend 后 collision 与
  compatible 不退化，则 ET-Flow 可进入 COF 域微调评估；
- 若只在 hard projection 后变为 compatible、raw geometry 不优于 ETKDG，则 ET-Flow 没有
  提供额外先验价值，停止该支路；
- 若 Sn/显式 H/图往返失败，属于 strict interface failure，不能靠删元素或映射为 C/Si 修复。

EF1-v1 已在读取其输出前冻结。首次启动完成第一个分子的计算后，在写逐分子 JSON 时因
`bridge_audit.model_to_canonical` 为 NumPy ndarray 而失败；没有形成有效 molecule JSON、
aggregate coordinates 或 geometry report，因此不是模型/Gate 失败。v1 原样保留。

EF1-v2 只新增 ndarray/NumPy scalar 的显式 JSON encoding；panel、checkpoint、candidate 数、
50 ODE steps、阈值和决策均与 v1 一致。panel 为 SHA-256 排序得到的 24 C2 + 8 C3 IID-test，原子数
14–78，不含 Sn（Sn 已由 EF0-B stress panel 单独覆盖）；每条分子两支路各 4 candidates，
ET-Flow 使用 50 ODE steps。candidate 只按 final UFF+repulsion objective 选择，reference XYZ
只作事后度量。v2 protocol SHA-256：
`e38afc3705d2ee122a0d0d6119ccdaba412cbf583df04514270153965f9f39aa`。

EF1-v2 的 geometry 与独立 point-group audit 均已完成。最终状态为
`PASS_ETFLOW_EF1_ZEROSHOT_ADVANCE_TO_FINETUNE_EVALUATION`，但这只是「允许设计有界微调评估」，
不是「ET-Flow 已优于 ETKDG」或「已可按 Target_PG 生成」。

| 指标 | ET-Flow 相对 ETKDG 结果 |
|---|---:|
| raw Kabsch RMSD 中位比 | 1.03329 |
| raw pair-distance MAE 中位比 | 0.87261 |
| projected Kabsch RMSD 中位比 | 0.98099 |
| projected pair-distance MAE 中位比 | 0.93518 |
| final Kabsch RMSD 中位比 | 0.99952 |
| final pair-distance MAE 中位比 | 0.99986 |
| final objective ET-Flow 胜率 | 14/32 = 43.75% |
| final collision-free / bond-quality | 32/32 / 32/32 |
| ET-Flow analyzer / compatible / exact | 32/32 / 32/32 / 20/32 |
| ETKDG analyzer / compatible / exact | 32/32 / 32/32 / 21/32 |

ET-Flow 在 raw pair-distance 和 projection 后的部分几何指标上提供了独立 learned prior
信号，但经过相同 F0.2 backend 后两者几乎收敛到同一结果；且 exact-PG 比
ETKDG 少 1 条。参考 XYZ 本身由 ETKDGv3 流程构建，RMSD 对 ETKDG 存在来源偏置，
所以本阶段将 compatible、碰撞、bond quality 和 reference-free objective 作为更重要的边界。

完整结果与下一阶段准入见
[26_ETFlow_EF1结果与微调评估.md](26_ETFlow_EF1结果与微调评估.md)。

## 6. 用户执行命令

复用 `env_uae3d` 的 PyTorch/PyG 文件，克隆为独立环境，避免每次重新下载 PyTorch，也不修改
冻结的 `env_uae3d`：

```bash
cd /home/tianyajun/MARL_for_COFs
conda create -n env_etflow --clone env_uae3d -y
conda activate env_etflow
python -m pip install etflow
python -c "import etflow, torch; print('ET-Flow import: OK'); print(torch.__version__, torch.cuda.is_available())"
```

EF0-B 已完成。下一条实际命令是在 tmux 中运行可断点续跑的 EF1 geometry：

```bash
cd /home/tianyajun/MARL_for_COFs
conda activate env_etflow
mkdir -p generative_model/checkpoints/etflow

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
python -m generative_model.smoke.run_etflow_ef1_zeroshot \
  --cache generative_model/checkpoints/etflow \
  --device cuda \
  --output generative_model/runs/etflow_ef1_zeroshot \
  2>&1 | tee generative_model/runs/etflow_ef1_zeroshot_console.log

sha256sum \
  generative_model/runs/etflow_ef1_zeroshot/geometry_report.json \
  generative_model/runs/etflow_ef1_zeroshot/coordinates.npz
```

geometry 完成后在 `env_cof` 运行独立 actual-PG audit：

```bash
cd /home/tianyajun/MARL_for_COFs
conda activate env_cof

python -m generative_model.smoke.audit_etflow_ef1_point_groups \
  --input-dir generative_model/runs/etflow_ef1_zeroshot

sha256sum generative_model/runs/etflow_ef1_zeroshot/point_group_report.json
```

## 7. 参考

- ET-Flow paper: <https://arxiv.org/abs/2410.22388>
- official repository: <https://github.com/shenoynikhil/ETFlow>
- official checkpoints: <https://zenodo.org/records/14226681>
