# 生成模型（当前实现）

本目录用于新的联合分子图 + 3D 生成模型，与 `symmcd_diffusion/` 中保留的历史实验隔离。
v1 冻结模型无关的图 + 3D canonical 接口；v2 在不改变 canonical 表示的前提下增加
点群操作、原子置换、轨道、RDKit 衍生特征和模型 adapter。

完整版本记录、验证数字、边界条件和后续任务见
[生成模型数据接口_v1_v2.md](../docs/生成模型数据接口_v1_v2.md)；总体训练 Gate 见
[生成模型训练规划.md](../docs/生成模型训练规划.md)。

当前可交付的 ET-Flow + graph action + hard projection + F0.2 模型说明、训练分支与代码地图见
[our_ET_Flow.md](../docs/our_ET_Flow.md)。
本地实验产物的保留等级、空间审计和清理边界见
[`RUNS_RETENTION.md`](RUNS_RETENTION.md)。

`smoke/` 不是当前 ET-Flow 生产目录。2026-08-28 清理后，其顶层只保留
SemlaFlow/MiDi/UAE-3D 的最小历史诊断入口；旧 ET-Flow adapter 和早期 geometry 路线的
可执行脚本已删除，但 `smoke/reports/` 中的冻结 JSON/日志/loss 仍作为科学证据保留。
Our ET-Flow v5 的代码位于 `inference/`、`conformer/`、`symmetry/`、`optimization/`，消融与
补充实验位于 `evaluation/`。v2–v4 推理文件是 v5 与消融的源码哈希依赖，不可按版本号删除。
具体入口与文件职责分别见 [`inference/README.md`](inference/README.md)、
[`evaluation/README.md`](evaluation/README.md) 和 [`tests/README.md`](tests/README.md)。

## 构建数据包

在仓库根目录、包含 NumPy、Pandas 和 RDKit 的环境中运行：

```bash
python -m generative_model.data.build_cof_package
```

默认读取 `cof_symmetry_pipeline/output/final_dataset.csv`，输出到
`generative_model/data/processed/v1/`。再次核验源文件和数据包哈希：

```bash
python -m generative_model.data.build_cof_package --verify-only
```

构建并严格验证 v2：

```bash
python -m generative_model.data.build_v2_package --workers 12
python -m generative_model.data.validate_v2 \
  --package generative_model/data/processed/v2
```

v2 仍从 `final_dataset.csv` 和 XYZ 直接重建 canonical layer，不修改或依赖 v1 中间结果。
`processed/v1/cof_graphs.npz` 与 `processed/v2/cof_graphs.npz` 字节一致。

## 加载

```python
from generative_model.data import COFGraphDataset, collate_graphs

dataset = COFGraphDataset(
    "generative_model/data/processed/v1",
    split="train",
    split_scheme="iid",  # 或 core_ood
)
sample = dataset[0]
print(sample["positions"].shape, sample["bond_index"].shape)
batch = collate_graphs([dataset[0], dataset[1]])
print(batch["batch_index"].shape)
```

每个样本包含 13 类原子 token、原子序数、形式电荷、自由基电子数、质心归零坐标，
以及由 SMILES 直接取得的键索引和键型。无向键只保存一次。`centroid` 可在需要时恢复
原 XYZ 坐标。

读取 v2 symmetry annotation：

```python
from generative_model.data import COFSymmetryDataset

dataset = COFSymmetryDataset(
    "generative_model/data/processed/v2",
    split="train",
    split_scheme="core_ood",
)
sample = dataset[0]
print(sample["target_pg"], sample["actual_pg"], sample["pg_compatible"])
print(sample["actual_symmetry"]["operation_matrices"].shape)  # [|G|, 3, 3]
print(sample["actual_symmetry"]["permutation_index"].shape)   # [|G|, N]
```

`permutation_index[g, i] = j` 表示操作 `R_g` 将第 `i` 个原子映射到第 `j` 个同元素原子：

```text
positions @ R_g.T ≈ positions[permutation_index[g]]
```

置换以长度 `N` 的索引保存，不构造 `N×N` permutation matrix。`orbit_id` 和
`orbit_size` 来自完整 actual operation group；另有对应的 target orbit。

### 点群标签约定

- `Target_PG` / `target_pg`：数据构建目标；
- `Analyzer_PG` / `analyzer_pg`：pymatgen 直接重算的候选标签；
- `Actual_PG` / `actual_pg`：候选操作经过标准有限群重建、Hungarian assignment、
  RMS tolerance 和群闭包验证后得到的最大完整点群；
- `PG_Exact_Match`：`target_pg == actual_pg`；
- `PG_Compatible`：target operation group 是 actual operation group 的子群。

统一协议为 pymatgen `PointGroupAnalyzer`，`tolerance=0.3 Å`、
`eigen_tolerance=0.01`、`matrix_tolerance=0.1`。每个操作在同元素块内使用 Hungarian
assignment，保存 operation RMS/max error；所有参数和软件版本记录在 v2 manifest。

### 模型 adapter

```python
from generative_model.data import MiDiAdapter, SemlaFlowAdapter, SparseGraphAdapter, UAE3DAdapter

eqgat_sample = SparseGraphAdapter(dataset, include_h=True)[0]
midi_sample = MiDiAdapter(dataset, include_h=True)[0]
print(midi_sample["full_pair_bond_types"].shape)  # [N, N]，0=none

uae = UAE3DAdapter(dataset, mode="compatible_subset")
# 或 mode="extended_vocabulary"：在官方 16 类 GEOM 词表后新增 Sn=16

semlaflow = SemlaFlowAdapter(dataset, mode="pretrained_compatible")
# Sn、非零自由基电子或不受支持的形式电荷都会显式报错
```

EQGAT adapter 保留 sparse graph；MiDi/UAE adapter 在加载时动态生成对称的 full-pair
bond tensor。canonical 数据始终保留显式 H，adapter 可以通过 `include_h=False` 删除 H，
并同步重编号边、置换和轨道。

UAE-3D 官方 GEOM 词表不含 Sn。v2 提供 2,530 条 compatible subset 和 17 类扩展词表
两种方案；两条含 Sn 分子绝不会映射为 C 或 Si。

## 文件

- `cof_graphs.npz`：展平的原子/键/坐标数组及分子 offsets；
- `metadata.csv`：分子元数据、可迁移的 XYZ 相对路径和两套 split 标签；
- `vocab.json`：原子与键词表和索引约定；
- `statistics.json`：数据分布和 split 统计；
- `split_iid.json`：按 `Target_PG` 分层的固定 80/10/10 划分；
- `split_core_ood.json`：按 `Core` 分组、无 Core 泄漏的固定划分；
- `manifest.json`：源 CSV、2,532 个 XYZ、数据包文件的 SHA-256 和数据指纹。

v2 额外包含：

- `symmetry_data.npz`：target/actual 点群、完整 O(3) 操作、局部 permutation、轨道和误差；
- `symmetry_vocab.json`：点群索引及矩阵/置换约定；
- `rdkit_features.npz`：hybridization、aromaticity、chirality、degree、显式 H 邻居数、
  ring membership、bond stereo/conjugation/ring；
- `uae_compatibility.json`：官方 UAE vocabulary 审计、compatible subset 和 Sn 扩展方案；
- `similarity_audit.csv`：IID/Core-OOD test 到各自 train 的最大 Morgan/Tanimoto 相似度；
- `validation_report.json`：点群一致率、分布、轨道、误差、元素、splits、相似度和 hashes。

当前 Core-OOD test 的最大 train Tanimoto 中位数为 0.653；258 条中有 57 条为 1.0。
因此该划分保证的是 **Core 标签隔离**，并不等价于严格的低结构相似度 OOD。后续主要
结论应同时报告连续 Tanimoto 分布，必要时再增加 fingerprint-threshold split。

运行测试：

```bash
conda run -n env_cof python -m unittest discover -s generative_model/tests -v
```

## 统一评测（P0/P1 已完成）

统一输入、`evaluate()`、稳定失败原因和 `evaluation_report.json` schema 位于
[evaluation/README.md](evaluation/README.md)。生成冻结 test split 的参考报告：

```bash
conda run -n env_cof python -m generative_model.evaluation.reference
```

当前参考报告：

- [IID test](evaluation/reports/v2/iid/reference_test.json)：253/253 完成评测；
- [Core-OOD test](evaluation/reports/v2/core_ood/reference_test.json)：258/258 完成评测。

两者均为零 schema/化学失败、有限且质心归零坐标、全图连通、零个低于 0.6 Å 的原子
碰撞，uniqueness/novelty 均为 100%，点群重算与 v2 保存的 actual label 100% 一致。
IID/Core-OOD 中分别有 247/250 个分子具有 MMFF 参数，其中 229/231 个在 200 步内收敛；
无参数分子单独统计。重复生成的报告字节一致。

## 当前训练结论与下一步

SemlaFlow 与 MiDi 已于 2026-08-14 收尾，均不进入 tier-4、2,532 条全量微调或 PG
conditioning。这两个结论只针对当前 COF 迁移 Gate，不是否定模型在官方原数据域的能力。

- **SemlaFlow**：官方 GEOM 原域采样、工程接口和等变 smoke 均通过；tier-1 曾出现
  2/5 raw 样本精确命中训练 SMILES，但原生化学有效性不稳定，既有 EMA、noise、decoder
  和 bond-loss 消融没有稳定修复。最终状态为 `FAIL_GATE2_SWITCH_TO_MIDI`。
- **MiDi**：官方 checkpoint 配合明确标注为近似的 GEOM marginals，在扩大到 100 样本后
  为 sanitize 74/100、connected 94/100、collision 0/100，证明权重本身可用；但最终
  bounded sqrt-Min-SNR 迁移审计虽有 aggregate ratio `0.98482`，只改善 14/32 cases 和
  4/8 timesteps，未过冻结的 60%/75% coverage Gate。最终状态为
  `FAIL_BOUNDED_SNR_DETERMINISTIC_GATE_STOP_BEFORE_SAMPLING`。

2026-08-14 时主候选切换为 **UAE-3D / UDM-3D**，原计划先验收 UAE-3D 的确定性近无损
重构，再训练无条件 latent DiT。正式 tier-4/explicit-head Gate 现已失败，因此 UDM 没有
启动，UAE 也不再作为唯一主线。当前推荐先固定 2D chemical graph，再做 graph-conditioned
3D 和 orbit/hard-symmetry 约束；阶段汇报与路线比较见
[生成模型阶段汇报材料](../docs/presentation/generative_model_progress_20260821/README.md)。

该新主线的 G0 strict interface 与 G1 hard projection 已完成。首个 G2 采用 4 层
sparse-bond E(3) coordinate denoiser：工程项通过、8/8 固定实例均改善，但预注册续训到
step 2,048 后 raw/projected mean RMSD 仍为 `0.19252/0.12663 Å`，未过 `0.08/0.05 Å`。
状态为 `FAIL_GRAPH_PG_3D_G2_TIER1_CONTINUATION_STOP_ARCHITECTURE`；下一实现改为扩大
non-bonded/global geometry 感受野，不再延长此 checkpoint。详见
[G2 tier-1 结果](../docs/presentation/generative_model_progress_20260821/10_Graph_PG_3D_G2_Tier1结果.md)。

单变量 G2.1 已进一步验证朴素 dynamic typed full-pair：工程项通过，但 step 1,024
raw/projected RMSD `0.22347/0.15233 Å`，弱于 sparse 同一步 `0.20373/0.13823 Å`。因此
G2.1 已停止。

G2.2 已采用 bonded/local/global 分通道聚合与显式 distance/angle 几何目标。其 C2
tier-1 step 1,024 达到 `0.18571/0.12376 Å`，相对 G2/G2.1 改善，但未过冻结
`0.08/0.05 Å`，状态 `FAIL_GRAPH_PG_3D_G22_C2_TIER1_STOP_AND_DIAGNOSE`。因此不进入
C2/C3 tier-4 或全数据微调；当时的下一步是有界架构诊断，该动作现已由后续 Fast-32
成熟 backbone 对照完成。

后续固定 32 IID-train + 8 IID-validation 的 Fast-32 证明：修正 GEOM 坐标尺度后，Semla
`EquiInvDynamics` 优于唯一备选 EQGAT，但两者作为 single-step absolute-x0 回归器均未过
独立验证 Gate。backbone 筛选已经停止；当前不延长这些 checkpoint，也不增加第三个模型。
新主线固定 Semla 架构，改用与其上游用途一致的 coordinate flow matching，在同一 32/8
面板通过多步 prior 采样 Gate 后才进入 2,026 条 IID-train。旧实验、旧计划与失败结论全部
保留。完整的新计划、可行性、优缺点和停止条件见
[Coordinate Flow Matching 新计划与可行性评估](../docs/presentation/generative_model_progress_20260821/13_Coordinate_Flow_Matching计划与可行性.md)。
F0 flow contract、显式时间 adapter 和 Euler/Heun 工程 Gate 已通过 13/13 测试，状态
`PASS_COORDINATE_FLOW_F0`。随后唯一一次 1,024-step Fast-32 质量实验已完成并以
`FAIL_COORDINATE_FLOW_FAST32_STOP_ENDPOINT_BRANCH` 收尾：`t=0.75` recovery、碰撞、raw
analyzer 和 correct-vs-swapped PG 未通过冻结 Gate，因此不执行 integrator repair 或
2,026 条 endpoint-flow 长训。

当前已切换到 symmetry-by-construction：orbit representative 的 NumPy/Torch 可微压缩与
展开接口通过 6/6 单元测试，并在 2,279 条 IID train/validation 上零严格失败；最大 operation
error `6.11e-7 Å`、collision-free 100%，包含 2 条 Sn。状态
`PASS_SYMMETRY_BY_CONSTRUCTION_ORBIT_INTERFACE` 只表示数据/表示层准入。结果见
[Fast-32 Flow 结果与 Orbit 接口](../docs/presentation/generative_model_progress_20260821/16_Fast32_Flow结果与Orbit硬对称接口.md)。

ETKDG-orbit E1 已进一步扩展为 E2 dataset-scale 审计模块。E2-v2 冻结 IID
train/validation 的全部 C2/C3（2,232 分子、8,928 候选、2 条 Sn），提供逐分子原子化落盘、
SHA 绑定的断点续跑 runner，以及在 `env_cof` 中独立重算 actual-PG/compatible 的 auditor。
正式全量现已通过：2,232/2,232 几何生成成功，独立 analyzer/compatible
2,223/2,232（99.60%），两条 Sn 均通过，状态
`PASS_ETKDG_ORBIT_E2_DATASET_SCALE_C23`。它是 CPU 几何/点群审计，不是训练，也不产生
checkpoint；结果证明的是 known graph + known target orbit action 的 C2/C3 3D baseline。
完整 Gate、结果和输出说明见
[ETKDG-Orbit E2 运行协议](../docs/presentation/generative_model_progress_20260821/23_ETKDG_Orbit_E2运行协议.md)。

E2 的五个代表性输出（C2/C3 exact、compatible supergroup 和 Sn）已导出到
`generative_model/visualization/e2_examples/`。在 Jupyter 中打开
[E2 示例可视化](visualization/visualize_etkdg_orbit_e2_examples.ipynb)，即可交互查看 XYZ；
同名 JSON 保留输入 graph、operation matrices、permutation/orbit 和完整输出审计语义。

E3-A 已进一步去掉 known target action 依赖。独立 graph-only loader 只读取 canonical
元素/电荷/自由基/稀疏键和 `Target_PG`，明确不能访问 XYZ、centroid 或预存 permutation。
确定性 1-WL + anchored VF2 C2/C3 automorphism recovery 在 IID train/validation 的
2,232/2,232 分子上成功，并通过完整显式 H graph 验证，Sn 2/2。E3-B 再将 recovered action
接入冻结 E2 后端：16 分子 stress panel 的 64/64 候选成功，独立 pymatgen analyzer/
compatible 为 16/16、exact 为 11/16。双环境冻结产物回归各 8/8。完整方法、指标、边界与
SHA 见 [E3-A/E3-B 结果](../docs/presentation/generative_model_progress_20260821/24_Graph_Action_E3A_E3B结果.md)。
这证明的是 `known graph + Target_PG → recovered action → 3D`，不是训练或新 graph 生成；
下一阶段 E3-C 才处理 `Target_PG → novel valid 2D graph`。

独立的 ET-Flow EF0/EF1 支路只评估 `known graph → initial conformer`。EF0-A 已完成
checkpoint-free canonical bridge：显式 H 通过 graph-only heavy-parent lifting 恢复 atom order，
Sn 保持 Z=50，元素/电荷/自由基/typed bonds 无 fallback；4/4 固定 C2/C3 样本通过，且未读取
reference XYZ 或 ETKDG 坐标。EF0-B 官方 `drugs-o3` CUDA/Sn forward 已通过，实际 ET-Flow
版本 0.1.2、checkpoint SHA `a24ae9…ccb2`。32 条 IID-test EF1 已完成并通过：ET-Flow
final analyzer/compatible/exact 为 32/32、32/32、20/32，ETKDG exact 为 21/32。两者经相同
F0.2 后 final 几何几乎持平，因此只放行 EF2 有界微调评估，不宣称 ET-Flow
已优于 ETKDG 或具备 Target_PG condition。详见
[ET-Flow zero-shot 与 symmetry projection](../docs/presentation/generative_model_progress_20260821/25_ETFlow_ZeroShot与Symmetry_Projection.md)
和 [EF1 结果与微调评估](../docs/presentation/generative_model_progress_20260821/26_ETFlow_EF1结果与微调评估.md)。

EF2 已不再仅做普通 domain adaptation，而是直接实现 PG-conditioned ET-Flow。新 wrapper
在不修改官方 checkpoint 的前提下加入 Target_PG embedding、O(3) operations、atom
permutations/orbit sizes 与 raw predicted-endpoint symmetry loss；forward 不使用 hard projection。
EF2-0 official-checkpoint engineering v2 已 10/10 通过。256 IID-train + 64 IID-validation
的 adapter-only EF2-A 已完成但未过冻结 Gate：endpoint symmetry ratio `0.91816`、整体/C2/C3
改善率 `0.51563/0.45833/0.68750`，flow ratio `0.97939`。四个 checkpoint 均失败；审计定位到
连续 `192 C2 → 64 C3` 的顺序遗忘，以及 noisy-`x_t` restoring direction 会扰动低误差 base
endpoint。下一步只做一次保持数据预算与阈值不变的 EF2-A2 修复。详见
[PG-conditioned ET-Flow EF2 实现](../docs/presentation/generative_model_progress_20260821/27_PG_Conditioned_ETFlow_EF2实现.md)。

EF2-A2 修复现已实现并通过 engineering Gate：base-endpoint direction 在固定 64 条 validation
上 64/64 收缩对称误差，`α=0.25` 的 mean MSE ratio 为 `0.56250003`，official base 不变。
相同 256 条训练记录按 `C2,C2,C2,C3` 交错，且保留 A1 每分子的噪声 seed；唯一训练协议
SHA-256 为 `eaf696764009ee46aa827dbd03a061f930c4d1c0825b73d79fc9d2a235ceb256`。训练已完成且
step 64/128/192/256 全部过 Gate；selected step 256 的 endpoint symmetry ratio 为
`0.005408`、C2/C3 改善率均 100%、flow ratio `0.699048`。正式放行 raw ODE sampling Gate，
尚未形成 actual-PG 条件生成结论。后续 EF2-B 已完成：raw correct/alternate symmetry ratio
`0.01421/0.01687`、改善 32/32 与 16/16，且两条件构象中位差异 `2.3777 Å`；但 raw
collision-free 只有 `26/32`，未过 90% Gate。按冻结规则停止 scalar conditioner，不进入
actual-PG/IID-test。

EF2-B 的 all-pair failure 已进一步分型为 3 个非键碰撞和 3 个 canonical short bond。新的
EF2-C2 overlap-aware head 将两类短程场分开计算，再做 Target_PG Reynolds average；冻结
方向审计已达到 6/6 failure 改善、32/32 all-pair collision-free/bond-quality，并保持对称误差。
256-step overlap-head-only 训练与后续 full raw ODE 已完成。小面板 collision/PG 证据为正，
但最终 EF2-C6 在全新 64 条 IID-validation 上因 C3 compatible `12/16 < 14/16` 正式失败；
虽然总体 compatible `59/64`、C2 `47/48`、collision/bond `62/64` 均过线，当前 checkpoint
仍按冻结规则停止在 IID-test 之前。后续若继续，必须新建 C3-specific 模型/训练分支。

orbit-coordinate O0 随后已通过：复用 G2.2 等变 encoder，把 atom proposals 池化为 orbit
representatives 后 hard-expand；5/5 单元测试和真实 C2/C3/Sn 工程检查全过，最大 operation
error `6.03e-7 Å`。状态 `PASS_ORBIT_COORDINATE_O0_FREEZE_O1`。这仍不是训练质量通过；随后
已实现并冻结唯一一次 512-step O1 trainer。详见
[Orbit-coordinate O0 工程结果](../docs/presentation/generative_model_progress_20260821/17_Orbit_Coordinate_O0工程结果.md)。

O1 随后已按冻结协议完成并失败：48-case validation improved `0.8333`、bond ratio `0.6769`、
hard operation error `1.23e-6 Å`，但 overall/low-noise coordinate ratio 为 `0.85997/1.01871`，
collision-free 仅 `0.64583`。状态 `FAIL_ORBIT_COORDINATE_O1_STOP_DENOISER`；不延长训练，
不进入 learned sampler。随后 orbit-constrained UFF 覆盖审计为 2,532/2,532；F0 原始
200-iteration Gate 因 optimizer success 失败并保留，唯一 F0.1 增至 500 iterations 后
10/10 checks 通过。随后独立 pymatgen audit 得到 44/48 compatible；4 个失败均为同一分子
的碰撞/C1 输出。冻结 F0.2 非键短程排斥已修复全部 4 个案例：最短间距 `0.76520 Å`，
独立重算 4/4 为 C2 exact/compatible，状态 `PASS_F02_COLLISION_REPAIR_ADVANCE_TO_ETKDG`。
ETKDG initial-conformer/atom-index/orbit interface 随后已实现：E0 4/4 通过，扩展 E1 在
24 C2 + 8 C3 IID-validation 上达到 128/128 embedding、32/32 collision-free/analyzer/
compatible；坐标重放字节一致。3 个 nitro/carboxyl-radical Lewis 共振表示由显式
terminal-resonance contract 处理，canonical graph 未修改。状态
`PASS_ETKDG_ORBIT_E1_ADVANCE_TO_DATASET_SCALE_C23`。详见
[Orbit-coordinate O1 结果](../docs/presentation/generative_model_progress_20260821/18_Orbit_Coordinate_O1结果.md)。
force-field 结果见
[Orbit force-field F0/F0.1](../docs/presentation/generative_model_progress_20260821/19_Orbit_Force_Field_F0_F01结果.md)。
独立审计见
[F0.1 坐标与点群审计](../docs/presentation/generative_model_progress_20260821/20_F01坐标归档与点群独立审计.md)。
[F0.2 碰撞修复结果](../docs/presentation/generative_model_progress_20260821/21_F02碰撞修复结果.md)。
[ETKDG + Orbit 初始构象结果](../docs/presentation/generative_model_progress_20260821/22_ETKDG_Orbit初始构象接口结果.md)。

UAE 官方 GEOM 词表缺 Sn，decoder 也没有 formal-charge/radical heads。因此必须区分：

- 2,530 条官方元素词表兼容记录；
- 2,242 条同时排除 Sn、非零形式电荷和非零自由基的首轮严格三头兼容记录。

U1 已通过：严格 bridge 按官方公式构造 55 维 node feature、`N²×5` edge 与标准化坐标；
105 原子往返无损。官方尺寸随机 UAE 在 CPU 和 A6000 均完成有限前后向，101 原子 CUDA
压力样本的峰值显存约 149 MiB，节点排列最大误差 `7.15e-7`。U2 的可恢复训练器、
`z_mean` 固定重构审计和严格 RDKit Gate 也已实现；A6000 两步工程短测证明连续训练与
断点恢复的最终参数指纹完全一致。后续全原子 tier-1 训练已到 1024 步，但原始 official
aggregate objective 未通过 bond/coordinate Gate，现已停止同配置延长。当前状态和下一项
受控 objective 对照统一见 [项目动态状态](../docs/项目动态状态.md)。balanced bond strata、
确定性 `z_mean` coordinate auxiliary、model-only checkpoint 分叉与恢复现已通过工程测试。
coordinate refinement 已使 tier-1 严格通过。tier-4 原分支到 step 3072 后 atom 为 4/4，
但 bond/sanitize/connected 停在 2/4。随后 256-step bond-focused refinement 使 balanced
bond loss 和三层 accuracy 持续改善，但恢复到 step 512 后完整 exact 仍为 2/4，触发停止
规则。hard-pair 审计选择原分支 step-3008；v3 objective、bond-head-only 冻结、梯度校准
和确定性 PCGrad 工程 Gate 均通过，但正式 32-step pilot 未过预冻结 gate：hard-pair 总数
下降，none/self 提升，present accuracy 却从 `85.135%` 降至 `57.432%`，完整 exact 仍为
2/4。该分支已停止，不进入 tier-32 或 UDM。受约束 hierarchical v4 已完成接口、只读审计
和两次确定性 1-step 工程复现：结构约束使 wrong unique pairs 179→119，分层平衡的一步
使其继续到 118，且 present/none/self/坐标均不退化。当时因此放行从 step-3008 初始化的
16-step pilot；该 pilot 随后因 raw zero-threshold 漂移失败：present 提升但 none 下降，wrong pairs
119→181，仍为 2/4 exact/sanitize。单一全局 threshold protocol 现已冻结并完成
train-fit → 32 分子 IID-validation 迁移审计；source 与 step-16 的阈值都无法同时保持
balanced existence 和 wrong pairs，validation 均为 0/32 exact。随后冻结 feature/capacity
审计表明，两个 checkpoint 的 post-GELU 特征在 IID-validation 都有约 `0.888` existence
ROC-AUC 和约 `0.80` balanced accuracy，明显高于官方共享六分类 head 的约 `0.727`；但
conditional type probe 只有约 `0.615` micro accuracy。独立 existence / conditional-type
head 的核心结构、unique-pair loss、独立 checkpoint/trainer 和连续/恢复确定性工程短测
均已通过，曾达到工程状态 `PASS_EXPLICIT_BOND_HEAD_ENGINEERING`。随后正式 64-step
head-only pilot 按训练前冻结协议运行：原子 `4/4`、坐标上限 `0.144109 Å`、conditional
type micro `0.95946`，但 none accuracy / present recall 只有 `0.61392/0.70946`，wrong
unique pairs 为 `1,464`，bond exact/sanitize/connected 均为 `0/4`。最终状态为
`FAIL_EXPLICIT_BOND_HEAD_PILOT_STOP_BRANCH`。当前 head-only 分支不再延长、不调 threshold、
不选择中间 checkpoint；严格 tier-4、tier-32 和 UDM 均未放行。对 UAE 的下一步仅做新的
只读线性可分性/表示上界诊断；该审计不再阻止 2D→3D 新主线的接口和 baseline 推进。

针对 C6 的 C3 泛化失败，C7-RS 已完成一次严格的重采样单变量实验：保持 A2 架构、loss、
256-step 预算、官方冻结 base 和 validation Gate 不变，只把训练 C2:C3 从 192:64 改为
128:128。step-256 的 C3 endpoint symmetry ratio 从 `0.012847` 降至 `0.003010`，总体
flow ratio 保持为 `0.698924`，状态
`PASS_ETFLOW_C7RS_BALANCED_A2_ENDPOINT_AWAIT_RAW_ODE`。这尚不是 raw actual-PG 通过；
下一步先做独立 C3-focused raw ODE Gate，再决定是否基于该权重重训 overlap head。

该 raw Gate 已在与旧验证面板零重叠的 17 C2 +17 C3 上完成。C3 operation RMS 虽有
14/17 改善、中位 ratio `0.77909`，compatible 也从 12/17 增至 13/17，但仍低于 15/17
门槛；3 个 C3 出现长尾 rollout 失稳，最坏 operation RMS `18.8438 Å`。C2 operation RMS
则 0/17 改善，candidate C2 compatible 14/17。正式状态
`FAIL_ETFLOW_C7RS_RAW_C3_CONFIRMATION_STOP_RESAMPLING_ONLY`。不重训 overlap head；下一步改为
operation/orbit-aware、worst-operation loss 与 bounded short-rollout 稳定性分支。

canonical 2,532 条数据不会被删改，其余字段也不会静默置零。详细历史证据见
[历史模型最小保留集](models/README.md)，环境状态见
[environment/README.md](environment/README.md)。机器可读的旧路线收尾状态与证据 hash 见
[model_evaluation_closeout_20260814.json](models/reports/model_evaluation_closeout_20260814.json)。

2026-08-28 已清理 SemlaFlow、MiDi、UAE-3D 的历史调参脚本、重复 run/checkpoint 与 MiDi
TABASCO 镜像，保留最终入口、冻结 JSON 结论、官方源码 checkout 和 UAE step-3008 唯一权重。
清理范围与不可恢复边界见
[legacy_cleanup_manifest_20260828.json](maintenance/legacy_cleanup_manifest_20260828.json)。
