# UAE-3D Gate 与指标定义

**文档版本**：1.2  
**适用范围**：本项目 UAE-3D 确定性 autoencoder 重构审计与 `1 → 4 → 32`
分阶段放行  
**指标 schema**：`uae3d-mean-reconstruction-v1.1`  
**实现真值源**：
[`uae3d_reconstruction.py`](../generative_model/models/uae3d_reconstruction.py)

本文档统一说明“指标怎样算”、“Gate 什么时候通过”和“哪些字样不代表
质量通过”。实验的具体 checkpoint、当期阈值和结论仍以对应的机器可读
stage report 为准；本文档不用来改写已冻结的历史 Gate。

## 1. 评估协议

当前重构评估使用固定输入和 encoder 的 `z_mean`，不从 VAE posterior 采样。
输出按固定的 canonical 原子索引与 target 逐项比较，不允许重排原子来提高
exact 指标。坐标乘回固定 `position_std` 后以 Å 报告。

键 target 是加载时动态构造的完整 `N×N` 有序 pair 表示：

| class | 语义 | 位置约束 |
|---:|---|---|
| 0 | `none` | 无化学键的非对角 pair |
| 1 | `single` | 单键 |
| 2 | `double` | 双键 |
| 3 | `triple` | 三键 |
| 4 | `aromatic` | 芳香键 |
| 5 | `self` | 只允许出现在对角线，是 UAE-3D 的 self-loop sentinel，不是化学键 |

无向化学键会在 `(i,j)` 和 `(j,i)` 各出现一次。因此 full-pair 指标包含大量
`none`，也对每条真实键计数两次；不能单独依靠它判断分子是否完整重构。

## 2. 原子与键指标

设评估集有 `M` 个分子，第 `m` 个分子有 `N_m` 个原子。除特别说明外，
accuracy 均为先汇总所有正确数和总数再相除的 **micro average**，不是先求每个
分子的 accuracy 再平均。

| 报告字段 | 计算 | 含义 |
|---|---|---|
| `atom_accuracy` | `Σ correct atoms / Σ N_m` | 所有原子的类别正确率 |
| `atom_exact_molecules` | `Σ 1[该分子所有原子类别正确]` | 原子类别整分子 exact 的分子数 |
| `bond_full_pair_accuracy` | `Σ correct(i,j) / Σ N_m²` | 六分类 full-pair 微平均正确率 |
| `bond_exact_molecules` | `Σ 1[该分子 N_m×N_m 全部正确]` | 项目的严格 `bond_exact` |
| `off_diagonal_accuracy` | `Σ correct(i,j), i≠j / Σ N_m(N_m-1)` | 排除 self 后的键分类正确率 |
| `none_bond_accuracy` | target 为 `none` 的非对角 pair 中预测为 `none` 的比例 | 无键识别能力 |
| `present_bond_accuracy` | target 为 class 1–4 的非对角 pair 中，预测的 **具体键型** 也正确的比例 | 不只是“有键/无键”二分类 |
| `self_loop_accuracy` | 对角线预测为 class 5 的比例 | self sentinel 正确率 |
| `self_loop_exact_molecules` | `Σ 1[该分子所有对角线均为 5]` | 按分子统计的 self exact |
| `categorical_exact_molecules` | atom exact 且 bond exact 的分子数 | 不包含坐标和 RDKit 检查 |

`bond_full_pair_accuracy` 是与 UAE-3D 官方 evaluator 的 bond type accuracy 最接近的
项目指标：官方实现也对完整 pair tensor 执行 `argmax` 后计算
`correct.sum()/correct.numel()`。`bond_exact_molecules` 是本项目额外增加的更严格标准，
不是论文表格中 bond accuracy 的同义词。

### Wrong unique pairs

阶段报告中的 `wrong_unique_pair_count` 将对称的两个有序非对角 pair 合并为
一个无向 pair：

```text
offdiag_wrong = off_diagonal_total - off_diagonal_correct
diag_wrong    = self_loop_total - self_loop_correct
wrong_unique_pair_count = offdiag_wrong / 2 + diag_wrong
```

这个公式只在预测矩阵已确认对称时具有“unique pair”语义。若预测非对称，
应先判定 representation 失败，不应用该数字代替错误分析。

## 3. 表示和化学有效性

| 报告字段 | 通过条件 | 不代表什么 |
|---|---|---|
| `bond_symmetric` | 预测键矩阵与转置逐项相同 | 不代表价态合法 |
| `off_diagonal_no_self_loop` | 非对角元素不含 class 5 | 不代表真实键型正确 |
| `bond_representation_valid` | 矩阵为方阵、对称，对角全为 5，非对角无 5 | 只是张量结构合法，不是化学有效 |
| `sanitize_valid` | 按预测原子和键构建 RDKit 分子后 `Chem.SanitizeMol` 完全通过 | 不验证坐标几何、点群或 novelty |
| `connected` | sanitize 通过且 `Chem.GetMolFrags` 只有一个 fragment | 不代表键图与 target exact |

RDKit 审计对预测原子设置 `formal_charge=0`、关闭 implicit H，使用预测的
single/double/triple/aromatic 键，不进行价态修复或键类 fallback。当前 U2 只使用已
排除非零形式电荷和自由基的 UAE 严格兼容子集；因此通过这些检查不等于
已实现 2,532 条 canonical 数据的电荷/自由基无损重构。

## 4. 坐标和几何指标

对第 `m` 个分子，项目直接坐标 RMSD 为：

```text
coordinate_rmsd_m = sqrt((1/N_m) * Σ_i ||x_pred(i) - x_target(i)||²)
```

- `coordinate_rmsd_mean_angstrom`：先按分子计算上式，再对 `M` 个分子等权
  平均；
- `coordinate_rmsd_max_angstrom`：所有分子 RMSD 的最大值，通用严格 Gate 使用
  该值；
- `pair_distance_mae_angstrom`：对一个分子所有 `i<j` 原子对，平均
  `abs(||x_pred(i)-x_pred(j)|| - ||x_target(i)-x_target(j)||)`；
- `pair_distance_mae_mean_angstrom`：上述每分子 MAE 的等权平均。

当前 `coordinate_rmsd` 在固定原子索引下直接比较，**不执行 Kabsch/旋转
对齐**。UAE-3D 官方 evaluator 会去质心并对齐后计算 RMSD，因此两者的
RMSD 不能混用或直接比较。`pair_distance_mae` 对整体平移和旋转不变，但它
只是辅助几何指标，不代替直接 RMSD Gate。

## 5. 通用严格重构 Gate

`reconstruction_gate(metrics, coordinate_rmsd_limit=0.05)` 只在下列五项 **全部**
通过时返回 `passed=true`：

| Gate check | 条件 |
|---|---|
| `atom_exact_all` | `atom_exact_molecules == molecule_count` |
| `bond_exact_all` | `bond_exact_molecules == molecule_count` |
| `sanitize_valid_all` | `sanitize_valid_molecules == molecule_count` |
| `connected_all` | `connected_molecules == molecule_count` |
| `coordinate_rmsd_max` | `coordinate_rmsd_max_angstrom <= 0.05` |

任何一项失败，该 tier 的通用严格 Gate 就失败。accuracy 接近 100%、平均
loss 下降或部分分子 exact 都不能替代这个判定。

`bond_representation_valid_all` 没有作为独立 check 重复写入这个函数：当
`bond_exact_all=true` 时，预测已与结构合法的 target 矩阵完全一致。在未达
bond exact 的诊断阶段，仍必须单独报告 representation validity。

## 6. 分阶段 Gate 与放行规则

项目使用 `tier-1 → tier-4 → tier-32 → full train` 顺序，不跳级。

1. **通用严格 Gate** 是 tier 最终验收标准。只有当前 tier 的所有分子同时
   满足第 5 节才算 `PASS_U2_TIER`。
2. **工程 Gate** 只检查代码和运行可信性，例如 loss/gradient 有限、checkpoint 可
   恢复、冻结参数未变和独立运行可复现。它通过后只能放行小规模实验，
   不证明模型质量通过。
3. **阶段冻结 Gate** 用于判断某个局部优化分支是否值得延长。阈值必须在运行前
   写入 stage report，运行后不得根据结果修改。该 Gate 可以比通用 Gate 宽，
   但只决定“继续还是停止这个诊断分支”，不代表 tier 通过。
4. tier-4 未通过通用严格 Gate 前，不得进入 tier-32；tier-32 未通过前，不得
   进入全量训练或 PG condition。

### 当前 hierarchical v4 16-step 的历史冻结 Gate

下表只记录已完成的 raw zero-threshold pilot，不是新实验的默认阈值：

| 检查 | step-16 预冻结条件 |
|---|---:|
| wrong unique pairs | `< 119` |
| none accuracy | `>= 0.9705320600272851` |
| present exact-type accuracy | `>= 0.9256756756756757` |
| self accuracy | `== 1.0` |
| bond exact molecules | `>= 2/4` |
| representation valid molecules | `== 4/4` |
| 已 exact 小分子 | package index `864` 和 `1135` 必须继续 exact |
| coordinate RMSD max | 不高于来源的 `0.14410921931266785 Å` |

该 pilot 的 raw Gate 已失败，不得因为后验 threshold 扫描可得到更好数字而追认
通过。

### 已冻结的全局 threshold calibration protocol

当前实现对每个 checkpoint 只保存一个 scalar threshold，固定语义为：

```text
score = max(logits[single:aromatic]) - logits[none]
present iff score > threshold
score == threshold 时判为 none
```

- fit panel：已用于训练当前 checkpoint 的 tier-4 IID-train 四分子；
- transfer panel：与 fit 互斥的 32 分子 IID-validation，按 C2/C3/S4/D6h 和原子数
  确定性覆盖；
- test/Core-OOD：不读取标签、不拟合、不参与 Gate；
- candidate 集：fit scores 的排序唯一值、`nextafter(min,-inf)` 和零；
- 目标：先最大化逐分子等权的 none/present balanced existence accuracy，再最大化
  最差分子 balanced accuracy，再最小化 existence errors；
- tie-break：优先绝对值更接近零的 threshold，最后选数值更小者；
- 每分子 threshold 在接口和 split policy 中都被禁止；table、candidate 集、split
  和报告均有 SHA-256/fingerprint。

该 protocol 的工程可复现性通过，但实际迁移 Gate 失败。source step-3008 拟合
threshold `-0.400896`，validation balanced existence 提升而 wrong pairs 恶化；step-16 拟合
threshold `0.125497`，wrong pairs 改善而 balanced existence/present exact-type 恶化。两者
都是 0/32 bond exact，因此 `FAIL_GLOBAL_THRESHOLD_TRANSFER_KEEP_TRAINING_STOPPED`。
validation-oracle threshold 只用于估计排序上界，不选择部署 threshold、不进入 Gate、不追认
任何历史运行。

### 冻结 bond capacity/representation 审计

全局 threshold 迁移失败后，项目在不更新 UAE 权重的前提下，对两处 64 维特征拟合辅助
probe：`pair_features=h_i+h_j`，以及官方 `bond_head` 第一层线性变换和 GELU 后、最终
六分类线性层之前的 `hidden_features`。probe fit 使用 128 个 UAE-compatible IID-train
分子，排除训练 checkpoint 的 tier-4 四分子；threshold 和 probe 参数不作修改地迁移到
既有 32 分子 IID-validation 面板。test/Core-OOD 均未使用。

- existence probe：每分子取全部 present unique pairs，并确定性抽取等量 none pairs，使用
  固定 `L2=0.01` 的 float64 logistic regression；
- conditional type probe：只在 target-present pair 上使用固定 `L2=0.01`、类别平衡的
  四分类 ridge least squares；
- `ROC-AUC`：在 validation 全部 unique pairs 上计算，可衡量 existence 排序，不依赖
  threshold；
- `macro_molecule_balanced_existence_accuracy`：先对每分子计算
  `(none accuracy + present recall)/2`，再对分子等权平均；
- `FEATURE_SEPARABLE` 是预冻结诊断标签：validation ROC-AUC `>=0.85`、迁移后的宏平均
  balanced accuracy `>=0.75`，且 none accuracy / present recall 均 `>=0.70`。

正式报告固定 CPU 单线程，两次完整运行字节一致。source step-3008 和 candidate step-16
的 post-GELU probe 均为 `FEATURE_SEPARABLE`：validation ROC-AUC 分别为 `0.88796` 和
`0.88793`，balanced accuracy 为 `0.80041` 和 `0.79893`。相同 split 上，官方六分类
head 的 calibrated balanced accuracy 仅为 `0.72688` 和 `0.72898`。这支持下一步建立
独立 existence head；它不证明 bond type、整分子 exact 或 tier-4 Gate 已通过，也不授权
进入 tier-32。

### 显式 head 工程 Gate

capacity 结论对应的本地 head 将键解码拆为结构 self、独立 existence binary logit 和四类
conditional type logits。首轮 trainer 冻结全部 UAE source 参数，只训练 325 个新 head
参数，并使用独立 checkpoint schema，禁止与旧 reconstruction checkpoint 静默混载。

工程 Gate 比较 CPU 上连续两步和 `step 1 → save → resume → step 2`，要求：

- head、optimizer、RNG、loss history 和 evaluation history 逐项相同；
- source UAE 参数指纹在两条路径中均不变化；
- 所有 loss/gradient 有限；
- checkpoint schema 与旧 schema 隔离；
- 结果必须标记 `ENGINEERING_ONLY`，不得冒充 tier-4 质量通过。

上述检查全部通过，状态为 `PASS_EXPLICIT_BOND_HEAD_ENGINEERING`。`.pt` 使用 `torch.save`
容器，文件字节不作为确定性标准；实际比较的是内部 tensor/state/RNG/history。两步后仍为
`0/4 bond_exact`，只证明训练和恢复链路可信，不证明该架构可以解决质量问题。

### 显式 head 64-step bounded pilot Gate

正式 pilot 在运行前冻结为：从 step-3008 source 重新初始化、只训练 325 个 head 参数、
64 步、batch 4、`lr=1e-4`、每 8 步审计、global existence threshold 始终为 0。工程
step-2 checkpoint 不作为初始化；中间 checkpoint 仅诊断，不能替换最终 step 64。

step 64 必须同时满足：source 指纹不变；atom exact 4/4；bond exact、sanitize、connected
均至少 2/4；wrong unique pairs `<=118`；macro/min molecule-balanced existence 至少
`0.90/0.75`；none accuracy 至少 `0.970532`；present recall 和 present exact type 至少
`0.925676`；conditional type micro accuracy 至少 `0.95`；坐标 RMSD max 不高于
`0.144109 Å`。通过只放行一个 combined tier-4 refinement 设计，不等于 U2 通过；失败则
停止 head-only 分支，禁止延长步数、后验 threshold 或挑选中间 checkpoint。

正式结果为 `FAIL_EXPLICIT_BOND_HEAD_PILOT_STOP_BRANCH`。运行成功且 source 指纹、raw
threshold、原子和坐标检查通过；conditional type micro accuracy 为 `0.95946`。但
macro/min molecule-balanced existence accuracy 为 `0.72588/0.64142`，none accuracy
为 `0.61392`，present existence recall 为 `0.70946`，wrong unique pairs 为 `1,464`，
最终 `bond_exact/sanitize/connected` 均为 `0/4`。这说明失败主因是 off-diagonal bond
existence，而不是 conditional type、原子或坐标。该结论关闭当前 head-only protocol，
不能用 `PASS_EXECUTION` 或连续 loss 下降覆盖。

## 7. 状态字段怎样解读

| 输出 | 准确语义 |
|---|---|
| `status=PASS_EXECUTION` | 脚本完整执行、产出报告；不是质量 Gate |
| `gate_conclusion=PENDING_MORE_TRAINING` | 通用严格 Gate 未通过；这个字样本身不授权继续训练 |
| `gate_conclusion=PASS_U2_TIER` | 该次报告的 tier 通过第 5 节全部五项检查 |
| `PASS_EXPLICIT_BOND_HEAD_ENGINEERING` | 新 head 的冻结、checkpoint 和恢复等价性通过；不是 tier 质量 Gate |
| `FAIL_EXPLICIT_BOND_HEAD_PILOT_STOP_BRANCH` | 冻结的 step-64 质量 Gate 失败；停止当前 head-only 分支，禁止延长、调 threshold、挑中间 checkpoint 或进入 tier-32 |
| stage report `passed=false` / `FAIL_*` | 预冻结阶段 Gate 失败；其停止决定优先于通用的 `PENDING_MORE_TRAINING` 文案 |

因此，“loss 下降 + `PASS_EXECUTION` + `PENDING_MORE_TRAINING`”的正确结论是“运行成功但
质量未通过”，不是“可以自动继续”。

## 8. 当前 Gate 不覆盖的内容

本文档定义的是 **autoencoder 确定性重构 Gate**，不是生成质量 Gate。它当前
不包含：

- UDM latent 无条件采样质量；
- uniqueness、novelty、collision 和样本多样性；
- `actual_pg`、`pg_exact_match`、`pg_compatible` 或 symmetry error；
- Core-OOD 和 train-test 最近邻相似度；
- 对全部 13 元素、Sn、形式电荷和自由基 head 的无损支持。

这些项目需要在 U2 重构通过后，由 U3/U4 和后续 PG-conditioned 阶段的独立
Gate 验收，不能从 `bond_exact` 或 `0.05 Å` RMSD 外推得出。

## 9. 实现与证据入口

- 指标、RDKit 审计和通用 Gate：
  [`generative_model/models/uae3d_reconstruction.py`](../generative_model/models/uae3d_reconstruction.py)
- canonical 到 UAE full-pair target：
  [`generative_model/models/uae3d_bridge.py`](../generative_model/models/uae3d_bridge.py)
- 训练报告和 `PASS_EXECUTION/PASS_U2_TIER` 状态：
  [`generative_model/smoke/train_uae3d_reconstruction.py`](../generative_model/smoke/train_uae3d_reconstruction.py)
- UAE 分阶段执行计划：
  [`generative_model/models/README.md`](../generative_model/models/README.md)
- 当前阶段状态与机器可读证据：
  [项目动态状态.md](项目动态状态.md)
- capacity split 与只读报告：
  [`uae3d_bond_capacity_v1.json`](../generative_model/smoke/splits/uae3d_bond_capacity_v1.json)、
  [`uae3d_bond_capacity_v1.json`](../generative_model/smoke/reports/uae3d_bond_capacity_v1.json)
