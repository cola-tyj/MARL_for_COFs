# Graph Action E3-A / E3-B 结果

## 1. 本阶段解决的问题

E2 已证明：给定 canonical graph、`Target_PG` 和已知 target permutation/orbits，冻结的
ETKDG + Reynolds projection + F0.2 后端可以在 C2/C3 开发域稳定生成 3D。但是，对一个新
graph 并不会天然拥有原子 permutation/orbits；因此 E3 首先去掉对数据集中已存 target
action 的依赖。

E3 分成两个有界步骤：

1. **E3-A**：`canonical graph + Target_PG → operation/permutation/orbits`；
2. **E3-B**：将 E3-A 恢复的 action 接入冻结 E2 后端，验证
   `known graph + Target_PG → recovered action → 3D`。

这两个步骤均为确定性接口/算法 Gate，不训练模型，也不产生 checkpoint。

## 2. E3-A 的严格输入边界

恢复过程只加载原子序数、形式电荷、自由基电子、稀疏无向 bond index/type 和
`Target_PG`。实现使用独立的 graph-only loader，不能读取：

- XYZ、positions 或 centroids；
- `symmetry_data.npz` 中已存的 target permutation/orbits；
- actual point group；
- reference 3D geometry。

当前只支持 C2/C3。未知元素、不连通图、不支持的点群、无法找到完整循环 automorphism、
搜索超过上限等情况均严格失败，不做元素 fallback；Sn 始终保持原子序数 50。

## 3. E3-A 方法

1. 按元素、形式电荷、自由基、芳香性、度数和局部键环境建立带属性 graph；
2. 对同元素 terminal Lewis-resonance 表示采用冻结的多重集等价语义，不修改 canonical graph；
3. 在重原子 graph 上进行 1-WL refinement，得到候选等价类和可移动原子上界；
4. 使用带完整 C2 二循环或 C3 三循环锚点的 NetworkX VF2 搜索 graph automorphism；
5. 按“移动重原子数、移动总原子数、canonical order”确定性选出 action；
6. 在被映射的重原子邻域内按 rank 保序提升显式 H；
7. 构造标准 z 轴 C2/C3 操作矩阵、各次幂 permutation 和 atomic orbits，并重新验证完整
   显式 H graph、元素、键类型与群闭包。

完整循环锚点很关键：朴素全 automorphism 枚举会在高对称图上爆炸；单节点锚点仍有两个
C3 图超过 10,000 次候选。完整三循环锚点后，最难的两个样本分别在 9,219 和 9,223 次
匹配内达到 1-WL 可移动上界。

## 4. E3-A 全量结果

冻结面板为 v2 IID train/validation 的全部 C2/C3，共 2,232 个分子；test/Core-OOD 未用。

| 指标 | 结果 |
|---|---:|
| graph-only action recovery | 2,232/2,232 |
| 完整显式 H graph 独立验证 | 2,232/2,232 |
| 达到 1-WL 可移动上界 | 2,232/2,232 |
| mean moved-atom fraction | 0.995432 |
| minimum moved-atom fraction | 0.55 |
| Sn strict recovery | 2/2 |
| maximum VF2 yielded matches | 9,223 |
| median yielded matches | 1 |
| terminal-resonance normalized | 2 个分子 |

正式状态：`PASS_GRAPH_ACTION_E3A_GRAPH_ONLY_C23_ADVANCE_TO_3D_HANDOFF`。

恢复 action 与数据中已存 action 的逐项相等为 980/2,232（43.91%）。该值不作为失败指标：
一个高对称 graph 往往存在多个同样合法的 C2/C3 子群 action；E3-A 选择的是确定性的
最大移动 action，而不是复刻原 XYZ 标注选择的某一个共轭/替代 action。所有恢复 action
均通过完整 graph automorphism 验证。

## 5. E3-B：恢复 action 到 3D 的交接

E3-B 冻结 16 分子 stress panel：C2/C3 各 8 个，包含两条 Sn、两个高搜索 C3、历史 E2
analyzer failure 以及 recovered action 与 stored action 不同的案例。每个分子使用 4 个
固定 ETKDG seed，共 64 个候选；真实 XYZ 仅用于事后 bond-length/RMSD 指标，不参与初始
构象、方向或候选选择。

几何 Gate 结果：

| 指标 | 结果 |
|---|---:|
| molecule / candidate success | 16/16；64/64 |
| collision-free | 16/16 |
| maximum bond-length MAE | 0.039498 Å |
| minimum pair distance | 1.043290 Å |
| maximum operation error | 7.54e-15 Å |
| Sn success | 2/2 |

独立 `env_cof`/pymatgen point-group audit：

| 指标 | 结果 |
|---|---:|
| analyzer success | 16/16 |
| target-PG compatible | 16/16 |
| target-PG exact | 11/16（68.75%） |
| actual C1 | 0/16 |

5 个非 exact 输出均为合法超群（C2h、D2h 或 C3h），因此 compatible 仍为 100%。正式状态：
`PASS_GRAPH_ACTION_E3B_RECOVERED_ACTION_TO_3D`。

## 6. 当前能够与不能够声称的结论

现在可以声称：在现有 C2/C3 开发域中，**不读取 XYZ 和预存 action**，仅从已知 canonical
graph 与 `Target_PG` 即可恢复合法 permutation/orbits，并驱动冻结 3D 后端；2,232 条 graph
action 全量通过，16 条 stress handoff 的独立 PG compatible 为 16/16。

仍不能声称：模型能够从 `Target_PG` 生成新的 SMILES/graph，也不能把 E3-A/E3-B 称为
learned generative model 或 dataset-level fine-tuning。当前剩余主缺口已经收敛为
`Target_PG → novel valid 2D graph`。

## 7. 下一阶段 E3-C

下一步冻结 symmetry-by-construction 的 2D quotient-graph/graph-grammar baseline，先生成
50 个 C2 与 50 个 C3 候选。建议 Gate 至少包含：

- RDKit sanitize、connected、显式价态与 13 元素严格通过；
- 对训练集 canonical SMILES 的 novelty 与批内 uniqueness；
- E3-A 能恢复目标阶完整 action；
- E3-B/E2 后端无碰撞且 independent `pg_compatible=true`；
- 单独报告 exact/compatible、元素分布、尺寸分布和最近邻 Morgan/Tanimoto。

该 Gate 通过后，才讨论用条件图生成模型替代/扩展规则 baseline；不再回到旧 UAE/MiDi
联合重猜键和坐标。

## 8. 冻结证据与 SHA-256

| 产物 | SHA-256 |
|---|---|
| E3-A protocol | `e97f2512a7cb6daf64cfeea1087aab874afe53e552a9e681f7781dc4911bf725` |
| E3-A actions NPZ | `b76bea3d95200bde0770605a21c5bad3ae539b10400bcce36b225777276618f3` |
| E3-A recovery report | `3f60314f390f4f80c8e589e20595860360bb2ca89c611144d6b8711d8f2d9c74` |
| E3-A independent audit | `12d1ddc07ba2fb117ed027cb305fb39f1a51902fd85966594a33459451abd6fa` |
| E3-B protocol | `a388d5506053ff26cb10b11fce50e86a3471f2de18dc7fdc1b7b84288056e6da` |
| E3-B coordinates NPZ | `e4270c7bca1a017a4808290d1193bbcfb9f86bf0fcc1088917cb707203d3681a` |
| E3-B geometry report | `5712829d7fe7e511e054ef5be6d12b96e7e1ba44c1e88f730372f4603349f750` |
| E3-B point-group report | `aeb38f1b5e658b7a25bad7192570386577ef51b6f5bcaf48255354a0f972fd38` |

冻结产物回归测试已在 `env_cof` 与 `env_uae3d` 各通过 8/8。
