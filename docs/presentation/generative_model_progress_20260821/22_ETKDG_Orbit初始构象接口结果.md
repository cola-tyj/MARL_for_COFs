# ETKDG + Orbit 初始构象接口结果

## 1. 实现目标与边界

本阶段实现的是：

`known explicit-H graph + known Target_PG operations/permutations/orbits`

`→ ETKDGv3 multi-start → reference-free orientation → Reynolds projection`

`→ F0.2 UFF+nonbonded repulsion → independent actual-PG audit`。

它不读取真实 XYZ 来生成、对齐或选择候选；真实 XYZ 只用于事后 bond-length MAE 和未对齐
coordinate RMSD。它也不从新 SMILES 推导 symmetry automorphism/orbit，因此仍不是
`Target_PG → new graph+3D`。

## 2. 接口设计

- 直接从 canonical graph 建立 RDKit molecule；数据已经显式含 H，因此禁止 `AddHs`，禁止
  重新解析 SMILES 后重排原子。
- 对每个分子固定 4 个 ETKDGv3 seed，保持 canonical atom index。
- ETKDG 坐标具有任意全局朝向。接口不使用真实 XYZ 做 Kabsch，而是计算 PCA frame，枚举
  24 个 proper signed-axis orientations，选择 Reynolds projection RMSD 最小的方向。
- 投影坐标保持 ETKDG radius of gyration，随后进入已经通过 F0.2 的 500-step
  orbit-constrained UFF+短程非键排斥后端。
- 每个分子的 4 个候选只按最终 UFF+repulsion objective、再按 seed 选优；不读取 actual PG。
- 最终 actual PG 必须由独立 `env_cof/pymatgen` 以 `0.3 Å` tolerance 复验。

## 3. 原子/orbit 对齐与共振语义

E0 的 4 分子 strict typed-graph action 全部成立。扩展到 E1 后，严格检查在 ETKDG 前拦截了
3/32 分子：`cof_000781`、`cof_002436`、`cof_002516`。它们的 connectivity 和元素映射完全
正确，差异只来自 nitro/carboxyl-radical 等价末端 O 的 Lewis 共振式：单/双键以及形式电荷或
自由基标签被对称 permutation 交换。

E1-v1 因此保留为工程失败。E1-v2 新增显式的
`terminal-resonance-compatible-graph-action-v1`：只允许同一 mapped center 下、同元素、
degree-1 末端原子的 `(bond type, formal charge, radical, aromaticity)` **多重集**交换；连接、
元素、中心原子电子特征及多重集必须完全不变。任意 connectivity 改动或一般键型错误仍严格
失败。该规则使 3 个共振表示分子通过，没有修改 canonical graph。

## 4. E0：4 分子低成本 Gate

固定 2 个 C2 + 2 个 C3 IID-validation 分子，每分子 4 seed；包含之前的 package-1490
碰撞压力案例，原子数覆盖 15–60。

| 指标 | 结果 |
|---|---:|
| ETKDG candidate success | 16/16 |
| selected collision-free | 4/4 |
| minimum pair distance | `1.08104 Å` |
| maximum bond-length MAE | `0.045031 Å` |
| maximum operation error | `2.88e-7 Å` |
| optimizer acceptable termination | 4/4 |
| independent analyzer success | 4/4 |
| PG exact / compatible | 1/4 / 4/4 |

actual PG 为 C2h、C2h、C3h、C3；C2h/C3h 是目标 C2/C3 的超群，故 compatible 为成功。
状态：`PASS_ETKDG_ORBIT_E0_ADVANCE_TO_EXPANDED_C23_PANEL`。

## 5. E1：32 分子跨分子 Gate

从 IID-validation 按 `(atom_count, package_index)` 排序后取居中等宽 quantiles，固定
24 个 C2 + 8 个 C3；每分子仍为 4 seed，算法、F0.2 后端和质量阈值均未调整。

| 指标 | 结果 |
|---|---:|
| ETKDG candidate success | 128/128 |
| selected collision-free | 32/32 |
| minimum pair distance | `1.01336 Å` |
| maximum bond-length MAE | `0.054641 Å` |
| maximum operation error | `4.71e-7 Å` |
| optimizer acceptable termination | 32/32 |
| resonance-normalized molecules | 3/32 |
| independent analyzer success | 32/32 |
| PG exact / compatible | 18/32 / 32/32 |

actual PG 分布为 C2 15、C2h 8、D2h 1、C3 3、C3h 4、D3h 1，全部与目标 C2/C3
compatible。状态：`PASS_ETKDG_ORBIT_E1_ADVANCE_TO_DATASET_SCALE_C23`。

报告中的未对齐 reference coordinate RMSD 约为 6 Å，不参与 Gate：输出与数据构象可能是
不同 conformer，且本接口刻意不使用 reference 做旋转对齐。化学局部几何、无碰撞、硬对称
误差和独立 actual PG 才是本阶段判据。

## 6. 可重复性、测试与工件

E0：

- protocol v2 SHA-256：`eb50483df5ae01a4d610a5a08c4c7ac3385411eeca089b05b6f4a86db49b3a4f`；
- geometry SHA-256：`00d350910c6689961eeaf84b584230e60abd2fc7c3f7661d4f9c3680a613c4f9`；
- coordinates SHA-256：`9d72a177456a55c469866b66e4cccf42f840bc1cda0f319655486d0d6e0a2aea`；
- point-group SHA-256：`b71ca7c02fd543d5087d8b4904cf146f447846b2476f4a9107fd1473d85e03c4`。

E1：

- protocol v2 SHA-256：`a9e187f8c75704ca22b921d9d5659a684257f35593854477a2f0598899727a4a`；
- geometry SHA-256：`6dc472a6d4cf4d28d2011dfe8c63d4f77f0d4132f13f1cfff4778d82e257734c`；
- coordinates SHA-256：`5202591b0dd826bed209dde5246ff29fa9c58b6c4036fea97491e0c419fd9e2e`；
- point-group SHA-256：`a14462387db0eeb284a7f1e5551ef03108dafa93a3f15ce7e9a589b08cddb52a`。

E0、E1 坐标在独立临时路径重放后分别得到相同 SHA-256。`env_uae3d` 与 `env_cof`
联合回归均为 12/12。

## 7. 当前结论和下一步

初始构象接口已从“待实现”升级为 32 分子跨分子通过组件。当前已验证链条为：

`known graph + known target orbit action → deterministic multi-start initial 3D`

`→ strict-symmetry force-field relaxation → compatible actual PG`。

dataset-scale C2/C3 E2 已完成并通过：2,232/2,232 分子和 8,928/8,928 候选成功，全部
collision-free、bond-quality 100%，两条 Sn 均成功。独立 pymatgen 审计为 2,223/2,232
analyzer/compatible（99.60%）和 1,280/2,232 exact；9 条 IID-train/C2 analyzer failure
保守计入全 panel 失败。正式状态 `PASS_ETKDG_ORBIT_E2_DATASET_SCALE_C23`。test/Core-OOD
仍未参与开发选择；详细结果和哈希见 [E2 运行协议](23_ETKDG_Orbit_E2运行协议.md)。
