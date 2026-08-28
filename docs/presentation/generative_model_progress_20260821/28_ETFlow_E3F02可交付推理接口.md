# ET-Flow + E3/F0.2 可交付推理接口

## 1. 目标与结论

当前已经实现一条可执行的：

```text
known canonical molecular graph + requested Target_PG (C2/C3/S4/D6h)
    -> official ET-Flow learned initial conformers
    -> graph-only cyclic action recovery
    -> E3 Reynolds/orbit hard-symmetry projection
    -> F0.2 orbit-constrained UFF + short-range repulsion
    -> final.xyz
```

它已在 C2/C3 的 64 条新鲜 IID-test 固定面板上达到 64/64 compatible，并在 canonical v2
全部稀有目标样本（S4=38、D6h=14）上达到 52/52 geometry、52/52 independent compatible。
因此四种受支持点群均已有冻结面板证据。给定已知分子图和请求点群，接口输出经独立
pymatgen 审计、至少包含该目标群作为子群的三维坐标。它不使用参考 XYZ、stored
permutation/orbit 或 ETKDG。

必须保留的科学边界是：ET-Flow 在这里是 learned initial-conformer prior；最终对称性由 E3
hard projection 和 orbit-constrained refinement 严格保证。因此这不是“ET-Flow backbone 已经
学会 raw Target_PG-conditioned generation”的证据。EF2-D4 已正式关闭的 learned raw 分支
不因本结果而改判。

## 2. 输入、输出与失败语义

输入：v2 canonical graph 的原子序数、形式电荷、自由基电子和 typed sparse bonds，另加用户请求
的 `C2`、`C3`、`S4` 或 `D6h`。显式 H 和 Sn 原样保留，不做元素 fallback。

输出目录包括：

- `raw.xyz`：官方 ET-Flow 生成、尚未施加 Target_PG 的初始构象；
- `projected.xyz`：E3 Reynolds/orbit projection 后坐标；
- `final.xyz`：F0.2 orbit-constrained UFF+repulsion 后的交付坐标；
- `coordinates.npz`：原子/键、operation、permutation、orbit 和三阶段坐标；
- `report.json`：候选、几何、模型、协议与输入范围审计；
- `point_group_report.json`：由独立 `env_cof/pymatgen` 产生的 actual-PG 审计。

每个分子固定生成 4 个 ET-Flow ODE 候选，每个 50 steps。只从满足以下条件的候选中选择最终
UFF+repulsion objective 最低者：优化终止可接受、最小任意原子间距 `>=0.6 Å`、最大群操作误差
`<=1e-5 Å`。若 graph 不存在请求群的非平凡 cyclic automorphism，或 4 个候选全部失败，则
严格报错，不静默降级为其他点群、ETKDG 或参考构象。

## 3. 冻结协议与测试

- 当前四点群推理协议：`generative_model/inference/etflow_e3f02_protocol_v5.json`
- protocol SHA-256：`cfcb872cf4d8902f2ccdba1bd2738b6476eba27b7a7be9e5108e7e0d4a2678d2`
- 官方 `drugs-o3.ckpt` SHA-256：`a24ae9a1fed2708696929308ed1dc10ab167fd66a2d51c44a4afb6c11badccb2`
- 相关单元测试通过：reference-free graph action、S4/D6h 多 action、S4 惯性筛选、严格候选
  计数及 Sn/H XYZ 保真。

单分子推理与独立审计命令：

```bash
cd /home/tianyajun/MARL_for_COFs
conda activate env_etflow
CUDA_VISIBLE_DEVICES=4 python -m \
  generative_model.inference.generate_etflow_symmetric_xyz_v5 \
  --package-index 1102 --target-pg S4 --seed 2026084202 \
  --protocol generative_model/inference/etflow_e3f02_protocol_v5.json \
  --output-dir generative_model/runs/etflow_e3f02_examples/s4_001102 \
  --device cuda

conda activate env_cof
python -m generative_model.inference.audit_etflow_e3f02_point_group \
  --run-dir generative_model/runs/etflow_e3f02_examples/s4_001102
```

## 4. 未见样本结果

以下分子均属于 IID-test，且不在既有 EF1 32 分子面板中：

| 输入 | 请求 PG | actual PG | exact | compatible | min pair (Å) | max op error (Å) | 结论 |
|---|---|---|:---:|:---:|---:|---:|---|
| `cof_000902`, index 902, 12 atoms | C2 | C2 | 是 | 是 | 0.991302 | `2.78e-17` | PASS |
| `cof_001160`, index 1160, 12 atoms | C3 | C3h | 否 | 是 | 1.013026 | `1.88e-15` | PASS；生成了 C3 的超群 |
| `cof_002400`, index 2400, 15 atoms；canonical label=C3 | C2 | D3h | 否 | 是 | 1.157395 | `1.45e-20` | PASS；反事实请求被满足 |

第三例说明输入请求可以覆盖 canonical label，但 actual D3h 同时含 C2/C3 子群，因此它不能单独
证明“同一 graph 对 C2/C3 产生互斥、可区分的 raw learned response”。另一次
`cof_001160: C3 -> requested C2` 尝试中，4 个候选均未通过严格几何 Gate；该失败被保留，说明
当前成功率尚未做 dataset-scale 量化，也不存在任意 graph/PG 组合必成功的承诺。

## 5. 当前可声明与不可声明

可以声明：

- 不依赖 ETKDG，从 learned ET-Flow prior 出发完成 known graph + requested C2/C3 -> XYZ；
- final coordinates 由独立 pymatgen 重算通过 requested-PG compatible 与 collision Gate；
- graph action 在不读取参考 XYZ/stored symmetry action 的条件下恢复；
- 失败严格显式暴露。

不可声明：

- ET-Flow raw output 本身已经响应 Target_PG；
- 不用 hard projection 也能获得正确点群；
- 已证明 Core-OOD 的总体成功率；
- 已解决 novel 2D graph/SMILES generation。

## 6. C2/C3 批量结果与 S4/D6h 扩展

C2/C3 的新鲜 IID-test 面板含 48 条 C2 与 16 条 C3，generation/geometry/独立 analyzer/
compatible 均为 `64/64`，exact 为 `33/64`。因此 C2/C3 composite route 已从示例提升为固定
面板证据。

S4/D6h 的 graph-only action 已在全部 52 条 canonical 分子上恢复成功。D6h 不能只取任意一个
合法 graph automorphism：不同合法 action 与 ET-Flow raw conformer 的相位/轨道对应可能不匹配，
造成 Reynolds projection 原子重叠。v3 固定恢复最多 64 个 graph-only D6h action，与 4 个 raw
构象形成 256 个组合，用不读取 reference XYZ 的几何指标排序，仅优化前 8 个。正式 index 858
结果为 `actual_pg=D6h`、exact/compatible、最小间距 `0.912665 Å`。

全量稀有点群的初始协议在坐标生成前冻结。v1/v2/v3 均保留，不覆盖历史结论：

- 面板：S4=38、D6h=14，canonical v2 全覆盖；
- v1：单 S4 action，geometry 仅 21/52；保留为 action-selection 失败基线；
- v2：S4/D6h 多 action 后 geometry 52/52，但独立 compatible 43/52；9 条 S4 落入
  pymatgen `eigen_tolerance=0.01` 的近球形惯性退化分支；
- v3：仅新增 reference-free 的 S4 mass-weighted inertia separation `>=0.012` 候选 guard，
  其余样本、seed、checkpoint、action/raw 预算、F0.2 与 Gate 全部保持不变；
- v3 协议：`generative_model/inference/etflow_e3f02_rare_targets_protocol_v3.json`；
- v3 SHA-256：`290b2127b079da2a27432e0f7ab225353791a0fc6e06b2ec7c63cee1eb73e2a1`；
- 运行：逐分子 JSON/NPZ、协议 SHA 校验、断点续跑、完整 52 分母；
- Gate：overall/S4/D6h geometry 与 independent compatible 分别预注册，不允许事后改阈值。

v3 正式结果：geometry S4 `38/38`、D6h `14/14`；独立 analyzer `52/52`，compatible
`52/52`。actual PG 为 S4 17、D2d 21、D6h 14；D2d 包含请求的 S4 子群，因此按项目冻结的
subgroup-compatible 定义判为成功。S4 最小选中惯性分离为 `0.0166247`，高于 0.012 guard。

## 7. 最终完成审计与下一阶段

四点群完成审计：

- 报告：`generative_model/smoke/reports/etflow_e3f02_completion_v1.json`；
- SHA-256：`84268a990af4e2ed50e8a35480467266fa62dce1525f04b46f198a937f068abf`；
- 状态：`PASS_ETFLOW_E3F02_KNOWN_GRAPH_TARGET_PG_TO_XYZ_DELIVERABLE`；
- 固定证据总计 116 条：C2 48、C3 16、S4 38、D6h 14，各组 geometry 与 compatible 均为 1.0；
- 单分子 v5 示例已实际写出 well-formed `final.xyz`，并通过 geometry 和独立 point-group 审计。

这关闭了“已知 canonical graph + 请求 C2/C3/S4/D6h → compatible 3D XYZ”的工程缺口。
下一阶段不再重复优化该后端，而应转向 `Target_PG → novel graph/SMILES`，再把生成 graph 接入
本接口。若要加强 3D 端证据，可另做 Core-OOD/反事实 graph-PG 组合的外推审计，但它不是当前
交付接口通过的前置条件。

## 8. 保留的旧计划与负结果

旧计划曾要求先运行全量 52 条 rare-target Gate；该计划现已按原阈值完成，结果见第 6–7 节。
raw learned Target_PG response 仍作为独立、已停止的研究分支：EF2-D4/operation-orbit 等实验
没有证明“不依赖 hard projection”的条件生成，不能用 composite route 的成功改判。若未来重启，
应采用新架构与新预注册协议，而不是继续延长旧 checkpoint。
