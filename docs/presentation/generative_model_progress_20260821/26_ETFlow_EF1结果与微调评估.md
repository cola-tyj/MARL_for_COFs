# ET-Flow EF1 结果与微调评估

## 1. 阶段结论

EF1 已通过预注册 Gate，状态为
`PASS_ETFLOW_EF1_ZEROSHOT_ADVANCE_TO_FINETUNE_EVALUATION`。它证明官方 `drugs-o3`
ET-Flow checkpoint 可在 32 条未用于当前实验训练的 IID-test COF 分子上，作为
`known graph → initial 3D conformer` 的独立 learned prior，并通过严格显式 H/13 元素接口。

它没有证明：

- ET-Flow 明显优于 ETKDGv3；
- ET-Flow 本身接收或学会了 `Target_PG`；
- 已经可以从 Target_PG 生成 novel graph；
- 可以不经 E3 symmetry action/projection 和 F0.2 backend 直接输出可用对称分子。

## 2. 实验设计

- panel：固定 IID-test 32 条，24 C2 + 8 C3；
- 输入：canonical explicit-H graph，不使用 reference XYZ 选样；
- ET-Flow：官方 `drugs-o3` checkpoint，4 candidates/molecule，50 ODE steps；
- 对照：ETKDGv3 4 candidates/molecule；
- 相同后端：E3 hard projection + 冻结 F0.2 optimization；
- 选样：只使用 final UFF + nonbonded-repulsion objective；
- 点群：在 `env_cof` 中使用 pymatgen 对 final coordinates 独立复算；
- strictness：禁止元素 fallback，禁止读取 stored target action，禁止用 reference 决定 candidate。

## 3. 核心结果

| 层级 | 指标 | ET-Flow / ETKDG 中位比 | ET-Flow 胜率 |
|---|---|---:|---:|
| raw | Kabsch RMSD | 1.03329 | 14/32 |
| raw | pair-distance MAE | 0.87261 | 19/32 |
| projected | Kabsch RMSD | 0.98099 | 19/32 |
| projected | pair-distance MAE | 0.93518 | 20/32 |
| final | Kabsch RMSD | 0.99952 | 21/32 |
| final | pair-distance MAE | 0.99986 | 22/32 |

其他结果：

- 两支 candidate success 均为 128/128；
- ET-Flow final bond-quality、collision-free、optimizer termination 均为 32/32；
- ET-Flow final objective 胜 14/32，中位 objective delta 仅 `4.48e-08 kcal/mol`；
- ET-Flow actual-PG analyzer/compatible/exact 为 32/32、32/32、20/32；
- ETKDG actual-PG analyzer/compatible/exact 为 32/32、32/32、21/32。

最关键的解读是：ET-Flow 在 raw pair-distance 上更好，projection 后仍有小幅优势；
但 F0.2 将两条路线修复到几乎一样的 final geometry/objective。因此 learned prior 是
「可用」而非「已显著替代 ETKDG」。

## 4. 可重复证据

- EF1-v2 protocol SHA-256：`e38afc3705d2ee122a0d0d6119ccdaba412cbf583df04514270153965f9f39aa`
- official checkpoint SHA-256：`a24ae9a1fed2708696929308ed1dc10ab167fd66a2d51c44a4afb6c11badccb2`
- geometry report SHA-256：`1d2350e89c1c29e970321f485c74aac8653480489c7feb41d7947f9783682f8e`
- coordinates NPZ SHA-256：`1dbf131e88b2fc43e80bf40f5e4757f802a8ee892b68e8c8401b0cac371bc221`
- point-group report SHA-256：`60a59d2997bc69d250801da9dd96e98fc3077e817d9167e685dfaa156da92c7f`

## 5. 微调准入决策

EF1 放行的下一步是 **EF2 有界微调评估设计**，不是立即开始长训。原因是：

1. final 结果已被强后端收敛，普通 COF domain adaptation 的边际收益可能很小；
2. canonical XYZ 来自 ETKDGv3 流程，直接拟合 XYZ 容易学到同源偏置；
3. 数据是 one-conformer-per-SMILES，不足以学习完整构象分布；
4. 官方 ET-Flow 无 Target_PG condition，普通微调只能改善构象先验，不能单独完成指定点群。

## 6. 推荐的 EF2 分段

### EF2-0：先冻结微调协议

- 只使用 IID-train 训练、IID-validation 选择 checkpoint；
- EF1 的 32 条 IID-test 不再参与调参，仅在最终一次决策时复用；
- 以 official `drugs-o3` 为初始 checkpoint，保留原始 SHA；
- 评估 zero-shot frozen baseline、普通 COF-domain fine-tune 与 symmetry-aware variant；
- 训练选择不得读取 test reference XYZ。

### EF2-A：低成本 domain-adaptation Gate

先做小面板、小步数微调，目标限定为改善 raw conformer prior：

- raw pair-distance/bond geometry 改善；
- projection displacement 降低；
- projected collision 不退化；
- analyzer/compatible 经相同 E3+F0.2 后不退化；
- Sn 和 13 元素 strict forward 不退化。

若不能在独立 validation 上稳定超过 frozen zero-shot，立即停止 ET-Flow 微调，继续使用
ETKDG/E3/F0.2 作为已通过的 known-graph baseline。

### EF2-B：仅在 EF2-A 通过后评估

将 Target_PG embedding/群操作上下文注入 ET-Flow，或加入 endpoint group-consistency /
projection-residual loss。这属于新的 PG-conditioned architecture，不应与官方 zero-shot ET-Flow
结果混写。即使它通过，`Target_PG → novel 2D graph` 仍是独立 E3-C 任务。

## 7. 当前主线边界

```text
E3-C: Target_PG → novel valid 2D graph             （尚未完成）
EF2:  known graph → improved initial conformer      （获得有界评估资格）
E3-A/B + E2/F0.2: graph + Target_PG/action → 3D      （C2/C3 已通过）
```

下一步应先实现和冻结 EF2-0，而不是立即用 2,532 条数据长训。

## 8. 2026-08-25 执行顺序更新（保留上述旧计划）

项目创新主线已进一步明确为 PG-conditioned ET-Flow。因此上述「先做普通 domain
adaptation，再做 PG-conditioned variant」仍作为历史计划保留，但新执行顺序为：

1. 直接实现 `Target_PG + operation/permutation/orbit + endpoint symmetry loss`；
2. 普通 COF fine-tune 降为必要对照组，不再阻塞创新模型；
3. 主要能力在 hard projection 之前的 raw coordinates 上评估；
4. deployment 的 E3/F0.2 仅作为另行报告的安全后端。

EF2-0 已实现并通过官方 checkpoint 工程 Gate，EF2-A 256-step adapter-only
训练协议已冻结待用户在 tmux 运行。详见
[27_PG_Conditioned_ETFlow_EF2实现.md](27_PG_Conditioned_ETFlow_EF2实现.md)。
