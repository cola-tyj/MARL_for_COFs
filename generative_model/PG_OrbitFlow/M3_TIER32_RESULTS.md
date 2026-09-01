# M3 Tier-32：非平面、手性、C2/C3 全学习内坐标闭环

## 1. 结论与边界

M3 已在固定的 32 分子封闭面板上通过 prediction Gate、无 oracle 3D reconstruction Gate 和
独立复现 Gate，正式状态为 `PASS_M3_TIER32`。这是 PG-OrbitFlow 当前第一个同时覆盖非平面
torsion、active chirality、C2/C3 分层和多起点解码的完整闭环。

该结果仍是 **Tier-32 训练面板内的记忆/容量 Gate**，不是 IID-test、Core-OOD 或新图泛化结果；
不能据此声称模型已经能够对 unseen molecular graph 生成可靠 3D。模型也尚未生成新的 2D
分子图。

## 2. 固定面板与输入输出

- 数据：canonical v2 的 IID-train 子集，32 个分子，C2/C3 各 16 个；严格包含已通过的
  Tier-16。
- 几何支持：157 个 nonplanar torsion orbit、87 个 active chirality、30 个 local rotor
  group。
- 模型输入：immutable 2D graph、元素/电荷/自由基、Target_PG、完整 `R_g/P_g` action、
  atomic orbit、quotient graph、relative group-phase、graph-centralizer automorphism。
- IC 输出：bond-orbit length、angle-orbit cosine、torsion-orbit `(sin φ, cos φ)`。
- shape 输出：16 个 long-range pair-distance quantiles，条件由 quotient IC embedding 与
  512-bit、radius-4、atom-order-invariant counted WL fingerprint 共同提供。
- 3D 输出：stabilizer-aware orbit parameters 经 exact group lifting 得到全原子 Cartesian
  coordinates。

decoder 不读取 target Cartesian coordinates，不使用 oracle bond/angle/torsion/local-pair/
chirality，不调用 ETKDG、posthoc hard projection 或 F0.2。Target coordinates 只在训练时定义
IC/shape 标签和在独立评估中计算误差。

## 3. 训练和修复路径

### 3.1 M2 v5 parent

M2 v5 在 Tier-16 上用 operation-coupled、terminal-sibling permutation-invariant circular set
loss 训练 2,048 steps。训练单位为 6 个 full-batch rotor sets，共 12 个 torsion views；只训练
50,210 个 set/head 参数。graph-centralizer 等价匹配后 16/16 分子及 47/47 active chirality
通过，随后独立复现通过。

### 3.2 M3.1–M3.2：表示与平均 loss 不足

- M3.1 把 graph automorphism head 扩展到 2--6 slots，并要求其与所有 Target-PG action
  对易；最差 torsion/nonplanar 降至 `24.0014°/2.31794°`，仍失败。
- M3.2 冻结 feature、bond、angle，只训练 base torsion head 与 automorphism head；32 分子
  full batch、512 steps。最差 torsion 降至 `6.96557°`，但平均 orbit loss 仍会掩盖少数
  hard molecules。

### 3.3 M3.3：Gate-aligned worst-molecule refinement

M3.3 从 M3.1 parent 独立训练 512 steps，batch size 32，学习率 `1e-3`；总参数 419,720，
只训练 torsion 与 automorphism heads 的 100,516 个参数。loss 为：

```text
mean molecule torsion
+ worst molecule torsion
+ worst molecule nonplanar torsion
```

prediction Gate 首次全部通过：bond `0.002910 Å`、angle `1.59751°`、torsion
`0.432139°`、nonplanar torsion `0.526255°`，且 C2/C3 分层均通过。

### 3.4 原 decoder 失败与错误分支排除

原 16-start decoder 仅 package 1054 失败：learned IC energy 选择 start 2，但只有 start 0
落入正确全局构象 basin。只读候选审计证明问题是局部 IC energy 缺少全局形状辨识，而不是
IC prediction、group lifting 或碰撞项失效。

- M3.4 试图预测 1,070 个全局 pair targets，但发现 310 处完全相同输入对应不同 target，
  表示发生 alias，停止该分支。
- M3.5/M3.5b 仅用 pooled quotient-IC embedding 预测 shape quantiles，最差 MAE 分别为
  `0.09225/0.08958 Å`，均未过 `0.03 Å`，说明容量/图身份信息不足。

### 3.5 M3.6：atom-order-invariant graph fingerprint + global shape

M3.6 加入 normalized counted WL fingerprint（radius 0--4，SHA-256 fold 到 512 bits）。32 个
面板分子的 fingerprint 全部唯一，没有 input-target conflict。shape head 总模型 670,104
参数，只训练 250,384 个 shape-head 参数；32 分子 full batch、2,048 steps、学习率 `1e-3`，
loss 同时惩罚 mean 与 worst molecule MSE。

shape quantile 最差 MAE 为 `0.004695 Å`，通过 `≤0.03 Å` Gate。decoder 对每个分子的 16 个
确定性起点使用：

```text
selection score = learned IC final energy + 10 × predicted shape MSE
```

该 selector 不读取 target coordinates。正式 v2 重新固定为原 M3 decoder seed `20262129`，
消除了第一次执行时的 seed drift。

## 4. 冻结 Gate 与最终数值

| 指标 | 阈值 | 最终最差值 | 结果 |
|---|---:|---:|---|
| bond-orbit prediction MAE | ≤0.03 Å | 0.002910 Å | PASS |
| angle-orbit prediction MAE | ≤5° | 1.59751° | PASS |
| torsion-orbit prediction MAE | ≤2° | 0.432139° | PASS |
| nonplanar torsion prediction MAE | ≤2° | 0.526255° | PASS |
| reconstructed bond MAE | ≤0.03 Å | 0.0037165 Å | PASS |
| reconstructed angle MAE | ≤5° | 1.532495° | PASS |
| reconstructed torsion MAE | ≤2° | 1.765254° | PASS |
| reconstructed nonplanar torsion MAE | ≤2° | 1.743004° | PASS |
| ring closure MAE | ≤0.03 Å | 0.0035736 Å | PASS |
| Kabsch RMSD | ≤0.2 Å | 0.0666573 Å | PASS |
| collision-free fraction | =1.0 | 1.0 | PASS |
| chirality preserved | =1.0，87 cases | 1.0，87/87 | PASS |
| group-action max atom error | ≤1e-6 Å | 3.1351e-7 Å | PASS |
| C2/C3 分层 | 各自全过 | 各自全过 | PASS |

这里的 `raw_reconstruction_gate` 记录“任意未选择起点”的总体分布，按预期失败；正式 Gate
评估的是不访问 target 的 learned selector 所选结果。不能把 raw candidate pool 的失败写成
最终 decoder 失败。

## 5. 独立复现与测试

独立脚本重新运行全部 selected starts，并验证：32/32 起点选择一致、coordinates 最大绝对差
`≤1e-5 Å`、energy/shape score 差 `≤1e-8`，且原 reconstruction Gate 再次全部通过。状态为
`PASS_M3_REPRODUCIBILITY`。

PG-OrbitFlow 全量单元测试为 53/53 通过；新增测试覆盖 WL fingerprint 的 atom-order
invariance、非法设置严格失败、parent 冻结边界以及 shape quantiles 的有限性和单调排序。

## 6. 关键证据和 SHA-256

| artifact | SHA-256 |
|---|---|
| M3.6 shape protocol | `5441bcf03c808ecd74f41ee072083a0490caa4156c538c802f698d67f6038a90` |
| M3.6 shape report | `b0d0dfbebf3bffa3dc9316d57fe42c6c6aa6cb647f0b3b2ea7780cb2ea7bacf0` |
| M3.6 shape checkpoint | `243f684cc24061d10fa53734304e991128c7fd2e158a8e6d8330b9edc85cc1ab` |
| final decoder protocol v2 | `c36d00d9ab9e8f1c9eb76bf738be1d9c24d09b4c630b5a5a59292898e7652109` |
| final M3 report | `08cfa3129bcb37659fe46c643cc08cebe62b3390c0e1dda5d30d8ffd3a203319` |
| final coordinates | `a3b7d6237366145335d0e4086cf5df2c7773fdd5ee4a2b1c8f598cc6fdf30acc` |
| final predictions | `8c6e77f32acf1175936c6f9b70670f3947d81bce7aae2c9f3c2302a2b15fa739` |
| independent reproduction report | `0f29102ba1b802e1106bc6d516f3c373b09717e836e3d44e61f45b9521705161` |

正式入口：

- `configs/m3p6_shape_decoder_v2.json`
- `runs/m3p6_shape_decoder_v2/report.json`
- `reports/m3_reproducibility_v1.json`
- `global_shape_fingerprint_model.py`
- `m3p6_fingerprint_shape_training.py`
- `m3p6_shape_decoder_gate_v2.py`
- `audit_m3_reproducibility.py`

## 7. 下一阶段

下一步应冻结当前 Tier-32 checkpoint 和 Gate，不再在相同 32 分子上调参。先做严格未见图的
IID-validation 小面板，只评估不训练；若失败，再扩大 IID-train 做 dataset-level training，
不能用 validation/test 反向选择结构。只有 unseen IID 通过后，才进入 Core-OOD 和 S4/D6h。
