# M2.1：phase-aware Tier-16 训练结果与 torsion 失败诊断

## 1. 正式结论

M2.1 已按冻结协议完整训练 2,048 steps，并在进入 3D decoder 前由 prediction Gate 拒绝：

```text
FAIL_M2P1_PREDICTION_GATE_STOP_BEFORE_DECODER
```

这不是全局的 bond/angle 学习失败。键长和键角均通过，失败集中在带局部可交换末端原子的
非平面扭转标签。只读诊断状态为：

```text
DIAGNOSED_M2P1_LOCAL_EXCHANGEABLE_TORSION_COLLAPSE
```

因此不应仅增加训练步数、提高 torsion loss 权重或放宽 `2°` Gate。下一项 M2.2 应把 NH₂、
CF₃、CH₃ 等局部 rotor 的 atom-labelled torsion 改为 permutation-invariant set/rotor 表示，
同时保留点群 action 和 exact group lifting。

## 2. 训练数据、初始化与预算

| 项目 | 冻结值 |
|---|---:|
| 分子数 | 16 |
| 点群组成 | C2 × 8，C3 × 8 |
| 与 Tier-4 的关系 | 严格嵌套原4条，新增12条 |
| 非平面 torsion orbit | 78 |
| active chirality | 47 |
| 最大原子数 | 21 |
| optimizer steps | 2,048 |
| batch size | 4（每批 C2 × 2、C3 × 2） |
| molecule exposures | 8,192；每条恰好512次 |
| optimizer | AdamW，lr `1e-3`，weight decay 0 |
| gradient clip | 5.0 |
| 参数量 | 318,724 |

模型从 M2 Tier-4 checkpoint 只继承 44 个 quotient message-passing backbone tensor，合计
200,736 个参数。由于 bond/angle/torsion head 新增了 9/18/27 维 action-only group-phase
feature，三个 head 均按固定 seed 重新初始化；optimizer 不继承。

训练损失保持：

```math
L=10L_{bond}+2L_{angle}+2L_{torsion},
\qquad L_{torsion}=1-\langle \hat u_\phi,u_\phi\rangle.
```

输入不含 XYZ。group-phase feature 只由配对的 `R_g/P_g` 构造；训练和 prediction Gate 都不
使用 ETKDG、hard projection 或 F0.2。

## 3. 冻结 Prediction Gate

| 指标 | 阈值 | 最差结果 | C2 最差 | C3 最差 | 判定 |
|---|---:|---:|---:|---:|---|
| bond-orbit MAE | ≤ 0.03 Å | `0.006317 Å` | `0.006317` | `0.006270` | PASS |
| angle-orbit MAE | ≤ 5° | `1.5452°` | `1.5452` | `1.4959` | PASS |
| torsion circular MAE | ≤ 2° | `50.9257°` | `38.6244` | `50.9257` | FAIL |
| nonplanar torsion circular MAE | ≤ 2° | `84.3727°` | `55.7798` | `84.3727` | FAIL |
| C2、C3 分层均通过 | true | false | false | false | FAIL |

程序依照冻结协议在 prediction Gate 后停止，没有运行 16 分子 × 16 起点的 decoder，也没有
产生 reconstruction PASS/FAIL。不能把“decoder 未运行”写成 decoder 失败。

训练里程碑也显示同一边界：bond/angle 持续收敛，而 nonplanar torsion 最差误差从 step 512 的
`90.76°` 仅降到 step 2,048 的 `84.37°`。这不是最后几十步的随机波动。

## 4. 失败集中在哪里

16 条分子中 10 条的全部轨道已通过 `2°`；只有以下6条失败：

| package index | PG | 局部基团 | failed torsion orbit | 分子 torsion MAE | 最大 orbit error |
|---:|---|---|---:|---:|---:|
| 892 | C2 | NH₂ | 4 | `20.55°` | `56.23°` |
| 875 | C2 | CF₃ | 6 | `38.62°` | `132.57°` |
| 1075 | C2 | NH₂ | 4 | `15.32°` | `67.29°` |
| 1768 | C2 | NH₂ | 5 | `15.06°` | `66.40°` |
| 1143 | C3 | CF₃ | 5 | `50.93°` | `145.24°` |
| 2403 | C3 | CH₃ | 7 | `41.76°` | `170.79°` |

总计184个 torsion orbit 中31个超过 `2°`：

- 30/31 位于“多个 orbit 得到相同预测角”的塌缩组；
- 30/31 在最终 checkpoint 的 torsion head 第一层之前已经具有逐元素完全相同的输入向量；
- 29/31 可进一步由“同一中心键、只替换同元素末端原子”直接解释；
- 例如 CF₃ 的三个 F 在2D图中可交换，但当前标签按 F 的 canonical atom index 分别要求约
  `0°/+120°/-120°`；permutation-equivariant head 不应凭任意编号区分它们，因而会输出同一
  circular mean。

这也说明此前“phase-aware exact feature conflict = 0”只排除了目标点群 coset phase 的精确
别名，没有覆盖分子图自身的局部 automorphism/stabilizer。点群 `G` 与局部同元素置换群是两个
不同层级，两者都必须进入 torsion 表示和 Gate。

## 5. M2.2 的受控修复边界

下一实验只改变 torsion 标签的局部置换语义：

1. 从 canonical atom/bond graph 严格求局部 rotor 的 equal-element automorphism slots；不能用
   XYZ 或 atom index 猜测；
2. 普通、不可交换 torsion 继续使用单个 `(sin φ, cos φ)`；
3. 可交换 rotor 改为无序 circular set，使用 Hungarian circular matching 或等价 Fourier
   set representation；
4. decoder 前确定性求 atom-to-slot assignment，再广播到原 torsion orbit；canonical graph、
   元素、键和点群 action 均不修改；
5. 原 `2°` 数值阈值保留，同时报告 atom-labelled diagnostic 与 permutation-matched primary
   metric，禁止用标签重排掩盖真实几何错误；
6. 仍先跑同一 Tier-16 记忆面板。通过 prediction Gate 后才能运行 oracle-free 3D decoder；
   通过后才能讨论 unseen 或扩大数据规模。

此修复不是 hard projection。它处理的是局部原子编号 gauge；最终 Cartesian 对称性仍由
stabilizer-aware orbit parameterization 和 exact group lifting 保证。

### M2.2 表示 Gate A（已通过）

已实现第一项受控验证，但尚未训练：`local_rotor.py` 只用 canonical graph、元素、形式电荷、
自由基和 bond type，将共享三个 directed-torsion 原子、仅替换同元素 degree-one 末端的 orbit
组成无序 circular set；不读取 XYZ 或 torsion target。冻结 Tier-16 上识别出12个 disjoint
local-rotor sets，slot 数均为2或3，覆盖29/31个 M2.1 失败 orbit（93.55%）。

同时实现了枚举2/3-slot permutation 的可微 circular assignment/loss；交换 target atom order
不改变 loss，梯度有限。正式状态：

```text
PASS_M2P2_LOCAL_ROTOR_SET_REPRESENTATION_GATE_A
```

该 PASS 只准入 set-valued head 的实现，不是训练或3D质量结论。下一步必须让 head 从共享 rotor
embedding 输出2/3个无序 slots；训练用 permutation-invariant matching，推理用确定性 slot 顺序，
并在 Cartesian Gate 中允许相同元素、相同局部 automorphism class 内的 assignment。

## 6. 制品与代码定位

- protocol：`configs/m2p1_phase_tier16_v1.json`，SHA-256
  `3a88b0eef28e27e104db031e2ce8e7c98b32211186bb5335fbcc3d436f6390b8`；
- formal report：`runs/m2p1_phase_tier16_v1/report.json`，SHA-256
  `858b486c04f9ebb2018c50722330d96c32addb1f5ab87d788a7b2afe4240bcbd`；
- losses：SHA-256 `06970ff7be6aac2e1a0d930306dc23fa0fd5719ac9c1dcff08d112e3b62b436e`；
- predictions：SHA-256 `3d59217e18083403a8903bdd86941dfe9ace205cebb426887305dd56d82c924a`；
- final checkpoint：SHA-256 `983fdb288f383a6f8b52eb8ee3524b56ecc8e8ef543b53548b8e624be59ec8e4`；
- read-only diagnosis：`reports/m2p1_phase_tier16_diagnosis_v1.json`，SHA-256
  `402188f7142f6e1925dc1bae8ac45f451564bebb13b769edc42960900a05b404`。

相关代码：

- `orbit_ic_model.py`：group-phase features 和三类 IC heads；
- `build_m2p1_protocol.py`：Tier-16 训练、迁移、双 Gate 与隔离条件；
- `m2p1_phase_training.py`：PG-balanced 训练和 prediction-before-decoder 停止逻辑；
- `diagnose_m2p1_torsion.py`：本次只读局部 rotor/重复预测诊断；
- `local_rotor.py`：graph-only terminal rotor sets、circular assignment 与可微 set loss；
- `audit_m2p2_rotor_sets.py`：M2.2 表示覆盖 Gate A；
- `tests/test_m2p1_phase.py`：面板、调度、权重迁移、非平面/手性合同测试。

M2.2 Gate A report SHA-256：
`877f4fb0abf051e1a65b6d114ff64a8ef03e2fbcb5014d2c235bde03b5e222e0`。
