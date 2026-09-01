# M2：全学习 bond/angle/torsion orbit 的 Tier-4 结果

## 1. 结论

M2 通过冻结的 Tier-4 双 Gate，状态为：

```text
PASS_M2_ALL_LEARNED_IC_ADVANCE_TO_TIER16_PANEL_AUDIT
```

这说明在固定的 4 分子记忆面板上，模型能够仅从 canonical graph、Target_PG 和 orbit/action
特征预测 bond、angle、torsion 三类 orbit-level 内坐标，再由对称子空间优化器重建 3D；decoder
不再读取真实 bond、angle、torsion、local-pair 或 chirality 数值，也没有 ETKDG、hard projection
或 F0.2 后处理。

结论边界必须同时保留：Tier-4 的 33 个 torsion orbit 全部接近 0° 或 180°。因此本结果验证的是
全学习接口和平面扭转记忆能力，不是非平面扭转、手性、unseen 分子或数据集尺度泛化。

## 2. 训练数据与批次

| 项目 | 固定值 |
|---|---:|
| 分子数 | 4 |
| 点群组成 | C2 × 2，C3 × 2 |
| package index | 864、1045、1135、2391 |
| 原子数 | 10、12、9、12 |
| torsion orbit 数 | 7、14、4、8 |
| optimizer steps | 1,024 |
| batch size | 4（每 step 看完整面板） |
| molecule exposures | 4,096 |
| optimizer | AdamW，lr `1e-3`，weight decay 0 |
| gradient clip | 5.0 |
| 参数量 | 313,540 |

M2 从已通过的 M1 step-1024 checkpoint 继承 quotient encoder、bond head 和 angle head；新增
torsion head 按 M2 固定 seed 初始化，optimizer 不复用。训练目标为：

```math
L=10L_{bond}+2L_{angle}+2L_{torsion},
```

其中 torsion 输出为单位圆上的 `(sin φ, cos φ)`，`L_torsion=1-dot(pred,target)`，避免
`-π/π` 边界不连续。

## 3. 输入、输出与模型改动

输入不含 Cartesian XYZ。quotient multigraph 的一个节点对应一个 atom orbit，节点特征包含
元素、形式电荷、自由基、群/稳定子特征和 C2/C3 条件；边特征包含 edge-orbit multiplicity 与
bond-type histogram。

M2 在 M1 基础上增加 reversal-invariant torsion-orbit 表示：四原子路径的两端与中心分别使用
embedding 的 sum/absolute-difference，再拼接 graph embedding、terminal bond histogram 和
central bond type。head 输出二维向量并归一化为 `(sin φ, cos φ)`。

checkpoint 的输入是 graph/group/orbit contract；输出是每个 bond orbit 的长度、每个 angle
orbit 的 cosine、每个 torsion orbit 的 `(sin,cos)`。这些预测按 orbit ID 广播到全原子图，再由
stabilizer-aware orbit parameterization 和 exact group lifting 在对称子空间内优化 Cartesian 3D。

## 4. Gate 与结果

### 4.1 轨道预测 Gate

| 指标 | 阈值 | 最差结果 | 判定 |
|---|---:|---:|---|
| bond-orbit MAE | ≤ 0.03 Å | `6.76e-7 Å` | PASS |
| angle-orbit MAE | ≤ 5° | `8.04e-5°` | PASS |
| torsion-orbit circular MAE | ≤ 2° | `0.001952°` | PASS |

### 4.2 无 oracle 3D 重建 Gate

每个分子使用 16 个与 target Cartesian 无关的固定随机 orbit-space 起点，选择 learned-energy
最低解。真实 XYZ 只在候选选定后用于评测。

| 指标 | 阈值 | 最差结果 | 判定 |
|---|---:|---:|---|
| bond MAE | ≤ 0.03 Å | `7.06e-7 Å` | PASS |
| angle MAE | ≤ 5° | `1.16e-4°` | PASS |
| torsion circular MAE | ≤ 2° | `5.50e-4°` | PASS |
| ring closure MAE | ≤ 0.03 Å | `8.91e-7 Å` | PASS |
| Kabsch RMSD | ≤ 0.20 Å | `3.74e-6 Å` | PASS |
| collision-free | = 1.0 | `4/4` | PASS |
| max group-operation atom error | ≤ `1e-6 Å` | `1.94e-7 Å` | PASS |
| all starts finite | true | true | PASS |

## 5. 复现与制品

- protocol：`configs/m2_torsion_tier4_v1.json`，SHA-256 `014b436c7618a570e1ef2a5aa4c99ddcd60686213bb87615fd40e546a816a3fa`；
- final checkpoint：SHA-256 `aac3fe61f2b5bfdea099cd1d4b69308b33ac8f8ddb6e7f80e5cb390d355eb995`；
- formal report：SHA-256 `f57eb55f5859ef5735f4fa15dd38ac8dda859565c72929dac6a913bc6c177677`；
- tracked summary：`reports/m2_torsion_tier4_v1_summary.json`；
- reproducibility：`PASS_M2_REPRODUCIBILITY`，同一 CUDA prediction device 下四条 prediction
  metrics 完全一致，selected-start Kabsch 差异均小于 `5.72e-8 Å`。

首次用 CPU 重算 prediction 时，约 `1e-6` 的跨设备浮点差使极低能量 optimizer 终点末位坐标
不同；按原运行设备 CUDA 重算后通过。该执行诊断没有改变 checkpoint、数据、seed、Gate 或科学
协议。

## 6. 代码位置与下一步

- `orbit_ic_model.py`：torsion orbit graph feature 与 circular head；
- `build_m2_protocol.py`：冻结训练、decoder、Gate 和隔离条件；
- `m2_torsion_training.py`：M1→M2 权重迁移、训练、无 oracle 3D 重建；
- `audit_m2_reproducibility.py`：prediction 与 selected-start 独立复算；
- `tests/test_m2_torsion.py`：target/shape/finite backward/oracle-free 测试。

下一步只进行 Tier-16 面板审计与冻结：优先纳入真正非平面 torsion 和有效 chirality 的分子，先
统计支持度与特征冲突，再决定是否训练。不得把当前四分子 PASS 直接写成生成模型泛化成功。
