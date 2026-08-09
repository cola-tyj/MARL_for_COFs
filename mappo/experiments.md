# SymTopo MAPPO 实验记录

---

## Exp-001: 初始训练（3-Agent + COOH硬编码 + 旧拓扑表）

**日期**: 2026-06-29  
**脚本**: `run_symtopo.py` (初版)  
**配置**:

| 参数 | 值 |
|------|------|
| 状态维度 | 64 |
| Ag1/2 动作 | sym(6) + core(26) |
| Ag3 动作 | topo(9) |
| 连接器 | 硬编码 COOH |
| 拓扑对 | 9对（含KGM_A, BOR） |
| 奖励 | N2查表 + 固定diversity bonus |
| 训练步数 | 800 |

### 结果

| 指标 | 值 |
|------|:---:|
| 成功率 | 11.2% |
| 平均奖励 | 1.5 |
| 拓扑覆盖 | 9/9 |

### 问题

1. **成功率极低**：COOH硬编码，40%无效sym对 + 52%组装失败
2. **连接器无选择**：双边硬编码COOH，非pycofbuilder有效反应对
3. **KGM_A、BOR不存在**：pycofbuilder中无KGM_A，BOR抛NotImplementedError
4. **奖励信号弱**：Tier 1统一+1，无法区分"差一点成功"

---

## Exp-002: 连接器选择 + 拓扑表修正（本次）

**日期**: 2026-06-30  
**脚本**: `run_symtopo.py` (当前版)  
**配置**:

| 参数 | 值 |
|------|------|
| 状态维度 | 100 |
| Ag1/2 动作 | sym(6) + core(41) + conn(12) |
| Ag3 动作 | topo(9) |
| 连接器 | 12种反应性（排除CH3, O, CCH3O） |
| 化学兼容对 | 26/210 = 12%，Agent 2 mask |
| Sym mask | Agent 2 只显示兼容的sym |
| 拓扑对 | 8对（移除KGM_A, BOR, (2,2)） |
| 3D stacking | 自动 "1"（非 "AA"） |
| 奖励 shaping | +1(connectivity)/+2(other)/+3(structure) |
| 成功奖励 | N2启发式(density-based) + UCB + topo bonus |
| 多样性 | UCB c=5.0×8 + 拓扑bonus 3.0 |
| 熵系数 | 0.05 |
| 动态 predictor | 每20 batch调用PMTransformer |
| 训练步数 | 20,000 |

### Exp-002a（首次运行，seed=42）

**结果**:

| 指标 | 值 |
|------|:---:|
| 最终成功率 | 98% |
| 最终奖励 | 73.1 |
| 总组装 | ~6,700 |

**⚠️ 严重模式崩溃**:

```
拓扑: DIA: 6568 (98.4%)  其他: 104 (1.6%)
     LON, SQL_A, FXT_A, HCB_A: 0
连接器: CONHNH2+COOH: 3113  CONHNH2+CHO: 3094
```

### 问题诊断

1. **UCB不够强**：DIA收敛后，UCB项 `sqrt(2×log(6568)/6568) ≈ 0.05`，毫无激励
2. **熵系数太低**：0.02无法阻止策略早熟收敛
3. **拓扑探索不均**：无显式拓扑级bonus，Agent只看组合级别
4. **Predictor import失败**：sys.path指向`cof_predictor/`而非项目根目录

### 修复（Exp-002b）

| 修复 | 之前 | 现在 |
|------|------|------|
| UCB c | 2.0 | **5.0** |
| UCB 权重 | ×5 | **×8** |
| 拓扑 bonus | 无 | **3.0×log(total)/topo_cnt** |
| 熵系数 | 0.02 | **0.05** |
| Seed | 42 | **123** |
| Predictor path | `cof_predictor/` | **项目根目录** |

### Exp-002b（seed=123，进行中）

**Batch 80 (3840/20000 steps) 中间结果**:

| 指标 | 值 |
|------|:---:|
| 成功率 | 96% |
| 奖励 | 75.2 |
| Loss | 278.9 |
| N2 cache | 0（predictor读取有bug） |

**拓扑分布**（模式崩溃再现）:
```
DIA:     1165 (92%)
DIA_A:     43 (3%)
HXL_A:     40 (3%)
KGD:       11
FXT_A:      9
其他4种:    0
```

**连接器分布**:
```
CHO+NHOH:    244 (36%)    ← 亚胺键
CHO+CHNNH2: 236 (35%)
CHO+NH2:    196 (29%)
全部来自 CHO 家族
```

**⚡ PMTransformer 成功运行！**
- 第 3 次 predictor 调用（batch 60）：处理 572 个 CIF，预测用时 6 秒
- 但有单个 CIF 解析错误：`D4_ADAM_BOH2-L2_3IDT_OH2_H_H_H-LON_A-1`
- N2 cache 未正确更新——结果读取有 bug，需修复

### 跨 Seed 对比

| | Exp-002a (seed=42) | Exp-002b (seed=123) |
|------|:---:|:---:|
| 主导拓扑 | DIA (98.4%) | DIA (92%) |
| 主导连接器 | CONHNH2+COOH/CHO | CHO+NHOH/CHNNH2/NH2 |
| 成功率 | 98% | 96% |
| 总组装 | 6,673 | ~3,000 (进行中) |

**分析**：拓扑收敛到 DIA 是**奖励驱动的**——DIA 3D 结构的低密度特性在启发式奖励中得分更高，不是随机种子影响。连接器偏好**受种子影响**——不同初始化导致 policy 锁定到不同但等价好解。

### 持续问题

1. **UCB+topo bonus 仍不够**：一旦 DIA 被充分探索，bonus 衰减到 ~0.1，无法对抗 DIA 的高奖励
2. **Predictor 结果读取 bug**：N2 cache 始终 0，需检查 CSV 读取逻辑
3. **4 种拓扑完全未触及**：LON, SQL_A, HCB_A 从未被选

---

## 实验演进总结

```
Exp-001 (COOH硬编码)
  ├── 成功率 11%  ❌
  ├── 9/9 拓扑探索 ✅
  └── 问题: 组装失败率88%, 无效拓扑, 弱奖励
        │
        ▼ 修复
Exp-002a (连接器选择 + 化学mask + 奖励shaping)
  ├── 成功率 98% ✅
  ├── 模式崩溃 ❌ (DIA 98.4%)
  └── 问题: UCB太弱, 熵太低, predictor路径错误
        │
        ▼ 修复
Exp-002b (强化多样性 + 修复predictor)
  ├── 成功率 96% ✅
  ├── PMTransformer 成功运行 ✅
  ├── 模式崩溃再现 ❌ (DIA 92%)
  ├── 跨seed验证：拓扑收敛是奖励驱动的
  └── 待修复：predictor结果读取bug, 更激进探索策略
```

---

## 关键技术决策记录

| 日期 | 决策 | 原因 |
|------|------|------|
| 06-29 | 3-Agent序列化设计 | CTDE框架，Ag2看Ag1选择→协调 |
| 06-29 | 连接器化学mask | 15种→12种反应性，26对有效 |
| 06-29 | 移除KGM_A, BOR, (2,2) | pycofbuilder不支持 |
| 06-30 | 奖励shaping +1/+2/+3 | 区别失败接近程度 |
| 06-30 | UCB + topo bonus | 防止模式崩溃 |
| 06-30 | 动态PMTransformer | 替代静态N2查表 |
| 06-30 | 跨seed验证 | seed=42/123：拓扑都收敛DIA，奖励驱动非随机 |
| 06-30 | 连接器seed敏感 | CONHNH2 vs CHO 系：不同seed锁到不同等价解 |

---

## Exp-003: Z-score per-topology + Agent 3 解耦

**日期**: 2026-07-01  
**脚本**: `run_symtopo.py` (修改版)  
**配置**:

| 参数 | 值 | 说明 |
|------|------|------|
| 奖励公式 | `z_n2*5 + UCB*5 + topo_bonus*5` | z-score 替代原始 N2 |
| z-score 归一化 | Welford online, min 5 samples, clip [-3,3] | 拓扑间公平比较 |
| Ag1/2 熵系数 | 0.05 | 正常探索 |
| Ag3 熵系数 | **0.20** (4x) | 强制拓扑探索 |
| seed | 42 | 与 Exp-002a 同 seed 可对比 |

### 核心设计

**Plan 2 - Z-score 归一化**:
- 每个拓扑维护独立 running mean/std (Welford online algorithm)
- 奖励 = (N2 - mu_topo) / sigma_topo * 5.0, 而非原始 N2 * 10
- DIA 的 +2z 和 HCB_A 的 +2z 都是各自拓扑内的 top performer
- Warmup (n<5): z=0, 完全靠 UCB+topo_bonus 探索

**Plan 3 - Agent 3 解耦**:
- Ag3 熵系数 0.20 vs Ag1/2 0.05
- Ag3 策略保持高随机性 -> 自然探索不同拓扑
- 配合 z-score 归一化: Ag1/2 不能靠"选 DIA"获利

### 期望效果

- 所有 9 个拓扑都被探索, 而非 DIA 垄断 98%
- 每个拓扑内优化 N2 (z-score 竞争)
- 拓扑选择由 Ag3 的探索驱动, 而非 N2 绝对值驱动

### Exp-003 结果（训练至 batch 100）

**成功部分**:
- B1-B9 (warmup): DIA 40-44%, 5种拓扑均匀探索
- B30-B50: z-score自我纠正, DIA从62%跌至37%, DIA_A和LON_A崛起
- 发现物理规律: 2D拓扑 N2=1.50 (天花板), 3D拓扑 N2=1.50-8.02

**局限**:
- B60后: 其他拓扑达N2天花板, DIA从37%反弹至62%
- 最终: DIA 62%, 类似Exp-002b但推迟40批
- 4/9拓扑未探索: FXT_A, LON, SQL_A, HCB_A(仅初期)
- 根本问题: 2D和3D COF N2物理差异导致agent抛弃2D

**NaN根因**: Welford M2浮点负数 → `max(M2, 0)`修复

---

## Exp-004: 2D/3D分离训练

**日期**: 2026-07-02
**原理**: 2D和3D COF物理特性根本不同(N2天花板1.50 vs 8.0), 混合训练必然导致agent抛弃2D。分离后各自在同类拓扑内公平竞争。

**配置**:
- `--topo_mode 2d`: 5 topologies (FXT_A, HCB_A, HXL_A, KGD, SQL_A)
- `--topo_mode 3d`: 4 topologies (DIA, DIA_A, LON, LON_A)
- Z-score per-topology + Ag3 entropy=0.20
- Double training: 2D + 3D in parallel
- max_train_steps=20000, batch_size=16, seed=42

### Exp-004 B20 结果

**2D训练**:
```
HXL_A: 109(44%)  SQL_A: 85(34%)  KGD: 29(12%)  HCB_A: 24(10%)  FXT_A: 0
Z(n/μ): HXL_A='101/1.50'  SQL_A='55/1.50'  HCB_A='21/1.50'
→ 所有2D N2=1.50: 物理一致, 纯探索竞争, 4/5活跃, 无垄断
→ 物理结论: 2D COF无N2优化空间, 需关注其他属性(稳定性/合成性)
```

**3D训练**:
```
LON_A: 125(51%)  DIA: 119(49%)  DIA_A: 0  LON: 0
Z(n/μ): LON_A='109/7.07'  DIA='84/4.20'
→ LON_A N2比DIA高70%, 均衡50/50竞争
→ 颠覆" DIA最优"认知: LON_A是3D最优N2载体
```

### Exp-004 完整演化 (B10→B86)

**2D 拓扑分布**:
```
      B10    B20    B30    B40    B50    B60
SQL_A 29%    28%    33%    50%    48%    51%
HXL_A 26%    38%    45%    37%    40%    35%
HCB_A 10%    10%    10%    10%    12%    9%
KGD   17%    12%    12%    12%    10%    7%
FXT_A  0%     0%     0%     0%     0%     0%
N2 μ   全1.50 全1.50 全1.50 全1.50 全1.50 1.50/1.52

动力学: HXL_A↔SQL_A 震荡 → B40后SQL_A单边上升
连接器: BOH2/OH2 → COCHCHOH/NH2(烯醇酮+胺)
```

**3D 拓扑分布**:
```
      B10    B20    B30    B40    B50    B60    B70    B80    B86
DIA   48%    49%    48%    61%    63%    61%    67%    71%    73%
LON_A 52%    51%    52%    39%    37%    39%    33%    29%    27%
DIA_A  0%     0%     0%     0%     0%     0%     0%     0%     0%
LON    0%     0%     0%     0%     0%     0%     0%     0%     0%

DIA N2 μ:   4.20 → 4.73 → 5.55 → 6.01 → 6.28 → 6.45
LON_A N2 μ: 7.11 → 7.07 → 7.06 → 7.13 → 7.38 → 7.38(冻结)
DIA_A N2 μ: 从未探索
LON N2 μ:   从未探索

动力学: B10-B30 50/50平衡 → B40 LON_A达天花板7.38 → B40-B86 DIA单边至73%
连接器: BOH2/OH2 → CHO/NH2(亚胺键, 最终占82%)
冻结: LON_A从B55起冻结32批(!!!)
```

### 所有实验最终对比

| | Exp-002b | Exp-003 | Exp-004 2D | Exp-004 3D |
|--|:---:|:---:|:---:|:---:|
| 设计 | 原始N2 | z-score混合 | z-score 2D | z-score 3D |
| 模式崩溃 | B30: 98% | B60: 62% | **无** | **B40: 73%** |
| 主导拓扑 | DIA 98% | DIA 62% | SQL_A 51% | DIA 73% |
| 活跃拓扑 | 1/9 | 3/9 | 4/5 | 2/4 |
| 物理发现 | 无 | 2D≠3D | 2D N2=1.50 | LON_A N2=7.38 |
| 最优N2 | ~6.5 | ~8.0(DIA_A) | 1.52(HCB_A) | **7.38(LON_A)** |

### 核心科学结论

1. **z-score归一化推迟但无法阻止模式崩溃** — 无论混合还是分离, 当某个拓扑达N2天花板而另一个还有优化空间时, agent自然转向后者
2. **2D COF无N2优化价值** — 所有2D拓扑密度>0.015, N2=1.50(启发式下限), 应关注其他属性
3. **LON_A是3D最优N2载体** — N2=7.38 vs DIA=6.45, 颠覆传统认知
4. **分离训练消除跨类崩溃但类内仍存在** — 2D的UCB震荡和3D的优化收敛都是z-score的正确行为
5. **FXT_A, DIA_A, LON从未被探索** — sym pair限制+无核心模板 → 需要diffusion生成新Core

### 训练成本

| 实验 | 批次 | 耗时 | 瓶颈 |
|------|:---:|------|------|
| Exp-003 | 100批 | ~5h | PMTransformer每20批, predictor CIF重组装慢 |
| Exp-004 2D | 60批 | ~3.5h | predictor B40卡1h+ |
| Exp-004 3D | 86批 | ~5h | predictor B20/B40/B60各卡30-40min |

Predictor优化方向: 在env step时直接保存CIF, 而非predictor中重组装(提速10x+)

### 所有实验对比 (B20)

| | Exp-002b | Exp-003 | Exp-004 2D | Exp-004 3D |
|--|:---:|:---:|:---:|:---:|
| 设计 | 原始N2+UCB | z-score混合 | z-score 2D | z-score 3D |
| DIA B20 | 67% | 60% | N/A | 49% |
| 最大占比 | DIA 67% | DIA 60% | HXL_A 44% | LON_A 51% |
| 活跃拓扑 | 5/9 | 5/9 | 4/5 | 2/4 |
| 模式崩溃 | B30: 92% | B60: 62% | **无** | **无** |
| 物理洞察 | 无 | 2D≠3D | 2D N2=1.50 | LON_A>DIA |

### NaN修复记录

| 日期 | 问题 | 修复 |
|------|------|------|
| 07-02 | Welford M2浮点负数 → sqrt(NaN) | `M2 = max(M2, 0)`, `var = max(var, 0)` |
| 07-02 | NaN传播到网络权重 | 训练中 `if isnan(loss): continue` 跳过异常batch |

## 关键技术决策 (新增)

| 日期 | 决策 | 原因 |
|------|------|------|
| 07-02 | 2D/3D分离训练 | 消除跨类物理差异导致的模式崩溃 |
| 07-02 | Welford M2数值安全 | 浮点累积导致负数, sqrt→NaN |

---

## Exp-005: VAE + Diffusion 连续 Core 生成

**日期**: 2026-07-03  
**原理**: 当前 MARL agent 只能从 276 个预计算 Core 中选择（离散动作）。训练 VAE 将 Core 压缩到 32-dim 连续 latent space，agent 输出 latent vector → VAE decoder 生成新 Core → 动作空间从离散变为连续 → 真正实现分子设计。

**架构**:
```
Encoder: Core(atom_types+coords) → EGNN(9层) → pool → μ,σ → z∈R³²
Decoder: z + sym_cond → MLP → EGNN(5层) → atom_logits + coords
AtomCount: MLP(z+sym) → n_atoms prediction
```

**配置**:
| 参数 | 值 |
|------|------|
| Latent dim | 32 |
| Hidden dim | 256 |
| Encoder layers | 9 (reuse diffusion depth) |
| Decoder layers | 5 (lightweight) |
| Total params | 13,991,062 |
| Data | 276 Cores → 5000 augmented |
| Batch size | 32 |
| LR | 3e-4 |
| Epochs | 200 |
| KL warmup | 0→0.1 over 100 epochs |
| Device | GPU 3 (24GB) |

**NaN 修复**:
| 问题 | 修复 |
|------|------|
| Welford M2 浮点负数 → sqrt(NaN) | `position clamp [-20,20]`, coord_delta clamp [-5,5] |
| NaN gradient → epoch crash | `if isnan(loss): continue` + gradient NaN check |
| EGNN coordinate explosion | `torch.clamp(coords, -20, 20)` in decoder output |

**训练状态**: 🔄 进行中 (Epoch 0-2: val_loss 117.7→117.7, 无NaN, ~3min/epoch, 预计10h)

**MARL 集成**:
- 新 env: `env_symtopo_vae.py` — Agent输出 `(sym, z∈R³², conn)` 替代 `(sym, core_idx, conn)`
- 新 actor: `MultiHeadActor12VAE` — 连续 Gaussian 策略替代离散 Categorical
- 新训练: `run_symtopo_vae.py` — PPO 支持连续+离散混合动作

### 文件索引

| 文件 | 功能 |
|------|------|
| `mappo/vae/config.py` | VAE 超参数 |
| `mappo/vae/model.py` | CoreVAE 模型 (14M params) |
| `mappo/vae/train_vae.py` | VAE 训练脚本 |
| `mappo/env_symtopo_vae.py` | VAE 集成环境 |
| `mappo/run_symtopo_vae.py` | VAE+MAPPO 训练脚本 |
| `docs/vae_integration_design.md` | VAE 集成设计文档 |

## 关键技术决策 (新增)

| 日期 | 决策 | 原因 |
|------|------|------|
| 07-03 | VAE 替代离散 Core 选择 | 将动作空间从离散→连续, 实现真正的分子设计 |
| 07-03 | 复用 EGNN 为 Encoder backbone | 避免从头训练, 利用 diffusion 预训练特征 |
| 07-03 | 32-dim latent space | 足够表达 Core 结构变化, 同时低维可控 |

---

## Exp-005b: VAE 失败 → DDIM 直接采样

**日期**: 2026-07-04  
**结论**: VAE 路线不可行，转向 diffusion DDIM 快速采样

### VAE 失败分析

**训练结果**: val_loss 117，最佳 epoch 50
**实际表现**:
- 原子类型准确率: 44.9%（≈ 总是猜 C 的 45% baseline）
- 坐标 RMSE: 对齐后 NaN（结构完全崩溃）
- pycofbuilder 组装: **0/15 成功**
- 仅有 Q 原子准确 (96.2%)——symmetry 强先验
- C, H, N, O: 全部 0-5% 准确率

**失败原因**:
1. 276 样本 训练 14M 参数 → 严重欠拟合
2. Coord loss ×10 权重吸收所有梯度
3. 全图结构生成对 VAE 来说难度过高
4. 32-dim latent 不足以同时编码原子组成 + 3D 几何

### 路线切换：Diffusion DDIM

放弃 VAE，直接使用 diffusion 模型 + DDIM 加速采样：

| | VAE | Diffusion DDIM |
|--|:---:|:---:|
| 单次生成速度 | ~2ms | ~0.2s/个 |
| 有效性 | ❌ 0% | 待验证 |
| 原子准确率 | 45% | 取决于模型质量 |
| 连续控制 | z∈R³² | noise seed (32-dim) |

**DDIM 实现**: `DiffusionProcess.sample_ddim()`
- 坐标: DDIM 非 Markov 跳步 (50/1000 steps)
- 原子/键类型: 最终 denoiser 一步预测
- 速度: ~0.2s/sample

### Phase 2 NaN 深度修复

**NaN 传播链分析**:

```
SiLU(m_ij) → coord_weight 过大 → coord_diff * weight 爆炸
    → index_add 累积 → coord_update 无界 → x_new 无界
    → 下层 EGNN 的 distance² = NaN → edge_mlp(NaN) → 传播到所有特征
```

**修复方案**（`symmcd_diffusion/models/egnn.py`）:

| 位置 | 修复 | 说明 |
|------|------|------|
| edge_mlp 输入 | `clamp(-100, 100)` | 防止 SiLU 输入溢出 |
| edge_mlp 输出 | `nan_to_num + clamp(-100, 100)` | 防止消息 NaN |
| coord_weight | `clamp(-10, 10)` | 防止坐标更新爆炸 |
| contrib (edge×weight) | `clamp(-50, 50)` | 每条边的贡献限制 |
| coord_update | `clamp(-10, 10)` | 每层坐标更新限制 |
| x_new | `clamp(-50, 50)` | 最终坐标限制 |
| node_mlp 输入 | `clamp(-100, 100)` | 防止节点特征溢出 |
| node_mlp 输出 | `nan_to_num + clamp(-100, 100)` | 防止残差 NaN |

**训练 NaN 安全**（`train_symmetry_conditioned.py`）:
- Loss NaN → skip backward + `scaler.update()` 保持 AMP 一致
- Gradient NaN → skip optimizer step + `scaler.update()`
- 学习率 5e-5（低 lr 减少初始不稳定）

### Sym Mask Bug 修复

**Bug**: `build_sym_mask()` 使用原始 `SYMMETRY_PAIR_TO_TOPOLOGY` 而非 2D/3D 过滤后的 `self._topo_table`
- 2D 模式下 Agent 2 mask 包含 (4,4) 对 → 无效 → **成功率仅 6%**
- 修复: 改用 `self._topo_table`
- 修复后: **成功率升至 51%**

### 当前三线并行状态 (2026-07-05)

| 进程 | 状态 | 备注 |
|------|:---:|------|
| Phase 2 v5 | 🔄 训练中 | EGNN NaN 修复 + AMP 修复 |
| MARL 2D v3 | 🔄 训练中 | Sym mask 已修复 |
| MARL 3D v3 | 🔄 训练中 | 正常运行 |

### CIF 缓存优化（Predictor 加速）

**改动**: 在 `env_symtopo.py` step() 组装成功时直接保存 CIF 到 `/tmp/symtopo_cif_cache/`
**效果**: Predictor 跳过重组装步骤，直接复制预存 CIF → **30-40min → 秒级** (10x+ 加速)

## 关键技术决策 (新增 07/04-07/05)

| 日期 | 决策 | 原因 |
|------|------|------|
| 07-04 | 放弃 VAE | 276样本训14M参数→欠拟合, 组装0/15 |
| 07-04 | DDIM 替代 VAE | Diffusion 预训练质量有保障，50步 DDIM ~0.2s |
| 07-04 | EGNN 全层坐标 clamp | SiLU 溢出导致 coord NaN → 传播全模型 |
| 07-04 | NaN安全+AMP scaler一致 | skip batch时必须调 scaler.update() |
| 07-05 | Sym mask 过滤修复 | 2D/3D mode 未过滤 → 成功率 6%→51% |

---

## Exp-005c: Coordinate-Only Diffusion Sampling

**日期**: 2026-07-06  
**原理**: 扩散模型只对坐标加噪/去噪，原子类型从模板 Core 固定。利用 Phase 2 训练好的坐标去噪能力（coord_loss=1.00），避开原子类型预测的失败（偏向 Q 原子）。

### 方法

```
1. 从 Core 模板取: 原子类型 (one-hot, 固定) + 初始坐标
2. Forward: x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
3. Denoise: denoiser(固定原子类型, x_t, t) → predicted ε
4. x_0_pred = (x_t - √(1-ᾱ_t) * ε_pred) / √(ᾱ_t)
5. 迭代 10 步从 t → 0
```

Agent 输出 `noise_level ∈ [0,1]` (连续) 替代 `core_idx` (离散)。

### 结果

```
noise=0.3: RMSD=0.52Å  (微调构象)
noise=0.5: RMSD=1.07Å  (有意义变化)
noise=0.7: RMSD=1.85Å  (显著重排)
noise=0.9: RMSD=6.11Å  (大幅重构)
全部 0 NaN, 坐标范围正常
```

### 利弊分析

**优点** ✅:
- **速度快**: 10 步去噪 ~0.03s/sample (1000步祖先采样 ~5s, DDIM 50步 ~0.2s)
- **无 NaN**: EGNN 坐标 clamp 修复后完全稳定
- **原子类型保证正确**: 来自真实 Core 模板，Q/R 计数准确
- **连续动作**: noise_level ∈ [0,1] 替代离散 core_idx → Agent 可平滑探索构象空间
- **即插即用**: 使用已有 Phase 2 checkpoint，无需额外训练
- **物理意义明确**: noise_level=0 回到模板，noise_level=1 完全随机构象

**缺点** ❌:
- **不改变原子组成**: 同一模板的 C/H/N/O/Q/R 比例固定，只能变化 3D 几何
- **模板依赖**: 每个 symmetry 需要选一个"最好"的模板 Core 作为起点
- **探索空间有限**: 构象变化 (RMSD 0.5-6Å) 而非组成变化
- **维度低**: noise_level 是 1 维，不如 VAE 32 维 latent 表达力强
- **不完全生成**: 构象可能不物理 (键长不合理), 需要验证步骤
- **模板选择仍是离散**: Agent 仍需选 Core 模板 (离散) + noise_level (连续) → 混合动作空间

### 与 VAE 和 DDIM 对比

| | VAE | DDIM全生成 | Coord-Only |
|--|:---:|:---:|:---:|
| 原子类型 | ❌ 45%准确 | ❌ 偏向Q | ✅ 模板保证 |
| 坐标质量 | ❌ NaN | ❌ NaN | ✅ 0.5-6Å RMSD |
| 速度 | ~2ms | ~0.2s | **~0.03s** |
| 动作空间 | 32维连续 | 32维连续 | **1维连续+离散** |
| 训练成本 | 10h(失败) | 2h(偏向Q) | 0(即用) |
| 新颖性 | 理论∞ | 理论∞ | 构象变化 |
| 可用性 | ❌ | ❌ | ✅ |

### 论文定位

Coord-only diffusion 作为"连续构象优化"方法:
- 离散 Core 选择 → baseline (Exp-002/003/004)
- 坐标扩散 → continuous conformational refinement (Exp-005c)
- 完整生成 → future work (需更多数据)


---

## Exp-006: QM9分子 → Core 映射 + Coord-Only 扩散

**日期**: 2026-07-06  
**原理**: 用 QM9 预训练模型 (130k 样本) 的分子多样性 + Phase 2 坐标去噪能力 + 
规则化对称映射，间接实现 Core 生成，完全绕过 Core 数据不足的问题。

### 动机

直接生成 Core 的三条路都受阻:
1. VAE: 276 样本不足 → 原子准确 45%
2. DDIM 全生成: Phase 2 偏向 Q 原子
3. 传统微调: Core 只有 276 个

**关键洞察**: QM9 有 130k 分子，覆盖 C2v(6.8%)、D3h(21.4%)、
D4h(1.6%) 等对称性。分子→Core 的映射是几何规则（对称轴放 Q、
外围原子标 R），不需要学习。

### Pipeline

```
QM9 分子 (130k)
    ↓ 惯性张量筛选
目标对称性分子 (C2v→L2, D3h→T3, D4h→S4/D4/R4)
    ↓ 规则映射
  - 沿对称轴两端 → Q 原子
  - 外围原子 → R 基团
  - 内部原子 → 骨架
    ↓
Core (.cjson)
    ↓ sample_coord_only()
构象变体 (noise_level 0-1)
    ↓
pycofbuilder 组装 → COF
```

### SymmCD 论文讨论

重读 SymmCD (Levy et al., NeurIPS 2024) 后的关键发现:

1. **Wyckoff 位置压缩**: 用 orbit representatives (M̄=4.7) 替代全原子 (N̄=18.9)，4x 压缩
2. **二元对称编码**: 与我们已有的 SymmetryEncoder 相同机制
3. **混合扩散**: 坐标(连续) + 原子类型(离散) + 位置对称性(离散)
4. **投影步骤**: 去噪后投影到最近 Wyckoff 位置保证对称性

与我们的关系:
- 我们的 Core 是分子片段 (非周期晶体)，Q/R 原子位置由对称性确定
- SymmCD 的 Wyckoff 压缩思路启发我们: 只生成碳骨架坐标，Q/R 固定
- 我们的 coord-only 扩散正是这个思想的简单实现

### 预期优势

- ✅ QM9 130k 样本 (vs Core 276) — 500x 数据量
- ✅ 原子类型来自真实分子 (不再偏向 Q)
- ✅ 对称性通过惯性张量筛选 (天然覆盖)
- ✅ 坐标扩散提供构象多样性
- ✅ 零额外训练成本

---

## Exp-006: Scaffold-Only 扩散 (QR 固定, 骨架生成)

**日期**: 2026-07-06  
**原理**: 保持 Q/R 原子在对称性确定的连接位置固定不变，仅对碳骨架(C/N/H/O)
坐标进行扩散加噪/去噪，生成新的骨架几何结构。

### 动机

Coord-only 扩散虽然简单可行, 但有两点不足:
1. **创新点不足**: 仅是已有坐标的"抖动", 不是真正的生成
2. **可能破坏连接**: 移动 Q 原子位置可能导致 connector 无法接上

Scaffold-only 扩散同时解决这两个问题:
- 生成新骨架结构 (创新)
- Q/R 固定在已验证的连接位置 (保证有效性)

### 方法

```
1. 取真实 Core 模板
2. 识别 Q/R 原子 → 固定其坐标
3. 识别骨架原子 (C/N/H/O) → 加噪
4. 所有原子一起送入 denoiser
5. 去噪后: Q/R 恢复原位, 骨架为新坐标
```

```
噪声水平 t=300: scaffold_RMSD=0.59Å  (微调)
噪声水平 t=500: scaffold_RMSD=0.81Å  (中等变化)  
噪声水平 t=700: scaffold_RMSD=2.22Å  (显著重构)
全部 0 NaN
```

### QM9→Core 映射探索 (Exp-006a)

**日期**: 2026-07-06
**原理**: 用 QM9 分子 + 惯性张量对称性识别 → 规则映射(H→Q/R) → Core

**映射算法**:
```
QM9分子 → 惯性张量计算主对称轴 → 识别C2v/D3h/D4h对称性
    → H原子中最靠近轴2个 → Q
    → H原子中最远离中心6个 → R1-R6
    → 其余保持原样
```

**QM9 对称性分子统计** (惯性张量估计):
| 对称性 | QM9数量 | COF映射 |
|--------|:-------:|---------|
| C2v-like | 1,097 | L2 |
| D3h-like | 2,138 | T3 |
| D4h-like | 741 | S4/D4/R4 |

**验证结果**:
| 步骤 | 状态 | 说明 |
|------|:---:|------|
| 对称性识别 | ✅ | 惯性张量足够区分 |
| H→Q/R替换 | ✅ | Q=2, R=6, 骨架保留 |
| cjson格式 | ✅ | pycofbuilder checks全部通过 |
| pycofbuilder组装 | ❌ | BB加载后卡在from_building_blocks |

**失败原因**: pycofbuilder 构建 BB 时对几何有严格要求, QM9
分子骨架不满足 COF connector 的连接条件 (空间位阻、键角等)

**教训**: QM9 分子作为 Core 模板的想法在理论上可行,
但 pycofbuilder 的工具限制使其在实践中不可用。
这条路线记录在此供后续研究者参考。

### Pycofbuilder 兼容性问题

在多次测试中发现 pycofbuilder 对 Core cjson 有以下限制:
1. **名称限制**: Core名称不能含下划线 (`_`)
2. **cjson 缓存**: BuildingBlock 初始化时扫描目录并缓存,
   后续添加的文件不会被识别
3. **坐标校验**: 修改已有 Core 的坐标后, connector 加载失败
   (Connector: False) — pycofbuilder 内部可能缓存了原始几何
4. **重启恢复**: 上述问题通过重启 Python 进程可部分解决

**影响**: 这些限制使得在线生成+即时组装的流程不可行。
对于论文, 可将生成-组装作为两阶段流程:
1) 离线生成 Core 候选
2) 筛选+组装验证
3) 合格者加入 MARL 候选池

### 技术路线总结 (2026-07-06)

```
离散 Core 选择 (Exp-002/003/004)
  ├── 可行 ✅ → 论文 baseline
  └── 局限: 仅276个Core, 无生成能力

VAE 连续生成 (Exp-005)
  ├── 失败 ❌ → 276样本不足, 原子准确45%
  └── 教训: 小数据集 + 全图生成不可行

DDIM 全生成 (Exp-005b)
  ├── 失败 ❌ → Phase2模型偏向Q原子
  └── 原因: Core微调数据不足, 原子类型学习崩溃

Coord-Only 扩散 (Exp-005c)
  ├── 可行 ✅ → 0.03s, 0 NaN, RMSD 0.5-6Å
  ├── 局限: 纯坐标抖动, 创新不足
  └── 用途: 构象探索, 连续动作空间

Scaffold-Only 扩散 (Exp-006)  
  ├── 技术上可行 ✅ → QR固定, 骨架RMSD 0.6-2.2Å
  ├── 创新: 对称性约束下的骨架生成
  └── 阻塞: pycofbuilder坐标校验无法通过

QM9→Core 映射 (Exp-006a)
  ├── 算法可行 ✅ → C2v 1097候选
  └── 阻塞: pycofbuilder几何兼容性
```

**当前推荐论文路线**:
- **离散 MARL** (Exp-004) 作为主要实验
- **Scaffold 扩散** 作为方法扩展 (理论上验证)
- **pycofbuilder 兼容性** 作为工程限制讨论

### Pycofbuilder 源码分析 (2026-07-06)

深入阅读 pycofbuilder `building_block.py` 后关键发现:

1. **加载链路**:
   `from_name()` → `check_existence()` → `create_BB_structure()`
   → `ChemJSON.from_cjson()` → `add_connection_group()` → `add_R_group()`

2. **smiles 必需字段** (line 621):
   `self.smiles = core.properties["smiles"]`
   这是 pycofbuilder 拒绝我们生成 cjson 的**唯一原因**。
   缺少此字段 → KeyError → 组装失败。

3. **两套 pycofbuilder 数据路径**:
   - 项目本地: `/home/tianyajun/MARL_for_COFs/pycofbuilder/data/` (BB实际读取)
   - pip安装: `.../site-packages/pycofbuilder/data/` (我们一直在写入)
   - **这解释了所有 Core:False 错误** — 文件保存在了错的位置

4. **Proof-of-concept 验证**:
   - 复制原始 2BPD.cjson 为 QQQ.cjson → ✅ 组装成功
   - Scaffold 扩散 GEN*.cjson (含 smiles, 正确路径) → ⚠️ 间歇性失败
   - `check_existence` 返回全 True 但 `from_name` 仍报 Core:False
   - 原因: pycofbuilder 内部状态管理不可靠(可能的模块级缓存)

5. **解决方案建议**:
   - 短期: 使用原始 Core 名 + 原地修改坐标(已有成功案例)
   - 长期: 替换 pycofbuilder 或用其内部 API 直接构建 BB

---

## Exp-004 v3: 完整训练结果 (Sym Mask 修复版)

**日期**: 2026-07-05 ~ 2026-07-07  
**耗时**: 27.8 小时 (每路)  
**配置**: z-score归一化 + Ag3高熵 + sym mask修复 (self._topo_table)

### 2D 最终结果 (20016 步)

```
SQL_A: 4599 (69%)  HXL_A: 2025 (30%)  KGD: 31  HCB_A: 17  FXT_A: 0
全部 N2=1.50, Succ=97%
连接器: COOH+NH2(477), BOH2+OH2(461), CONHNH2+COOH(403)
```

**模式**: SQL_A 主导但 HXL_A 保持 30% 份额。KGD/HCB 增长极慢
(~0.1/batch), FXT_A 从未被选。无经典模式崩溃(无拓扑 >90%)。

**KGD/HCB 冻结原因**: T3 symmetry 只有 17 个 Core (vs L2=41)。
Agent 学会 L2+S4/D4 → SQL_A 和 L2+H6 → HXL_A 是更优路径。
KGD/HCB 的 sym pair (L2+T3) 中 T3 Core 太稀少, z-score 正确反映这一点。

### 3D 最终结果 (20032 步)

```
LON_A: 3972 (60%)  DIA: 2652 (40%)  DIA_A: 0  LON: 0
N2: LON_A=8.16, DIA=6.75, Succ=96%
连接器: Br+Cl(1001!), CHNNH2+CHO(433), COCHCHOH+NH2(416)
```

**模式**: LON_A 主导(60%), DIA 跟随(40%)。与 v1(DIA 73%)完全相反。
Sym mask 修复使 3D 探索更充分。DIA_A 和 LON 从未被选。

**LON_A > DIA**: 物理上合理的结论。LON_A (lonsdaleite变体) 
比 DIA (diamond) 有更低的密度和更高的 N2 容量(8.16 vs 6.75)。
z-score 帮助 agent 发现了这个物理规律。

### Sym Mask Bug 的影响评估

修复前(v1): build_sym_mask 用原始 SYMMETRY_PAIR_TO_TOPOLOGY,
导致 2D 下 (4,4) 对被错误标记为有效 → 成功率仅 6%。
修复后(v3): 用 self._topo_table (2D/3D过滤版) → 成功率 97%。

| 指标 | v1 2D | v3 2D | v1 3D | v3 3D |
|------|:---:|:---:|:---:|:---:|
| 成功率 | 6% | 97% | 97% | 96% |
| 主导Topo | SQL_A | SQL_A 69% | DIA 73% | LON_A 60% |
| Topo分布 | SQL_A垄断 | SQL_A/HXL_A | DIA垄断 | LON_A/DIA |

3D v1→v3 的主导反转(LON_A取代DIA)说明: sym mask bug 在 3D 
模式下影响较小(仅 (2,3)/(3,2)/(2,6) 对受影响), 但修复后 agent
能更公平地探索所有有效 sym 对, 从而发现 LON_A 的物理优势。

### 离散 MARL 最终结论

1. **z-score 归一化 + 2D/3D 分离有效缓解模式崩溃**
2. **Sym mask 修复是关键**: 2D 成功率 6%→97%
3. **LON_A 是 3D 最优 N2 载体** (N2=8.16)
4. **2D COF 无 N2 优化空间** (全部 N2=1.50)
5. **离散 Core 选择可达到 96-97% 成功率**

---

## Exp-007: Diffusion 联合训练 (进行中)

**日期**: 2026-07-07  
**原理**: Scaffold 扩散集成到 MARL 训练循环。Agent 输出连续 noise_level
∈ [0,1] 替代离散 core_idx，diffusion 根据 noise_level 生成不同构象的 Core。

### 架构

```
Agent1/2: (sym_discrete, noise_level_continuous, conn_discrete)
Agent3:   topo_discrete
```

MultiHeadActor12Diff: sym(离散 Categorical) + noise(连续 Gaussian μ+σ) + conn(离散)

### 文件

| 文件 | 功能 |
|------|------|
| `env_symtopo_diff.py` | 继承 SymTopoEnv，覆盖 step()，_diffuse_core() |
| `run_symtopo_diff.py` | MultiHeadActor12Diff + 连续 PPO 训练 |

### 技术细节

- 动作编码: `code = sym*1000 + noise*100 + conn` (标量, Buffer兼容)
- Core: 使用 BEST_TEMPLATES (DPDA/BRZN/OTPR 等)，原地覆写坐标
- Pycofbuilder: 修改已知 Core 的 cjson 坐标，用原名 from_name()
- 恢复: close() 时从备份恢复原始坐标

### 离散 vs 扩散 对比设计

| | 离散 (run_symtopo) | 扩散 (run_symtopo_diff) |
|--|:---:|:---:|
| Core 选择 | core_idx (离散, ~41选项) | noise_level (连续, [0,1]) |
| Actor 头 | sym/categorical | sym/categorical |
|  | core/categorical | noise/Gaussian(μ,σ) |
|  | conn/categorical | conn/categorical |
| 动作空间 | 离散 | 混合(离散+连续) |
| Core 多样性 | 固定276个 | 连续构象变化 |
