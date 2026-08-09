# 训练实验记录

> 每次训练的方法、参数、结果统一记录于此

---

## MARL 训练

### Exp-001: 单Agent + 预验证组合 (baseline)

| 项目 | 内容 |
|------|------|
| 日期 | 2026-06-18 |
| 脚本 | `train_mappo_cof.py` |
| 环境 | env_v2 (单Agent, 55 预验证设计) |
| 动作空间 | 55 (HCB_A/KGD, T3+L2) |
| 奖励 | Mock (拓扑基础吸附值 + size + symmetry bonus) |
| PPO | lr=1e-4, γ=0.99, ε=0.2, update_every=20 |
| Episodes | 2,000 |
| 结果 | **成功率 72%, Best=22.1, Mean=14.4** |
| 日志 | `/tmp/mappo_cof_final.log` |

### Exp-002: 单Agent + 真实 N₂ reward

| 项目 | 内容 |
|------|------|
| 日期 | 2026-06-22 |
| 脚本 | `train_mappo_cof.py --real_reward` |
| 环境 | 同上, 40 COF N₂ 查找表 |
| Episodes | 2,000 |
| 结果 | **成功率 72%, Best=22.2, Mean=14.5** |
| 日志 | `/tmp/mappo_real_n2.log` |

### Exp-003: 3-Agent + Action Masking (topo→sym)

| 项目 | 内容 |
|------|------|
| 日期 | 2026-06-22 |
| 脚本 | `train_mappo_3agent.py` |
| 环境 | env_v3 (Agent 0=拓扑, Agent 1/2=对称性, action masked) |
| 动作空间 | Agent 0: 9 拓扑, Agent 1/2: 6 对称性 |
| 奖励 | Tiered (Tier0=-1, Tier1=+1, Tier2=N₂+8) |
| Episodes | 2,000 |
| 结果 | **成功率 6-8%, Best=16.4, Mean=0.2** |
| 日志 | `/tmp/mappo_3agent.log`, `/tmp/mappo_3agent_v2.log` |
| 备注 | 成功率低因 Core 模板组装失败 |

### Exp-004: 3-Agent + Action Masking + 分层奖励 (topo→sym)

| 项目 | 内容 |
|------|------|
| 日期 | 2026-06-24 |
| 脚本 | `train_mappo_3agent.py --no_rnd` |
| 环境 | env_v3 (topo→sym) |
| 奖励 | Tiered (Tier0=-1, Tier1=+1, Tier2=N₂+8) |
| Episodes | 2,000 |
| 结果 | **成功率 28%, Best=18.4 (SQL_A R4+L2), Mean=5.9** |
| 日志 | `/tmp/mappo_3agent_noRND.log` |
| 备注 | 分层奖励显著改善（6%→28%） |

### Exp-005: 3-Agent + Action Masking + 分层奖励 + RND (topo→sym)

| 项目 | 内容 |
|------|------|
| 日期 | 2026-06-24 |
| 脚本 | `train_mappo_3agent.py` |
| 环境 | env_v3 + RND (β=0.3) |
| 奖励 | R_tiered + β × R_rnd |
| Episodes | 2,000 |
| 结果 | 🔄 训练中 |
| 日志 | `/tmp/mappo_3agent_RND.log` |

### Exp-006: 3-Agent v4 (sym→topo) — 计划中

| 项目 | 内容 |
|------|------|
| 环境 | env_v4 (Agent 1/2 选对称性 → Agent 3 选拓扑, masked) |
| 奖励 | Tiered (Tier0=-1, Tier1=+1, Tier2=N₂+8) |
| 备注 | 正确因果顺序：对称性决定拓扑 |

---

## 扩散模型训练

### D-001: Phase 1 QM9 预训练

| 项目 | 内容 |
|------|------|
| 日期 | 2026-06-11 ~ 06-14 |
| 脚本 | `train_qm9.py` |
| 配置 | hidden=256, layers=9, steps=1000, batch=64, lr=1e-4 |
| Epochs | 500 |
| 结果 | **Best epoch 444, val_loss=1.462, RDKit 97%, Uniqueness 100%** |
| 权重 | `qm9_best.pt` (60MB) |

### D-002: Phase 2 Core 微调 (100ep)

| 项目 | 内容 |
|------|------|
| 日期 | 2026-06-15 |
| 脚本 | `train_symmetry_conditioned.py` |
| Epochs | 100 |
| 结果 | **Symm loss 0.008, Atom acc 100%** |
| 权重 | `symmcd_denoiser_finetuned.pt` (27MB) |

### D-003: Phase 2 Core 微调 (300ep)

| 项目 | 内容 |
|------|------|
| 日期 | 2026-06-16 |
| 结果 | **Symm loss 0.003, Atom acc 100%** |
| 权重 | `symmcd_latest.pt` (56MB) |

### D-004: Phase 2 Core 微调 (500ep)

| 项目 | 内容 |
|------|------|
| 日期 | 2026-06-17 |
| 结果 | **Symm loss 0.004, Atom acc 100%** |
| 权重 | `symmcd_500ep.pt` (56MB, released) |

### D-005: Phase 2 + Uniform Prior (500ep)

| 项目 | 内容 |
|------|------|
| 日期 | 2026-06-24 |
| 结果 | Loss=1.76, de novo 不可行 |
| 备注 | uniform prior 无法解决 83 Core 数据不足问题 |

---

## 对比总结

| ID | 成功率 | Best | Mean | 关键设计 |
|:---:|:---:|:---:|:---:|------|
| E001 | 72% | 22.1 | 14.4 | 单Agent+预验证, Mock |
| E002 | 72% | 22.2 | 14.5 | 单Agent+预验证, Real N₂ |
| E003 | 6-8% | 16.4 | 0.2 | 3-Agent, 原始奖励 |
| E004 | 28% | 18.4 | 5.9 | 3-Agent, 分层奖励 |
| E005 | **31%** | **18.5** | **6.4** | 3-Agent, 分层+RND |
| E006 | 10% | 18.6 | 2.7 | 3-Agent v4 (sym→topo) |
| E007 | 8% | 21.5 | 2.5 | **3-Agent v4b + diversity bonus** |
| E008 | 13% | **21.5** | 1.7 | **3-Agent v5 (connector+FG)** |

### v4b 多样性奖励效果

| 拓扑 | 探索次数 | 占比 |
|------|:---:|:---:|
| HCB_A | 906 | 42% |
| BOR | 381 | 18% |
| DIA_A | 337 | 16% |
| SQL_A | 178 | 8% |
| HXL_A | 110 | 5% |
| 其他 4 种 | 250 | 11% |

全部 9 种拓扑均匀覆盖，Best=21.5 (SQL_A S4+L2)。

### v5 连接子配对测试

11×11=121 配对中 118 可组装 (97.5%)，仅 OH2-NHOH 等 3 对失败。Connector 独立选择不受化学约束限制。

### 关键发现

1. **成功率 ≠ 设计质量**：v4 成功率最低(10%)但探索了全部 9 拓扑
2. **多样性奖励有效**：v4b 全部 9 拓扑覆盖，Best 反而从 18.6→21.5
3. **丰富动作空间有价值**：v5 探索 15×31 化学组合，成功率 13% > v4 的 10%
4. **Predictor 多属性**：N₂吸附、O₂吸附、带隙 — 可用于多目标优化
