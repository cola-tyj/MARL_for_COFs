# 强化学习稀疏奖励问题：方法与调研

> 2026年6月24日 | 针对 COF 设计 MARL 任务中的稀疏奖励问题

---

## 一、问题定义

在 COF 设计的 RL 任务中，Agent 的有效动作空间由拓扑-对称性兼容性决定。原始设计中：
- 无效对称性对 → -1.0（~94% episode）
- 成功组装 → N₂ 吸附值（~6% episode）

**稀疏奖励导致**：（1）Agent 难以区分"接近成功"和"完全错误"；（2）探索效率极低；（3）信用分配困难——多层决策（拓扑→对称性A→对称性B）中哪一步出错无法判断。

---

## 二、改善方法分类

### 2.1 奖励塑形（Reward Shaping）

**核心思想**：为中间状态提供稠密的启发式奖励，引导 Agent 向目标前进。

**经典方法**：

| 方法 | 公式 | 适用场景 |
|------|------|------|
| 势函数塑形 | F(s,a,s') = γΦ(s') - Φ(s) | 已知启发式势函数 |
| 距离奖励 | r = -dist(s, goal) | 目标可达性已知 |
| 分层奖励 | r = tiered(success_level) | 有明显中间状态 |

**势函数塑形（Potential-Based）** (Ng et al., 1999)：
```
R_shaped = R_original + F(s, a, s')
F = γΦ(s') - Φ(s)  ← 保证最优策略不变
```

**在 COF 中的应用**（我们的分层奖励）：
```
Tier 0: 无效对称性对 → -1.0
Tier 1: 有效对但组装失败 → +1.0（部分信用）
Tier 2: 组装成功 → N₂×10 + 8（完整奖励）
```

**参考文献**：
- Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy invariance under reward transformations: Theory and application to reward shaping." *ICML*.
- Grzes, M., & Kudenko, D. (2008). "Plan-based reward shaping for reinforcement learning." *ICAIS*.

---

### 2.2 课程学习（Curriculum Learning）

**核心思想**：从简单任务开始，逐步增加难度。

**方法**：

| 方法 | 说明 |
|------|------|
| 手动课程 | 人工设计难度递增的任务序列 |
| 自动课程 | 根据 Agent 当前能力自动调整难度 |
| 反向课程 (Reverse Curriculum) | 从目标状态附近开始，逐步扩大起始范围 |

**在 COF 中的应用**：
```
阶段 1: 只训练 HCB_A（成功率最高，~100% 组装）
阶段 2: 加入 KGD, SQL_A
阶段 3: 全拓扑覆盖（9 种）
```

**参考文献**：
- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). "Curriculum learning." *ICML*.
- Florensa, C., Held, D., Wulfmeier, M., Zhang, M., & Abbeel, P. (2017). "Reverse curriculum generation for reinforcement learning." *CoRL*.
- Narvekar, S., Peng, B., Leonetti, M., Sinapov, J., Taylor, M. E., & Stone, P. (2020). "Curriculum learning for reinforcement learning domains: A framework and survey." *JMLR*.

---

### 2.3 事后经验回放（Hindsight Experience Replay, HER）

**核心思想**：失败的 episode 也有价值——把"实际达到的状态"作为虚拟目标重新标记。

**HER 算法** (Andrychowicz et al., 2017)：
```
对每个 episode:
  1. 原始目标 g → 经验 (s, a, r, s', g)
  2. 虚拟目标 g' = 实际达到的最终状态
     → 额外经验 (s, a, r', s', g')  ← r' 相对于 g' 重新计算
```

**在 COF 中的应用**：
- Agent 设计了 COF_A（失败），但该 COF 本身可能有价值
- 将该 COF 作为"新目标"加入设计空间
- 对应 Self-Play 闭环：好 COF → Core 加入训练集 → 微调扩散模型

**参考文献**：
- Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, P., & Zaremba, W. (2017). "Hindsight experience replay." *NeurIPS*.
- Fang, M., Zhou, C., Shi, B., Zhang, Y., Zhou, J., & Li, Z. (2019). "Curriculum-guided hindsight experience replay." *NeurIPS*.

---

### 2.4 内在动机 / 好奇心驱动（Intrinsic Motivation）

**核心思想**：除了外部奖励，Agent 还有内在的"好奇心"奖励——探索未访问过的状态。

**方法**：

| 方法 | 内在奖励 | 说明 |
|------|------|------|
| ICM | 状态预测误差 | 预测越不准 → 越好奇 |
| RND | 随机网络蒸馏 | 两个随机网络的预测差 |
| Novelty Search | 状态访问计数 | 访问越少 → 奖励越大 |

**RND (Random Network Distillation)** (Burda et al., 2019)：
```
r_intrinsic = ||f_target(s) - f_predictor(s)||²
f_target: 固定随机网络
f_predictor: 可训练网络 → 见过越多 s，预测越准
```

**在 COF 中的应用**：
- Agent 探索新的对称性-Core 组合时有内在奖励
- 防止过早收敛到单一设计模式

**参考文献**：
- Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). "Curiosity-driven exploration by self-supervised prediction." *ICML*.
- Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2019). "Exploration by random network distillation." *ICLR*.
- Bellemare, M. G., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D., & Munos, R. (2016). "Unifying count-based exploration and intrinsic motivation." *NeurIPS*.

---

### 2.5 分层强化学习（Hierarchical RL）

**核心思想**：高层 Agent 设定子目标 → 低层 Agent 执行子目标。

**方法**：

| 方法 | 高层 | 低层 |
|------|------|------|
| Feudal Networks | 设定方向 | 执行方向 |
| Option-Critic | 选择 option | 执行 primitive action |
| HAC | 设定子目标 | 达到子目标 |

**在 COF 中的应用**：
```
高层 Agent: 选择拓扑 (HCB_A)
低层 Agent 1: 选择 Core-A 对称性 (T3)
低层 Agent 2: 选择 Core-B 对称性 (L2)
```

这正是我们 3-Agent 设计的理论依据——分层决策自然适合 MARL 框架。

**参考文献**：
- Dayan, P., & Hinton, G. E. (1993). "Feudal reinforcement learning." *NeurIPS*.
- Bacon, P. L., Harb, J., & Precup, D. (2017). "The option-critic architecture." *AAAI*.
- Levy, A., Platt, R., & Saenko, K. (2019). "Hierarchical reinforcement learning with hindsight." *ICLR*.

---

### 2.6 离线 RL / 从演示中学习

**核心思想**：先用次优策略或人类演示收集数据，再从数据中学习。

| 方法 | 说明 |
|------|------|
| Behavior Cloning | 模仿演示中的动作 |
| DQfD | 演示数据 + 在线探索 |
| CQL | 保守 Q-learning，避免外推错误 |

**在 COF 中的应用**：
- 用已验证的 (拓扑, Core-A, Core-B) 组合作为"演示数据"
- Agent 从这些已知有效组合开始学习
- 然后再探索新颖组合

**参考文献**：
- Hester, T., Vecerik, M., Pietquin, O., Lanctot, M., Schaul, T., Piot, B., Horgan, D., Quan, J., Sendonaris, A., Dulac-Arnold, G., Osband, I., Agapiou, J., Leibo, J. Z., & Gruslys, A. (2018). "Deep Q-learning from demonstrations." *AAAI*.
- Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). "Conservative Q-learning for offline reinforcement learning." *NeurIPS*.

---

## 三、方法对比

| 方法 | 稀疏奖励改善 | 实现难度 | COF适用性 | 我们的状态 |
|------|:---:|:---:|:---:|:---:|
| **分层奖励 (Tiered)** | ★★★ | 低 | 高 | ✅ 已实现 |
| 课程学习 | ★★★ | 中 | 高 | 📋 计划中 |
| HER | ★★★★★ | 中 | 中 | 可结合 Self-Play |
| 内在动机 (RND) | ★★★★ | 中 | 高 | 可集成 |
| 分层 RL (HRL) | ★★★★ | 高 | 高 | 架构契合 |
| 离线 RL | ★★★ | 中 | 中 | 可做 warm-start |

---

## 四、COF 任务的推荐方案

**短期（已实现）**：
1. ✅ 分层奖励（Tiered Reward）— 成功从 6%→30%

**中期（计划）**：
2. 📋 课程学习 — 从 HCB_A 逐步扩展到全拓扑
3. 📋 预验证动作空间 — 用已知有效组合 warm-start

**长期（研究）**：
4. 📋 Self-Play + HER — 发现的 COF → 加入训练集
5. 📋 RND 探索奖励 — 鼓励新颖性

---

## 五、参考文献汇总

1. Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.
2. Andrychowicz, M. et al. (2017). Hindsight experience replay. *NeurIPS*.
3. Burda, Y. et al. (2019). Exploration by random network distillation. *ICLR*.
4. Pathak, D. et al. (2017). Curiosity-driven exploration by self-supervised prediction. *ICML*.
5. Bengio, Y. et al. (2009). Curriculum learning. *ICML*.
6. Florensa, C. et al. (2017). Reverse curriculum generation for reinforcement learning. *CoRL*.
7. Bacon, P. L. et al. (2017). The option-critic architecture. *AAAI*.
8. Hester, T. et al. (2018). Deep Q-learning from demonstrations. *AAAI*.
9. Kumar, A. et al. (2020). Conservative Q-learning for offline reinforcement learning. *NeurIPS*.
10. Bellemare, M. G. et al. (2016). Unifying count-based exploration and intrinsic motivation. *NeurIPS*.
