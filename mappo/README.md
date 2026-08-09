# MAPPO for COF Material Design

Multi-Agent PPO framework for designing Covalent Organic Frameworks (COFs).

## 目录结构

```
mappo/
├── README.md
├── __init__.py                    # 包初始化 + 目录说明
│
│  ── MAPPO 核心算法 ──
├── mappo_mpe.py                  # MAPPO 实现 (Actor_MLP/RNN, Critic, PPO训练)
├── mappo_mpe_v2.py               # V2: MultiDiscrete 动作空间
├── replay_buffer.py              # 经验回放缓冲区 (GAE, 批量采样)
├── normalization.py              # 奖励归一化
│
│  ── 环境 ──
├── env.py                        # 原版: 6-Agent BB构建 (词汇表+Transformer)
├── env_v2.py                     # V2: 3-Agent 扩散集成
├── env_symtopo.py                # ★ 当前: 3-Agent SymTopo选择
│
│  ── 训练入口 ──
├── run.py                        # 6-Agent BB构建训练
├── run_symtopo.py                # ★ 当前: SymTopo MAPPO训练
│
│  ── 工具模块 ──
├── transformer.py                # Transformer 分子状态编码器
├── reward.py                     # 奖励函数 + RND探索
├── rnd_module.py                 # 独立RND模块
├── ccjson.py                     # ChemJSON读取
├── cof_fromname.py               # COF组装（从Agent命名）
├── combine_substructures.py      # RDKit分子子结构组装
├── xyz.py                        # 2D坐标生成
│
│  ── 预训练 + 数据 ──
├── model/                        # 预训练权重
├── data_train/                   # 训练数据
├── runs/                         # TensorBoard日志
├── out/                          # 输出
│
│  ── 旧版/测试 ──
└── legacy/                       # 历史代码 (不再使用)
```

## 两种训练模式

### 1. SymTopo MAPPO (`env_symtopo.py` + `run_symtopo.py`) ★ 当前

- **3 Agent**: Ag1(sym+core) → Ag2(sym+core) → Ag3(topo)
- **状态**: 64维（Core特征+协调信息+拓扑mask+多样性统计）
- **动作**: Ag1/2共享Actor_12(26维), Ag3用Actor_3(9维)
- **奖励**: N2查表 + 多样性bonus + RND探索

```bash
python run_symtopo.py --max_train_steps 5000 --batch_size 16
```

### 2. COF构建 MAPPO (`env.py` + `run.py`)

- **6 Agent**: 每个Agent从53个词汇token中逐步构建BB
- **状态**: 128维 Transformer编码
- **奖励**: PMTransformer预测N2/O2

```bash
python run.py
```

## 实验记录

| 实验 | 环境 | Agent数 | 状态 | 成功率 |
|------|------|:---:|------|:---:|
| E001 | v1单Agent | 1 | 预验证 | 72% |
| E004-E008 | v2-v5 | 3 | 16维 | 8-31% |
| SymTopo | v6 | 3 | 64维 | 11% (WIP) |
