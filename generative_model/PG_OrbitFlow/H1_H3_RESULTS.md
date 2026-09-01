# PG-OrbitFlow H1–H3 Tier-4 单变量消融结果

## 1. 冻结问题与实验边界

本轮只回答一个问题：能否在不改变 Cartesian E(3) backbone、训练面板和 Gate 的前提下，
通过修复早期 transport prior/target/loss，使 4 个已知分子的 raw 3D generation 通过准入。

三组实验共同使用：

- canonical v2 IID-train 的 package index `864, 1045, 1135, 2391`；
- `2 C2 + 2 C3`，不使用 validation、test 或 Core-OOD；
- 740,424 参数的同一 PG-OrbitFlow 模型；
- 每组独立从相同随机初始化开始；
- 1,024 optimizer steps、batch size 4，共 4,096 molecule exposures；
- C2/C3 各暴露 2,048 次；
- `t ~ Uniform(0.02,0.98)`、coordinate scale `3.0 Å`；
- raw inference 为 50-step Heun；
- 禁止 posthoc hard projection、F0.2、外部预训练和历史 checkpoint。

模型输入为 frozen atom/bond graph、原子属性、`Target_PG`、完整 `R_g/p_g` action、atomic
orbits 和当前时间坐标；输出是全原子 Cartesian velocity
`v_theta(X_t,t,graph,PG)`。checkpoint 因此保存的是速度场模型和 optimizer state，不是某个
分子的最终 XYZ。`raw_coordinates.npz` 才保存模型从 prior 独立积分得到的 raw 坐标。

## 2. 三个累计变量

### H1：graph-harmonic prior

由 canonical 无向 bond topology 构造 `L=D-A`，在 float64 下从非零模采样
`Z=U Lambda^(-1/2) epsilon`，再做 Reynolds prior projection 和质心归零。严格要求：

- graph 连通且只有一个平移零模；
- `P_g L P_g^T = L`；
- 特征值、坐标和群作用误差有限；
- 不做逐样本 RMS 归一化；
- Sn 保持 frozen vocabulary index 12，不允许 fallback。

H1 仍使用旧的 single best Cn phase alignment。

### H2：centralizer-averaged transport

在 H1 上删除 single best-phase label。C3 使用绕主轴 24 相位，C2 使用 centralizer 的两个
连通分量各 24 相位，共 48 candidates。确定性选择一个 candidate 构造 `X_t`，再计算：

```math
log w_k=-E_L(X_t-tX_{1,k})/[2(1-t)^2],
```

并用 posterior mean `bar X_1=sum_k w_k X_{1,k}` 定义 velocity target。真实旋转 candidate
仍单独提供 bond-length ground truth；posterior mean 不替代 canonical chemistry target。
raw inference 不读取 target，也不做 projection。

### H3：endpoint auxiliary

在 H2 上只增加权重为 1.0 的：

```math
L_endpoint=||X_t+(1-t)v_theta-bar X_1||^2.
```

它与 `(1-t)^2` 倍的 velocity error 数值等价；没有同时改变时间采样。

## 3. 冻结 Gate 与结果

| 指标 | 阈值 | H1 | H2 | H3 |
|---|---:|---:|---:|---:|
| training loss window ratio | <= 0.50 | 0.5583 | 0.6266 | 0.6006 |
| fixed-time endpoint RMSD | <= 0.75 Å | 0.7804 | 1.4280 | 1.4114 |
| raw Kabsch RMSD | <= 1.00 Å | 1.4537 | 1.6555 | 1.5527 |
| raw pair-distance MAE | <= 0.60 Å | 0.8538 | 0.9366 | 0.8557 |
| raw bond-length MAE | <= 0.20 Å | 0.5432 | 0.5957 | 0.5742 |
| collision-free fraction | >= 0.75 | 1.000 | 0.875 | 0.875 |
| max operation atom error | <= 1e-4 Å | 1.52e-6 | 1.64e-6 | 1.32e-6 |
| 最终判定 | 全部通过 | **FAIL** | **FAIL** | **FAIL** |

三组的有限 loss/gradient、collision 和 raw group-action Gate 均通过；失败集中于 Cartesian
几何恢复、化学键几何和收敛窗口。H1 是三组中最简单且总体最好的候选，fixed-time 指标接近
阈值，但它仍同时失败 5 个冻结检查，不能作为 Tier-16 parent。H2 引入的 phase mixture 在当前
容量/预算下明显增加优化难度；H3 有小幅恢复，但不足以抵消 H2 的退化。

## 4. 排除项与停止决定

- 每组 step 1,024 都是其 fixed-time 表现最好的保存 checkpoint，不是 checkpoint 选择错误；
- 25/50/100-step Heun 指标近似不变，不是积分步数不足；
- 实际 final checkpoint 的同进程双重复 raw sampling 坐标逐数组完全一致，指标完全一致；
- protocol structured diff 已证明 H0→H1、H1→H2、H2→H3 只改变预先声明的变量族；
- 没有放宽任何 Gate，也没有执行 Tier-16/Tier-32。

按预注册停止规则，当前 Cartesian backbone 到此停止；下一路线为 fixed-graph 的
bond/angle/torsion internal-coordinate manifold decomposition。该决定不是“对称性失败”：raw
operation error 已稳定在 `1e-6 Å` 量级；失败的是从 prior 恢复合法化学几何的能力。

## 5. 代码与证据索引

| 文件 | 作用 |
|---|---|
| `data.py` | strict graph Laplacian、unnormalized harmonic sampling、Sn strict |
| `flow.py` | C2/C3 centralizer quadrature、posterior target、raw Heun |
| `losses.py` | canonical bond loss、symmetry/overlap、H3 endpoint auxiliary |
| `overfit.py` | 固定 Tier、训练、fixed-time/raw Gate、checkpoint |
| `build_overfit_protocol.py` | 冻结 H0–H3 protocol |
| `audit_h_protocols.py` | 相邻实验 structured diff 和 protocol SHA-256 |
| `audit_h_reproducibility.py` | actual checkpoint 双重复 raw 字节审计 |
| `diagnose_overfit.py` | checkpoint/time/ODE-step 只读诊断 |
| `summarize_h_experiments.py` | 结果、artifact hash 和停止决定汇总 |
| `tests/test_harmonic_transport.py` | harmonic、centralizer、endpoint、Sn、repro tests |

机器可读总表为
[`reports/h1_h3_tier4_summary.json`](reports/h1_h3_tier4_summary.json)，protocol diff 为
[`reports/h1_h3_protocol_audit.json`](reports/h1_h3_protocol_audit.json)，重复性证据为
[`reports/h1_h3_raw_reproducibility.json`](reports/h1_h3_raw_reproducibility.json)。每个 run
目录保留 `report.json`、`losses.npz`、`raw_coordinates.npz`、step checkpoints、`last.pt`、
`console.log` 和 artifact manifest。
