# 7. 开题 PPT 页纲与讲稿

建议 18 页，汇报时间约 15–20 分钟。每页只讲一个结论。

## 第 1 页：题目

**标题**：面向 COF 构筑单元的指定点群三维分子生成方法研究  
**副标题**：从对称数据构建、约束生成基线到轨道空间条件流模型

讲解重点：研究对象是“具有指定分子点群的 3D 构筑单元”，不是完整周期 COF 晶体生成。

## 第 2 页：研究背景

内容：

- COF 由具有特定连接数和空间构型的 building blocks 组装；
- 分子对称性影响连接位点等价性和网络拓扑；
- 随机生成后筛选高对称分子的效率较低。

一句话：希望把“先生成再筛选”转为“按目标点群条件生成”。

## 第 3 页：任务为什么困难

画三角形：化学合法性—三维几何—点群约束。

强调：

- E(3) equivariance 只保证坐标系变换一致；
- C3 symmetry 要求分子自身在 120° 旋转和原子置换后不变；
- 两者不能混为一谈。

## 第 4 页：研究问题分层

```text
known graph → conformer
known graph + PG → symmetric conformer
PG → novel graph + symmetric conformer
```

标出当前完成第二层 composite baseline，后续主攻 raw learned 第二层，再扩展第三层。

## 第 5 页：数据集构建总流程

```text
511 cores × 77 arms
→ 2D combination
→ ETKDGv3
→ MMFF/UFF
→ point-group screening
→ 2,532 molecules
```

建议图：`cof_symmetry_pipeline/visualization/08_summary_card.png`。

## 第 6 页：数据集构建细节

内容：

- `*` 位于图自同构等价位置；
- ReplaceSubstructs 同臂替换；
- 显式 H、多 seed 三维嵌入；
- 三维点群复验排除柔性破缺。

强调：2D 拓扑对称不等于 3D 点群对称。

## 第 7 页：canonical v2 数据资产

大数字：

- 2,532 molecules；
- 91,113 atoms；
- 96,555 bonds；
- 13 elements；
- 2,532/2,532 target-compatible。

旁边放 Target_PG 分布：2019/461/38/14，说明类别不平衡和 D6h low-support。

## 第 8 页：数据表示创新

图示：

```text
R_g: 3D operation
p_g: atom permutation
orbit_id: equivalent atoms
XR_g^T = X[p_g]
```

讲解：点群标签被拆成可用于模型训练的原子级群作用，而不是只保存字符串。

## 第 9 页：为什么测试多个生成模型

简表：SemlaFlow、MiDi、UAE-3D、ET-Flow。

只保留结论：

- 通用模型与本数据 vocab/规模/任务接口存在差异；
- reconstruction pass 不等于 generation pass；
- 最终选择 ET-Flow 作为 frozen conformer prior，而不是宣布其 PG adapter 成功。

不要在本页展示逐步 loss 日志。

## 第 10 页：our_ET_Flow 总架构

使用双输入流程图：

```text
known graph → ET-Flow raw ───────────────┐
Target_PG → graph action {R,p,orbit} ────┤
                                        ↓
                              orientation + projection
                                        ↓
                              orbit UFF + repulsion
                                        ↓
                                  independent audit
```

一句话：ET-Flow 给初态，对称模块给约束，F0.2 给几何修复。

## 第 11 页：hard projection 数学含义

展示：

```math
Y=\Pi_G(X),\qquad YR_g^T=Y[p_g].
```

解释：把坐标直接放入目标群不变子空间，所以对称误差接近机器精度；但它不考虑键长和碰撞。

## 第 12 页：F0.2 几何修复

展示：

```math
X=Aq,
\quad E=E_{UFF}(Aq)+E_{repulsion}(Aq).
```

讲解：只优化 orbit representative，既修复局部几何，又不破坏对称性。

建议图：hard projection 与 F0.2 前后 XYZ 对照。

## 第 13 页：关键消融结果

直接使用 `paired_ablation_overview.png`，旁边写：

- raw：6/32 compatible；
- projection：30/32 compatible，但 collision-free 27/32；
- full F0.2：32/32 compatible + 32/32 collision-free。

结论：projection 和 F0.2 缺一不可。

## 第 14 页：泛化与物理稳定性

三栏：

1. Core-OOD：32/32 compatible；
2. 外部 Cn：98/98 C3-compatible，94 PASS + 4 WARNING；
3. xTB：31/32 松弛后保持 compatible。

强调分母和负例：D6h→D3h 真实降对称。

## 第 15 页：当前瓶颈

列出四点：

- 不是 novel graph generation；
- raw ET-Flow 没有 PG 条件；
- C2/C3 same-graph response 只有 5/8 通过；
- 数据小、类别不平衡、单图单构象、ETKDG/MMFF 偏置。

这页是引出后续研究的关键，不应回避负结果。

## 第 16 页：后续模型 PG-OrbitFlow

```text
graph + structured PG/action
        ├─ group order / generators / relations
        ├─ conjugacy classes / character table / irreps
        └─ permutation cycles / orbit / stabilizer
        ↓
orbit harmonic prior q0
        ↓
orbit-space conditional flow
        ↓  X_t = A q_t at every step
raw symmetric 3D
```

本页建议增加一个“群特征四层表示”小图：

```text
abstract group → operations → irreps/characters → molecular action
```

核心创新：不把 C2/C3 当成普通类别标签，而是使用群乘法、共轭类、特征标和 orbit–stabilizer
信息；同时把 hard symmetry 从 endpoint 后处理变成生成过程的参数化。最终坐标属于 trivial
irrep，网络内部允许 non-trivial irrep channels 表达方向与形变。

## 第 17 页：扩展为 novel molecule

```text
Target_PG
  → quotient graph/orbit types
  → group expansion to full graph
  → PG-OrbitFlow coordinates
  → chemistry/PG/xTB audit
```

讲解：先生成不对称单元/轨道代表图，比任意生成完整 graph 后筛选 automorphism 更有效。

## 第 18 页：研究计划与预期成果

展示 12 个月甘特图和五项成果：

1. symmetry dataset v3；
2. PG-OrbitFlow C2/C3；
3. Core-OOD/controllability/xTB benchmark；
4. quotient graph pilot；
5. 学位论文和可复现代码。

结尾句：

> 本课题的目标不是让模型“尽量接近对称”，而是让目标点群成为图和三维坐标生成空间本身的一部分。

## 答辩常见问题准备

### Q1：ET-Flow 已经能生成，为什么还需要新模型？

ET-Flow raw 只做 graph-conditioned conformer generation，不读取 Target_PG；当前 100% compatible
依赖生成后的 graph action/projection/F0.2。后续模型要让 raw output 自身响应目标点群。

### Q2：hard projection 既然严格，为什么还要学习？

projection 不理解化学能量，消融中会使 collision-free 从 32/32 降到 27/32，bond MAE 从
0.02349 Å 恶化到 0.11777 Å。学习模型的任务是在对称子空间内直接生成合理几何。

### Q3：2,532 条数据够吗？

不够从头训练高容量 de novo 模型。方案是大规模通用构象预训练、COF 对称数据微调，并扩充
多构象和平衡点群数据。quotient graph 生成放在第二阶段。

### Q4：compatible 为什么不要求 exact？

实际结构可能具有目标群的超群，例如 requested C3、actual D3h。它仍严格满足所有 C3 操作。
但为防止超群掩盖条件坍缩，还要同时报告 exact、actual distribution 和 swapped-PG response。

### Q5：生成结构是否真实稳定？

UFF 只是快速筛选。当前 32 条 xTB pilot 中 31 条保持 compatible，1 条 D6h 降为 D3h。后续将
增加 xTB/DFT gold subset 和无约束松弛后的对称保持率。
