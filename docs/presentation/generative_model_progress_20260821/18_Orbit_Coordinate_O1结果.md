# Orbit-coordinate O1 最终结果

**日期**：2026-08-22  
**正式状态**：`FAIL_ORBIT_COORDINATE_O1_STOP_DENOISER`  
**实验类型**：固定 32/8 C2/C3 面板、512-step hard-symmetry denoising quality Gate

## 1. 实验设计

O1 沿用 O0 已通过的架构：G2.2 bonded/local/global 等变图编码器产生 atom proposals，随后
Reynolds-pool 为 orbit representatives 并硬展开。训练输入不是任意全原子 Gaussian prior，
而是对 representatives 加 `0.15/0.5/1.0 Å` 噪声后展开得到的严格对称坐标。

- train：32 条 IID-train，24 C2 + 8 C3；
- validation：8 条 IID-validation，6 C2 + 2 C3；
- validation cases：8 molecules × 3 sigmas × 2 seeds = 48；
- optimizer：AdamW，learning rate `2e-4`，batch size 4；
- budget：唯一一次 512 steps；
- output/target：同一 graph、atom order 和 group action 下的全原子严格对称坐标；
- test/Core-OOD：未使用。

目标为 molecule-balanced coordinate MSE + `0.25 ×` bond-length MSE + `0.1 ×` all-pair
distance MSE。没有 soft symmetry loss，因为输出层已经构造保证对称性。

`Target_PG` token swap 只报告诊断，不作为 Gate：目标 operation matrices 和 permutations 已经
显式编码群作用，要求冗余标签必须再产生额外收益并不构成正确的必要条件。

## 2. 冻结 Gate 与结果

| 检查 | 阈值 | 结果 | 判定 |
|---|---:|---:|---|
| finite loss/gradient | 全部有限 | 全部有限 | 通过 |
| loss tail/first ratio | `≤0.70` | `0.71174` | **失败** |
| coordinate improved fraction | `≥0.75` | `0.83333` | 通过 |
| mean coordinate RMSD ratio | `≤0.85` | `0.85997` | **失败** |
| each-sigma RMSD ratio | `≤0.95` | max `1.01871` | **失败** |
| mean bond MAE ratio | `≤0.85` | `0.67691` | 通过 |
| collision-free fraction | `≥0.75` | `0.64583` | **失败** |
| max operation error | `≤1e-5 Å` | `1.22891e-6 Å` | 通过 |

虽然总体 ratio 只比阈值高约 `0.01`，本次不能解释为“接近通过”：低噪声和碰撞揭示了两个
结构性问题，不能靠继续训练合理解决。

## 3. 分噪声结果

| sigma | coordinate ratio | improved | bond ratio | collision-free |
|---:|---:|---:|---:|---:|
| `0.15 Å` | `1.01871` | `8/16` | `0.87020` | `16/16` |
| `0.50 Å` | `0.77534` | `16/16` | `0.57095` | `8/16` |
| `1.00 Å` | `0.78587` | `16/16` | `0.58956` | `7/16` |

模型能有效处理 `0.5/1.0 Å` 扰动并改善键长，但在 `0.15 Å` 已接近正确构象时出现过度修正；
中高噪声输出虽然比输入更接近目标，仍有大量原子碰撞。因此它学到了粗粒度收缩/恢复，不是
稳定、化学可用的坐标更新场。

C2 的 coordinate ratio/collision-free 为 `0.85582/0.66667`，C3 为
`0.87243/0.58333`；问题并非只来自一个点群。PG token swap 诊断为 correct better `25/48`、
平均输出差异 `0.04050 Å`，说明标签通路有响应但不是主要瓶颈。

## 4. 工程与 artifact

训练前 O1 工程 Gate 已通过：3/3 protocol tests、两步 finite、连续/恢复 model 和 optimizer
bit-exact、Sn strict forward、无 test/Core-OOD。正式训练在物理 GPU 4 的 tmux `orbit_o1`
运行；命令正常结束后 session 自动关闭。

- O1 protocol SHA-256：`d7d94fb4be1a4fa8e3b91faa424cbde0c48245634ac086f0f585295263f497ea`
- O1 engineering report SHA-256：`bf23085bf70ef2b6cf043a38e074c32e595142a3a081b2fd9a0b92eac5e80498`
- O1 final report SHA-256：`747168fee4099b14aae922361379c9f7a6f601edfcd761e9153495a7dd5be114`
- losses SHA-256：`c8b4a65e3d7072436770cc3faefcd0e327cf27d6dd0546aef6c0713c273e4ca4`
- last checkpoint SHA-256：`17f58709dac300369954f81c6043df52f0cc9a0380e9110ace3e37337cfee69b`

## 5. 正式结论

根据训练前协议，O1 失败后：

- 不延长到 1,024 steps；
- 不降低 `0.85/0.75` Gate；
- 不用中间 checkpoint 选择结果；
- 不进入 orbit-space learned sampler；
- checkpoint 只保留为诊断证据。

hard-symmetry 表示层仍然有效，失败的是当前 learned denoiser 的几何质量。下一条更合理的低成本
路线是非学习式 orbit-constrained force-field baseline：从 graph 生成候选 conformer，以 orbit
representatives 为优化变量，在每一步严格展开全原子坐标，并用 UFF/MMFF 能量与碰撞约束进行
relaxation。这样把模型当前最弱的 collision/局部化学几何交给显式物理约束，同时保留严格点群。
