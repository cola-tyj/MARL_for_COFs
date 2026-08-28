# Coordinate Flow Matching F0 工程 Gate 结果

**执行日期**：2026-08-22  
**状态**：`PASS_COORDINATE_FLOW_F0`  
**结论边界**：只证明 flow 数学和工程接口正确；尚未训练，不能据此声称生成质量通过。

## 1. 本阶段实现

- 新增模型无关 coordinate-flow contract：centroid-zero Gaussian prior、
  `x_t=(1-t)x0+t x1`、endpoint-to-velocity、固定 Euler/Heun 积分；
- 新增 Semla `EquiInvDynamics` endpoint-flow adapter；旧 absolute-x0 adapter 未修改；
- 时间通过 `[t,t²,sin(πt),cos(πt)]` 显式嵌入，不再以 noise sigma 代替；
- 训练配对支持只做 SO(3) Kabsch 旋转，不做 reflection、节点 permutation 或 graph mutation；
- 13 元素词表完整保留，Sn index 12 正向通过，越界元素严格失败。

## 2. Gate 结果

两个测试模块共 13 项全部通过：

- prior 同 seed 字节一致、质心归零且 padding 为零；
- interpolation 的 `t=0/1/0.5` 和 endpoint velocity 公式正确；
- oracle endpoint 经 Euler/Heun 均恢复目标；
- Kabsch 只改变刚体朝向并保持节点顺序/成对距离；
- Semla adapter forward/backward 有限，显式 time embedding 获得非零梯度；
- E(3) 与节点排列等变；
- continuous two-step 与 checkpoint-resume two-step 参数逐 tensor 完全一致；
- Sn 可用且没有 C/Si fallback。

test 和 Core-OOD 均未读取，canonical graph 未修改，训练未启动。

## 3. 机器证据

- F0 报告：`generative_model/smoke/reports/graph_pg_3d_coordinate_flow_f0_v1.json`
- 报告 SHA-256：`c44a0adec079c98f8c175cea007e76ff3b0c74ca974705967ac3b2975675b95d`
- flow contract SHA-256：`7fd0782f5c3f7106654d3c8bd3bde5a01ed86cc8e043ca8b16506a004c9bfb6b`
- Semla flow adapter SHA-256：`60dda1aec4465c7efcb1f922c90ed140ae3d3670173827f3498878506d087266`
- contract tests SHA-256：`f53664a7fed75136dca4404ac2033baa6ec5a0fb4cc7d7a67255868e131e7810`
- adapter tests SHA-256：`5fc81595b49d94939bc8dcf4f1bce6a3484e6cebfdc8aeb0f78a7bbc88815faf`

## 4. 下一准入动作

下一步只冻结 Fast-32 coordinate-flow protocol，明确 prior scale、time distribution、训练预算、
多步积分、correct-PG/swapped-PG 对照和相对 Gate。协议冻结前不启动 GPU 训练；Fast-32 未过前
不提供 2,026 条正式长训命令。
