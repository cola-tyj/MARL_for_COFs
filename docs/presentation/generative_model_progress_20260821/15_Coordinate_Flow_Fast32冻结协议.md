# Coordinate Flow Fast-32 冻结协议

**冻结日期**：2026-08-22  
**工程状态**：`PASS_COORDINATE_FLOW_FAST32_ENGINEERING_PROTOCOL_FROZEN`  
**质量状态**：尚未运行，不能声称 flow 模型通过。

## 1. 固定面板与数据边界

- train：原 Fast-32 的 32 条 IID-train，24 C2 + 8 C3，12–63 原子；
- validation：原 Fast-32 的 8 条 IID-validation，6 C2 + 2 C3，15–60 原子；
- 两个面板无交集，test/Core-OOD 不读取；
- 面板本身没有 Sn，因此另取 IID-train package index `1684` 做 strict forward，不能计入
  模型质量统计；
- chemical graph、节点顺序、显式 H 和 13 元素词表保持不变。

## 2. Flow 与训练配置

- 路径：`x_t=(1-t)x0+t*x1`；
- 目标：centroid-zero endpoint `x1`；
- 训练 prior：Gaussian，使用 fixed-node SO(3) Kabsch 朝向配对；
- 采样 prior：独立、未对齐、与 target 无关的 Gaussian；
- prior component std：`3.3742073179010283 Å`，只由 2,026 条 IID-train 坐标计算；
- time：`Uniform[0,0.99]`；
- 模型：Semla `EquiInvDynamics`，约 77 万参数，显式 time embedding；
- optimizer：AdamW，LR `2e-4`，weight decay `1e-6`，gradient clip `1.0`；
- batch size 4，训练 1,024 steps，每 256 steps 保存 checkpoint；
- objective：coordinate `1.0`、bond length `0.25`、all-pair distance `0.1`、soft symmetry `0.05`。

## 3. 两层评估

第一层是 interpolation recovery：在 `t=0.25/0.5/0.75`、2 seeds、8 validation 分子上，
判断 endpoint 相对 `x_t` 是否稳定改善。

第二层是真正 prior sampling：从未对齐 Gaussian prior 做 16-step Euler，终点 `t=0.99`；
评估 pair-distance、bond-length、碰撞、raw point group，以及同一 prior 下 correct-PG 与
swapped-PG 的差异。

reference Kabsch alignment 只用于和单一参考构象比较，以及单独展示 projection；raw analyzer
和 collision 使用未对齐模型输出。reference-aligned projection 不计作 raw 模型通过。

v1 只比较 C2↔C3 swapped condition，不使用 no-PG，因为冻结的四类 vocabulary 没有经过训练的
null-PG token；是否增加 condition dropout 属于后续唯一一次 PG 增强，不能在本轮偷加。

## 4. 冻结 Gate

- training loss 尾/首窗口比例 `≤0.70`；
- recovery：改善比例 `≥0.75`，平均 RMSD ratio `≤0.90`，每个 time ratio `≤0.95`；
- sampling pair-distance MAE / prior baseline `≤0.90`；
- collision-free `≥0.75`；
- symmetry 相对 prior 改善比例 `≥0.625`；
- correct-PG 优于 swapped-PG 比例 `≥0.625`，平均 symmetry ratio `≤0.95`；
- PG effect Kabsch RMSD `≥0.001 Å`；
- raw analyzer success `≥0.90`，raw PG-compatible `≥0.25`。

没有恢复单分子 `0.05 Å` absolute Gate。

## 5. 三种预注册结论

1. 全部通过：`PASS_COORDINATE_FLOW_FAST32_ADVANCE_TO_IID_TRAIN`，准备 2,026/253 正式训练；
2. recovery 通过但 sampling 失败：
   `FAIL_COORDINATE_FLOW_FAST32_ALLOW_ONE_INTEGRATOR_REPAIR`，只允许一次积分器修复，不重训；
3. recovery 失败：`FAIL_COORDINATE_FLOW_FAST32_STOP_ENDPOINT_BRANCH`，停止 learned endpoint
   flow，转 symmetry-by-construction baseline。

## 6. 环境与机器证据

复用现有 `env_cof`，无需安装或下载：Python 3.10.16、PyTorch 1.13.1/CUDA 11.7、NumPy
1.26.4、SciPy 1.15.2、pymatgen 2025.3.10。CPU 工程测试 7/7 通过，包含完整配置 one-step、
Sn forward 和两步积分有限性。

- 冻结协议：`generative_model/smoke/reports/graph_pg_3d_coordinate_flow_fast32_protocol_v1.json`
- 协议 SHA-256：`791443909591b08246a5f755269ed09638bf5a4a891350788829cd942fa26545`
- 工程报告：`generative_model/smoke/reports/graph_pg_3d_coordinate_flow_fast32_engineering_v1.json`
- 工程报告 SHA-256：`f6f24aabda591026a101e13d6d5e228910dd1d0655d738d6e551284f21824fa4`

正式训练启动前应再次校验协议 SHA-256；训练过程中不得修改源码、协议或 Gate。

## 7. 正式结果（2026-08-22）

唯一一次 1,024-step CUDA 实验已结束，状态
`FAIL_COORDINATE_FLOW_FAST32_STOP_ENDPOINT_BRANCH`。`t=0.75` recovery ratio 为 `1.0370`，
prior sampling 只有 `1/16` collision-free、raw analyzer/compatible 均为 `0/16`，且 correct PG
仅在 `6/16` cases 优于 swapped PG。根据本文件第 5 节预注册规则，不允许 integrator repair，
endpoint flow 分支正式停止。完整指标、hash 与后续 orbit 路线见
[Fast-32 Flow 最终结果与 Orbit 硬对称接口](16_Fast32_Flow结果与Orbit硬对称接口.md)。
