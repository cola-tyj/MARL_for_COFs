# 生成模型环境说明

## 当前状态

`env_gen` 已于 2026-08-12 创建并通过核心验收。实际环境路径为
`/home/tianyajun/anaconda3/envs/env_gen`，不修改数据构建环境 `env_cof`。

已核验的直接依赖版本：

| 组件 | 版本 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.5.0+cu121 |
| PyTorch CUDA build | 12.1 |
| NumPy | 1.26.2 |
| Pandas | 2.2.2 |
| SciPy | 1.11.4 |
| RDKit | 2024.03.5 |
| openbabel-wheel | 3.1.1.22（底层 `OBReleaseVersion=3.1.0`） |
| Lightning | 2.5.0.post0 |
| TorchMetrics | 1.6.1 |
| tqdm | 4.70.0 |
| typing-extensions | 4.12.2 |

[env_gen.yml](env_gen.yml) 保留为环境目标规格。由于当前服务器上的 Conda 24.5.0 在
整份 YAML 的依赖事务中会停滞，本机实际采用“最小 Conda 环境 + 分组 pip 安装”。该问题
不是 SemlaFlow 依赖冲突。

## 本机实际创建流程

在终端依次运行：

```bash
CONDA_NO_PLUGINS=true conda create -n env_gen python=3.11 pip \
  --solver classic --no-default-packages -y
conda activate env_gen
```

安装 PyTorch CUDA 12.1 wheel：

```bash
python -m pip install torch==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

安装基础科学计算依赖：

```bash
python -m pip install \
  numpy==1.26.2 \
  pandas==2.2.2 \
  scipy==1.11.4 \
  tqdm==4.70.0 \
  typing-extensions==4.12.2
```

安装化学与训练依赖：

```bash
python -m pip install rdkit==2024.3.5
python -m pip install openbabel-wheel==3.1.1.22
python -m pip install lightning==2.5.0.post0 torchmetrics==1.6.1
```

下载官方 Google Drive 资产时额外安装了工具依赖（不参与模型计算）：

```bash
python -m pip install gdown==5.2.0
```

锁定并校验官方 SemlaFlow 源码：

```bash
cd /home/tianyajun/MARL_for_COFs
python -m generative_model.environment.bootstrap_sources --model semlaflow

PYTHONPATH="$PWD/generative_model/external/semla-flow:$PWD" \
python -c "from semlaflow.util.molrepr import GeometricMol; \
from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator; \
print('SemlaFlow import: OK')"
```

## 验收命令

基础版本与 GPU：

```bash
python -c "import torch, numpy, pandas, scipy, lightning, torchmetrics; \
from rdkit import rdBase; from openbabel import openbabel; \
print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
print(numpy.__version__, pandas.__version__, scipy.__version__); \
print(rdBase.rdkitVersion, openbabel.OBReleaseVersion()); \
print(lightning.__version__, torchmetrics.__version__)"
```

canonical 数据与官方对象无损往返：

```bash
python -m unittest \
  generative_model.tests.test_smoke_split.TestSemlaFlowSmokeSplit.test_official_geometric_mol_bytes_roundtrip_is_lossless \
  -v
```

CUDA 前向、反向、旋转与排列一致性：

```bash
CUDA_VISIBLE_DEVICES=4 python -m generative_model.smoke.run_semlaflow_forward \
  --device cuda:0 \
  --output generative_model/smoke/reports/semlaflow_random_forward_env_gen_cuda.json
```

上述三组检查均已通过。正式报告为
[semlaflow_random_forward_env_gen_cuda.json](../smoke/reports/semlaflow_random_forward_env_gen_cuda.json)：
使用冻结 smoke split 中的 87 原子分子，输出形状正确、梯度有限，旋转/排列最大误差
小于 `3.6e-7`。这证明环境和模型接口可工作，但不等价于官方 checkpoint 已复现。

## MiDi 隔离环境（已冻结）

SemlaFlow 的 `env_gen` 保留用于复现实验，不在其中降级 PyTorch/Lightning。MiDi 官方
checkout 已锁定，但官方测试栈明显更旧，因此使用独立环境：

`env_midi` 已于 2026-08-12 实际创建并通过验收；[env_midi.yml](env_midi.yml) 记录实际
版本目标。由于 Conda 在线完整索引求解会停滞，本机先用缓存离线创建最小环境，再分组 pip
安装。最终核心版本为 Python 3.9.25、PyTorch 2.0.1+cu118、NumPy 1.25.0、PyG 2.3.1、
RDKit 2023.03.2、Lightning 2.0.4。

```bash
CONDA_NO_PLUGINS=true conda create -n env_midi python=3.9 pip \
  --solver classic --offline --no-default-packages -y
conda activate env_midi

python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
python -m pip install torch-geometric==2.3.1

python -m pip install \
  hydra-core==1.3.2 omegaconf==2.3.0 numpy==1.25.0 pandas==2.0.2 \
  scikit-learn==1.2.2 pytorch-lightning==2.0.4 torchmetrics==0.11.4 \
  wandb==0.15.4 matplotlib==3.7.0 imageio==2.31.1 Pillow==9.5.0 \
  rdkit==2023.3.2

cd /home/tianyajun/MARL_for_COFs
python -m generative_model.environment.bootstrap_sources --model midi
python -c "import torch, torch_geometric, hydra, omegaconf; \
print(torch.__version__, torch_geometric.__version__)"
```

安装完成后执行官方 MiDi import、canonical→PyG bridge 和单 batch CPU/CUDA Gate。官方
GEOM 显式 H checkpoint 已实际取得并登记 SHA-256，详见模型文档。

实际验收结果：

- 官方 `GraphTransformer`、`MarginalUniformTransition`、`FullDenoisingDiffusion` 导入通过；
- PyG CUDA `scatter` 扩展通过；
- canonical→PyG→canonical 的 atom/charge/bond/coordinate 往返逐数组一致；
- 官方 `to_dense` 保持 atom/charge target 一致；
- 87 原子随机 mini `GraphTransformer` 在 CPU/CUDA 上前后向均通过，输出和梯度有限；
- CUDA 前后向约 0.29 秒，峰值显存 93,937,664 bytes。

报告：

- [MiDi compatibility](../models/reports/midi_compatibility.json)
- [CPU forward](../smoke/reports/midi_random_forward_cpu.json)
- [CUDA forward](../smoke/reports/midi_random_forward_cuda.json)

## 下一步

`env_gen` 与 `env_midi` 均保留用于复现，不再修改其依赖。SemlaFlow 与 MiDi 已完成实验
收尾，均不进入 tier-4 或 2,532 条全量训练；详见
[models/README.md](../models/README.md)。

当前主候选已切换为 UAE-3D / UDM-3D。为避免破坏两个历史环境，已从 `env_gen` 克隆出
独立 `env_uae3d`，从而复用 PyTorch/CUDA 与化学依赖而不重新下载。随后安装了严格匹配
PyTorch 2.5.0+cu121 的 PyG wheels 和 `overrides`。

| 组件 | `env_uae3d` 实际版本 |
|---|---|
| Python | 3.11.15 |
| PyTorch / CUDA build | 2.5.0+cu121 / 12.1 |
| PyG | 2.6.1 |
| torch-scatter | 2.1.2+pt25cu121 |
| torch-sparse | 0.6.18+pt25cu121 |
| torch-cluster | 1.6.3+pt25cu121 |
| NumPy / Pandas / SciPy | 1.26.2 / 2.2.2 / 1.11.4 |
| RDKit / OpenBabel | 2024.03.5 / 3.1.0 |
| Lightning / TorchMetrics | 2.5.0.post0 / 1.6.1 |
| overrides | 7.7.0 |

官方 `UnifiedAutoEncoder`、`DiffusionTransformer` 和 `LatentDiffusion` 核心导入通过；PyG
CUDA scatter、严格 bridge、CPU 10 原子和 A6000 101 原子随机前后向均通过。当前不安装
存在旧 RDKit 兼容风险的 MOSES，也不需要 W&B/LoRA 依赖。U2 reconstruction trainer、
确定性固定评估和 checkpoint/resume 短测也已通过；下一步是在该环境运行 tier-1 512-step
校准。清理后的最终状态和最小复现入口见 [历史模型最小保留集](../models/README.md)。

本机默认 CUDA 顺序为 `FASTEST_FIRST`，与 `nvidia-smi` index 不一致；指定物理 GPU 前应
先设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。当前 tier-1 任务使用该设置映射到物理 GPU 4。

## Coordinate Flow Fast-32 环境复用

2026-08-22 的新 coordinate-flow 训练不新建环境，复用已有 `env_cof`，因为它同时具备
Semla 所需 PyTorch/CUDA 和最终 point-group analyzer 所需 pymatgen：

| 组件 | 冻结版本 |
|---|---|
| Python | 3.10.16 |
| PyTorch / CUDA build | 1.13.1 / 11.7 |
| NumPy / SciPy | 1.26.4 / 1.15.2 |
| pymatgen | 2025.3.10 |

完整配置 one-step、Sn forward、两步 Euler sampling 和协议身份测试已在该环境通过 7/7。
正式 Fast-32 protocol 会严格检查这些版本；不需要下载新数据或安装新依赖。

## GFN2-xTB 物理松弛补充依赖

2026-08-27 在同一 `env_cof` 中追加安装 `tblite==0.7.0`，保留原有 Python、NumPy、ASE 和
pymatgen 版本不变。安装命令：

```bash
conda activate env_cof
python -m pip install --no-deps tblite==0.7.0
python -c "from importlib.metadata import version; from tblite.ase import TBLite; print(version('tblite'))"
```

当前验证版本为 Python `3.10.16`、NumPy `1.26.4`、ASE `3.24.0`、tblite `0.7.0`。
Our ET-Flow 32 条无约束 GFN2-xTB pilot 已在该环境运行完成；正式协议固定
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`，避免线程设置造成数值漂移。
