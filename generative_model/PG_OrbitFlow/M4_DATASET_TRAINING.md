# M4 unseen 诊断与数据级训练准入

## 1. 为什么必须进入数据级训练

M3 在固定 Tier-32 面板上通过，只证明模型容量、orbit-IC 接口和 shape-aware decoder 能闭环，
不证明对新图泛化。冻结 M3 后，在未参加训练的 IID-validation C2/C3 面板上做了只读预测，正式
状态为 `FAIL_M4_UNSEEN_PREDICTION_PROCEED_TO_DATASET_TRAINING`。其中最差 bond、angle、torsion、
nonplanar torsion 和 shape 误差分别为 `0.0434787 Å`、`7.79241°`、`75.4199°`、
`115.081°` 和 `2.02556 Å`。因此不得把 M3 checkpoint 直接当作可泛化基座，也不在 validation
上调参；下一步必须从确定性随机初始化做 IID-train 数据级学习。

## 2. 数据合同

- canonical 数据：`generative_model/data/processed/v2`，唯一身份由 manifest SHA-256 固定；
- 训练：1,963 条严格支持样本，C2 1,594、C3 369；
- 验证：245 条严格支持样本，C2 199、C3 46；
- 训练 21 条、验证 3 条因 `torsion orbit has an undefined circular mean` 显式列入
  `*_unsupported`，不静默修复、不 fallback；
- IID-test 与 Core-OOD 完全保留，不参与当前训练、调参或模型选择；
- 每步 batch 8，固定 4 C2 + 4 C3，使较少的 C3 获得 1:1 exposure；验证仍保留自然分布。

模型输入只有 canonical 2D graph、Target-PG action/orbit 和 counted WL fingerprint，不读 target
Cartesian coordinates。监督输出为 bond-orbit length、angle-orbit cosine、在 graph automorphism
下集合等价的 torsion-orbit sin/cos，以及16维 long-range distance quantiles。完整训练联合更新
369,510 个 active 参数；旧 local-rotor head 已被更一般的 factorized automorphism head 取代并冻结。

## 3. 256-step 工程 smoke

smoke 使用64条训练（C2/C3各32）和16条 descriptive-only validation（各8），训练256步。
首32步/末32步平均 loss 为 `15.3869/2.65993`，比值 `0.172870`；loss/gradient 有限、每批
PG-balanced、train/validation 无交集，checkpoint 重载输出逐元素完全相同。

原 runner report 唯一失败项来自报告代码把“validation gradient 实际为 false”直接当作失败，
而协议本来就要求 false。训练科学协议和权重均未改变；独立审计修正为比较
`actual false == expected false` 后，状态为
`PASS_DATASET_TRAINER_SMOKE_READY_FOR_FULL_RUN`。审计文件：
`reports/dataset_training_smoke_audit_v1.json`，SHA-256
`2412d7a61056c222cbf3328b2f2d350088197254cd4f6deda624232e91a3cd5f`。

## 4. 为什么需要 factorized contract 缓存

graph-action-compatible automorphism witness 只依赖 canonical graph 和 Target-PG action，却需要
CPU GraphMatcher。训练时即时构建会让每次重启重复昂贵预处理。因此新增 pickle-free、逐分子
NPZ 缓存：

1. 每条记录独立保存 torsion representatives、允许的 torsion permutation closure 和诊断计数；
2. 每个 NPZ 写后立即严格回读，并验证 torsion support；
3. manifest 固定 record selection fingerprint、canonical manifest、实现 hash 和每个 NPZ hash；
4. local branch swap 先沿 Target-PG action 的共轭轨道同步组合；每个组合都重新验证 canonical graph
   automorphism 和与全部 `P_g` 对易；
5. 若同步生成元已经形成超过模型冻结上限的6槽等价组，立即严格失败，不再进入昂贵搜索；只有仍
   未连接的全局等价组才运行 anchored GraphMatcher，单次冻结上限5秒；
6. 超6槽、超时或其他合同失败均写成独立、可哈希的 `*.unsupported.json`，final 协议显式排除，
   不静默 fallback，也不会终止整个缓存任务；
7. 全量训练协议必须绑定完成的 cache manifest，启动时再次逐文件验 hash。

第一次 v1 全量缓存在第25条 `cof_000028` 停留约35分钟。诊断发现它有24个局部支链交换，沿C2
同步后形成18槽 torsion 等价组，超出当前6槽 head；新版在数秒内严格判定。新版80条端到端 smoke
缓存用77.11秒完成，79条成功，1条 validation 因5秒上限显式 unsupported，79/79均可重建模型
输入。加入缓存、Gate 语义与 optimizer-resume 回归后，PG-OrbitFlow 完整测试为61/61通过；续训测试
验证连续4步与“2步保存 + 2步恢复”的 loss 和全部参数逐位一致。

## 5. 当前状态与执行顺序

全量v2缓存已于2026-09-01完成，但尚未开始20,000-step训练。冻结的预缓存协议是
`configs/dataset_training_full_precache_v2.json`，SHA-256
`2b6d002989415f78a5683b4ba795f6c040bf505fcbf6b2eca709e7588d0e2d96`。v1协议及已完成的24个
v1缓存只保留为失败历史，因实现身份已经变化，不得混入v2。

v2 manifest 状态为 `PASS_FACTORIZED_CACHE_COMPLETE_WITH_EXPLICIT_UNSUPPORTED`：2,208条来源记录中
1,925条生成合同，283条显式unsupported。成功集合按split/PG为 train C2/C3
`1,373/327`、validation C2/C3 `182/43`；unsupported原因为 anchored 5秒上限39条、attempt cap
214条、超过6槽30条。manifest SHA-256为
`b1800439535b5a8d873e203d6f3b7d57ae308bf3ba05b27022ad211f3e98ab63`。

绑定manifest后的最终训练协议为 `configs/dataset_training_full_v1.json`，SHA-256
`c94a1841a290a15b52f1a2f2268ac3ef832bd6b8da10d8cee093b1177ed22470`。1,925个NPZ/unsupported
JSON均已逐文件验hash并pickle-free回读；C2/C3及最大87原子的代表样本完成forward/backward，全部
梯度有限。当前状态为 `PASS_FULL_TRAINING_READINESS`，可以启动长跑。

### 5.1 在 tmux 中预计算缓存（已完成，保留为复现命令）

```bash
cd /home/tianyajun/MARL_for_COFs
conda activate env_etflow
# 长跑允许 Ctrl-C 后保留交互 shell；不要启用 `set -e`。
set -uo pipefail

CACHE=generative_model/PG_OrbitFlow/cache/factorized_full_v2
python -m generative_model.PG_OrbitFlow.precompute_factorized_cache \
  --protocol generative_model/PG_OrbitFlow/configs/dataset_training_full_precache_v2.json \
  --output-dir "$CACHE" \
  2>&1 | tee "${CACHE}.console.log"

sha256sum "$CACHE/manifest.json"
```

该步骤为单核为主的 CPU 图同构预处理，不会占满48核服务器。可中断；用完全相同的命令重启会
复用已经完成的成功 NPZ 和 unsupported JSON。日志在第1条、每25条以及每个 unsupported 条目更新。

### 5.2 缓存完成后冻结 final 全量协议（已完成）

```bash
python -m generative_model.PG_OrbitFlow.build_dataset_training_protocol \
  --mode full \
  --m3-protocol generative_model/PG_OrbitFlow/configs/m3p6_shape_decoder_v2.json \
  --m3-report generative_model/PG_OrbitFlow/runs/m3p6_shape_decoder_v2/report.json \
  --m4-report generative_model/PG_OrbitFlow/runs/m4_unseen_prediction_v1/report.json \
  --factorized-audit generative_model/PG_OrbitFlow/reports/factorized_automorphism_m3_audit_v7.json \
  --smoke-audit generative_model/PG_OrbitFlow/reports/dataset_training_smoke_audit_v1.json \
  --factorized-cache-manifest "$CACHE/manifest.json" \
  --package-dir generative_model/data/processed/v2 \
  --output generative_model/PG_OrbitFlow/configs/dataset_training_full_v1.json
```

### 5.3 启动20,000-step训练

```bash
set +e
GPU_ID=4
OUT=generative_model/PG_OrbitFlow/runs/dataset_training_full_v1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -m generative_model.PG_OrbitFlow.dataset_training \
  --protocol generative_model/PG_OrbitFlow/configs/dataset_training_full_v1.json \
  --output-dir "$OUT" --device cuda \
  2>&1 | tee "${OUT}.console.log"
```

如果 tmux 内的 shell 曾执行过 `set -e`，必须先执行 `set +e`。否则 `Ctrl-C` 会让训练管道返回130，
shell 随即退出；当该 shell 是 tmux 窗口中的唯一进程时，整个 session 也会消失。checkpoint 不受影响。

每1,000步保存 checkpoint，并在 checkpoint 内保存完整 optimizer 与 loss history。若中断，从最近
完整 checkpoint 精确续训，例如：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -m generative_model.PG_OrbitFlow.dataset_training \
  --protocol generative_model/PG_OrbitFlow/configs/dataset_training_full_v1.json \
  --output-dir "$OUT" --device cuda \
  --resume-from "$OUT/step-001000.pt" \
  2>&1 | tee -a "${OUT}.console.log"
```

全量训练完成仍只准入独立的 IID-validation raw prediction/decoder Gate；在该 Gate 通过前，不能使用
IID-test/Core-OOD，也不能宣称模型已获得指定点群3D分子的泛化生成能力。

## 6. 最终结果（2026-09-02）

20,000-step训练已完成，执行状态为`PASS_DATASET_TRAINING_EXECUTION`。冻结checkpoint在全部225条
IID-validation记录上的独立复算与训练report逐项一致，但原prediction/shape Gate失败，因此按规则
停止在decoder之前，不查看IID-test/Core-OOD。详细结果与hash见
[M5_DATASET_TRAINING_RESULTS.md](M5_DATASET_TRAINING_RESULTS.md)。
