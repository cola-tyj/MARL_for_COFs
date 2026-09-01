# Cn：our_ET_Flow 2D graph → C3 3D 结果

输入来自 `generative_model/data/Cn`，共 98 个外部 2D 重原子图：C3k2 24 个、C3k3
74 个。批处理严格验证 `x_idx/adj_corr/SMILES` 一致性，再显式补全 H；随后原样复用冻结的
our_ET_Flow v5 路线：ET-Flow ODE 初始构象 → C3 graph action → hard symmetry projection →
F0.2 orbit-constrained UFF + nonbonded repulsion。输入不含、流程也不读取参考 3D 坐标。

## 本次结果（2026-08-28）

- 98/98 完成生成并通过 geometry gate，失败 0 条；
- 98/98 直接使用的 `xyz/*.xyz` 与 `coordinates.npz` 中的 final 坐标逐原子一致；
- 独立 pymatgen 审计为 98/98 C3-compatible，其中 32/98 exact C3；
- actual PG：C3 32、C3h 6、C3v 39、D3 7、D3d 3、D3h 9、D6h 1、Oh 1；
- 全批最小非键/全原子间距为 `0.7614427425 Å`，最大目标操作误差为
  `2.7200464e-15 Å`；
- 最终状态：`PASS_OUR_ETFLOW_CN_2D_GRAPH_TO_C3_XYZ`。

`manifest.json` 覆盖输入、适配/推理/审计源码、冻结协议、所有 XYZ、NPZ 和 JSON 报告；
本次 manifest SHA-256 为
`ad48c437dd087c18a4a6561ae4630ce7bf8a83c96edc1feb333d331f03bfcec4`。

## 怎样判断“是否合法”

“合法”不是单一布尔值。本目录新增 [validity_report.json](validity_report.json) 和
[validity_records.csv](validity_records.csv)，分四层报告：

1. **化学图合法**：元素/键/电荷保持不变，RDKit sanitize 通过、分子连通、重建 SMILES
   与源 SMILES 一致。本批为 **98/98**。
2. **pipeline 3D 合法**：坐标有限且质心归零、全原子和非键最小距离均不低于 `0.6 Å`、
   F0.2 optimizer termination 可接受、独立审计 C3-compatible。本批为 **98/98**。
3. **保守键长筛查**：每条键长除以两原子共价半径之和，要求落在 `[0.65, 1.35]`。
   本批为 **95/98**。
4. **解除约束 UFF 稳定性**：独立无约束 UFF 松弛后能量不升，Kabsch RMSD 不超过
   `0.5 Å`。本批为 **97/98**。

综合保守筛查为 **94/98 PASS + 4 WARNING**。WARNING 不等于化学图非法：

| 分子 | 警告 |
|---|---|
| `mol_019_C3k2` | 高张力含氮 cage 的最短键/共价半径比为 0.594 |
| `mol_002_C3k3` | cage 中最长键/共价半径比为 1.434 |
| `mol_043_C3k3` | 最长键/共价半径比为 1.363，略超保守阈值 |
| `mol_066_C3k3` | 无约束 UFF 松弛 RMSD 为 1.330 Å；表明强制 C3 构象可能有较高应变 |

机器可读警告见 [validity_warnings.csv](validity_warnings.csv)。若这些结构要进入后续材料
计算，建议优先对 4 条 warning 做 GFN2-xTB，再对候选做 DFT 优化；UFF 筛查不能证明
热力学稳定性。

## 批量可视化

- [浏览器图像索引](visualization/index.html)：98 张 2D 输入图 / final 3D 对照图；
- [C3k2 分页汇总](visualization/contact_sheets/C3k2_page_01.png)；
- [C3k3 分页汇总](visualization/contact_sheets/C3k3_page_01.png)；
- 单分子 PNG 位于 `visualization/C3k2/` 和 `visualization/C3k3/`；
- 图像 SHA-256 清单为 `visualization/manifest.json`。

重新审计和绘图：

```bash
conda run -n env_cof python -m generative_model.evaluation.audit_our_etflow_cn_validity
conda run -n env_cof python -m generative_model.visualization.render_our_etflow_cn
```

结果布局：

```text
Cn/
├── batch_report.json
├── index.csv
├── point_group_batch_report.json
├── point_group_index.csv
├── C3k2/
│   ├── xyz/mol_001_C3k2.xyz ...       # 推荐直接使用的最终坐标
│   └── records/mol_001_C3k2/           # raw/projected/final/NPZ/审计报告
└── C3k3/
    ├── xyz/mol_001_C3k3.xyz ...
    └── records/mol_001_C3k3/...
```

生成环境为 `env_etflow`，独立点群审计环境为 `env_cof`。可中断续跑命令：

```bash
conda run -n env_etflow python -m generative_model.inference.run_our_etflow_cn \
  --device cuda --resume

conda run -n env_cof python -m generative_model.inference.audit_our_etflow_cn
```

最终结论以 `point_group_batch_report.json` 为准；单个分子的 `final.xyz` 不能替代独立
actual-PG/compatible 审计。
