# 生成模型可视化

## E2 代表性 XYZ

从正式 E2 产物确定性导出五类代表样本：

```bash
conda activate env_cof
cd /home/tianyajun/MARL_for_COFs
python -m generative_model.visualization.export_etkdg_orbit_e2_examples
```

输出位于 `generative_model/visualization/e2_examples/`。每个示例包含：

- `.xyz`：E2 生成的质心归零坐标，单位 Å；
- `.json`：输入 canonical graph、Target_PG operation matrices、原子 permutation/orbit、
  选中 seed、几何指标和独立 actual-PG 结果；
- `examples.json`：示例索引和产物 SHA-256。

在 VS Code/Jupyter 中打开 `visualize_etkdg_orbit_e2_examples.ipynb`，选择示例后即可在单元格
内旋转、缩放。显示的键由 3Dmol.js 根据 XYZ 距离推断；化学键的 canonical ground truth
保存在同名 JSON 的 `input.known_graph` 中。
