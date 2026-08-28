# COF graph + 3D + symmetry dataset v2

`cof_graphs.npz` 保持模型无关的 canonical molecular dataset；`symmetry_data.npz` 通过 package/molecule index 一一对应。

重新构建与验证：

```bash
python -m generative_model.data.build_v2_package
python -m generative_model.data.validate_v2 --package generative_model/data/processed/v2
```
