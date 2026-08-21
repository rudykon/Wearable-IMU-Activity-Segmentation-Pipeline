# 数据与模型资产

此页面保留仓库原有的 `docs/ASSETS.md` 链接。当前资产映射、完整性检查表、环境变量
覆盖与许可边界，请参阅扩展后的[数据与模型资产指南](reference/assets.md)。

## 关键边界

- 本 GitHub 仓库**不分发**参与者记录。
- 经授权的记录保留在已被忽略的本地 `data/` 目录中。
- 选定的 Python 检查点与归一化文件在 `saved_models/` 下进行版本管理。
- 选定的 Android ONNX 与 JSON 资产在
  `android_realtime_app/app/src/main/assets/` 下进行版本管理。
- 其他本地生成的检查点、结果、图表与日志默认被忽略，除非有意整理发布。

数据访问方式与规范本地目录结构记录在[数据](guide/data.md)页面。
