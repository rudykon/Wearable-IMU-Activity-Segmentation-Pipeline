# 数据与模型资产

此页面保留仓库原有的 `docs/ASSETS.md` 链接。当前资产映射、完整性检查表、环境变量
覆盖与许可边界，请参阅扩展后的[数据与模型资产指南](reference/assets.md)。

## 关键边界

- 本 GitHub 仓库**不分发**参与者记录。
- 经授权的记录保留在已被忽略的本地 `data/` 目录中。
- Python 检查点、归一化文件和 Android ONNX 权重在
  [Hugging Face Model 仓库](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline)
  公开发布。
- GitHub 仅跟踪资产清单与小型配置，不保存模型二进制文件；缺失资产会在使用前下载并校验。
- 其他本地生成的检查点、结果、图表与日志默认被忽略，除非有意整理发布。

数据访问方式与规范本地目录结构记录在[数据](guide/data.md)页面。
