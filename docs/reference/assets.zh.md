# 资产

代码、参与者数据、选定的可复现权重、Android 资产与生成的实验材料具有不同的
分发边界。使用或扩展仓库时，请明确保持这些边界。

## 范围

| 资产类别 | 仓库状态 | 说明 |
| --- | --- | --- |
| 仓库原创源码 | 已跟踪 | Apache-2.0 |
| 文档与公开冒烟测试 | 已跟踪 | 不需要私有数据 |
| 参与者传感器流 | 不分发 | 仅保存在已被忽略的本地 `data/` 路径 |
| 选定的 Python 检查点 | 已跟踪 | `saved_models/` 下的可复现／推理资产 |
| 选定的归一化文件 | 已跟踪 | 必须与检查点尺度配对 |
| Android ONNX 资产 | 已跟踪 | 存储在 Android 应用资产目录中 |
| 生成的检查点与日志 | 默认仅本地 | 除非有意整理，否则被忽略 |
| 可选公开数据集 | 由用户下载 | 遵循原始许可与引用要求 |

## Python

~~~text
saved_models/
├── ensemble_config.json
├── combined_model_3s_seed42.pth
├── combined_model_5s_seed123.pth
├── combined_model_8s_seed123.pth
├── norm_params_3s.pkl
├── norm_params_5s.pkl
└── norm_params_8s.pkl
~~~

`ensemble_config.example.json` 记录配置结构。

!!! danger "不要混用尺度或训练运行"

    加载模型时必须使用训练该模型时的归一化参数、通道顺序、窗口长度与类别映射。
    某个文件组合能够成功加载，并不表示该组合有效。

## 研究目录

~~~text
data/
  signals/{train,internal_eval,external_test}/
  annotations/
  splits/
  metadata/
  public_external/
  raw/

saved_models/
experiments/results/
experiments/figures/
experiments/logs/
~~~

本地数据目录树只有占位文件和说明被纳入版本控制。仓库 `.gitignore` 规则可以降低
意外公开风险，但不能替代正常的数据治理审查。

## Android 资产

应用包含选定的 3 秒、5 秒、8 秒 ONNX 模型和 JSON 归一化文件，以及一个旧版
回退模型。SHA-256 校验和与运行假设请参阅
[Android 模型卡](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/android_realtime_app/MODEL_CARD.md)。

## 路径

| 变量 | 默认值 |
| --- | --- |
| `HLS_HAR_DATA_ROOT` | `<runtime>/data` |
| `HLS_HAR_TRAIN_DATA_DIR` | `data/signals/train` |
| `HLS_HAR_INTERNAL_EVAL_DATA_DIR` | `data/signals/internal_eval` |
| `HLS_HAR_EXTERNAL_TEST_DATA_DIR` | `data/signals/external_test` |
| `HLS_HAR_MODEL_DIR` | `<bundle>/saved_models` |
| `HLS_HAR_INFERENCE_SPLIT` | `external_test` |
| `HLS_HAR_EVALUATION_SPLIT` | `external_test` |

## 完整性

报告或部署结果前，请记录：

- Git 提交；
- 选定检查点的文件名与哈希；
- 归一化文件名与哈希；
- `ensemble_config.json`；
- 数据划分／清单版本；
- 后处理策略参数；
- 运行时与依赖版本；
- 准确的评估命令。

## 许可

- 仓库原创代码：[Apache License 2.0](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/LICENSE)。
- Python 模型资产：`saved_models/WEIGHTS_LICENSE`。
- Android 源码与权重：`android_realtime_app/LICENSE` 和
  `android_realtime_app/WEIGHTS_LICENSE`。
- 数据集与第三方依赖：遵循各自条款。
