# 资产

代码、参与者数据、选定的可复现权重、Android 资产与生成的实验材料具有不同的
分发边界。使用或扩展仓库时，请明确保持这些边界。

## 范围

| 资产类别 | 仓库状态 | 说明 |
| --- | --- | --- |
| 仓库原创源码 | 已跟踪 | Apache-2.0 |
| 文档与公开冒烟测试 | 已跟踪 | 不需要私有数据 |
| 参与者传感器流 | 不分发 | 仅保存在已被忽略的本地 `data/` 路径 |
| 选定的 Python 检查点 | Hugging Face | 需要时下载到 `saved_models/` |
| 选定的归一化文件 | Hugging Face | 与各尺度检查点配套 |
| Android ONNX 资产 | Hugging Face | Android 构建前自动下载 |
| 模型清单 | 已跟踪 | `model-assets.json` 中的文件大小和 SHA-256 |
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

离线工作前可提前下载：

~~~bash
python scripts/download_model_assets.py python
~~~

Python 推理会在文件缺失时自动完成这一步。已有的本地重训练文件不会被覆盖，除非明确使用
`--force`。

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

构建过程会从公开模型仓库下载选定的 3 秒、5 秒、8 秒 ONNX 模型与旧版回退模型。
JSON 归一化参数体积很小，仍作为运行配置跟踪。SHA-256 与运行假设请参阅
[Android 模型卡](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/android_realtime_app/MODEL_CARD.md)。

~~~bash
python scripts/download_model_assets.py android
~~~

## 路径

| 变量 | 默认值 |
| --- | --- |
| `HLS_HAR_DATA_ROOT` | `<runtime>/data` |
| `HLS_HAR_TRAIN_DATA_DIR` | `data/signals/train` |
| `HLS_HAR_INTERNAL_EVAL_DATA_DIR` | `data/signals/internal_eval` |
| `HLS_HAR_EXTERNAL_TEST_DATA_DIR` | `data/signals/external_test` |
| `HLS_HAR_MODEL_DIR` | `<bundle>/saved_models` |
| `HLS_HAR_MODEL_REPO_ID` | `config-h/Wearable-IMU-Activity-Segmentation-Pipeline` |
| `HLS_HAR_MODEL_REVISION` | `main` |
| `HLS_HAR_OFFLINE` | 默认未设置；设为 `1` 时禁止下载 |
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
- Python 与 Android 权重：[公开 Hugging Face Model 仓库](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline)（Apache-2.0）。
- Android 源码：`android_realtime_app/LICENSE`。
- 数据集与第三方依赖：遵循各自条款。
