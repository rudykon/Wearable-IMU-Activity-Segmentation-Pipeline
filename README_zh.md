<p align="center">
  <a href="README.md">English</a> · <strong>中文</strong>
</p>

<p align="center">
  <img src="docs/assets/logo.svg" width="260" alt="可穿戴 IMU 活动分割流程 Logo">
</p>

<h1 align="center">可穿戴 IMU 活动分割流程</h1>

<p align="center">
  <strong>面向长时程可穿戴加速度计与陀螺仪信号的活动片段识别</strong><br>
  一套可复现 Python 研究流程，包含多尺度时序后处理和 Android 端侧 ONNX 演示。
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-%E2%89%A53.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12 或更高版本"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.5.1"></a>
  <a href="android_realtime_app/"><img src="https://img.shields.io/badge/Android-ONNX%20Runtime-3DDC84?style=flat-square&logo=android&logoColor=white" alt="Android ONNX Runtime 演示"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Smoke%20test-no%20raw%20data-2CA02C?style=flat-square" alt="轻量测试不需要原始数据"></a>
  <a href="https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/"><img src="https://img.shields.io/badge/Docs-项目网站-0F8F8C?style=flat-square&logo=materialformkdocs&logoColor=white" alt="项目网站"></a>
  <a href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Spaces-FFD21E?style=flat-square" alt="Hugging Face Spaces 在线演示"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-4C78A8?style=flat-square" alt="Apache License 2.0"></a>
</p>

<p align="center">
  <a href="https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/">项目网站</a> ·
  <a href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline">在线演示</a> ·
  <a href="#overview">项目概览</a> ·
  <a href="#pipeline">流程</a> ·
  <a href="#quick-start">快速开始</a> ·
  <a href="#data">数据</a> ·
  <a href="#model-assets">模型资产</a> ·
  <a href="#android-app">Android</a> ·
  <a href="#reproduction">复现</a> ·
  <a href="#license">许可证</a>
</p>

> [!IMPORTANT]
> 本 GitHub 仓库不分发参与者传感器记录。仓库作者编写的源码以及随仓库分发的 Python/Android 模型资产采用 Apache-2.0；数据集和第三方依赖仍遵循各自条款。

<a id="overview"></a>
## 项目概览

本项目将长时程可穿戴 IMU 会话分割为活动片段记录：

```text
user_id, category, start, end
```

Python 流程读取加速度计与陀螺仪信号，训练多尺度神经网络分类器，对齐 3 秒、5 秒和 8 秒窗口预测，执行时序解码与边界细化，并写出片段级预测表。仓库还包含面向 WT9011DCL-BT50 蓝牙 IMU 的 Android App，用于实时采集、可视化、CSV 录制和端侧 ONNX 推理。

| 目标 | 已实现方法 | 公开边界 |
| --- | --- | --- |
| 分割长时程可穿戴运动信号 | 多核 1D-CNN + BiLSTM 滑窗分类器 | 完整推理/训练需要授权本地传感器文件 |
| 提升时序一致性 | 多尺度概率对齐、LBSA 融合、平滑、Viterbi 解码、边界细化、重叠消解、置信度过滤和 Top-K 裁剪 | 轻量测试只使用临时文件 |
| 支持可部署演示 | Android BLE 采集与 ONNX Runtime 推理 | 随仓库模型资产与私有数据集分开说明 |
| 在浏览器中体验真实模型 | 免费 ZeroGPU Gradio Space，支持上传、曲线、时序解码、片段表与 CSV 导出 | 内置合成示例；公开上传不得包含敏感受试者数据 |
| 保持实验可复现 | 评估、鲁棒性、可视化和公开数据集可迁移性脚本 | 生成产物保留在被忽略的本地目录 |

支持的前景活动包括 `羽毛球`、`跳绳`、`飞鸟`、`跑步` 和 `乒乓球`。背景/无活动在必要时作为内部类别建模，但提交的片段记录只包含前景活动。

<a id="pipeline"></a>
## 流程

<p align="center">
  <a href="experiments/figures/fig02_overall_framework.png">
    <img src="experiments/figures/fig02_overall_framework.png" alt="可穿戴 IMU 活动分割整体框架" width="92%">
  </a>
</p>
<p align="center"><em>图 1｜从原始 IMU 信号到活动片段记录的整体活动分割框架。</em></p>

核心组成：

- `data/` 下统一管理信号、标注、划分、元数据和可选公开外部数据集。
- 核心源码位于 `src/imu_activity_pipeline/`，根目录脚本作为兼容入口保留。
- 提供顺序训练、并行训练和单模型训练入口。
- 片段级评估采用同类别一对一 IoU 匹配。
- 实验脚本覆盖内部评估、后处理策略检查、公开数据集可迁移性检查和图件生成。
- Android 演示包含 BLE 采集、实时视图、离线识别和端侧多尺度推理。

<a id="quick-start"></a>
## 快速开始

```bash
git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
cd Wearable-IMU-Activity-Segmentation-Pipeline

conda env create -f environment.yml
conda activate imu-activity-pipeline
python -m pip install -e .
python tests/smoke_test.py
```

轻量测试会检查导入、规范路径、临时小信号读取、标注读取和工作簿写出，不需要私有原始数据或训练检查点。

### 浏览器演示

[`demo/`](demo/) 模块通过双语 Gradio 界面运行仓库跟踪的 3 秒、5 秒和 8 秒
检查点。你可以上传规范的 100 Hz TXT/TSV 记录，或直接使用内置合成示例，
查看六路信号、类别概率、解码时间线、最终片段表，并下载 CSV。

[**打开 Hugging Face Space →**](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline)

也可在本地运行同一界面：

```bash
python -m pip install -r requirements.txt
python -m pip install spaces
python -m pip install -e .
python demo/app.py
```

若使用纯 pip 环境，请先安装固定版本依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

将授权数据放入 `data/`、模型资产放入 `saved_models/` 后，运行推理：

```bash
python run_inference.py
```

默认划分为 `external_test`，默认输出为：

```text
predictions_external_test.xlsx
```

常用根目录命令：

```bash
python train.py
python train_parallel.py
python evaluate.py --split external_test
python -m imu_activity_pipeline.inference \
  --data_dir data/signals/internal_eval \
  --output predictions_internal_eval.xlsx
```

训练、评估、Python 接口、打包可执行文件布局和实验脚本见 [docs/USAGE.md](docs/USAGE.md)。

<a id="data"></a>
## 数据

本 GitHub 仓库不直接分发数据集。仓库保留预期本地目录结构和数据使用说明，便于获授权用户一致放置文件。

| 组成 | 默认本地路径 | 说明 |
| --- | --- | --- |
| 信号流 | `data/signals/{train,internal_eval,external_test}/` | UTF-8 制表符分隔 `.txt` 文件 |
| 标注 | `data/annotations/*_annotations.csv` | `split,user_id,category,start,end` |
| 划分与元数据 | `data/splits/`, `data/metadata/` | 用户列表、清单、标签汇总和数据集元数据 |
| 可选公开数据集 | `data/public_external/` | 用户自行下载；各数据集遵循自身许可 |

在 PhysioNet 仓库正式发布前，研究用途访问通过 [data/README_zh.md](data/README_zh.md) 中维护的腾讯问卷链接申请。PhysioNet 发布后，请遵循仓库文档中维护的 PhysioNet 链接和引用信息。

<a id="model-assets"></a>
## 模型资产

Python 研究流程以代码为主。`saved_models/` 下跟踪部分复现检查点、归一化参数和集成配置；额外本地训练输出由 Git 忽略。

默认多尺度推理需要：

```text
saved_models/ensemble_config.json
saved_models/combined_model_3s_seed42.pth
saved_models/combined_model_5s_seed123.pth
saved_models/combined_model_8s_seed123.pth
saved_models/norm_params_3s.pkl
saved_models/norm_params_5s.pkl
saved_models/norm_params_8s.pkl
```

资产说明：

- [docs/ASSETS.md](docs/ASSETS.md) 描述本地数据、检查点和生成产物边界。
- [saved_models/WEIGHTS_LICENSE](saved_models/WEIGHTS_LICENSE) 适用于随仓库分发的 Python 模型资产。
- [android_realtime_app/MODEL_CARD.md](android_realtime_app/MODEL_CARD.md) 说明 Android ONNX 资产、校验和、预期用途和限制。
- [android_realtime_app/WEIGHTS_LICENSE](android_realtime_app/WEIGHTS_LICENSE) 适用于随仓库分发的 Android 模型资产。

<a id="android-app"></a>
## Android App

[android_realtime_app/](android_realtime_app/) 中的 Android 演示支持 WT9011DCL-BT50 BLE 扫描/连接、实时加速度和角速度图、姿态/指南针/轨迹视图、CSV 录制、离线文件识别和端侧 3 秒/5 秒/8 秒 ONNX 推理。

<p align="center">
  <a href="experiments/figures/fig03_physical_deployment_chain.png">
    <img src="experiments/figures/fig03_physical_deployment_chain.png" alt="可穿戴 IMU 采集与 Android 推理物理部署链路" width="92%">
  </a>
</p>
<p align="center"><em>图 2｜从可穿戴 IMU 采集到 Android 端识别的物理部署链路。</em></p>

可使用 Android Studio 构建，或在 JDK 17 + Android SDK 环境下运行：

```bash
cd android_realtime_app
./gradlew assembleDebug
```

BLE 集成说明和桌面调试工具见 [android_realtime_app/docs/README.md](android_realtime_app/docs/README.md) 与 [android_realtime_app/tools/desktop/README.md](android_realtime_app/tools/desktop/README.md)。

<a id="reproduction"></a>
## 复现

顶层实验封装脚本为：

```bash
bash run_reproducibility_experiments.sh
```

该脚本会运行已保存模型评估、内部鲁棒性检查、策略选择检查、PPG 信号质量分析、代表性时间线图、外部无标注队列压力测试和汇总图生成。它需要 [docs/ASSETS.md](docs/ASSETS.md) 中说明的本地数据和检查点资产。

输出目录：

```text
experiments/results/
experiments/figures/
experiments/logs/
```

指定解释器：

```bash
PYTHON_BIN=/path/to/python bash run_reproducibility_experiments.sh
```

<a id="repository-map"></a>
## 项目结构

| 路径 | 作用 |
| --- | --- |
| `src/imu_activity_pipeline/` | 核心 Python 包，覆盖配置、加载、训练、推理、后处理和评估 |
| `run_inference.py`, `train.py`, `train_parallel.py`, `evaluate.py` | 源码检出环境下的兼容入口 |
| `saved_models/` | 已跟踪复现资产及被忽略的本地训练输出 |
| `data/` | 本地数据目录占位与访问说明 |
| `experiments/` | 评估、鲁棒性、可视化和公开数据集可迁移性脚本 |
| `scripts/` | 辅助分析、调参和图件工具 |
| `android_realtime_app/` | Android BLE 采集、可视化、录制和 ONNX 推理 App |
| `docs/` | 使用说明与资产边界文档 |
| `tests/` | 轻量公开健康检查 |

<a id="license"></a>
## 许可证

仓库作者编写的源码以及随仓库分发的 Python、Android 模型资产均采用 [Apache License 2.0](LICENSE)。适用范围副本位于 [saved_models/WEIGHTS_LICENSE](saved_models/WEIGHTS_LICENSE)、[android_realtime_app/LICENSE](android_realtime_app/LICENSE) 和 [android_realtime_app/WEIGHTS_LICENSE](android_realtime_app/WEIGHTS_LICENSE)。数据集和第三方依赖仍分别遵循各自条款。
