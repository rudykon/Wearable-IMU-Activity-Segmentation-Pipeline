<p align="center">
  <a href="README.md">English</a> · <strong>中文</strong>
</p>

<p align="center">
  <img src="docs/assets/logo.svg" width="260" alt="可穿戴 IMU 活动分割流程 Logo">
</p>

<h1 align="center">可穿戴 IMU 活动分割流程</h1>

<p align="center">
  <strong>把连续腕部运动转换为带时间戳的活动记录</strong><br>
  回答“做了什么、何时开始、何时结束”，并提供 Python 研究代码、浏览器演示和 Android 应用。
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

### 为什么需要这个项目

可穿戴惯性测量单元（IMU）会连续记录加速度和旋转运动。一小时数据可能包含几十万行
传感器数值，但研究人员或应用使用者通常只需要几个清楚的答案：

- 做了什么活动？
- 活动什么时候开始、什么时候结束？
- 一共发生了几段活动？

本项目把连续信号转换为下面这样的活动记录：

```text
user_id, category, start, end
```

系统会用 3 秒、5 秒和 8 秒三种时间窗口观察同一段记录。短窗口更容易找到活动
切换位置，长窗口则能提供更稳定的动作背景。系统随后组合三种预测，并处理短暂中断、
误报和边界偏移，最后输出活动记录表。

### 哪些场景会用到

- **长时活动识别研究：**比较完整活动时间线，而不只评价相互独立的短窗口。
- **受控训练日志：**为羽毛球、跳绳、哑铃飞鸟、跑步和乒乓球生成候选记录。
- **辅助标注与质量检查：**帮助审核者快速找到可能的活动区间和边界问题。
- **移动端部署实验：**通过蓝牙采集六轴信号，并在 Android 端运行对应 ONNX 模型。

仓库包含 Python 流水线、固定模型资产、免费浏览器演示、Android 原型，以及评估和
复现实验脚本。完整训练和数据集推理需要获授权的本地传感器文件。公开 Demo 默认使用
合成数据，请勿向公共 Space 上传敏感受试者记录。

<a id="pipeline"></a>
## 系统怎样工作

<p align="center">
  <a href="experiments/figures/fig02_overall_framework.png">
    <img src="experiments/figures/fig02_overall_framework.png" alt="可穿戴 IMU 活动分割整体框架" width="92%">
  </a>
</p>
<p align="center"><em>图 1｜从连续 IMU 记录到简洁活动列表的完整流程。</em></p>

1. **读取完整会话。**载入时间戳、三路加速度和三路角速度。
2. **同时观察不同时间范围。**3 秒、5 秒和 8 秒模型分别关注快速切换与持续动作。
3. **整理稳定时间线。**对齐三种预测，减少标签抖动，合并合理间隔，修正边界并过滤弱记录。
4. **输出可用记录。**导出活动类别、开始时间和结束时间，再与人工标注片段进行比较。

CNN–BiLSTM、局部边界尺度仲裁（LBSA）、Viterbi 解码和时间记录层（TRL）等技术名词，
会在[架构页面](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/zh/guide/pipeline/)中逐项解释。

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

浏览器 Demo 是理解输出最快的方式。你可以使用内置合成会话，或上传兼容的
100 Hz TXT/TSV 文件。页面会展示六路传感器信号、活动可能性随时间的变化、最终
开始—结束记录和可下载 CSV。Demo 运行仓库中的 3 秒、5 秒和 8 秒真实模型。

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
