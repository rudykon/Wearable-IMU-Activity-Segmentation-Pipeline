<p align="center">
  <a href="README.md">English</a> · <strong>中文</strong>
</p>

<p align="center">
  <img src="docs/assets/logo-horizontal.svg" width="520" alt="可穿戴 IMU 活动分割流程 Logo">
</p>

<h1 align="center">基于多尺度仲裁与时间记录层的端到端可穿戴 IMU 片段级活动识别系统</h1>

<p align="center">
  将连续 100 Hz 腕部 IMU 信号转换为包含活动类别与起止时间的片段记录。
</p>

<p align="center">
  <a href="https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/zh/">网站</a> ·
  <a href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline">演示</a> ·
  <a href="https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline">模型</a> ·
  <a href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/releases/tag/v0.1.0-research-preview">研究版本</a> ·
  <a href="README.md">English</a>
</p>

> GitHub 不分发参与者记录。公开模型权重托管于 Hugging Face，下载时会校验文件完整性。

## 项目概览

窗口分类器只能给出局部活动证据，长时记录还需要完整的活动类别、开始时间与结束时间。本项目通过三个部分生成这些片段记录：

1. **多尺度后验模型**：同时分析同一段六通道腕部 IMU 的 3、5、8 秒视图。
2. **局部边界尺度仲裁（LBSA）**：在活动切换附近加强短窗口证据，在稳定区域保留长窗口上下文。
3. **时间记录层（TRL）**：把融合时间线确定性地转换为可进行记录级评估的活动片段。

<p align="center">
  <a href="docs/assets/fig02_overall_framework.png">
    <img src="docs/assets/fig02_overall_framework.png" alt="从腕部 IMU 输入到活动记录的总体框架" width="92%">
  </a>
</p>
<p align="center"><em>三个分尺度模型生成对齐后验，LBSA 完成融合，TRL 构建活动记录。</em></p>

## 主要结果

| 证据 | 数值 |
| --- | --- |
| 传感器数据 | 259.6 小时 |
| 独立外部测试 | 37 条记录 / 114 个片段 |
| 片段级性能 | 平均用户 F1 0.89 / Micro-F1 0.90 |

外部测试采用同类别、一对一、IoU > 0.5 的记录匹配。固定方案对比、分类结果、代表性案例与限制见[结果页](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/zh/research/paper/)。

## 快速验证

```bash
git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
cd Wearable-IMU-Activity-Segmentation-Pipeline
conda env create -f environment.yml
conda activate imu-activity-pipeline
python -m pip install -e .
python tests/smoke_test.py
```

该冒烟测试可在 CPU 上运行，用于验证公开代码包与文件接口，不需要参与者数据或训练权重。

## 数据与模型

| 资源 | 获取方式 |
| --- | --- |
| 参与者记录 | 不存放于 GitHub；按[数据访问说明](data/README_zh.md)申请 |
| PyTorch 与 ONNX 权重 | 公开的 [Hugging Face 模型仓库](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline) |
| 文件哈希与许可证 | [资产说明](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/zh/reference/assets/) |

缺失的公开权重会下载到约定目录，并在使用前依据 `model-assets.json` 校验。

## 文档入口

| 页面 | 内容 |
| --- | --- |
| [方法](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/zh/guide/pipeline/) | 任务、数据、后验模型、LBSA 与 TRL |
| [结果](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/zh/research/paper/) | 独立外部测试证据与失败案例 |
| [补充分析](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/zh/research/supplementary/) | 开发集诊断、可移植性与 Android 证据 |
| [复现](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/zh/reproduce/) | 安装、数据、模型、训练、推理与评估 |
| [演示](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline) | 在浏览器运行合成示例 |

## 引用与许可

正式归档论文引用发布前，请按[引用页](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/zh/reference/citation/)引用 [v0.1.0 研究预览版](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/releases/tag/v0.1.0-research-preview)。仓库原创代码与公开模型资产采用 [Apache-2.0](LICENSE)；数据集和第三方依赖保留各自条款。
