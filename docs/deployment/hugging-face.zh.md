# Hugging Face Space 在线演示

浏览器 Demo 是了解项目最直接的入口。选择内置样例或上传兼容的腕部运动记录后，
仓库中的 3 秒、5 秒和 8 秒真实模型会生成带时间戳的活动时间线。模型组合和时间线
整理步骤与 Python 研究流程一致。

[打开 Hugging Face Space](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline){ .md-button .md-button--primary target="_blank" rel="noopener" }
[查看 Demo 源码](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo){ .md-button target="_blank" rel="noopener" }

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="../../../assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的总体框架图">
    <img src="../../../assets/fig02_overall_framework.png" alt="项目总体框架，展示 IMU 输入、分尺度 CNN–BiLSTM、LBSA 融合、时序记录层和活动片段记录" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Space 遵循仓库原始流程：六路 IMU 信号 → 三种窗口模型 → 尺度选择 → 时间线整理 → 活动记录。</figcaption>
</figure>

## 可以查看和下载什么

一次请求会生成：

- 六路上传 IMU 信号的预览；
- 背景与五类已支持活动的可能性曲线；
- 清理短暂预测跳变后的最终活动时间线；
- 包含开始时间、结束时间、持续时间和置信度的中英文记录表；
- 包含绝对毫秒时间戳的 UTF-8 CSV 下载文件。

ZeroGPU Space 启动时会注册模型，每次完整模型推理使用一次最长 30 秒的 GPU 配额。
推理采用单任务串行方式，以保持内存占用可预测并避免请求竞争。公开界面最多接收
60,000 个样本，相当于 100 Hz 下的十分钟记录；访客消耗自己的 Hugging Face
ZeroGPU 配额。

## 使用内置示例

选择 **Bundled synthetic example / 内置合成示例**，然后点击
**Run segmentation / 开始分割**。该 120 秒文件由确定性的静止与周期运动阶段组成，
用于贯通完整流水线，不含任何受试者记录；它既不是验证集样本，也不代表生物学真实性。

演示默认控制值有意比论文最终的长时报告策略更宽松：

| 控制项 | 演示默认值 | 用途 |
| --- | ---: | --- |
| 三个模型怎样组合 | `local_boundary` | 在可能发生活动切换的位置自适应调整模型权重 |
| 最短持续时间 | 5 秒 | 让较短的合成阶段可见 |
| 最低置信度 | 0.30 | 避免短时演示隐藏全部输出 |
| Top-K | 5 | 限制结果表长度 |

复现论文时，应使用仓库中固定的评估脚本。Demo 中可调整的选项只用于体验，不能当作
论文报告设置。

## 上传格式

上传 UTF-8 制表符分隔的 `.txt` 或 `.tsv` 文件，时间戳单位为毫秒，且至少包含：

```text
ACC_TIME	ACC_X	ACC_Y	ACC_Z	GYRO_X	GYRO_Y	GYRO_Z
```

演示会检查时间戳严格递增，且中位采样间隔位于 8–12 ms。额外源数据列会被忽略。
比较预测前，请确认传感器佩戴位置、坐标轴方向、物理单位和预处理均符合文档中的输入格式。

## 本地运行

```bash
git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
cd Wearable-IMU-Activity-Segmentation-Pipeline
python -m pip install -r requirements.txt
python -m pip install spaces
python -m pip install -e .
python demo/app.py
```

Space 还会把同一 Gradio 函数公开为名为 `/segment` 的 API 端点。

## 隐私与局限

!!! warning "请勿上传敏感受试者记录"

    公开 Space 属于共享托管基础设施。应用不会主动持久化上传文件，但机密或可识别的
    记录应始终保留在经授权的本地环境中。

预测仅用于研究展示，不构成医疗、安全或训练建议。论文尚未证明跨设备或跨人群泛化，
内置合成文件也不能代替在经授权记录上的正式评估。
