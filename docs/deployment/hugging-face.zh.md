# 在线演示

在浏览器运行 3、5、8 秒模型。可选内置样例，也可上传兼容的腕部 IMU 文件。

[打开 Hugging Face Space](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline){ .md-button .md-button--primary target="_blank" rel="noopener" }
[打开模型权重](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline){ .md-button target="_blank" rel="noopener" }
[查看 Demo 源码](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo){ .md-button target="_blank" rel="noopener" }

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="../../../assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的总体框架图">
    <img src="../../../assets/fig02_overall_framework.png" alt="项目总体框架，展示 IMU 输入、分尺度 CNN–BiLSTM、LBSA 融合、时序记录层和活动片段记录" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">六路信号 → 三个模型 → 融合 → 活动记录。</figcaption>
</figure>

## 输出

- 六路 IMU 信号；
- 类别概率；
- 解码时间线；
- 活动表；
- CSV 下载。

上限为 60,000 个样本，即 100 Hz 下 10 分钟。每次运行使用一次 ZeroGPU 请求。

## 样例

选择 **Bundled synthetic example / 内置合成示例**，再点击
**Run segmentation / 开始分割**。120 秒样例为合成数据，不含参与者记录，也不是验证数据。

| 控制项 | 演示默认值 | 用途 |
| --- | ---: | --- |
| 三个模型怎样组合 | `local_boundary` | 在可能发生活动切换的位置自适应调整模型权重 |
| 最短持续时间 | 5 秒 | 让较短的合成阶段可见 |
| 最低置信度 | 0.30 | 避免短时演示隐藏全部输出 |
| Top-K | 5 | 限制结果表长度 |

Demo 选项只用于体验；复现论文请使用固定评估脚本。

## 上传格式

上传 UTF-8 制表符分隔的 `.txt` 或 `.tsv` 文件，时间戳单位为毫秒，且至少包含：

```text
ACC_TIME	ACC_X	ACC_Y	ACC_Z	GYRO_X	GYRO_Y	GYRO_Z
```

时间戳必须递增，中位间隔必须为 8–12 ms。佩戴位置、坐标轴、单位和预处理需与文档一致。

## 本地运行

```bash
git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
cd Wearable-IMU-Activity-Segmentation-Pipeline
python -m pip install -r requirements.txt
python -m pip install spaces
python -m pip install -e .
python demo/app.py
```

缺少的检查点会从公开 HF Model 仓库自动下载。

API 端点：`/segment`。

## 隐私

!!! warning "请勿上传敏感受试者记录"

    请勿向公开 Space 上传机密或可识别的记录。

预测只用于研究，不构成医疗、安全或训练建议。跨设备和跨人群表现尚未证明。
