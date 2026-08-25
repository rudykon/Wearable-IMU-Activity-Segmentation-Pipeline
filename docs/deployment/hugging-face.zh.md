# 在线演示

在浏览器运行公开的 3、5、8 秒模型。可选择内置合成样例，也可上传兼容的 100 Hz 腕部 IMU 文件。

[打开 Hugging Face Space](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline){ .md-button .md-button--primary target="_blank" rel="noopener" }
[打开模型权重](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline){ .md-button target="_blank" rel="noopener" }

<figure class="paper-figure demo-figure">
  <a class="pipeline-image-link" href="../../../assets/demo-results.jpg" target="_blank" rel="noopener" aria-label="打开完整分辨率的演示截图">
    <img src="../../../assets/demo-results.jpg" alt="Hugging Face 演示界面，展示合成 IMU 输入、模型控制项和两条带时间戳的活动记录" loading="eager" decoding="async">
  </a>
  <figcaption class="pipeline-caption">内置合成记录的真实演示输出；画面不含参与者数据。</figcaption>
</figure>

## 输入

上传 UTF-8 制表符分隔的 `.txt` 或 `.tsv` 文件，时间戳单位为毫秒，并包含：

~~~text
ACC_TIME  ACC_X  ACC_Y  ACC_Z  GYRO_X  GYRO_Y  GYRO_Z
~~~

时间戳必须递增，中位间隔为 8–12 ms。公开界面最多接收 60,000 个样本，即 100 Hz 下 10 分钟。

## 输出

- 六路 IMU 信号图；
- 类别概率与解码时间线；
- 每条记录的活动、开始、结束、持续时间和置信度；
- 可下载 CSV。

内置 120 秒样例由程序生成，不属于验证数据。Demo 控制项只用于体验；研究结果使用固定评估脚本。

## 隐私

!!! warning "请勿上传敏感记录"

    请勿向公开 Space 发送机密或可识别的参与者数据。预测只用于研究，不构成医疗、安全或训练建议。

## 本地复现

安装、模型下载、数据边界与本地命令统一见[复现页](../reproduce.md)。界面源码位于 [`demo/`](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo)。
