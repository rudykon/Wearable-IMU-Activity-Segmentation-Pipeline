<section class="demo-page-hero">
  <div>
    <p class="hero-kicker">交互式浏览器演示</p>
    <h1>从腕部 IMU 信号到活动记录</h1>
    <p>使用内置合成记录或兼容的 100 Hz 腕部 IMU 文件，运行公开的 3、5、8 秒模型。</p>
    <div class="demo-facts" aria-label="演示能力">
      <span>真实公开模型</span>
      <span>六路传感器信号</span>
      <span>时间线 + CSV</span>
    </div>
    <div class="demo-actions">
      <a class="demo-action primary" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">打开在线演示</a>
      <a class="demo-action github" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.22c-3.23.7-3.91-1.37-3.91-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.72 1.27 3.39.97.1-.75.4-1.27.74-1.56-2.58-.29-5.29-1.29-5.29-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.48.11-3.05 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.19-1.49 3.16-1.18 3.16-1.18.63 1.57.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.4-2.72 5.38-5.31 5.67.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .7Z"/></svg>
        查看 Demo 源码
      </a>
      <a class="demo-action" href="https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">模型权重</a>
    </div>
  </div>
  <a class="demo-page-image" href="../../../assets/demo-results.jpg" target="_blank" rel="noopener" aria-label="打开完整分辨率的演示截图">
    <img src="../../../assets/demo-results.jpg" alt="真实 Hugging Face 演示输出，展示合成 IMU 输入、模型控制项和带时间戳的活动记录" loading="eager" decoding="async">
    <span>内置样例真实输出</span>
  </a>
</section>

## 演示展示什么

界面保留完整推理过程，而不是只返回一个活动标签：

- 六路加速度计和陀螺仪信号图；
- 类别概率与解码后的活动时间线；
- 每条记录的活动、开始、结束、持续时间和置信度；
- 可下载的 CSV 文件。

内置 120 秒样例由程序生成，不含参与者数据，也不属于验证集。Demo 控制项用于交互体验；论文结果使用固定评估脚本。

## 输入格式

上传 UTF-8 制表符分隔的 `.txt` 或 `.tsv` 文件，时间戳单位为毫秒，并包含：

~~~text
ACC_TIME  ACC_X  ACC_Y  ACC_Z  GYRO_X  GYRO_Y  GYRO_Z
~~~

时间戳必须递增，中位间隔为 8–12 ms。公开界面最多接收 60,000 个样本，即 100 Hz 下 10 分钟。设备佩戴位置、坐标轴、单位和预处理方式应与文档协议一致。

## 隐私

!!! warning "请勿上传敏感记录"

    请勿向公开 Space 发送机密或可识别的参与者数据。预测只用于研究，不构成医疗、安全或训练建议。

## 本地运行

安装、模型校验下载、数据边界与本地命令统一维护在[复现页](../reproduce.md)。完整 Gradio 实现位于 GitHub 仓库的 [`demo/`](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo) 目录。
