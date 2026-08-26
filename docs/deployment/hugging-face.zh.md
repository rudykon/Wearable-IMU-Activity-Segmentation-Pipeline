<section class="demo-page-hero">
  <div>
    <p class="hero-kicker">交互式浏览器演示</p>
    <h1>从腕部 IMU 信号到活动记录</h1>
    <p>页面默认载入 <code>synthetic_activity_imu.tsv</code>，可以直接运行，也可以替换为兼容的 100 Hz 腕部 IMU 文件。</p>
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

<nav class="demo-page-nav" aria-label="Demo 使用指南目录">
  <a href="#run-the-bundled-example">运行内置样例</a>
  <a href="#raw-signals">原始信号</a>
  <a href="#activity-likelihood-and-timeline">概率与时间线</a>
  <a href="#activity-records">活动记录</a>
  <a href="#use-your-own-recording">上传自己的文件</a>
</nav>

## 运行内置合成样例 {#run-the-bundled-example}

理解界面最快的方法，是直接运行确定性生成的 [`synthetic_activity_imu.tsv`](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/demo/examples/synthetic_activity_imu.tsv)。该文件包含 **12,000 个样本**，对应 **100 Hz 下的 120 秒记录**，不含任何参与者数据。在线 Space 打开后，该文件已经默认选中。

<div class="demo-steps">
  <article class="demo-step">
    <span class="demo-step__number">1</span>
    <h3>打开在线 Demo</h3>
    <p>进入 Hugging Face Space，等待 Gradio 界面加载完成。</p>
    <a href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">打开在线 Demo ↗</a>
  </article>
  <article class="demo-step">
    <span class="demo-step__number">2</span>
    <h3>确认默认样例</h3>
    <p>文件输入框已经载入 <code>synthetic_activity_imu.tsv</code>，不需要上传文件，也不需要再点击示例行。</p>
  </article>
  <article class="demo-step">
    <span class="demo-step__number">3</span>
    <h3>保留默认参数</h3>
    <p>融合方式使用 <code>local_boundary</code>，最小时长 <code>5 秒</code>，置信度 <code>0.30</code>，Top-K 为 <code>5</code>。</p>
  </article>
  <article class="demo-step">
    <span class="demo-step__number">4</span>
    <h3>运行当前样例</h3>
    <p>点击 <strong>Run the loaded sample / 运行当前样例</strong>，然后依次查看三个结果标签页。</p>
  </article>
</div>

<div class="demo-defaults" aria-label="可复现示例所用设置">
  <div><span>输入文件</span><strong>synthetic_activity_imu.tsv</strong></div>
  <div><span>融合方式</span><strong>local_boundary</strong></div>
  <div><span>最小时长</span><strong>5 秒</strong></div>
  <div><span>最低置信度</span><strong>0.30</strong></div>
  <div><span>Top-K</span><strong>5</strong></div>
</div>

!!! tip "点击运行后，系统会做什么"

    Space 会先检查默认载入的文件，再载入公开模型权重，分别运行 3、5、8 秒模型，融合三条后验概率轨迹，经过时间记录层处理，最后在一次请求中返回信号图、活动时间线、记录表和 CSV。

## 可复现的示例输出 {#reproducible-example-output}

下面三张图由**当前公开模型权重**和上述默认参数真实生成。本次运行得到 118 个时间线点，以及两条活动记录。

<div class="demo-run-summary" aria-label="合成样例运行摘要">
  <div class="demo-run-stat"><strong>12,000</strong><span>输入样本</span></div>
  <div class="demo-run-stat"><strong>120.0 秒</strong><span>记录时长</span></div>
  <div class="demo-run-stat"><strong>118</strong><span>时间线点</span></div>
  <div class="demo-run-stat"><strong>2</strong><span>活动记录</span></div>
</div>

!!! note "这是流程演示，不是准确率验证"

    该样例是合成信号，用于展示从输入到输出的完整路径。识别类别与边界是模型输出，不是带有真实标签的准确率证据。修改融合方式、时长、置信度或 Top-K 后，结果可能发生变化。

### 1. Raw signals / 原始信号 {#raw-signals}

打开第一个结果标签页，可以在分类前检查六路信号。上图是三轴加速度，下图是三轴角速度。

<figure class="demo-result-figure">
  <a href="../../../assets/demo/synthetic-raw-signals.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的原始信号结果图">
    <img src="../../../assets/demo/synthetic-raw-signals.png" alt="由 synthetic_activity_imu.tsv 生成的三轴加速度和三轴角速度原始信号图" loading="lazy" decoding="async">
  </a>
  <figcaption><strong>Raw signals / 原始信号。</strong>内置 120 秒合成样例的真实 Demo 输出。</figcaption>
</figure>

<div class="demo-reading-grid">
  <div class="demo-reading">
    <strong>怎样看这张图</strong>
    <p>开头、两段运动之间以及末尾是相对平静区间；两块幅度较大的周期振荡，对应合成样例中的两段高运动信号。</p>
  </div>
  <div class="demo-reading">
    <strong>为什么先看原始信号</strong>
    <p>它可以帮助检查通道缺失、数值截断、异常偏置、采样率问题，以及上传文件中是否确实存在可见运动。</p>
  </div>
</div>

### 2. Activity likelihood and timeline / 活动概率与时间线 {#activity-likelihood-and-timeline}

第二个标签页包含两类相关结果。上半部分是各活动类别随时间变化的平滑概率；下半部分是多尺度融合和时间后处理后的最终解码状态。

<figure class="demo-result-figure">
  <a href="../../../assets/demo/synthetic-activity-likelihood-timeline.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的活动概率与时间线结果图">
    <img src="../../../assets/demo/synthetic-activity-likelihood-timeline.png" alt="由 synthetic_activity_imu.tsv 生成的类别概率曲线和最终活动时间线" loading="lazy" decoding="async">
  </a>
  <figcaption><strong>Activity likelihood and timeline / 活动概率与时间线。</strong>上方为类别后验轨迹，下方为最终解码活动。</figcaption>
</figure>

<div class="demo-reading-grid">
  <div class="demo-reading">
    <strong>上半部分</strong>
    <p>在较平静区间，Background 概率最高；第一段运动使 Fly（飞鸟）概率升高，第二段运动中 Running（跑步）成为最高的前景活动概率。</p>
  </div>
  <div class="demo-reading">
    <strong>下半部分</strong>
    <p>解码时间线抑制了快速标签抖动，最终形成“背景—飞鸟—背景—跑步—背景”的稳定状态序列。</p>
  </div>
</div>

### 3. Activity records / 活动记录 {#activity-records}

第三个标签页把解码时间线转换为真正可使用的输出：每个活动区间占一行，并给出活动类别、开始时间、结束时间、持续时间和置信度。

<figure class="demo-result-figure demo-result-figure--records">
  <a href="../../../assets/demo/synthetic-activity-records.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的活动记录结果图">
    <img src="../../../assets/demo/synthetic-activity-records.png" alt="由 synthetic_activity_imu.tsv 生成的活动记录表" loading="lazy" decoding="async">
  </a>
  <figcaption><strong>Activity records / 活动记录。</strong>默认 Demo 参数返回的两条记录。</figcaption>
</figure>

| 活动 | 开始（秒） | 结束（秒） | 持续时间（秒） | 置信度 |
| --- | ---: | ---: | ---: | ---: |
| Fly / 飞鸟 | 29.84 | 73.15 | 43.31 | 0.4038 |
| Running / 跑步 | 76.06 | 98.24 | 22.18 | 0.3186 |

这些边界不是从合成样例生成脚本中直接复制的，而是由模型概率、Viterbi 解码、间隔处理、边界修正、时长过滤和置信度过滤共同产生。

<div class="demo-download-row">
  <a class="demo-action primary" href="../../../assets/demo/synthetic-activity-records.csv" download>下载本次结果 CSV</a>
  <a class="demo-action" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/demo/examples/synthetic_activity_imu.tsv" target="_blank" rel="noopener">查看样例文件</a>
  <a class="demo-action github" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo" target="_blank" rel="noopener">浏览 Demo 源码</a>
</div>

## 各项参数会改变什么

| 参数 | 示例设置 | 作用 |
| --- | --- | --- |
| 多尺度融合 | `local_boundary` | 改变 3、5、8 秒后验轨迹的组合方式，尤其影响可能发生活动切换的位置。 |
| 最小时长 | `5 秒` | 删除更短的解码记录。提高该值会偏向保留长活动，降低该值则会保留更短事件。 |
| 最低置信度 | `0.30` | 删除较弱的活动记录。提高阈值可能减少误报，也可能删除真实但置信度较低的活动。 |
| Top-K | `5` | 限制时间处理后最多返回多少条记录；设为 `0` 时不限制数量。 |

上述较短参数是为了便于观察 120 秒样例，属于 **Demo 体验参数**，不是论文实验中面向分钟级活动所采用的固定设置。

## 上传自己的记录 {#use-your-own-recording}

1. 准备 UTF-8 制表符分隔的 `.txt` 或 `.tsv` 文件。
2. 在文件输入控件中，用自己的记录替换默认载入的合成样例。
3. 第一次运行时先保留默认参数，并优先检查 **Raw signals / 原始信号**，再解释模型结果。
4. 后续每次只调整一个参数，便于看清该参数带来的变化。

必需列名：

~~~text
ACC_TIME  ACC_X  ACC_Y  ACC_Z  GYRO_X  GYRO_Y  GYRO_Z
~~~

`ACC_TIME` 必须是严格递增的毫秒时间戳，中位采样间隔需为 8–12 ms，对应约 100 Hz。公开界面接收 **800–60,000 个有效样本**，即 100 Hz 下约 8 秒至 10 分钟；额外列会被忽略。

!!! warning "文件格式正确，不代表模型一定适用"

    传感器佩戴位置、坐标轴方向、数值单位、设备特性和预处理方式都应与文档协议匹配。新设备、新人群、新佩戴位置和新活动类别都需要重新验证。

## 隐私

!!! warning "请勿上传敏感记录"

    请勿向公开 Space 发送机密或可识别的参与者数据。预测只用于研究，不构成医疗、安全或训练建议。

## 本地运行

安装、模型校验下载、数据边界与本地命令统一维护在[复现页](../reproduce.md)。完整 Gradio 实现位于 GitHub 仓库的 [`demo/`](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo) 目录。
