---
title: 浏览器 Demo
description: 在浏览器本地运行真实的多尺度 IMU 活动分段流水线。
hide:
  - toc
---

<section class="demo-page-hero imu-browser-page-hero">
  <div>
    <p class="hero-kicker">访问者设备上的真实推理</p>
    <h1>从腕部 IMU 信号到活动记录</h1>
    <p>直接运行内置的 120 秒合成样例，或换成兼容的 100 Hz 腕部 IMU 记录。无需安装，也不会上传传感器数据。</p>
    <div class="demo-facts" aria-label="浏览器 Demo 能力">
      <span>真实公开模型</span>
      <span>WebGPU / WASM</span>
      <span>时间线 + CSV</span>
      <span>数据留在本机</span>
    </div>
    <div class="demo-actions">
      <a class="demo-action primary" href="#run-browser-demo">开始运行</a>
      <a class="demo-action github" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/docs/assets/javascripts/imu-demo-worker.mjs" target="_blank" rel="noopener">查看浏览器源码</a>
      <a class="demo-action" href="https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">模型权重</a>
    </div>
  </div>
  <a class="demo-page-image" href="../../assets/demo/synthetic-activity-likelihood-timeline.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的内置样例参考输出">
    <img src="../../assets/demo/synthetic-activity-likelihood-timeline.png" alt="内置合成样例的活动概率曲线与最终活动时间线" loading="eager" decoding="async">
    <span>内置样例参考输出</span>
  </a>
</section>

<nav class="demo-page-nav" aria-label="浏览器 Demo 页面目录">
  <a href="#run-browser-demo">运行 Demo</a>
  <a href="#quick-start">操作步骤</a>
  <a href="#expected-output">参考结果</a>
  <a href="#understand-results">读懂结果</a>
  <a href="#parameters">参数说明</a>
  <a href="#use-your-own-recording">上传记录</a>
</nav>

## 直接运行 {#run-browser-demo}

内置样例已经就绪。保留默认参数并点击**运行当前记录**，页面会在当前设备上执行完整的三尺度推理。

<div id="imu-browser-demo" class="imu-demo-shell" data-locale="zh">
  <section class="imu-demo-card" aria-labelledby="imu-demo-input-title-zh">
    <div class="imu-demo-input-grid">
      <div>
        <span class="imu-demo-section-label" id="imu-demo-input-title-zh">1 · 传感器记录</span>
        <div class="imu-demo-drop" data-demo-id="dropZone">
          <input id="imu-demo-file-zh" data-demo-id="file" type="file" accept=".txt,.tsv,text/tab-separated-values,text/plain">
          <label for="imu-demo-file-zh">
            <span class="imu-demo-drop__icon" aria-hidden="true">↥</span>
            <strong>使用内置样例或选择 TSV 文件</strong>
            <small>拖入 UTF-8 文件 · 800–60,000 个样本 · 最大 20 MB</small>
            <span class="imu-demo-file-name" data-demo-id="fileName"></span>
          </label>
        </div>
      </div>

      <div>
        <span class="imu-demo-section-label">2 · 流水线参数</span>
        <div class="imu-demo-controls">
          <div class="imu-demo-control imu-demo-control--wide">
            <label for="imu-demo-fusion-zh">模型融合</label>
            <select id="imu-demo-fusion-zh" data-demo-id="fusion">
              <option value="local_boundary">局部边界（默认）</option>
              <option value="average">平均融合</option>
              <option value="dynamic_boundary">动态边界</option>
              <option value="confident_conflict">高置信冲突</option>
              <option value="weighted_long">偏向长窗口</option>
              <option value="weighted_balanced">均衡权重</option>
            </select>
          </div>
          <div class="imu-demo-control">
            <label for="imu-demo-duration-zh">最小时长 <output data-demo-id="durationOutput">5</output> 秒</label>
            <input id="imu-demo-duration-zh" data-demo-id="duration" type="range" min="1" max="180" step="1" value="5">
          </div>
          <div class="imu-demo-control">
            <label for="imu-demo-confidence-zh">最低置信度 <output data-demo-id="confidenceOutput">0.30</output></label>
            <input id="imu-demo-confidence-zh" data-demo-id="confidence" type="range" min="0" max="1" step="0.05" value="0.30">
          </div>
          <div class="imu-demo-control imu-demo-control--wide">
            <label for="imu-demo-top-k-zh">Top-K 记录 <output data-demo-id="topKOutput">5</output></label>
            <input id="imu-demo-top-k-zh" data-demo-id="topK" type="range" min="0" max="10" step="1" value="5">
          </div>
        </div>
        <div class="imu-demo-actions">
          <button class="imu-demo-button imu-demo-button--primary" data-demo-id="run" type="button">运行当前记录</button>
          <button class="imu-demo-button" data-demo-id="reset" type="button">恢复内置样例</button>
        </div>
      </div>
    </div>
    <p class="imu-demo-privacy"><span aria-hidden="true">⌁</span><span><strong>本地计算。</strong>页面只从 Hugging Face 下载经过校验的公开模型；IMU 样本的解析、滤波、推理、绘图和 CSV 导出全部在当前设备的 Web Worker 中完成。</span></p>
  </section>

  <section class="imu-demo-status" data-demo-id="status" data-kind="ready" aria-live="polite" aria-atomic="true">
    <span class="imu-demo-status__dot" aria-hidden="true"></span>
    <strong data-demo-id="statusTitle"></strong>
    <span data-demo-id="statusDetail"></span>
    <progress data-demo-id="progress" max="1" hidden></progress>
  </section>

  <section class="imu-demo-results" data-demo-id="results" hidden aria-label="推理结果">
    <div class="imu-demo-summary">
      <div class="imu-demo-stat"><strong data-demo-id="samples">—</strong><span>输入样本</span></div>
      <div class="imu-demo-stat"><strong data-demo-id="recordingDuration">—</strong><span>记录时长</span></div>
      <div class="imu-demo-stat"><strong data-demo-id="points">—</strong><span>时间线点</span></div>
      <div class="imu-demo-stat"><strong data-demo-id="records">—</strong><span>活动记录</span></div>
    </div>
    <div class="imu-demo-runtime" aria-label="本地运行信息">
      <span data-demo-id="backend"></span>
      <span data-demo-id="cache"></span>
      <span>ONNX · 3 秒 + 5 秒 + 8 秒</span>
    </div>

    <div class="imu-demo-tabs">
      <div class="imu-demo-tabs__list" role="tablist" aria-label="结果视图">
        <button class="imu-demo-tab" type="button" role="tab" data-demo-tab="signal" aria-selected="true">原始信号</button>
        <button class="imu-demo-tab" type="button" role="tab" data-demo-tab="timeline" aria-selected="false" tabindex="-1">概率 + 时间线</button>
        <button class="imu-demo-tab" type="button" role="tab" data-demo-tab="records" aria-selected="false" tabindex="-1">活动记录</button>
      </div>
      <div class="imu-demo-panel" role="tabpanel" data-demo-panel="signal">
        <canvas data-demo-id="signalCanvas" role="img" aria-label="原始加速度与角速度信号图"></canvas>
        <p class="imu-demo-chart-note">图表只为显示而降采样；模型推理仍使用全部有效样本。</p>
      </div>
      <div class="imu-demo-panel" role="tabpanel" data-demo-panel="timeline" hidden>
        <canvas data-demo-id="timelineCanvas" role="img" aria-label="平滑活动概率和最终解码时间线"></canvas>
        <p class="imu-demo-chart-note">上图是融合并平滑后的后验概率，下方色带是 Viterbi 解码状态。</p>
      </div>
      <div class="imu-demo-panel" role="tabpanel" data-demo-panel="records" hidden>
        <div class="imu-demo-table-wrap">
          <table class="imu-demo-table">
            <thead data-demo-id="tableHead"></thead>
            <tbody data-demo-id="tableBody"></tbody>
          </table>
        </div>
        <div class="imu-demo-download">
          <a class="imu-demo-button imu-demo-button--primary" data-demo-id="download" download>下载结果 CSV</a>
        </div>
      </div>
    </div>
  </section>
</div>

<noscript>
该 Demo 需要 JavaScript，因为完整推理流水线会在浏览器内执行。
</noscript>

## 三步完成一次推理 {#quick-start}

<div class="demo-steps imu-browser-steps">
  <article class="demo-step">
    <span class="demo-step__number">1</span>
    <h3>确认内置样例</h3>
    <p><code>synthetic_activity_imu.tsv</code> 已经载入，包含 12,000 个样本和 120 秒六路 IMU 信号。</p>
  </article>
  <article class="demo-step">
    <span class="demo-step__number">2</span>
    <h3>保留默认参数</h3>
    <p>先使用局部边界融合、5 秒最小时长、0.30 最低置信度和 Top-K 5。</p>
  </article>
  <article class="demo-step">
    <span class="demo-step__number">3</span>
    <h3>运行并查看结果</h3>
    <p>点击<strong>运行当前记录</strong>，完成后依次查看原始信号、概率时间线和活动记录。</p>
  </article>
</div>

<div class="demo-defaults" aria-label="内置样例默认设置">
  <div><span>输入文件</span><strong>synthetic_activity_imu.tsv</strong></div>
  <div><span>融合方式</span><strong>local_boundary</strong></div>
  <div><span>最小时长</span><strong>5 秒</strong></div>
  <div><span>最低置信度</span><strong>0.30</strong></div>
  <div><span>Top-K</span><strong>5</strong></div>
</div>

!!! tip "点击运行后"

    Worker 会校验记录并执行零相位 Butterworth 滤波，再运行 3、5、8 秒三份真实 ONNX 模型，最后完成尺度融合、Viterbi 解码、边界修正和记录筛选。

## 内置样例参考结果 {#expected-output}

使用上述默认参数时，公开模型会生成 118 个时间线点和两条活动记录。这些数值可用于确认浏览器端流水线是否正常工作。

<div class="demo-run-summary" aria-label="内置样例参考结果摘要">
  <div class="demo-run-stat"><strong>12,000</strong><span>输入样本</span></div>
  <div class="demo-run-stat"><strong>120.0 秒</strong><span>记录时长</span></div>
  <div class="demo-run-stat"><strong>118</strong><span>时间线点</span></div>
  <div class="demo-run-stat"><strong>2</strong><span>活动记录</span></div>
</div>

| 活动 | 开始（秒） | 结束（秒） | 持续时间（秒） | 置信度 |
| --- | ---: | ---: | ---: | ---: |
| 飞鸟 | 29.84 | 73.15 | 43.31 | 0.4038 |
| 跑步 | 76.06 | 98.24 | 22.18 | 0.3186 |

!!! note "流程演示，不是准确率验证"

    内置文件是确定性生成的合成信号，不含参与者数据。表中的类别和边界来自模型推理，不是真实标签或准确率证据。

## 读懂三个结果标签页 {#understand-results}

<div class="imu-demo-details-grid">
  <article class="imu-demo-detail"><strong>原始信号</strong><p>先检查三轴加速度与三轴角速度，确认通道完整、采样稳定，并观察记录中是否存在明显运动。</p></article>
  <article class="imu-demo-detail"><strong>概率 + 时间线</strong><p>上图显示六类活动的平滑概率；下方色带显示多尺度融合和 Viterbi 解码后的最终状态。</p></article>
  <article class="imu-demo-detail"><strong>活动记录</strong><p>每行给出活动类别、开始时间、结束时间、持续时间和置信度，可直接下载为 CSV。</p></article>
</div>

## 参数如何影响结果 {#parameters}

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| 多尺度融合 | `local_boundary` | 决定 3、5、8 秒后验概率的组合方式，主要影响活动切换附近。 |
| 最小时长 | `5 秒` | 删除更短的活动区间；数值越大，越偏向保留长活动。 |
| 最低置信度 | `0.30` | 删除较弱的活动记录；提高阈值可能同时减少误报和召回。 |
| Top-K | `5` | 限制最多返回多少条记录；设为 `0` 时不限制。 |

这些是便于观察 120 秒样例的 **Demo 参数**，不是论文实验中固定不变的活动定义。

## 上传自己的记录 {#use-your-own-recording}

1. 准备 UTF-8 编码、制表符分隔的 `.txt` 或 `.tsv` 文件。
2. 选择或拖入文件，第一次运行时保留默认参数。
3. 先检查**原始信号**，再解释模型输出；后续每次只调整一个参数。

必需列名：

```text
ACC_TIME  ACC_X  ACC_Y  ACC_Z  GYRO_X  GYRO_Y  GYRO_Z
```

`ACC_TIME` 必须是严格递增的毫秒时间戳，采样间隔中位数需为 8–12 毫秒。多余列和无效行会被忽略。公开 Demo 接受 800–60,000 个有效样本，即 100 Hz 下约 8 秒到 10 分钟。

## 本地计算与隐私 {#local-compute}

<div class="demo-reading-grid">
  <div class="demo-reading"><strong>浏览器下载什么</strong><p>首次运行会下载约 17 MB 的 ONNX 权重和约 26 MB 的运行时。文件经过 SHA-256 校验，并在浏览器允许时缓存。</p></div>
  <div class="demo-reading"><strong>数据去了哪里</strong><p>解析、滤波、模型推理、绘图和 CSV 导出都在当前设备的 Web Worker 中完成；IMU 记录不会上传。</p></div>
</div>

模型固定到 Hugging Face 修订 `e0f89bb6…`，校验值发布在 [`model-assets.json`](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/model-assets.json)。支持 WebGPU 时优先使用 GPU；否则自动切换到 WASM CPU，因此不同电脑和手机的耗时会不同。

!!! warning "研究用途输出"

    文件格式正确不代表数据符合训练协议。传感器佩戴位置、坐标轴方向、单位、设备特性和预处理方式都应匹配；识别结果不能用作医疗、安全或训练指导。

需要服务器端对照时，可使用原 [Hugging Face Space](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline)；当前页面是使用访问者算力、数据留在本机的版本。
