---
title: 浏览器 Demo
description: 在浏览器本地运行真实的多尺度 IMU 活动分段流水线。
hide:
  - toc
---

<section class="imu-demo-hero">
  <p class="imu-demo-kicker">本地优先的交互式 Demo</p>
  <h1>你的信号，你的算力，真实模型结果</h1>
  <p>直接在当前页面运行公开的 3 秒、5 秒和 8 秒 ONNX 模型。六路 IMU 记录不会离开访问者的设备。</p>
  <div class="imu-demo-hero__facts" aria-label="Demo 能力">
    <span>真实 ONNX 推理</span>
    <span>WebGPU → WASM 回退</span>
    <span>时间线 + CSV</span>
    <span>不上传数据</span>
  </div>
</section>

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

## 点击“运行”后会发生什么

<div class="imu-demo-details-grid">
  <article class="imu-demo-detail"><strong>1 · 校验并滤波</strong><p>Worker 检查七列 100 Hz 格式、排序时间戳、移除无效行，并执行仓库中的四阶零相位 Butterworth 滤波。</p></article>
  <article class="imu-demo-detail"><strong>2 · 执行真实模型</strong><p>ONNX Runtime Web 逐一计算 3 秒、5 秒和 8 秒滑动窗口。页面优先尝试 WebGPU，WASM 提供跨平台 CPU 回退。</p></article>
  <article class="imu-demo-detail"><strong>3 · 生成活动记录</strong><p>尺度对齐、融合、平滑、Viterbi 解码、边界细化、时长筛选、置信度筛选和 Top-K 逻辑与项目流水线保持一致。</p></article>
</div>

首次运行会下载约 **17 MB 的 ONNX 权重**和适合当前浏览器的运行时；浏览器允许持久存储时会缓存这些文件。以后再次点击仍会重新执行模型计算，只是不再重复下载相同权重。

模型地址固定到 Hugging Face 修订 `e0f89bb6…`。创建推理会话前，每个权重文件都会根据 [`model-assets.json`](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/model-assets.json) 中发布的 SHA-256 值进行完整性校验。

## 使用自己的记录

请提供 UTF-8 编码、制表符分隔的文本，并包含以下精确列名：

```text
ACC_TIME  ACC_X  ACC_Y  ACC_Z  GYRO_X  GYRO_Y  GYRO_Z
```

`ACC_TIME` 必须是唯一的毫秒时间戳，采样间隔中位数需为 8–12 毫秒。多余列和无效行会被忽略。公开 Demo 接受 800–60,000 个有效样本，即 100 Hz 下约 8 秒到 10 分钟。

!!! warning "研究用途输出"

    通过格式检查并不代表数据符合训练协议。传感器佩戴位置、坐标轴方向、单位、设备特性和预处理方式都应与文档保持一致。识别结果不能用作医疗、安全或训练指导。

如需与服务器端版本对照，原 [Hugging Face Space](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline) 仍可访问；当前页面则是保护隐私、使用访问者算力的版本。
