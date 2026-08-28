---
title: Browser Demo
description: Run the real multi-scale IMU activity segmentation pipeline locally in your browser.
hide:
  - toc
---

<section class="imu-demo-hero">
  <p class="imu-demo-kicker">Local-first interactive demo</p>
  <h1>Your signals. Your processor. Real model output.</h1>
  <p>Run the public 3-, 5-, and 8-second ONNX checkpoints directly in this page. The six-channel recording never leaves your device.</p>
  <div class="imu-demo-hero__facts" aria-label="Demo capabilities">
    <span>Real ONNX inference</span>
    <span>WebGPU → WASM fallback</span>
    <span>Timeline + CSV</span>
    <span>No upload</span>
  </div>
</section>

<div id="imu-browser-demo" class="imu-demo-shell" data-locale="en">
  <section class="imu-demo-card" aria-labelledby="imu-demo-input-title">
    <div class="imu-demo-input-grid">
      <div>
        <span class="imu-demo-section-label" id="imu-demo-input-title">1 · Recording</span>
        <div class="imu-demo-drop" data-demo-id="dropZone">
          <input id="imu-demo-file-en" data-demo-id="file" type="file" accept=".txt,.tsv,text/tab-separated-values,text/plain">
          <label for="imu-demo-file-en">
            <span class="imu-demo-drop__icon" aria-hidden="true">↥</span>
            <strong>Use the bundled sample or choose a TSV</strong>
            <small>Drop a UTF-8 file here · 800–60,000 samples · up to 20 MB</small>
            <span class="imu-demo-file-name" data-demo-id="fileName"></span>
          </label>
        </div>
      </div>

      <div>
        <span class="imu-demo-section-label">2 · Pipeline settings</span>
        <div class="imu-demo-controls">
          <div class="imu-demo-control imu-demo-control--wide">
            <label for="imu-demo-fusion-en">Model fusion</label>
            <select id="imu-demo-fusion-en" data-demo-id="fusion">
              <option value="local_boundary">Local boundary (default)</option>
              <option value="average">Average</option>
              <option value="dynamic_boundary">Dynamic boundary</option>
              <option value="confident_conflict">Confident conflict</option>
              <option value="weighted_long">Weighted toward long windows</option>
              <option value="weighted_balanced">Balanced weights</option>
            </select>
          </div>
          <div class="imu-demo-control">
            <label for="imu-demo-duration-en">Minimum duration <output data-demo-id="durationOutput">5</output> s</label>
            <input id="imu-demo-duration-en" data-demo-id="duration" type="range" min="1" max="180" step="1" value="5">
          </div>
          <div class="imu-demo-control">
            <label for="imu-demo-confidence-en">Confidence <output data-demo-id="confidenceOutput">0.30</output></label>
            <input id="imu-demo-confidence-en" data-demo-id="confidence" type="range" min="0" max="1" step="0.05" value="0.30">
          </div>
          <div class="imu-demo-control imu-demo-control--wide">
            <label for="imu-demo-top-k-en">Top-K records <output data-demo-id="topKOutput">5</output></label>
            <input id="imu-demo-top-k-en" data-demo-id="topK" type="range" min="0" max="10" step="1" value="5">
          </div>
        </div>
        <div class="imu-demo-actions">
          <button class="imu-demo-button imu-demo-button--primary" data-demo-id="run" type="button">Run current recording</button>
          <button class="imu-demo-button" data-demo-id="reset" type="button">Reset to sample</button>
        </div>
      </div>
    </div>
    <p class="imu-demo-privacy"><span aria-hidden="true">⌁</span><span><strong>Local by design.</strong> The page downloads public, checksum-verified model files from Hugging Face. Your IMU samples are parsed, filtered, inferred, plotted, and exported inside a Web Worker on this device.</span></p>
  </section>

  <section class="imu-demo-status" data-demo-id="status" data-kind="ready" aria-live="polite" aria-atomic="true">
    <span class="imu-demo-status__dot" aria-hidden="true"></span>
    <strong data-demo-id="statusTitle"></strong>
    <span data-demo-id="statusDetail"></span>
    <progress data-demo-id="progress" max="1" hidden></progress>
  </section>

  <section class="imu-demo-results" data-demo-id="results" hidden aria-label="Inference results">
    <div class="imu-demo-summary">
      <div class="imu-demo-stat"><strong data-demo-id="samples">—</strong><span>input samples</span></div>
      <div class="imu-demo-stat"><strong data-demo-id="recordingDuration">—</strong><span>recording duration</span></div>
      <div class="imu-demo-stat"><strong data-demo-id="points">—</strong><span>timeline points</span></div>
      <div class="imu-demo-stat"><strong data-demo-id="records">—</strong><span>activity records</span></div>
    </div>
    <div class="imu-demo-runtime" aria-label="Local runtime information">
      <span data-demo-id="backend"></span>
      <span data-demo-id="cache"></span>
      <span>ONNX · 3 s + 5 s + 8 s</span>
    </div>

    <div class="imu-demo-tabs">
      <div class="imu-demo-tabs__list" role="tablist" aria-label="Result views">
        <button class="imu-demo-tab" type="button" role="tab" data-demo-tab="signal" aria-selected="true">Raw signals</button>
        <button class="imu-demo-tab" type="button" role="tab" data-demo-tab="timeline" aria-selected="false" tabindex="-1">Likelihood + timeline</button>
        <button class="imu-demo-tab" type="button" role="tab" data-demo-tab="records" aria-selected="false" tabindex="-1">Activity records</button>
      </div>
      <div class="imu-demo-panel" role="tabpanel" data-demo-panel="signal">
        <canvas data-demo-id="signalCanvas" role="img" aria-label="Raw accelerometer and gyroscope signal chart"></canvas>
        <p class="imu-demo-chart-note">The chart is downsampled only for display. Model inference uses every validated sample.</p>
      </div>
      <div class="imu-demo-panel" role="tabpanel" data-demo-panel="timeline" hidden>
        <canvas data-demo-id="timelineCanvas" role="img" aria-label="Smoothed activity likelihood and decoded timeline chart"></canvas>
        <p class="imu-demo-chart-note">The upper plot is the fused, smoothed posterior; the lower band is the Viterbi-decoded state.</p>
      </div>
      <div class="imu-demo-panel" role="tabpanel" data-demo-panel="records" hidden>
        <div class="imu-demo-table-wrap">
          <table class="imu-demo-table">
            <thead data-demo-id="tableHead"></thead>
            <tbody data-demo-id="tableBody"></tbody>
          </table>
        </div>
        <div class="imu-demo-download">
          <a class="imu-demo-button imu-demo-button--primary" data-demo-id="download" download>Download result CSV</a>
        </div>
      </div>
    </div>
  </section>
</div>

<noscript>
This Demo requires JavaScript because the complete inference pipeline executes in your browser.
</noscript>

## What happens after Run

<div class="imu-demo-details-grid">
  <article class="imu-demo-detail"><strong>1 · Validate and filter</strong><p>The Worker checks the seven-column 100 Hz format, sorts timestamps, removes invalid rows, and applies the repository's zero-phase fourth-order Butterworth filter.</p></article>
  <article class="imu-demo-detail"><strong>2 · Execute real models</strong><p>ONNX Runtime Web evaluates every 3 s, 5 s, and 8 s sliding window. WebGPU is attempted first; WASM provides the portable CPU fallback.</p></article>
  <article class="imu-demo-detail"><strong>3 · Build records</strong><p>The same scale alignment, fusion, smoothing, Viterbi decoding, boundary refinement, duration filtering, confidence filtering, and Top-K logic create the table and CSV.</p></article>
</div>

The first run downloads about **17 MB of ONNX weights** plus the appropriate browser runtime. Both are cached by the browser when storage is available. Later clicks still perform the model computations again; they simply avoid downloading the same weights.

The model URLs are pinned to Hugging Face revision `e0f89bb6…`. Every weight file is checked against the SHA-256 values published in [`model-assets.json`](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/model-assets.json) before a session is created.

## Use your own recording

Provide UTF-8 tab-separated text with these exact columns:

```text
ACC_TIME  ACC_X  ACC_Y  ACC_Z  GYRO_X  GYRO_Y  GYRO_Z
```

`ACC_TIME` must contain unique millisecond timestamps with a median interval of 8–12 ms. Extra columns and invalid rows are ignored. The public Demo accepts 800–60,000 valid samples (about 8 seconds to 10 minutes at 100 Hz).

!!! warning "Research output"

    Passing the file-format checks does not guarantee compatibility with the training protocol. Sensor placement, axis orientation, units, device characteristics, and preprocessing must match the documented setup. Predictions are not medical, safety, or coaching advice.

Need a server-side comparison? The original [Hugging Face Space](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline) remains available, while this page is the privacy-preserving, visitor-compute version.
