---
title: Browser Demo
description: Run the real multi-scale IMU activity segmentation pipeline locally in your browser.
hide:
  - toc
---

<section class="demo-page-hero imu-browser-page-hero">
  <div>
    <p class="hero-kicker">Real inference on the visitor's device</p>
    <h1>From wrist IMU signals to activity records</h1>
    <p>Run the bundled 120-second synthetic example, or replace it with a compatible 100 Hz wrist-IMU recording. Nothing to install, and no sensor data is uploaded.</p>
    <div class="demo-facts" aria-label="Browser Demo capabilities">
      <span>Real public models</span>
      <span>WebGPU / WASM</span>
      <span>Timeline + CSV</span>
      <span>Data stays local</span>
    </div>
    <div class="demo-actions">
      <a class="demo-action primary" href="#run-browser-demo">Run the Demo</a>
      <a class="demo-action github" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/docs/assets/javascripts/imu-demo-worker.mjs" target="_blank" rel="noopener">Browser source</a>
      <a class="demo-action" href="https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">Model weights</a>
    </div>
  </div>
  <a class="demo-page-image" href="../assets/demo/synthetic-activity-likelihood-timeline.png" target="_blank" rel="noopener" aria-label="Open the full-resolution bundled-example reference output">
    <img src="../assets/demo/synthetic-activity-likelihood-timeline.png" alt="Activity likelihood curves and final activity timeline for the bundled synthetic example" loading="eager" decoding="async">
    <span>Bundled-example reference output</span>
  </a>
</section>

<nav class="demo-page-nav" aria-label="Browser Demo sections">
  <a href="#run-browser-demo">Run Demo</a>
  <a href="#quick-start">Quick start</a>
  <a href="#expected-output">Reference output</a>
  <a href="#understand-results">Read the results</a>
  <a href="#parameters">Parameters</a>
  <a href="#use-your-own-recording">Upload a recording</a>
</nav>

## Run it here {#run-browser-demo}

The bundled example is ready. Keep the defaults and choose **Run current recording** to execute the complete three-scale pipeline on this device.

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
        <p class="imu-demo-chart-note">The upper plot is the fused, smoothed posterior; the lower stepped trace is the Viterbi-decoded class index, matching the HF server figure.</p>
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

## Complete an inference run in three steps {#quick-start}

<div class="demo-steps imu-browser-steps">
  <article class="demo-step">
    <span class="demo-step__number">1</span>
    <h3>Confirm the bundled example</h3>
    <p><code>synthetic_activity_imu.tsv</code> is already loaded with 12,000 samples and 120 seconds of six-channel IMU data.</p>
  </article>
  <article class="demo-step">
    <span class="demo-step__number">2</span>
    <h3>Keep the defaults</h3>
    <p>Start with local-boundary fusion, a 5-second minimum duration, 0.30 confidence, and Top-K 5.</p>
  </article>
  <article class="demo-step">
    <span class="demo-step__number">3</span>
    <h3>Run and inspect</h3>
    <p>Choose <strong>Run current recording</strong>, then inspect raw signals, the likelihood timeline, and activity records.</p>
  </article>
</div>

<div class="demo-defaults" aria-label="Bundled-example default settings">
  <div><span>Input file</span><strong>synthetic_activity_imu.tsv</strong></div>
  <div><span>Fusion</span><strong>local_boundary</strong></div>
  <div><span>Minimum duration</span><strong>5 s</strong></div>
  <div><span>Confidence</span><strong>0.30</strong></div>
  <div><span>Top-K</span><strong>5</strong></div>
</div>

!!! tip "After you choose Run"

    The Worker validates and zero-phase Butterworth-filters the recording, evaluates the real 3-, 5-, and 8-second ONNX models, then performs scale fusion, Viterbi decoding, boundary refinement, and record filtering.

## Bundled-example reference output {#expected-output}

With the defaults above, the public models produce 118 timeline points and two activity records. Use these values to confirm that the browser pipeline is operating correctly.

<div class="demo-run-summary" aria-label="Bundled-example reference summary">
  <div class="demo-run-stat"><strong>12,000</strong><span>input samples</span></div>
  <div class="demo-run-stat"><strong>120.0 s</strong><span>recording duration</span></div>
  <div class="demo-run-stat"><strong>118</strong><span>timeline points</span></div>
  <div class="demo-run-stat"><strong>2</strong><span>activity records</span></div>
</div>

| Activity | Start (s) | End (s) | Duration (s) | Confidence |
| --- | ---: | ---: | ---: | ---: |
| Fly | 29.84 | 73.15 | 43.31 | 0.4038 |
| Running | 76.06 | 98.24 | 22.18 | 0.3186 |

!!! note "A pipeline demonstration, not an accuracy test"

    The bundled file is deterministically generated and contains no participant data. Its predicted classes and boundaries are model outputs, not ground-truth labels or accuracy evidence.

## Read the three result tabs {#understand-results}

<div class="imu-demo-details-grid">
  <article class="imu-demo-detail"><strong>Raw signals</strong><p>Start with the three accelerometer and three gyroscope channels to check completeness, sampling stability, and visible movement.</p></article>
  <article class="imu-demo-detail"><strong>Likelihood + timeline</strong><p>The upper chart shows six smoothed class probabilities; the lower stepped trace shows the final class index after multi-scale fusion and Viterbi decoding.</p></article>
  <article class="imu-demo-detail"><strong>Activity records</strong><p>Each row reports the activity, start, end, duration, and confidence and can be downloaded directly as CSV.</p></article>
</div>

## How parameters change the result {#parameters}

| Parameter | Default | Effect |
| --- | --- | --- |
| Multi-scale fusion | `local_boundary` | Combines the 3-, 5-, and 8-second posteriors, with the largest effect near activity transitions. |
| Minimum duration | `5 s` | Removes shorter intervals; larger values favor longer activities. |
| Confidence | `0.30` | Removes weaker records; raising it may reduce both false positives and recall. |
| Top-K | `5` | Limits the number of returned records; `0` keeps all records. |

These are **Demo settings** chosen to make the 120-second example easy to inspect, not immutable activity definitions from the paper experiments.

## Upload your own recording {#use-your-own-recording}

1. Prepare a UTF-8, tab-separated `.txt` or `.tsv` file.
2. Choose or drop the file and keep the defaults for the first run.
3. Inspect **Raw signals** before interpreting model output; change only one parameter at a time afterward.

Required column names:

```text
ACC_TIME  ACC_X  ACC_Y  ACC_Z  GYRO_X  GYRO_Y  GYRO_Z
```

`ACC_TIME` must contain strictly increasing millisecond timestamps with a median interval of 8–12 ms. Extra columns and invalid rows are ignored. The public Demo accepts 800–60,000 valid samples (about 8 seconds to 10 minutes at 100 Hz).

## Local compute and privacy {#local-compute}

<div class="demo-reading-grid">
  <div class="demo-reading"><strong>What the browser downloads</strong><p>The first run downloads about 17 MB of ONNX weights and a roughly 26 MB runtime. Files are SHA-256 verified and cached when browser storage is available.</p></div>
  <div class="demo-reading"><strong>Where the data goes</strong><p>Parsing, filtering, inference, plotting, and CSV export all run in a Web Worker on this device; the IMU recording is never uploaded.</p></div>
</div>

Models are pinned to Hugging Face revision `e0f89bb6…`, with checksums published in [`model-assets.json`](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/model-assets.json). WebGPU is preferred when available; otherwise the Demo falls back to WASM CPU, so runtime varies across computers and phones.

!!! warning "Research output"

    A valid file format does not guarantee compatibility with the training protocol. Sensor placement, axis orientation, units, device characteristics, and preprocessing must match. Predictions are not medical, safety, or coaching advice.

For a server-side comparison, use the original [Hugging Face Space](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline). This page uses visitor compute and keeps the recording local.
