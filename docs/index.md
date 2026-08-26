---
hide:
  - navigation
  - toc
---

<section class="home-hero showcase-hero">
  <div class="hero-copy">
    <p class="hero-kicker">Wearable IMU activity records</p>
    <h1 class="paper-title">An End-to-End Wearable IMU System for Segment-Level Activity Recognition via <span class="title-accent">Multi-Scale Arbitration and a Temporal Record Layer</span></h1>
    <p class="hero-lead">A multi-scale wrist-IMU pipeline that converts continuous 100 Hz signals into timestamped activity records.</p>
    <div class="hero-actions">
      <a class="hero-button primary" href="deployment/android/">
        Android Demo
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
      <a class="hero-button" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">HF Demo</a>
      <a class="hero-button" href="guide/pipeline/">Method</a>
      <a class="hero-button github-button" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.22c-3.23.7-3.91-1.37-3.91-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.72 1.27 3.39.97.1-.75.4-1.27.74-1.56-2.58-.29-5.29-1.29-5.29-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.48.11-3.05 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.19-1.49 3.16-1.18 3.16-1.18.63 1.57.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.4-2.72 5.38-5.31 5.67.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .7Z"/></svg>
        GitHub
      </a>
    </div>
    <div class="hero-proof" aria-label="Key evidence">
      <span>259.6 h sensor data</span>
      <span>37 external-test recordings</span>
      <span>0.90 micro-F1</span>
    </div>
  </div>

  <div class="hero-visual" aria-label="Illustrative IMU-to-record pipeline">
    <div class="hero-system-card">
      <div class="system-card-header">
        <span>Live IMU stream</span>
        <span>100 Hz · 6 channels</span>
      </div>
      <svg class="hero-wave" viewBox="0 0 520 210" role="img" aria-label="Stylized accelerometer and gyroscope traces decoded into activity segments">
        <g stroke="#dce6f1" stroke-width="1">
          <path d="M0 40H520M0 80H520M0 120H520M0 160H520"/>
          <path d="M65 0V170M130 0V170M195 0V170M260 0V170M325 0V170M390 0V170M455 0V170"/>
        </g>
        <path d="M0 92C18 84 28 50 45 78s27 51 44 17 25-63 44-18 33 32 49-5 27-62 45 8 34 16 48-16 27-20 44 17 31 35 48-20 30-50 44 5 30 55 47 2 28-49 45-5 28 18 38-2 22-14 30 2" fill="none" stroke="#3d6fb6" stroke-width="4" stroke-linecap="round"/>
        <path d="M0 128C24 112 34 151 55 126s31-68 49-13 31 42 47 5 29-35 45 15 32 18 47-13 27-53 45 7 34 18 49-17 30-25 45 15 29 33 46-8 27-28 43 2 28 23 44-4" fill="none" stroke="#756bb1" stroke-width="3" stroke-linecap="round" opacity=".9"/>
        <g>
          <rect x="8" y="184" width="96" height="12" rx="6" fill="#168c7e"/>
          <rect x="111" y="184" width="126" height="12" rx="6" fill="#d9822b"/>
          <rect x="244" y="184" width="72" height="12" rx="6" fill="#3d6fb6"/>
          <rect x="323" y="184" width="86" height="12" rx="6" fill="#756bb1"/>
          <rect x="416" y="184" width="96" height="12" rx="6" fill="#168c7e"/>
        </g>
      </svg>
      <div class="hero-model-row" aria-label="Three models followed by fusion and record decoding">
        <span>3 s model</span>
        <span>5 s model</span>
        <span>8 s model</span>
        <strong>LBSA</strong>
        <strong>TRL</strong>
      </div>
      <div class="hero-record-list" aria-label="Illustrative output records">
        <div><time>09:02–09:17</time><strong>Badminton</strong></div>
        <div><time>09:25–09:34</time><strong>Jump rope</strong></div>
        <div><time>09:41–09:53</time><strong>Running</strong></div>
      </div>
      <div class="system-card-footer">
        <span>Continuous signal</span>
        <span>Stable records</span>
      </div>
    </div>
  </div>
</section>

<section class="home-section" markdown="1">

## Problem and output

Window-level predictions do not directly provide reliable activity records. A long-session system must recover the activity class, event count, duration, and boundaries from continuous sensor streams.

<div class="record-transform showcase-transform" aria-label="System input and output">
  <code>6-channel wrist IMU stream</code>
  <span aria-hidden="true">→</span>
  <code>{activity, start, end}</code>
</div>

The evaluated task covers five sports recorded from a 100 Hz wrist accelerometer and gyroscope. Background motion supports decoding but is not emitted as a workout record.

</section>

<section class="home-section" markdown="1">

## Method

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="Open the full-resolution framework figure">
    <img src="assets/fig02_overall_framework.png" alt="Overall framework from wrist IMU input through three scale-specific models, LBSA, and TRL to activity records" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">IMU stream → 3/5/8-second posteriors → LBSA → TRL → activity records.</figcaption>
</figure>

<div class="method-points">
  <article class="method-point">
    <span>01</span>
    <h3>Multi-scale models</h3>
    <p>Scale-specific CNN–BiLSTMs analyze 3-, 5-, and 8-second views.</p>
  </article>
  <article class="method-point">
    <span>02</span>
    <h3>Boundary-aware fusion</h3>
    <p>LBSA favors short evidence near transitions and longer context in stable regions.</p>
  </article>
  <article class="method-point">
    <span>03</span>
    <h3>Record construction</h3>
    <p>TRL smooths, decodes, merges, refines, and filters the fused timeline.</p>
  </article>
</div>

[Read the full method](guide/pipeline.md){ .md-button }

</section>

<section class="home-section" markdown="1">

## Evidence

<p class="section-intro">The operating point was frozen before evaluation on the independent 37-recording external test.</p>

<div class="metric-strip metric-strip--three showcase-metrics">
  <div class="metric"><strong>259.6 h</strong><span>sensor data</span></div>
  <div class="metric"><strong>37 / 114</strong><span>test recordings / segments</span></div>
  <div class="metric"><strong>0.90</strong><span>micro-F1</span></div>
</div>

<div class="evidence-summary">
  <div class="evidence-highlight">
    <strong>0.89</strong>
    <span>mean-user F1 · LBSA + TRL</span>
  </div>
  <div class="evidence-scope">
    <h3>Evaluated scope</h3>
    <p>Five activities under the studied device, wrist placement, and long-session protocol.</p>
    <p><strong>New devices, placements, users, activities, and deployment conditions require new validation.</strong></p>
  </div>
</div>

[Inspect results and failure cases](research/paper.md){ .md-button }

</section>

<section class="home-section" markdown="1">

## See the pipeline run

<div class="demo-showcase">
  <a class="demo-showcase__media" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener" aria-label="Open the live Hugging Face demo">
    <img src="assets/demo-results-paper-notation.jpg" alt="Actual Hugging Face demo output showing paper-style wrist IMU channel notation and timestamped activity records" loading="lazy" decoding="async">
    <span>Open live demo ↗</span>
  </a>
  <div class="demo-showcase__copy">
    <p class="hero-kicker">Browser demo</p>
    <h3>Signals, probabilities, timeline, and records in one view</h3>
    <p>Run the public 3-, 5-, and 8-second models with the bundled synthetic recording or a compatible wrist-IMU file.</p>
    <ul>
      <li>Six-channel signal plots</li>
      <li>Decoded activity timeline</li>
      <li>Record table and CSV export</li>
    </ul>
    <div class="demo-actions">
      <a class="demo-action primary" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">Run live demo</a>
      <a class="demo-action github" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.22c-3.23.7-3.91-1.37-3.91-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.72 1.27 3.39.97.1-.75.4-1.27.74-1.56-2.58-.29-5.29-1.29-5.29-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.48.11-3.05 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.19-1.49 3.16-1.18 3.16-1.18.63 1.57.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.4-2.72 5.38-5.31 5.67.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .7Z"/></svg>
        Demo source
      </a>
      <a class="demo-action" href="deployment/hugging-face/">Input and privacy notes</a>
    </div>
  </div>
</div>

</section>

<section class="home-section" markdown="1">

## Reproduce and inspect

<div class="resource-grid">
  <a class="resource-card" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
    <span>Source</span>
    <h3>GitHub</h3>
    <p>Browse the package, experiments, Android app, issues, and release history.</p>
  </a>
  <a class="resource-card" href="https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
    <span>Weights</span>
    <h3>Models</h3>
    <p>Download verified PyTorch and ONNX assets from Hugging Face.</p>
  </a>
  <a class="resource-card" href="getting-started/quickstart/">
    <span>Code</span>
    <h3>Quickstart</h3>
    <p>Verify the public package and run authorized-data inference.</p>
  </a>
  <a class="resource-card" href="deployment/android/">
    <span>On-device demo</span>
    <h3>Android APK</h3>
    <p>Download the app and a synthetic sample for offline on-device inference.</p>
  </a>
</div>

Participant recordings are not distributed on GitHub. The [reproduction guide](reproduce.md) separates public verification from workflows that require authorized data.

</section>
