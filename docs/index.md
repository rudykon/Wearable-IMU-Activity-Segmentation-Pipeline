---
hide:
  - toc
---

<section class="home-hero">
  <span class="hero-blob one" aria-hidden="true"></span>
  <span class="hero-blob two" aria-hidden="true"></span>
  <div class="hero-copy">
    <span class="hero-kicker">Open-source · Research to edge</span>
    <h1>Turn long IMU streams into <span class="gradient-text">auditable activity timelines.</span></h1>
    <p class="hero-lead">
      A reproducible Python and Android pipeline for multi-scale segmentation of
      wrist-worn accelerometer and gyroscope data—from model training and
      temporal decoding to segment evaluation and on-device inference.
    </p>
    <div class="hero-actions">
      <a class="hero-button primary" href="getting-started/quickstart/">
        Run the pipeline
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
      <a class="hero-button" href="guide/pipeline/">Explore the architecture</a>
    </div>
    <div class="hero-proof" aria-label="Project capabilities">
      <span>Reproducible Python</span>
      <span>Multi-scale models</span>
      <span>Android ONNX</span>
    </div>
  </div>
  <div class="hero-visual">
    <div class="floating-badge badge-model">3-scale ensemble</div>
    <div class="floating-badge badge-edge">Edge-ready ONNX</div>
    <div class="signal-card">
      <div class="signal-toolbar">
        <span class="signal-live"><span class="signal-dot"></span>LIVE SENSOR STREAM</span>
        <span>WRIST IMU · 100 HZ</span>
      </div>
      <div class="signal-window">
        <svg viewBox="0 0 520 224" role="img" aria-label="Stylized accelerometer and gyroscope traces decoded into activity segments">
          <g stroke="#c7d2fe" stroke-width="1" opacity=".65">
            <path d="M0 42H520M0 84H520M0 126H520M0 168H520"/>
            <path d="M65 0V180M130 0V180M195 0V180M260 0V180M325 0V180M390 0V180M455 0V180"/>
          </g>
          <path d="M0 96C20 91 28 58 44 83S70 132 86 99s24-66 44-20 34 31 49-6 26-65 43 9 35 14 47-17 25-18 41 21 32 33 48-22 29-55 43 4 31 62 47 3 30-53 44-6 27 20 38-2 20-17 30 3" fill="none" stroke="#4f46e5" stroke-width="4" stroke-linecap="round"/>
          <path d="M0 130C24 112 33 154 53 129s33-74 49-15 31 45 47 5 29-38 44 17 33 19 47-14 27-57 45 7 34 19 49-18 30-27 45 16 28 36 45-8 27-30 42 2 28 26 44-4" fill="none" stroke="#7c3aed" stroke-width="3" stroke-linecap="round" opacity=".9"/>
          <g>
            <rect x="5" y="192" width="78" height="13" rx="6.5" fill="#10b981"/>
            <rect x="89" y="192" width="116" height="13" rx="6.5" fill="#f59e0b"/>
            <rect x="211" y="192" width="55" height="13" rx="6.5" fill="#6366f1"/>
            <rect x="272" y="192" width="98" height="13" rx="6.5" fill="#8b5cf6"/>
            <rect x="376" y="192" width="139" height="13" rx="6.5" fill="#14b8a6"/>
          </g>
        </svg>
      </div>
      <div class="signal-foot">
        <span>ACC + GYRO · 6 CHANNELS</span>
        <span>DECODED TIMELINE</span>
      </div>
      <div class="signal-meta">
        <div><strong>3 / 5 / 8 s</strong><span>temporal scales</span></div>
        <div><strong>CNN–BiLSTM</strong><span>window model</span></div>
        <div><strong>Viterbi</strong><span>sequence decode</span></div>
      </div>
    </div>
  </div>
</section>

<div class="metric-strip">
  <div class="metric"><strong>100 Hz</strong><span>sampling rate</span></div>
  <div class="metric"><strong>6</strong><span>physical IMU channels</span></div>
  <div class="metric"><strong>3</strong><span>temporal scales</span></div>
  <div class="metric"><strong>5 + bg</strong><span>output classes</span></div>
</div>

<span class="section-eyebrow">Research to deployment</span>

## One repository, the complete research-to-device path {: .section-title}

<p class="section-lead">A coherent workflow keeps every stage inspectable—from physical-unit sensor input to deployable activity segments—without hiding decisions behind a black box.</p>

<div class="feature-grid">
  <article class="feature-card">
    <span class="feature-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24"><path d="M3 12h3l2-6 4 13 3-10 2 6h4"/></svg>
    </span>
    <h3>Long-session segmentation</h3>
    <p>Convert continuous per-user sensor recordings into explicit <code>user_id, category, start, end</code> activity records.</p>
  </article>
  <article class="feature-card">
    <span class="feature-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24"><rect x="3" y="5" width="9" height="5" rx="1"/><rect x="6" y="10" width="12" height="5" rx="1"/><rect x="9" y="15" width="12" height="5" rx="1"/></svg>
    </span>
    <h3>Multi-scale modeling</h3>
    <p>Align 3-, 5-, and 8-second CNN–BiLSTM predictions so short motion signatures and longer context inform one timeline.</p>
  </article>
  <article class="feature-card">
    <span class="feature-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24"><path d="M3 16l5-5 4 3 5-7 4 3"/><path d="M17 7h4v4"/></svg>
    </span>
    <h3>Temporal decoding</h3>
    <p>Apply LBSA fusion, probability smoothing, Viterbi decoding, boundary refinement, overlap handling, and segment filtering.</p>
  </article>
  <article class="feature-card">
    <span class="feature-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24"><rect x="6" y="2" width="12" height="20" rx="2"/><path d="M10 5h4M10 18h4"/><circle cx="12" cy="12" r="3"/></svg>
    </span>
    <h3>Android deployment</h3>
    <p>Acquire WT9011DCL-BT50 BLE data, visualize signals, record CSV, and execute the selected ONNX ensemble on device.</p>
  </article>
</div>

<span class="section-eyebrow">Auditable by design</span>

## Pipeline at a glance {: .section-title}

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="Open the full-resolution overall framework figure">
    <img src="assets/fig02_overall_framework.png" alt="Existing project framework figure showing the IMU stream, scale-specific CNN–BiLSTM models, LBSA fusion, temporal record layer, and segment records" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Repository figure: IMU stream → scale-specific CNN–BiLSTM models → LBSA → temporal record layer → segment records. Select the image to view it at full resolution.</figcaption>
</figure>

<p class="pipeline-summary">The Python research path and Android demonstration share the same observable contract: six physical-unit IMU channels enter the system; time-aligned activity segments leave it. The repository keeps each intermediate choice inspectable through configuration files, fixed model assets, experiment scripts, and segment-level evaluation.</p>

| Layer | Repository implementation | Main artifact |
| --- | --- | --- |
| Input | UTF-8 tab-separated ACC/GYRO recordings | Per-user signal files |
| Representation | Multi-kernel 1D CNN + BiLSTM | Window probabilities |
| Fusion | 3 s / 5 s / 8 s probability alignment | Multi-scale sequence |
| Temporal logic | LBSA, smoothing, Viterbi, refinement | Activity timeline |
| Output | Segment writer and evaluator | XLSX records and F1 metrics |
| Deployment | Android BLE + ONNX Runtime | Real-time or offline recognition |

### Physical deployment path

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" target="_blank" rel="noopener" aria-label="Open the full-resolution physical deployment chain figure">
    <img src="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" alt="Existing project figure showing the physical deployment chain from the wearable IMU sensor through BLE and Android on-device inference to activity recognition" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Repository figure: wearable IMU → BLE acquisition → Android signal processing → on-device multi-scale inference → activity recognition. Select the image to view it at full resolution.</figcaption>
</figure>

<span class="section-eyebrow">Fast validation</span>

## Start with a public smoke test {: .section-title}

The lightweight smoke test validates imports, canonical paths, temporary signal
loading, annotation loading, and workbook writing. It does **not** require
participant recordings or a GPU.

=== "Conda"

    ~~~bash
    git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
    cd Wearable-IMU-Activity-Segmentation-Pipeline
    conda env create -f environment.yml
    conda activate imu-activity-pipeline
    python -m pip install -e .
    python tests/smoke_test.py
    ~~~

=== "pip"

    ~~~bash
    git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
    cd Wearable-IMU-Activity-Segmentation-Pipeline
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m pip install -e .
    python tests/smoke_test.py
    ~~~

!!! important "Data access boundary"

    Participant sensor streams are not distributed in the GitHub repository.
    Authorized data remain local under the ignored `data/` layout. Selected
    Python checkpoints, normalization assets, and Android ONNX files are
    versioned separately from those recordings.

<span class="section-eyebrow">Output vocabulary</span>

## Supported foreground activities {: .section-title}

| Chinese label | English label | Output behavior |
| --- | --- | --- |
| 羽毛球 | Badminton | Foreground segment |
| 跳绳 | Jump rope | Foreground segment |
| 飞鸟 | Fly | Foreground segment |
| 跑步 | Running | Foreground segment |
| 乒乓球 | Table tennis | Foreground segment |

Background / no-activity is modeled internally where required, while submitted
segment records contain foreground activities.

<div class="cta-panel">
  <div>
    <h3>Ready to run an authorized dataset?</h3>
    <p>Prepare the canonical folder layout, then follow the inference walkthrough.</p>
  </div>
  <a class="md-button md-button--primary" href="getting-started/quickstart/">Open quick start</a>
</div>
