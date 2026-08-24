---
hide:
  - toc
---

<section class="home-hero">
  <span class="hero-blob one" aria-hidden="true"></span>
  <span class="hero-blob two" aria-hidden="true"></span>
  <div class="hero-copy">
    <span class="hero-kicker">Wearable IMU</span>
    <h1>Motion to <span class="gradient-text">activity records.</span></h1>
    <p class="hero-lead">Segment 100 Hz wrist IMU data into timestamped activities.</p>
    <div class="hero-actions">
      <a class="hero-button primary" href="context/use-cases/">
        Scenarios
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
      <a class="hero-button" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
        Demo
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
      <a class="hero-button" href="https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">Models</a>
      <a class="hero-button" href="research/paper/">Results</a>
    </div>
    <div class="hero-proof" aria-label="Project focus">
      <span>Long sessions</span>
      <span>5 activities</span>
      <span>Web + Android</span>
    </div>
  </div>
  <div class="hero-visual">
    <div class="floating-badge badge-model">3 scales</div>
    <div class="floating-badge badge-edge">Stable records</div>
    <div class="signal-card">
      <div class="signal-toolbar">
        <span class="signal-live"><span class="signal-dot"></span>LIVE IMU</span>
        <span>100 HZ</span>
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
        <span>6 CHANNELS</span>
        <span>TIMELINE</span>
      </div>
      <div class="signal-meta">
        <div><strong>3 / 5 / 8 s</strong><span>windows</span></div>
        <div><strong>3 models</strong><span>combined</span></div>
        <div><strong>1 timeline</strong><span>output</span></div>
      </div>
    </div>
  </div>
</section>

<span class="section-eyebrow">Problem</span>

## Signals are not records. {: .section-title}

<p class="section-lead">One hour contains 2.16 million readings. The useful output is a short activity log.</p>

<div class="story-grid">
  <article class="story-card story-card-input">
    <span class="story-card-kicker">Input</span>
    <strong>Six motion channels</strong>
    <div class="signal-token-list" aria-label="Six input channels">
      <span>ACC_X</span><span>ACC_Y</span><span>ACC_Z</span>
      <span>GYRO_X</span><span>GYRO_Y</span><span>GYRO_Z</span>
    </div>
    <p>Movement, transitions, pauses, and noise share one stream.</p>
  </article>
  <article class="story-card story-card-output">
    <span class="story-card-kicker">Output</span>
    <strong>Timestamped activities</strong>
    <div class="record-list" aria-label="Illustrative activity records">
      <div><time>09:02–09:17</time><span>Badminton</span></div>
      <div><time>09:25–09:34</time><span>Jump rope</span></div>
      <div><time>09:41–09:53</time><span>Running</span></div>
    </div>
    <p>Each record gives the activity, start, end, and duration.</p>
  </article>
</div>

<p class="story-caption">Illustrative times; no participant data.</p>

<div class="metric-strip context-metrics">
  <div class="metric"><strong>100 Hz</strong><span>sampling</span></div>
  <div class="metric"><strong>6</strong><span>channels</span></div>
  <div class="metric"><strong>5</strong><span>activities</span></div>
  <div class="metric"><strong>4</strong><span>output fields</span></div>
</div>

<span class="section-eyebrow">Scenarios</span>

## Where it fits {: .section-title}

<div class="scenario-grid">
  <article class="scenario-card">
    <span class="scenario-tag established">Research</span>
    <h3>Long-session benchmarks</h3>
    <p>Compare records, boundaries, counts, and false alarms.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag scoped">Prototype</span>
    <h3>Workout logs</h3>
    <p>Create reviewable records for the five supported activities.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag established">Mobile</span>
    <h3>Phone deployment</h3>
    <p>Run the sensor-to-Android ONNX path over BLE.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag exploratory">Review</span>
    <h3>Annotation aid</h3>
    <p>Locate likely activities and boundary errors for human review.</p>
  </article>
</div>

<div class="paper-home-cta story-cta">
  <p>New devices, placements, and populations need new validation.</p>
  <a class="md-button md-button--primary" href="context/use-cases/">All scenarios</a>
</div>

<span class="section-eyebrow">Challenge</span>

## Good windows can make bad records. {: .section-title}

<p class="section-lead">Confidence dips can split events, shift boundaries, or create false alarms.</p>

<figure class="paper-figure">
  <a class="pipeline-image-link" href="assets/manuscript-figures/fig01_window_to_record_gap.png" target="_blank" rel="noopener" aria-label="Open the full-resolution window-to-record gap figure">
    <img src="assets/manuscript-figures/fig01_window_to_record_gap.png" alt="Posterior trajectories, naive fragmented records, and the stabilized activity records produced by the Temporal Record Layer" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Paper Fig. 1. Timeline decoding joins unstable window predictions.</figcaption>
</figure>

<div class="metric-strip paper-metrics">
  <div class="metric"><strong>137</strong><span>recordings</span></div>
  <div class="metric"><strong>259.6 h</strong><span>sensor data</span></div>
  <div class="metric"><strong>0.89</strong><span>mean-user F1</span></div>
  <div class="metric"><strong>0.90</strong><span>micro-F1</span></div>
</div>

<div class="paper-home-cta">
  <p>These F1 scores measure complete records on the fixed external test.</p>
  <a class="md-button md-button--primary" href="research/paper/">Full results</a>
</div>

<span class="section-eyebrow">Method</span>

## Four steps {: .section-title}

<div class="feature-grid process-grid">
  <article class="feature-card process-card">
    <span class="process-step">01</span>
    <h3>Capture</h3>
    <p>Keep six IMU channels and timestamps.</p>
  </article>
  <article class="feature-card process-card">
    <span class="process-step">02</span>
    <h3>Classify</h3>
    <p>Run 3-, 5-, and 8-second models.</p>
  </article>
  <article class="feature-card process-card">
    <span class="process-step">03</span>
    <h3>Decode</h3>
    <p>LBSA selects scale; TRL builds stable records.</p>
  </article>
  <article class="feature-card process-card">
    <span class="process-step">04</span>
    <h3>Use</h3>
    <p>Export records, inspect plots, or run on Android.</p>
  </article>
</div>

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="Open the full-resolution overall framework figure">
    <img src="assets/fig02_overall_framework.png" alt="Existing project framework figure showing the IMU stream, scale-specific CNN–BiLSTM models, LBSA fusion, temporal record layer, and segment records" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">IMU → three models → LBSA → TRL → records.</figcaption>
</figure>

<span class="section-eyebrow">Evidence</span>

## Results and limits {: .section-title}

<div class="evidence-grid">
  <article class="evidence-card">
    <span class="evidence-kicker">Supported</span>
    <h3>Tested here</h3>
    <ul>
      <li>Five activities in long sessions.</li>
      <li>Fixed external test: 37 recordings.</li>
      <li>Python, web demo, and Android paths.</li>
    </ul>
  </article>
  <article class="evidence-card caution">
    <span class="evidence-kicker">Not established</span>
    <h3>Test again</h3>
    <ul>
      <li>New devices, placements, users, or activities.</li>
      <li>Clinical, coaching, safety, or production use.</li>
      <li>Dense or adjacent same-class events.</li>
    </ul>
  </article>
</div>

<span class="section-eyebrow">Next</span>

## Choose a path {: .section-title}

<div class="route-grid">
  <a class="route-card" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
    <span>2 min</span>
    <h3>Demo</h3>
    <p>Run a synthetic session in the browser.</p>
  </a>
  <a class="route-card" href="context/use-cases/">
    <span>Context</span>
    <h3>Scenarios</h3>
    <p>See uses, assumptions, and limits.</p>
  </a>
  <a class="route-card" href="guide/pipeline/">
    <span>Technical</span>
    <h3>Pipeline</h3>
    <p>Trace input, models, decoding, and output.</p>
  </a>
  <a class="route-card" href="deployment/android/">
    <span>Mobile</span>
    <h3>Android</h3>
    <p>Build the BLE and ONNX app.</p>
  </a>
</div>
