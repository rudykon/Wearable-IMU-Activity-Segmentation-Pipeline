---
hide:
  - toc
---

<section class="home-hero">
  <span class="hero-blob one" aria-hidden="true"></span>
  <span class="hero-blob two" aria-hidden="true"></span>
  <div class="hero-copy">
    <span class="hero-kicker">Open source · From movement to meaning</span>
    <h1>Turn raw wrist motion into <span class="gradient-text">activity records people can inspect.</span></h1>
    <p class="hero-lead">
      A wrist IMU streams six channels at 100 Hz. This project turns that
      continuous signal into timestamped records of what happened and when—then
      exposes how every record was produced, evaluated, and deployed.
    </p>
    <div class="hero-actions">
      <a class="hero-button primary" href="context/use-cases/">
        See where it fits
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
      <a class="hero-button" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
        Try the live demo
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
      <a class="hero-button" href="research/paper/">Inspect the evidence</a>
    </div>
    <div class="hero-proof" aria-label="Project focus">
      <span>Long recordings</span>
      <span>Full activity records</span>
      <span>Web + Android</span>
    </div>
  </div>
  <div class="hero-visual">
    <div class="floating-badge badge-model">3-scale context</div>
    <div class="floating-badge badge-edge">Stable timeline</div>
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
        <div><strong>3 models</strong><span>short + long views</span></div>
        <div><strong>1 timeline</strong><span>final activity records</span></div>
      </div>
    </div>
  </div>
</section>

<span class="section-eyebrow">Start with the real problem</span>

## A sensor captures movement. People need an account of the session. {: .section-title}

<p class="section-lead">An hour of six-axis sensing produces 360,000 timestamps and 2.16 million channel readings. None of those rows says “badminton started at 09:02 and ended at 09:17.” The useful artifact is not another waveform—it is a small, reviewable set of activity records.</p>

<div class="story-grid">
  <article class="story-card story-card-input">
    <span class="story-card-kicker">What the device sees</span>
    <strong>Continuous motion, without meaning</strong>
    <div class="signal-token-list" aria-label="Six input channels">
      <span>ACC_X</span><span>ACC_Y</span><span>ACC_Z</span>
      <span>GYRO_X</span><span>GYRO_Y</span><span>GYRO_Z</span>
    </div>
    <p>Background movement, transitions, repeated actions, pauses, and sensor noise all arrive in the same stream.</p>
  </article>
  <article class="story-card story-card-output">
    <span class="story-card-kicker">What a person needs</span>
    <strong>A short, timestamped activity log</strong>
    <div class="record-list" aria-label="Illustrative activity records">
      <div><time>09:02–09:17</time><span>Badminton</span></div>
      <div><time>09:25–09:34</time><span>Jump rope</span></div>
      <div><time>09:41–09:53</time><span>Running</span></div>
    </div>
    <p>Each record answers three practical questions: what happened, when did it happen, and how long did it last?</p>
  </article>
</div>

<p class="story-caption">The times above are illustrative—not participant data. They show the translation this repository is designed to make.</p>

<div class="metric-strip context-metrics">
  <div class="metric"><strong>100 Hz</strong><span>continuous sampling</span></div>
  <div class="metric"><strong>6</strong><span>physical channels</span></div>
  <div class="metric"><strong>5</strong><span>foreground activities</span></div>
  <div class="metric"><strong>4 fields</strong><span>record output</span></div>
</div>

<span class="section-eyebrow">Use scenarios</span>

## Where would this pipeline be useful? {: .section-title}

<p class="section-lead">The same motion-to-record chain supports several concrete workflows, but the strength of the evidence differs by scenario. The cards below separate demonstrated research use from applications that still require task-specific validation.</p>

<div class="scenario-grid">
  <article class="scenario-card">
    <span class="scenario-tag established">Direct research fit</span>
    <h3>Evaluate long-session activity recognition</h3>
    <p>Compare systems using segment F1, boundary overlap, false positives per hour, event counts, and durations—not only isolated-window accuracy.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag scoped">Scoped prototype</span>
    <h3>Build an automatic workout diary</h3>
    <p>Turn controlled sessions containing the five supported activities into candidate records that a participant, coach, or researcher can review.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag established">Implemented path</span>
    <h3>Test research-to-phone deployment</h3>
    <p>Follow the same six-channel contract from a WT9011DCL-BT50 sensor over BLE to Android-side ONNX inference and a visible timeline.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag exploratory">Human-in-the-loop</span>
    <h3>Accelerate annotation and quality review</h3>
    <p>Use proposed segments to focus a reviewer on likely activity intervals and boundary errors; the predictions are not a substitute for ground truth.</p>
  </article>
</div>

<div class="paper-home-cta story-cta">
  <p>See one example session, who this system is for, what it returns, and what must be tested again before using it with a new device or population.</p>
  <a class="md-button md-button--primary" href="context/use-cases/">Read background & use cases</a>
</div>

<span class="section-eyebrow">The hidden difficulty</span>

## A correct window label can still become a wrong record. {: .section-title}

<p class="section-lead">Short-window probabilities may look convincing while brief confidence dips split one activity into several events, motion-like background creates false positives, or boundaries drift. Those mistakes directly change the session count, duration, and timeline.</p>

<figure class="paper-figure">
  <a class="pipeline-image-link" href="assets/manuscript-figures/fig01_window_to_record_gap.png" target="_blank" rel="noopener" aria-label="Open the full-resolution window-to-record gap figure">
    <img src="assets/manuscript-figures/fig01_window_to_record_gap.png" alt="Posterior trajectories, naive fragmented records, and the stabilized activity records produced by the Temporal Record Layer" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Paper Fig. 1. The final timeline step joins unstable short-window predictions into fewer, more reliable activity records.</figcaption>
</figure>

<div class="metric-strip paper-metrics">
  <div class="metric"><strong>137</strong><span>long recordings</span></div>
  <div class="metric"><strong>259.6 h</strong><span>continuous sensing</span></div>
  <div class="metric"><strong>0.89</strong><span>mean-user F1</span></div>
  <div class="metric"><strong>0.90</strong><span>micro-F1</span></div>
</div>

<div class="paper-home-cta">
  <p>The 0.89 and 0.90 values measure complete activity records on the fixed external test; they are not short-window accuracy. The paper page separates development analyses, final results, success cases, failure cases, and limitations.</p>
  <a class="md-button md-button--primary" href="research/paper/">Inspect the paper evidence</a>
</div>

<span class="section-eyebrow">How the system closes the gap</span>

## Four steps from a wrist sensor to a reviewable timeline {: .section-title}

<p class="section-lead">Each technical component has a practical job: preserve the original session, compare short and long views of the motion, build stable records, and make the result easy to check.</p>

<div class="feature-grid process-grid">
  <article class="feature-card process-card">
    <span class="process-step">01</span>
    <h3>Capture the whole session</h3>
    <p>Keep the six physical-unit ACC/GYRO channels and timestamps so the final boundaries remain anchored to the source recording.</p>
  </article>
  <article class="feature-card process-card">
    <span class="process-step">02</span>
    <h3>Compare short and long views</h3>
    <p>Use 3-, 5-, and 8-second models together: short windows help locate changes, while longer windows provide steadier activity context.</p>
  </article>
  <article class="feature-card process-card">
    <span class="process-step">03</span>
    <h3>Construct stable records</h3>
    <p>Choose the most useful time scale, reduce rapid label changes, join appropriate gaps, refine boundaries, and filter weak records. The paper calls these stages LBSA and TRL.</p>
  </article>
  <article class="feature-card process-card">
    <span class="process-step">04</span>
    <h3>Review or deploy</h3>
    <p>Export <code>user_id, category, start, end</code>, inspect the browser plots, or run the corresponding ONNX path on Android.</p>
  </article>
</div>

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="Open the full-resolution overall framework figure">
    <img src="assets/fig02_overall_framework.png" alt="Existing project framework figure showing the IMU stream, scale-specific CNN–BiLSTM models, LBSA fusion, temporal record layer, and segment records" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Repository figure: IMU stream → three window models → scale selection (LBSA) → timeline cleanup (TRL) → activity records. Select the image to view it at full resolution.</figcaption>
</figure>

<p class="pipeline-summary">The Python code, browser demo, and Android app follow the same basic rule: six IMU channels go in, and timestamped activity records come out. Saved settings, model files, intermediate probabilities, and segment-level tests make each step easier to reproduce and check.</p>

<span class="section-eyebrow">Evidence before expansion</span>

## What is demonstrated—and what is not yet established {: .section-title}

<div class="evidence-grid">
  <article class="evidence-card">
    <span class="evidence-kicker">Supported by this repository</span>
    <h3>A complete, inspectable research prototype</h3>
    <ul>
      <li>Five foreground activities in the evaluated long-session protocol.</li>
      <li>Fixed segment-level external testing on 37 recordings.</li>
      <li>Python reproduction, public synthetic demo, and Android field-test path.</li>
      <li>Success and failure timelines reported together.</li>
    </ul>
  </article>
  <article class="evidence-card caution">
    <span class="evidence-kicker">Requires new validation</span>
    <h3>Uses outside the tested setting</h3>
    <ul>
      <li>New devices, sensor placements, populations, or activity protocols.</li>
      <li>Clinical benefit, coaching quality, safety decisions, or production reliability.</li>
      <li>Highly interleaved sessions and separate counting of adjacent same-class events.</li>
      <li>Automatic labels without human review in consequential workflows.</li>
    </ul>
  </article>
</div>

<span class="section-eyebrow">Choose your depth</span>

## Start with the question you have now {: .section-title}

<div class="route-grid">
  <a class="route-card" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
    <span>2 minutes</span>
    <h3>Try a synthetic session</h3>
    <p>Run the tracked models and inspect signals, probabilities, records, and CSV output without installing anything.</p>
  </a>
  <a class="route-card" href="context/use-cases/">
    <span>Understand the setting</span>
    <h3>Read background & scenarios</h3>
    <p>See who would use this workflow, what decision it supports, and where human review or new validation is required.</p>
  </a>
  <a class="route-card" href="guide/pipeline/">
    <span>Technical path</span>
    <h3>Follow the architecture</h3>
    <p>Trace channel order, temporal scales, model structure, fusion, record construction, and output.</p>
  </a>
  <a class="route-card" href="deployment/android/">
    <span>Physical path</span>
    <h3>Build the Android demo</h3>
    <p>Connect the documented BLE sensor, inspect live signals, record CSV, and run on-device ONNX inference.</p>
  </a>
</div>

<div class="cta-panel">
  <div>
    <h3>Want the shortest path from a sample to the results?</h3>
    <p>Start with the live synthetic example, then compare its timeline with the paper’s fixed evaluation protocol.</p>
  </div>
  <a class="md-button md-button--primary" href="deployment/hugging-face/">Open the demo guide</a>
</div>
