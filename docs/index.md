---
hide:
  - toc
---

<section class="home-hero">
  <p class="hero-kicker">Wearable IMU activity records</p>
  <h1 class="paper-title">An End-to-End Wearable IMU System for Segment-Level Activity Recognition via Multi-Scale Arbitration and a Temporal Record Layer</h1>
  <p class="hero-lead">A multi-scale wrist-IMU pipeline that converts continuous 100 Hz signals into timestamped activity records.</p>
  <div class="hero-actions">
    <a class="hero-button primary" href="deployment/hugging-face/">Demo</a>
    <a class="hero-button" href="guide/pipeline/">Method</a>
    <a class="hero-button" href="research/paper/">Results</a>
  </div>
</section>

## Problem and output

Window-level predictions do not directly provide reliable activity records. A long-session system must recover the activity class, event count, duration, and boundaries from continuous sensor streams.

<div class="record-transform" aria-label="System input and output">
  <code>6-channel wrist IMU stream</code>
  <span aria-hidden="true">→</span>
  <code>{activity, start, end}</code>
</div>

The evaluated task covers five sports recorded from a 100 Hz wrist accelerometer and gyroscope. Background motion supports decoding but is not emitted as a workout record.

## Method

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="Open the full-resolution framework figure">
    <img src="assets/fig02_overall_framework.png" alt="Overall framework from wrist IMU input through three scale-specific models, LBSA, and TRL to activity records" loading="eager" decoding="async">
  </a>
  <figcaption class="pipeline-caption">IMU stream → 3/5/8-second posteriors → LBSA → TRL → activity records.</figcaption>
</figure>

1. **Multi-scale posterior models** analyze 3-, 5-, and 8-second views with scale-specific CNN–BiLSTMs.
2. **Local-Boundary Scale Arbitration (LBSA)** emphasizes shorter evidence near transitions and longer context in stable regions.
3. **Temporal Record Layer (TRL)** smooths, decodes, merges, refines, and filters the fused timeline into deterministic records.

[Read the full method](guide/pipeline.md){ .md-button }

## Evidence

<div class="metric-strip metric-strip--three">
  <div class="metric"><strong>259.6 h</strong><span>sensor data</span></div>
  <div class="metric"><strong>37 / 114</strong><span>test recordings / segments</span></div>
  <div class="metric"><strong>0.90</strong><span>micro-F1</span></div>
</div>

| Fixed operating point | Mean-user F1 | Micro-F1 | TP / FP / FN |
| --- | ---: | ---: | ---: |
| **LBSA + TRL** | **0.89** | **0.90** | **99 / 7 / 15** |

| Evaluated scope | Requires new validation |
| --- | --- |
| Five activities under the studied device, placement, and protocol | New devices, placements, users, activities, and deployment conditions |

[Inspect the results and failure cases](research/paper.md){ .md-button }

## Reproduce

| Resource | Purpose |
| --- | --- |
| [Demo](deployment/hugging-face.md) | Run a synthetic example and inspect the record output |
| [Models](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline) | Download verified PyTorch and ONNX weights |
| [Quickstart](getting-started/quickstart.md) | Verify the public package and run authorized-data inference |
| [Android](deployment/android.md) | Build the BLE and ONNX research prototype |

Participant recordings are not distributed on GitHub. The [reproduction guide](reproduce.md) separates public verification from workflows that require authorized data.
