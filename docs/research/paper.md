# Paper highlights

<p class="research-lead"><strong>An End-to-End Wearable IMU System for Segment-Level Activity Recognition via Multi-Scale Arbitration and a Temporal Record Layer</strong> studies wearable activity recognition as a complete sensing-to-record measurement chain. The central question is not only whether short windows are classified correctly, but whether a long workout becomes a reliable list of labeled records with usable boundaries, durations, and event counts.</p>

<div class="metric-strip paper-metrics">
  <div class="metric"><strong>137</strong><span>long-session recordings</span></div>
  <div class="metric"><strong>259.6 h</strong><span>continuous sensing</span></div>
  <div class="metric"><strong>0.89</strong><span>mean-user F1</span></div>
  <div class="metric"><strong>0.90</strong><span>micro-F1</span></div>
</div>

!!! note "How to read the evidence"

    The headline values are **segment-record metrics**, not window-classification
    accuracy. Models, fusion, and temporal policies were fixed before the 37
    independent external-test labels were used for final scoring. Internal
    development diagnostics and external-test results are identified separately
    below.

## The window-to-record gap

A locally plausible posterior trajectory can still produce the wrong record
list. Brief confidence dips may split one activity into two records, weak motion
may become a short false positive, and window-centered decisions may shift an
otherwise correct boundary. These errors directly bias activity counts,
durations, and timelines.

<figure class="paper-figure">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig01_window_to_record_gap.png" target="_blank" rel="noopener" aria-label="Open the full-resolution window-to-record gap figure">
    <img src="../../assets/manuscript-figures/fig01_window_to_record_gap.png" alt="Posterior trajectories, naive fragmented activity records, and the stabilized record list produced by the Temporal Record Layer" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Paper Fig. 1. Window posteriors can look reasonable while naive record extraction still creates false splits, short false positives, and shifted boundaries. TRL merges and stabilizes the final record list.</figcaption>
</figure>

## What the system contributes

<div class="research-grid">
  <article class="research-card">
    <span class="research-card-kicker">Local evidence</span>
    <h3>CNN + BiLSTM posterior generator</h3>
    <p>Multi-kernel one-dimensional convolutions capture local motion motifs while a bidirectional recurrent path models their evolution inside each 3 s, 5 s, or 8 s window.</p>
  </article>
  <article class="research-card">
    <span class="research-card-kicker">Scale uncertainty</span>
    <h3>Local-Boundary Scale Arbitration</h3>
    <p>LBSA retains longer-window stability in steady regions and raises the contribution of the 3 s branch near candidate transitions, where localization matters most.</p>
  </article>
  <article class="research-card">
    <span class="research-card-kicker">Record construction</span>
    <h3>Deterministic Temporal Record Layer</h3>
    <p>TRL exposes smoothing, constrained Viterbi decoding, same-class gap merging, boundary refinement, overlap handling, duration filtering, confidence clipping, and pruning as auditable steps.</p>
  </article>
</div>

The reported measurand is a variable-length record set
`(activity, start, end)`. A prediction is matched to at most one reference
segment of the same class when IoU is greater than 0.5. This makes fragmented,
merged, mistimed, and misclassified records visible in the score.

## Fixed evaluation protocol

| Role | Recordings | Purpose |
| --- | ---: | --- |
| Training | 80 | Model fitting and training-stage checkpoint selection |
| Development / calibration | 20 | Temporal-policy calibration, diagnostics, and split-separated selection |
| Independent external test | 37 | One final evaluation after the operating point was frozen |

The corpus contains about **46.8 million** valid 100 Hz ACC/GYRO samples and
five foreground activities: badminton, rope skipping, dumbbell fly, running,
and table tennis. The external set contains 114 labeled activity segments.

## Independent external-test result

<figure class="paper-figure portrait">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig04_external_variant_comparison.png" target="_blank" rel="noopener" aria-label="Open the full-resolution external-test comparison figure">
    <img src="../../assets/manuscript-figures/fig04_external_variant_comparison.png" alt="External-test mean-user F1 and micro-F1 together with false-positive and false-negative counts for five fixed system variants" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Paper Fig. 4. Fixed external-test operating points on the same 37 recordings. LBSA + TRL reaches the top rounded mean-user F1 and the lowest false-positive count among the three-scale variants.</figcaption>
</figure>

| Fixed variant | Mean-user F1 | 95% CI | Micro-F1 | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: |
| 5 s + 8 s + TRL | 0.88 | 0.80-0.94 | 0.88 | 98 / 11 / 16 |
| Three-model average + TRL | 0.88 | 0.80-0.94 | 0.89 | 98 / 9 / 16 |
| Three-model weighted + TRL | 0.89 | 0.81-0.94 | 0.89 | 99 / 9 / 15 |
| LBSA + relaxed Top-K | 0.88 | 0.80-0.95 | 0.88 | 103 / 17 / 11 |
| **LBSA + TRL** | **0.89** | **0.82-0.94** | **0.90** | **99 / 7 / 15** |

<div class="result-callout"><strong>Interpretation.</strong> Relaxed pruning recovers more true segments but also produces substantially more false positives. The final frozen LBSA + TRL setting retains the strongest rounded F1 while suppressing spurious records.</div>

### Per-class external-test outcomes

| Activity | Ground truth | TP / FP / FN | Precision | Recall | F1 | Matched IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Badminton | 32 | 26 / 3 / 6 | 0.90 | 0.81 | 0.85 | 0.87 |
| Rope skipping | 20 | 19 / 0 / 1 | 1.00 | 0.95 | 0.97 | 0.84 |
| Dumbbell fly | 20 | 19 / 1 / 1 | 0.95 | 0.95 | 0.95 | 0.78 |
| Running | 20 | 18 / 1 / 2 | 0.95 | 0.90 | 0.92 | 0.82 |
| Table tennis | 22 | 17 / 2 / 5 | 0.90 | 0.77 | 0.83 | 0.86 |

These descriptive class results show that the main residual errors are not
uniform: table tennis has the lowest recall, while rope skipping has the
strongest segment F1 in the fixed external set.

## Why multiple time scales help

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig05_multiscale_tsne_diagnostic.png" target="_blank" rel="noopener" aria-label="Open the full-resolution multi-scale t-SNE figure">
    <img src="../../assets/manuscript-figures/fig05_multiscale_tsne_diagnostic.png" alt="Three t-SNE plots of penultimate model embeddings for 3-second, 5-second, and 8-second windows across six activity states" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Paper Fig. 5. Independently embedded 3 s, 5 s, and 8 s representations show complementary qualitative structure across activities.</figcaption>
</figure>

Short windows preserve local transition detail but are more susceptible to
noise. Longer windows provide steadier context but blur boundaries. The t-SNE
panels are a **qualitative internal diagnostic**, not a headline performance
result; they motivate cross-scale arbitration but do not replace the fixed
segment-level external evaluation.

## What the Temporal Record Layer changes

<figure class="paper-figure portrait">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig06_outer_split_boundary_summary.png" target="_blank" rel="noopener" aria-label="Open the full-resolution TRL outer-split diagnostic figure">
    <img src="../../assets/manuscript-figures/fig06_outer_split_boundary_summary.png" alt="Repeated outer-split F1 distributions and cumulative Temporal Record Layer boundary quality and false-positive cost" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Paper Fig. 6. Split-separated diagnostics on a fixed 3 s posterior source. Temporal consistency and confidence clipping provide the clearest gains over the baseline record-construction policy.</figcaption>
</figure>

Across 50 random 10/10 development-user splits, the selected temporal policy
has mean outer-split F1 **0.913**, compared with **0.802** for the baseline
post-processing setting. In the cumulative boundary diagnostic, false
positives fall from **0.862 to 0.190 per recording hour**, while matched IoU
changes from **0.835 to 0.843**. The evidence supports a narrow conclusion:
most recovered record quality comes from temporal consistency control and
false-positive suppression, not from changing the fixed local predictor.

## Representative timelines: success and failure

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig07_representative_timeline_cases.png" target="_blank" rel="noopener" aria-label="Open the full-resolution representative timeline figure">
    <img src="../../assets/manuscript-figures/fig07_representative_timeline_cases.png" alt="Ground truth and four fixed temporal fusion variants for one successful and one partially failed long-session activity timeline" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Paper Fig. 7. The upper case shows the 3 s branch extending a table-tennis record enough to meet the IoU rule. The lower case preserves a real limitation: table tennis remains unmatched and later background motion becomes a badminton false positive.</figcaption>
</figure>

The paired cases are intentionally shown together. They demonstrate both the
benefit of boundary-sensitive multi-scale fusion and the remaining failure
modes when weak activity evidence and motion-like background intervals stress
the record assumptions.

## Physical demonstration and field tests

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig08_app_field_test.png" target="_blank" rel="noopener" aria-label="Open the full-resolution app field-test figure">
    <img src="../../assets/manuscript-figures/fig08_app_field_test.png" alt="Privacy-preserving action renderings paired with Android recognition screenshots for background and five target activities" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Paper Fig. 8. App-facing field-test examples for background motion, badminton, rope skipping, dumbbell fly, running, and table tennis.</figcaption>
</figure>

The physical chain pairs a WT9011DCL-BT50 BLE six-axis IMU with the Android
recognition app. The final selected-per-scale three-model configuration,
including LBSA and TRL, requires an average **2.1 seconds per user recording**
on the evaluated active device. This is slower than the compact single-model
diagnostic profile (0.21 seconds), but remains practical for offline session
analysis, pre-sync workout processing, or near-real-time record review.

??? info "Public-corpus portability checks - not leaderboard claims"

    The manuscript also reconnects TRL-style decoding to neural window
    posterior generators on HAR70+, WISDM-phone, PAMAP2, and OPPORTUNITY. These
    datasets do not match the long-session wrist-sport protocol, so the checks
    only test whether the temporal interface can be re-parameterized.

    | Dataset | Argmax F1 | TRL F1 | Argmax FP/h | TRL FP/h |
    | --- | ---: | ---: | ---: | ---: |
    | HAR70+ | 0.70 | 0.70 | 47 | 45 |
    | WISDM-phone | 0.05 | 0.37 | 83 | 5.7 |
    | PAMAP2 | 0.09 | 0.53 | 90 | 9.3 |
    | OPPORTUNITY | 0.29 | 0.29 | 320 | 110 |

## Scope and limitations

!!! warning "Use the claims at the level that was evaluated"

    - The evidence concerns **segment-record quality**, not clinical benefit,
      coaching quality, or safety decisions.
    - Generalization to new devices, populations, sensor placements, sport
      protocols, and annotation practices has not yet been established.
    - User-level statistical power is governed by 20 development/calibration
      recordings and 37 independent external-test recordings, not by the raw
      sample count alone.
    - The current system is best interpreted as a minute-scale workout-record
      generator. Highly interleaved sessions and separately counting adjacent
      same-class events remain difficult.

## Reproduce the software path

- [Run the public smoke test](../getting-started/quickstart.md)
- [Inspect the pipeline architecture](../guide/pipeline.md)
- [Review the segment-level evaluator](../guide/evaluation.md)
- [Build the Android demonstration](../deployment/android.md)
- [Check data and model-asset boundaries](../reference/assets.md)
