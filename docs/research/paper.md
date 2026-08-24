# Results

<p class="research-lead">The paper evaluates full activity records: class, start, end, duration, and count.</p>

<div class="metric-strip paper-metrics">
  <div class="metric"><strong>137</strong><span>recordings</span></div>
  <div class="metric"><strong>259.6 h</strong><span>sensor data</span></div>
  <div class="metric"><strong>0.89</strong><span>mean-user F1</span></div>
  <div class="metric"><strong>0.90</strong><span>micro-F1</span></div>
</div>

!!! note "Protocol"

    Scores measure complete records, not windows. Models and rules were frozen
    before the 37-recording external test.

## Record errors

Good window labels can still split events, shift boundaries, or create false alarms.

<figure class="paper-figure">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig01_window_to_record_gap.png" target="_blank" rel="noopener" aria-label="Open the full-resolution window-to-record gap figure">
    <img src="../../assets/manuscript-figures/fig01_window_to_record_gap.png" alt="Posterior trajectories, naive fragmented activity records, and the stabilized record list produced by the Temporal Record Layer" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Fig. 1. Window predictions can fragment the final record list.</figcaption>
</figure>

## Method

<div class="research-grid">
  <article class="research-card">
    <span class="research-card-kicker">Model</span>
    <h3>CNN–BiLSTM</h3>
    <p>Classifies 3-, 5-, and 8-second windows.</p>
  </article>
  <article class="research-card">
    <span class="research-card-kicker">Fusion</span>
    <h3>LBSA</h3>
    <p>Selects useful scales near and away from boundaries.</p>
  </article>
  <article class="research-card">
    <span class="research-card-kicker">Decoding</span>
    <h3>TRL</h3>
    <p>Smooths, joins, refines, and filters records.</p>
  </article>
</div>

Predictions match same-class labels one-to-one at IoU > 0.5. Splits, merges,
wrong times, and wrong classes reduce the score.

## Data split

| Role | Recordings | Purpose |
| --- | ---: | --- |
| Training | 80 | Model fitting and training-stage checkpoint selection |
| Development / calibration | 20 | Temporal-policy calibration, diagnostics, and split-separated selection |
| Independent external test | 37 | One final evaluation after the operating point was frozen |

The corpus has **46.8 million** valid samples and five activities. The external
set has 114 labeled segments.

## External test

<figure class="paper-figure portrait">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig04_external_variant_comparison.png" target="_blank" rel="noopener" aria-label="Open the full-resolution external-test comparison figure">
    <img src="../../assets/manuscript-figures/fig04_external_variant_comparison.png" alt="External-test mean-user F1 and micro-F1 together with false-positive and false-negative counts for five fixed system variants" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Fig. 4. Five fixed variants on the same 37 recordings.</figcaption>
</figure>

| Fixed variant | Mean-user F1 | 95% CI | Micro-F1 | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: |
| 5 s + 8 s + TRL | 0.88 | 0.80-0.94 | 0.88 | 98 / 11 / 16 |
| Three-model average + TRL | 0.88 | 0.80-0.94 | 0.89 | 98 / 9 / 16 |
| Three-model weighted + TRL | 0.89 | 0.81-0.94 | 0.89 | 99 / 9 / 15 |
| LBSA + relaxed Top-K | 0.88 | 0.80-0.95 | 0.88 | 103 / 17 / 11 |
| **LBSA + TRL** | **0.89** | **0.82-0.94** | **0.90** | **99 / 7 / 15** |

<div class="result-callout"><strong>Takeaway.</strong> LBSA + TRL keeps the top rounded F1 with fewer false alarms.</div>

### By activity

| Activity | Ground truth | TP / FP / FN | Precision | Recall | F1 | Matched IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Badminton | 32 | 26 / 3 / 6 | 0.90 | 0.81 | 0.85 | 0.87 |
| Rope skipping | 20 | 19 / 0 / 1 | 1.00 | 0.95 | 0.97 | 0.84 |
| Dumbbell fly | 20 | 19 / 1 / 1 | 0.95 | 0.95 | 0.95 | 0.78 |
| Running | 20 | 18 / 1 / 2 | 0.95 | 0.90 | 0.92 | 0.82 |
| Table tennis | 22 | 17 / 2 / 5 | 0.90 | 0.77 | 0.83 | 0.86 |

Table tennis has the lowest recall; rope skipping has the highest F1.

## Time scales

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig05_multiscale_tsne_diagnostic.png" target="_blank" rel="noopener" aria-label="Open the full-resolution multi-scale t-SNE figure">
    <img src="../../assets/manuscript-figures/fig05_multiscale_tsne_diagnostic.png" alt="Three t-SNE plots of penultimate model embeddings for 3-second, 5-second, and 8-second windows across six activity states" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Fig. 5. The three scales learn complementary structure.</figcaption>
</figure>

Short windows preserve boundaries; long windows add context. The t-SNE panels
are **qualitative diagnostics**, not performance evidence.

## TRL effect

<figure class="paper-figure portrait">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig06_outer_split_boundary_summary.png" target="_blank" rel="noopener" aria-label="Open the full-resolution TRL outer-split diagnostic figure">
    <img src="../../assets/manuscript-figures/fig06_outer_split_boundary_summary.png" alt="Repeated outer-split F1 distributions and cumulative Temporal Record Layer boundary quality and false-positive cost" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Fig. 6. Temporal consistency and confidence clipping drive the gains.</figcaption>
</figure>

Across 50 development splits, TRL raises mean F1 from **0.802 to 0.913**.
False positives fall from **0.862 to 0.190 per hour**; matched IoU changes from
**0.835 to 0.843**. These are development diagnostics, not the external result.

## Cases

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig07_representative_timeline_cases.png" target="_blank" rel="noopener" aria-label="Open the full-resolution representative timeline figure">
    <img src="../../assets/manuscript-figures/fig07_representative_timeline_cases.png" alt="Ground truth and four fixed temporal fusion variants for one successful and one partially failed long-session activity timeline" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Fig. 7. One success and one failure, shown together.</figcaption>
</figure>

Fusion improves one boundary; weak evidence still causes a miss and a false alarm.

## Android test

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig08_app_field_test.png" target="_blank" rel="noopener" aria-label="Open the full-resolution app field-test figure">
    <img src="../../assets/manuscript-figures/fig08_app_field_test.png" alt="Privacy-preserving action renderings paired with Android recognition screenshots for background and five target activities" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Fig. 8. Android tests for background and five activities.</figcaption>
</figure>

The Android app uses a WT9011DCL-BT50 BLE IMU. The full three-model pipeline
averages **2.1 seconds per recording** on the tested device.

??? info "Public datasets — portability only"

    These datasets do not match the wrist-sport protocol. Results only test
    whether the temporal interface can be re-parameterized.

    | Dataset | Argmax F1 | TRL F1 | Argmax FP/h | TRL FP/h |
    | --- | ---: | ---: | ---: | ---: |
    | HAR70+ | 0.70 | 0.70 | 47 | 45 |
    | WISDM-phone | 0.05 | 0.37 | 83 | 5.7 |
    | PAMAP2 | 0.09 | 0.53 | 90 | 9.3 |
    | OPPORTUNITY | 0.29 | 0.29 | 320 | 110 |

## Limits

!!! warning "Evaluated scope"

    - Results measure **segment records**, not clinical, coaching, or safety value.
    - New devices, users, placements, and protocols need new tests.
    - Evidence comes from 20 development and 37 external-test recordings.
    - Dense sessions and adjacent same-class events remain difficult.

## Reproduce

- [Run the public smoke test](../getting-started/quickstart.md)
- [Inspect the pipeline architecture](../guide/pipeline.md)
- [Review the segment-level evaluator](../guide/evaluation.md)
- [Build the Android demonstration](../deployment/android.md)
- [Check data and model-asset boundaries](../reference/assets.md)
