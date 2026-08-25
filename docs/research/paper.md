# Results

<p class="research-lead">The primary analysis evaluates complete activity records on an independent external test set.</p>

<div class="metric-strip metric-strip--three">
  <div class="metric"><strong>37</strong><span>test recordings</span></div>
  <div class="metric"><strong>114</strong><span>reference segments</span></div>
  <div class="metric"><strong>0.90</strong><span>micro-F1</span></div>
</div>

## Evaluation protocol

Predictions are matched one-to-one with same-class reference segments at IoU > 0.5. This record-level protocol penalizes missed and false activities, fragmentation, merging, wrong classes, and shifted boundaries.

| Data role | Recordings | Use |
| --- | ---: | --- |
| Training | 80 | Model fitting and training-stage checkpoint selection |
| Development / calibration | 20 | Temporal-policy calibration and diagnostics |
| Independent external test | 37 | One final evaluation after all choices were frozen |

External labels were not used to select checkpoints, fusion rules, TRL parameters, or reported variants. Mean-user F1 weights each user equally; micro-F1 pools TP, FP, and FN over the complete set.

## External test

<figure class="paper-figure portrait">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig04_external_variant_comparison.png" target="_blank" rel="noopener" aria-label="Open the full-resolution external-test comparison figure">
    <img src="../../assets/manuscript-figures/fig04_external_variant_comparison.png" alt="External-test mean-user F1 and micro-F1 with false-positive and false-negative counts for five fixed variants" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Five fixed variants evaluated on the same 37 recordings.</figcaption>
</figure>

| Fixed variant | Mean-user F1 | 95% CI | Micro-F1 | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: |
| 5 s + 8 s + TRL | 0.88 | 0.80–0.94 | 0.88 | 98 / 11 / 16 |
| Three-model average + TRL | 0.88 | 0.80–0.94 | 0.89 | 98 / 9 / 16 |
| Three-model weighted + TRL | 0.89 | 0.81–0.94 | 0.89 | 99 / 9 / 15 |
| LBSA + relaxed Top-K | 0.88 | 0.80–0.95 | 0.88 | 103 / 17 / 11 |
| **LBSA + TRL** | **0.89** | **0.82–0.94** | **0.90** | **99 / 7 / 15** |

<div class="result-callout"><strong>Primary result.</strong> LBSA + TRL retains the top rounded F1 while producing the fewest false-positive records among the compared variants.</div>

## By activity

| Activity | Reference segments | TP / FP / FN | Precision | Recall | F1 | Matched IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Badminton | 32 | 26 / 3 / 6 | 0.90 | 0.81 | 0.85 | 0.87 |
| Rope skipping | 20 | 19 / 0 / 1 | 1.00 | 0.95 | 0.97 | 0.84 |
| Dumbbell fly | 20 | 19 / 1 / 1 | 0.95 | 0.95 | 0.95 | 0.78 |
| Running | 20 | 18 / 1 / 2 | 0.95 | 0.90 | 0.92 | 0.82 |
| Table tennis | 22 | 17 / 2 / 5 | 0.90 | 0.77 | 0.83 | 0.86 |

Rope skipping has the highest F1. Table tennis has the lowest recall, indicating that weak or ambiguous evidence still causes missed records.

## Representative cases

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig07_representative_timeline_cases.png" target="_blank" rel="noopener" aria-label="Open the full-resolution representative timeline figure">
    <img src="../../assets/manuscript-figures/fig07_representative_timeline_cases.png" alt="Ground truth and fixed temporal fusion variants for one successful and one partially failed activity timeline" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">One successful case and one partial failure.</figcaption>
</figure>

Scale arbitration improves a boundary in the successful case. The failure case retains a miss and a false alarm where posterior evidence is weak; TRL cannot reconstruct activity evidence that the classifiers never provide.

## Limitations

!!! warning "Evaluated scope"

    - Results measure segment-record quality, not clinical, coaching, or safety value.
    - Evidence covers five activities under the studied device, placement, and protocol.
    - New devices, placements, users, activities, and deployment conditions require new validation.
    - Dense sessions and adjacent same-class events remain difficult.

Development diagnostics, public-dataset portability checks, and Android implementation evidence are reported separately in [Supplementary analyses](supplementary.md). Reproduction commands and asset boundaries are collected on the [Reproduce](../reproduce.md) page.
