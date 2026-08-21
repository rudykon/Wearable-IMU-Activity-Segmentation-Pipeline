# Evaluation

The evaluator measures the quality of **segments**, not just per-window class
accuracy.

!!! info "Paper evidence"

    The [paper highlights](../research/paper.md) page separates the fixed
    37-recording external-test result from internal TRL diagnostics and shows
    the corresponding comparison, boundary, and timeline figures.

## Run evaluation

For the default external split:

~~~bash
python evaluate.py --split external_test
~~~

For a named prediction workbook:

~~~bash
python evaluate.py --split internal_eval --predictions predictions_internal_eval.xlsx
~~~

Required files include the prediction workbook and the matching annotation CSV:

~~~text
predictions_internal_eval.xlsx
data/annotations/internal_eval_annotations.csv
~~~

## Matching rule

A predicted segment is eligible to match a reference segment only when:

1. both segments belong to the same user;
2. both have the same activity category; and
3. their intersection-over-union exceeds 0.5.

Matching is one-to-one, so one prediction cannot explain multiple reference
segments and vice versa.

For prediction interval `P` and reference interval `G`:

> **IoU(P, G) = duration(P ∩ G) / duration(P ∪ G)**

The evaluator then reports segment-level precision, recall, and F1:

> **F1 = 2 × precision × recall / (precision + recall)**

The project uses mean user-level segmental F1 as its default summary, preventing
users with very long sessions from automatically dominating the score.

## Interpreting errors

| Failure mode | Metric effect | Typical diagnostic |
| --- | --- | --- |
| Missed activity | False negative | Recall falls |
| Spurious activity | False positive | Precision falls |
| Correct class, poor boundary | IoU may fail | Both precision and recall fall |
| Fragmented prediction | Extra unmatched pieces | Precision falls |
| Merged adjacent activities | One-to-one match conflict | Recall and precision may fall |
| Wrong activity class | No eligible match | Both precision and recall fall |

## Evaluation discipline

- Use `internal_eval` for post-processing calibration.
- Reserve `external_test` for the intended final evaluation workflow.
- Record the checkpoint hashes, normalization assets, ensemble configuration,
  and any changed policy thresholds.
- Compare temporal policies with fixed window probabilities when possible; this
  isolates decoder changes from model retraining.
- Do not infer performance from the website. Run the released evaluator on the
  authorized split and report the exact asset/configuration version.

## Experiment outputs

The reproducibility wrapper writes evaluation and diagnostic material under:

~~~text
experiments/results/
experiments/figures/
experiments/logs/
~~~

These are generated outputs and remain local unless explicitly curated for a
release.
