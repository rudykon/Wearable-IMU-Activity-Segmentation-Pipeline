# Evaluation

The study evaluates complete activity records. Window accuracy alone cannot
measure event count, false alarms, fragmentation, or boundary error.

## Protocol

| Split | Role in evaluation |
| --- | --- |
| Training, 80 recordings | fit posterior models |
| Development / calibration, 20 | tune and diagnose LBSA/TRL policies |
| External test, 37 | report the frozen operating point once |

The external set contains 114 labeled segments. Its labels are not used to
select checkpoints, fusion variants, temporal parameters, or table rows.
Development analyses and external results must therefore be reported separately.

## Matching

A predicted segment \(P\) can match a reference segment \(G\) only when both
belong to the same recording, have the same activity class, and satisfy
\(\operatorname{IoU}(P,G)>0.5\). Matching is one-to-one.

\[
\operatorname{IoU}(P,G)
=
\frac{
\operatorname{dur}(P\cap G)
}{
\operatorname{dur}(P\cup G)
}.
\tag{1}
\]

<p class="equation-context">The numerator is the overlapping duration; the denominator is the total duration covered by either interval.</p>

This rule penalizes wrong classes, shifted boundaries, splits, and merges in the
same record-level framework.

## Metrics

| Metric | What it measures |
| --- | --- |
| Mean-user F1 | segment F1 averaged equally across users |
| Micro-F1 | F1 from pooled TP, FP, and FN counts |
| Matched IoU | overlap quality of true-positive records |
| Start / end MAE | boundary error in seconds |
| Duration error | absolute activity-time error in seconds |
| FP/hour | false records per recording hour |

For each user, precision \(P\), recall \(R\), and F1 are computed from matched
and unmatched segments:

\[
\begin{aligned}
P &= \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},
&
R &= \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},
\\[4pt]
\mathrm{F1} &= \frac{2PR}{P+R}.
\end{aligned}
\tag{2}
\]

If the evaluation contains \(U\) users, the headline mean-user score gives each
user equal weight:

\[
\mathrm{F1}_{\mathrm{mean\text{-}user}}
=
\frac{1}{U}
\sum_{u=1}^{U}\mathrm{F1}_{u}.
\tag{3}
\]

Mean-user F1 prevents users with longer recordings from automatically
dominating the headline score. Micro-F1 instead pools TP, FP, and FN over all
users before applying Equation (2), preserving the global event-count view.

## Error reading

| Record error | Count effect | Boundary effect |
| --- | --- | --- |
| Missed activity | FN | no matched boundary |
| False activity | FP | no reference boundary |
| Fragmented activity | extra FP pieces | unstable starts and ends |
| Merged activities | one-to-one conflict | overlong duration |
| Correct class, shifted interval | may become FP + FN | larger boundary error |
| Wrong class | FP + FN | no eligible match |

## Evidence levels

The repeated 10/10 development splits isolate TRL behavior on fixed 3 s
posteriors. They are mechanism diagnostics. The final LBSA + TRL result is a
fixed three-model operating point evaluated on the 37-recording external set.
The two evidence levels answer different questions and are not interchangeable.

The headline external result is mean-user F1 **0.89**, micro-F1 **0.90**, and
TP/FP/FN **99/7/15**. See [Results](../research/paper.md) for variant tables,
confidence intervals, class outcomes, and failure cases.

!!! warning "Claim boundary"

    These metrics measure segment-record quality. They do not establish
    clinical benefit, coaching quality, safety value, or transfer to new
    devices and populations.

??? info "Reproduce"

    The evaluator entry point is `python evaluate.py --split external_test`.
    Run it only with the matching authorized labels and frozen model/policy
    assets documented in [Assets](../reference/assets.md).
