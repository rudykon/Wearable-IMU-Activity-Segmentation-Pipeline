# Inference

Inference is the posterior-to-record stage of the method. It applies the three
selected window models to a complete session, fuses their probability
trajectories, and emits activity segments.

## Posteriors

The 3 s, 5 s, and 8 s models advance at one-second intervals. Each produces a
six-class posterior matrix: background plus five sports. The 5 s and 8 s
matrices are interpolated onto the 3 s reference grid before fusion.

The scales provide complementary evidence:

| Scale | Strength | Weakness |
| --- | --- | --- |
| 3 s | transition localization | noisier stable regions |
| 5 s | balance of detail and context | moderate boundary blur |
| 8 s | stable repetitive-motion context | weakest boundary precision |

## Arbitration

LBSA begins with stable-region weights `(0.20, 0.35, 0.45)`. Around class
changes in the 3 s trajectory, the weights move toward `(0.50, 0.27, 0.23)`.
The adjustment is local: long-window support remains present while the short
branch receives enough weight to sharpen a possible boundary.

## Record decoding

TRL is applied once to the fused matrix:

1. smooth each class trajectory;
2. decode a consistent state path with constrained Viterbi;
3. extract contiguous foreground intervals;
4. merge same-class gaps shorter than 60 s;
5. refine boundaries within ±15 s using acceleration energy;
6. resolve overlaps;
7. remove records shorter than 180 s;
8. apply Top-K and confidence pruning.

The final paper operating point uses Top-K 3 and confidence ≥ 0.45. These
thresholds reflect minute-scale workout records and must be recalibrated for a
different activity protocol.

<figure class="paper-figure">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig01_window_to_record_gap.png" target="_blank" rel="noopener" aria-label="Open the full-resolution window-to-record figure">
    <img src="../../assets/manuscript-figures/fig01_window_to_record_gap.png" alt="Window posteriors, fragmented naive records, and stabilized TRL records" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">TRL reduces false splits, short false records, and boundary drift.</figcaption>
</figure>

## Records

Each foreground output contains:

| Field | Meaning |
| --- | --- |
| `user_id` | recording identifier |
| `category` | one of five activities |
| `start` | segment start in milliseconds |
| `end` | segment end in milliseconds |

Background supports decoding but is not emitted as an activity record.

## Complexity

Smoothing and segment refinement are linear in the number of windows. Viterbi
decoding is `O(TC²)`; with six classes, neural forward passes dominate runtime.
For fixed inputs and parameters, the record layer is deterministic and every
merge, trim, and filter has an explicit interpretation.

??? info "Run the software"

    The default entry point is `python run_inference.py`. Input paths and the
    Python interface are documented in [Quickstart](../getting-started/quickstart.md)
    and the [API reference](../reference/api.md).
