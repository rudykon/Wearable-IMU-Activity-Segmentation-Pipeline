# Training

Training produces calibrated window-level posterior models for the later LBSA
and TRL stages. The goal is stable class evidence across time, not a standalone
window-accuracy result.

## Protocol

Only the 80-record training split is used to fit model weights and select
checkpoints during training. The 20-record development/calibration split
supports temporal-policy calibration and diagnostics. The 37 external-test
recordings remain untouched until final scoring.

All retained models share the same architecture and training recipe. They differ
only in window length and random seed.

| Setting | Value |
| --- | --- |
| Windows | 3 s, 5 s, 8 s |
| Step | 1 s |
| Seeds per scale | 42, 123, 456 |
| Batch size | 256 |
| Optimizer | AdamW |
| Learning rate | \(1\times10^{-3}\) |
| Schedule | cosine annealing |
| Label smoothing | 0.1 |

## Model

The 1.41M-parameter classifier combines three 1D convolution branches with a
BiLSTM. Kernels 3, 7, and 15 capture motion at different receptive fields;
the BiLSTM adds order-sensitive context across the window. The output has six
classes: background plus five sports.

The backbone is deliberately compact. Its role is to provide reliable
posterior trajectories so the later analysis can isolate scale arbitration and
record construction.

## Objective

The training objective combines cross-entropy, focal, and triplet terms:

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{CE}}
+
\lambda_{\mathrm{focal}}\mathcal{L}_{\mathrm{focal}}
+
\lambda_{\mathrm{triplet}}\mathcal{L}_{\mathrm{triplet}},
\qquad
\lambda_{\mathrm{focal}}=0.2,
\quad
\lambda_{\mathrm{triplet}}=0.1.
\tag{1}
\]

- cross-entropy learns the six-class decision;
- focal loss emphasizes harder examples;
- triplet loss improves embedding separation;
- class-balanced sampling limits dominance by frequent classes.

Augmentation combines amplitude scaling, Gaussian noise, random shift, time
warp, and Mixup. Label smoothing and Mixup also reduce over-confident posterior
spikes near ambiguous boundaries.

## Retained models

Nine models are trained: three seeds at each scale. The table reports
training-phase internal-validation F1; it is not external-test evidence.

| Scale | Seed | Best epoch | Internal-val F1 |
| --- | ---: | ---: | ---: |
| **3 s** | **42** | **53** | **0.81** |
| 3 s | 123 | 37 | 0.80 |
| 3 s | 456 | 73 | 0.81 |
| 5 s | 42 | 37 | 0.81 |
| **5 s** | **123** | **38** | **0.86** |
| 5 s | 456 | 32 | 0.81 |
| 8 s | 42 | 31 | 0.85 |
| **8 s** | **123** | **38** | **0.85** |
| 8 s | 456 | 27 | 0.85 |

The final stack keeps one selected model per scale: seed 42 at 3 s and seed 123
at 5 s and 8 s. It does not average seeds within a scale. Cross-scale fusion is
performed later by LBSA.

## Interpretation

Longer windows show a stronger mean internal-validation trend, but they blur
transitions. Shorter windows localize changes better but are noisier. Training
therefore supplies complementary posterior sources; it does not decide the
final record policy.

!!! note "Reporting boundary"

    Internal-validation results select assets. They must not be mixed with the
    fixed external-test segment scores reported on the [Results](../research/paper.md) page.

??? info "Reproduce"

    The default training entry point is `python train.py`. Environment setup,
    authorized data placement, and saved-asset checks are documented in
    [Quickstart](../getting-started/quickstart.md) and [Assets](../reference/assets.md).
