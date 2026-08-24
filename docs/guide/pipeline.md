# Pipeline

The pipeline turns a long IMU recording into timestamped activity records.

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="../../assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="Open the full-resolution overall framework figure">
    <img src="../../assets/fig02_overall_framework.png" alt="Existing project framework figure showing the wearable IMU activity segmentation architecture" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Six channels → three window scales → fusion → timeline → records.</figcaption>
</figure>

## I/O

| Item | What the system expects or returns |
| --- | --- |
| Sampling | 100 Hz accelerometer and gyroscope |
| Model input | `(window, 6)` physical-unit IMU channels |
| Window scales | 3 s (300 samples), 5 s (500), 8 s (800) |
| Window step | 1 s (100 samples) |
| Internal classes | background + five foreground activities |
| Public output | `user_id, category, start, end` |
| Evaluation | same-class one-to-one matching at IoU > 0.5 |

## 1. Read

`DataReader` loads tab-separated `.txt` sessions. Required channels:

~~~text
ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z
~~~

`ACC_TIME` stores millisecond timestamps.

## 2. Preprocessing

Python and Android use the same channel order and normalization:

- low-pass Butterworth filtering;
- fixed channel order;
- scale-specific normalization parameters; and
- timestamp-preserving window construction.

Keep each checkpoint with its matching normalization file.

## 3. Window

| Scale | Samples | Why it helps |
| --- | ---: | --- |
| 3 seconds | 300 | Captures short local motion signatures |
| 5 seconds | 500 | Balances local detail and activity context |
| 8 seconds | 800 | Stabilizes longer or repetitive actions |

All scales advance by one second.

## 4. Classify

`CombinedModel` has six classes and five parts:

1. parallel 1D convolution branches with kernels 3, 7, and 15;
2. concatenated multi-resolution feature maps;
3. a deeper CNN path with adaptive pooling;
4. a two-layer bidirectional LSTM path; and
5. a fused classification head.

The source also keeps two-stage classifiers for experiments.

!!! info "CNN + BiLSTM"

    CNN captures local patterns; BiLSTM captures their order.

## 5. Fuse

The three probability timelines share a one-second grid. **LBSA** favors short
windows near changes and long windows during steady motion.

The ensemble configuration is explicit in:

~~~text
saved_models/ensemble_config.json
~~~

It records the selected models and timeline settings.

## 6. Decode

The **Temporal Record Layer (TRL)** applies:

- multi-scale fusion;
- probability smoothing;
- Viterbi sequence decoding;
- boundary refinement;
- short-gap handling;
- overlap resolution;
- confidence filtering; and
- final Top-K or duration policies.

The result is a set of continuous activity periods.

## 7. Export

`DataOutput` writes an Excel workbook:

~~~text
user_id, category, start, end
~~~

Background is not exported.

## Python and Android

| Python research pipeline | Android app |
| --- | --- |
| PyTorch checkpoints (`.pth`) | ONNX models (`.onnx`) |
| Pickled normalization assets | JSON normalization assets |
| Batch files from `data/signals/` | BLE history or a selected offline file |
| Experiment scripts and evaluator | Live views and on-device timeline |
| XLSX segment output | UI segments with confidence |

Both use the same channels, windows, labels, and record format.
