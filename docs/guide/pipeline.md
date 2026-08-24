# Pipeline architecture

This page explains how the system changes a long sensor recording into a short
activity log. It does more than label separate clips: it must decide how many
activities occurred and where each one started and ended. This task is called
**temporal activity segmentation**.

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="../../assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="Open the full-resolution overall framework figure">
    <img src="../../assets/fig02_overall_framework.png" alt="Existing project framework figure showing the wearable IMU activity segmentation architecture" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Project overview: read six IMU channels, examine 3-, 5-, and 8-second windows, combine their predictions, clean up the timeline, and output activity records.</figcaption>
</figure>

## Inputs and outputs at a glance

| Item | What the system expects or returns |
| --- | --- |
| Sampling | 100 Hz accelerometer and gyroscope |
| Model input | `(window, 6)` physical-unit IMU channels |
| Window scales | 3 s (300 samples), 5 s (500), 8 s (800) |
| Window step | 1 s (100 samples) |
| Internal classes | background + five foreground activities |
| Public output | `user_id, category, start, end` |
| Evaluation | same-class one-to-one matching at IoU > 0.5 |

## 1. Signal ingestion

`imu_activity_pipeline.signal_file_reader.DataReader` loads each tab-separated
`.txt` file in an input directory and keys the resulting sessions by file
stem. The model uses:

~~~text
ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z
~~~

Millisecond timestamps in `ACC_TIME` anchor windows and final boundaries to
the source recording.

## 2. Preprocessing

The research and Android paths preserve the same physical-channel order and
normalization contract. The temporal pipeline includes:

- low-pass Butterworth filtering;
- fixed channel order;
- scale-specific normalization parameters; and
- timestamp-preserving window construction.

Keeping normalization assets beside their corresponding checkpoint is
essential. A checkpoint and a normalization file from different training runs
are not interchangeable.

## 3. Look at short and long windows

The same session is viewed at three temporal resolutions:

| Scale | Samples | Why it helps |
| --- | ---: | --- |
| 3 seconds | 300 | Captures short local motion signatures |
| 5 seconds | 500 | Balances local detail and activity context |
| 8 seconds | 800 | Stabilizes longer or repetitive actions |

All scales advance by one second, which makes their probability sequences
alignable before temporal decoding.

## 4. Classify each window

The practical `CombinedModel` is a six-class network:

1. parallel 1D convolution branches with kernels 3, 7, and 15;
2. concatenated multi-resolution feature maps;
3. a deeper CNN path with adaptive pooling;
4. a two-layer bidirectional LSTM path; and
5. a fused classification head.

The source also retains separate stage-one activity/background and stage-two
foreground classifiers for controlled experiments.

!!! info "Why both CNN and BiLSTM?"

    The CNN looks for short waveform patterns. The BiLSTM looks at how those
    patterns change from the beginning to the end of a window. Combining them
    gives the model both local detail and within-window context.

## 5. Combine the 3-, 5-, and 8-second views

Each model produces one probability timeline. The pipeline aligns all three on
the same one-second grid. It then gives more influence to the time scale that is
most useful at each point—for example, a shorter window near a possible change
and a longer window during steady movement. The paper calls this rule
**Local-Boundary Scale Arbitration (LBSA)**.

The ensemble configuration is explicit in:

~~~text
saved_models/ensemble_config.json
~~~

This file records which models and timeline settings were used, so an
experiment can be checked and repeated.

## 6. Turn window guesses into continuous records

The system does not turn every one-second prediction directly into a record.
Instead, a final timeline stage can:

- multi-scale fusion;
- probability smoothing;
- Viterbi sequence decoding;
- boundary refinement;
- short-gap handling;
- overlap resolution;
- confidence filtering; and
- final Top-K or duration policies.

Together, these operations reduce rapid label changes and turn a grid of
short-window guesses into continuous start-to-end activity periods. The paper
calls this stage the **Temporal Record Layer (TRL)**.

## 7. Segment output

`imu_activity_pipeline.prediction_writer.DataOutput` writes records to an
Excel workbook:

~~~text
user_id, category, start, end
~~~

Background is useful internally for decoding but is excluded from foreground
submission records.

## How the Python and Android versions stay aligned

| Python research pipeline | Android app |
| --- | --- |
| PyTorch checkpoints (`.pth`) | ONNX models (`.onnx`) |
| Pickled normalization assets | JSON normalization assets |
| Batch files from `data/signals/` | BLE history or a selected offline file |
| Experiment scripts and evaluator | Live views and on-device timeline |
| XLSX segment output | UI segments with confidence |

The software formats differ, but both versions use the same channel order,
window lengths, activity names, and start-to-end record format.
