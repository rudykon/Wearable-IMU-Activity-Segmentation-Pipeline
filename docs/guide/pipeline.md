# Pipeline architecture

The project solves **temporal activity segmentation**, not isolated clip
classification. Its input is a long per-user sensor stream; its output is a
collection of timestamped foreground activity records.

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="../../assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="Open the full-resolution overall framework figure">
    <img src="../../assets/fig02_overall_framework.png" alt="Existing project framework figure showing the wearable IMU activity segmentation architecture" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Existing project figure covering signal input, scale-specific models, LBSA fusion, deterministic temporal decoding, and final segment records.</figcaption>
</figure>

## System contract

| Item | Contract |
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

## 3. Multi-scale windows

The same session is viewed at three temporal resolutions:

| Scale | Samples | Why it helps |
| --- | ---: | --- |
| 3 seconds | 300 | Captures short local motion signatures |
| 5 seconds | 500 | Balances local detail and activity context |
| 8 seconds | 800 | Stabilizes longer or repetitive actions |

All scales advance by one second, which makes their probability sequences
alignable before temporal decoding.

## 4. Window classifier

The practical `CombinedModel` is a six-class network:

1. parallel 1D convolution branches with kernels 3, 7, and 15;
2. concatenated multi-resolution feature maps;
3. a deeper CNN path with adaptive pooling;
4. a two-layer bidirectional LSTM path; and
5. a fused classification head.

The source also retains separate stage-one activity/background and stage-two
foreground classifiers for controlled experiments.

!!! info "Why both CNN and BiLSTM?"

    Convolution branches detect local waveform motifs at several receptive
    fields. The bidirectional recurrent branch encodes how those motifs evolve
    across a window. Their fused representation is more expressive than either
    path alone.

## 5. Probability alignment and LBSA

Each selected 3 s, 5 s, and 8 s checkpoint produces a class-probability stream.
The pipeline aligns these streams on the common one-second grid and applies
local-boundary scale arbitration (LBSA), allowing the preferred temporal scale
to change around uncertain transitions.

The ensemble configuration is explicit in:

~~~text
saved_models/ensemble_config.json
~~~

This keeps checkpoint selection and temporal-policy parameters reviewable.

## 6. Temporal decoding

Window predictions are not emitted directly as segments. They pass through a
structured temporal layer that can apply:

- multi-scale fusion;
- probability smoothing;
- Viterbi sequence decoding;
- boundary refinement;
- short-gap handling;
- overlap resolution;
- confidence filtering; and
- final Top-K or duration policies.

These operations reduce isolated label flicker and convert the regular
one-second probability grid into contiguous timestamp intervals.

## 7. Segment output

`imu_activity_pipeline.prediction_writer.DataOutput` writes records to an
Excel workbook:

~~~text
user_id, category, start, end
~~~

Background is useful internally for decoding but is excluded from foreground
submission records.

## Research and Android parity

| Python research pipeline | Android app |
| --- | --- |
| PyTorch checkpoints (`.pth`) | ONNX models (`.onnx`) |
| Pickled normalization assets | JSON normalization assets |
| Batch files from `data/signals/` | BLE history or a selected offline file |
| Experiment scripts and evaluator | Live views and on-device timeline |
| XLSX segment output | UI segments with confidence |

The runtimes differ, but both implement the same channel order, temporal scales,
class map, and segment-oriented interpretation.
