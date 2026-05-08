# Model Card

## Model Summary

This repository includes ONNX model weights for real-time activity recognition
from WT9011DCL-BT50 BLE IMU data. The Android app loads the weights directly
from `app/src/main/assets/` and runs inference on-device with ONNX Runtime.

The default recognition pipeline uses selected 3s, 5s, and 8s models, then
applies multi-scale fusion and temporal post-processing in
`MotionClassifier.kt`.

## Intended Use

- Real-time motion recognition from a wrist-mounted WT9011DCL-BT50 IMU.
- Android on-device demo and competition prototype.
- Research, teaching, and reproducible app-level evaluation of the bundled
  motion-recognition pipeline.

This model card does not claim production safety, clinical validity, or
cross-device generalization.

## Input

- Sampling rate: 100 Hz.
- Channels: 6 physical-unit IMU channels.
- Channel order: `accX`, `accY`, `accZ`, `gyroX`, `gyroY`, `gyroZ`.
- Window sizes:
  - 3 seconds: 300 samples.
  - 5 seconds: 500 samples.
  - 8 seconds: 800 samples.
- Step size: 100 samples, approximately 1 second.

## Output Classes

The classifier outputs 6 classes:

| Index | Chinese | English |
| --- | --- | --- |
| 0 | 无活动 | No activity |
| 1 | 羽毛球 | Badminton |
| 2 | 跳绳 | Jump rope |
| 3 | 飞鸟 | Fly |
| 4 | 跑步 | Running |
| 5 | 乒乓球 | Table tennis |

## Runtime Pipeline

The Android implementation:

1. Accumulates complete session IMU history.
2. Applies zero-phase Butterworth filtering.
3. Runs the selected 3s, 5s, and 8s ONNX models when enough data is available.
4. Aligns probabilities across scales.
5. Applies local-boundary scale arbitration (LBSA).
6. Applies probability smoothing.
7. Runs Viterbi decoding over the session.
8. Produces filtered activity segments with confidence values.

The fallback path loads `hand_motion.onnx` and `norm_params.json` if selected
multi-scale assets are replaced or unavailable.

## Bundled Assets

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `app/src/main/assets/combined_model_3s_seed42.onnx` | Selected 3s model | `5adb8807bfc737e11ee40cca0af0690c22fdd8d29b5c5aaa35c3e83f9f646839` |
| `app/src/main/assets/combined_model_5s_seed123.onnx` | Selected 5s model | `d812c5fc04df6c1ca249e3cec8c977a28c2de8036736c134210b3173490f2681` |
| `app/src/main/assets/combined_model_8s_seed123.onnx` | Selected 8s model | `75fb093eca823003c6f5e44b0a0363f24b04659ab2ae08ddb08c334481087f8d` |
| `app/src/main/assets/hand_motion.onnx` | Legacy fallback model | `c41f8c7fab431d407504eba0141976047f5f010597e7fa8e0a6b57872f8f9456` |
| `app/src/main/assets/norm_params_3s.json` | 3s normalization parameters | `21d38c29325d691e49a5835254dbf1b08fb216f2e4c9af0cb31a39412e1d98c9` |
| `app/src/main/assets/norm_params_5s.json` | 5s normalization parameters | `180b60e4306972072e07fd29a7c2065bab47ea8fe6daaa3c2c7eb4942b6f4ad2` |
| `app/src/main/assets/norm_params_8s.json` | 8s normalization parameters | `7416395acceafc45803d8a63cc79919fc203970a061e5bc16d00337822418b7e` |
| `app/src/main/assets/norm_params.json` | Fallback normalization parameters | `396daf64b26765910065c0a46535e1595b12a12e13fb65485f2edfb29141cc3b` |

## Known Limitations

- Performance depends on sensor placement, device calibration, sampling
  stability, and whether the runtime scenario matches the model's training
  conditions.
- The app assumes 100 Hz input and the channel order listed above.
- The current repository includes inference assets and app code, not training
  data or training scripts.
- Validation metrics should be reported separately if this model is used in a
  paper, benchmark, or competition submission.

## License

The bundled model weights and normalization files are distributed under
`WEIGHTS_LICENSE`. The application source code is distributed under `LICENSE`.
