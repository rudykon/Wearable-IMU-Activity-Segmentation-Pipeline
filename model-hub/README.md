---
license: apache-2.0
library_name: pytorch
tags:
  - pytorch
  - onnx
  - wearable
  - imu
  - activity-recognition
  - time-series
---

# Wearable IMU Activity Segmentation Pipeline

Public inference assets for the multi-scale wearable IMU activity-segmentation
pipeline. The repository contains the selected 3 s, 5 s, and 8 s PyTorch
checkpoints used by the Python pipeline and their ONNX exports used by the
Android app.

## Model

The classifier combines a compact one-dimensional CNN with a bidirectional
LSTM. Three temporal views produce six-class posteriors. Local-Boundary Scale
Arbitration (LBSA) fuses the views, and the Temporal Record Layer (TRL) converts
the posterior sequence into activity records.

Input is a 100 Hz wrist-IMU stream with six channels in this order:
`ACC_X`, `ACC_Y`, `ACC_Z`, `GYRO_X`, `GYRO_Y`, `GYRO_Z`.

| Index | Class |
| ---: | --- |
| 0 | Background |
| 1 | Badminton |
| 2 | Jump rope |
| 3 | Fly |
| 4 | Running |
| 5 | Table tennis |

## Files

- `saved_models/`: selected PyTorch checkpoints, normalization parameters, and
  the ensemble configuration.
- `android_realtime_app/app/src/main/assets/`: selected ONNX exports and the
  Android normalization parameters.
- `model-assets.json`: file sizes and SHA-256 checksums.

The 3 s fallback files duplicate the selected 3 s model for backward
compatibility. The manifest makes that relationship explicit and provides an
integrity check for every published asset.

## Download

Download the Python assets into a source checkout:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="config-h/Wearable-IMU-Activity-Segmentation-Pipeline",
    allow_patterns=["saved_models/*", "model-assets.json"],
    local_dir=".",
)
```

The GitHub project also provides a checksum-verifying download command and
downloads missing assets automatically before inference.

## Intended use

- Research and reproducible evaluation of the associated activity-segmentation
  pipeline.
- The public Gradio demo.
- On-device Android inference with the supplied ONNX exports.

These weights are not validated for clinical, safety-critical, or unrestricted
cross-device use. Performance can change with sensor placement, sampling
stability, calibration, device hardware, and population shift.

## Project links

- [Source and documentation](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline)
- [Project website](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/)
- [ZeroGPU demo](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline)

## License

The model assets are released under the Apache License 2.0.
