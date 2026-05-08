# Wearable IMU Activity Segmentation Pipeline

中文版本：[README_zh.md](README_zh.md)


This repository contains a research-code implementation for long-session wearable IMU activity segmentation. It reads accelerometer and gyroscope streams, predicts activity segments, and writes records in the form:

```text
user_id, category, start, end
```

The release keeps the algorithm, training scripts, inference pipeline, post-processing logic, experiment scripts, and reusable public-dataset checks. Private recordings, held-out evaluation files, trained checkpoints, non-public writing files, local environments, build artifacts, logs, and third-party PDFs are not included.

The repository also includes an Android on-device demo in `android_realtime_app/`, with bundled ONNX inference assets and documentation for the WT9011DCL-BT50 BLE IMU workflow.

## Pipeline Figures

![Overall activity-segmentation framework](experiments/figures/fig02_overall_framework.png)

![Physical deployment chain](experiments/figures/fig03_physical_deployment_chain.png)

## Highlights

- Multi-kernel 1D-CNN + BiLSTM window classifier.
- 3 s, 5 s, and 8 s sliding-window training and inference support.
- Multi-scale probability alignment and Local-Boundary Scale Arbitration.
- Temporal decoding with smoothing, Viterbi constraints, segment extraction, boundary refinement, overlap resolution, confidence filtering, and Top-K pruning.
- Segment-level evaluation using same-class one-to-one IoU matching.
- Experiment, robustness, visualization, and public-dataset portability scripts.
- Android real-time acquisition and on-device inference demo.

## Repository Layout

```text
.
|-- run_inference.py                 # Thin inference wrapper for source checkouts and packaged builds
|-- train.py                # Thin sequential-training wrapper
|-- train_parallel.py       # Thin parallel-training wrapper
|-- train_single_model.py            # Thin single-model training wrapper
|-- evaluate.py             # Thin evaluation wrapper
|-- src/imu_activity_pipeline/
|   |-- config.py           # Paths, sensor settings, and model/post-processing parameters
|   |-- sensor_data_processing.py       # Sensor loading, filtering, windowing, labels, and augmentation
|   |-- neural_network_models.py        # Neural network definitions and auxiliary losses
|   |-- inference.py        # End-to-end inference and post-processing
|   |-- train*.py           # Training implementations
|   `-- input.py/output.py  # Lightweight file I/O interfaces
|-- scripts/                # Auxiliary analysis, sweep, and figure helpers
|-- experiments/            # Experiment, robustness, figure, and public portability scripts
|-- docs/                   # Usage and asset layout notes
|-- android_realtime_app/   # Android BLE acquisition and on-device ONNX inference app
|-- tests/                  # Lightweight repository health checks
|-- saved_models/           # Local checkpoint placeholder
`-- data/                   # Unified local dataset root
    |-- signals/
    |   |-- train/
    |   |-- internal_eval/
    |   `-- external_test/
    |-- annotations/
    |-- splits/
    |-- metadata/
    `-- public_external/
```

## Environment

Use Python 3.12 and the Conda environment file as the recommended setup. A CUDA-capable GPU is recommended for training; inference can also run on CPU.

```bash
conda env create -f environment.yml
conda activate imu-activity-pipeline
python -m pip install -e .
```

Then run:

```bash
python tests/smoke_test.py
```

## Quick Start

After obtaining the dataset through the process described in [Data Access](#data-access), place the files under `data/` and model assets under `saved_models/`, then run:

```bash
python run_inference.py
```

The default output is:

```text
predictions_external_test.xlsx
```

For training, evaluation, Python interfaces, packaged executable usage, and experiment scripts, see [docs/USAGE.md](docs/USAGE.md).

## Android App

The Android demo is located in [android_realtime_app/](android_realtime_app/). It supports WT9011DCL-BT50 BLE acquisition, visualization, CSV recording, and on-device ONNX inference.

Use Android Studio, or build from a JDK 17 + Android SDK environment:

```bash
cd android_realtime_app
./gradlew assembleDebug
```

## Data Access

The dataset is not distributed directly in this GitHub repository. Dataset access instructions are maintained on the project GitHub page.

Before the PhysioNet repository is formally released, research-use access requests should be submitted through the access request form linked from the GitHub page. Requests are reviewed by the Hainan University organizer responsible for data management. After approval, readers can download the dataset and place it under `data/` following [data/README.md](data/README.md).

After the PhysioNet repository is released, the request form will be closed and readers should obtain the dataset directly from PhysioNet. The GitHub page will keep the current PhysioNet link and citation information.

## Data and Model Assets

The Python research pipeline is code-first. The `data/` directory contains only layout placeholders and data-access instructions; authorized local data files are ignored by Git. Required private or user-provided assets are documented in [docs/ASSETS.md](docs/ASSETS.md), including expected sensor columns, label format, checkpoint names, and ignored local-output locations. The Android app ships small ONNX demo assets documented in [android_realtime_app/MODEL_CARD.md](android_realtime_app/MODEL_CARD.md) and licensed by [android_realtime_app/WEIGHTS_LICENSE](android_realtime_app/WEIGHTS_LICENSE).

## Experiment Reproduction

The top-level experiment wrapper is:

```bash
bash run_reproducibility_experiments.sh
```

This requires the local data and checkpoint assets described in [docs/ASSETS.md](docs/ASSETS.md).

## License

MIT License. See [LICENSE](LICENSE).
