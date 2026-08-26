<p align="center">
  <strong>English</strong> · <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <img src="docs/assets/logo-horizontal.svg" width="520" alt="Wearable IMU Activity Segmentation Pipeline logo">
</p>

<h1 align="center">An End-to-End Wearable IMU System for Segment-Level Activity Recognition via Multi-Scale Arbitration and a Temporal Record Layer</h1>

<p align="center">
  A multi-scale wrist-IMU pipeline that converts continuous 100 Hz signals into timestamped activity records.
</p>

<p align="center">
  <a href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/actions/workflows/deploy-docs.yml"><img src="https://img.shields.io/github/actions/workflow/status/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/deploy-docs.yml?branch=main&amp;style=flat-square&amp;label=docs" alt="Documentation build status"></a>
  <a href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/actions/workflows/demo.yml"><img src="https://img.shields.io/github/actions/workflow/status/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/demo.yml?branch=main&amp;style=flat-square&amp;label=demo" alt="Demo test status"></a>
  <a href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/releases/tag/v0.1.0-research-preview"><img src="https://img.shields.io/badge/research%20preview-v0.1.0-3D6FB6?style=flat-square" alt="Research preview v0.1.0"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.12 or newer"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline?style=flat-square" alt="Apache 2.0 license"></a>
  <a href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-Live-FFD21E?style=flat-square" alt="Live Hugging Face Space"></a>
  <a href="https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/deployment/android/"><img src="https://img.shields.io/badge/Android-Demo-3DDC84?style=flat-square&amp;logo=android&amp;logoColor=white" alt="Android Demo"></a>
</p>

<p align="center">
  <a href="https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/">Website</a> ·
  <a href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline">Demo</a> ·
  <a href="https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline">Models</a> ·
  <a href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/releases/tag/v0.1.0-research-preview">Research release</a> ·
  <a href="README_zh.md">中文</a>
</p>

> Participant recordings are not distributed on GitHub. Public model weights are hosted on Hugging Face and downloaded with integrity verification.

## Overview

Window classifiers provide local activity evidence, but a long recording needs complete records: activity class, start time, and end time. This project builds those records with three components:

1. **Multi-scale posterior models** analyze 3-, 5-, and 8-second views of the same six-channel wrist IMU stream.
2. **Local-Boundary Scale Arbitration (LBSA)** emphasizes short-window evidence near transitions and longer context in stable regions.
3. **Temporal Record Layer (TRL)** converts the fused timeline into deterministic segment records for record-level evaluation.

<p align="center">
  <a href="docs/assets/fig02_overall_framework.png">
    <img src="docs/assets/fig02_overall_framework.png" alt="Overall framework from wrist IMU input to activity records" width="92%">
  </a>
</p>
<p align="center"><em>Three scale-specific models produce aligned posteriors; LBSA fuses them, and TRL constructs activity records.</em></p>

## Results

| Evidence | Value |
| --- | --- |
| Sensor data | 259.6 h |
| Independent external test | 37 recordings / 114 segments |
| Segment performance | 0.89 mean-user F1 / 0.90 micro-F1 |

External-test records are matched one-to-one with same-class references at IoU > 0.5. See the [results page](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/research/paper/) for the fixed-variant comparison, per-activity outcomes, representative cases, and limitations.

## Quick Start

```bash
git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
cd Wearable-IMU-Activity-Segmentation-Pipeline
conda env create -f environment.yml
conda activate imu-activity-pipeline
python -m pip install -e .
python tests/smoke_test.py
```

The CPU-safe smoke test verifies the public package and file interfaces. It requires neither participant data nor trained checkpoints.

## Data and Models

| Resource | Access |
| --- | --- |
| Participant recordings | Not stored on GitHub; follow the [dataset access instructions](data/README.md) |
| PyTorch and ONNX weights | Public [Hugging Face model repository](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline) |
| Asset hashes and licenses | [Asset documentation](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/reference/assets/) |

Missing public weights are downloaded into the expected local paths and checked against `model-assets.json` before use.

## Documentation

| Page | Purpose |
| --- | --- |
| [Method](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/guide/pipeline/) | Task, dataset, posterior models, LBSA, and TRL |
| [Results](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/research/paper/) | Independent external-test evidence and failure cases |
| [Supplementary analyses](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/research/supplementary/) | Development diagnostics, portability, and Android evidence |
| [Reproduce](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/reproduce/) | Installation, data, models, training, inference, and evaluation |
| [Demo](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline) | Run a synthetic example in the browser |

## Citation and License

Until an archival paper citation is available, cite the [v0.1.0 research preview](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/releases/tag/v0.1.0-research-preview) as described on the [citation page](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/reference/citation/). Repository-authored source and public model assets are licensed under [Apache-2.0](LICENSE); datasets and third-party dependencies retain their own terms.
