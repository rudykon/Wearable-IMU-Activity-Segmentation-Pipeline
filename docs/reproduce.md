# Reproduce

The public repository supports two distinct workflows: a data-free package check and full experiments with authorized recordings. Keeping them separate avoids implying that participant data are bundled with the code.

## Public verification

~~~bash
git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
cd Wearable-IMU-Activity-Segmentation-Pipeline
conda env create -f environment.yml
conda activate imu-activity-pipeline
python -m pip install -e .
python tests/smoke_test.py
~~~

The smoke test is CPU-safe and requires neither participant recordings nor model checkpoints.

## Workflow map

| Step | Documentation | Main command |
| --- | --- | --- |
| Install | [Environment and verification](getting-started/installation.md) | `python tests/smoke_test.py` |
| Place authorized data | [Dataset](guide/data.md) and [asset boundaries](reference/assets.md) | — |
| Train posterior models | [Training protocol](guide/training.md) | `python train.py` |
| Generate records | [Inference and TRL](guide/inference.md) | `python run_inference.py` |
| Score records | [Segment evaluation](guide/evaluation.md) | `python evaluate.py --split external_test` |
| Build the mobile prototype | [Android](deployment/android.md) | `./gradlew assembleDebug` |

Python and Android weights are hosted in the public [Hugging Face model repository](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline). Missing assets are downloaded into their expected paths and checked against `model-assets.json`.

## Full experiment wrapper

After placing the required local data and fixed assets, run:

~~~bash
bash run_reproducibility_experiments.sh
~~~

The wrapper executes the saved-model external evaluation, internal robustness and policy-selection checks, representative timeline generation, an external unlabeled-cohort stress test, and summary-figure generation. Outputs remain under:

~~~text
experiments/results/
experiments/figures/
experiments/logs/
~~~

Use a specific interpreter when needed:

~~~bash
PYTHON_BIN=/absolute/path/to/python bash run_reproducibility_experiments.sh
~~~

## Interfaces

- [Quickstart](getting-started/quickstart.md) — minimum authorized-data path.
- [API](reference/api.md) — package imports and output schema.
- [Models and licenses](reference/assets.md) — filenames, hashes, and distribution boundaries.
- [Citation](reference/citation.md) — versioned software citation.

!!! warning "Data boundary"

    Participant recordings are not stored in the public repository. Preserve the documented access, privacy, split, and licensing conditions when reproducing or extending the study.
