---
search:
  exclude: true
robots: noindex, follow
---

# Usage guide

This page preserves the repository's original `docs/USAGE.md` link. The
expanded project website now separates each workflow into focused guides:

- [Installation](getting-started/installation.md)
- [Quick start](getting-started/quickstart.md)
- [Training](guide/training.md)
- [Inference](guide/inference.md)
- [Evaluation](guide/evaluation.md)
- [Python API](reference/api.md)
- [Android deployment](deployment/android.md)

## Minimal public verification

~~~bash
conda env create -f environment.yml
conda activate imu-activity-pipeline
python -m pip install -e .
python tests/smoke_test.py
~~~

## Default authorized-data inference

~~~bash
python run_inference.py
~~~

The default split is `external_test`, and the output is
`predictions_external_test.xlsx`.

## Experiment wrapper

~~~bash
bash run_reproducibility_experiments.sh
~~~

This command requires the local data and model assets described in the
[asset guide](reference/assets.md).
