# Training

Training builds scale-specific activity classifiers and normalization assets
from authorized long-session recordings and annotations.

## Setup

Confirm that:

1. the package is installed with `python -m pip install -e .`;
2. authorized training streams are under `data/signals/train/`;
3. labels are under `data/annotations/train_annotations.csv`; and
4. the selected PyTorch installation can access the intended CUDA device.

Run the public health check first:

~~~bash
python tests/smoke_test.py
~~~

## Commands

=== "Sequential"

    ~~~bash
    python train.py
    ~~~

    Trains the configured workflow in one process.

=== "Parallel"

    ~~~bash
    python train_parallel.py
    ~~~

    Coordinates independent scale/seed jobs for a machine with sufficient
    resources.

=== "Shell wrapper"

    ~~~bash
    bash run_training.sh
    ~~~

    Convenient for a long-running source checkout.

=== "tmux wrapper"

    ~~~bash
    bash run_training_in_tmux.sh
    ~~~

    Starts and records a persistent terminal training session.

## Experiments

The default configuration covers:

| Dimension | Values |
| --- | --- |
| Window length | 3 s, 5 s, 8 s |
| Step | 1 s |
| Ensemble seeds | 42, 123, 456 |
| Input channels | 6 ACC/GYRO channels |
| Classifier | Background + 5 foreground activities |

The model uses parallel convolution kernels, a bidirectional LSTM, and a fused
classification head. Focal and triplet losses are available for imbalance and
embedding separation experiments.

## Runtime

Several long-run controls can be changed without editing source:

~~~bash
export NUM_EPOCHS_STAGE2=100
export EARLY_STOPPING_PATIENCE=30
export MIN_EPOCHS_BEFORE_EARLY_STOP=40
python train.py
~~~

Batch size, learning rate, window construction, augmentation, and device
defaults live in `imu_activity_pipeline.config`. Record any non-default values
with experiment outputs.

## Outputs

A complete run can write:

~~~text
saved_models/
├── combined_model_3s_seed42.pth
├── combined_model_5s_seed123.pth
├── combined_model_8s_seed123.pth
├── combined_model_{3s,5s,8s}_seed{42,123,456}.pth
├── norm_params_3s.pkl
├── norm_params_5s.pkl
├── norm_params_8s.pkl
├── ensemble_config.json
├── logs/
└── plots/
~~~

Locally generated checkpoints, logs, and plots are ignored unless intentionally
curated as reproducibility assets. Keep every checkpoint paired with the
normalization parameters and configuration used to create it.

## Reproduce

After training and calibration:

~~~bash
bash run_reproducibility_experiments.sh
~~~

The wrapper coordinates saved-model evaluation, internal robustness checks,
post-processing policy comparisons, signal-quality analyses, timeline figures,
external-cohort stress tests, and summary figure generation.

!!! warning

    The wrapper is not a substitute for asset preparation. It expects the local
    files documented in [Data & model assets](../reference/assets.md) and will
    not download private or public datasets.
