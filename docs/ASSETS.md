# Data and Model Assets

This repository intentionally excludes private datasets, held-out evaluation files, trained checkpoint weights, non-public writing files, local caches, and generated experiment outputs. The `data/` directory contains only layout placeholders and access instructions.

## Dataset Access

Dataset access instructions are maintained on the project GitHub page.

Before the PhysioNet repository is formally released, research-use access requests should be submitted through the access request form linked from the GitHub page. Requests are reviewed by the Hainan University organizer responsible for data management. After approval, place the downloaded files under `data/` using the layout below.

After the PhysioNet repository is released, the GitHub access request form will be closed and readers should obtain the dataset directly from PhysioNet.

## Expected Local Layout

```text
data/
  signals/
    train/                  # authorized training streams
      HNUxxxxx.txt
    internal_eval/          # development/calibration streams
      HNUxxxxx.txt
    external_test/          # final evaluation or inference streams
      HNUxxxxx.txt
  annotations/
    train_annotations.csv
    internal_eval_annotations.csv
    external_test_annotations.csv
    all_annotations.csv
  splits/
    train_users.txt
    internal_eval_users.txt
    external_test_users.txt
    split_manifest.csv
  metadata/
    signal_manifest.csv
    split_summary.csv
    label_summary_by_split.csv
    dataset_metadata.json
  public_external/          # optional public-dataset downloads
  raw/                      # optional raw archives for selected experiment scripts
    raw.csv                 # optional raw index

saved_models/               # local checkpoints and normalization parameters
experiments/results/        # generated experiment tables and summaries
experiments/figures/        # generated experiment figures
```

## Sensor Text Format

Each `.txt` file is a tab-separated sensor stream. The released pipeline uses these columns:

```text
ACC_TIME, ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z
```

Additional columns may be present, but the default model uses only the six IMU channels. Timestamps are millisecond timestamps.

## Label Format

Annotation CSV files should contain:

```text
split, user_id, category, start, end
```

`category` must be one of:

```text
羽毛球, 跳绳, 飞鸟, 跑步, 乒乓球
```

## Model Assets

For the default multi-scale inference configuration, place these files under `saved_models/`:

```text
ensemble_config.json
combined_model_3s_seed42.pth
combined_model_5s_seed123.pth
combined_model_8s_seed123.pth
norm_params_3s.pkl
norm_params_5s.pkl
norm_params_8s.pkl
```

A complete training run may also produce:

```text
combined_model_{3s,5s,8s}_seed{42,123,456}.pth
norm_params_{3s,5s,8s}.pkl
combined_model_best.pth
norm_params.pkl
```

`saved_models/ensemble_config.example.json` documents the expected JSON structure.

## Open-Source Scope

Included:

- Core algorithm source code.
- Training, inference, evaluation, and post-processing scripts.
- Experiment, robustness, visualization, and public portability scripts.
- Lightweight documentation for running the code with authorized local assets.

Excluded:

- Non-public writing files, bibliography files, and generated document figures.
- Private participant sensor streams.
- Local held-out evaluation streams and annotations.
- Trained checkpoint weights and normalization pickle files.
- Local virtual environments, PyInstaller builds, caches, logs, and temporary PDF previews.
- Third-party reference PDFs.

The `.gitignore` file is configured to keep local data, checkpoints, caches, and generated assets untracked.
