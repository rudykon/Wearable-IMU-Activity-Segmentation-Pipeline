# Usage

## Task

The pipeline performs long-session wearable IMU activity segmentation. Each input file is one continuous sensor stream, and the output is a list of activity records:

```text
user_id, category, start, end
```

Supported foreground categories are:

```text
羽毛球, 跳绳, 飞鸟, 跑步, 乒乓球
```

Timestamps are millisecond timestamps. The default metric is mean user-level segmental F1 with same-class one-to-one matching and IoU > 0.5.

## Installation

Create the Conda environment or the CPU-only pip environment described in the repository README first. Then install the source package in editable mode:

```bash
python -m pip install -e .
```

This makes `imu_activity_pipeline` imports available for notebooks, scripts, and interactive Python sessions.

## Inference

Prepare local files:

```text
data/signals/external_test/*.txt
saved_models/ensemble_config.json
saved_models/*.pth
saved_models/*.pkl
```

Run:

```bash
python run_inference.py
```

The default split is `external_test`, and the workbook `predictions_external_test.xlsx` is written at the repository root. Each row contains:

```text
user_id, category, start, end
```

To run the same pipeline on the development/calibration split, keep the split name explicit:

```bash
python -m imu_activity_pipeline.inference \
  --data_dir data/signals/internal_eval \
  --output predictions_internal_eval.xlsx
```

## Evaluation

Evaluation names follow the released data split names: `train`, `internal_eval`, and `external_test`.

With `predictions_external_test.xlsx` and `data/annotations/external_test_annotations.csv` available:

```bash
python evaluate.py --split external_test
```

For internal development/calibration evaluation:

```bash
python evaluate.py --split internal_eval --predictions predictions_internal_eval.xlsx
```

The evaluator reports segment-level precision, recall, and F1 using same-class IoU matching.

## Training

Place authorized training streams and labels in the expected local layout, then run:

```bash
python train.py
```

For shell-based or long-running training:

```bash
bash run_training.sh
```

Training writes checkpoints, normalization parameters, logs, and plots under `saved_models/`. These generated assets are ignored by Git.

## Python Interfaces

`imu_activity_pipeline.signal_file_reader` provides `DataReader`, which reads all `.txt` files in a directory and returns a dictionary keyed by file stem:

```python
from imu_activity_pipeline.signal_file_reader import DataReader

reader = DataReader("./data/signals/external_test")
data = reader.read_data()
```

`imu_activity_pipeline.prediction_writer` provides `DataOutput`, which writes prediction rows to an Excel workbook:

```python
from imu_activity_pipeline.prediction_writer import DataOutput

results = [
    ["HNU00001", "跑步", 1760000000000, 1760000600000],
]
DataOutput(results, output_file="./predictions_external_test.xlsx").save_predictions()
```

## Packaged Executable

`main.spec` can be used with PyInstaller to package the inference entry point. A packaged layout should keep `data/` next to the executable and model assets inside the bundled runtime directory or another path configured in `imu_activity_pipeline.config`.

Example runtime layout:

```text
main/
|-- main.exe
|-- _internal/
|   `-- saved_models/
|-- data/
`-- predictions_external_test.xlsx
```

## Experiment Scripts

The experiment wrapper is:

```bash
bash run_reproducibility_experiments.sh
```

Generated tables and figures are written to `experiments/results/` and `experiments/figures/`. Reproduction requires local data and model assets; see [ASSETS.md](ASSETS.md).

Optional public-dataset portability checks are under:

```text
experiments/public_temporal_record_layer_checks/
```

These scripts do not download datasets. Put each public dataset under `data/public_external/` and run the dataset-specific script, or run the combined table builder after summaries exist:

```bash
python experiments/public_temporal_record_layer_checks/run_public_temporal_record_layer_checks.py
```

## Limitations

- This repository does not include private sensor recordings or trained checkpoints.
- Exact result reproduction depends on fixed local checkpoints and post-processing parameters.
- The sensor loader is centered on the tab-separated stream schema described in [ASSETS.md](ASSETS.md).
- Public-dataset extension scripts require users to download those datasets separately and follow their licenses.
