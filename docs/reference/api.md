# Python API

The package is deliberately small and source-oriented. Install it in editable
mode before importing:

~~~bash
python -m pip install -e .
~~~

## Package map

| Module | Main responsibility |
| --- | --- |
| `imu_activity_pipeline.config` | Paths, split names, channels, windows, classes, and training/decoder defaults |
| `signal_file_reader` | Read per-user tab-separated sensor streams |
| `sensor_data_processing` | Filtering, feature preparation, window construction, and labels |
| `neural_network_models` | Losses and PyTorch detector/classifier definitions |
| `train`, `train_parallel` | Sequential and parallel training workflows |
| `inference` | Model loading, multi-scale prediction, decoding, and segment generation |
| `evaluate` | Same-class segment matching and metrics |
| `prediction_writer` | Write output rows to Excel |

## Version

~~~python
import imu_activity_pipeline

print(imu_activity_pipeline.__version__)
~~~

Current package version:

~~~text
0.1.0
~~~

## Read signal files

`DataReader` reads every `.txt` file in a directory and returns a dictionary
keyed by file stem.

~~~python
from imu_activity_pipeline.signal_file_reader import DataReader

reader = DataReader("data/signals/external_test")
sessions = reader.read_data()

for user_id, frame in sessions.items():
    print(user_id, frame.shape)
~~~

Each frame should contain the canonical timestamp and six IMU channels described
in [Data](../guide/data.md).

## Run end-to-end inference

~~~python
from imu_activity_pipeline.inference import run_inference

segments = run_inference(
    data_dir="data/signals/internal_eval",
    output_file="predictions_internal_eval.xlsx",
)
~~~

The returned segment rows and workbook use:

~~~text
user_id, category, start, end
~~~

## Write segment records

~~~python
from imu_activity_pipeline.prediction_writer import DataOutput

rows = [
    ["HNU00001", "跑步", 1760000000000, 1760000600000],
]

DataOutput(
    rows,
    output_file="predictions_external_test.xlsx",
).save_predictions()
~~~

## Instantiate a window classifier

~~~python
import torch

from imu_activity_pipeline.neural_network_models import CombinedModel

model = CombinedModel(
    input_channels=6,
    num_classes=6,
    window_size=300,
)

x = torch.randn(2, 300, 6)
logits = model(x)
print(logits.shape)  # torch.Size([2, 6])
~~~

The practical model combines three convolution kernel sizes with a
bidirectional LSTM and fused classification head.

## Configuration

~~~python
from imu_activity_pipeline import config

print(config.SPLIT_NAMES)
print(config.WINDOW_CONFIGS)
print(config.ACTIVITIES)
~~~

Path settings such as `HLS_HAR_DATA_ROOT` and `HLS_HAR_MODEL_DIR` are read
when `config` is imported. Set environment variables before launching Python.

## Compatibility entry points

The scripts at repository root preserve source-checkout commands while
delegating to the package:

~~~text
run_inference.py
train.py
train_parallel.py
train_single_model.py
evaluate.py
~~~

For reproducible automation, prefer `python -m imu_activity_pipeline.<module>`
when a package module exposes a CLI, and record the package/versioned commit.
