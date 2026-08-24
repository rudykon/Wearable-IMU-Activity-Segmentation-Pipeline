# Quickstart

This walkthrough first proves that the public code is healthy, then shows the
minimum authorized-data path from one folder of sensor streams to an activity
workbook.

## 1. Install

~~~bash
conda env create -f environment.yml
conda activate imu-activity-pipeline
python -m pip install -e .
python tests/smoke_test.py
~~~

No private recordings or trained checkpoints are needed for this step.

## 2. Layout

~~~text
data/
├── signals/
│   ├── train/
│   ├── internal_eval/
│   └── external_test/
├── annotations/
├── splits/
└── metadata/

saved_models/
├── ensemble_config.json
├── combined_model_3s_seed42.pth
├── combined_model_5s_seed123.pth
├── combined_model_8s_seed123.pth
├── norm_params_3s.pkl
├── norm_params_5s.pkl
└── norm_params_8s.pkl
~~~

Each sensor file is UTF-8, tab-separated, and contains at least:

~~~text
ACC_TIME  ACC_X  ACC_Y  ACC_Z  GYRO_X  GYRO_Y  GYRO_Z
~~~

The default model consumes the six physical channels. `ACC_TIME` contains
millisecond timestamps; released files may retain additional columns.

## 3. Inference

Put authorized input files under `data/signals/external_test/`, then run:

~~~bash
python run_inference.py
~~~

The compatibility entry point calls the installed package and writes:

~~~text
predictions_external_test.xlsx
~~~

Every prediction row follows:

~~~text
user_id, category, start, end
~~~

## 4. Paths

~~~bash
python -m imu_activity_pipeline.inference \
  --data_dir data/signals/internal_eval \
  --output predictions_internal_eval.xlsx
~~~

Canonical locations can also be redirected without moving the checkout:

~~~bash
export HLS_HAR_DATA_ROOT=/absolute/path/to/data
export HLS_HAR_MODEL_DIR=/absolute/path/to/saved_models
export HLS_HAR_INFERENCE_SPLIT=internal_eval
python run_inference.py
~~~

## 5. Evaluate

~~~bash
python evaluate.py \
  --split internal_eval \
  --predictions predictions_internal_eval.xlsx
~~~

The evaluator performs same-class, one-to-one segment matching and reports
precision, recall, and F1 at IoU > 0.5.

## 6. Reproduce

When the required local data and fixed assets are present:

~~~bash
bash run_reproducibility_experiments.sh
~~~

Generated material stays in ignored directories:

~~~text
experiments/results/
experiments/figures/
experiments/logs/
~~~

Use a specific interpreter when needed:

~~~bash
PYTHON_BIN=/absolute/path/to/python bash run_reproducibility_experiments.sh
~~~

## Next

- Understand the [end-to-end architecture](../guide/pipeline.md).
- Check the exact [data schema and access boundary](../guide/data.md).
- Inspect [inference and temporal post-processing](../guide/inference.md).
- Build the [Android on-device demonstration](../deployment/android.md).
