# Data

The repository publishes code, selected reproducibility model assets, layout
placeholders, and access instructions. It does **not** publish participant
sensor recordings.

!!! important "Use only authorized data"

    Keep downloaded recordings in the ignored `data/` tree. Do not commit,
    redistribute, or upload them to issues, pull requests, or experiment
    artifacts.

## Access

Before the planned PhysioNet release, research-use requests follow the form and
review procedure maintained in the repository's
[dataset access instructions](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/data/README.md).
After a PhysioNet release, that file is the canonical place for the current
repository link and citation information.

## Layout

~~~text
data/
├── signals/
│   ├── train/
│   │   └── HNUxxxxx.txt
│   ├── internal_eval/
│   │   └── HNUxxxxx.txt
│   └── external_test/
│       └── HNUxxxxx.txt
├── annotations/
│   ├── train_annotations.csv
│   ├── internal_eval_annotations.csv
│   ├── external_test_annotations.csv
│   └── all_annotations.csv
├── splits/
│   ├── train_users.txt
│   ├── internal_eval_users.txt
│   ├── external_test_users.txt
│   └── split_manifest.csv
├── metadata/
│   ├── signal_manifest.csv
│   ├── split_summary.csv
│   ├── label_summary_by_split.csv
│   └── dataset_metadata.json
└── public_external/
    ├── har70plus/
    ├── opportunity/
    ├── pamap2/
    └── wisdm_phone/
~~~

The split names are part of the public interface:

- `train` — model development;
- `internal_eval` — development/calibration evaluation; and
- `external_test` — final evaluation or default inference.

## Signals

Each recording is a UTF-8 tab-separated file. The default model requires:

| Column | Meaning |
| --- | --- |
| `ACC_TIME` | Millisecond timestamp used for output boundaries |
| `ACC_X`, `ACC_Y`, `ACC_Z` | Three-axis acceleration |
| `GYRO_X`, `GYRO_Y`, `GYRO_Z` | Three-axis angular velocity |

Released files may preserve PPG timestamps, PPG channels, or other original
columns. The default activity model reads the six ACC/GYRO channels.

Example header:

~~~text
ACC_TIME	ACC_X	ACC_Y	ACC_Z	GYRO_X	GYRO_Y	GYRO_Z
~~~

## Labels

CSV annotation files use:

~~~text
split,user_id,category,start,end
~~~

`start` and `end` are millisecond timestamps. `category` is one of:

| Label | English |
| --- | --- |
| 羽毛球 | Badminton |
| 跳绳 | Jump rope |
| 飞鸟 | Fly |
| 跑步 | Running |
| 乒乓球 | Table tennis |

## Paths

Canonical paths can be replaced through environment variables:

| Variable | Purpose |
| --- | --- |
| `HLS_HAR_DATA_ROOT` | Replace the complete `data/` root |
| `HLS_HAR_TRAIN_DATA_DIR` | Replace training signals only |
| `HLS_HAR_INTERNAL_EVAL_DATA_DIR` | Replace internal evaluation signals |
| `HLS_HAR_EXTERNAL_TEST_DATA_DIR` | Replace external-test signals |
| `HLS_HAR_*_ANNOTATIONS_FILE` | Replace a split's annotation CSV |
| `HLS_HAR_MODEL_DIR` | Replace `saved_models/` |

Example:

~~~bash
export HLS_HAR_DATA_ROOT=/mnt/authorized/imu_dataset
export HLS_HAR_MODEL_DIR=/mnt/models/imu_activity
python run_inference.py
~~~

## Portability

Optional adapters under
`experiments/public_temporal_record_layer_checks/` exercise the segment-record
layer on separately downloaded public datasets. The scripts do not download
those datasets and do not replace each dataset's own license or citation terms.
