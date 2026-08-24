# Inference

Inference converts every long sensor file in an input directory into
timestamped foreground activity segments.

## Default command

~~~bash
python run_inference.py
~~~

By default, the wrapper reads:

~~~text
data/signals/external_test/*.txt
saved_models/ensemble_config.json
saved_models/*.pth
saved_models/*.pkl
~~~

and writes:

~~~text
predictions_external_test.xlsx
~~~

## Explicit paths

Use the package module when choosing an input directory and output workbook:

~~~bash
python -m imu_activity_pipeline.inference \
  --data_dir data/signals/internal_eval \
  --output predictions_internal_eval.xlsx
~~~

Use environment variables when data and models live outside the repository:

~~~bash
export HLS_HAR_DATA_ROOT=/absolute/path/to/data
export HLS_HAR_MODEL_DIR=/absolute/path/to/saved_models
export HLS_HAR_INFERENCE_SPLIT=external_test
python run_inference.py
~~~

## What happens during a run

1. Load the selected scale checkpoints and matching normalization parameters.
2. Read each tab-separated session.
3. Filter and normalize the six physical IMU channels.
4. Build 3-, 5-, and 8-second windows on a common one-second step.
5. Produce and align scale-specific class probabilities.
6. Combine the three window lengths using the configured scale-selection rule (LBSA).
7. Reduce rapid label changes and choose a consistent activity sequence (Viterbi decoding).
8. Refine boundaries, resolve overlaps, and apply segment policies.
9. Write foreground segment records to the workbook.

## Output records

| Column | Description |
| --- | --- |
| `user_id` | File stem / session identifier |
| `category` | One of the five foreground activity labels |
| `start` | Segment start in milliseconds |
| `end` | Segment end in milliseconds |

Example:

~~~text
HNU00001,跑步,1760000000000,1760000600000
~~~

## Python interface

~~~python
from imu_activity_pipeline.inference import run_inference

segments = run_inference(
    data_dir="data/signals/internal_eval",
    output_file="predictions_internal_eval.xlsx",
)

print(f"generated {len(segments)} segments")
~~~

For lower-level I/O, see the [Python API](../reference/api.md).

## Troubleshooting

??? question "No input files are found"

    Confirm that the selected directory contains `.txt` files and that the
    active split or `--data_dir` points to that directory.

??? question "A checkpoint loads but tensor shapes do not match"

    Check that the selected checkpoint corresponds to its configured window
    length and that the expected six-channel input order is unchanged.

??? question "Predictions look unstable"

    Verify sampling rate, sensor placement, physical units, filtering,
    scale-specific normalization, and the ensemble configuration before tuning
    temporal thresholds.

??? question "The workbook is empty"

    The decoder may have classified the recording as background or removed all
    foreground candidates through duration/confidence policies. Inspect the
    probability plots and filtering settings before weakening the filters.
