# Inference

Convert long sensor files into timestamped activities.

## Run

~~~bash
python run_inference.py
~~~

Reads:

~~~text
data/signals/external_test/*.txt
saved_models/ensemble_config.json
saved_models/*.pth
saved_models/*.pkl
~~~

Writes:

~~~text
predictions_external_test.xlsx
~~~

## Paths

Set input and output:

~~~bash
python -m imu_activity_pipeline.inference \
  --data_dir data/signals/internal_eval \
  --output predictions_internal_eval.xlsx
~~~

For external data and models:

~~~bash
export HLS_HAR_DATA_ROOT=/absolute/path/to/data
export HLS_HAR_MODEL_DIR=/absolute/path/to/saved_models
export HLS_HAR_INFERENCE_SPLIT=external_test
python run_inference.py
~~~

## Steps

1. Load checkpoints and normalization.
2. Read each session.
3. Filter and normalize six channels.
4. Build 3-, 5-, and 8-second windows.
5. Align probabilities.
6. Fuse scales with LBSA.
7. Decode the sequence with Viterbi.
8. Refine and filter segments.
9. Write the workbook.

## Output

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

See [Python API](../reference/api.md) for lower-level I/O.

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
