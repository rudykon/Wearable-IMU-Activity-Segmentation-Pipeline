# Assets

Code, participant data, selected reproducibility weights, Android assets, and
generated experiments have different distribution boundaries. Keep those
boundaries explicit when using or extending the repository.

## Scope

| Asset class | Repository status | Notes |
| --- | --- | --- |
| Repository-authored source | Tracked | Apache-2.0 |
| Documentation and public smoke tests | Tracked | Require no private data |
| Participant sensor streams | Not distributed | Keep only in ignored local `data/` paths |
| Selected Python checkpoints | Tracked | Reproducibility/inference assets under `saved_models/` |
| Selected normalization files | Tracked | Must remain paired with checkpoint scale |
| Android ONNX assets | Tracked | Stored under the Android app's assets directory |
| Generated checkpoints and logs | Local by default | Ignored unless intentionally curated |
| Optional public datasets | User-downloaded | Original licenses and citations apply |

## Python

~~~text
saved_models/
├── ensemble_config.json
├── combined_model_3s_seed42.pth
├── combined_model_5s_seed123.pth
├── combined_model_8s_seed123.pth
├── norm_params_3s.pkl
├── norm_params_5s.pkl
└── norm_params_8s.pkl
~~~

`ensemble_config.example.json` documents the configuration structure.

!!! danger "Do not mix scales or runs"

    A model must be loaded with the normalization parameters, channel order,
    window length, and class map used during its training. A file that happens
    to load successfully is not evidence that the combination is valid.

## Research layout

~~~text
data/
  signals/{train,internal_eval,external_test}/
  annotations/
  splits/
  metadata/
  public_external/
  raw/

saved_models/
experiments/results/
experiments/figures/
experiments/logs/
~~~

Only placeholders and instructions for the local data tree are versioned.
Repository `.gitignore` rules reduce accidental publication risk but do not
replace normal data-governance review.

## Android assets

The app ships selected 3-, 5-, and 8-second ONNX models and JSON normalization
files, plus a legacy fallback model. See the
[Android model card](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/android_realtime_app/MODEL_CARD.md)
for SHA-256 checksums and runtime assumptions.

## Paths

| Variable | Default |
| --- | --- |
| `HLS_HAR_DATA_ROOT` | `<runtime>/data` |
| `HLS_HAR_TRAIN_DATA_DIR` | `data/signals/train` |
| `HLS_HAR_INTERNAL_EVAL_DATA_DIR` | `data/signals/internal_eval` |
| `HLS_HAR_EXTERNAL_TEST_DATA_DIR` | `data/signals/external_test` |
| `HLS_HAR_MODEL_DIR` | `<bundle>/saved_models` |
| `HLS_HAR_INFERENCE_SPLIT` | `external_test` |
| `HLS_HAR_EVALUATION_SPLIT` | `external_test` |

## Integrity

Before reporting or deploying a result, record:

- Git commit;
- selected checkpoint filenames and hashes;
- normalization filenames and hashes;
- `ensemble_config.json`;
- data split/manifest version;
- post-processing policy parameters;
- runtime and dependency versions; and
- the exact evaluation command.

## Licenses

- Repository-authored code: [Apache License 2.0](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/LICENSE).
- Python model assets: `saved_models/WEIGHTS_LICENSE`.
- Android source and weights: `android_realtime_app/LICENSE` and
  `android_realtime_app/WEIGHTS_LICENSE`.
- Datasets and third-party dependencies: their own terms.
