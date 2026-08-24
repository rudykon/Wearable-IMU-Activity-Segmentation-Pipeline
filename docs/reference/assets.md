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
| Selected Python checkpoints | Hugging Face | Downloaded to `saved_models/` when needed |
| Selected normalization files | Hugging Face | Paired with each checkpoint scale |
| Android ONNX assets | Hugging Face | Downloaded before the Android build |
| Model manifest | Tracked | File sizes and SHA-256 values in `model-assets.json` |
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

Download the assets in advance when working offline:

~~~bash
python scripts/download_model_assets.py python
~~~

Python inference performs this step automatically when a required file is
missing. Existing locally retrained files are not overwritten unless `--force`
is supplied.

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

The build downloads selected 3-, 5-, and 8-second ONNX models plus the legacy
fallback from the public model repository. JSON normalization parameters remain
small tracked runtime configuration. See the
[Android model card](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/android_realtime_app/MODEL_CARD.md)
for SHA-256 checksums and runtime assumptions.

~~~bash
python scripts/download_model_assets.py android
~~~

## Paths

| Variable | Default |
| --- | --- |
| `HLS_HAR_DATA_ROOT` | `<runtime>/data` |
| `HLS_HAR_TRAIN_DATA_DIR` | `data/signals/train` |
| `HLS_HAR_INTERNAL_EVAL_DATA_DIR` | `data/signals/internal_eval` |
| `HLS_HAR_EXTERNAL_TEST_DATA_DIR` | `data/signals/external_test` |
| `HLS_HAR_MODEL_DIR` | `<bundle>/saved_models` |
| `HLS_HAR_MODEL_REPO_ID` | `config-h/Wearable-IMU-Activity-Segmentation-Pipeline` |
| `HLS_HAR_MODEL_REVISION` | `main` |
| `HLS_HAR_OFFLINE` | unset; set to `1` to disable downloads |
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
- Python and Android weights: [public Hugging Face Model repository](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline) (Apache-2.0).
- Android source: `android_realtime_app/LICENSE`.
- Datasets and third-party dependencies: their own terms.
