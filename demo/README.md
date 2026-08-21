# Hugging Face Space demo

This directory contains a CPU-safe Gradio interface for the repository's real
tracked 3 s / 5 s / 8 s CNN–BiLSTM checkpoints and temporal record layer.

## What the demo does

- accepts one UTF-8 tab-separated `.txt` or `.tsv` recording;
- validates the canonical millisecond timestamp and six-channel IMU contract;
- runs the tracked multi-scale models with selectable probability fusion;
- shows the uploaded signals, smoothed class probabilities, and decoded path;
- reports bilingual activity segments and provides a downloadable CSV;
- includes a deterministic synthetic example with no participant data.

The public interface limits uploads to 20 MB, 60,000 samples, and one serialized
inference request at a time. Models are loaded lazily on the first request and
cached for subsequent runs.

## Run locally

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python demo/app.py
```

Open the local Gradio URL printed by the process. The first inference request is
slower because it loads all three tracked checkpoints.

Regenerate the bundled synthetic example with:

```bash
python demo/generate_sample.py
```

Validate all three tracked checkpoints on CPU with:

```bash
python demo/model_smoke_test.py
```

## Deploy to Hugging Face Spaces

Deploy the **whole repository**, not only this directory. The app imports the
core package from `src/` and loads the tracked assets under `saved_models/`.

The YAML metadata at the top of the root `README.md` configures a Gradio Space
with Python 3.12 and `demo/app.py` as the entry point. The root
`requirements.txt` contains the runtime dependencies. Create a Gradio Space,
then push or mirror this repository to it; no dataset or secret is required.

Suggested public Space ID:

```text
rudykon/Wearable-IMU-Activity-Segmentation-Pipeline
```

## Input contract

The first row must contain at least these tab-separated columns:

```text
ACC_TIME	ACC_X	ACC_Y	ACC_Z	GYRO_X	GYRO_Y	GYRO_Z
```

`ACC_TIME` contains strictly increasing millisecond timestamps. The model
expects approximately 100 Hz data (median interval 8–12 ms). Additional columns
are ignored.

## Privacy and scientific scope

Do not upload confidential or identifiable participant recordings to a public
Space. This application does not intentionally persist uploads, but the hosting
platform is shared infrastructure. Results are research outputs, not medical,
safety, or coaching advice. Device, units, sensor placement, and population
shift can materially affect predictions.
