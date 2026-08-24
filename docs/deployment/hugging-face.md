# Demo

Run the 3-, 5-, and 8-second models in your browser. Use the sample or upload a
compatible wrist IMU file.

[Open the Hugging Face Space](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline){ .md-button .md-button--primary target="_blank" rel="noopener" }
[Inspect the demo source](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo){ .md-button target="_blank" rel="noopener" }

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="../../assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="Open the full-resolution overall framework figure">
    <img src="../../assets/fig02_overall_framework.png" alt="Project framework showing IMU input, scale-specific CNN–BiLSTM models, LBSA fusion, temporal record layer, and segment records" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Six channels → three models → fusion → records.</figcaption>
</figure>

## Output

- six IMU signals;
- class probabilities;
- decoded timeline;
- activity table;
- CSV download.

Limit: 60,000 samples, or 10 minutes at 100 Hz. Each run uses one ZeroGPU request.

## Sample

Choose **Bundled synthetic example**, then **Run segmentation**. The 120-second
sample is synthetic, contains no participant data, and is not validation data.

| Control | Demo default | Purpose |
| --- | ---: | --- |
| How the three models are combined | `local_boundary` | Adapt the model weights near possible activity changes |
| Minimum duration | 5 s | Make short synthetic phases visible |
| Minimum confidence | 0.30 | Avoid hiding all outputs in a short demonstration |
| Top-K | 5 | Limit the result table |

Demo controls are for exploration. Use the fixed scripts to reproduce the paper.

## Upload format

Upload a UTF-8 tab-separated `.txt` or `.tsv` file with millisecond timestamps
and at least these columns:

```text
ACC_TIME	ACC_X	ACC_Y	ACC_Z	GYRO_X	GYRO_Y	GYRO_Z
```

Timestamps must increase; median spacing must be 8–12 ms. Match the documented
placement, axes, units, and preprocessing.

## Run locally

```bash
git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
cd Wearable-IMU-Activity-Segmentation-Pipeline
python -m pip install -r requirements.txt
python -m pip install spaces
python -m pip install -e .
python demo/app.py
```

API endpoint: `/segment`.

## Privacy

!!! warning "Do not upload sensitive participant recordings"

    Do not upload confidential or identifiable recordings to a public Space.

Predictions are research outputs, not medical, safety, or coaching advice.
Cross-device and cross-population performance is not established.
