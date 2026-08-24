# Hugging Face Space demo

The browser demo is the fastest way to see the project work. Choose the built-in
sample or upload a compatible wrist-motion recording; the real 3-, 5-, and
8-second repository models then produce a timestamped activity timeline for
that recording. The same model combination and timeline-cleanup
steps are used by the Python research pipeline.

[Open the Hugging Face Space](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline){ .md-button .md-button--primary target="_blank" rel="noopener" }
[Inspect the demo source](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo){ .md-button target="_blank" rel="noopener" }

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="../../assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="Open the full-resolution overall framework figure">
    <img src="../../assets/fig02_overall_framework.png" alt="Project framework showing IMU input, scale-specific CNN–BiLSTM models, LBSA fusion, temporal record layer, and segment records" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">The Space follows the repository's original path: six IMU channels → three window models → scale selection → timeline cleanup → activity records.</figcaption>
</figure>

## What you can see and download

One request produces:

- a preview of all six uploaded IMU channels;
- activity-likelihood curves for background and five supported activities;
- the final activity timeline after short prediction changes are cleaned up;
- a bilingual record table with start time, end time, duration, and confidence;
- a downloadable UTF-8 CSV with absolute millisecond timestamps.

Models are registered when the ZeroGPU Space starts, and each complete model
pass uses one 30-second GPU allocation. Inference is serialized to keep memory
use predictable and avoid competing requests. The public interface accepts no
more than 60,000 samples, equivalent to ten minutes at 100 Hz. Visitors use
their own Hugging Face ZeroGPU quota.

## Try the bundled example

Select **Bundled synthetic example**, then choose **Run segmentation**. The
120-second file contains deterministic quiet and periodic-motion phases that
exercise the complete pipeline. It contains no participant recording and is
not a validation example or a claim of biological realism.

Default demo controls are deliberately more permissive than the paper's final
long-session reporting policy:

| Control | Demo default | Purpose |
| --- | ---: | --- |
| How the three models are combined | `local_boundary` | Adapt the model weights near possible activity changes |
| Minimum duration | 5 s | Make short synthetic phases visible |
| Minimum confidence | 0.30 | Avoid hiding all outputs in a short demonstration |
| Top-K | 5 | Limit the result table |

To reproduce the paper, use the repository's fixed evaluation scripts. The
adjustable Demo controls are for exploration and are not the reported study
settings.

## Upload format

Upload a UTF-8 tab-separated `.txt` or `.tsv` file with millisecond timestamps
and at least these columns:

```text
ACC_TIME	ACC_X	ACC_Y	ACC_Z	GYRO_X	GYRO_Y	GYRO_Z
```

The demo validates strictly increasing timestamps and a median sampling
interval from 8 to 12 ms. Extra source columns are ignored. Before comparing
predictions, confirm that sensor placement, axis orientation, physical units,
and preprocessing match the documented input format.

## Run locally

```bash
git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
cd Wearable-IMU-Activity-Segmentation-Pipeline
python -m pip install -r requirements.txt
python -m pip install spaces
python -m pip install -e .
python demo/app.py
```

The same Gradio function is exposed by the Space as an API endpoint named
`/segment`.

## Privacy and limitations

!!! warning "Do not upload sensitive participant recordings"

    A public Space is shared hosting infrastructure. The application does not
    intentionally persist uploaded files, but confidential or identifiable
    recordings should remain in an authorized local environment.

Predictions are research outputs, not medical, safety, or coaching advice. The
paper does not establish cross-device or cross-population generalization, and
the bundled synthetic file cannot replace evaluation on authorized recordings.
