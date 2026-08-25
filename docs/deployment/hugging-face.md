# Demo

Run the public 3-, 5-, and 8-second models in a browser. Choose the bundled synthetic example or upload a compatible 100 Hz wrist-IMU file.

[Open the Hugging Face Space](https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline){ .md-button .md-button--primary target="_blank" rel="noopener" }
[Open the model weights](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline){ .md-button target="_blank" rel="noopener" }

<figure class="paper-figure demo-figure">
  <a class="pipeline-image-link" href="../../assets/demo-results.jpg" target="_blank" rel="noopener" aria-label="Open the full-resolution demo screenshot">
    <img src="../../assets/demo-results.jpg" alt="Hugging Face demo showing a synthetic IMU input, model controls, and two timestamped activity records" loading="eager" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Actual demo output for the bundled synthetic recording; no participant data are shown.</figcaption>
</figure>

## Input

Upload a UTF-8 tab-separated `.txt` or `.tsv` file with millisecond timestamps and these columns:

~~~text
ACC_TIME  ACC_X  ACC_Y  ACC_Z  GYRO_X  GYRO_Y  GYRO_Z
~~~

Timestamps must increase, with a median spacing of 8–12 ms. The public interface accepts at most 60,000 samples, equivalent to 10 minutes at 100 Hz.

## Output

- six IMU signal plots;
- class probabilities and a decoded timeline;
- activity, start, end, duration, and confidence for each record; and
- a downloadable CSV.

The bundled 120-second example is computer-generated and is not validation data. Demo controls are exploratory; the reported study uses the fixed evaluation scripts.

## Privacy

!!! warning "Do not upload sensitive recordings"

    Do not send confidential or identifiable participant data to a public Space. Predictions are research outputs, not medical, safety, or coaching advice.

## Reproduce locally

Use the [Reproduce](../reproduce.md) page for installation, model downloads, data boundaries, and local commands. The interface source is available under [`demo/`](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo).
