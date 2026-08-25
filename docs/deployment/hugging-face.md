<section class="demo-page-hero">
  <div>
    <p class="hero-kicker">Interactive browser demo</p>
    <h1>From wrist IMU signals to activity records</h1>
    <p>Run the public 3-, 5-, and 8-second models with the bundled synthetic recording or a compatible 100 Hz wrist-IMU file.</p>
    <div class="demo-facts" aria-label="Demo capabilities">
      <span>Real public models</span>
      <span>Six signal channels</span>
      <span>Timeline + CSV</span>
    </div>
    <div class="demo-actions">
      <a class="demo-action primary" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">Open live demo</a>
      <a class="demo-action github" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.22c-3.23.7-3.91-1.37-3.91-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.72 1.27 3.39.97.1-.75.4-1.27.74-1.56-2.58-.29-5.29-1.29-5.29-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.48.11-3.05 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.19-1.49 3.16-1.18 3.16-1.18.63 1.57.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.4-2.72 5.38-5.31 5.67.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .7Z"/></svg>
        View demo source
      </a>
      <a class="demo-action" href="https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">Model weights</a>
    </div>
  </div>
  <a class="demo-page-image" href="../../assets/demo-results.jpg" target="_blank" rel="noopener" aria-label="Open the full-resolution demo screenshot">
    <img src="../../assets/demo-results.jpg" alt="Actual Hugging Face demo output showing a synthetic IMU input, model controls, and timestamped activity records" loading="eager" decoding="async">
    <span>Actual bundled example</span>
  </a>
</section>

## What the demo shows

The interface keeps the complete inference path visible instead of returning only a label:

- six accelerometer and gyroscope signal plots;
- class probabilities and the decoded timeline;
- activity, start, end, duration, and confidence for every record; and
- a downloadable CSV file.

The bundled 120-second example is computer-generated, contains no participant data, and is not part of the validation set. Demo controls are intended for exploration; reported results use the fixed evaluation scripts.

## Input format

Upload a UTF-8 tab-separated `.txt` or `.tsv` file with millisecond timestamps and these columns:

~~~text
ACC_TIME  ACC_X  ACC_Y  ACC_Z  GYRO_X  GYRO_Y  GYRO_Z
~~~

Timestamps must increase, with a median spacing of 8–12 ms. The public interface accepts at most 60,000 samples, equivalent to 10 minutes at 100 Hz. Device placement, axes, units, and preprocessing should match the documented protocol.

## Privacy

!!! warning "Do not upload sensitive recordings"

    Do not send confidential or identifiable participant data to a public Space. Predictions are research outputs, not medical, safety, or coaching advice.

## Run locally

Installation, verified model downloads, data boundaries, and local commands are maintained on the [Reproduce](../reproduce.md) page. The complete Gradio implementation is available in the repository's [`demo/`](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo) directory.
