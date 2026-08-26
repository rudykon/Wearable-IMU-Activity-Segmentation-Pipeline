<section class="demo-page-hero">
  <div>
    <p class="hero-kicker">Interactive browser demo</p>
    <h1>From wrist IMU signals to activity records</h1>
    <p>The bundled <code>synthetic_activity_imu.tsv</code> recording is loaded by default. Run it immediately, or replace it with a compatible 100 Hz wrist-IMU file.</p>
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
  <a class="demo-page-image" href="../../assets/demo-results-paper-notation.jpg" target="_blank" rel="noopener" aria-label="Open the full-resolution demo screenshot">
    <img src="../../assets/demo-results-paper-notation.jpg" alt="Actual Hugging Face demo output showing paper-style IMU channel notation, model controls, and timestamped activity records" loading="eager" decoding="async">
    <span>Actual bundled example</span>
  </a>
</section>

<nav class="demo-page-nav" aria-label="Demo guide sections">
  <a href="#run-the-bundled-example">Run the sample</a>
  <a href="#raw-signals">Raw signals</a>
  <a href="#activity-likelihood-and-timeline">Likelihood + timeline</a>
  <a href="#activity-records">Activity records</a>
  <a href="#use-your-own-recording">Use your own file</a>
</nav>

## Run the bundled example {#run-the-bundled-example}

The fastest way to understand the interface is to run the deterministic [`synthetic_activity_imu.tsv`](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/demo/examples/synthetic_activity_imu.tsv) example. It contains **12,000 samples**, covers **120 seconds** at **100 Hz**, and contains no participant data. The live Space opens with this file already selected.

Select **English** in the language switch at the top of the Space. The complete
interface—including controls, status messages, validation errors, result-table
headers, and CSV export—then stays in English.

<div class="demo-steps">
  <article class="demo-step">
    <span class="demo-step__number">1</span>
    <h3>Open the Space</h3>
    <p>Open the live Demo and wait until the Gradio interface is ready.</p>
    <a href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">Open live Demo ↗</a>
  </article>
  <article class="demo-step">
    <span class="demo-step__number">2</span>
    <h3>Confirm the preloaded sample</h3>
    <p>The file input already contains <code>synthetic_activity_imu.tsv</code>. Upload another file only if you want to test your own compatible recording.</p>
  </article>
  <article class="demo-step">
    <span class="demo-step__number">3</span>
    <h3>Keep the defaults</h3>
    <p>Use <code>local_boundary</code>, minimum duration <code>5 s</code>, confidence <code>0.30</code>, and Top-K <code>5</code>.</p>
  </article>
  <article class="demo-step">
    <span class="demo-step__number">4</span>
    <h3>Run the current recording</h3>
    <p>Click <strong>Run current recording</strong>, then inspect the three result tabs.</p>
  </article>
</div>

<div class="demo-defaults" aria-label="Settings used for the reproducible example">
  <div><span>Input</span><strong>synthetic_activity_imu.tsv</strong></div>
  <div><span>Fusion</span><strong>local_boundary</strong></div>
  <div><span>Minimum duration</span><strong>5 s</strong></div>
  <div><span>Confidence</span><strong>0.30</strong></div>
  <div><span>Top-K</span><strong>5</strong></div>
</div>

Use <strong>Reset to synthetic sample</strong> at any time to restore the bundled file and all five defaults.

!!! tip "What happens after you click Run"

    The Space validates the preloaded file, loads the public checkpoints, runs the 3-, 5-, and 8-second models, fuses their posterior trajectories, applies the Temporal Record Layer, and returns plots, records, and a CSV in one request.

## Reproducible example output {#reproducible-example-output}

The figures below were exported from the **current public checkpoints** with the settings shown above. This run produced 118 timeline points and two activity records.

<div class="demo-run-summary" aria-label="Synthetic example run summary">
  <div class="demo-run-stat"><strong>12,000</strong><span>input samples</span></div>
  <div class="demo-run-stat"><strong>120.0 s</strong><span>recording duration</span></div>
  <div class="demo-run-stat"><strong>118</strong><span>timeline points</span></div>
  <div class="demo-run-stat"><strong>2</strong><span>activity records</span></div>
</div>

!!! note "Demonstration, not validation"

    The sample is synthetic and is intended to show the complete input-to-output path. Its detected labels and boundaries are model outputs, not ground-truth accuracy evidence. Changing fusion, duration, confidence, or Top-K can change the records.

### 1. Raw signals {#raw-signals}

Open the first result tab to inspect the six channels before classification. The upper panel contains the acceleration channels \(a_x,a_y,a_z\); the lower panel contains the angular-velocity channels \(\omega_x,\omega_y,\omega_z\).

<figure class="demo-result-figure">
  <a href="../../assets/demo/synthetic-raw-signals.png" target="_blank" rel="noopener" aria-label="Open the full-resolution raw-signal result">
    <img src="../../assets/demo/synthetic-raw-signals.png" alt="Raw accelerometer and gyroscope signals generated from synthetic_activity_imu.tsv" loading="lazy" decoding="async">
  </a>
  <figcaption><strong>Raw signals.</strong> Real output from the bundled 120-second sample.</figcaption>
</figure>

<div class="demo-reading-grid">
  <div class="demo-reading">
    <strong>What to look for</strong>
    <p>Quiet intervals appear near the beginning, between the two motion blocks, and at the end. Large periodic oscillations mark the two synthetic high-motion regions.</p>
  </div>
  <div class="demo-reading">
    <strong>Why this view matters</strong>
    <p>Use it to check missing channels, clipped values, unexpected offsets, sampling problems, and whether the uploaded recording contains visible motion.</p>
  </div>
</div>

### 2. Activity likelihood and timeline {#activity-likelihood-and-timeline}

The second tab shows two related outputs. The upper panel is the smoothed likelihood assigned to each class over time. The lower panel is the final decoded state after multi-scale fusion and temporal post-processing.

<figure class="demo-result-figure">
  <a href="../../assets/demo/synthetic-activity-likelihood-timeline.png" target="_blank" rel="noopener" aria-label="Open the full-resolution activity-likelihood and timeline result">
    <img src="../../assets/demo/synthetic-activity-likelihood-timeline.png" alt="Class likelihood curves and decoded activity timeline generated from synthetic_activity_imu.tsv" loading="lazy" decoding="async">
  </a>
  <figcaption><strong>Activity likelihood and timeline.</strong> Posterior trajectories above; final decoded activity below.</figcaption>
</figure>

<div class="demo-reading-grid">
  <div class="demo-reading">
    <strong>Upper panel</strong>
    <p>Background is strongest during quieter intervals. The first motion region raises the <em>Fly</em> likelihood, while <em>Running</em> becomes the strongest foreground likelihood in the second region.</p>
  </div>
  <div class="demo-reading">
    <strong>Lower panel</strong>
    <p>The decoded path suppresses rapid label changes and returns a stable sequence: background, Fly, background, Running, then background.</p>
  </div>
</div>

### 3. Activity records {#activity-records}

The third tab converts the decoded path into the output people can use: one row per activity period, with its class, start, end, duration, and confidence.

<figure class="demo-result-figure demo-result-figure--records">
  <a href="../../assets/demo/synthetic-activity-records.png" target="_blank" rel="noopener" aria-label="Open the full-resolution activity-record table">
    <img src="../../assets/demo/synthetic-activity-records.png" alt="Activity-record table generated from synthetic_activity_imu.tsv" loading="lazy" decoding="async">
  </a>
  <figcaption><strong>Activity records.</strong> Two records returned by the default Demo configuration.</figcaption>
</figure>

| Activity | Start (s) | End (s) | Duration (s) | Confidence |
| --- | ---: | ---: | ---: | ---: |
| Fly | 29.84 | 73.15 | 43.31 | 0.4038 |
| Running | 76.06 | 98.24 | 22.18 | 0.3186 |

The record boundaries are not copied from the sample generator. They are produced by the model probabilities, Viterbi decoding, gap handling, boundary refinement, duration filtering, and confidence filtering.

<div class="demo-download-row">
  <a class="demo-action primary" href="../../assets/demo/synthetic-activity-records.csv" download>Download this result CSV</a>
  <a class="demo-action" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/demo/examples/synthetic_activity_imu.tsv" target="_blank" rel="noopener">Inspect the sample file</a>
  <a class="demo-action github" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo" target="_blank" rel="noopener">Browse Demo source</a>
</div>

## What the controls change

| Control | Example setting | Effect |
| --- | --- | --- |
| Model fusion | `local_boundary` | Changes how the 3-, 5-, and 8-second posterior trajectories are combined, especially near possible transitions. |
| Minimum duration | `5 s` | Removes shorter decoded records. Raising it favors longer bouts; lowering it retains brief events. |
| Confidence threshold | `0.30` | Removes weak records. Raising it may reduce false alarms but can also remove true low-confidence periods. |
| Top-K | `5` | Limits the number of returned records after temporal processing; `0` disables this limit. |

The short settings above make the 120-second sample easy to inspect. They are **Demo settings**, not the fixed minute-scale settings used for the study results.

## Use your own recording {#use-your-own-recording}

1. Prepare a UTF-8 tab-separated `.txt` or `.tsv` file.
2. Replace the preloaded synthetic file in the input control with your recording.
3. Start with the Demo defaults, run once, and inspect **Raw signals** before interpreting the model output.
4. Adjust one control at a time so its effect remains understandable.

Required columns:

~~~text
ACC_TIME  ACC_X  ACC_Y  ACC_Z  GYRO_X  GYRO_Y  GYRO_Z
~~~

The plots use paper notation: file columns `ACC_X`, `ACC_Y`, and `ACC_Z` are
shown as \(a_x,a_y,a_z\), while `GYRO_X`, `GYRO_Y`, and `GYRO_Z` are shown as
\(\omega_x,\omega_y,\omega_z\). The literal names above remain the required
TSV headers.

`ACC_TIME` must contain strictly increasing millisecond timestamps. The median interval must be 8–12 ms, corresponding to approximately 100 Hz. The public interface accepts **800–60,000 valid samples**, or about 8 seconds to 10 minutes at 100 Hz. Extra columns are ignored.

!!! warning "Match the sensing protocol"

    A file can pass the format check and still be incompatible with the trained models. Sensor placement, axis orientation, units, device characteristics, and preprocessing should match the documented protocol. New devices, users, placements, and activities require new validation.

## Privacy

!!! warning "Do not upload sensitive recordings"

    Do not send confidential or identifiable participant data to a public Space. Predictions are research outputs, not medical, safety, or coaching advice.

## Run locally

Installation, verified model downloads, data boundaries, and local commands are maintained on the [Reproduce](../reproduce.md) page. The complete Gradio implementation is available in the repository's [`demo/`](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo) directory.
