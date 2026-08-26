<section class="demo-page-hero android-demo-hero">
  <div>
    <p class="hero-kicker">On-device research demo</p>
    <h1>Android Demo</h1>
    <p>Install the public preview APK for WT9011DCL-BT50 BLE acquisition and on-device ONNX inference. No sensor is required for the offline path: download the synthetic IMU sample and select it in the app.</p>
    <div class="demo-facts" aria-label="Android Demo compatibility">
      <span>Android 7.0+</span>
      <span>arm64-v8a</span>
      <span>On-device ONNX</span>
    </div>
    <div class="demo-actions">
      <a class="demo-action primary" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/releases/download/android-demo-v1.0-preview/hls-har-android-demo-v1.0-arm64-v8a-debug.apk">Download APK · 33 MB</a>
      <a class="demo-action" href="../../assets/android/synthetic_activity_imu.tsv" download>Download sample data</a>
      <a class="demo-action github" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/releases/tag/android-demo-v1.0-preview" target="_blank" rel="noopener">Release notes</a>
    </div>
  </div>
  <a class="demo-page-image" href="../../assets/manuscript-figures/fig08_app_field_test.png" target="_blank" rel="noopener" aria-label="Open the full-resolution Android field-test figure">
    <img src="../../assets/manuscript-figures/fig08_app_field_test.png" alt="Android activity-recognition field tests covering background and five target activities" loading="eager" decoding="async">
    <span>Field-tested prototype</span>
  </a>
</section>

<nav class="demo-page-nav" aria-label="Android Demo sections">
  <a href="#download-and-try">Download</a>
  <a href="#capabilities">Capabilities</a>
  <a href="#offline">Offline sample</a>
  <a href="#sensor">BLE sensor</a>
  <a href="#build">Build</a>
</nav>

## Download and try {#download-and-try}

<div class="demo-run-summary" aria-label="Android preview package facts">
  <div class="demo-run-stat"><strong>v1.0</strong><span>preview build</span></div>
  <div class="demo-run-stat"><strong>33 MB</strong><span>debug APK</span></div>
  <div class="demo-run-stat"><strong>API 24+</strong><span>Android 7.0+</span></div>
  <div class="demo-run-stat"><strong>120 s</strong><span>synthetic sample</span></div>
</div>

1. Download and install the **arm64-v8a APK**.
2. Download `synthetic_activity_imu.tsv` to the phone.
3. Open **Recognition**, choose the downloaded file, and run offline inference.

The sample contains 12,000 computer-generated rows at 100 Hz and no participant
data. The file picker accepts the `.tsv` file directly.

!!! warning "Preview APK"

    This is a debug-signed research preview, not a Play Store production build.
    Install only the APK linked above. Its SHA-256 is
    `cdde56db9d915eb10918724d503597a84fb18deace096086fe87509f60348be6`.

## Capabilities {#capabilities}

| Area | Implemented behavior |
| --- | --- |
| Acquisition | Scan and connect to a WT9011DCL-BT50 over BLE |
| Visualization | Acceleration, angular velocity, attitude, compass, trajectory, hand-motion, and dashboard views |
| Recording | Save timestamped IMU CSV files to Android Downloads |
| Online inference | Run the selected 3 s / 5 s / 8 s ensemble on session history |
| Offline inference | Recognize the downloadable synthetic sample or a user-selected ACC/GYRO text file |
| Temporal logic | Filtering, LBSA fusion, smoothing, Viterbi decoding, boundary refinement, and segment filtering |
| Localization | Chinese and English UI strings |

## Runtime

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" target="_blank" rel="noopener" aria-label="Open the full-resolution physical deployment chain figure">
    <img src="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" alt="Physical deployment chain from the WT9011DCL-BT50 wearable IMU through BLE acquisition and Android on-device inference to activity recognition" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Physical deployment chain from wearable IMU acquisition to Android-side recognition. Select the image to view it at full resolution.</figcaption>
</figure>

## Requirements

- Android Studio, or JDK 17 with an Android SDK;
- Android Gradle Plugin 8.1.0 and the included Gradle 8.0 wrapper;
- an Android device with BLE support; and
- a WT9011DCL-BT50 sensor for live acquisition.

The offline recognition path can be explored without the physical sensor by
copying a compatible derived segment text file to the phone.

## Build {#build}

From the app directory:

~~~bash
cd android_realtime_app
./gradlew assembleDebug
~~~

The build downloads the four ONNX weights from the public
[HF Model repository](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline)
and verifies their SHA-256 values. GitHub stores the app source and small JSON
configuration, not the model binaries.

Android Studio creates `local.properties` automatically. Do not commit that
machine-specific file.

## Sensor {#sensor}

1. Install the debug APK on a BLE-capable Android device.
2. Turn on the WT9011DCL-BT50 sensor.
3. Grant the required Bluetooth and location permissions.
4. Tap **Scan**, select a device whose name contains `WT`, and connect.
5. Use bottom navigation to inspect charts, attitude, hand, trajectory,
   dashboard, and recognition views.
6. Start recognition or record a CSV session.

Recorded files use names such as:

~~~text
imu_yyyyMMdd_HHmmss.csv
~~~

and are saved in the device Downloads directory.

## Offline {#offline}

On the recognition view, either:

- select the downloadable `synthetic_activity_imu.tsv` sample;
- choose a compatible paper-format ACC/GYRO `.txt` file; or
- copy one of the derived files from
  `android_realtime_app/motion_segments/` to the device and select it.

This exercises the app-side model and temporal layer without a live BLE
connection.

## Assets

| Asset | Role |
| --- | --- |
| `combined_model_3s_seed42.onnx` | Selected 3-second model |
| `combined_model_5s_seed123.onnx` | Selected 5-second model |
| `combined_model_8s_seed123.onnx` | Selected 8-second model |
| `norm_params_3s.json` | 3-second normalization |
| `norm_params_5s.json` | 5-second normalization |
| `norm_params_8s.json` | 8-second normalization |
| `hand_motion.onnx` | Legacy fallback model |
| `norm_params.json` | Legacy fallback normalization |

The detailed
[model card](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/android_realtime_app/MODEL_CARD.md)
contains input assumptions, output mapping, checksums, intended use, and known
limitations.

## Classes

| Index | Chinese | English |
| ---: | --- | --- |
| 0 | 无活动 | No activity |
| 1 | 羽毛球 | Badminton |
| 2 | 跳绳 | Jump rope |
| 3 | 飞鸟 | Fly |
| 4 | 跑步 | Running |
| 5 | 乒乓球 | Table tennis |

!!! warning "Research demonstration"

    The app and public models are intended for research, teaching, and
    reproducible prototype evaluation. They do not claim production safety,
    clinical validity, or cross-device generalization.

## BLE tools

Optional utilities under `android_realtime_app/tools/desktop/` provide:

- `collect.py` for direct BLE collection and matplotlib plots;
- `server.py` for a FastAPI + WebSocket service; and
- `index.html` for a browser dashboard.

Hardware protocol details, UUIDs, packet parsing, unit conversion, and Android
mapping are documented in the app's `docs/` directory.
