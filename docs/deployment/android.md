# Android

The `android_realtime_app/` module is a deployable companion to the Python
research pipeline. It supports WT9011DCL-BT50 BLE acquisition, live
visualization, CSV recording, offline-file recognition, and on-device ONNX
inference.

## Capabilities

| Area | Implemented behavior |
| --- | --- |
| Acquisition | Scan and connect to a WT9011DCL-BT50 over BLE |
| Visualization | Acceleration, angular velocity, attitude, compass, trajectory, hand-motion, and dashboard views |
| Recording | Save timestamped IMU CSV files to Android Downloads |
| Online inference | Run the selected 3 s / 5 s / 8 s ensemble on session history |
| Offline inference | Recognize bundled samples or a user-selected ACC/GYRO text file |
| Temporal logic | Filtering, LBSA fusion, smoothing, Viterbi decoding, boundary refinement, and segment filtering |
| Localization | Chinese and English UI strings |

## Runtime

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" target="_blank" rel="noopener" aria-label="Open the full-resolution physical deployment chain figure">
    <img src="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" alt="Physical deployment chain from the WT9011DCL-BT50 wearable IMU through BLE acquisition and Android on-device inference to activity recognition" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Physical deployment chain from wearable IMU acquisition to Android-side recognition. Select the image to view it at full resolution.</figcaption>
</figure>

## Field tests

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig08_app_field_test.png" target="_blank" rel="noopener" aria-label="Open the full-resolution app field-test figure">
    <img src="../../assets/manuscript-figures/fig08_app_field_test.png" alt="Privacy-preserving action renderings paired with Android recognition screenshots for background and five target activities" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Paper field-test examples for background motion, badminton, rope skipping, dumbbell fly, running, and table tennis.</figcaption>
</figure>

These examples document end-to-end app behavior across every target class.
They are implementation evidence, not an additional quantitative benchmark.

## Requirements

- Android Studio, or JDK 17 with an Android SDK;
- Android Gradle Plugin 8.1.0 and the included Gradle 8.0 wrapper;
- an Android device with BLE support; and
- a WT9011DCL-BT50 sensor for live acquisition.

The offline recognition path can be explored without the physical sensor by
copying a compatible derived segment text file to the phone.

## Build

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

## Sensor

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

## Offline

On the recognition view, either:

- run the built-in offline sample;
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
