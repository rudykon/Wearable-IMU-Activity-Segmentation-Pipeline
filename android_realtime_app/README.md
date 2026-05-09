# IMU Realtime Activity Recognition App

[中文说明](README.zh-CN.md)

Android app for real-time WT9011DCL-BT50 BLE IMU acquisition, visualization,
CSV recording, and on-device activity recognition with ONNX Runtime.

The app is designed for a motion-recognition competition workflow. It receives
100 Hz IMU frames over BLE, renders live charts and attitude views, and runs a
multi-scale 3s/5s/8s motion classifier on the phone.

## Features

- BLE scan/connect flow for WT9011DCL-BT50 sensors.
- Real-time charts for acceleration, angular velocity, and attitude.
- Dashboard, artificial horizon, compass, trajectory, and hand-motion views.
- CSV recording to the Android Downloads directory.
- On-device ONNX inference with selected 3s, 5s, and 8s models.
- Offline recognition for bundled sample data, derived segment TXT files under
  `android_realtime_app/motion_segments/`, or user-selected paper-format
  ACC/GYRO `.txt` files.
- Offline-style post-processing: zero-phase filtering, LBSA fusion, smoothing,
  Viterbi decoding, boundary refinement, and segment filtering.
- Chinese and English UI strings.

## Activity Classes

The bundled classifier outputs six classes:

1. No activity
2. Badminton
3. Jump rope
4. Fly
5. Running
6. Table tennis

## Repository Layout

```text
.
├── app/                         # Android application module
│   └── src/main/
│       ├── assets/              # ONNX models, normalization parameters, sample data
│       ├── java/com/imu/realtime/
│       └── res/
├── gradle/                      # Gradle wrapper
├── tools/desktop/               # Optional PC-side BLE debug tools
├── build.gradle
├── settings.gradle
└── README.md
```

## Build

Requirements:

- Android Studio or JDK 17 with the Android SDK installed.
- Android Gradle Plugin 8.1.0 / Gradle wrapper 8.0.
- Android device with BLE support.

Build from the repository root:

```bash
./gradlew assembleDebug
```

Android Studio will create `local.properties` automatically when the project is
opened. Do not commit that file.

## Run

1. Install the debug APK on an Android device.
2. Turn on the WT9011DCL-BT50 sensor.
3. Grant Bluetooth and location permissions when prompted.
4. Tap Scan, select the sensor whose name contains `WT`, and connect.
5. Use the bottom navigation to switch between charts, attitude, hand,
   trajectory, dashboard, and recognition views.
6. On the recognition view, run the built-in offline sample or choose a
   paper-format ACC/GYRO `.txt` file to classify offline data with the same
   pipeline. If you do not have a similar physical sensor, copy one of the
   derived segment TXT files from `android_realtime_app/motion_segments/` to
   the Android device and select it from the app to experience offline activity
   recognition.

Recorded CSV files are saved under the device Downloads directory with names
like `imu_yyyyMMdd_HHmmss.csv`.

## Model Assets

The current `.onnx` weights are stored directly in this repository. Git LFS is
not required because each bundled weight file is well below GitHub's regular
file-size limit.

The Android app loads these selected assets:

- `combined_model_3s_seed42.onnx`
- `combined_model_5s_seed123.onnx`
- `combined_model_8s_seed123.onnx`
- `norm_params_3s.json`
- `norm_params_5s.json`
- `norm_params_8s.json`

The legacy fallback model `hand_motion.onnx` and `norm_params.json` are also
included so the app can still run if the selected ensemble assets are replaced.

The repository includes derived segment TXT files under
`android_realtime_app/motion_segments/` (relative to this app directory:
`motion_segments/`). If you do not have a similar physical sensor, copy one of
these files to the Android device and select it from the recognition page to
experience the app-side offline activity-recognition path.

For model details, checksums, intended use, and limitations, see
`MODEL_CARD.md`. The bundled model weights and normalization files are covered
by `WEIGHTS_LICENSE`.

## Desktop Debug Tools

Optional Python tools live in `tools/desktop/`:

- `collect.py`: direct BLE collection with matplotlib plots.
- `server.py`: FastAPI + WebSocket service.
- `index.html`: browser dashboard for the WebSocket service.

See `tools/desktop/README.md` for usage.

## Hardware Development Notes

Open-source-ready WT9011DCL-BT50 integration notes extracted from the local
hardware-development materials are available in `docs/`:

- `docs/wt9011dcl-bt50-integration.zh-CN.md`: BLE UUIDs, packet parsing,
  conversion formulas, return-rate/bandwidth commands, and Android app mapping.
- `docs/source-materials.zh-CN.md`: source-material selection notes and files
  intentionally not copied into the repository.

## License

Code is provided under the MIT License. Bundled model weights and normalization
files are provided under `WEIGHTS_LICENSE`.
