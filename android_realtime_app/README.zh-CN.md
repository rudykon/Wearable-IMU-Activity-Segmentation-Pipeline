# IMU 实时运动识别 App

[English README](README.md)

这是一个面向 WT9011DCL-BT50 蓝牙 IMU 的 Android 应用，用于实时采集、可视化、CSV 录制和端侧运动识别。应用通过 BLE 接收约 100 Hz 的 IMU 数据帧，在手机端展示曲线、姿态、轨迹、手部运动等视图，并使用 ONNX Runtime 运行 3s/5s/8s 多尺度运动识别模型。

## 功能

- 扫描并连接名称包含 `WT` 的 WT9011DCL-BT50 BLE 传感器。
- 实时显示加速度、角速度和姿态角曲线。
- 提供图表、姿态、手部、轨迹、仪表盘和识别六个页面。
- 支持将 IMU 数据录制为 CSV 文件，保存到 Android Downloads 目录。
- 在 Android 端使用 ONNX Runtime 进行模型推理。
- 支持运行内置离线示例，或选择 `android_realtime_app/motion_segments/`
  中的真实切片 TXT 文件、论文格式 ACC/GYRO `.txt` 文件进行离线识别。
- 识别流程包含零相位滤波、多尺度概率融合、平滑、Viterbi 解码、边界细化和片段过滤。
- 提供中文和英文界面字符串。

## 识别类别

内置模型输出 6 个类别：

1. 无活动
2. 羽毛球
3. 跳绳
4. 飞鸟
5. 跑步
6. 乒乓球

## 目录结构

```text
.
├── app/                         # Android 应用模块
│   └── src/main/
│       ├── assets/              # ONNX 模型、归一化参数和示例数据
│       ├── java/com/imu/realtime/
│       └── res/
├── gradle/                      # Gradle Wrapper
├── tools/desktop/               # 可选的电脑端 BLE 调试工具
├── build.gradle
├── settings.gradle
└── README.md
```

## 构建

环境要求：

- Android Studio，或已安装 Android SDK 的 JDK 17 环境。
- Android Gradle Plugin 8.1.0 / Gradle Wrapper 8.0。
- 一台支持 BLE 的 Android 真机。

在仓库根目录执行：

```bash
./gradlew assembleDebug
```

使用 Android Studio 打开项目时会自动生成 `local.properties`。该文件包含本机 Android SDK 路径，不应提交到 GitHub。

## 运行

1. 将 debug APK 安装到 Android 真机。
2. 打开 WT9011DCL-BT50 传感器。
3. 按提示授予蓝牙和定位权限。
4. 点击“扫描”，选择名称包含 `WT` 的设备并连接。
5. 通过底部导航切换图表、姿态、手部、轨迹、仪表盘和识别页面。
6. 在识别页面可运行内置离线示例，或选择论文格式 ACC/GYRO `.txt` 文件，使用同一套论文推理流程进行离线识别。没有类似实物传感器时，可以将 `android_realtime_app/motion_segments/` 下的真实切片 TXT 文件复制到 Android 设备，再在 App 中选择该文件体验离线数据运动识别。

录制的 CSV 文件会保存到设备 Downloads 目录，文件名格式类似：

```text
imu_yyyyMMdd_HHmmss.csv
```

## 模型资产

当前 `.onnx` 权重文件直接保存在本仓库中。由于每个权重文件都明显低于 GitHub 普通文件大小限制，因此不需要 Git LFS。

Android 应用会加载以下模型和归一化参数：

- `combined_model_3s_seed42.onnx`
- `combined_model_5s_seed123.onnx`
- `combined_model_8s_seed123.onnx`
- `norm_params_3s.json`
- `norm_params_5s.json`
- `norm_params_8s.json`

仓库还包含 fallback 模型：

- `hand_motion.onnx`
- `norm_params.json`

这样在替换多尺度模型资产时，应用仍然可以退回到单模型推理路径。

本仓库已在 `android_realtime_app/motion_segments/` 下提供真实切片 TXT 示例（相对于本 App 目录为 `motion_segments/`）。没有类似实物传感器时，可以将这些 TXT 文件复制到 Android 设备，在 App 的识别页面通过“选择TXT文件”载入，从而体验离线数据运动识别。

模型用途、输入格式、校验和、已知限制等信息见 `MODEL_CARD.md`。随仓库分发的模型权重和归一化参数适用 `WEIGHTS_LICENSE`。训练数据不包含在本 app 仓库中。

## 桌面调试工具

可选 Python 调试工具位于 `tools/desktop/`：

- `collect.py`：直接连接 BLE 设备，并用 matplotlib 显示实时曲线。
- `server.py`：FastAPI + WebSocket 本地服务。
- `index.html`：浏览器实时看板。

使用方式见：

```text
tools/desktop/README.md
```

## 硬件开发资料

从开源前的 WT9011DCL-BT50 物理实现资料中抽取出的开发笔记位于：

- `docs/wt9011dcl-bt50-integration.zh-CN.md`：BLE UUID、数据帧解析、单位换算、回传速率/带宽/校准命令，以及当前 Android App 的接入关系。
- `docs/source-materials.zh-CN.md`：原始资料筛选说明，以及为什么没有把厂商 PDF、Windows 软件包、驱动、SDK 压缩包和临时状态直接复制进开源仓库。

## 开源说明

代码使用 MIT License。随仓库分发的模型权重和归一化参数使用 `WEIGHTS_LICENSE`。
