# Android 应用

`android_realtime_app/` 模块是 Python 研究流水线的可部署配套应用。它支持
WT9011DCL-BT50 BLE 采集、实时可视化、CSV 记录、离线文件识别与端侧 ONNX 推理。

## 功能

| 领域 | 已实现行为 |
| --- | --- |
| 采集 | 通过 BLE 扫描并连接 WT9011DCL-BT50 |
| 可视化 | 加速度、角速度、姿态、指南针、轨迹、手部动作与仪表盘视图 |
| 记录 | 将带时间戳的 IMU CSV 文件保存到 Android Downloads |
| 在线推理 | 在会话历史上运行选定的 3 秒 / 5 秒 / 8 秒集成模型 |
| 离线推理 | 识别内置样例或用户选择的 ACC/GYRO 文本文件 |
| 时序逻辑 | 滤波、LBSA 融合、平滑、Viterbi 解码、边界细化与片段过滤 |
| 本地化 | 中文与英文界面文本 |

## 运行链路

~~~text
WT9011DCL-BT50
       │ BLE · 100 Hz
       ▼
Android 传感器缓冲区
       │ 六个物理 IMU 通道
       ▼
3 秒 / 5 秒 / 8 秒 ONNX Runtime 模型
       │ 对齐后的概率
       ▼
多尺度时序后处理
       │
       ▼
实时活动时间线 + 置信度
~~~

## 环境要求

- Android Studio，或 JDK 17 与 Android SDK；
- Android Gradle Plugin 8.1.0 以及随附的 Gradle 8.0 wrapper；
- 支持 BLE 的 Android 设备；
- 用于实时采集的 WT9011DCL-BT50 传感器。

无需物理传感器也可以体验离线识别流程：只需将兼容的派生片段文本文件复制到手机。

## 构建

在应用目录中运行：

~~~bash
cd android_realtime_app
./gradlew assembleDebug
~~~

Android Studio 会自动创建 `local.properties`。请勿提交该机器专用文件。

## 连接传感器运行

1. 在支持 BLE 的 Android 设备上安装调试 APK。
2. 打开 WT9011DCL-BT50 传感器。
3. 授予所需的蓝牙与位置权限。
4. 点击 **扫描**，选择名称中包含 `WT` 的设备并连接。
5. 使用底部导航查看图表、姿态、手部、轨迹、仪表盘与识别视图。
6. 开始识别或记录 CSV 会话。

记录文件使用如下名称：

~~~text
imu_yyyyMMdd_HHmmss.csv
~~~

并保存在设备的 Downloads 目录中。

## 离线识别

在识别视图中，可以：

- 运行内置离线样例；
- 选择兼容论文格式的 ACC/GYRO `.txt` 文件；
- 将 `android_realtime_app/motion_segments/` 中的派生文件复制到设备后选择该文件。

这样无需实时 BLE 连接即可运行应用侧模型与时序层。

## 内置推理资产

| 资产 | 作用 |
| --- | --- |
| `combined_model_3s_seed42.onnx` | 选定的 3 秒模型 |
| `combined_model_5s_seed123.onnx` | 选定的 5 秒模型 |
| `combined_model_8s_seed123.onnx` | 选定的 8 秒模型 |
| `norm_params_3s.json` | 3 秒归一化参数 |
| `norm_params_5s.json` | 5 秒归一化参数 |
| `norm_params_8s.json` | 8 秒归一化参数 |
| `hand_motion.onnx` | 旧版回退模型 |
| `norm_params.json` | 旧版回退归一化参数 |

详细的[模型卡](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/android_realtime_app/MODEL_CARD.md)
包含输入假设、输出映射、校验和、预期用途与已知限制。

## 输出类别

| 索引 | 中文 | 英文 |
| ---: | --- | --- |
| 0 | 无活动 | No activity |
| 1 | 羽毛球 | Badminton |
| 2 | 跳绳 | Jump rope |
| 3 | 飞鸟 | Fly |
| 4 | 跑步 | Running |
| 5 | 乒乓球 | Table tennis |

!!! warning "研究演示"

    应用与内置模型仅用于研究、教学和可复现原型评估，不声称具备生产安全性、
    临床有效性或跨设备泛化能力。

## 桌面 BLE 工具

`android_realtime_app/tools/desktop/` 下的可选工具包括：

- `collect.py`：直接 BLE 采集与 matplotlib 绘图；
- `server.py`：FastAPI + WebSocket 服务；
- `index.html`：浏览器仪表盘。

硬件协议细节、UUID、数据包解析、单位转换与 Android 映射记录在应用的 `docs/`
目录中。
