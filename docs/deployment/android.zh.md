<section class="demo-page-hero android-demo-hero">
  <div>
    <p class="hero-kicker">端侧研究演示</p>
    <h1>Android Demo</h1>
    <p>安装公开体验版 APK，可连接 WT9011DCL-BT50 进行 BLE 采集并在手机端运行 ONNX 推理。离线体验无需传感器：下载合成 IMU 样例后，在 App 中选择该文件即可。</p>
    <div class="demo-facts" aria-label="Android Demo 兼容性">
      <span>Android 7.0+</span>
      <span>arm64-v8a</span>
      <span>端侧 ONNX</span>
    </div>
    <div class="demo-actions">
      <a class="demo-action primary" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/releases/download/android-demo-v1.0-preview/hls-har-android-demo-v1.0-arm64-v8a-debug.apk">下载 APK · 33 MB</a>
      <a class="demo-action" href="../../../assets/android/synthetic_activity_imu.tsv" download>下载示例数据</a>
      <a class="demo-action github" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/releases/tag/android-demo-v1.0-preview" target="_blank" rel="noopener">版本说明</a>
    </div>
  </div>
  <a class="demo-page-image" href="../../../assets/manuscript-figures/fig08_app_field_test.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的 Android 现场测试图">
    <img src="../../../assets/manuscript-figures/fig08_app_field_test.png" alt="Android 活动识别现场测试，覆盖背景和五种目标活动" loading="eager" decoding="async">
    <span>已完成现场测试</span>
  </a>
</section>

<nav class="demo-page-nav" aria-label="Android Demo 页面目录">
  <a href="#download-and-try">下载体验</a>
  <a href="#capabilities">功能</a>
  <a href="#offline">离线样例</a>
  <a href="#sensor">BLE 传感器</a>
  <a href="#build">自行构建</a>
</nav>

## 下载与体验 {#download-and-try}

<div class="demo-run-summary" aria-label="Android 体验版信息">
  <div class="demo-run-stat"><strong>v1.0</strong><span>体验版</span></div>
  <div class="demo-run-stat"><strong>33 MB</strong><span>调试版 APK</span></div>
  <div class="demo-run-stat"><strong>API 24+</strong><span>Android 7.0+</span></div>
  <div class="demo-run-stat"><strong>120 秒</strong><span>合成样例</span></div>
</div>

1. 下载并安装 **arm64-v8a APK**。
2. 将 `synthetic_activity_imu.tsv` 示例数据下载到手机。
3. 打开 App 的**识别**页，选择下载的文件并运行离线推理。

样例包含 100 Hz 下的 12,000 行计算机生成数据，不含参与者信息。文件选择器可直接
读取 `.tsv` 文件。

!!! warning "体验版 APK"

    当前安装包使用调试签名，用于研究体验，不是 Play Store 正式生产版本。请只安装
    上方链接提供的 APK。其 SHA-256 为
    `cdde56db9d915eb10918724d503597a84fb18deace096086fe87509f60348be6`。

## 功能 {#capabilities}

| 领域 | 已实现行为 |
| --- | --- |
| 采集 | 通过 BLE 扫描并连接 WT9011DCL-BT50 |
| 可视化 | 加速度、角速度、姿态、指南针、轨迹、手部动作与仪表盘视图 |
| 记录 | 将带时间戳的 IMU CSV 文件保存到 Android Downloads |
| 在线推理 | 在会话历史上运行选定的 3 秒 / 5 秒 / 8 秒集成模型 |
| 离线推理 | 识别可下载的合成样例或用户选择的 ACC/GYRO 文本文件 |
| 时序逻辑 | 滤波、LBSA 融合、平滑、Viterbi 解码、边界细化与片段过滤 |
| 本地化 | 中文与英文界面文本 |

## 运行

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的物理部署链路图">
    <img src="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" alt="从 WT9011DCL-BT50 可穿戴 IMU 经 BLE 采集与 Android 端侧推理到活动识别的物理部署链路" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">从可穿戴 IMU 采集到 Android 端识别的物理部署链路。点击图片可查看完整分辨率。</figcaption>
</figure>

## 环境要求

- Android Studio，或 JDK 17 与 Android SDK；
- Android Gradle Plugin 8.1.0 以及随附的 Gradle 8.0 wrapper；
- 支持 BLE 的 Android 设备；
- 用于实时采集的 WT9011DCL-BT50 传感器。

无需物理传感器也可以体验离线识别流程：只需将兼容的派生片段文本文件复制到手机。

## 构建 {#build}

在应用目录中运行：

~~~bash
cd android_realtime_app
./gradlew assembleDebug
~~~

构建过程会从公开的
[HF Model 仓库](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline)
下载 4 个 ONNX 权重并核对 SHA-256。GitHub 只保存应用源码与小型 JSON 配置，
不保存模型二进制文件。

Android Studio 会自动创建 `local.properties`。请勿提交该机器专用文件。

## 传感器 {#sensor}

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

## 离线 {#offline}

在识别视图中，可以：

- 选择下载的 `synthetic_activity_imu.tsv` 合成样例；
- 选择兼容论文格式的 ACC/GYRO `.txt` 文件；
- 将 `android_realtime_app/motion_segments/` 中的派生文件复制到设备后选择该文件。

这样无需实时 BLE 连接即可运行应用侧模型与时序层。

## 资产

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

## 类别

| 索引 | 中文 | 英文 |
| ---: | --- | --- |
| 0 | 无活动 | No activity |
| 1 | 羽毛球 | Badminton |
| 2 | 跳绳 | Jump rope |
| 3 | 飞鸟 | Fly |
| 4 | 跑步 | Running |
| 5 | 乒乓球 | Table tennis |

!!! warning "研究演示"

    应用与公开模型仅用于研究、教学和可复现原型评估，不声称具备生产安全性、
    临床有效性或跨设备泛化能力。

## BLE 工具

`android_realtime_app/tools/desktop/` 下的可选工具包括：

- `collect.py`：直接 BLE 采集与 matplotlib 绘图；
- `server.py`：FastAPI + WebSocket 服务；
- `index.html`：浏览器仪表盘。

硬件协议细节、UUID、数据包解析、单位转换与 Android 映射记录在应用的 `docs/`
目录中。
