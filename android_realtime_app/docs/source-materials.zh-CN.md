# WT9011DCL-BT50 资料筛选说明

本文记录从开源前物理实现资料目录中抽取到本仓库的内容范围。

源目录：

```text
/home/wangrj/桌面/KMH/运动识别竞赛/5.运动识别的物理实现
```

目标目录：

```text
/home/wangrj/桌面/KMH/运动识别竞赛/4.开源代码/app/docs
```

## 已抽取内容

抽取后的开发笔记位于：

```text
docs/wt9011dcl-bt50-integration.zh-CN.md
```

主要来源包括：

- `WT9011DCL-BT50中文开发笔记.md`
- `WT9011DCL-BT50_资料/WT9011DCL-BT50 BLE SDK说明.md`
- `WT9011DCL-BT50_资料/README.md`
- `WT9011DCL-BT50_资料/WT9011DCL-BT50 Communication Protocol.pdf`
- `WT9011DCL-BT50_资料/WT9011DCL-BT50 Product Specifications.pdf`
- 本仓库当前 Android 代码：
  - `app/src/main/java/com/imu/realtime/BleManager.kt`
  - `app/src/main/java/com/imu/realtime/MotionClassifier.kt`
  - `tools/desktop/collect.py`
  - `tools/desktop/server.py`

抽取重点是 App 开发真正需要的事实：

- BLE Service/Notify/Write UUID。
- `55 61` 默认 IMU 数据帧布局。
- `acc + gyro + angle` 的小端 `int16` 换算规则。
- 读寄存器、设置回传速率、设置带宽、校准、保存配置等常用命令。
- 当前 App 对 100 Hz 数据流和 3s/5s/8s 模型窗口的依赖。

## 未并入内容

以下资料保留在原始目录，不复制到开源仓库。

| 类型 | 示例 | 未并入原因 |
| --- | --- | --- |
| 厂商 PDF 原件 | `WT9011DCL Manual.pdf`, `WT9011DCL-BT50 Communication Protocol.pdf` | 授权边界不清；只抽取必要事实并注明来源类别 |
| Windows 软件包 | `Standard Software for Windows PC.zip`, `Download the WitMotion Software.zip` | 体积大，且属于厂商二进制软件 |
| 驱动和工具 exe/zip | `CH340 & CP2102 Driver.zip`, `sscom5.13.1.exe` | 与 Android App 构建无关，且不适合重新分发 |
| 官方 SDK 压缩包 | `WitBluetooth_BWT901BLE5_0-main.zip` | 可通过官方仓库获取；本项目已有最小 Android/Python 实现 |
| 临时状态 | `.omx/`, `.gradle/`, `tmp/` | 本地工作状态，不属于项目文档 |
| 旧版 Android 工程 | `实时采集/android_app/` | 当前开源 App 已包含整理后的实现，复制会制造重复代码 |
| 截图和示意图 | `识别app截图.png`, `BLE采集实物图.png`, `物理实现示意图.png` | 对代码使用不是必需；部分图像包含产品渲染或展示元素，授权边界不清 |

## 外部资料入口

原始资料清单中记录过以下上游入口，后续需要完整厂商资料时可以从官方来源获取：

- WITMOTION BLE SDK GitHub 仓库：`https://github.com/WITMOTION/WitBluetooth_BWT901BLE5_0`
- WITMOTION 产品页：`https://www.wit-motion.com/proinertialsensor/31.html`
- WITMOTION 资料页和 SDK 页：见原始目录中的 `WT9011DCL-BT50_资料/README.md`

这些链接仅作为资料定位入口；开源仓库不重新分发厂商软件包和完整手册。
