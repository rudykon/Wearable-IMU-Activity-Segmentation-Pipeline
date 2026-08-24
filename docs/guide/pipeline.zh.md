# 流水线

流水线把长时 IMU 记录转换成带时间戳的活动记录。

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="../../../assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的总体框架图">
    <img src="../../../assets/fig02_overall_framework.png" alt="仓库现有可穿戴 IMU 活动分割总体框架图" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">六路信号 → 三种窗口 → 融合 → 时间线 → 活动记录。</figcaption>
</figure>

## 输入输出

| 项目 | 系统要求或返回的内容 |
| --- | --- |
| 采样 | 100 Hz 加速度计与陀螺仪 |
| 模型输入 | `(window, 6)` 物理单位 IMU 通道 |
| 窗口尺度 | 3 秒（300 个样本）、5 秒（500）、8 秒（800） |
| 窗口步长 | 1 秒（100 个样本） |
| 内部类别 | 背景 + 五种前景活动 |
| 公开输出 | `user_id, category, start, end` |
| 评估 | IoU > 0.5 时同类别一对一匹配 |

## 1. 读取

`DataReader` 加载制表符分隔的 `.txt` 会话。必需通道：

~~~text
ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z
~~~

`ACC_TIME` 保存毫秒时间戳。

## 2. 预处理

Python 与 Android 使用相同的通道顺序和归一化：

- 低通 Butterworth 滤波；
- 固定通道顺序；
- 各尺度独立的归一化参数；
- 保留时间戳的窗口构建。

每个检查点必须配套对应的归一化文件。

## 3. 分窗

| 尺度 | 样本数 | 作用 |
| --- | ---: | --- |
| 3 秒 | 300 | 捕获短时局部运动特征 |
| 5 秒 | 500 | 平衡局部细节与活动上下文 |
| 8 秒 | 800 | 稳定更长或重复性动作 |

所有尺度都以一秒为步长。

## 4. 分类

`CombinedModel` 有六个类别、五个部分：

1. 核大小为 3、7、15 的并行一维卷积分支；
2. 拼接后的多分辨率特征图；
3. 带自适应池化的更深 CNN 路径；
4. 双层双向 LSTM 路径；
5. 融合分类头。

源码还保留用于实验的两阶段分类器。

!!! info "CNN + BiLSTM"

    CNN 捕捉局部模式，BiLSTM 捕捉前后顺序。

## 5. 融合

三条概率时间线共用一秒网格。**LBSA** 在切换附近侧重短窗口，在稳定动作中侧重长窗口。

集成配置明确记录在：

~~~text
saved_models/ensemble_config.json
~~~

它记录模型和时间线设置。

## 6. 解码

**时间记录层（TRL）**执行：

- 多尺度融合；
- 概率平滑；
- Viterbi 序列解码；
- 边界细化；
- 短间隔处理；
- 重叠消解；
- 置信度过滤；
- 最终 Top-K 或持续时间策略。

输出是一组连续活动区间。

## 7. 导出

`DataOutput` 写入 Excel 工作簿：

~~~text
user_id, category, start, end
~~~

背景类别不导出。

## Python 与 Android

| Python 研究流水线 | Android 应用 |
| --- | --- |
| PyTorch 检查点（`.pth`） | ONNX 模型（`.onnx`） |
| Pickle 归一化资产 | JSON 归一化资产 |
| 来自 `data/signals/` 的批量文件 | BLE 历史数据或选定离线文件 |
| 实验脚本与评估器 | 实时视图与端侧时间线 |
| XLSX 片段输出 | 带置信度的 UI 片段 |

两端使用相同的通道、窗口、标签和记录格式。
