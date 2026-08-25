---
hide:
  - toc
---

<section class="home-hero">
  <p class="hero-kicker">可穿戴 IMU 活动记录</p>
  <h1 class="paper-title">基于多尺度仲裁与时间记录层的端到端可穿戴 IMU 片段级活动识别系统</h1>
  <p class="hero-lead">将连续 100 Hz 腕部 IMU 信号转换为包含活动类别与起止时间的片段记录。</p>
  <div class="hero-actions">
    <a class="hero-button primary" href="deployment/hugging-face/">演示</a>
    <a class="hero-button" href="guide/pipeline/">方法</a>
    <a class="hero-button" href="research/paper/">结果</a>
  </div>
</section>

## 问题与输出

窗口级预测不能直接形成可靠的活动记录。面对长时传感器数据，系统还必须恢复活动类别、事件数量、持续时间与起止边界。

<div class="record-transform" aria-label="系统输入与输出">
  <code>六通道腕部 IMU 流</code>
  <span aria-hidden="true">→</span>
  <code>{活动, 开始, 结束}</code>
</div>

当前任务在 100 Hz 腕部加速度计和陀螺仪数据上评估五类运动。背景运动参与解码，但不会作为运动记录输出。

## 方法

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的总体框架图">
    <img src="assets/fig02_overall_framework.png" alt="腕部 IMU 输入经过三个分尺度模型、LBSA 与 TRL 后生成活动记录的总体框架" loading="eager" decoding="async">
  </a>
  <figcaption class="pipeline-caption">IMU 流 → 3/5/8 秒后验 → LBSA → TRL → 活动记录。</figcaption>
</figure>

1. **多尺度后验模型**：以分尺度 CNN–BiLSTM 分析 3、5、8 秒视图。
2. **局部边界尺度仲裁（LBSA）**：在切换附近加强短窗口证据，在稳定区域保留长窗口上下文。
3. **时间记录层（TRL）**：对融合时间线进行平滑、解码、合并、边界修正与过滤，生成确定性记录。

[阅读完整方法](guide/pipeline.md){ .md-button }

## 证据

<div class="metric-strip metric-strip--three">
  <div class="metric"><strong>259.6 h</strong><span>传感器数据</span></div>
  <div class="metric"><strong>37 / 114</strong><span>测试记录 / 片段</span></div>
  <div class="metric"><strong>0.90</strong><span>Micro-F1</span></div>
</div>

| 固定工作点 | 平均用户 F1 | Micro-F1 | TP / FP / FN |
| --- | ---: | ---: | ---: |
| **LBSA + TRL** | **0.89** | **0.90** | **99 / 7 / 15** |

| 已评估范围 | 需要重新验证 |
| --- | --- |
| 研究所用设备、佩戴位置和协议下的五类活动 | 新设备、佩戴位置、用户、活动类别与部署条件 |

[查看完整结果与失败案例](research/paper.md){ .md-button }

## 复现

| 资源 | 用途 |
| --- | --- |
| [演示](deployment/hugging-face.md) | 运行合成样例并查看活动记录输出 |
| [模型](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline) | 下载已校验的 PyTorch 与 ONNX 权重 |
| [快速开始](getting-started/quickstart.md) | 验证公开代码包并运行获授权数据推理 |
| [Android](deployment/android.md) | 构建 BLE 与 ONNX 研究原型 |

GitHub 不分发参与者记录。[复现指南](reproduce.md)明确区分了公开验证与需要授权数据的完整流程。
