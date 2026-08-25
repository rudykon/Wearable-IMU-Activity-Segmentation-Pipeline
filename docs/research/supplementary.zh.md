# 补充分析

<p class="research-lead">以下内容用于诊断机制与工程表现，与独立外部测试的主要结果分开报告。</p>

## 多尺度表征

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig05_multiscale_tsne_diagnostic.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的多尺度 t-SNE 图">
    <img src="../../../assets/manuscript-figures/fig05_multiscale_tsne_diagnostic.png" alt="3、5、8 秒窗口模型表征的 t-SNE 图" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">三种尺度学习到互补的表征结构。</figcaption>
</figure>

短窗口保留切换细节，长窗口补充稳定运动上下文。t-SNE 只用于定性诊断，不是性能证据。

## TRL 开发集诊断

<figure class="paper-figure portrait">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig06_outer_split_boundary_summary.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的 TRL 重复划分诊断图">
    <img src="../../../assets/manuscript-figures/fig06_outer_split_boundary_summary.png" alt="重复开发集划分 F1 分布与 TRL 累积边界诊断" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">重复开发集划分用于分离时间记录构建的作用。</figcaption>
</figure>

在 50 次开发集划分中，TRL 将平均 F1 从 **0.802 提高到 0.913**，每小时误报从 **0.862 降至 0.190**，匹配 IoU 从 **0.835 变为 0.843**。这些数值是开发集诊断，不能当作独立外部测试性能报告。

## Android 工程证据

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig08_app_field_test.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的 Android 现场测试图">
    <img src="../../../assets/manuscript-figures/fig08_app_field_test.png" alt="隐私保护动作示意与 Android 识别截图" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">App 端检查覆盖背景运动与五类目标活动。</figcaption>
</figure>

Android 原型连接 WT9011DCL-BT50 BLE IMU，并运行选定 ONNX 模型与时间记录层。完整三模型流程在被测设备上平均每条记录耗时 **2.1 秒**。这说明工程实现可行，不构成新的基准测试。

## 公共数据集可移植性

下列数据集与腕部运动协议并不一致。实验只检查时间接口能否配合数据集专用模型与策略重新配置。

| 数据集 | Argmax F1 | TRL F1 | Argmax FP/h | TRL FP/h |
| --- | ---: | ---: | ---: | ---: |
| HAR70+ | 0.70 | 0.70 | 47 | 45 |
| WISDM-phone | 0.05 | 0.37 | 83 | 5.7 |
| PAMAP2 | 0.09 | 0.53 | 90 | 9.3 |
| OPPORTUNITY | 0.29 | 0.29 | 320 | 110 |

这些结果不是排行榜对比，也不能证明 HLS-HAR 结果可直接迁移。返回[主要结果](paper.md)，或打开[复现指南](../reproduce.md)。
