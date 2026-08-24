# 研究结果

<p class="research-lead">论文评价完整活动记录：类别、开始、结束、时长和次数。</p>

<div class="metric-strip paper-metrics">
  <div class="metric"><strong>137</strong><span>条记录</span></div>
  <div class="metric"><strong>259.6 h</strong><span>传感器数据</span></div>
  <div class="metric"><strong>0.89</strong><span>平均用户 F1</span></div>
  <div class="metric"><strong>0.90</strong><span>Micro-F1</span></div>
</div>

!!! note "评估协议"

    分数衡量完整记录，不是窗口。模型和规则在 37 条外部记录测试前已固定。

## 记录错误

窗口分类正确，仍可能拆分活动、移动边界或产生误报。

<figure class="paper-figure">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig01_window_to_record_gap.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的窗口到记录落差图">
    <img src="../../../assets/manuscript-figures/fig01_window_to_record_gap.png" alt="后验概率轨迹、朴素提取得到的碎片记录以及时间记录层稳定后的记录列表" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">图 1。窗口预测可能把最终记录拆碎。</figcaption>
</figure>

## 方法

<div class="research-grid">
  <article class="research-card">
    <span class="research-card-kicker">模型</span>
    <h3>CNN–BiLSTM</h3>
    <p>分类 3、5、8 秒窗口。</p>
  </article>
  <article class="research-card">
    <span class="research-card-kicker">融合</span>
    <h3>LBSA</h3>
    <p>在边界内外选择合适尺度。</p>
  </article>
  <article class="research-card">
    <span class="research-card-kicker">解码</span>
    <h3>TRL</h3>
    <p>平滑、合并、修正并过滤记录。</p>
  </article>
</div>

预测与同类别标注按 IoU > 0.5 一对一匹配。拆分、合并、时间和类别错误都会扣分。

## 数据划分

| 数据角色 | 记录数 | 用途 |
| --- | ---: | --- |
| 训练集 | 80 | 模型拟合与训练阶段检查点选择 |
| 开发／校准集 | 20 | 时序策略校准、诊断与划分分离选择 |
| 独立外部测试集 | 37 | 工作点冻结后进行一次最终评估 |

语料含 **4680 万**个有效样本和五类活动；外部测试集有 114 个标注片段。

## 外部测试

<figure class="paper-figure portrait">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig04_external_variant_comparison.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的外部测试变体对比图">
    <img src="../../../assets/manuscript-figures/fig04_external_variant_comparison.png" alt="五种固定系统变体的外部测试平均用户 F1、Micro-F1、假阳性和假阴性计数" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">图 4。五种固定方案测试同一组 37 条记录。</figcaption>
</figure>

| 固定变体 | 平均用户 F1 | 95% CI | Micro-F1 | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: |
| 5 秒 + 8 秒 + TRL | 0.88 | 0.80-0.94 | 0.88 | 98 / 11 / 16 |
| 三模型平均 + TRL | 0.88 | 0.80-0.94 | 0.89 | 98 / 9 / 16 |
| 三模型加权 + TRL | 0.89 | 0.81-0.94 | 0.89 | 99 / 9 / 15 |
| LBSA + 宽松 Top-K | 0.88 | 0.80-0.95 | 0.88 | 103 / 17 / 11 |
| **LBSA + TRL** | **0.89** | **0.82-0.94** | **0.90** | **99 / 7 / 15** |

<div class="result-callout"><strong>结论。</strong>LBSA + TRL 保持最高的四舍五入 F1，同时减少误报。</div>

### 分类结果

| 活动 | 真实片段 | TP / FP / FN | 精确率 | 召回率 | F1 | 匹配 IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 羽毛球 | 32 | 26 / 3 / 6 | 0.90 | 0.81 | 0.85 | 0.87 |
| 跳绳 | 20 | 19 / 0 / 1 | 1.00 | 0.95 | 0.97 | 0.84 |
| 哑铃飞鸟 | 20 | 19 / 1 / 1 | 0.95 | 0.95 | 0.95 | 0.78 |
| 跑步 | 20 | 18 / 1 / 2 | 0.95 | 0.90 | 0.92 | 0.82 |
| 乒乓球 | 22 | 17 / 2 / 5 | 0.90 | 0.77 | 0.83 | 0.86 |

乒乓球召回率最低，跳绳 F1 最高。

## 时间尺度

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig05_multiscale_tsne_diagnostic.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的多尺度 t-SNE 图">
    <img src="../../../assets/manuscript-figures/fig05_multiscale_tsne_diagnostic.png" alt="3 秒、5 秒和 8 秒窗口模型在六种活动状态上的倒数第二层表征 t-SNE 图" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">图 5。三种尺度学习到互补结构。</figcaption>
</figure>

短窗口保留边界，长窗口提供上下文。t-SNE 是**定性诊断**，不是性能证据。

## TRL 效果

<figure class="paper-figure portrait">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig06_outer_split_boundary_summary.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的 TRL 外层划分诊断图">
    <img src="../../../assets/manuscript-figures/fig06_outer_split_boundary_summary.png" alt="重复外层划分 F1 分布，以及时间记录层累积策略的边界质量与假阳性代价" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">图 6。主要增益来自时序一致性和置信裁剪。</figcaption>
</figure>

在 50 次开发划分中，TRL 把平均 F1 从 **0.802 提高到 0.913**。每小时误报从
**0.862 降至 0.190**，匹配 IoU 从 **0.835 变为 0.843**。这些是开发诊断，
不是外部测试结果。

## 案例

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig07_representative_timeline_cases.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的代表性时间线图">
    <img src="../../../assets/manuscript-figures/fig07_representative_timeline_cases.png" alt="一个成功案例和一个部分失败案例中真实时间线及四种固定融合时序变体的对比" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">图 7。一个成功案例和一个失败案例。</figcaption>
</figure>

融合改善了一个边界；弱证据仍造成漏检和误报。

## Android 测试

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig08_app_field_test.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的 App 现场测试图">
    <img src="../../../assets/manuscript-figures/fig08_app_field_test.png" alt="隐私保护动作模型与 Android 识别截图，覆盖背景和五种目标活动" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">图 8。Android 上的背景和五类活动测试。</figcaption>
</figure>

Android 应用连接 WT9011DCL-BT50 BLE IMU。完整三模型流水线在被测设备上平均
每条记录耗时 **2.1 秒**。

??? info "公共数据集——仅检查可移植性"

    这些数据集不匹配腕部运动协议。结果只检验时序接口能否重新配置。

    | 数据集 | Argmax F1 | TRL F1 | Argmax FP/h | TRL FP/h |
    | --- | ---: | ---: | ---: | ---: |
    | HAR70+ | 0.70 | 0.70 | 47 | 45 |
    | WISDM-phone | 0.05 | 0.37 | 83 | 5.7 |
    | PAMAP2 | 0.09 | 0.53 | 90 | 9.3 |
    | OPPORTUNITY | 0.29 | 0.29 | 320 | 110 |

## 边界

!!! warning "评估范围"

    - 结果衡量**片段记录**，不代表临床、训练或安全价值。
    - 新设备、人群、佩戴位置和协议需要重测。
    - 证据来自 20 条开发记录和 37 条外部测试记录。
    - 密集会话和相邻同类事件仍然困难。

## 复现

- [运行公开冒烟测试](../getting-started/quickstart.md)
- [查看流水线架构](../guide/pipeline.md)
- [理解片段级评估器](../guide/evaluation.md)
- [构建 Android 演示](../deployment/android.md)
- [核对数据与模型资产边界](../reference/assets.md)
