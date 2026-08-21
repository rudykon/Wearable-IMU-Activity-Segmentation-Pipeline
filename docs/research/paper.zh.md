# 论文要点

<p class="research-lead"><strong>《An End-to-End Wearable IMU System for Segment-Level Activity Recognition via Multi-Scale Arbitration and a Temporal Record Layer》</strong>将可穿戴活动识别研究为一条完整的“感知到记录”测量链。核心问题不仅是短窗口能否分类正确，而是一次长时训练能否变成标签、边界、持续时间和事件计数均可用的可靠记录列表。</p>

<div class="metric-strip paper-metrics">
  <div class="metric"><strong>137</strong><span>条长时记录</span></div>
  <div class="metric"><strong>259.6 h</strong><span>连续感知数据</span></div>
  <div class="metric"><strong>0.89</strong><span>平均用户 F1</span></div>
  <div class="metric"><strong>0.90</strong><span>Micro-F1</span></div>
</div>

!!! note "如何阅读这些证据"

    主要结果是**片段记录指标**，并非窗口分类准确率。模型、融合方式与时序策略在
    使用 37 条独立外部测试记录的标签进行最终评分前已经固定。下文会明确区分
    内部开发诊断与独立外部测试结果。

## 窗口到记录的落差

局部看似合理的后验轨迹仍可能生成错误的记录列表。短暂置信下陷会把一段活动拆成
两条记录，弱运动可能变成短假阳性，以窗口中心为基础的决策也可能使正确类别的
边界发生偏移。这些错误会直接影响活动次数、持续时间和训练时间线。

<figure class="paper-figure">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig01_window_to_record_gap.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的窗口到记录落差图">
    <img src="../../../assets/manuscript-figures/fig01_window_to_record_gap.png" alt="后验概率轨迹、朴素提取得到的碎片记录以及时间记录层稳定后的记录列表" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">论文图 1。窗口后验看似合理时，朴素记录提取仍会产生假分裂、短假阳性与边界偏移；TRL 负责合并并稳定最终记录列表。</figcaption>
</figure>

## 系统的三项关键作用

<div class="research-grid">
  <article class="research-card">
    <span class="research-card-kicker">局部证据</span>
    <h3>CNN + BiLSTM 后验生成器</h3>
    <p>多卷积核一维卷积捕获局部运动模式，双向循环路径建模这些模式在 3 秒、5 秒或 8 秒窗口内的演化。</p>
  </article>
  <article class="research-card">
    <span class="research-card-kicker">尺度不确定性</span>
    <h3>局部边界尺度仲裁</h3>
    <p>LBSA 在稳定区域保留长窗口证据，在候选活动转换附近提高 3 秒分支的贡献，以兼顾稳定性与边界定位。</p>
  </article>
  <article class="research-card">
    <span class="research-card-kicker">记录构建</span>
    <h3>确定性时间记录层</h3>
    <p>TRL 将平滑、受约束 Viterbi、同类间隔合并、边界细化、重叠处理、持续时间过滤、置信裁剪与剪枝公开为可审计步骤。</p>
  </article>
</div>

系统报告的是可变长度的 `(活动, 开始, 结束)` 记录集合。仅当预测与真实片段
类别一致且 IoU 大于 0.5 时，二者才可进行一对一匹配。因此，碎片化、错误合并、
边界偏移和类别错误都会反映在评分中。

## 固定评估协议

| 数据角色 | 记录数 | 用途 |
| --- | ---: | --- |
| 训练集 | 80 | 模型拟合与训练阶段检查点选择 |
| 开发／校准集 | 20 | 时序策略校准、诊断与划分分离选择 |
| 独立外部测试集 | 37 | 工作点冻结后进行一次最终评估 |

完整语料包含约 **4680 万**个 100 Hz ACC/GYRO 有效样本，覆盖羽毛球、跳绳、
哑铃飞鸟、跑步和乒乓球五类前景活动。外部测试集含 114 个标注活动片段。

## 独立外部测试结果

<figure class="paper-figure portrait">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig04_external_variant_comparison.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的外部测试变体对比图">
    <img src="../../../assets/manuscript-figures/fig04_external_variant_comparison.png" alt="五种固定系统变体的外部测试平均用户 F1、Micro-F1、假阳性和假阴性计数" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">论文图 4。五个固定工作点使用同一组 37 条外部记录。LBSA + TRL 获得最高的四舍五入平均用户 F1，并在三尺度变体中具有最低假阳性计数。</figcaption>
</figure>

| 固定变体 | 平均用户 F1 | 95% CI | Micro-F1 | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: |
| 5 秒 + 8 秒 + TRL | 0.88 | 0.80-0.94 | 0.88 | 98 / 11 / 16 |
| 三模型平均 + TRL | 0.88 | 0.80-0.94 | 0.89 | 98 / 9 / 16 |
| 三模型加权 + TRL | 0.89 | 0.81-0.94 | 0.89 | 99 / 9 / 15 |
| LBSA + 宽松 Top-K | 0.88 | 0.80-0.95 | 0.88 | 103 / 17 / 11 |
| **LBSA + TRL** | **0.89** | **0.82-0.94** | **0.90** | **99 / 7 / 15** |

<div class="result-callout"><strong>解释。</strong>放宽剪枝可以找回更多真实片段，但也会明显增加假阳性；最终冻结的 LBSA + TRL 工作点在保持最强四舍五入 F1 的同时抑制了虚假记录。</div>

### 外部测试分类别结果

| 活动 | 真实片段 | TP / FP / FN | 精确率 | 召回率 | F1 | 匹配 IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 羽毛球 | 32 | 26 / 3 / 6 | 0.90 | 0.81 | 0.85 | 0.87 |
| 跳绳 | 20 | 19 / 0 / 1 | 1.00 | 0.95 | 0.97 | 0.84 |
| 哑铃飞鸟 | 20 | 19 / 1 / 1 | 0.95 | 0.95 | 0.95 | 0.78 |
| 跑步 | 20 | 18 / 1 / 2 | 0.95 | 0.90 | 0.92 | 0.82 |
| 乒乓球 | 22 | 17 / 2 / 5 | 0.90 | 0.77 | 0.83 | 0.86 |

描述性分类别结果显示，剩余错误并不均匀：固定外部集上乒乓球召回率最低，跳绳
则具有最高的片段 F1。

## 为什么需要多个时间尺度

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig05_multiscale_tsne_diagnostic.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的多尺度 t-SNE 图">
    <img src="../../../assets/manuscript-figures/fig05_multiscale_tsne_diagnostic.png" alt="3 秒、5 秒和 8 秒窗口模型在六种活动状态上的倒数第二层表征 t-SNE 图" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">论文图 5。独立嵌入的 3 秒、5 秒与 8 秒表征在各活动上呈现互补的定性结构。</figcaption>
</figure>

短窗口保留局部转换细节，但更容易受噪声影响；长窗口提供更稳定的上下文，却会
模糊活动边界。t-SNE 面板是**内部定性诊断**，不是主要性能结果；它用于解释
跨尺度仲裁的动机，不能替代固定的片段级外部测试。

## 时间记录层带来了什么变化

<figure class="paper-figure portrait">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig06_outer_split_boundary_summary.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的 TRL 外层划分诊断图">
    <img src="../../../assets/manuscript-figures/fig06_outer_split_boundary_summary.png" alt="重复外层划分 F1 分布，以及时间记录层累积策略的边界质量与假阳性代价" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">论文图 6。固定 3 秒后验源上的划分分离诊断；时序一致性与置信裁剪相较基线记录构建策略带来最清晰的改善。</figcaption>
</figure>

在 50 次随机 10/10 开发用户划分中，选定时序策略的平均外层 F1 为 **0.913**，
基线后处理为 **0.802**。在累积边界诊断中，每记录小时假阳性由 **0.862**
降至 **0.190**，匹配 IoU 则由 **0.835** 变为 **0.843**。这些结果支持一个
较窄的结论：恢复的记录质量主要来自时序一致性控制与假阳性抑制，而不是改变
固定的局部预测器。

## 代表性时间线：成功与失败

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig07_representative_timeline_cases.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的代表性时间线图">
    <img src="../../../assets/manuscript-figures/fig07_representative_timeline_cases.png" alt="一个成功案例和一个部分失败案例中真实时间线及四种固定融合时序变体的对比" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">论文图 7。上图中 3 秒分支将乒乓球记录延长到满足 IoU 规则；下图保留真实局限：乒乓球仍未匹配，后续背景运动成为羽毛球假阳性。</figcaption>
</figure>

两个案例被有意并列展示：它们同时说明边界敏感多尺度融合的收益，以及弱活动证据
与运动样背景区间对记录假设造成压力时仍会出现的失败模式。

## 物理演示与现场测试

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig08_app_field_test.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的 App 现场测试图">
    <img src="../../../assets/manuscript-figures/fig08_app_field_test.png" alt="隐私保护动作模型与 Android 识别截图，覆盖背景和五种目标活动" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">论文图 8。面向 App 的现场测试示例，覆盖背景运动、羽毛球、跳绳、哑铃飞鸟、跑步与乒乓球。</figcaption>
</figure>

物理链路将 WT9011DCL-BT50 BLE 六轴 IMU 与 Android 识别应用连接。最终每尺度
选择三模型并结合 LBSA 与 TRL 的配置，在被测主动设备上平均每条用户记录耗时
**2.1 秒**。它慢于紧凑单模型诊断配置的 0.21 秒，但仍适合离线会话分析、同步前
训练记录处理或近实时记录复核。

??? info "公共语料可移植性检查——不是排行榜结果"

    论文还把 TRL 风格解码连接到 HAR70+、WISDM-phone、PAMAP2 与 OPPORTUNITY
    的神经窗口后验生成器。这些数据集并不匹配长时腕部运动协议，因此这里只检验
    时序接口能否重新参数化。

    | 数据集 | Argmax F1 | TRL F1 | Argmax FP/h | TRL FP/h |
    | --- | ---: | ---: | ---: | ---: |
    | HAR70+ | 0.70 | 0.70 | 47 | 45 |
    | WISDM-phone | 0.05 | 0.37 | 83 | 5.7 |
    | PAMAP2 | 0.09 | 0.53 | 90 | 9.3 |
    | OPPORTUNITY | 0.29 | 0.29 | 320 | 110 |

## 适用范围与局限

!!! warning "只在已评估的层级使用论文结论"

    - 现有证据衡量的是**片段记录质量**，不是临床收益、训练指导质量或安全决策。
    - 对新设备、新人群、新传感器位置、运动协议和标注方式的泛化尚未建立。
    - 用户级统计功效由 20 条开发／校准记录和 37 条独立外部测试记录决定，不能仅
      根据原始样本数量判断。
    - 当前系统更适合作为分钟级训练记录生成器。高度交错的会话以及相邻同类事件的
      分别计数仍然困难。

## 复现软件流程

- [运行公开冒烟测试](../getting-started/quickstart.md)
- [查看流水线架构](../guide/pipeline.md)
- [理解片段级评估器](../guide/evaluation.md)
- [构建 Android 演示](../deployment/android.md)
- [核对数据与模型资产边界](../reference/assets.md)
