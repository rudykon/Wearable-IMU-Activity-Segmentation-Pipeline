# 主要结果

<p class="research-lead">主要分析在独立外部测试集上评价完整活动记录。</p>

<div class="metric-strip metric-strip--three">
  <div class="metric"><strong>37</strong><span>测试记录</span></div>
  <div class="metric"><strong>114</strong><span>真实片段</span></div>
  <div class="metric"><strong>0.90</strong><span>Micro-F1</span></div>
</div>

## 评估协议

预测片段与同类别真实片段按 IoU > 0.5 一对一匹配。这种记录级评估会同时惩罚漏检、误报、拆分、合并、类别错误与边界偏移。

| 数据角色 | 记录数 | 用途 |
| --- | ---: | --- |
| 训练集 | 80 | 模型拟合与训练阶段检查点选择 |
| 开发／校准集 | 20 | 时序策略校准与诊断 |
| 独立外部测试集 | 37 | 所有选择冻结后进行一次最终评估 |

外部测试标签不参与检查点、融合规则、TRL 参数或报告方案的选择。平均用户 F1 对每位用户等权；Micro-F1 汇总全部 TP、FP 与 FN。

## 外部测试

<figure class="paper-figure portrait">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig04_external_variant_comparison.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的外部测试对比图">
    <img src="../../../assets/manuscript-figures/fig04_external_variant_comparison.png" alt="五种固定方案的外部测试平均用户 F1、Micro-F1、误报与漏检数量" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">五种固定方案评估同一组 37 条记录。</figcaption>
</figure>

| 固定方案 | 平均用户 F1 | 95% CI | Micro-F1 | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: |
| 5 秒 + 8 秒 + TRL | 0.88 | 0.80–0.94 | 0.88 | 98 / 11 / 16 |
| 三模型平均 + TRL | 0.88 | 0.80–0.94 | 0.89 | 98 / 9 / 16 |
| 三模型加权 + TRL | 0.89 | 0.81–0.94 | 0.89 | 99 / 9 / 15 |
| LBSA + 宽松 Top-K | 0.88 | 0.80–0.95 | 0.88 | 103 / 17 / 11 |
| **LBSA + TRL** | **0.89** | **0.82–0.94** | **0.90** | **99 / 7 / 15** |

<div class="result-callout"><strong>主要结果。</strong>LBSA + TRL 保持最高的四舍五入 F1，同时在对比方案中产生最少的假阳性记录。</div>

## 分类结果

| 活动 | 真实片段 | TP / FP / FN | 精确率 | 召回率 | F1 | 匹配 IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 羽毛球 | 32 | 26 / 3 / 6 | 0.90 | 0.81 | 0.85 | 0.87 |
| 跳绳 | 20 | 19 / 0 / 1 | 1.00 | 0.95 | 0.97 | 0.84 |
| 哑铃飞鸟 | 20 | 19 / 1 / 1 | 0.95 | 0.95 | 0.95 | 0.78 |
| 跑步 | 20 | 18 / 1 / 2 | 0.95 | 0.90 | 0.92 | 0.82 |
| 乒乓球 | 22 | 17 / 2 / 5 | 0.90 | 0.77 | 0.83 | 0.86 |

跳绳的 F1 最高。乒乓球召回率最低，说明较弱或含混的证据仍会造成漏检。

## 代表性案例

<figure class="paper-figure compact">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig07_representative_timeline_cases.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的代表性时间线图">
    <img src="../../../assets/manuscript-figures/fig07_representative_timeline_cases.png" alt="一个成功案例和一个部分失败案例中真实时间线与固定时序融合方案的对比" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">一个成功案例与一个部分失败案例。</figcaption>
</figure>

尺度仲裁改善了成功案例中的边界。失败案例在后验证据较弱的位置仍存在一次漏检和一次误报；分类器没有提供的活动证据，TRL 无法凭空恢复。

## 限制

!!! warning "评估范围"

    - 结果衡量片段记录质量，不代表临床、训练或安全价值。
    - 证据覆盖研究所用设备、佩戴位置和协议下的五类活动。
    - 新设备、佩戴位置、用户、活动类别和部署条件需要重新验证。
    - 密集会话和相邻同类事件仍然困难。

开发集诊断、公共数据集可移植性检查与 Android 工程证据见[补充分析](supplementary.md)。复现命令和资产边界统一放在[复现页](../reproduce.md)。
