# 应用场景

<p class="research-lead">可穿戴设备记录运动，本项目把它整理成活动日志。</p>

!!! info "范围"

    本项目是**片段级活动识别**研究原型。新设备、新人群、新活动和重要用途都需
    重新验证。

## 示例

六轴 IMU 不接收活动指令，只记录加速度、角速度和时间。

<div class="session-story" aria-label="混合训练示例时间线">
  <article class="story-moment">
    <time>08:55</time>
    <div><strong>连接</strong><span>整理和走动进入数据流。</span></div>
  </article>
  <article class="story-moment">
    <time>09:02</time>
    <div><strong>羽毛球</strong><span>重复腕部运动逐渐可识别。</span></div>
  </article>
  <article class="story-moment">
    <time>09:17</time>
    <div><strong>休息</strong><span>上一条记录需要准确结束。</span></div>
  </article>
  <article class="story-moment">
    <time>09:25</time>
    <div><strong>跳绳</strong><span>短、长窗口看到不同信息。</span></div>
  </article>
  <article class="story-moment">
    <time>09:34</time>
    <div><strong>复核</strong><span>输出一份简短活动列表。</span></div>
  </article>
</div>

<p class="story-caption">时间仅作示意，不含参与者数据。</p>

输出：

~~~text
user_id, category, start, end
~~~

## 窗口与记录

窗口分类正确，活动日志仍可能出错：

- 一次真实活动可能被拆成多条短记录；
- 两个相邻事件可能被错误合并；
- 类似运动的背景区间可能变成假活动；
- 类别虽然正确，但开始或结束时间不可用；
- 相邻的同类活动可能难以分别计数。

<figure class="paper-figure">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig01_window_to_record_gap.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的窗口到记录落差图">
    <img src="../../../assets/manuscript-figures/fig01_window_to_record_gap.png" alt="后验概率轨迹、朴素提取得到的碎片活动记录以及时间记录层稳定后的记录" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">论文图 1。局部预测正确，最终记录仍可能出错。</figcaption>
</figure>

## 用途

<div class="scenario-grid detailed">
  <article class="scenario-card">
    <span class="scenario-tag established">研究</span>
    <h3>长时评测</h3>
    <p>用固定划分、片段 F1、重叠和误报比较完整记录。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag scoped">原型</span>
    <h3>训练日志</h3>
    <p>为五类已支持活动生成候选记录；新场景需重测。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag established">移动端</span>
    <h3>端侧部署</h3>
    <p>测试 Android 的 BLE 采集、ONNX 推理和时间线。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag exploratory">复核</span>
    <h3>辅助标注</h3>
    <p>定位候选活动和边界错误，再由人工确认。</p>
  </article>
</div>

## 传感器到手机

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的物理部署链路图">
    <img src="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" alt="从可穿戴 IMU 经 BLE 和 Android 推理到活动记录的物理部署链路" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">WT9011DCL-BT50 → BLE → Android → 活动记录。</figcaption>
</figure>

[浏览器演示](../deployment/hugging-face.md)会运行 3、5、8 秒模型，并返回图表、记录和 CSV。

## 证据

| 问题 | 当前答案 |
| --- | --- |
| 完整链路？ | 是：Python、Gradio、Android。 |
| 片段级评估？ | 是：同类别 IoU > 0.5。 |
| 固定外部测试？ | 是：37 条记录、114 个片段。 |
| 任意活动？ | 否：仅评估五类。 |
| 新设备或新人群？ | 尚未证明。 |
| 临床、训练或安全用途？ | 否。 |

!!! warning "边界"

    现有结果只衡量研究协议下的**活动记录质量**，不代表临床、训练、安全或生产表现。

## 下一步

<div class="route-grid compact">
  <a class="route-card" href="../../deployment/hugging-face/">
    <span>访客</span>
    <h3>演示</h3>
    <p>运行一次完整样例。</p>
  </a>
  <a class="route-card" href="../../guide/pipeline/">
    <span>研究人员</span>
    <h3>流水线</h3>
    <p>追踪并复现方法。</p>
  </a>
  <a class="route-card" href="../../deployment/android/">
    <span>工程人员</span>
    <h3>Android</h3>
    <p>构建 BLE 与 ONNX 链路。</p>
  </a>
</div>
