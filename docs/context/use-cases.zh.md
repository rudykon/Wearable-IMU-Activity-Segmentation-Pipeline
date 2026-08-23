# 背景与应用场景

<p class="research-lead">这个项目起源于一个很实际的错位：可穿戴设备记录的是密集运动信号，而人们依靠的是稀疏事件来理解和决策。一套真正有用的系统，必须在两者之间建立可核查的联系，不能隐藏每条活动记录从哪里开始、在哪里结束，又可能在哪里出错。</p>

!!! info "预期使用范围"

    本仓库是一套面向**片段级活动识别**的研究与教学原型。它最适合用于可复现的
    长时会话评估与端到端部署实验。迁移到新设备、新人群、新运动协议、训练指导或
    临床场景时，必须重新完成针对性验证。

## 一段示例会话

设想一位参与者在一次混合训练中佩戴六轴 IMU。设备并不会收到“开始打羽毛球”
这样的指令；它只会持续接收加速度、角速度与时间戳，其中同时包含准备、运动、
停顿、切换和离场过程。

<div class="session-story" aria-label="混合训练示例时间线">
  <article class="story-moment">
    <time>08:55</time>
    <div><strong>连接传感器</strong><span>设备整理、走动与其他背景动作都进入同一条数据流。</span></div>
  </article>
  <article class="story-moment">
    <time>09:02</time>
    <div><strong>开始羽毛球</strong><span>重复的腕部运动逐渐可以识别，但短暂停顿可能让置信度中断。</span></div>
  </article>
  <article class="story-moment">
    <time>09:17</time>
    <div><strong>切换与休息</strong><span>系统需要准确结束上一条记录，又不能把附近的背景动作误报为新事件。</span></div>
  </article>
  <article class="story-moment">
    <time>09:25</time>
    <div><strong>开始跳绳</strong><span>短窗口与长窗口从不同角度观察同一组重复动作。</span></div>
  </article>
  <article class="story-moment">
    <time>09:34</time>
    <div><strong>复核训练记录</strong><span>最终需要的是带类别、起点、终点和持续时间，并且能够追溯证据的简短记录表。</span></div>
  </article>
</div>

<p class="story-caption">这条时间线完全是说明性示例，不包含任何参与者数据。它用于解释为什么连续感知是“时间分割”问题，而不是一组互不相关的短片段分类。</p>

系统的输出契约刻意保持简洁：

~~~text
user_id, category, start, end
~~~

但要可靠地产生这四个字段，必须经过完整链路：物理单位信号处理、多尺度窗口证据、
时序一致性控制、边界细化、假阳性抑制与片段级评估。

## 为什么普通窗口准确率还不够

窗口分类器回答的是“这几秒看起来像什么”；训练记录回答的却是另一个问题：
“一共发生了多少次活动、分别是什么、边界在哪里？”把前一个答案转换成后一个答案，
会引入窗口准确率无法衡量的错误：

- 一次真实活动可能被拆成多条短记录；
- 两个相邻事件可能被错误合并；
- 类似运动的背景区间可能变成假活动；
- 类别虽然正确，但开始或结束时间不可用；
- 相邻的同类活动可能难以分别计数。

<figure class="paper-figure">
  <a class="pipeline-image-link" href="../../../assets/manuscript-figures/fig01_window_to_record_gap.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的窗口到记录落差图">
    <img src="../../../assets/manuscript-figures/fig01_window_to_record_gap.png" alt="后验概率轨迹、朴素提取得到的碎片活动记录以及时间记录层稳定后的记录" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">论文图 1 直观展示了核心落差：局部证据看似可信，最终记录表仍然可能出错。</figcaption>
</figure>

## 四个具体应用场景

<div class="scenario-grid detailed">
  <article class="scenario-card">
    <span class="scenario-tag established">主要研究用途</span>
    <h3>长时人体活动识别评估</h3>
    <p><strong>情境。</strong>研究人员拥有连续可穿戴记录，希望比较完整的时间分割系统。</p>
    <p><strong>项目支持。</strong>仓库提供用户级划分、多尺度推理、同类别一对一 IoU 匹配、片段 F1、匹配 IoU 与假阳性分析。</p>
    <p><strong>支持的判断。</strong>一个方法是否能产生更好的活动记录，而不只是更好的窗口标签。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag scoped">受控原型</span>
    <h3>自动训练日志</h3>
    <p><strong>情境。</strong>一段受控训练包含羽毛球、跳绳、哑铃飞鸟、跑步或乒乓球。</p>
    <p><strong>项目支持。</strong>系统输出可作为候选活动日志，提供事件次数、发生时间与持续时间，供后续复核。</p>
    <p><strong>必须验证。</strong>新人群、新设备、新传感器位置和新训练协议均需重新测试，才能把日志视为可靠结果。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag established">已有系统实现</span>
    <h3>端侧与移动部署研究</h3>
    <p><strong>情境。</strong>工程人员希望确认研究流水线从离线文件迁移到物理传感器和手机后是否仍能完整运行。</p>
    <p><strong>项目支持。</strong>Android 模块覆盖 BLE 采集、信号视图、CSV 记录、离线识别与选定 ONNX 模型。</p>
    <p><strong>支持的判断。</strong>数据契约、模型资产与时序逻辑能否在端到端链路中保持一致。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag exploratory">候选工作流</span>
    <h3>人工辅助标注与质量检查</h3>
    <p><strong>情境。</strong>审核者需要检查数小时连续运动数据，并找到可能的前景活动区间。</p>
    <p><strong>项目支持。</strong>候选记录与概率图可以帮助集中检查边界、拆分、合并和误报问题。</p>
    <p><strong>人工职责。</strong>所有重要标签都必须由人确认或修正；这并不是论文主要评估的终点任务。</p>
  </article>
</div>

## 物理链路：从手腕到活动记录

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的物理部署链路图">
    <img src="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" alt="从可穿戴 IMU 经 BLE 和 Android 推理到活动记录的物理部署链路" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">仓库原图：WT9011DCL-BT50 可穿戴 IMU → BLE 采集 → Android 信号处理 → 端侧多尺度推理 → 活动识别。</figcaption>
</figure>

浏览器演示提供了第二条入口。它会在合成示例或兼容的上传会话上运行仓库中的
3 秒、5 秒与 8 秒模型，并展示六路信号、类别概率、解码时间线、最终记录以及
CSV 下载。在阅读实现细节之前，这是理解系统输出最快的方式。

## 当前证据能够支持什么

| 问题 | 当前答案 |
| --- | --- |
| 仓库是否实现了完整的“感知 → 记录”链路？ | 是。Python、公开 Gradio 与 Android 路径均有文档和测试。 |
| 是否按活动片段而不是独立窗口评估？ | 是。同类别一对一匹配，阈值为 IoU > 0.5。 |
| 是否有固定外部测试？ | 是。37 条长时记录，共 114 个已标注前景片段。 |
| 最终系统是否覆盖任意活动？ | 否。已评估的前景词汇只包含五类活动。 |
| 是否已经证明跨设备或跨人群泛化？ | 否。 |
| 是否属于临床、训练指导或安全产品？ | 否。 |

!!! warning "不要把结论悄悄扩大"

    现有结果衡量的是研究协议下的**活动记录质量**，并不能证明临床结局、训练指导
    正确性、损伤预防、安全监测或生产级可靠性。研究原型可以帮助设计这些后续研究，
    但不能替代这些验证。

## 按照你的角色继续阅读

<div class="route-grid compact">
  <a class="route-card" href="../../deployment/hugging-face/">
    <span>访客或审稿人</span>
    <h3>先看一次完整运行</h3>
    <p>从在线演示开始，再检查论文的固定证据与明确局限。</p>
  </a>
  <a class="route-card" href="../../guide/pipeline/">
    <span>人体活动识别研究人员</span>
    <h3>追踪并复现方法</h3>
    <p>依次阅读架构、数据契约、训练、推理与片段评估器。</p>
  </a>
  <a class="route-card" href="../../deployment/android/">
    <span>移动端或边缘工程人员</span>
    <h3>沿物理部署链路实现</h3>
    <p>查看传感器假设、ONNX 资产、Android 构建与 BLE 运行流程。</p>
  </a>
</div>

<div class="cta-panel">
  <div>
    <h3>现在背景已经清楚，可以继续看活动记录怎样生成。</h3>
    <p>架构页面将按照通道顺序、多尺度证据、LBSA、TRL 与最终输出逐层展开。</p>
  </div>
  <a class="md-button md-button--primary" href="../../guide/pipeline/">继续查看系统架构</a>
</div>
