---
hide:
  - toc
---

<section class="home-hero">
  <span class="hero-blob one" aria-hidden="true"></span>
  <span class="hero-blob two" aria-hidden="true"></span>
  <div class="hero-copy">
    <span class="hero-kicker">开源 · 从运动信号到可读信息</span>
    <h1>把连续腕部运动变成<span class="gradient-text">可以阅读、可以核查的活动记录。</span></h1>
    <p class="hero-lead">
      腕戴式 IMU 以 100 Hz 持续输出六路信号。本项目把这些连续数字转化为“发生了什么、
      何时开始、何时结束”的时间戳记录，并公开每条记录的生成、评估与部署过程。
    </p>
    <div class="hero-actions">
      <a class="hero-button primary" href="context/use-cases/">
        先看应用场景
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
      <a class="hero-button" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
        体验在线演示
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
      <a class="hero-button" href="research/paper/">查看研究证据</a>
    </div>
    <div class="hero-proof" aria-label="项目重点">
      <span>面向长时记录</span>
      <span>片段级证据</span>
      <span>网页 + Android</span>
    </div>
  </div>
  <div class="hero-visual">
    <div class="floating-badge badge-model">三尺度上下文</div>
    <div class="floating-badge badge-edge">可审计解码</div>
    <div class="signal-card">
      <div class="signal-toolbar">
        <span class="signal-live"><span class="signal-dot"></span>实时传感器流</span>
        <span>腕部 IMU · 100 HZ</span>
      </div>
      <div class="signal-window">
        <svg viewBox="0 0 520 224" role="img" aria-label="加速度计与陀螺仪轨迹解码为活动片段的示意图">
          <g stroke="#c7d2fe" stroke-width="1" opacity=".65">
            <path d="M0 42H520M0 84H520M0 126H520M0 168H520"/>
            <path d="M65 0V180M130 0V180M195 0V180M260 0V180M325 0V180M390 0V180M455 0V180"/>
          </g>
          <path d="M0 96C20 91 28 58 44 83S70 132 86 99s24-66 44-20 34 31 49-6 26-65 43 9 35 14 47-17 25-18 41 21 32 33 48-22 29-55 43 4 31 62 47 3 30-53 44-6 27 20 38-2 20-17 30 3" fill="none" stroke="#4f46e5" stroke-width="4" stroke-linecap="round"/>
          <path d="M0 130C24 112 33 154 53 129s33-74 49-15 31 45 47 5 29-38 44 17 33 19 47-14 27-57 45 7 34 19 49-18 30-27 45 16 28 36 45-8 27-30 42 2 28 26 44-4" fill="none" stroke="#7c3aed" stroke-width="3" stroke-linecap="round" opacity=".9"/>
          <g>
            <rect x="5" y="192" width="78" height="13" rx="6.5" fill="#10b981"/>
            <rect x="89" y="192" width="116" height="13" rx="6.5" fill="#f59e0b"/>
            <rect x="211" y="192" width="55" height="13" rx="6.5" fill="#6366f1"/>
            <rect x="272" y="192" width="98" height="13" rx="6.5" fill="#8b5cf6"/>
            <rect x="376" y="192" width="139" height="13" rx="6.5" fill="#14b8a6"/>
          </g>
        </svg>
      </div>
      <div class="signal-foot">
        <span>ACC + GYRO · 6 通道</span>
        <span>解码时间线</span>
      </div>
      <div class="signal-meta">
        <div><strong>3 / 5 / 8 秒</strong><span>时间尺度</span></div>
        <div><strong>CNN–BiLSTM</strong><span>窗口证据</span></div>
        <div><strong>TRL</strong><span>记录构建</span></div>
      </div>
    </div>
  </div>
</section>

<span class="section-eyebrow">先从现实问题讲起</span>

## 传感器记录的是运动，人真正需要的是一段活动故事 {: .section-title}

<p class="section-lead">六轴传感器连续记录 1 小时，会产生 36 万个时间点、216 万个通道读数，但其中没有一行会直接写着“09:02 开始打羽毛球，09:17 结束”。真正有用的结果不是另一张波形图，而是一组数量不多、可以复核的活动记录。</p>

<div class="story-grid">
  <article class="story-card story-card-input">
    <span class="story-card-kicker">设备看见的内容</span>
    <strong>连续运动信号，本身没有语义</strong>
    <div class="signal-token-list" aria-label="六路输入通道">
      <span>ACC_X</span><span>ACC_Y</span><span>ACC_Z</span>
      <span>GYRO_X</span><span>GYRO_Y</span><span>GYRO_Z</span>
    </div>
    <p>背景动作、活动切换、重复动作、停顿与传感器噪声都混在同一条连续数据流里。</p>
  </article>
  <article class="story-card story-card-output">
    <span class="story-card-kicker">使用者真正需要的内容</span>
    <strong>简洁、带时间戳的训练记录</strong>
    <div class="record-list" aria-label="示例活动记录">
      <div><time>09:02–09:17</time><span>羽毛球</span></div>
      <div><time>09:25–09:34</time><span>跳绳</span></div>
      <div><time>09:41–09:53</time><span>跑步</span></div>
    </div>
    <p>每条记录回答三个实际问题：做了什么、什么时候做、持续了多久。</p>
  </article>
</div>

<p class="story-caption">以上时间仅用于说明流程，并非参与者数据；它展示了本仓库试图完成的“从信号到记录”的转化。</p>

<div class="metric-strip context-metrics">
  <div class="metric"><strong>100 Hz</strong><span>连续采样</span></div>
  <div class="metric"><strong>6</strong><span>物理通道</span></div>
  <div class="metric"><strong>5</strong><span>前景活动</span></div>
  <div class="metric"><strong>4 个字段</strong><span>记录输出</span></div>
</div>

<span class="section-eyebrow">应用场景</span>

## 哪些场景会用到这套流水线？ {: .section-title}

<p class="section-lead">同一条“运动信号 → 活动记录”链路可以支持多种具体工作，但不同场景的证据强度并不相同。下面明确区分已经得到研究支持的用途，以及仍需针对性验证的原型用途。</p>

<div class="scenario-grid">
  <article class="scenario-card">
    <span class="scenario-tag established">直接适用于研究</span>
    <h3>评估长时活动识别系统</h3>
    <p>使用片段 F1、边界重叠、每小时假阳性、事件次数与持续时间比较方法，而不只看孤立窗口的分类准确率。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag scoped">限定范围原型</span>
    <h3>制作自动训练日志</h3>
    <p>把包含五类已支持活动的受控训练过程转换成候选记录，供参与者、教练或研究人员查看与修订。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag established">已有完整实现</span>
    <h3>验证研究模型到手机的部署</h3>
    <p>沿用同一六通道数据契约，从 WT9011DCL-BT50 传感器经 BLE 进入 Android ONNX 推理并显示时间线。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag exploratory">人在回路</span>
    <h3>辅助标注与质量复核</h3>
    <p>用候选片段帮助审核者快速定位可能的活动区间与边界错误；模型预测不能直接代替真实标注。</p>
  </article>
</div>

<div class="paper-home-cta story-cta">
  <p>进一步查看一段完整示例、适用人群、准确的输入输出契约，以及迁移到新设备或新人群前必须完成的验证。</p>
  <a class="md-button md-button--primary" href="context/use-cases/">阅读背景与应用场景</a>
</div>

<span class="section-eyebrow">真正困难的地方</span>

## 窗口标签正确，最后的活动记录仍可能是错的 {: .section-title}

<p class="section-lead">短窗口概率看起来可信，并不代表最终记录可靠：短暂的置信度下降可能把一次活动拆成多段，类似运动的背景会造成假阳性，窗口中心决策也会让边界偏移。这些错误会直接改变活动次数、持续时间与整条时间线。</p>

<figure class="paper-figure">
  <a class="pipeline-image-link" href="../assets/manuscript-figures/fig01_window_to_record_gap.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的窗口到记录落差图">
    <img src="../assets/manuscript-figures/fig01_window_to_record_gap.png" alt="后验概率轨迹、朴素提取得到的碎片记录以及时间记录层稳定后的活动记录" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">论文图 1。TRL 将局部看似合理但不稳定的窗口证据转换为更少、更稳定的活动片段记录。</figcaption>
</figure>

<div class="metric-strip paper-metrics">
  <div class="metric"><strong>137</strong><span>条长时记录</span></div>
  <div class="metric"><strong>259.6 h</strong><span>连续感知数据</span></div>
  <div class="metric"><strong>0.89</strong><span>平均用户 F1</span></div>
  <div class="metric"><strong>0.90</strong><span>Micro-F1</span></div>
</div>

<div class="paper-home-cta">
  <p>以上核心指标来自固定外部测试的片段记录评估，并非窗口准确率。内部诊断、外部结果、成功案例、失败案例与研究局限均在论文页面中分别标注。</p>
  <a class="md-button md-button--primary" href="research/paper/">查看论文证据</a>
</div>

<span class="section-eyebrow">系统如何解决问题</span>

## 从腕部传感器到可复核时间线的四个步骤 {: .section-title}

<p class="section-lead">先理解每个模块在故事中的任务，再看技术名词会更容易：保留完整会话、在多个时间尺度收集证据、稳定地构建记录，最后让结果可以审查与部署。</p>

<div class="feature-grid process-grid">
  <article class="feature-card process-card">
    <span class="process-step">01</span>
    <h3>采集完整会话</h3>
    <p>保留六路物理单位 ACC/GYRO 信号与时间戳，让最终活动边界始终能够回到原始记录核对。</p>
  </article>
  <article class="feature-card process-card">
    <span class="process-step">02</span>
    <h3>同时观察短时与长时上下文</h3>
    <p>对齐 3、5、8 秒 CNN–BiLSTM 证据，让局部切换细节和更稳定的动作上下文共同参与判断。</p>
  </article>
  <article class="feature-card process-card">
    <span class="process-step">03</span>
    <h3>构建稳定活动记录</h3>
    <p>LBSA 与确定性的时间记录层依次完成平滑、解码、合并、边界细化和过滤，并公开所有时序策略。</p>
  </article>
  <article class="feature-card process-card">
    <span class="process-step">04</span>
    <h3>复核或部署结果</h3>
    <p>导出 <code>user_id, category, start, end</code>，在浏览器中检查图表，或在 Android 上运行对应 ONNX 路径。</p>
  </article>
</div>

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="../assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的总体框架图">
    <img src="../assets/fig02_overall_framework.png" alt="仓库总体框架图，展示 IMU 数据流、分尺度 CNN–BiLSTM 模型、LBSA 融合、时序记录层与活动片段记录" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">仓库框架图：IMU 数据流 → 分尺度 CNN–BiLSTM → LBSA → 时间记录层 → 活动片段记录。点击图片可查看完整分辨率。</figcaption>
</figure>

<p class="pipeline-summary">Python 研究流程、公开 Gradio 演示与 Android 实现遵循同一项可观察契约：六路物理 IMU 信号进入系统，时间对齐的前景活动记录离开系统。配置文件、固定模型资产、实验脚本、中间概率与片段级评估让整个过程可以审计。</p>

<span class="section-eyebrow">先看证据，再谈扩展</span>

## 已经证明了什么，还有什么尚未证明？ {: .section-title}

<div class="evidence-grid">
  <article class="evidence-card">
    <span class="evidence-kicker">本仓库已有支持</span>
    <h3>完整、可检查的研究原型</h3>
    <ul>
      <li>已评估长时协议中的五类前景活动。</li>
      <li>在 37 条记录上完成固定的片段级外部测试。</li>
      <li>提供 Python 复现、公开合成示例与 Android 现场测试路径。</li>
      <li>同时报告成功与失败时间线。</li>
    </ul>
  </article>
  <article class="evidence-card caution">
    <span class="evidence-kicker">必须重新验证</span>
    <h3>超出当前评估范围的结论</h3>
    <ul>
      <li>新设备、新传感器位置、新人群或新活动协议。</li>
      <li>临床收益、训练指导质量、安全决策或生产级可靠性。</li>
      <li>高度交错的会话，以及相邻同类活动的分别计数。</li>
      <li>在重要场景中未经人工复核就直接采用自动标签。</li>
    </ul>
  </article>
</div>

<span class="section-eyebrow">按需要逐层深入</span>

## 从你现在最关心的问题开始 {: .section-title}

<div class="route-grid">
  <a class="route-card" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
    <span>2 分钟</span>
    <h3>体验一段合成会话</h3>
    <p>无需安装，直接运行仓库内模型并查看信号、概率、活动记录与 CSV 输出。</p>
  </a>
  <a class="route-card" href="context/use-cases/">
    <span>理解现实背景</span>
    <h3>阅读背景与应用场景</h3>
    <p>了解谁会使用这套流程、它支持什么判断，以及哪些环节需要人工复核或新增验证。</p>
  </a>
  <a class="route-card" href="guide/pipeline/">
    <span>技术路线</span>
    <h3>沿架构逐层理解</h3>
    <p>依次查看通道顺序、时间尺度、模型结构、融合、记录构建与输出。</p>
  </a>
  <a class="route-card" href="deployment/android/">
    <span>物理部署路线</span>
    <h3>构建 Android 演示</h3>
    <p>连接文档中的 BLE 传感器，查看实时信号、记录 CSV，并运行端侧 ONNX 推理。</p>
  </a>
</div>

<div class="cta-panel">
  <div>
    <h3>想走最短的“故事 → 证据”路线？</h3>
    <p>先运行公开合成示例，再把输出时间线与论文中的固定评估协议对应起来。</p>
  </div>
  <a class="md-button md-button--primary" href="deployment/hugging-face/">打开在线演示指南</a>
</div>
