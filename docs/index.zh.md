---
hide:
  - toc
---

<section class="home-hero">
  <span class="hero-blob one" aria-hidden="true"></span>
  <span class="hero-blob two" aria-hidden="true"></span>
  <div class="hero-copy">
    <span class="hero-kicker">可穿戴 IMU</span>
    <h1>从运动信号到<span class="gradient-text">活动记录。</span></h1>
    <p class="hero-lead">把 100 Hz 腕部 IMU 数据分割成带时间戳的活动。</p>
    <div class="hero-actions">
      <a class="hero-button primary" href="context/use-cases/">
        场景
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
      <a class="hero-button" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
        演示
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
      <a class="hero-button" href="research/paper/">结果</a>
    </div>
    <div class="hero-proof" aria-label="项目重点">
      <span>长时会话</span>
      <span>5 类活动</span>
      <span>网页 + Android</span>
    </div>
  </div>
  <div class="hero-visual">
    <div class="floating-badge badge-model">3 种尺度</div>
    <div class="floating-badge badge-edge">稳定记录</div>
    <div class="signal-card">
      <div class="signal-toolbar">
        <span class="signal-live"><span class="signal-dot"></span>实时 IMU</span>
        <span>100 HZ</span>
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
        <span>6 通道</span>
        <span>时间线</span>
      </div>
      <div class="signal-meta">
        <div><strong>3 / 5 / 8 秒</strong><span>窗口</span></div>
        <div><strong>3 个模型</strong><span>融合</span></div>
        <div><strong>1 条时间线</strong><span>输出</span></div>
      </div>
    </div>
  </div>
</section>

<span class="section-eyebrow">问题</span>

## 信号不是记录 {: .section-title}

<p class="section-lead">一小时有 216 万个读数，真正需要的是一份简短活动日志。</p>

<div class="story-grid">
  <article class="story-card story-card-input">
    <span class="story-card-kicker">输入</span>
    <strong>六路运动信号</strong>
    <div class="signal-token-list" aria-label="六路输入通道">
      <span>ACC_X</span><span>ACC_Y</span><span>ACC_Z</span>
      <span>GYRO_X</span><span>GYRO_Y</span><span>GYRO_Z</span>
    </div>
    <p>动作、切换、停顿和噪声混在一起。</p>
  </article>
  <article class="story-card story-card-output">
    <span class="story-card-kicker">输出</span>
    <strong>带时间戳的活动</strong>
    <div class="record-list" aria-label="示例活动记录">
      <div><time>09:02–09:17</time><span>羽毛球</span></div>
      <div><time>09:25–09:34</time><span>跳绳</span></div>
      <div><time>09:41–09:53</time><span>跑步</span></div>
    </div>
    <p>每条记录包含活动、开始、结束和时长。</p>
  </article>
</div>

<p class="story-caption">以上时间仅作示意，不含参与者数据。</p>

<div class="metric-strip context-metrics">
  <div class="metric"><strong>100 Hz</strong><span>采样</span></div>
  <div class="metric"><strong>6</strong><span>通道</span></div>
  <div class="metric"><strong>5</strong><span>活动</span></div>
  <div class="metric"><strong>4</strong><span>输出字段</span></div>
</div>

<span class="section-eyebrow">应用场景</span>

## 用在哪里 {: .section-title}

<div class="scenario-grid">
  <article class="scenario-card">
    <span class="scenario-tag established">研究</span>
    <h3>长时评测</h3>
    <p>比较记录、边界、次数和误报。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag scoped">原型</span>
    <h3>训练日志</h3>
    <p>为五类已支持活动生成待复核记录。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag established">移动端</span>
    <h3>手机部署</h3>
    <p>通过 BLE 运行传感器到 Android ONNX 链路。</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag exploratory">复核</span>
    <h3>辅助标注</h3>
    <p>定位候选活动和边界错误，再由人工确认。</p>
  </article>
</div>

<div class="paper-home-cta story-cta">
  <p>新设备、佩戴位置和人群都需重新验证。</p>
  <a class="md-button md-button--primary" href="context/use-cases/">全部场景</a>
</div>

<span class="section-eyebrow">难点</span>

## 窗口正确，记录仍会出错 {: .section-title}

<p class="section-lead">置信度波动会拆分活动、移动边界或产生误报。</p>

<figure class="paper-figure">
  <a class="pipeline-image-link" href="../assets/manuscript-figures/fig01_window_to_record_gap.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的窗口到记录落差图">
    <img src="../assets/manuscript-figures/fig01_window_to_record_gap.png" alt="后验概率轨迹、朴素提取得到的碎片记录以及时间记录层稳定后的活动记录" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">论文图 1。时间线解码合并不稳定的窗口预测。</figcaption>
</figure>

<div class="metric-strip paper-metrics">
  <div class="metric"><strong>137</strong><span>条记录</span></div>
  <div class="metric"><strong>259.6 h</strong><span>传感器数据</span></div>
  <div class="metric"><strong>0.89</strong><span>平均用户 F1</span></div>
  <div class="metric"><strong>0.90</strong><span>Micro-F1</span></div>
</div>

<div class="paper-home-cta">
  <p>这些 F1 衡量固定外部测试中的完整记录。</p>
  <a class="md-button md-button--primary" href="research/paper/">完整结果</a>
</div>

<span class="section-eyebrow">方法</span>

## 四步完成 {: .section-title}

<div class="feature-grid process-grid">
  <article class="feature-card process-card">
    <span class="process-step">01</span>
    <h3>采集</h3>
    <p>保留六路 IMU 信号和时间戳。</p>
  </article>
  <article class="feature-card process-card">
    <span class="process-step">02</span>
    <h3>分类</h3>
    <p>运行 3、5、8 秒模型。</p>
  </article>
  <article class="feature-card process-card">
    <span class="process-step">03</span>
    <h3>解码</h3>
    <p>LBSA 选尺度，TRL 生成稳定记录。</p>
  </article>
  <article class="feature-card process-card">
    <span class="process-step">04</span>
    <h3>使用</h3>
    <p>导出记录、查看图表或部署到 Android。</p>
  </article>
</div>

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="../assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的总体框架图">
    <img src="../assets/fig02_overall_framework.png" alt="仓库总体框架图，展示 IMU 数据流、分尺度 CNN–BiLSTM 模型、LBSA 融合、时序记录层与活动片段记录" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">IMU → 三个模型 → LBSA → TRL → 活动记录。</figcaption>
</figure>

<span class="section-eyebrow">证据</span>

## 结果与边界 {: .section-title}

<div class="evidence-grid">
  <article class="evidence-card">
    <span class="evidence-kicker">已有支持</span>
    <h3>本项目已测试</h3>
    <ul>
      <li>长时会话中的五类活动。</li>
      <li>固定外部测试：37 条记录。</li>
      <li>Python、网页和 Android 路径。</li>
    </ul>
  </article>
  <article class="evidence-card caution">
    <span class="evidence-kicker">尚未证明</span>
    <h3>需要重测</h3>
    <ul>
      <li>新设备、佩戴位置、人群或活动。</li>
      <li>临床、训练、安全或生产用途。</li>
      <li>密集活动和相邻同类事件。</li>
    </ul>
  </article>
</div>

<span class="section-eyebrow">下一步</span>

## 选择入口 {: .section-title}

<div class="route-grid">
  <a class="route-card" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
    <span>2 分钟</span>
    <h3>演示</h3>
    <p>在浏览器运行合成会话。</p>
  </a>
  <a class="route-card" href="context/use-cases/">
    <span>背景</span>
    <h3>场景</h3>
    <p>查看用途、前提和边界。</p>
  </a>
  <a class="route-card" href="guide/pipeline/">
    <span>技术</span>
    <h3>流水线</h3>
    <p>查看输入、模型、解码和输出。</p>
  </a>
  <a class="route-card" href="deployment/android/">
    <span>移动端</span>
    <h3>Android</h3>
    <p>构建 BLE 与 ONNX 应用。</p>
  </a>
</div>
