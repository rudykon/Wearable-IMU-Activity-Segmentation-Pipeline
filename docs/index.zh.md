---
hide:
  - navigation
  - toc
---

<section class="home-hero showcase-hero">
  <div class="hero-copy">
    <p class="hero-kicker">可穿戴 IMU 活动记录</p>
    <h1 class="paper-title">基于<span class="title-accent">多尺度仲裁与时间记录层</span>的端到端可穿戴 IMU 片段级活动识别系统</h1>
    <p class="hero-lead">将连续 100 Hz 腕部 IMU 信号转换为包含活动类别与起止时间的片段记录。</p>
    <div class="hero-actions">
      <a class="hero-button primary" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
        体验在线演示
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
      <a class="hero-button" href="guide/pipeline/">方法</a>
      <a class="hero-button" href="research/paper/">结果</a>
      <a class="hero-button github-button" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.22c-3.23.7-3.91-1.37-3.91-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.72 1.27 3.39.97.1-.75.4-1.27.74-1.56-2.58-.29-5.29-1.29-5.29-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.48.11-3.05 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.19-1.49 3.16-1.18 3.16-1.18.63 1.57.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.4-2.72 5.38-5.31 5.67.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .7Z"/></svg>
        GitHub
      </a>
    </div>
    <div class="hero-proof" aria-label="关键证据">
      <span>259.6 小时传感器数据</span>
      <span>37 条独立外部测试记录</span>
      <span>0.90 Micro-F1</span>
    </div>
  </div>

  <div class="hero-visual" aria-label="IMU 信号到活动记录的示意流程">
    <div class="hero-system-card">
      <div class="system-card-header">
        <span>实时 IMU 数据流</span>
        <span>100 Hz · 6 通道</span>
      </div>
      <svg class="hero-wave" viewBox="0 0 520 210" role="img" aria-label="加速度计与陀螺仪信号被解码为活动片段的示意图">
        <g stroke="#dce6f1" stroke-width="1">
          <path d="M0 40H520M0 80H520M0 120H520M0 160H520"/>
          <path d="M65 0V170M130 0V170M195 0V170M260 0V170M325 0V170M390 0V170M455 0V170"/>
        </g>
        <path d="M0 92C18 84 28 50 45 78s27 51 44 17 25-63 44-18 33 32 49-5 27-62 45 8 34 16 48-16 27-20 44 17 31 35 48-20 30-50 44 5 30 55 47 2 28-49 45-5 28 18 38-2 22-14 30 2" fill="none" stroke="#3d6fb6" stroke-width="4" stroke-linecap="round"/>
        <path d="M0 128C24 112 34 151 55 126s31-68 49-13 31 42 47 5 29-35 45 15 32 18 47-13 27-53 45 7 34 18 49-17 30-25 45 15 29 33 46-8 27-28 43 2 28 23 44-4" fill="none" stroke="#756bb1" stroke-width="3" stroke-linecap="round" opacity=".9"/>
        <g>
          <rect x="8" y="184" width="96" height="12" rx="6" fill="#168c7e"/>
          <rect x="111" y="184" width="126" height="12" rx="6" fill="#d9822b"/>
          <rect x="244" y="184" width="72" height="12" rx="6" fill="#3d6fb6"/>
          <rect x="323" y="184" width="86" height="12" rx="6" fill="#756bb1"/>
          <rect x="416" y="184" width="96" height="12" rx="6" fill="#168c7e"/>
        </g>
      </svg>
      <div class="hero-model-row" aria-label="三个模型以及融合和记录解码阶段">
        <span>3 秒模型</span>
        <span>5 秒模型</span>
        <span>8 秒模型</span>
        <strong>LBSA</strong>
        <strong>TRL</strong>
      </div>
      <div class="hero-record-list" aria-label="示例输出记录">
        <div><time>09:02–09:17</time><strong>羽毛球</strong></div>
        <div><time>09:25–09:34</time><strong>跳绳</strong></div>
        <div><time>09:41–09:53</time><strong>跑步</strong></div>
      </div>
      <div class="system-card-footer">
        <span>连续信号</span>
        <span>稳定记录</span>
      </div>
    </div>
  </div>
</section>

<section class="home-section" markdown="1">

## 问题与输出

窗口级预测不能直接形成可靠的活动记录。面对长时传感器数据，系统还必须恢复活动类别、事件数量、持续时间与起止边界。

<div class="record-transform showcase-transform" aria-label="系统输入与输出">
  <code>六通道腕部 IMU 流</code>
  <span aria-hidden="true">→</span>
  <code>{活动, 开始, 结束}</code>
</div>

当前任务在 100 Hz 腕部加速度计和陀螺仪数据上评估五类运动。背景运动参与解码，但不会作为运动记录输出。

</section>

<section class="home-section" markdown="1">

## 方法

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的总体框架图">
    <img src="assets/fig02_overall_framework.png" alt="腕部 IMU 输入经过三个分尺度模型、LBSA 与 TRL 后生成活动记录的总体框架" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">IMU 流 → 3/5/8 秒后验 → LBSA → TRL → 活动记录。</figcaption>
</figure>

<div class="method-points">
  <article class="method-point">
    <span>01</span>
    <h3>多尺度模型</h3>
    <p>使用分尺度 CNN–BiLSTM 分析 3、5、8 秒时间视图。</p>
  </article>
  <article class="method-point">
    <span>02</span>
    <h3>边界感知融合</h3>
    <p>LBSA 在切换附近加强短窗口证据，在稳定区域保留长窗口上下文。</p>
  </article>
  <article class="method-point">
    <span>03</span>
    <h3>活动记录构建</h3>
    <p>TRL 对融合时间线进行平滑、解码、合并、边界修正与过滤。</p>
  </article>
</div>

[阅读完整方法](guide/pipeline.md){ .md-button }

</section>

<section class="home-section" markdown="1">

## 实验证据

<p class="section-intro">系统工作点在独立的 37 条外部测试记录评估之前完成冻结。</p>

<div class="metric-strip metric-strip--three showcase-metrics">
  <div class="metric"><strong>259.6 h</strong><span>传感器数据</span></div>
  <div class="metric"><strong>37 / 114</strong><span>测试记录 / 活动片段</span></div>
  <div class="metric"><strong>0.90</strong><span>Micro-F1</span></div>
</div>

<div class="evidence-summary">
  <div class="evidence-highlight">
    <strong>0.89</strong>
    <span>平均用户 F1 · LBSA + TRL</span>
  </div>
  <div class="evidence-scope">
    <h3>已评估范围</h3>
    <p>研究所用设备、腕部佩戴位置和长时会话协议下的五类活动。</p>
    <p><strong>新设备、佩戴位置、用户、活动类别和部署条件都需要重新验证。</strong></p>
  </div>
</div>

[查看完整结果与失败案例](research/paper.md){ .md-button }

</section>

<section class="home-section" markdown="1">

## 查看完整演示流程

<div class="demo-showcase">
  <a class="demo-showcase__media" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener" aria-label="打开 Hugging Face 在线演示">
    <img src="assets/demo-results-paper-notation.jpg" alt="真实 Hugging Face 演示输出，展示论文式腕部 IMU 通道记号和带时间戳的活动记录" loading="lazy" decoding="async">
    <span>打开在线演示 ↗</span>
  </a>
  <div class="demo-showcase__copy">
    <p class="hero-kicker">浏览器演示</p>
    <h3>在一个页面查看信号、概率、时间线和活动记录</h3>
    <p>使用内置合成记录或兼容的腕部 IMU 文件，运行公开的 3、5、8 秒模型。</p>
    <ul>
      <li>六通道信号曲线</li>
      <li>解码后的活动时间线</li>
      <li>活动记录表与 CSV 导出</li>
    </ul>
    <div class="demo-actions">
      <a class="demo-action primary" href="https://huggingface.co/spaces/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">运行在线演示</a>
      <a class="demo-action github" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/tree/main/demo" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.22c-3.23.7-3.91-1.37-3.91-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.72 1.27 3.39.97.1-.75.4-1.27.74-1.56-2.58-.29-5.29-1.29-5.29-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.48.11-3.05 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.19-1.49 3.16-1.18 3.16-1.18.63 1.57.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.4-2.72 5.38-5.31 5.67.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .7Z"/></svg>
        Demo 源码
      </a>
      <a class="demo-action" href="deployment/hugging-face/">输入格式与隐私说明</a>
    </div>
  </div>
</div>

</section>

<section class="home-section" markdown="1">

## 复现与查看源码

<div class="resource-grid">
  <a class="resource-card" href="https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
    <span>源码</span>
    <h3>GitHub</h3>
    <p>查看 Python 包、实验脚本、Android 应用、问题记录和版本历史。</p>
  </a>
  <a class="resource-card" href="https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline" target="_blank" rel="noopener">
    <span>权重</span>
    <h3>模型</h3>
    <p>从 Hugging Face 下载已校验的 PyTorch 和 ONNX 模型资产。</p>
  </a>
  <a class="resource-card" href="getting-started/quickstart/">
    <span>代码</span>
    <h3>快速开始</h3>
    <p>验证公开代码包，并运行需要授权数据的推理流程。</p>
  </a>
  <a class="resource-card" href="deployment/android/">
    <span>端侧</span>
    <h3>Android</h3>
    <p>构建 BLE 采集与端侧 ONNX 推理研究原型。</p>
  </a>
</div>

GitHub 不分发参与者记录。[复现指南](reproduce.md)明确区分公开验证与需要授权数据的完整流程。

</section>
