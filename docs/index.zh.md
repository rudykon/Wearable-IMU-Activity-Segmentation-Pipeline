---
hide:
  - toc
---

<section class="home-hero">
  <span class="hero-blob one" aria-hidden="true"></span>
  <span class="hero-blob two" aria-hidden="true"></span>
  <div class="hero-copy">
    <span class="hero-kicker">开源 · 从研究到端侧</span>
    <h1>将长时 IMU 流转化为<span class="gradient-text">可审计的活动时间线。</span></h1>
    <p class="hero-lead">
      一套可复现的 Python 与 Android 多尺度分割流水线，面向腕戴式加速度计和陀螺仪数据，
      覆盖模型训练、时序解码、片段评估与端侧推理。
    </p>
    <div class="hero-actions">
      <a class="hero-button primary" href="getting-started/quickstart/">
        运行流水线
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
      <a class="hero-button" href="guide/pipeline/">探索系统架构</a>
    </div>
    <div class="hero-proof" aria-label="项目能力">
      <span>可复现 Python 流程</span>
      <span>多尺度模型</span>
      <span>Android ONNX</span>
    </div>
  </div>
  <div class="hero-visual">
    <div class="floating-badge badge-model">三尺度集成</div>
    <div class="floating-badge badge-edge">端侧就绪 ONNX</div>
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
        <div><strong>CNN–BiLSTM</strong><span>窗口模型</span></div>
        <div><strong>Viterbi</strong><span>序列解码</span></div>
      </div>
    </div>
  </div>
</section>

<div class="metric-strip">
  <div class="metric"><strong>100 Hz</strong><span>采样率</span></div>
  <div class="metric"><strong>6</strong><span>物理 IMU 通道</span></div>
  <div class="metric"><strong>3</strong><span>时间尺度</span></div>
  <div class="metric"><strong>5 + 背景</strong><span>输出类别</span></div>
</div>

<span class="section-eyebrow">论文洞见</span>

## 为什么窗口准确率还不够 {: .section-title}

<p class="section-lead">长时识别器即使产生局部可信的概率，也可能报告错误的活动次数、持续时间或边界。论文将这一“窗口到记录的落差”作为核心测量问题。</p>

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
  <p>查看固定外部测试对比、多尺度诊断、代表性时间线、现场测试与论文明确说明的局限。</p>
  <a class="md-button md-button--primary" href="research/paper/">查看论文证据</a>
</div>

<span class="section-eyebrow">从研究到部署</span>

## 一个仓库，贯通研究到设备的完整路径 {: .section-title}

<p class="section-lead">从物理单位传感器输入到可部署的活动片段，一套连贯的工作流让每个阶段均可检查，避免把关键决策隐藏在黑盒之中。</p>

<div class="feature-grid">
  <article class="feature-card">
    <span class="feature-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24"><path d="M3 12h3l2-6 4 13 3-10 2 6h4"/></svg>
    </span>
    <h3>长时记录分割</h3>
    <p>将每位用户的连续传感器记录转换为明确的 <code>user_id, category, start, end</code> 活动片段。</p>
  </article>
  <article class="feature-card">
    <span class="feature-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24"><rect x="3" y="5" width="9" height="5" rx="1"/><rect x="6" y="10" width="12" height="5" rx="1"/><rect x="9" y="15" width="12" height="5" rx="1"/></svg>
    </span>
    <h3>多尺度建模</h3>
    <p>对齐 3、5、8 秒 CNN–BiLSTM 预测，让短时运动特征与更长上下文共同决定一条时间线。</p>
  </article>
  <article class="feature-card">
    <span class="feature-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24"><path d="M3 16l5-5 4 3 5-7 4 3"/><path d="M17 7h4v4"/></svg>
    </span>
    <h3>时序解码</h3>
    <p>依次执行 LBSA 融合、概率平滑、Viterbi 解码、边界细化、重叠处理与片段过滤。</p>
  </article>
  <article class="feature-card">
    <span class="feature-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24"><rect x="6" y="2" width="12" height="20" rx="2"/><path d="M10 5h4M10 18h4"/><circle cx="12" cy="12" r="3"/></svg>
    </span>
    <h3>Android 部署</h3>
    <p>采集 WT9011DCL-BT50 BLE 数据、可视化信号、记录 CSV，并在设备上运行选定的 ONNX 集成模型。</p>
  </article>
</div>

<span class="section-eyebrow">可审计设计</span>

## 流水线概览 {: .section-title}

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="../assets/fig02_overall_framework.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的总体框架图">
    <img src="../assets/fig02_overall_framework.png" alt="仓库现有总体框架图，展示 IMU 数据流、分尺度 CNN–BiLSTM 模型、LBSA 融合、时序记录层与活动片段记录" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">仓库现有框架图：IMU 数据流 → 分尺度 CNN–BiLSTM → LBSA → 时序记录层 → 活动片段记录。点击图片可查看完整分辨率。</figcaption>
</figure>

<p class="pipeline-summary">Python 研究流程与 Android 演示遵循同一项可观察契约：六个物理单位 IMU 通道进入系统，时间对齐的活动片段离开系统。仓库通过配置文件、固定模型资产、实验脚本与片段级评估，让每项中间决策都可检查。</p>

| 层级 | 仓库实现 | 主要产物 |
| --- | --- | --- |
| 输入 | UTF-8 制表符分隔的 ACC/GYRO 记录 | 每用户信号文件 |
| 表征 | 多卷积核一维 CNN + BiLSTM | 窗口概率 |
| 融合 | 对齐 3 秒 / 5 秒 / 8 秒概率 | 多尺度序列 |
| 时序逻辑 | LBSA、平滑、Viterbi、边界细化 | 活动时间线 |
| 输出 | 片段写入器与评估器 | XLSX 记录与 F1 指标 |
| 部署 | Android BLE + ONNX Runtime | 实时或离线识别 |

### 物理部署链路

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" target="_blank" rel="noopener" aria-label="打开完整分辨率的物理部署链路图">
    <img src="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" alt="仓库现有物理部署链路图，展示从可穿戴 IMU 传感器经 BLE 与 Android 端侧推理到活动识别的完整路径" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">仓库现有部署图：可穿戴 IMU → BLE 采集 → Android 信号处理 → 端侧多尺度推理 → 活动识别。点击图片可查看完整分辨率。</figcaption>
</figure>

<span class="section-eyebrow">快速验证</span>

## 从公开冒烟测试开始 {: .section-title}

轻量级冒烟测试会验证导入、规范路径、临时信号加载、标注加载与工作簿写入，
**不需要**参与者记录或 GPU。

=== "Conda"

    ~~~bash
    git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
    cd Wearable-IMU-Activity-Segmentation-Pipeline
    conda env create -f environment.yml
    conda activate imu-activity-pipeline
    python -m pip install -e .
    python tests/smoke_test.py
    ~~~

=== "pip"

    ~~~bash
    git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
    cd Wearable-IMU-Activity-Segmentation-Pipeline
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m pip install -e .
    python tests/smoke_test.py
    ~~~

!!! important "数据访问边界"

    GitHub 仓库不分发参与者传感器流。经授权的数据应保留在已被忽略的本地 `data/`
    目录结构中。选定的 Python 检查点、归一化资产和 Android ONNX 文件与这些记录
    分开进行版本管理。

<span class="section-eyebrow">输出类别</span>

## 支持的前景活动 {: .section-title}

| 中文标签 | 英文标签 | 输出行为 |
| --- | --- | --- |
| 羽毛球 | Badminton | 前景片段 |
| 跳绳 | Jump rope | 前景片段 |
| 飞鸟 | Fly | 前景片段 |
| 跑步 | Running | 前景片段 |
| 乒乓球 | Table tennis | 前景片段 |

系统在需要时于内部建模背景／无活动状态，而提交的片段记录仅包含前景活动。

<div class="cta-panel">
  <div>
    <h3>准备好运行经授权的数据集了吗？</h3>
    <p>先准备规范目录结构，再按照推理指南运行。</p>
  </div>
  <a class="md-button md-button--primary" href="getting-started/quickstart/">打开快速开始</a>
</div>
