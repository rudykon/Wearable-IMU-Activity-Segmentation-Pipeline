"""Gradio entry point for the Hugging Face Space demo."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import spaces
import gradio as gr

from demo.runtime import (
    FUSION_MODE_LABELS,
    FUSION_MODES,
    SEGMENT_COLUMNS,
    DemoInputError,
    export_segments,
    make_signal_figure,
    make_timeline_figure,
    run_pipeline,
    segments_dataframe,
    status_markdown,
)

LOGGER = logging.getLogger(__name__)
EXAMPLE_PATH = ROOT / "demo" / "examples" / "synthetic_activity_imu.tsv"
FUSION_CHOICES = [(FUSION_MODE_LABELS[mode], mode) for mode in FUSION_MODES]
PROJECT_WEBSITE_URL = (
    "https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/"
)
GITHUB_URL = "https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline"
SYNTHETIC_READY_STATUS = (
    "### Synthetic sample ready / 合成样例已就绪\n"
    "`synthetic_activity_imu.tsv` is selected. Click **Run current recording / 运行当前记录** "
    "to generate the three outputs. / `synthetic_activity_imu.tsv` 已选中，点击 "
    "**运行当前记录** 即可生成三类结果。"
)
CUSTOM_RECORDING_STATUS = (
    "### Custom recording selected / 已选择自定义记录\n"
    "Review the settings, then click **Run current recording / 运行当前记录**. / "
    "请确认参数设置，然后点击 **运行当前记录**。"
)
NO_RECORDING_STATUS = (
    "### Select a recording / 请选择记录\n"
    "Upload a compatible TXT/TSV file, or restore the synthetic sample. / "
    "请上传兼容的 TXT/TSV 文件，或恢复合成样例。"
)

CSS = """
:root {
  --imu-indigo: #4f46e5;
  --imu-violet: #7c3aed;
  --imu-ink: #172033;
  --imu-muted: #5b6573;
  --imu-border: #d9e1e8;
}
.gradio-container { max-width: 1240px !important; }
.imu-hero {
  padding: 1.35rem 1.5rem;
  border: 1px solid rgba(79, 70, 229, .18);
  border-radius: 22px;
  background: linear-gradient(135deg, rgba(238,242,255,.98), rgba(250,245,255,.98));
  box-shadow: 0 18px 48px rgba(79, 70, 229, .10);
  margin-bottom: .75rem;
}
.imu-hero h1 {
  margin: 0 0 .45rem;
  color: var(--imu-ink);
  font-size: clamp(1.7rem, 4vw, 2.6rem);
}
.imu-hero p { margin: .2rem 0; color: #475569; max-width: 900px; }
.imu-links {
  display: flex;
  flex-wrap: wrap;
  gap: .6rem;
  margin-top: 1rem;
}
.imu-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 2.55rem;
  padding: .62rem .92rem;
  border: 1px solid var(--imu-border);
  border-radius: 10px;
  color: var(--imu-ink) !important;
  background: #fff;
  font-size: .9rem;
  font-weight: 700;
  text-decoration: none !important;
}
.imu-link.primary {
  border-color: transparent;
  color: #fff !important;
  background: linear-gradient(135deg, var(--imu-indigo), var(--imu-violet));
}
.imu-link.github {
  border-color: #24292f;
  color: #fff !important;
  background: #24292f;
}
.imu-link:hover { transform: translateY(-1px); }
.imu-link:focus-visible { outline: 3px solid rgba(79, 70, 229, .28); outline-offset: 2px; }
.imu-overview {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: .75rem;
  margin: 0 0 1rem;
}
.imu-card {
  padding: 1rem 1.05rem;
  border: 1px solid var(--imu-border);
  border-radius: 16px;
  background: #fff;
}
.imu-card-kicker {
  color: #4f46e5;
  font-size: .75rem;
  font-weight: 800;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.imu-card h2 {
  margin: .35rem 0 .45rem;
  color: var(--imu-ink);
  font-size: 1rem;
  line-height: 1.35;
}
.imu-card p,
.imu-card li {
  color: var(--imu-muted);
  font-size: .87rem;
  line-height: 1.55;
}
.imu-card p { margin: .25rem 0; }
.imu-card .zh { color: #475569; }
.imu-flow {
  margin: .55rem 0;
  padding: .52rem .62rem;
  border-radius: 9px;
  color: var(--imu-ink);
  background: #f3f5f9;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .78rem;
  font-weight: 700;
  line-height: 1.5;
}
.imu-steps {
  margin: .45rem 0 0;
  padding-left: 1.15rem;
}
.imu-steps li + li { margin-top: .28rem; }
.imu-pill {
  display: inline-block;
  margin-bottom: .65rem;
  padding: .25rem .65rem;
  color: #4338ca;
  border-radius: 999px;
  background: rgba(99, 102, 241, .10);
  font-size: .78rem;
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.primary-action { background: linear-gradient(135deg, var(--imu-indigo), var(--imu-violet)) !important; }
.privacy-note { font-size: .9rem; color: #64748b; }
.sensor-schema {
  margin: .1rem 0 .55rem;
  color: #475569;
  font-size: .94rem;
  line-height: 1.55;
}
.sensor-schema p { margin: .18rem 0; }
.sensor-symbol {
  display: inline-block;
  min-width: 1.35em;
  color: var(--imu-ink);
  font-family: "Cambria Math", "STIX Two Math", "Times New Roman", serif;
  font-size: 1.08em;
  white-space: nowrap;
}
.sensor-symbol var { font-family: inherit; }
.sensor-symbol sub { font-size: .68em; }
@media (max-width: 850px) {
  .imu-overview { grid-template-columns: 1fr; }
  .imu-links { align-items: stretch; flex-direction: column; }
  .imu-link { width: 100%; }
}
"""
THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet")


@spaces.GPU(duration=30)
def _run_zero_gpu_pipeline(upload, fusion_mode, min_duration_sec, confidence_min, top_k):
    """Run one complete model pass inside a single ZeroGPU allocation."""

    return run_pipeline(
        upload,
        fusion_mode=str(fusion_mode),
        min_duration_sec=float(min_duration_sec),
        confidence_min=float(confidence_min),
        top_k=int(top_k),
    )


def segment_recording(upload, fusion_mode, min_duration_sec, confidence_min, top_k):
    """Gradio callback that turns one upload into plots, segments, and CSV."""

    try:
        result = _run_zero_gpu_pipeline(
            upload,
            fusion_mode,
            min_duration_sec,
            confidence_min,
            top_k,
        )
        return (
            status_markdown(result, str(fusion_mode)),
            make_signal_figure(result.recording),
            make_timeline_figure(result),
            segments_dataframe(result),
            export_segments(result),
        )
    except DemoInputError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # Avoid exposing server internals in the public UI.
        LOGGER.exception("Space inference failed")
        raise gr.Error(
            "No activity records could be generated. Check the file columns, "
            "timestamps, and sampling rate, then try again. / 无法生成活动记录，请检查文件列名、时间戳和采样率后重试。"
        ) from exc


def recording_selection_status(upload) -> str:
    """Describe the selected file without running a model or exposing its name."""

    if upload is None:
        return NO_RECORDING_STATUS

    candidate = upload if isinstance(upload, (str, Path)) else None
    if candidate is None:
        candidate = getattr(upload, "path", None) or getattr(upload, "name", None)

    if candidate:
        try:
            if Path(candidate).resolve() == EXAMPLE_PATH.resolve():
                return SYNTHETIC_READY_STATUS
        except (OSError, TypeError, ValueError):
            pass
    return CUSTOM_RECORDING_STATUS


def reset_demo():
    """Restore the bundled recording and the documented demo defaults."""

    return str(EXAMPLE_PATH), "local_boundary", 5, 0.30, 5, SYNTHETIC_READY_STATUS


def build_app() -> gr.Blocks:
    """Build the bilingual, ZeroGPU-compatible Gradio application."""

    with gr.Blocks(title="Wearable IMU Activity Timeline Demo") as app:
        gr.HTML(
            f"""
            <section class="imu-hero">
              <span class="imu-pill">Free ZeroGPU · Repository models · 仓库真实模型</span>
              <h1>Wearable IMU Activity Timeline Demo</h1>
              <p>Turn continuous 100 Hz wrist-IMU signals into timestamped activity records with the project's public multi-scale models.</p>
              <p>使用项目公开的多尺度模型，将连续 100 Hz 腕部 IMU 信号转换为带起止时间的活动记录。</p>
              <div class="imu-links" aria-label="Project links">
                <a class="imu-link primary" href="{PROJECT_WEBSITE_URL}" target="_blank" rel="noopener noreferrer">Project website / 项目主页 ↗</a>
                <a class="imu-link github" href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer">GitHub / 源码 ↗</a>
              </div>
            </section>

            <section class="imu-overview" aria-label="Project background, method, and demo guide">
              <article class="imu-card">
                <span class="imu-card-kicker">Background / 项目背景</span>
                <h2>From window labels to complete activity records</h2>
                <p>Long recordings need more than a class label: the system must recover each activity's start, end, count, and duration.</p>
                <p class="zh">长时记录不能只做窗口分类；系统还要恢复每段活动的类别、起止时间、次数与持续时长。</p>
              </article>
              <article class="imu-card">
                <span class="imu-card-kicker">Method / 方法</span>
                <h2>Multi-scale recognition with temporal decoding</h2>
                <div class="imu-flow">3 / 5 / 8 s CNN–BiLSTM → LBSA → TRL → &#123;activity, start, end&#125;</div>
                <p class="zh">三种时间尺度提取互补证据，LBSA 自适应融合，TRL 负责平滑、解码与边界修正。</p>
              </article>
              <article class="imu-card">
                <span class="imu-card-kicker">Run in 3 steps / 三步运行</span>
                <h2>Sample → infer → export</h2>
                <ol class="imu-steps">
                  <li>Keep the loaded sample, or upload a compatible TXT/TSV file. / 使用默认样例，或上传兼容文件。</li>
                  <li>Keep the defaults or adjust fusion and filtering. / 保持默认参数，或调整融合与过滤设置。</li>
                  <li>Click <strong>Run current recording</strong>, inspect the three result tabs, and download CSV. / 点击<strong>运行当前记录</strong>，查看三类结果并下载 CSV。</li>
                </ol>
              </article>
            </section>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                upload = gr.File(
                    value=str(EXAMPLE_PATH),
                    label="Input recording — synthetic sample loaded by default / 输入记录——默认载入合成样例",
                    file_types=[".txt", ".tsv"],
                    type="filepath",
                    interactive=True,
                )
                gr.HTML(
                    """
                    <div class="sensor-schema">
                      <p><strong>Paper notation / 论文记号：</strong>
                        <span class="sensor-symbol"><var>a</var><sub>x</sub></span>,
                        <span class="sensor-symbol"><var>a</var><sub>y</sub></span>,
                        <span class="sensor-symbol"><var>a</var><sub>z</sub></span>;
                        <span class="sensor-symbol"><var>&omega;</var><sub>x</sub></span>,
                        <span class="sensor-symbol"><var>&omega;</var><sub>y</sub></span>,
                        <span class="sensor-symbol"><var>&omega;</var><sub>z</sub></span>。
                      </p>
                      <p><strong>Required TSV headers / 必需文件列名：</strong>
                        <code>ACC_TIME</code>, <code>ACC_X</code>, <code>ACC_Y</code>,
                        <code>ACC_Z</code>, <code>GYRO_X</code>, <code>GYRO_Y</code>,
                        <code>GYRO_Z</code>。<code>ACC_TIME</code> uses milliseconds / 单位为毫秒。
                      </p>
                    </div>
                    """
                )
                with gr.Row():
                    run_button = gr.Button(
                        "Run current recording / 运行当前记录",
                        variant="primary",
                        elem_classes=["primary-action"],
                    )
                    reset_button = gr.Button(
                        "Reset to synthetic sample / 恢复合成样例",
                        variant="secondary",
                    )

            with gr.Column(scale=4):
                fusion_mode = gr.Dropdown(
                    choices=FUSION_CHOICES,
                    value="local_boundary",
                    label="How to combine the 3 / 5 / 8 s models / 怎样组合 3 / 5 / 8 秒模型",
                )
                min_duration = gr.Slider(
                    1,
                    180,
                    value=5,
                    step=1,
                    label="Ignore activity periods shorter than (s) / 忽略短于多少秒的活动",
                )
                confidence = gr.Slider(
                    0.0,
                    1.0,
                    value=0.30,
                    step=0.05,
                    label="Ignore predictions below this confidence / 忽略低于该置信度的结果",
                )
                top_k = gr.Slider(
                    0,
                    10,
                    value=5,
                    step=1,
                    label="Maximum records to show (0 = all) / 最多显示记录数（0 = 全部）",
                )

        gr.Markdown(
            "The sample is computer-generated and contains no participant data. "
            "Its short activity periods are for demonstration only, not a paper result. / "
            "样例由程序生成，不含参与者数据；其中的短活动区间仅用于演示，不是论文结果。",
            elem_classes=["privacy-note"],
        )

        status = gr.Markdown(SYNTHETIC_READY_STATUS)
        with gr.Tabs():
            with gr.Tab("Raw signals / 原始信号"):
                signal_plot = gr.Plot(label="Six IMU channels / 六路 IMU 信号")
            with gr.Tab("Activity likelihood and timeline / 活动概率与时间线"):
                timeline_plot = gr.Plot(label="Activity likelihood and final timeline / 活动概率与最终时间线")
            with gr.Tab("Activity records / 活动记录"):
                segment_table = gr.Dataframe(
                    headers=SEGMENT_COLUMNS,
                    interactive=False,
                    label="Detected activity records / 识别到的活动记录",
                )
                download = gr.File(label="Download CSV / 下载 CSV", interactive=False)

        with gr.Accordion("Before you upload: format, privacy, and limits / 上传前须知：格式、隐私与限制", open=False):
            gr.Markdown(
                """
                - The public demo accepts UTF-8 tab-separated TXT/TSV files, 800–60,000 valid samples,
                  at approximately 100 Hz. Extra columns are ignored.
                - Do not upload confidential or identifiable participant data to a public Space.
                - Files are processed for the current request; this application does not intentionally
                  persist them, but the hosting platform is shared infrastructure.
                - Predictions are research outputs, not medical, safety, or coaching advice. Sensor
                  placement, units, device differences, and population shift can materially affect results.

                - 公开演示接收 UTF-8 制表符分隔的 TXT/TSV 文件，支持 800–60,000 个有效样本，采样率约 100 Hz；额外列会被忽略。
                - 请勿向公开 Space 上传机密或可识别的受试者数据。
                - 预测仅用于研究展示，不构成医疗、安全或训练建议；佩戴位置、单位、设备差异与人群偏移都会影响结果。
                """
            )

        run_button.click(
            fn=segment_recording,
            inputs=[upload, fusion_mode, min_duration, confidence, top_k],
            outputs=[status, signal_plot, timeline_plot, segment_table, download],
            api_name="segment",
            concurrency_limit=1,
        )
        upload.change(
            fn=recording_selection_status,
            inputs=upload,
            outputs=status,
            queue=False,
            api_visibility="private",
        )
        reset_button.click(
            fn=reset_demo,
            inputs=None,
            outputs=[upload, fusion_mode, min_duration, confidence, top_k, status],
            queue=False,
            api_visibility="private",
        )

    return app


demo = build_app()


if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        theme=THEME,
        css=CSS,
        max_file_size="20mb",
    )
