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

CSS = """
:root {
  --imu-indigo: #4f46e5;
  --imu-violet: #7c3aed;
  --imu-ink: #172033;
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
            "Inference failed. Confirm the input format and try again; "
            "see the repository issue tracker if the problem persists."
        ) from exc


def build_app() -> gr.Blocks:
    """Build the bilingual, ZeroGPU-compatible Gradio application."""

    with gr.Blocks(title="Wearable IMU Activity Segmentation") as app:
        gr.HTML(
            """
            <section class="imu-hero">
              <span class="imu-pill">ZeroGPU · Real tracked models · 真实模型</span>
              <h1>Wearable IMU Activity Segmentation</h1>
              <p>Upload a 100 Hz accelerometer + gyroscope recording and inspect
              multi-scale CNN–BiLSTM probabilities, temporal decoding, and final
              activity segments.</p>
              <p>上传 100 Hz 加速度计与陀螺仪记录，查看多尺度概率、时序解码和最终活动片段。</p>
            </section>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                upload = gr.File(
                    label="IMU recording / IMU 记录",
                    file_types=[".txt", ".tsv"],
                    type="filepath",
                )
                gr.Markdown(
                    "Required columns: `ACC_TIME`, `ACC_X`, `ACC_Y`, `ACC_Z`, "
                    "`GYRO_X`, `GYRO_Y`, `GYRO_Z`. `ACC_TIME` uses milliseconds."
                )
                run_button = gr.Button(
                    "Run segmentation / 开始分割",
                    variant="primary",
                    elem_classes=["primary-action"],
                )

            with gr.Column(scale=4):
                fusion_mode = gr.Dropdown(
                    choices=list(FUSION_MODES),
                    value="local_boundary",
                    label="Multi-scale fusion / 多尺度融合",
                )
                min_duration = gr.Slider(
                    1,
                    180,
                    value=5,
                    step=1,
                    label="Minimum segment duration (s) / 最短片段（秒）",
                )
                confidence = gr.Slider(
                    0.0,
                    1.0,
                    value=0.30,
                    step=0.05,
                    label="Minimum confidence / 最低置信度",
                )
                top_k = gr.Slider(
                    0,
                    10,
                    value=5,
                    step=1,
                    label="Top-K segments (0 = off) / 最多片段数",
                )

        gr.Examples(
            examples=[[str(EXAMPLE_PATH), "local_boundary", 5, 0.30, 5]],
            inputs=[upload, fusion_mode, min_duration, confidence, top_k],
            label="Bundled synthetic example / 内置合成示例",
            cache_examples=False,
        )

        gr.Markdown(
            "This example is synthetic and contains no participant recording. "
            "Demo thresholds are intentionally shorter than the paper's conservative "
            "long-session reporting settings. / 示例为合成数据，不含受试者记录；演示阈值短于论文的长时记录设置。",
            elem_classes=["privacy-note"],
        )

        status = gr.Markdown("### Ready / 就绪\nUpload a file or select the bundled example.")
        with gr.Tabs():
            with gr.Tab("Signals / 信号"):
                signal_plot = gr.Plot(label="Six-channel IMU preview")
            with gr.Tab("Model timeline / 模型时间线"):
                timeline_plot = gr.Plot(label="Probabilities and decoded path")
            with gr.Tab("Segments / 活动片段"):
                segment_table = gr.Dataframe(
                    headers=SEGMENT_COLUMNS,
                    interactive=False,
                    label="Detected segments / 检测片段",
                )
                download = gr.File(label="Download CSV / 下载 CSV", interactive=False)

        with gr.Accordion("Input, privacy, and limitations / 输入、隐私与局限", open=False):
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

    return app


demo = build_app()


if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        theme=THEME,
        css=CSS,
        max_file_size="20mb",
    )
