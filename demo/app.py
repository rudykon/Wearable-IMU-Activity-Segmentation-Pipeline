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
            "No activity records could be generated. Check the file columns, "
            "timestamps, and sampling rate, then try again. / 无法生成活动记录，请检查文件列名、时间戳和采样率后重试。"
        ) from exc


def build_app() -> gr.Blocks:
    """Build the bilingual, ZeroGPU-compatible Gradio application."""

    with gr.Blocks(title="Wearable IMU Activity Timeline Demo") as app:
        gr.HTML(
            """
            <section class="imu-hero">
              <span class="imu-pill">Free ZeroGPU · Repository models · 仓库真实模型</span>
              <h1>Wearable IMU Activity Timeline Demo</h1>
              <p><code>synthetic_activity_imu.tsv</code> is loaded by default. Run it immediately,
              or replace it with a compatible 100 Hz wrist-motion recording.</p>
              <p>页面默认载入 <code>synthetic_activity_imu.tsv</code>。可以直接运行，也可以替换为兼容的 100 Hz 腕部运动记录。</p>
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
                gr.Markdown(
                    "The bundled sample is ready. Replace it only when using your own file. / "
                    "内置样例已经就绪；只有在测试自己的数据时才需要替换文件。\n\n"
                    "File columns / 文件列名：`ACC_TIME`, `ACC_X`, `ACC_Y`, `ACC_Z`, "
                    "`GYRO_X`, `GYRO_Y`, `GYRO_Z`。`ACC_TIME` uses milliseconds / 单位为毫秒。"
                )
                run_button = gr.Button(
                    "Run the loaded sample / 运行当前样例",
                    variant="primary",
                    elem_classes=["primary-action"],
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

        gr.Examples(
            examples=[[str(EXAMPLE_PATH), "local_boundary", 5, 0.30, 5]],
            inputs=[upload, fusion_mode, min_duration, confidence, top_k],
            label="Restore the bundled sample and defaults / 恢复内置样例与默认参数",
            cache_examples=False,
        )

        gr.Markdown(
            "The sample is computer-generated and contains no participant data. "
            "Its short activity periods are for demonstration only, not a paper result. / "
            "样例由程序生成，不含参与者数据；其中的短活动区间仅用于演示，不是论文结果。",
            elem_classes=["privacy-note"],
        )

        status = gr.Markdown(
            "### Synthetic sample ready / 合成样例已就绪\n"
            "`synthetic_activity_imu.tsv` is already selected. Click **Run the loaded sample / 运行当前样例** to generate the three outputs. / "
            "`synthetic_activity_imu.tsv` 已默认选中，点击 **运行当前样例** 即可生成三类结果。"
        )
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

    return app


demo = build_app()


if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        theme=THEME,
        css=CSS,
        max_file_size="20mb",
    )
