"""Gradio entry point for the Hugging Face Space demo."""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import spaces
import gradio as gr

from demo.runtime import (
    SEGMENT_COLUMNS,
    DemoInputError,
    DemoResult,
    export_segments,
    fusion_choices,
    make_signal_figure,
    make_timeline_figure,
    normalise_language,
    run_pipeline,
    segments_dataframe,
    status_markdown,
    timeline_class_key,
)

LOGGER = logging.getLogger(__name__)
EXAMPLE_PATH = ROOT / "demo" / "examples" / "synthetic_activity_imu.tsv"
EXAMPLE_SIZE = EXAMPLE_PATH.stat().st_size
with EXAMPLE_PATH.open("rb") as _example_file:
    EXAMPLE_SHA256 = hashlib.file_digest(_example_file, "sha256").hexdigest()
PROJECT_WEBSITE_URL = (
    "https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/"
)
GITHUB_URL = "https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline"

COPY = {
    "en": {
        "upload": "Recording file — choose to replace the loaded sample",
        "run": "Run current recording",
        "reset": "Reset to synthetic sample",
        "fusion": "How to combine the 3, 5, and 8 s models",
        "min_duration": "Ignore activity periods shorter than (s)",
        "confidence": "Ignore predictions below this confidence",
        "top_k": "Maximum records to show (0 = all)",
        "privacy": (
            "The sample is computer-generated and contains no participant data. "
            "Its short activity periods are for demonstration only, not a paper result."
        ),
        "tab_signal": "Raw signals",
        "signal_plot": "Six IMU channels",
        "tab_timeline": "Activity likelihood and timeline",
        "timeline_plot": "Activity likelihood and final timeline",
        "tab_records": "Activity records",
        "segment_table": "Detected activity records",
        "download": "Download CSV",
        "accordion": "Before you upload: format, privacy, and limits",
        "generic_error": (
            "No activity records could be generated. Check the file columns, "
            "timestamps, and sampling rate, then try again."
        ),
        "synthetic_status": (
            "### Synthetic sample ready\n"
            "`synthetic_activity_imu.tsv` is selected. Click **Run current recording** "
            "to generate the three outputs."
        ),
        "custom_status": (
            "### Custom recording selected\n"
            "Review the settings, then click **Run current recording**."
        ),
        "empty_status": (
            "### Select a recording\n"
            "Upload a compatible TXT/TSV file, or restore the synthetic sample."
        ),
        "limits": (
            "- The public demo accepts UTF-8 tab-separated TXT/TSV files, "
            "800–60,000 valid samples, at approximately 100 Hz. Extra columns are ignored.\n"
            "- Do not upload confidential or identifiable participant data to a public Space.\n"
            "- Files are processed for the current request; this application does not "
            "intentionally persist them, but the hosting platform is shared infrastructure.\n"
            "- Predictions are research outputs, not medical, safety, or coaching advice. "
            "Sensor placement, units, device differences, and population shift can materially "
            "affect results."
        ),
    },
    "zh": {
        "upload": "记录文件——点击可替换当前样例",
        "run": "运行当前记录",
        "reset": "恢复合成样例",
        "fusion": "怎样组合 3、5、8 秒模型",
        "min_duration": "忽略短于多少秒的活动",
        "confidence": "忽略低于该置信度的结果",
        "top_k": "最多显示记录数（0 表示全部）",
        "privacy": (
            "样例由程序生成，不含参与者数据；其中的短活动区间仅用于演示，不是论文结果。"
        ),
        "tab_signal": "原始信号",
        "signal_plot": "六路 IMU 信号",
        "tab_timeline": "活动概率与时间线",
        "timeline_plot": "活动概率与最终时间线",
        "tab_records": "活动记录",
        "segment_table": "识别到的活动记录",
        "download": "下载 CSV",
        "accordion": "上传前须知：格式、隐私与限制",
        "generic_error": "无法生成活动记录，请检查文件列名、时间戳和采样率后重试。",
        "synthetic_status": (
            "### 合成样例已就绪\n"
            "已选择 `synthetic_activity_imu.tsv`。点击 **运行当前记录** 即可生成三类结果。"
        ),
        "custom_status": "### 已选择自定义记录\n请确认参数设置，然后点击 **运行当前记录**。",
        "empty_status": "### 请选择记录\n请上传兼容的 TXT/TSV 文件，或恢复合成样例。",
        "limits": (
            "- 公开演示接收 UTF-8 制表符分隔的 TXT/TSV 文件，支持 800–60,000 个有效样本，"
            "采样率约 100 Hz；额外列会被忽略。\n"
            "- 请勿向公开 Space 上传机密或可识别的参与者数据。\n"
            "- 文件仅用于处理当前请求，本应用不会主动持久化文件；但托管平台属于共享基础设施。\n"
            "- 预测仅用于研究展示，不构成医疗、安全或训练建议；佩戴位置、单位、设备差异与"
            "人群偏移都会显著影响结果。"
        ),
    },
}


def _copy(language: str | None) -> dict[str, str]:
    return COPY[normalise_language(language)]


def hero_html(language: str | None = "en") -> str:
    """Return the project introduction in exactly one language."""

    if normalise_language(language) == "zh":
        return f"""
        <section class="imu-hero" lang="zh-CN">
          <span class="imu-pill">免费 ZeroGPU · 公开仓库模型</span>
          <h1>可穿戴 IMU 活动时间线演示</h1>
          <p>使用项目公开的多尺度模型，将连续 100 Hz 腕部 IMU 信号转换为带起止时间的活动记录。</p>
          <div class="imu-links" aria-label="项目链接">
            <a class="imu-link primary" href="{PROJECT_WEBSITE_URL}" target="_blank" rel="noopener noreferrer">项目网站 ↗</a>
            <a class="imu-link github" href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer">GitHub 源码 ↗</a>
          </div>
        </section>
        <section class="imu-overview" aria-label="项目背景、方法与演示步骤">
          <article class="imu-card">
            <span class="imu-card-kicker">项目背景</span>
            <h2>从窗口标签到完整活动记录</h2>
            <p>长时记录不能只做窗口分类；系统还要恢复每段活动的类别、起止时间、次数与持续时长。</p>
          </article>
          <article class="imu-card">
            <span class="imu-card-kicker">方法</span>
            <h2>多尺度识别与时间解码</h2>
            <div class="imu-flow">3、5、8 秒 CNN–BiLSTM → LBSA → TRL → &#123;活动，开始，结束&#125;</div>
            <p>三种时间尺度提取互补证据，LBSA 自适应融合，TRL 负责平滑、解码与边界修正。</p>
          </article>
          <article class="imu-card">
            <span class="imu-card-kicker">三步运行</span>
            <h2>样例 → 推理 → 导出</h2>
            <ol class="imu-steps">
              <li>使用默认样例，或上传兼容的 TXT/TSV 文件。</li>
              <li>保持默认参数，或调整融合与过滤设置。</li>
              <li>点击<strong>运行当前记录</strong>，查看三类结果并下载 CSV。</li>
            </ol>
          </article>
        </section>
        """
    return f"""
    <section class="imu-hero" lang="en">
      <span class="imu-pill">Free ZeroGPU · Public repository models</span>
      <h1>Wearable IMU Activity Timeline Demo</h1>
      <p>Turn continuous 100 Hz wrist-IMU signals into timestamped activity records with the project's public multi-scale models.</p>
      <div class="imu-links" aria-label="Project links">
        <a class="imu-link primary" href="{PROJECT_WEBSITE_URL}" target="_blank" rel="noopener noreferrer">Project website ↗</a>
        <a class="imu-link github" href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer">GitHub source ↗</a>
      </div>
    </section>
    <section class="imu-overview" aria-label="Project background, method, and demo guide">
      <article class="imu-card">
        <span class="imu-card-kicker">Background</span>
        <h2>From window labels to complete activity records</h2>
        <p>Long recordings need more than a class label: the system must recover each activity's start, end, count, and duration.</p>
      </article>
      <article class="imu-card">
        <span class="imu-card-kicker">Method</span>
        <h2>Multi-scale recognition with temporal decoding</h2>
        <div class="imu-flow">3 / 5 / 8 s CNN–BiLSTM → LBSA → TRL → &#123;activity, start, end&#125;</div>
        <p>Three temporal scales provide complementary evidence, LBSA adapts their fusion, and TRL smooths, decodes, and refines boundaries.</p>
      </article>
      <article class="imu-card">
        <span class="imu-card-kicker">Run in 3 steps</span>
        <h2>Sample → infer → export</h2>
        <ol class="imu-steps">
          <li>Keep the loaded sample, or upload a compatible TXT/TSV file.</li>
          <li>Keep the defaults or adjust fusion and filtering.</li>
          <li>Click <strong>Run current recording</strong>, inspect the three result tabs, and download CSV.</li>
        </ol>
      </article>
    </section>
    """


def sensor_schema_html(language: str | None = "en") -> str:
    """Return single-language input guidance with paper-style channel notation."""

    notation = """
      <span class="sensor-symbol"><var>a</var><sub>x</sub></span>,
      <span class="sensor-symbol"><var>a</var><sub>y</sub></span>,
      <span class="sensor-symbol"><var>a</var><sub>z</sub></span>;
      <span class="sensor-symbol"><var>&omega;</var><sub>x</sub></span>,
      <span class="sensor-symbol"><var>&omega;</var><sub>y</sub></span>,
      <span class="sensor-symbol"><var>&omega;</var><sub>z</sub></span>
    """
    headers = (
        "<code>ACC_TIME</code>, <code>ACC_X</code>, <code>ACC_Y</code>, "
        "<code>ACC_Z</code>, <code>GYRO_X</code>, <code>GYRO_Y</code>, "
        "<code>GYRO_Z</code>"
    )
    if normalise_language(language) == "zh":
        return f"""
        <div class="sensor-schema" lang="zh-CN">
          <p><strong>论文记号：</strong>{notation}。</p>
          <p><strong>必需 TSV 列名：</strong>{headers}。<code>ACC_TIME</code> 的单位为毫秒。</p>
        </div>
        """
    return f"""
    <div class="sensor-schema" lang="en">
      <p><strong>Paper notation:</strong>{notation}.</p>
      <p><strong>Required TSV headers:</strong>{headers}. <code>ACC_TIME</code> uses milliseconds.</p>
    </div>
    """

CSS = """
:root {
  --imu-indigo: #4f46e5;
  --imu-violet: #7c3aed;
  --imu-ink: #172033;
  --imu-muted: #5b6573;
  --imu-border: #d9e1e8;
}
.gradio-container { max-width: 1240px !important; }
#language-switch {
  max-width: 260px;
  margin: 0 0 .7rem auto;
}
.recording-upload button { width: 100%; }
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
  #language-switch { width: 100%; max-width: none; }
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


def segment_recording(
    upload,
    fusion_mode,
    min_duration_sec,
    confidence_min,
    top_k,
    language="en",
):
    """Turn one upload into localized summaries, records, and a CSV export."""

    locale = normalise_language(language)
    try:
        result = _run_zero_gpu_pipeline(
            upload,
            fusion_mode,
            min_duration_sec,
            confidence_min,
            top_k,
        )
        return (
            result,
            status_markdown(result, str(fusion_mode), locale),
            make_signal_figure(result.recording),
            make_timeline_figure(result),
            segments_dataframe(result, locale),
            export_segments(result, locale),
        )
    except DemoInputError as exc:
        raise gr.Error(exc.localized(locale)) from exc
    except Exception as exc:  # Avoid exposing server internals in the public UI.
        LOGGER.exception("Space inference failed")
        raise gr.Error(_copy(locale)["generic_error"]) from exc


def recording_selection_status(upload, language="en") -> str:
    """Describe the selected file in one language without exposing its name."""

    copy = _copy(language)
    if upload is None:
        return copy["empty_status"]

    candidate = upload if isinstance(upload, (str, Path)) else None
    if candidate is None:
        candidate = getattr(upload, "path", None) or getattr(upload, "name", None)

    if candidate:
        try:
            candidate_path = Path(candidate)
            if candidate_path.resolve() == EXAMPLE_PATH.resolve():
                return copy["synthetic_status"]
            if (
                candidate_path.name == EXAMPLE_PATH.name
                and candidate_path.stat().st_size == EXAMPLE_SIZE
            ):
                with candidate_path.open("rb") as selected_file:
                    selected_sha256 = hashlib.file_digest(
                        selected_file, "sha256"
                    ).hexdigest()
                if selected_sha256 == EXAMPLE_SHA256:
                    return copy["synthetic_status"]
        except (OSError, TypeError, ValueError):
            pass
    return copy["custom_status"]


def recording_selection_update(upload, language="en"):
    """Clear stale outputs when the selected recording changes."""

    locale = normalise_language(language)
    return (
        None,
        recording_selection_status(upload, locale),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None, headers=SEGMENT_COLUMNS[locale]),
        gr.update(value=None),
    )


def reset_demo(language="en"):
    """Restore the sample, defaults, and an empty localized result area."""

    locale = normalise_language(language)
    return (
        str(EXAMPLE_PATH),
        "local_boundary",
        5,
        0.30,
        5,
        None,
        _copy(locale)["synthetic_status"],
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None, headers=SEGMENT_COLUMNS[locale]),
        gr.update(value=None),
    )


def localize_interface(language, upload, last_result, fusion_mode):
    """Update every visible app-authored string without rerunning the model."""

    locale = normalise_language(language)
    copy = _copy(locale)
    result = last_result if isinstance(last_result, DemoResult) else None
    if result is None:
        status = recording_selection_status(upload, locale)
        table_update = gr.update(
            headers=SEGMENT_COLUMNS[locale],
            label=copy["segment_table"],
        )
        download_update = gr.update(label=copy["download"])
    else:
        status = status_markdown(result, str(fusion_mode), locale)
        table_update = gr.update(
            value=segments_dataframe(result, locale),
            headers=SEGMENT_COLUMNS[locale],
            label=copy["segment_table"],
        )
        download_update = gr.update(
            value=export_segments(result, locale),
            label=copy["download"],
        )

    return (
        locale,
        gr.update(value=hero_html(locale)),
        gr.update(label=copy["upload"]),
        gr.update(value=sensor_schema_html(locale)),
        gr.update(value=copy["run"]),
        gr.update(value=copy["reset"]),
        gr.update(
            choices=fusion_choices(locale),
            value=str(fusion_mode),
            label=copy["fusion"],
        ),
        gr.update(label=copy["min_duration"]),
        gr.update(label=copy["confidence"]),
        gr.update(label=copy["top_k"]),
        gr.update(value=copy["privacy"]),
        status,
        gr.update(label=copy["tab_signal"]),
        gr.update(label=copy["signal_plot"]),
        gr.update(label=copy["tab_timeline"]),
        gr.update(label=copy["timeline_plot"]),
        gr.update(value=timeline_class_key(locale)),
        gr.update(label=copy["tab_records"]),
        table_update,
        download_update,
        gr.update(label=copy["accordion"]),
        gr.update(value=copy["limits"]),
    )


def build_app() -> gr.Blocks:
    """Build the switchable single-language, ZeroGPU-compatible application."""

    with gr.Blocks(title="Wearable IMU Activity Timeline Demo") as app:
        language_state = gr.State("en")
        last_result = gr.State(None, time_to_live=1_800)
        language_picker = gr.Radio(
            choices=[("English", "en"), ("简体中文", "zh")],
            value="en",
            show_label=False,
            container=False,
            elem_id="language-switch",
        )
        hero = gr.HTML(hero_html("en"))

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                upload = gr.UploadButton(
                    value=str(EXAMPLE_PATH),
                    label=COPY["en"]["upload"],
                    file_types=[".txt", ".tsv"],
                    type="filepath",
                    interactive=True,
                    elem_classes=["recording-upload"],
                )
                sensor_schema = gr.HTML(sensor_schema_html("en"))
                with gr.Row():
                    run_button = gr.Button(
                        COPY["en"]["run"],
                        variant="primary",
                        elem_classes=["primary-action"],
                    )
                    reset_button = gr.Button(
                        COPY["en"]["reset"],
                        variant="secondary",
                    )

            with gr.Column(scale=4):
                fusion_mode = gr.Dropdown(
                    choices=fusion_choices("en"),
                    value="local_boundary",
                    label=COPY["en"]["fusion"],
                )
                min_duration = gr.Slider(
                    1,
                    180,
                    value=5,
                    step=1,
                    label=COPY["en"]["min_duration"],
                )
                confidence = gr.Slider(
                    0.0,
                    1.0,
                    value=0.30,
                    step=0.05,
                    label=COPY["en"]["confidence"],
                )
                top_k = gr.Slider(
                    0,
                    10,
                    value=5,
                    step=1,
                    label=COPY["en"]["top_k"],
                )

        privacy = gr.Markdown(
            COPY["en"]["privacy"],
            elem_classes=["privacy-note"],
        )

        status = gr.Markdown(COPY["en"]["synthetic_status"])
        with gr.Tabs():
            with gr.Tab(COPY["en"]["tab_signal"]) as signal_tab:
                signal_plot = gr.Plot(label=COPY["en"]["signal_plot"])
            with gr.Tab(COPY["en"]["tab_timeline"]) as timeline_tab:
                timeline_plot = gr.Plot(label=COPY["en"]["timeline_plot"])
                timeline_key = gr.Markdown(timeline_class_key("en"))
            with gr.Tab(COPY["en"]["tab_records"]) as records_tab:
                segment_table = gr.Dataframe(
                    headers=SEGMENT_COLUMNS["en"],
                    interactive=False,
                    label=COPY["en"]["segment_table"],
                )
                download = gr.File(label=COPY["en"]["download"], interactive=False)

        with gr.Accordion(COPY["en"]["accordion"], open=False) as upload_accordion:
            limits = gr.Markdown(COPY["en"]["limits"])

        language_picker.change(
            fn=localize_interface,
            inputs=[language_picker, upload, last_result, fusion_mode],
            outputs=[
                language_state,
                hero,
                upload,
                sensor_schema,
                run_button,
                reset_button,
                fusion_mode,
                min_duration,
                confidence,
                top_k,
                privacy,
                status,
                signal_tab,
                signal_plot,
                timeline_tab,
                timeline_plot,
                timeline_key,
                records_tab,
                segment_table,
                download,
                upload_accordion,
                limits,
            ],
            queue=False,
            api_visibility="private",
        )

        run_button.click(
            fn=segment_recording,
            inputs=[
                upload,
                fusion_mode,
                min_duration,
                confidence,
                top_k,
                language_state,
            ],
            outputs=[
                last_result,
                status,
                signal_plot,
                timeline_plot,
                segment_table,
                download,
            ],
            api_name="segment",
            concurrency_limit=1,
        )
        upload.change(
            fn=recording_selection_update,
            inputs=[upload, language_state],
            outputs=[
                last_result,
                status,
                signal_plot,
                timeline_plot,
                segment_table,
                download,
            ],
            queue=False,
            api_visibility="private",
        )
        reset_button.click(
            fn=reset_demo,
            inputs=language_state,
            outputs=[
                upload,
                fusion_mode,
                min_duration,
                confidence,
                top_k,
                last_result,
                status,
                signal_plot,
                timeline_plot,
                segment_table,
                download,
            ],
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
        footer_links=[],
        run_history=False,
    )
