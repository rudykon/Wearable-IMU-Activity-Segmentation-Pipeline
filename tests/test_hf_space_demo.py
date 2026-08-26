"""Lightweight tests for the Hugging Face Space adapter (no torch required)."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "imu-matplotlib-tests"))

import matplotlib.pyplot as plt
import pandas as pd

from demo.runtime import (
    CLASS_LABELS,
    DemoInputError,
    DemoResult,
    SEGMENT_COLUMNS,
    export_segments,
    load_imu_recording,
    make_signal_figure,
    make_timeline_figure,
    segments_dataframe,
    status_markdown,
    timeline_class_key,
)

EXAMPLE = ROOT / "demo" / "examples" / "synthetic_activity_imu.tsv"


class DemoRuntimeTests(unittest.TestCase):
    def test_bundled_example_matches_input_contract(self) -> None:
        recording = load_imu_recording(EXAMPLE, apply_filter=False)
        self.assertEqual(recording.data.shape, (12_000, 7))
        self.assertAlmostEqual(recording.sample_rate_hz, 100.0, places=4)
        self.assertAlmostEqual(recording.duration_sec, 119.99, places=2)
        self.assertEqual(recording.user_id, "synthetic_activity_imu")

    def test_space_defaults_to_bundled_example(self) -> None:
        app_source = (ROOT / "demo" / "app.py").read_text(encoding="utf-8")
        self.assertIn("upload = gr.UploadButton(", app_source)
        self.assertIn("value=str(EXAMPLE_PATH)", app_source)
        self.assertIn('choices=[("English", "en"), ("简体中文", "zh")]', app_source)
        self.assertIn('"### Synthetic sample ready\\n"', app_source)
        self.assertIn('"### 合成样例已就绪\\n"', app_source)
        self.assertIn('fn=localize_interface', app_source)
        self.assertNotIn("Run current recording / 运行当前记录", app_source)
        self.assertNotIn("Synthetic sample ready / 合成样例已就绪", app_source)

    def test_space_updates_selection_status_and_has_a_real_reset(self) -> None:
        app_source = (ROOT / "demo" / "app.py").read_text(encoding="utf-8")
        self.assertIn('"### Custom recording selected\\n"', app_source)
        self.assertIn('"### 已选择自定义记录\\n', app_source)
        self.assertIn("fn=recording_selection_update", app_source)
        self.assertIn("upload.change(", app_source)
        self.assertIn("hashlib.file_digest", app_source)
        self.assertIn("selected_sha256 == EXAMPLE_SHA256", app_source)
        self.assertIn("fn=reset_demo", app_source)
        self.assertNotIn("gr.Examples(", app_source)
        self.assertIn('def reset_demo(language="en"):', app_source)
        self.assertIn(
            'str(EXAMPLE_PATH),\n        "local_boundary",\n        5,\n        0.30,\n        5,',
            app_source,
        )
        self.assertIn('gr.update(value=None, headers=SEGMENT_COLUMNS[locale])', app_source)

    def test_space_explains_the_project_and_links_to_primary_resources(self) -> None:
        app_source = (ROOT / "demo" / "app.py").read_text(encoding="utf-8")
        self.assertIn(
            "https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/",
            app_source,
        )
        self.assertIn(
            "https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline",
            app_source,
        )
        self.assertIn(">Background</span>", app_source)
        self.assertIn(">项目背景</span>", app_source)
        self.assertIn(">Method</span>", app_source)
        self.assertIn(">方法</span>", app_source)
        self.assertIn(">Run in 3 steps</span>", app_source)
        self.assertIn(">三步运行</span>", app_source)
        self.assertIn("3 / 5 / 8 s CNN–BiLSTM → LBSA → TRL", app_source)
        self.assertIn('target="_blank" rel="noopener noreferrer"', app_source)

    def test_language_switch_updates_every_visible_app_region(self) -> None:
        app_source = (ROOT / "demo" / "app.py").read_text(encoding="utf-8")
        self.assertIn(
            "def localize_interface(language, upload, last_result, fusion_mode):",
            app_source,
        )
        self.assertIn('gr.update(value=hero_html(locale))', app_source)
        self.assertIn('gr.update(value=sensor_schema_html(locale))', app_source)
        self.assertIn('choices=fusion_choices(locale)', app_source)
        self.assertIn('headers=SEGMENT_COLUMNS[locale]', app_source)
        self.assertIn('value=export_segments(result, locale)', app_source)
        self.assertIn('gr.update(value=timeline_class_key(locale))', app_source)
        self.assertIn('language_state = gr.State("en")', app_source)
        self.assertIn('last_result = gr.State(None, time_to_live=1_800)', app_source)

    def test_public_segment_api_keeps_five_user_supplied_parameters(self) -> None:
        app_source = (ROOT / "demo" / "app.py").read_text(encoding="utf-8")
        self.assertIn('api_name="segment"', app_source)
        self.assertIn('top_k,\n                language_state,', app_source)
        self.assertIn('inputs=[language_picker, upload, last_result, fusion_mode]', app_source)

    @unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio is not installed")
    def test_gradio_applies_chinese_layout_and_reset_updates(self) -> None:
        if importlib.util.find_spec("spaces") is None:
            spaces_stub = types.ModuleType("spaces")
            spaces_stub.GPU = lambda **_kwargs: lambda function: function
            sys.modules["spaces"] = spaces_stub

        from gradio.state_holder import SessionState
        from demo import app as space_app

        endpoint = space_app.demo.get_api_info()["named_endpoints"]["/segment"]
        self.assertEqual(len(endpoint["parameters"]), 5)

        async def exercise_events() -> None:
            state = SessionState(space_app.demo)
            upload = {
                "path": str(EXAMPLE),
                "orig_name": EXAMPLE.name,
                "meta": {"_type": "gradio.FileData"},
            }
            localized = await space_app.demo.process_api(
                0,
                ["zh", upload, None, "local_boundary"],
                state=state,
                explicit_call=True,
            )
            self.assertEqual(localized["data"][12]["label"], "原始信号")
            self.assertEqual(
                localized["data"][20]["label"],
                "上传前须知：格式、隐私与限制",
            )

            reset = await space_app.demo.process_api(
                3,
                [None],
                state=state,
                explicit_call=True,
            )
            self.assertIn("合成样例已就绪", reset["data"][6])
            self.assertEqual(reset["data"][9]["headers"], SEGMENT_COLUMNS["zh"])

        asyncio.run(exercise_events())

    def test_missing_channel_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.tsv"
            path.write_text("ACC_TIME\tACC_X\n1\t0\n", encoding="utf-8")
            with self.assertRaisesRegex(DemoInputError, "Missing required columns"):
                load_imu_recording(path, apply_filter=False)

            try:
                load_imu_recording(path, apply_filter=False)
            except DemoInputError as exc:
                self.assertEqual(
                    exc.localized("zh"),
                    "缺少必需列：ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z",
                )

    def test_figures_and_segment_table_are_bounded(self) -> None:
        recording = load_imu_recording(EXAMPLE, apply_filter=False)
        timestamps = recording.data[150:350:20, 0]
        probabilities = __import__("numpy").full((len(timestamps), len(CLASS_LABELS)), 1 / 6)
        path = __import__("numpy").zeros(len(timestamps), dtype=int)
        result = DemoResult(
            recording=recording,
            timestamps=timestamps,
            probabilities=probabilities,
            decoded_path=path,
            segments=[
                {
                    "class_name": "跑步",
                    "start_ts": int(recording.data[200, 0]),
                    "end_ts": int(recording.data[700, 0]),
                    "duration": 5.0,
                    "confidence": 0.81234,
                }
            ],
            device="cpu",
            model_scales=("3s", "5s", "8s"),
        )
        table_en = segments_dataframe(result, "en")
        table_zh = segments_dataframe(result, "zh")
        self.assertEqual(list(table_en.columns), SEGMENT_COLUMNS["en"])
        self.assertEqual(list(table_zh.columns), SEGMENT_COLUMNS["zh"])
        self.assertEqual(table_en.loc[0, "Activity"], "Running")
        self.assertEqual(table_zh.loc[0, "活动"], "跑步")
        self.assertNotIn("活动", table_en.columns)
        self.assertNotIn("Activity", table_zh.columns)

        export_en = Path(export_segments(result, "en"))
        export_zh = Path(export_segments(result, "zh"))
        self.assertEqual(pd.read_csv(export_en).columns[1], "activity")
        self.assertEqual(pd.read_csv(export_zh).columns[1], "活动")
        self.assertTrue(export_en.name.endswith("_segments_en.csv"))
        self.assertTrue(export_zh.name.endswith("_segments_zh.csv"))

        status_en = status_markdown(result, "local_boundary", "en")
        status_zh = status_markdown(result, "local_boundary", "zh")
        self.assertIn("Activity records ready", status_en)
        self.assertNotIn("活动记录已生成", status_en)
        self.assertIn("活动记录已生成", status_zh)
        self.assertNotIn("Activity records ready", status_zh)
        self.assertIn("$c_4$ = Running", timeline_class_key("en"))
        self.assertIn("$c_4$ = 跑步", timeline_class_key("zh"))

        signal_figure = make_signal_figure(recording)
        timeline_figure = make_timeline_figure(result)
        self.assertEqual(len(signal_figure.axes), 2)
        self.assertEqual(len(timeline_figure.axes), 2)
        self.assertEqual(
            [text.get_text() for text in signal_figure.axes[0].get_legend().get_texts()],
            [r"$a_x$", r"$a_y$", r"$a_z$"],
        )
        self.assertEqual(
            [text.get_text() for text in signal_figure.axes[1].get_legend().get_texts()],
            [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"],
        )
        self.assertEqual(
            [text.get_text() for text in timeline_figure.axes[0].get_legend().get_texts()],
            [rf"$c_{index}$" for index in range(len(CLASS_LABELS))],
        )
        plt.close(signal_figure)
        plt.close(timeline_figure)

    def test_space_metadata_is_kept_out_of_github_readme(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        metadata = (ROOT / "demo" / "space-readme-frontmatter.md").read_text(
            encoding="utf-8"
        )
        self.assertFalse(readme.startswith("---\n"))
        self.assertTrue(readme.startswith('<p align="center">'))
        self.assertTrue(metadata.startswith("---\n"))
        self.assertTrue(metadata.endswith("---\n"))
        self.assertIn("sdk: gradio", metadata)
        self.assertIn("app_file: demo/app.py", metadata)
        self.assertIn("suggested_hardware: zero-a10g", metadata)

    def test_deployment_requests_free_zerogpu_hardware(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "hugging-face-space.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('--flavor zero-a10g', workflow)
        self.assertIn("demo/space-readme-frontmatter.md", workflow)
        self.assertIn('mktemp -d "${RUNNER_TEMP}/hugging-face-space.XXXXXX"', workflow)
        self.assertNotIn("${{ runner.temp }}", workflow)
        self.assertIn('uvx hf upload "${HF_SPACE_ID}" "${HF_SPACE_STAGE}" .', workflow)

        app_source = (ROOT / "demo" / "app.py").read_text(encoding="utf-8")
        self.assertIn("@spaces.GPU(duration=30)", app_source)


if __name__ == "__main__":
    unittest.main()
