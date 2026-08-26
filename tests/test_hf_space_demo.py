"""Lightweight tests for the Hugging Face Space adapter (no torch required)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "imu-matplotlib-tests"))

import matplotlib.pyplot as plt

from demo.runtime import (
    CLASS_LABELS,
    DemoInputError,
    DemoResult,
    load_imu_recording,
    make_signal_figure,
    make_timeline_figure,
    segments_dataframe,
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
        self.assertIn("value=str(EXAMPLE_PATH)", app_source)
        self.assertIn("Synthetic sample ready / 合成样例已就绪", app_source)
        self.assertIn("Run the loaded sample / 运行当前样例", app_source)

    def test_missing_channel_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.tsv"
            path.write_text("ACC_TIME\tACC_X\n1\t0\n", encoding="utf-8")
            with self.assertRaisesRegex(DemoInputError, "Missing required columns"):
                load_imu_recording(path, apply_filter=False)

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
        table = segments_dataframe(result)
        self.assertEqual(table.loc[0, "Activity"], "Running")
        self.assertEqual(table.loc[0, "活动"], "跑步")
        signal_figure = make_signal_figure(recording)
        timeline_figure = make_timeline_figure(result)
        self.assertEqual(len(signal_figure.axes), 2)
        self.assertEqual(len(timeline_figure.axes), 2)
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
