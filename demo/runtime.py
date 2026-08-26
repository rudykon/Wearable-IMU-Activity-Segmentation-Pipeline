"""Runtime adapter for the Hugging Face Space demo.

The adapter deliberately keeps model imports lazy. Importing the Gradio app is
therefore inexpensive, while the public Hugging Face checkpoints are downloaded
when needed and loaded only before the first inference request.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "imu-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

REQUIRED_COLUMNS = [
    "ACC_TIME",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "GYRO_X",
    "GYRO_Y",
    "GYRO_Z",
]
FUSION_MODES = (
    "local_boundary",
    "average",
    "dynamic_boundary",
    "confident_conflict",
    "weighted_long",
    "weighted_balanced",
)
FUSION_MODE_LABELS = {
    "local_boundary": "Adaptive near activity changes / 活动切换处自适应",
    "average": "Simple average / 简单平均",
    "dynamic_boundary": "Dynamic boundary weighting / 动态边界加权",
    "confident_conflict": "Prefer confident agreement / 优先高置信一致判断",
    "weighted_long": "Prefer longer windows / 偏重长窗口",
    "weighted_balanced": "Balanced fixed weights / 固定均衡权重",
}
CLASS_LABELS = [
    "Background",
    "Badminton",
    "Jump rope",
    "Fly",
    "Running",
    "Table tennis",
]
ACTIVITY_EN = {
    "羽毛球": "Badminton",
    "跳绳": "Jump rope",
    "飞鸟": "Fly",
    "跑步": "Running",
    "乒乓球": "Table tennis",
}
SEGMENT_COLUMNS = [
    "Activity",
    "活动",
    "Start (s)",
    "End (s)",
    "Duration (s)",
    "Confidence",
]

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SAMPLES = 60_000
MIN_SAMPLES = 800
EXPECTED_SAMPLE_INTERVAL_MS = (8.0, 12.0)

_INFERENCE_LOCK = threading.Lock()


class DemoInputError(ValueError):
    """Raised when an uploaded recording does not match the public contract."""


@dataclass(frozen=True)
class Recording:
    """Validated IMU recording and derived metadata."""

    data: np.ndarray
    user_id: str
    sample_rate_hz: float
    duration_sec: float


@dataclass(frozen=True)
class DemoResult:
    """Structured outputs returned by one real model inference request."""

    recording: Recording
    timestamps: np.ndarray
    probabilities: np.ndarray
    decoded_path: np.ndarray
    segments: list[dict[str, Any]]
    device: str
    model_scales: tuple[str, ...]


def _normalise_upload_path(upload: str | Path | Any) -> Path:
    if upload is None:
        raise DemoInputError("Upload a TSV/TXT recording or choose the bundled example.")

    candidate = upload if isinstance(upload, (str, os.PathLike)) else getattr(upload, "name", upload)
    path = Path(str(candidate)).expanduser()
    if not path.is_file():
        raise DemoInputError("The uploaded file is no longer available. Please upload it again.")
    if path.suffix.lower() not in {".txt", ".tsv"}:
        raise DemoInputError("Use a UTF-8 tab-separated .txt or .tsv file.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise DemoInputError("The upload exceeds the 20 MB demo limit.")
    return path


def load_imu_recording(upload: str | Path | Any, *, apply_filter: bool = True) -> Recording:
    """Read and validate the canonical seven-column 100 Hz input format."""

    path = _normalise_upload_path(upload)
    try:
        frame = pd.read_csv(path, sep="\t", low_memory=False)
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DemoInputError("Could not parse the file as UTF-8 tab-separated text.") from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    if "GYRO_Z" not in frame.columns and "GYRO_" in frame.columns:
        frame = frame.rename(columns={"GYRO_": "GYRO_Z"})

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DemoInputError("Missing required columns: " + ", ".join(missing))

    numeric = frame[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    numeric = numeric[numeric["ACC_TIME"] > 0]
    numeric = numeric.sort_values("ACC_TIME").reset_index(drop=True)

    if len(numeric) < MIN_SAMPLES:
        raise DemoInputError(
            f"At least {MIN_SAMPLES:,} valid samples (8 seconds at 100 Hz) are required."
        )
    if len(numeric) > MAX_SAMPLES:
        raise DemoInputError(
            f"This public demo accepts at most {MAX_SAMPLES:,} samples (10 minutes at 100 Hz)."
        )

    timestamps = numeric["ACC_TIME"].to_numpy(dtype=np.int64)
    intervals = np.diff(timestamps).astype(np.float64)
    if np.any(intervals <= 0):
        raise DemoInputError("ACC_TIME must contain strictly increasing millisecond timestamps.")

    median_interval = float(np.median(intervals))
    low_ms, high_ms = EXPECTED_SAMPLE_INTERVAL_MS
    if not low_ms <= median_interval <= high_ms:
        raise DemoInputError(
            "The model expects approximately 100 Hz input "
            f"(median interval 8–12 ms); this file has {median_interval:.2f} ms."
        )

    imu = numeric[REQUIRED_COLUMNS[1:]].to_numpy(dtype=np.float32)
    if not np.isfinite(imu).all():
        raise DemoInputError("The six IMU channels must contain finite numeric values.")

    if apply_filter:
        from imu_activity_pipeline.sensor_data_processing import butterworth_filter

        imu = butterworth_filter(imu)

    data = np.column_stack((timestamps, imu))
    duration_sec = float((timestamps[-1] - timestamps[0]) / 1000.0)
    sample_rate_hz = float(1000.0 / median_interval)
    user_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "recording"
    return Recording(
        data=data,
        user_id=user_id,
        sample_rate_hz=sample_rate_hz,
        duration_sec=duration_sec,
    )


@lru_cache(maxsize=1)
def _load_model_bundle():
    """Prepare and load the public multi-scale checkpoints once per process."""

    import torch

    thread_count = max(1, min(int(os.getenv("TORCH_NUM_THREADS", "2")), 4))
    torch.set_num_threads(thread_count)
    from imu_activity_pipeline.inference import load_ensemble_models

    return load_ensemble_models()


# ZeroGPU captures module-scope CUDA model placement during Space startup. Keep
# local and ordinary CPU imports lazy, but eagerly register the tracked models
# when the platform explicitly enables its ZeroGPU runtime.
if os.getenv("SPACES_ZERO_GPU", "").lower() in {"1", "true", "yes"}:
    _load_model_bundle()


def clear_model_cache() -> None:
    """Clear the lazy model cache; useful for tests and controlled restarts."""

    _load_model_bundle.cache_clear()


def _postprocess(
    data: np.ndarray,
    timestamps: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_duration_sec: float,
    confidence_min: float,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Apply the repository's temporal record layer without a second model pass."""

    from imu_activity_pipeline.inference import (
        _select_top_k,
        extract_segments,
        filter_short_segments,
        merge_same_class_segments,
        refine_boundaries,
        resolve_overlaps,
        smooth_predictions,
        viterbi_decode,
    )

    smoothed = smooth_predictions(probabilities, timestamps)
    decoded = viterbi_decode(smoothed)
    segments = extract_segments(decoded, timestamps, smoothed)
    segments = merge_same_class_segments(segments)
    segments = refine_boundaries(segments, data, data[:, 0])
    segments = resolve_overlaps(segments)
    segments = filter_short_segments(segments, min_duration_sec=min_duration_sec)
    if top_k > 0 and len(segments) > top_k:
        segments = _select_top_k(segments, k=top_k)
    if confidence_min > 0:
        segments = [segment for segment in segments if segment["confidence"] >= confidence_min]
    return smoothed, decoded, segments


def run_pipeline(
    upload: str | Path | Any,
    *,
    fusion_mode: str = "local_boundary",
    min_duration_sec: float = 5.0,
    confidence_min: float = 0.30,
    top_k: int = 5,
) -> DemoResult:
    """Run one bounded, serialized inference request through the real pipeline."""

    if fusion_mode not in FUSION_MODES:
        raise DemoInputError(f"Unsupported fusion mode: {fusion_mode}")
    if not 1.0 <= float(min_duration_sec) <= 180.0:
        raise DemoInputError("Minimum duration must be between 1 and 180 seconds.")
    if not 0.0 <= float(confidence_min) <= 1.0:
        raise DemoInputError("Confidence threshold must be between 0 and 1.")
    if not 0 <= int(top_k) <= 10:
        raise DemoInputError("Top-K must be between 0 and 10; use 0 to disable it.")

    recording = load_imu_recording(upload)

    with _INFERENCE_LOCK:
        model_groups, device = _load_model_bundle()
        from imu_activity_pipeline.inference import predict_multiscale_ensemble

        timestamps, probabilities = predict_multiscale_ensemble(
            recording.data,
            model_groups,
            device,
            fusion_mode=fusion_mode,
        )
        if len(timestamps) == 0:
            raise DemoInputError("The recording did not produce any model windows.")
        smoothed, decoded, segments = _postprocess(
            recording.data,
            timestamps,
            probabilities,
            min_duration_sec=float(min_duration_sec),
            confidence_min=float(confidence_min),
            top_k=int(top_k),
        )

    return DemoResult(
        recording=recording,
        timestamps=timestamps,
        probabilities=smoothed,
        decoded_path=decoded,
        segments=segments,
        device=str(device),
        model_scales=tuple(model_groups.keys()),
    )


def segments_dataframe(result: DemoResult) -> pd.DataFrame:
    """Convert segment dictionaries into a compact bilingual table."""

    origin_ms = float(result.recording.data[0, 0])
    rows = []
    for segment in result.segments:
        chinese = str(segment["class_name"])
        rows.append(
            {
                "Activity": ACTIVITY_EN.get(chinese, chinese),
                "活动": chinese,
                "Start (s)": round((float(segment["start_ts"]) - origin_ms) / 1000.0, 2),
                "End (s)": round((float(segment["end_ts"]) - origin_ms) / 1000.0, 2),
                "Duration (s)": round(float(segment["duration"]), 2),
                "Confidence": round(float(segment["confidence"]), 4),
            }
        )
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)


def export_segments(result: DemoResult) -> str:
    """Write a downloadable CSV containing absolute and relative boundaries."""

    origin_ms = int(result.recording.data[0, 0])
    rows = []
    for segment in result.segments:
        start_ms = int(segment["start_ts"])
        end_ms = int(segment["end_ts"])
        chinese = str(segment["class_name"])
        rows.append(
            {
                "user_id": result.recording.user_id,
                "category": chinese,
                "activity_en": ACTIVITY_EN.get(chinese, chinese),
                "start": start_ms,
                "end": end_ms,
                "start_relative_sec": round((start_ms - origin_ms) / 1000.0, 3),
                "end_relative_sec": round((end_ms - origin_ms) / 1000.0, 3),
                "duration_sec": round(float(segment["duration"]), 3),
                "confidence": round(float(segment["confidence"]), 6),
            }
        )

    export_dir = Path(tempfile.mkdtemp(prefix="imu-space-demo-"))
    export_path = export_dir / f"{result.recording.user_id}_segments.csv"
    columns = [
        "user_id",
        "category",
        "activity_en",
        "start",
        "end",
        "start_relative_sec",
        "end_relative_sec",
        "duration_sec",
        "confidence",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(export_path, index=False, encoding="utf-8-sig")
    return str(export_path)


def make_signal_figure(recording: Recording):
    """Render a bounded-resolution preview of the six uploaded channels."""

    data = recording.data
    step = max(1, int(np.ceil(len(data) / 4_000)))
    view = data[::step]
    time_sec = (view[:, 0] - data[0, 0]) / 1000.0

    figure, axes = plt.subplots(2, 1, figsize=(10.8, 5.8), sharex=True)
    colors = ("#4f46e5", "#7c3aed", "#0f9f8f")
    for channel, color, label in zip(range(1, 4), colors, (r"$a_x$", r"$a_y$", r"$a_z$")):
        axes[0].plot(time_sec, view[:, channel], color=color, linewidth=0.9, label=label)
    for channel, color, label in zip(
        range(4, 7), colors, (r"$\omega_x$", r"$\omega_y$", r"$\omega_z$")
    ):
        axes[1].plot(time_sec, view[:, channel], color=color, linewidth=0.9, label=label)

    axes[0].set_ylabel("Acceleration")
    axes[1].set_ylabel("Angular velocity")
    axes[1].set_xlabel("Time from recording start (s)")
    for axis in axes:
        axis.grid(color="#dbe4f0", linewidth=0.7, alpha=0.8)
        axis.legend(loc="upper right", ncols=3, frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Motion recorded by the six IMU channels", fontsize=13, fontweight="bold")
    figure.tight_layout()
    plt.close(figure)
    return figure


def make_timeline_figure(result: DemoResult):
    """Render smoothed class probabilities and the decoded activity timeline."""

    origin_ms = float(result.recording.data[0, 0])
    time_sec = (result.timestamps.astype(np.float64) - origin_ms) / 1000.0
    palette = ("#64748b", "#4f46e5", "#7c3aed", "#0f9f8f", "#f59e0b", "#ef4444")

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.8, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.25]},
    )
    for index, (label, color) in enumerate(zip(CLASS_LABELS, palette)):
        axes[0].plot(
            time_sec,
            result.probabilities[:, index],
            label=label,
            color=color,
            linewidth=1.5 if index else 1.0,
            alpha=0.95 if index else 0.75,
        )
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Activity likelihood")
    axes[0].grid(color="#dbe4f0", linewidth=0.7, alpha=0.8)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncols=3, frameon=False, fontsize=8)

    axes[1].step(time_sec, result.decoded_path, where="mid", color="#312e81", linewidth=1.8)
    axes[1].fill_between(time_sec, 0, result.decoded_path, step="mid", color="#c7d2fe", alpha=0.55)
    axes[1].set_yticks(range(len(CLASS_LABELS)), labels=CLASS_LABELS, fontsize=8)
    axes[1].set_ylim(-0.35, len(CLASS_LABELS) - 0.65)
    axes[1].set_xlabel("Time from recording start (s)")
    axes[1].set_ylabel("Final activity")
    axes[1].grid(axis="x", color="#dbe4f0", linewidth=0.7, alpha=0.8)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Activity likelihood and final timeline", fontsize=13, fontweight="bold")
    figure.tight_layout()
    plt.close(figure)
    return figure


def status_markdown(result: DemoResult, fusion_mode: str) -> str:
    """Create a concise, non-sensitive inference summary for the UI."""

    segment_count = len(result.segments)
    segment_text = "segment" if segment_count == 1 else "segments"
    scales = " / ".join(result.model_scales)
    fusion_label = FUSION_MODE_LABELS.get(fusion_mode, fusion_mode)
    return (
        "### Activity records ready / 活动记录已生成\n"
        f"- **Input / 输入：** `{result.recording.user_id}` · "
        f"{len(result.recording.data):,} samples · {result.recording.duration_sec:.1f} s · "
        f"{result.recording.sample_rate_hz:.1f} Hz\n"
        f"- **Models / 模型：** {scales} windows · {fusion_label} · {result.device.upper()}\n"
        f"- **Result / 结果：** {len(result.timestamps):,} timeline points · "
        f"**{segment_count} {segment_text} / {segment_count} 条活动记录**"
    )
