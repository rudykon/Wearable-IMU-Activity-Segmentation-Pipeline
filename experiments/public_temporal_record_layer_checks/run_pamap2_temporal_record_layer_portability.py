"""PAMAP2 feature-based TRL portability experiment.

Purpose:
    Loads PAMAP2 protocol files, interpolates sensor channels, builds fixed
    windows, trains a lightweight baseline, and evaluates decoded activity
    segments.
Inputs:
    Expects a local PAMAP2 dataset copy in the configured public-data locations.
Outputs:
    Prints metrics and writes a row for the public portability summary.
"""
from __future__ import annotations


import json
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_record_layer_common import SubjectSequence, run_portability_experiment


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "public_external" / "pamap2"
RESULT_DIR = SCRIPT_DIR / "results"

FS = 100.0
WINDOW_S = 5.0
STEP_S = 1.0

PAMAP2_LABEL_NAMES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic walking",
    9: "watching TV",
    10: "computer work",
    11: "car driving",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    18: "folding laundry",
    19: "house cleaning",
    20: "playing soccer",
    24: "rope jumping",
}

# Zero-based columns: timestamp, activityID, heart-rate, then 3 IMU blocks.
# Use accelerometer-16g and gyroscope channels from hand/chest/ankle.
PAMAP2_SIGNAL_COLS = [
    4,
    5,
    6,
    10,
    11,
    12,
    21,
    22,
    23,
    27,
    28,
    29,
    38,
    39,
    40,
    44,
    45,
    46,
]


def discover_pamap2_protocol_files(data_dir: Path) -> list[Path]:
    protocol_dirs = [path for path in data_dir.rglob("Protocol") if path.is_dir()]
    if protocol_dirs:
        files = []
        for protocol_dir in protocol_dirs:
            files.extend(protocol_dir.glob("subject*.dat"))
        return sorted(files)
    return sorted(data_dir.rglob("subject*.dat"))


def interpolate_signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.interpolate(method="linear", limit_direction="both", axis=0)
    frame = frame.fillna(frame.median(numeric_only=True)).fillna(0.0)
    return frame


def load_pamap2_sequences(data_dir: Path) -> tuple[dict[str, SubjectSequence], dict[int, str], dict[str, list[str]]]:
    files = discover_pamap2_protocol_files(data_dir)
    if not files:
        raise FileNotFoundError(
            "Missing PAMAP2 protocol files. Place PAMAP2_Dataset/Protocol/subject*.dat "
            "under data/public_external/pamap2/."
        )

    sequences: dict[str, SubjectSequence] = {}
    used_activity_ids: set[int] = set()
    for path in files:
        subject = path.stem.replace("subject", "")
        df = pd.read_csv(path, sep=r"\s+", header=None, na_values="NaN", engine="python")
        if df.shape[1] <= max(PAMAP2_SIGNAL_COLS):
            raise ValueError(f"Unexpected PAMAP2 column count in {path}: {df.shape[1]}")

        df = df[df.iloc[:, 1].fillna(0).astype(int) > 0].copy()
        if df.empty:
            continue
        activity_ids = df.iloc[:, 1].astype(int).to_numpy()
        used_activity_ids.update(int(value) for value in np.unique(activity_ids))

        signal_df = interpolate_signal_frame(df.iloc[:, PAMAP2_SIGNAL_COLS].astype(float))
        signal = signal_df.to_numpy(dtype=np.float32)
        timestamps = df.iloc[:, 0].astype(float).to_numpy()
        timestamps = timestamps - float(timestamps[0])

        if len(signal) >= int(WINDOW_S * FS):
            sequences[subject] = SubjectSequence(subject, signal, activity_ids.astype(np.int32), timestamps)

    if len(sequences) < 5:
        raise ValueError(f"Need at least 5 PAMAP2 subjects with enough data, got {len(sequences)}")

    activity_order = sorted(used_activity_ids)
    label_to_id = {label: idx for idx, label in enumerate(activity_order)}
    label_names = {
        idx: PAMAP2_LABEL_NAMES.get(label, f"activity {label}")
        for label, idx in label_to_id.items()
    }

    remapped = {}
    for subject, sequence in sequences.items():
        remapped_labels = np.asarray([label_to_id[int(label)] for label in sequence.labels], dtype=np.int32)
        remapped[subject] = SubjectSequence(sequence.subject, sequence.signal, remapped_labels, sequence.timestamps_s)

    subjects = sorted(remapped.keys())
    split = {
        "train_subjects": subjects[:5],
        "dev_subjects": subjects[5:7],
        "test_subjects": subjects[7:],
    }
    if not split["test_subjects"]:
        split = {
            "train_subjects": subjects[: max(1, len(subjects) - 3)],
            "dev_subjects": subjects[max(1, len(subjects) - 3) : max(2, len(subjects) - 1)],
            "test_subjects": subjects[max(2, len(subjects) - 1) :],
        }
    return remapped, label_names, split


def main() -> None:
    sequences, label_names, split = load_pamap2_sequences(DATA_DIR)
    caption = (
        "\\parbox{\\columnwidth}{\\centering\\scshape Lightweight PAMAP2 portability check. "
        "A simple linear classifier is trained on protocol subjects 101--105 when available, "
        "TRL constants are selected on the next held-out subjects, and the final subjects are "
        "reported. The input uses accelerometer and gyroscope channels from hand, chest, and "
        "ankle IMUs with 5 s windows and 1 s step. This is a temporal-interface sanity check, "
        "not a public leaderboard comparison.}"
    )
    summary = run_portability_experiment(
        dataset_name="PAMAP2",
        source="UCI PAMAP2 Physical Activity Monitoring protocol files",
        sequences=sequences,
        split=split,
        label_names=label_names,
        window_s=WINDOW_S,
        step_s=STEP_S,
        fs_hz=FS,
        result_dir=RESULT_DIR,
        prefix="pamap2",
        caption=caption,
    )
    print(json.dumps(summary["selected_trl_params"], indent=2))
    print(
        pd.DataFrame(
            [{"setting": key, **value} for key, value in summary["test_metrics"].items()]
        ).to_string(index=False)
    )
    print(f"Wrote {RESULT_DIR / 'pamap2_trl_summary.json'}")


if __name__ == "__main__":
    main()
