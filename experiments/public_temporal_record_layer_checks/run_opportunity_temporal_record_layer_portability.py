"""OPPORTUNITY locomotion feature-based TRL portability experiment.

Purpose:
    Parses OPPORTUNITY sensor files, selects body-motion channels, builds window
    features, trains a lightweight classifier, and evaluates decoded locomotion
    segments.
Inputs:
    Expects a local OPPORTUNITY dataset directory with subject data files.
Outputs:
    Prints and saves portability metrics for the shared public-corpus summary.
"""
from __future__ import annotations


import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_record_layer_common import SubjectSequence, run_portability_experiment


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "public_external" / "opportunity"
RESULT_DIR = SCRIPT_DIR / "results"

FS = 30.0
WINDOW_S = 2.0
STEP_S = 1.0

# Zero-based positions in the source .dat files.
TIMESTAMP_COL = 0
LOCOMOTION_LABEL_COL = 243
FIRST_BODY_COL = 1
LAST_BODY_COL_EXCLUSIVE = 134
BACKGROUND_LABEL = 0

RAW_LOCOMOTION_LABEL_NAMES = {
    0: "Null",
    1: "Stand",
    2: "Walk",
    4: "Sit",
    5: "Lie",
}


def find_dataset_dir(data_dir: Path) -> Path:
    candidates = [data_dir / "OpportunityUCIDataset" / "dataset"]
    candidates.extend(path for path in data_dir.rglob("dataset") if path.is_dir())
    for candidate in candidates:
        if (candidate / "column_names.txt").exists() and list(candidate.glob("S*-ADL*.dat")):
            return candidate
    raise FileNotFoundError(
        "Missing OPPORTUNITY data. Expected OpportunityUCIDataset/dataset/S*-ADL*.dat "
        "under data/public_external/opportunity/."
    )


def parse_column_names(dataset_dir: Path) -> dict[int, str]:
    column_file = dataset_dir / "column_names.txt"
    pattern = re.compile(r"Column:\s+(\d+)\s+(.+)")
    names: dict[int, str] = {}
    for line in column_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            names[int(match.group(1))] = match.group(2).strip()
    return names


def is_body_motion_column(one_based_col: int, name: str) -> bool:
    if not (FIRST_BODY_COL + 1 <= one_based_col <= LAST_BODY_COL_EXCLUSIVE):
        return False
    if name.startswith("Accelerometer "):
        return "accX" in name or "accY" in name or "accZ" in name
    if not name.startswith("InertialMeasurementUnit "):
        return False
    motion_tokens = (
        " accX",
        " accY",
        " accZ",
        " gyroX",
        " gyroY",
        " gyroZ",
        " Nav_Ax",
        " Nav_Ay",
        " Nav_Az",
        " Body_Ax",
        " Body_Ay",
        " Body_Az",
        " AngVelBodyFrameX",
        " AngVelBodyFrameY",
        " AngVelBodyFrameZ",
        " AngVelNavFrameX",
        " AngVelNavFrameY",
        " AngVelNavFrameZ",
    )
    return any(token in name for token in motion_tokens)


def discover_signal_columns(dataset_dir: Path) -> list[int]:
    names = parse_column_names(dataset_dir)
    cols = [
        one_based_col - 1
        for one_based_col, name in sorted(names.items())
        if is_body_motion_column(one_based_col, name)
    ]
    if not cols:
        cols = list(range(FIRST_BODY_COL, LAST_BODY_COL_EXCLUSIVE))
    return cols


def interpolate_signal(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32, copy=True)
    row_idx = np.arange(signal.shape[0], dtype=np.float32)
    for col in range(signal.shape[1]):
        values = signal[:, col]
        finite = np.isfinite(values)
        if finite.all():
            continue
        if not finite.any():
            signal[:, col] = 0.0
            continue
        signal[:, col] = np.interp(row_idx, row_idx[finite], values[finite]).astype(np.float32)
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)


def parse_subject_id(path: Path) -> str:
    return path.stem.split("-", 1)[0]


def load_opportunity_sequences(data_dir: Path) -> tuple[dict[str, SubjectSequence], dict[int, str], dict[str, list[str]]]:
    dataset_dir = find_dataset_dir(data_dir)
    signal_cols = discover_signal_columns(dataset_dir)
    usecols = [TIMESTAMP_COL, *signal_cols, LOCOMOTION_LABEL_COL]
    files = sorted(dataset_dir.glob("S*-ADL*.dat"))
    if not files:
        raise FileNotFoundError(f"No OPPORTUNITY ADL files found in {dataset_dir}")

    raw_sequences: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    used_raw_labels: set[int] = set()
    for path in files:
        data = np.loadtxt(path, dtype=np.float32, usecols=usecols)
        timestamps_s = (data[:, 0].astype(np.float64) - float(data[0, 0])) / 1000.0
        signal = interpolate_signal(data[:, 1:-1])
        raw_labels = np.nan_to_num(data[:, -1], nan=BACKGROUND_LABEL).astype(np.int32)
        used_raw_labels.update(int(label) for label in np.unique(raw_labels))
        key = path.stem
        if len(signal) >= int(WINDOW_S * FS):
            raw_sequences[key] = (signal, raw_labels, timestamps_s)

    label_order = [BACKGROUND_LABEL] + sorted(label for label in used_raw_labels if label != BACKGROUND_LABEL)
    label_to_id = {raw_label: idx for idx, raw_label in enumerate(label_order)}
    label_names = {
        idx: RAW_LOCOMOTION_LABEL_NAMES.get(raw_label, f"locomotion {raw_label}")
        for raw_label, idx in label_to_id.items()
    }

    sequences: dict[str, SubjectSequence] = {}
    for key, (signal, raw_labels, timestamps_s) in raw_sequences.items():
        labels = np.asarray([label_to_id[int(label)] for label in raw_labels], dtype=np.int32)
        sequences[key] = SubjectSequence(key, signal, labels, timestamps_s)

    split = {
        "train_subjects": sorted(key for key in sequences if parse_subject_id(Path(key)) in {"S1", "S2"}),
        "dev_subjects": sorted(key for key in sequences if parse_subject_id(Path(key)) == "S3"),
        "test_subjects": sorted(key for key in sequences if parse_subject_id(Path(key)) == "S4"),
    }
    if not all(split.values()):
        raise ValueError(f"Unexpected OPPORTUNITY split from {list(sequences)}: {split}")
    return sequences, label_names, split


def main() -> None:
    sequences, label_names, split = load_opportunity_sequences(DATA_DIR)
    caption = (
        "\\parbox{\\columnwidth}{\\centering\\scshape Lightweight OPPORTUNITY locomotion "
        "portability check. A linear window classifier is trained on ADL runs from subjects "
        "S1--S2, TRL constants are selected on S3, and S4 is reported. Inputs use body-worn "
        "motion sensor columns with 2 s windows and 1 s step. Label 0 is treated as "
        "background and excluded from record matching. This is a temporal-interface sanity "
        "check, not a public leaderboard comparison.}"
    )
    summary = run_portability_experiment(
        dataset_name="OPPORTUNITY-locomotion",
        source="UCI OPPORTUNITY Activity Recognition locomotion ADL files",
        sequences=sequences,
        split=split,
        label_names=label_names,
        window_s=WINDOW_S,
        step_s=STEP_S,
        fs_hz=FS,
        result_dir=RESULT_DIR,
        prefix="opportunity",
        caption=caption,
        ignore_labels={BACKGROUND_LABEL},
    )
    print(json.dumps(summary["selected_trl_params"], indent=2))
    print(
        pd.DataFrame(
            [{"setting": key, **value} for key, value in summary["test_metrics"].items()]
        ).to_string(index=False)
    )
    print(f"Wrote {RESULT_DIR / 'opportunity_trl_summary.json'}")


if __name__ == "__main__":
    main()
