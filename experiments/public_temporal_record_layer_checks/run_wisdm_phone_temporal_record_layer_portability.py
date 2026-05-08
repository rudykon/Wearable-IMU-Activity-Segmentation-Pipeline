"""WISDM-phone feature-based TRL portability experiment.

Purpose:
    Parses WISDM phone accelerometer records, builds subject sequences and window
    features, trains a lightweight classifier, and evaluates decoded activity
    segments.
Inputs:
    Expects a local WISDM phone dataset file or directory.
Outputs:
    Prints metrics and writes a result row for the public portability summary.
"""
from __future__ import annotations


import json
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_record_layer_common import SubjectSequence, fixed_subject_split, run_portability_experiment


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "public_external" / "wisdm_phone"
RESULT_DIR = SCRIPT_DIR / "results"

FS = 20.0
WINDOW_S = 10.0
STEP_S = 1.0


def parse_wisdm_line(line: str) -> tuple[str, str, float, float, float] | None:
    line = line.strip().rstrip(";")
    if not line or "," not in line:
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 6:
        return None
    try:
        user = parts[0]
        activity = parts[1]
        x = float(parts[3])
        y = float(parts[4])
        z = float(parts[5])
    except ValueError:
        return None
    if not user or not activity:
        return None
    return user, activity, x, y, z


def discover_wisdm_files(data_dir: Path) -> list[Path]:
    phone_accel_dir = data_dir / "raw" / "phone" / "accel"
    if phone_accel_dir.exists():
        return sorted(phone_accel_dir.glob("*.txt"))

    preferred = [
        data_dir / "WISDM_ar_v1.1_raw.txt",
        data_dir / "WISDM_ar_latest" / "WISDM_ar_v1.1" / "WISDM_ar_v1.1_raw.txt",
    ]
    found = [path for path in preferred if path.exists()]
    if found:
        return found

    return sorted(
        path
        for path in data_dir.rglob("*.txt")
        if "readme" not in path.name.lower() and "about" not in path.name.lower()
    )


def load_wisdm_phone_sequences(data_dir: Path) -> tuple[dict[str, SubjectSequence], dict[int, str]]:
    files = discover_wisdm_files(data_dir)
    if not files:
        raise FileNotFoundError(
            "Missing WISDM-phone data. Place WISDM_ar_v1.1_raw.txt or "
            "raw/phone/accel/*.txt under data/public_external/wisdm_phone/."
        )

    rows = []
    for path in files:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parsed = parse_wisdm_line(line)
                if parsed is None:
                    continue
                rows.append(parsed)

    if not rows:
        raise ValueError(f"No WISDM-format sensor rows parsed from {data_dir}")

    df = pd.DataFrame(rows, columns=["subject", "activity", "x", "y", "z"])
    df["subject"] = df["subject"].astype(str)
    df["activity"] = df["activity"].astype(str)

    activity_names = sorted(df["activity"].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(activity_names)}
    label_names = {idx: label for label, idx in label_to_id.items()}
    df["label"] = df["activity"].map(label_to_id).astype(np.int32)

    sequences: dict[str, SubjectSequence] = {}
    for subject, group in df.groupby("subject", sort=True):
        signal = group[["x", "y", "z"]].to_numpy(dtype=np.float32)
        labels = group["label"].to_numpy(dtype=np.int32)
        timestamps_s = np.arange(len(group), dtype=np.float64) / FS
        if len(signal) >= int(WINDOW_S * FS):
            sequences[str(subject)] = SubjectSequence(str(subject), signal, labels, timestamps_s)

    if len(sequences) < 5:
        raise ValueError(f"Need at least 5 WISDM subjects with enough data, got {len(sequences)}")
    return sequences, label_names


def main() -> None:
    sequences, label_names = load_wisdm_phone_sequences(DATA_DIR)
    split = fixed_subject_split(sequences.keys(), train_frac=0.6, dev_frac=0.2)
    caption = (
        "\\parbox{\\columnwidth}{\\centering\\scshape Lightweight WISDM-phone portability check. "
        "A simple linear classifier is trained on the first 60\\% of subject IDs, TRL constants are "
        "selected on the next 20\\%, and held-out subjects form the final split. Phone accelerometer "
        "streams use 10 s windows with 1 s step. This is a temporal-interface sanity check, not a "
        "public leaderboard comparison.}"
    )
    summary = run_portability_experiment(
        dataset_name="WISDM-phone",
        source="WISDM phone accelerometer raw time-series",
        sequences=sequences,
        split=split,
        label_names=label_names,
        window_s=WINDOW_S,
        step_s=STEP_S,
        fs_hz=FS,
        result_dir=RESULT_DIR,
        prefix="wisdm_phone",
        caption=caption,
    )
    print(json.dumps(summary["selected_trl_params"], indent=2))
    print(
        pd.DataFrame(
            [{"setting": key, **value} for key, value in summary["test_metrics"].items()]
        ).to_string(index=False)
    )
    print(f"Wrote {RESULT_DIR / 'wisdm_phone_trl_summary.json'}")


if __name__ == "__main__":
    main()
