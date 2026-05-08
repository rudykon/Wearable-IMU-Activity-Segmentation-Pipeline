"""PPG quality-analysis script for supplementary signal diagnostics.

Purpose:
    Measures PPG availability and quality proxies alongside IMU segments so the
    repository can document why the main pipeline focuses on accelerometer and
    gyroscope streams.
Inputs:
    Reads raw signal files and annotation tables from the configured data layout.
Outputs:
    Writes summary tables or logs under `experiments/results/`.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from imu_activity_pipeline.config import INTERNAL_EVAL_DATA_DIR, INTERNAL_EVAL_GOLD_FILE  # noqa: E402
from imu_activity_pipeline.sensor_data_processing import load_gold_labels  # noqa: E402

OUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    gold = load_gold_labels(INTERNAL_EVAL_GOLD_FILE)
    gold["user_id"] = gold["user_id"].astype(str)

    file_map = {p.stem: p for p in Path(INTERNAL_EVAL_DATA_DIR).glob("*.txt")}
    ppg_cols = [f"PPG{i}" for i in range(1, 25)]
    active_cols = [f"PPG{i}" for i in range(1, 21)]

    nonzero_by_channel = {c: False for c in ppg_cols}
    update_diffs_ms = []
    segment_features = []

    for uid, group in gold.groupby("user_id"):
        file_path = file_map.get(uid)
        if file_path is None:
            continue

        df = pd.read_csv(file_path, sep="\t", low_memory=False)
        for col in ["ACC_TIME", "PPG_TIME", *ppg_cols]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["ACC_TIME"]).reset_index(drop=True)

        ppg_time = df["PPG_TIME"].dropna()
        ppg_time = ppg_time[ppg_time > 0]
        if len(ppg_time) > 1:
            diffs = ppg_time.diff().dropna()
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                update_diffs_ms.append(float(diffs.median()))

        for col in ppg_cols:
            if (df[col].fillna(0).to_numpy() != 0).any():
                nonzero_by_channel[col] = True

        for _, row in group.iterrows():
            seg = df[(df["ACC_TIME"] >= row["start"]) & (df["ACC_TIME"] <= row["end"])]
            if len(seg) == 0:
                continue
            feat = seg[active_cols].mean().to_numpy(dtype=float)
            segment_features.append((row["category"], feat))

    x = np.stack([feat for _, feat in segment_features], axis=0)
    y = [cls for cls, _ in segment_features]
    overall = x.mean(axis=0)
    between = 0.0
    within = 0.0
    for cls in sorted(set(y)):
        xc = x[[i for i, label in enumerate(y) if label == cls]]
        mu = xc.mean(axis=0)
        between += len(xc) * float(((mu - overall) ** 2).sum())
        within += float(((xc - mu) ** 2).sum())

    median_update_ms = float(np.median(update_diffs_ms)) if update_diffs_ms else 0.0
    summary = {
        "users": int(gold["user_id"].nunique()),
        "segments": int(len(gold)),
        "dead_channels": [c for c, has_signal in nonzero_by_channel.items() if not has_signal],
        "median_ppg_update_ms": median_update_ms,
        "effective_ppg_rate_hz": float(1000.0 / median_update_ms) if median_update_ms > 0 else 0.0,
        "between_within_variance_ratio": float(between / within) if within > 0 else 0.0,
        "between_within_variance_ratio_pct": float(100.0 * between / within) if within > 0 else 0.0,
    }

    out_path = os.path.join(OUT_DIR, "ppg_quality_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved PPG quality summary to:", out_path)


if __name__ == "__main__":
    main()
