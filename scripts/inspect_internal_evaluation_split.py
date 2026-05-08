"""Internal-evaluation split inspection utility.

Purpose:
    Checks which user signal files are available under the unified data layout
    and reports whether they match the expected evaluation split.
Inputs:
    Reads `data/signals/internal_eval/` and related metadata/annotation files.
Outputs:
    Prints a concise availability report for debugging data packaging issues.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os

from imu_activity_pipeline.config import *
from imu_activity_pipeline.sensor_data_processing import load_sensor_data, load_gold_labels, create_windows


def get_valid_users(data_dir=INTERNAL_EVAL_DATA_DIR, gold_file=INTERNAL_EVAL_GOLD_FILE):
    """Get valid users from one unified split directory."""
    gold_labels = load_gold_labels(gold_file)
    users = sorted([f.replace('.txt', '') for f in os.listdir(data_dir) if f.endswith('.txt')])
    gold_users = set(gold_labels['user_id'].unique())

    valid_users = []
    for user_id in users:
        if user_id not in gold_users:
            continue
        file_path = os.path.join(data_dir, f"{user_id}.txt")
        try:
            with open(file_path, 'rb') as f:
                header_bytes = f.read(100)
            header_text = header_bytes.decode('utf-8')
            if 'ACC_TIME' not in header_text:
                continue
        except:
            continue
        data = load_sensor_data(file_path, apply_filter=False)  # no filter needed just to check
        if data is None or len(data) < WINDOW_SIZE:
            continue
        timestamps, windows = create_windows(data, WINDOW_SIZE, WINDOW_STEP)
        if len(windows) == 0:
            continue
        valid_users.append(user_id)

    return valid_users, gold_labels


def main():
    print("=" * 60)
    print("UNIFIED INTERNAL EVALUATION SPLIT CHECK")
    print("=" * 60)
    print(f"Signal dir: {INTERNAL_EVAL_DATA_DIR}")
    print(f"Gold file:  {INTERNAL_EVAL_GOLD_FILE}")

    valid_users, gold_labels = get_valid_users()
    valid_gold = gold_labels[gold_labels['user_id'].isin(valid_users)].copy()
    valid_gold = valid_gold.sort_values(['user_id', 'start']).reset_index(drop=True)

    print(f"Valid internal-eval users: {len(valid_users)}")
    print(f"  {valid_users}")
    print(f"\nAnnotations: {len(valid_gold)} segments from {valid_gold['user_id'].nunique()} users")

    # Summary
    print(f"\n{'='*60}")
    print("CATEGORY DISTRIBUTION IN INTERNAL EVAL GOLD:")
    for cat in ACTIVITIES:
        count = len(valid_gold[valid_gold['category'] == cat])
        print(f"  {cat}: {count} segments")
    print(f"  Total: {len(valid_gold)} segments")
    print(f"{'='*60}")
    print("DONE! Run internal-eval inference with:")
    print(f"  python -m imu_activity_pipeline.inference --data_dir {INTERNAL_EVAL_DATA_DIR} --output predictions_internal_eval.xlsx")
    print("  python evaluate.py --split internal_eval --predictions predictions_internal_eval.xlsx")


if __name__ == '__main__':
    main()
