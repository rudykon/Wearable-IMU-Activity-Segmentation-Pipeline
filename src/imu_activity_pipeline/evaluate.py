"""Segment-level evaluation for activity prediction workbooks.

Purpose:
    Compares predicted activity segments against reference annotations using
    same-class one-to-one temporal IoU matching.
Inputs:
    Reads prediction workbooks and reference annotation files in CSV or Excel
    format with `user_id`, `category`, `start`, and `end` columns.
Outputs:
    Prints per-user true positives, false positives, false negatives, and the
    final average segmental F1 score.
"""
import argparse
import os

import numpy as np
import pandas as pd

from .config import BASE_DIR, DEFAULT_EVALUATION_SPLIT, SPLIT_ANNOTATION_FILES, SPLIT_NAMES
from .sensor_data_processing import load_gold_labels

EXPECTED_COLUMNS = ["user_id", "category", "start", "end"]


def calculate_iou(start1, end1, start2, end2):
    """Calculate intersection-over-union for two temporal segments."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = max(0, union_end - union_start)

    if union == 0:
        return 0

    return intersection / union


def load_predictions(prediction_file):
    """Load a prediction workbook with or without a header row."""
    df = pd.read_excel(prediction_file)
    if list(df.columns[:4]) == EXPECTED_COLUMNS:
        return df[EXPECTED_COLUMNS].copy()

    df = pd.read_excel(prediction_file, header=None, names=EXPECTED_COLUMNS)
    if len(df) > 0 and list(df.iloc[0].astype(str)) == EXPECTED_COLUMNS:
        df = df.iloc[1:].reset_index(drop=True)
    return df


def default_prediction_file(split_name: str) -> str:
    """Return the default prediction workbook for one canonical data split."""
    return os.path.join(BASE_DIR, f"predictions_{split_name}.xlsx")


def default_gold_file(split_name: str) -> str:
    """Return the annotation file for one canonical data split."""
    try:
        return SPLIT_ANNOTATION_FILES[split_name]
    except KeyError as exc:
        valid = ", ".join(SPLIT_NAMES)
        raise ValueError(f"Unknown split '{split_name}'. Expected one of: {valid}") from exc


def evaluate_metrics(prediction_file, gold_file, split_name=None):
    """Print per-user and average segmental F1 against the reference labels."""
    label = f" for split '{split_name}'" if split_name else ""
    print(f"\nEvaluating{label}: {prediction_file} against {gold_file}...")

    try:
        df_sub = load_predictions(prediction_file)
        df_gold = load_gold_labels(gold_file)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    df_sub["user_id"] = df_sub["user_id"].astype(str)
    df_gold["user_id"] = df_gold["user_id"].astype(str)

    split_users = df_gold["user_id"].unique()
    user_f1_scores = []

    for user_id in split_users:
        g_segments = df_gold[df_gold["user_id"] == user_id]
        p_segments = df_sub[df_sub["user_id"] == user_id]

        gold_segments = [
            {
                "start": row["start"],
                "end": row["end"],
                "category": row["category"],
            }
            for _, row in g_segments.iterrows()
        ]
        pred_segments = [
            {
                "start": row["start"],
                "end": row["end"],
                "category": row["category"],
            }
            for _, row in p_segments.iterrows()
        ]

        # Greedily accept same-class segment pairs in descending IoU order.
        matches = []
        for pred_idx, pred in enumerate(pred_segments):
            for gold_idx, gold in enumerate(gold_segments):
                if pred["category"] == gold["category"]:
                    iou = calculate_iou(pred["start"], pred["end"], gold["start"], gold["end"])
                    if iou > 0.5:
                        matches.append({"pred_idx": pred_idx, "gold_idx": gold_idx, "iou": iou})

        matches.sort(key=lambda x: x["iou"], reverse=True)

        matched_pred_indices = set()
        matched_gold_indices = set()
        true_positive = 0

        for match in matches:
            if match["pred_idx"] not in matched_pred_indices and match["gold_idx"] not in matched_gold_indices:
                true_positive += 1
                matched_pred_indices.add(match["pred_idx"])
                matched_gold_indices.add(match["gold_idx"])

        false_positive = len(pred_segments) - len(matched_pred_indices)
        false_negative = len(gold_segments) - len(matched_gold_indices)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0

        user_f1_scores.append(f1)
        print(
            f"User {user_id}: TP={true_positive}, FP={false_positive}, "
            f"FN={false_negative} -> F1={f1:.4f}"
        )

    avg_f1 = np.mean(user_f1_scores) if user_f1_scores else 0
    print(f"\n{'=' * 30}")
    print(f"Final Average Segmental F1-Score: {avg_f1:.4f}")
    print(f"{'=' * 30}")


def main():
    """Parse CLI arguments and run segment-level evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=SPLIT_NAMES,
        default=DEFAULT_EVALUATION_SPLIT,
        help="Canonical data split to evaluate. Defaults to external_test.",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Prediction workbook path. Defaults to predictions_<split>.xlsx.",
    )
    parser.add_argument(
        "--gold",
        default=None,
        help="Reference annotation file path (.csv or .xlsx). Defaults to the selected split annotation file.",
    )
    args = parser.parse_args()

    prediction_file = args.predictions or default_prediction_file(args.split)
    gold_file = args.gold or default_gold_file(args.split)
    evaluate_metrics(prediction_file, gold_file, split_name=args.split)


if __name__ == "__main__":
    main()
