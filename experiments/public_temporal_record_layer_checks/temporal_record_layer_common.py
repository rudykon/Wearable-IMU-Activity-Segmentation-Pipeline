"""Shared utilities for public-dataset TRL portability experiments.

Purpose:
    Provides dataset containers, subject splits, robust feature extraction,
    temporal decoding, segment metrics, and common reporting helpers used across
    lightweight public-corpus experiments.
Inputs:
    Receives per-subject sensor arrays and labels from dataset-specific loader
    scripts.
Outputs:
    Returns window features, decoded segments, metrics, and table rows consumed
    by public portability runners.
"""
from __future__ import annotations


import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SubjectSequence:
    subject: str
    signal: np.ndarray
    labels: np.ndarray
    timestamps_s: np.ndarray


@dataclass
class SubjectWindows:
    subject: str
    features: np.ndarray
    labels: np.ndarray
    centers_s: np.ndarray
    duration_s: float
    gt_segments: list[dict]


def numeric_sort_key(value: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return (int(digits) if digits else 0, str(value))


def fixed_subject_split(subjects: Iterable[str], train_frac: float = 0.6, dev_frac: float = 0.2) -> dict[str, list[str]]:
    ordered = sorted([str(s) for s in subjects], key=numeric_sort_key)
    if len(ordered) < 5:
        raise ValueError(f"Need at least 5 subjects for train/dev/test split, got {len(ordered)}")
    n_train = max(1, int(round(len(ordered) * train_frac)))
    n_dev = max(1, int(round(len(ordered) * dev_frac)))
    if n_train + n_dev >= len(ordered):
        n_train = max(1, len(ordered) - 2)
        n_dev = 1
    return {
        "train_subjects": ordered[:n_train],
        "dev_subjects": ordered[n_train : n_train + n_dev],
        "test_subjects": ordered[n_train + n_dev :],
    }


def sample_period_s(timestamps_s: np.ndarray, fallback_fs: float) -> float:
    diffs = np.diff(timestamps_s.astype(float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs):
        return float(np.median(diffs))
    return 1.0 / fallback_fs


def labels_to_segments(labels: np.ndarray, timestamps_s: np.ndarray, fallback_fs: float) -> tuple[list[dict], float]:
    if len(labels) == 0:
        return [], 0.0
    dt = sample_period_s(timestamps_s, fallback_fs)
    duration_s = float(timestamps_s[-1] + dt)
    segments: list[dict] = []
    start = 0
    for idx in range(1, len(labels) + 1):
        if idx == len(labels) or labels[idx] != labels[start]:
            segments.append(
                {
                    "label": int(labels[start]),
                    "start": float(timestamps_s[start]),
                    "end": float(timestamps_s[idx - 1] + dt),
                }
            )
            start = idx
    return segments, duration_s


def robust_stats(values: np.ndarray) -> list[float]:
    values = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return [
        float(values.mean()),
        float(values.std()),
        float(values.min()),
        float(values.max()),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
    ]


def robust_stats_windows(windows: np.ndarray) -> np.ndarray:
    """Return per-channel robust stats for a window view.

    `windows` is shaped as (n_windows, n_channels, window_n). The output keeps
    the same feature order as repeated `featurize_window` calls: all six stats
    for channel 0, then all six stats for channel 1, and so on.
    """

    windows = np.nan_to_num(windows.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    stats = np.stack(
        [
            windows.mean(axis=2),
            windows.std(axis=2),
            windows.min(axis=2),
            windows.max(axis=2),
            np.percentile(windows, 25, axis=2),
            np.percentile(windows, 75, axis=2),
        ],
        axis=2,
    )
    return stats.reshape(windows.shape[0], -1).astype(np.float32, copy=False)


def featurize_window(block: np.ndarray) -> np.ndarray:
    block = np.nan_to_num(block.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    features: list[float] = []
    for idx in range(block.shape[1]):
        features.extend(robust_stats(block[:, idx]))

    # Add norm summaries for every consecutive 3-axis sensor group, plus a
    # global norm. This keeps the same feature type across 3-axis phone data and
    # multi-IMU PAMAP2 data without assuming a specific body placement.
    if block.shape[1] >= 3:
        for start in range(0, block.shape[1] - 2, 3):
            features.extend(robust_stats(np.linalg.norm(block[:, start : start + 3], axis=1)))
        features.extend(robust_stats(np.linalg.norm(block, axis=1)))
    return np.asarray(features, dtype=np.float32)


def featurize_windows(signal: np.ndarray, window_n: int, step_n: int) -> np.ndarray:
    signal = np.nan_to_num(signal.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_shape=window_n, axis=0)[::step_n]
    if windows.shape[-1] != window_n:
        windows = np.moveaxis(windows, 1, -1)

    feature_blocks = [robust_stats_windows(windows)]
    if signal.shape[1] >= 3:
        for start in range(0, signal.shape[1] - 2, 3):
            norm = np.linalg.norm(signal[:, start : start + 3], axis=1).astype(np.float32, copy=False)
            norm_windows = np.lib.stride_tricks.sliding_window_view(norm, window_shape=window_n)[::step_n]
            feature_blocks.append(robust_stats_windows(norm_windows[:, None, :]))
        global_norm = np.linalg.norm(signal, axis=1).astype(np.float32, copy=False)
        global_windows = np.lib.stride_tricks.sliding_window_view(global_norm, window_shape=window_n)[::step_n]
        feature_blocks.append(robust_stats_windows(global_windows[:, None, :]))
    return np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)


def majority_label(labels: np.ndarray) -> int:
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def build_subject_windows(
    sequence: SubjectSequence,
    *,
    window_s: float,
    step_s: float,
    fs_hz: float,
) -> SubjectWindows:
    window_n = int(round(window_s * fs_hz))
    step_n = int(round(step_s * fs_hz))
    if window_n <= 0 or step_n <= 0:
        raise ValueError("window_s and step_s must produce positive sample counts")
    if len(sequence.signal) < window_n:
        raise ValueError(f"{sequence.subject} is shorter than one analysis window")

    starts = np.arange(0, len(sequence.signal) - window_n + 1, step_n, dtype=np.int64)
    features = featurize_windows(sequence.signal, window_n, step_n)
    label_windows = np.lib.stride_tricks.sliding_window_view(sequence.labels, window_shape=window_n)[::step_n]
    window_labels = [majority_label(labels) for labels in label_windows]
    centers = (sequence.timestamps_s[starts] + sequence.timestamps_s[starts + window_n - 1]) / 2.0

    gt_segments, duration_s = labels_to_segments(sequence.labels, sequence.timestamps_s, fs_hz)
    return SubjectWindows(
        subject=sequence.subject,
        features=features,
        labels=np.asarray(window_labels, dtype=np.int32),
        centers_s=np.asarray(centers, dtype=np.float64),
        duration_s=duration_s,
        gt_segments=gt_segments,
    )


def _normalize_proba(proba: np.ndarray) -> np.ndarray:
    sums = proba.sum(axis=1, keepdims=True)
    zero = sums.squeeze(axis=1) <= 0
    if np.any(zero):
        proba[zero] = 1.0 / proba.shape[1]
        sums = proba.sum(axis=1, keepdims=True)
    return proba / sums


def moving_average(proba: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return proba.copy()
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(proba, ((pad_left, pad_right), (0, 0)), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    out = np.empty_like(proba, dtype=np.float64)
    for col in range(proba.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return _normalize_proba(out)


def median_filter(proba: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return proba.copy()
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(proba, ((pad_left, pad_right), (0, 0)), mode="edge")
    out = np.empty_like(proba, dtype=np.float64)
    for idx in range(proba.shape[0]):
        out[idx] = np.median(padded[idx : idx + width], axis=0)
    return _normalize_proba(out)


def viterbi_decode(proba: np.ndarray, self_prob: float) -> np.ndarray:
    eps = 1e-12
    n_steps, n_classes = proba.shape
    off_prob = (1.0 - self_prob) / max(n_classes - 1, 1)
    trans = np.full((n_classes, n_classes), off_prob, dtype=np.float64)
    np.fill_diagonal(trans, self_prob)
    log_emit = np.log(np.clip(proba, eps, 1.0))
    log_trans = np.log(np.clip(trans, eps, 1.0))

    score = np.empty((n_steps, n_classes), dtype=np.float64)
    back = np.zeros((n_steps, n_classes), dtype=np.int32)
    score[0] = log_emit[0]
    for t in range(1, n_steps):
        candidate = score[t - 1][:, None] + log_trans
        back[t] = np.argmax(candidate, axis=0)
        score[t] = candidate[back[t], np.arange(n_classes)] + log_emit[t]

    states = np.empty(n_steps, dtype=np.int32)
    states[-1] = int(np.argmax(score[-1]))
    for t in range(n_steps - 2, -1, -1):
        states[t] = int(back[t + 1, states[t + 1]])
    return states


def repair_short_runs(states: np.ndarray, proba: np.ndarray, min_run: int) -> np.ndarray:
    repaired = states.copy()
    for _ in range(2):
        runs = []
        start = 0
        for idx in range(1, len(repaired) + 1):
            if idx == len(repaired) or repaired[idx] != repaired[start]:
                runs.append((start, idx, repaired[start]))
                start = idx
        for run_idx, (start, end, label) in enumerate(runs):
            if end - start >= min_run:
                continue
            left = runs[run_idx - 1][2] if run_idx > 0 else None
            right = runs[run_idx + 1][2] if run_idx + 1 < len(runs) else None
            if left is None and right is None:
                continue
            if left == right and left is not None:
                replacement = left
            else:
                candidates = [c for c in (left, right) if c is not None]
                scores = [float(proba[start:end, c].mean()) for c in candidates]
                replacement = candidates[int(np.argmax(scores))]
            if replacement != label:
                repaired[start:end] = replacement
    return repaired


def decode_trl(proba: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    smoothed = moving_average(proba, int(params["smooth_width"]))
    smoothed = median_filter(smoothed, int(params.get("median_width", 1)))
    decoded = viterbi_decode(smoothed, float(params["self_prob"]))
    return repair_short_runs(decoded, smoothed, int(params["min_run"])), smoothed


def segment_confidence(proba: np.ndarray, start: int, end: int, state: int) -> float:
    return float(proba[start:end, state].mean()) if end > start else 0.0


def states_to_segments(states: np.ndarray, centers_s: np.ndarray, duration_s: float, classes: np.ndarray, proba: np.ndarray) -> list[dict]:
    labels = classes[states]
    segments: list[dict] = []
    start = 0
    for idx in range(1, len(labels) + 1):
        if idx == len(labels) or labels[idx] != labels[start]:
            seg_start = 0.0 if start == 0 else float((centers_s[start - 1] + centers_s[start]) / 2.0)
            seg_end = duration_s if idx == len(labels) else float((centers_s[idx - 1] + centers_s[idx]) / 2.0)
            state = int(states[start])
            segments.append(
                {
                    "label": int(labels[start]),
                    "state": state,
                    "start": seg_start,
                    "end": seg_end,
                    "start_idx": start,
                    "end_idx": idx,
                    "confidence": segment_confidence(proba, start, idx, state),
                }
            )
            start = idx
    return segments


def refresh_confidence(segment: dict, proba: np.ndarray) -> dict:
    refreshed = dict(segment)
    refreshed["confidence"] = segment_confidence(
        proba,
        int(refreshed["start_idx"]),
        int(refreshed["end_idx"]),
        int(refreshed["state"]),
    )
    return refreshed


def merge_same_label_interruptions(segments: list[dict], max_gap_s: float, proba: np.ndarray) -> list[dict]:
    if max_gap_s <= 0 or len(segments) <= 2:
        return segments
    merged: list[dict] = []
    idx = 0
    while idx < len(segments):
        current = dict(segments[idx])
        next_idx = idx + 1
        while True:
            merge_idx = None
            for cand_idx in range(next_idx, len(segments)):
                if segments[cand_idx]["start"] - current["end"] > max_gap_s:
                    break
                if segments[cand_idx]["label"] == current["label"]:
                    merge_idx = cand_idx
                    break
            if merge_idx is None:
                break
            current["end"] = segments[merge_idx]["end"]
            current["end_idx"] = segments[merge_idx]["end_idx"]
            current = refresh_confidence(current, proba)
            next_idx = merge_idx + 1
        merged.append(current)
        idx = next_idx
    return merged


def postprocess_segments(segments: list[dict], proba: np.ndarray, params: dict | None) -> list[dict]:
    if params is None:
        return segments
    processed = merge_same_label_interruptions(
        segments,
        max_gap_s=float(params.get("merge_gap_s", 0.0)),
        proba=proba,
    )
    min_segment_s = float(params.get("min_segment_s", 0.0))
    if min_segment_s > 0:
        processed = [seg for seg in processed if seg["end"] - seg["start"] >= min_segment_s]
    conf_threshold = float(params.get("conf_threshold", 0.0))
    if conf_threshold > 0:
        processed = [seg for seg in processed if seg.get("confidence", 0.0) >= conf_threshold]
    return processed


def interval_iou(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def filter_eval_segments(
    segments: list[dict],
    min_segment_s: float,
    ignore_labels: set[int] | None = None,
) -> list[dict]:
    ignored = ignore_labels or set()
    return [
        seg
        for seg in segments
        if seg["end"] - seg["start"] >= min_segment_s and int(seg["label"]) not in ignored
    ]


def match_segments(
    gt_segments: list[dict],
    pred_segments: list[dict],
    min_segment_s: float,
    threshold: float = 0.5,
    ignore_labels: set[int] | None = None,
) -> dict:
    gt_segments = filter_eval_segments(gt_segments, min_segment_s, ignore_labels)
    pred_segments = filter_eval_segments(pred_segments, min_segment_s, ignore_labels)
    candidates = []
    for gi, gt in enumerate(gt_segments):
        for pi, pred in enumerate(pred_segments):
            if gt["label"] != pred["label"]:
                continue
            iou = interval_iou(gt, pred)
            if iou > threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    ious = []
    boundary_errors = []
    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        gt = gt_segments[gi]
        pred = pred_segments[pi]
        ious.append(iou)
        boundary_errors.extend([abs(pred["start"] - gt["start"]), abs(pred["end"] - gt["end"])])

    tp = len(ious)
    fp = len(pred_segments) - tp
    fn = len(gt_segments) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "ious": ious, "boundary_errors": boundary_errors}


def summarize_scores(scores: list[dict], total_hours: float, total_pred_segments: int, window_acc: float) -> dict:
    tp = sum(item["tp"] for item in scores)
    fp = sum(item["fp"] for item in scores)
    fn = sum(item["fn"] for item in scores)
    denom = 2 * tp + fp + fn
    ious = [value for item in scores for value in item["ious"]]
    boundary_errors = [value for item in scores for value in item["boundary_errors"]]
    return {
        "window_accuracy": window_acc,
        "segment_f1": (2 * tp / denom) if denom else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "miou": float(np.mean(ious)) if ious else float("nan"),
        "boundary_mae_s": float(np.mean(boundary_errors)) if boundary_errors else float("nan"),
        "fp_per_hour": fp / total_hours if total_hours > 0 else 0.0,
        "records_per_hour": total_pred_segments / total_hours if total_hours > 0 else 0.0,
    }


def decode_subjects(subjects: list[SubjectWindows], proba_by_subject: dict[str, np.ndarray], mode: str, params: dict | None = None) -> dict[str, dict]:
    decoded = {}
    for subject in subjects:
        proba = proba_by_subject[subject.subject]
        if mode == "argmax":
            states = np.argmax(proba, axis=1)
            segment_proba = proba
        elif mode == "trl":
            assert params is not None
            states, segment_proba = decode_trl(proba, params)
        else:
            raise ValueError(mode)
        decoded[subject.subject] = {"states": states, "proba": segment_proba}
    return decoded


def evaluate_decoded_subjects(
    subjects: list[SubjectWindows],
    decoded_by_subject: dict[str, dict],
    classes: np.ndarray,
    *,
    min_segment_s: float,
    params: dict | None = None,
    ignore_labels: set[int] | None = None,
) -> dict:
    scores = []
    all_true = []
    all_pred = []
    total_hours = 0.0
    total_pred_segments = 0
    for subject in subjects:
        decoded = decoded_by_subject[subject.subject]
        states = decoded["states"]
        segment_proba = decoded["proba"]
        pred_segments = states_to_segments(states, subject.centers_s, subject.duration_s, classes, segment_proba)
        pred_segments = postprocess_segments(pred_segments, segment_proba, params)
        scores.append(match_segments(subject.gt_segments, pred_segments, min_segment_s, ignore_labels=ignore_labels))
        total_hours += subject.duration_s / 3600.0
        total_pred_segments += len(filter_eval_segments(pred_segments, min_segment_s, ignore_labels))
        ignored = ignore_labels or set()
        for true_label, pred_label in zip(subject.labels.tolist(), classes[states].tolist(), strict=False):
            if int(true_label) in ignored:
                continue
            all_true.append(true_label)
            all_pred.append(pred_label)
    window_acc = accuracy_score(all_true, all_pred) if all_true else 0.0
    return summarize_scores(scores, total_hours, total_pred_segments, window_acc)


def evaluate_subjects(
    subjects: list[SubjectWindows],
    proba_by_subject: dict[str, np.ndarray],
    classes: np.ndarray,
    *,
    min_segment_s: float,
    mode: str,
    params: dict | None = None,
    ignore_labels: set[int] | None = None,
) -> dict:
    decoded = decode_subjects(subjects, proba_by_subject, mode, params)
    return evaluate_decoded_subjects(
        subjects,
        decoded,
        classes,
        min_segment_s=min_segment_s,
        params=params,
        ignore_labels=ignore_labels,
    )


def fit_classifier(train_subjects: list[SubjectWindows]):
    x_train = np.vstack([subject.features for subject in train_subjects])
    y_train = np.concatenate([subject.labels for subject in train_subjects])
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    clf.fit(x_train, y_train)
    return clf


def predict_all(clf, subjects: list[SubjectWindows]) -> dict[str, np.ndarray]:
    return {subject.subject: clf.predict_proba(subject.features) for subject in subjects}


def select_trl_params(
    dev_subjects: list[SubjectWindows],
    proba_by_subject: dict[str, np.ndarray],
    classes: np.ndarray,
    *,
    min_segment_s: float,
    merge_gaps_s: tuple[float, ...],
    min_segments_s: tuple[float, ...],
    ignore_labels: set[int] | None = None,
) -> tuple[dict, list[dict]]:
    sequence_candidates = []
    for smooth_width in (1, 3, 5, 7):
        for median_width in (1, 3, 5):
            for self_prob in (0.35, 0.50, 0.65, 0.80, 0.90, 0.97):
                for min_run in (1, 2, 3, 5):
                    sequence_candidates.append(
                        {
                            "smooth_width": smooth_width,
                            "median_width": median_width,
                            "self_prob": self_prob,
                            "min_run": min_run,
                            "merge_gap_s": 0.0,
                            "min_segment_s": 0.0,
                            "conf_threshold": 0.0,
                        }
                    )

    evaluated = []
    decoded_cache = {}
    for params in sequence_candidates:
        seq_key = tuple(params[key] for key in ("smooth_width", "median_width", "self_prob", "min_run"))
        decoded_cache[seq_key] = decode_subjects(dev_subjects, proba_by_subject, "trl", params)
        metrics = evaluate_decoded_subjects(
            dev_subjects,
            decoded_cache[seq_key],
            classes,
            min_segment_s=min_segment_s,
            params=params,
            ignore_labels=ignore_labels,
        )
        evaluated.append({**params, **metrics, "stage": "sequence_only"})

    sequence_rank = sorted(evaluated, key=lambda item: (item["segment_f1"], item["miou"], -item["records_per_hour"]), reverse=True)
    expanded_keys = set()
    for base in sequence_rank[:40]:
        seq_key = tuple(base[key] for key in ("smooth_width", "median_width", "self_prob", "min_run"))
        decoded = decoded_cache[seq_key]
        for merge_gap_s in merge_gaps_s:
            for min_segment_s in min_segments_s:
                for conf_threshold in (0.0, 0.15, 0.25, 0.35, 0.45, 0.55):
                    params = {
                        **{key: base[key] for key in ("smooth_width", "median_width", "self_prob", "min_run")},
                        "merge_gap_s": merge_gap_s,
                        "min_segment_s": min_segment_s,
                        "conf_threshold": conf_threshold,
                    }
                    full_key = tuple(params[key] for key in ("smooth_width", "median_width", "self_prob", "min_run", "merge_gap_s", "min_segment_s", "conf_threshold"))
                    if full_key in expanded_keys:
                        continue
                    expanded_keys.add(full_key)
                    metrics = evaluate_decoded_subjects(
                        dev_subjects,
                        decoded,
                        classes,
                        min_segment_s=min_segment_s,
                        params=params,
                        ignore_labels=ignore_labels,
                    )
                    evaluated.append({**params, **metrics, "stage": "record_ops"})

    evaluated.sort(key=lambda item: (item["segment_f1"], item["miou"], -item["records_per_hour"]), reverse=True)
    param_keys = ("smooth_width", "median_width", "self_prob", "min_run", "merge_gap_s", "min_segment_s", "conf_threshold")
    return {key: evaluated[0][key] for key in param_keys}, evaluated


def format_table_row(name: str, metrics: dict) -> str:
    counts = f"{metrics['tp']}/{metrics['fp']}/{metrics['fn']}"
    return (
        f"{name} & {metrics['segment_f1']:.3f} & {metrics['miou']:.3f} & "
        f"{metrics['boundary_mae_s']:.2f} & {metrics['fp_per_hour']:.2f} & "
        f"{metrics['records_per_hour']:.2f} & {counts} \\\\"
    )


def write_dataset_outputs(summary: dict, result_dir: Path, prefix: str, caption: str) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f"{prefix}_trl_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [{"setting": setting, **metrics} for setting, metrics in summary["test_metrics"].items()]
    pd.DataFrame(rows).to_csv(result_dir / f"{prefix}_trl_metrics.csv", index=False)
    pd.DataFrame(summary["dev_grid"]).to_csv(result_dir / f"{prefix}_trl_dev_grid.csv", index=False)

    baseline = summary["test_metrics"]["Window argmax + merge"]
    trl = summary["test_metrics"]["Window argmax + dev-selected TRL-style record layer"]
    table = "\n".join(
        [
            "\\begin{center}",
            "\\begin{minipage}{\\columnwidth}",
            "\\centering",
            "\\footnotesize",
            caption,
            "\\\\[0.8ex]",
            "{\\setlength{\\tabcolsep}{2.0pt}",
            "\\begin{tabular*}{\\columnwidth}{@{}l@{\\extracolsep{\\fill}}cccccc@{}}",
            "\\toprule",
            "Setting & F1 & mIoU & B-MAE & FP/h & Rec./h & TP/FP/FN \\\\",
            "\\midrule",
            format_table_row("Argmax + merge", baseline),
            format_table_row("+ TRL-style layer", trl),
            "\\bottomrule",
            "\\end{tabular*}",
            "}",
            "\\end{minipage}",
            "\\end{center}",
            "",
        ]
    )
    (result_dir / f"{prefix}_portability_table.tex").write_text(table, encoding="utf-8")


def run_portability_experiment(
    *,
    dataset_name: str,
    source: str,
    sequences: dict[str, SubjectSequence],
    split: dict[str, list[str]],
    label_names: dict[int, str],
    window_s: float,
    step_s: float,
    fs_hz: float,
    result_dir: Path,
    prefix: str,
    caption: str,
    ignore_labels: set[int] | None = None,
) -> dict:
    all_subject_ids = split["train_subjects"] + split["dev_subjects"] + split["test_subjects"]
    missing = [subject for subject in all_subject_ids if subject not in sequences]
    if missing:
        raise FileNotFoundError(f"Missing subjects for {dataset_name}: {missing}")

    windows = {
        subject: build_subject_windows(sequences[subject], window_s=window_s, step_s=step_s, fs_hz=fs_hz)
        for subject in all_subject_ids
    }
    train = [windows[s] for s in split["train_subjects"]]
    dev = [windows[s] for s in split["dev_subjects"]]
    test = [windows[s] for s in split["test_subjects"]]

    clf = fit_classifier(train)
    dev_proba = predict_all(clf, dev)
    test_proba = predict_all(clf, test)
    classes = clf.classes_

    selected_params, dev_grid = select_trl_params(
        dev,
        dev_proba,
        classes,
        min_segment_s=window_s,
        merge_gaps_s=(0.0, step_s, window_s, 2.0 * window_s),
        min_segments_s=(0.0, window_s, 1.5 * window_s, 2.0 * window_s),
        ignore_labels=ignore_labels,
    )
    test_baseline = evaluate_subjects(
        test,
        test_proba,
        classes,
        min_segment_s=window_s,
        mode="argmax",
        ignore_labels=ignore_labels,
    )
    test_trl = evaluate_subjects(
        test,
        test_proba,
        classes,
        min_segment_s=window_s,
        mode="trl",
        params=selected_params,
        ignore_labels=ignore_labels,
    )

    summary = {
        "dataset": dataset_name,
        "source": source,
        "split": split,
        "window": {"length_s": window_s, "step_s": step_s, "fs_hz": fs_hz},
        "evaluation": {
            "matching": "class-consistent one-to-one IoU > 0.5",
            "minimum_episode_s": window_s,
            "ignored_labels": sorted(ignore_labels) if ignore_labels else [],
            "note": "Record-level matching excludes episodes shorter than one analysis window.",
        },
        "classifier": "StandardScaler + multinomial LogisticRegression(class_weight=balanced)",
        "labels": label_names,
        "selected_trl_params": selected_params,
        "dev_grid": dev_grid[:250],
        "test_metrics": {
            "Window argmax + merge": test_baseline,
            "Window argmax + dev-selected TRL-style record layer": test_trl,
        },
    }
    summary["dev_grid_csv"] = str(result_dir / f"{prefix}_trl_dev_grid.csv")
    write_dataset_outputs(summary, result_dir, prefix, caption)
    return summary
