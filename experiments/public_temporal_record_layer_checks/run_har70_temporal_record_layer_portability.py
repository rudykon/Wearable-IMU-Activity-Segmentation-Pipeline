"""HAR70+ feature-based TRL portability experiment.

Purpose:
    Loads HAR70+ subject sequences, builds hand-crafted window features, trains a
    lightweight baseline, applies temporal decoding, and evaluates segment-level
    transfer behavior.
Inputs:
    Expects a local HAR70+ dataset directory in the configured public-data search
    locations.
Outputs:
    Prints metrics and writes a result row for combined public portability
    summaries.
"""
from __future__ import annotations


import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "public_external" / "har70plus"
RESULT_DIR = SCRIPT_DIR / "results"

FS = 50.0
WINDOW_S = 4.0
STEP_S = 1.0
WINDOW_N = int(WINDOW_S * FS)
STEP_N = int(STEP_S * FS)
EVAL_MIN_SEGMENT_S = WINDOW_S

TRAIN_SUBJECTS = [str(i) for i in range(501, 513)]
DEV_SUBJECTS = [str(i) for i in range(513, 516)]
TEST_SUBJECTS = [str(i) for i in range(516, 519)]

SIGNAL_COLS = [
    "back_x",
    "back_y",
    "back_z",
    "thigh_x",
    "thigh_y",
    "thigh_z",
]

LABEL_NAMES = {
    1: "walking",
    3: "shuffling",
    4: "stairs up",
    5: "stairs down",
    6: "standing",
    7: "sitting",
    8: "lying",
}


@dataclass
class SubjectWindows:
    subject: str
    features: np.ndarray
    labels: np.ndarray
    centers_s: np.ndarray
    duration_s: float
    gt_segments: list[dict]


def load_subject(subject: str) -> pd.DataFrame:
    path = DATA_DIR / f"{subject}.csv"
    df = pd.read_csv(path, usecols=[*SIGNAL_COLS, "label"])
    df["label"] = df["label"].astype(int)
    return df


def robust_stats(values: np.ndarray) -> list[float]:
    return [
        float(values.mean()),
        float(values.std()),
        float(values.min()),
        float(values.max()),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
    ]


def featurize_window(block: np.ndarray) -> np.ndarray:
    features: list[float] = []
    series = [block[:, idx] for idx in range(block.shape[1])]
    series.append(np.linalg.norm(block[:, 0:3], axis=1))
    series.append(np.linalg.norm(block[:, 3:6], axis=1))
    for values in series:
        features.extend(robust_stats(values))
    return np.asarray(features, dtype=np.float32)


def majority_label(labels: np.ndarray) -> int:
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def labels_to_segments(labels: np.ndarray, duration_s: float) -> list[dict]:
    segments: list[dict] = []
    start = 0
    for idx in range(1, len(labels) + 1):
        if idx == len(labels) or labels[idx] != labels[start]:
            segments.append(
                {
                    "label": int(labels[start]),
                    "start": start / FS,
                    "end": min(idx / FS, duration_s),
                }
            )
            start = idx
    return segments


def build_subject_windows(subject: str) -> SubjectWindows:
    df = load_subject(subject)
    signal = df[SIGNAL_COLS].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int32)
    duration_s = len(df) / FS
    starts = range(0, len(df) - WINDOW_N + 1, STEP_N)

    features = []
    window_labels = []
    centers = []
    for start in starts:
        end = start + WINDOW_N
        features.append(featurize_window(signal[start:end]))
        window_labels.append(majority_label(labels[start:end]))
        centers.append((start + WINDOW_N / 2) / FS)

    return SubjectWindows(
        subject=subject,
        features=np.vstack(features),
        labels=np.asarray(window_labels, dtype=np.int32),
        centers_s=np.asarray(centers, dtype=np.float64),
        duration_s=duration_s,
        gt_segments=labels_to_segments(labels, duration_s),
    )


def moving_average(proba: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return proba.copy()
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(proba, ((pad_left, pad_right), (0, 0)), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    out = np.empty_like(proba)
    for col in range(proba.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    out /= out.sum(axis=1, keepdims=True)
    return out


def median_filter(proba: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return proba.copy()
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(proba, ((pad_left, pad_right), (0, 0)), mode="edge")
    out = np.empty_like(proba)
    for idx in range(proba.shape[0]):
        out[idx] = np.median(padded[idx : idx + width], axis=0)
    out /= out.sum(axis=1, keepdims=True)
    return out


def viterbi_decode(proba: np.ndarray, self_prob: float) -> np.ndarray:
    eps = 1e-12
    n_steps, n_classes = proba.shape
    off_prob = (1.0 - self_prob) / max(n_classes - 1, 1)
    trans = np.full((n_classes, n_classes), off_prob, dtype=np.float64)
    np.fill_diagonal(trans, self_prob)
    log_emit = np.log(np.clip(proba, eps, 1.0))
    log_trans = np.log(np.clip(trans, eps, 1.0))

    score = np.empty((n_steps, n_classes), dtype=np.float64)
    back = np.zeros((n_steps, n_classes), dtype=np.int16)
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


def decode_trl(proba: np.ndarray, smooth_width: int, median_width: int, self_prob: float, min_run: int) -> tuple[np.ndarray, np.ndarray]:
    smoothed = moving_average(proba, smooth_width)
    smoothed = median_filter(smoothed, median_width)
    decoded = viterbi_decode(smoothed, self_prob)
    return repair_short_runs(decoded, smoothed, min_run), smoothed


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
            segments.append(
                {
                    "label": int(labels[start]),
                    "state": int(states[start]),
                    "start": seg_start,
                    "end": seg_end,
                    "start_idx": start,
                    "end_idx": idx,
                    "confidence": segment_confidence(proba, start, idx, int(states[start])),
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

    top_k = int(params.get("top_k", 0) or 0)
    if top_k > 0 and len(processed) > top_k:
        processed = sorted(
            processed,
            key=lambda seg: (seg.get("confidence", 0.0), seg["end"] - seg["start"]),
            reverse=True,
        )[:top_k]
        processed = sorted(processed, key=lambda seg: seg["start"])

    return processed


def interval_iou(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def filter_eval_segments(segments: list[dict]) -> list[dict]:
    return [seg for seg in segments if seg["end"] - seg["start"] >= EVAL_MIN_SEGMENT_S]


def match_segments(gt_segments: list[dict], pred_segments: list[dict], threshold: float = 0.5) -> dict:
    gt_segments = filter_eval_segments(gt_segments)
    pred_segments = filter_eval_segments(pred_segments)
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
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ious": ious,
        "boundary_errors": boundary_errors,
    }


def summarize_scores(scores: list[dict], total_hours: float, total_pred_segments: int, window_acc: float) -> dict:
    tp = sum(item["tp"] for item in scores)
    fp = sum(item["fp"] for item in scores)
    fn = sum(item["fn"] for item in scores)
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom else 0.0
    ious = [value for item in scores for value in item["ious"]]
    boundary_errors = [value for item in scores for value in item["boundary_errors"]]
    return {
        "window_accuracy": window_acc,
        "segment_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "miou": float(np.mean(ious)) if ious else float("nan"),
        "boundary_mae_s": float(np.mean(boundary_errors)) if boundary_errors else float("nan"),
        "fp_per_hour": fp / total_hours,
        "records_per_hour": total_pred_segments / total_hours,
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
            states, segment_proba = decode_trl(
                proba,
                smooth_width=int(params["smooth_width"]),
                median_width=int(params.get("median_width", 1)),
                self_prob=float(params["self_prob"]),
                min_run=int(params["min_run"]),
            )
        else:
            raise ValueError(mode)
        decoded[subject.subject] = {"states": states, "proba": segment_proba}
    return decoded


def evaluate_decoded_subjects(subjects: list[SubjectWindows], decoded_by_subject: dict[str, dict], classes: np.ndarray, params: dict | None = None) -> dict:
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
        scores.append(match_segments(subject.gt_segments, pred_segments))
        total_hours += subject.duration_s / 3600.0
        total_pred_segments += len(filter_eval_segments(pred_segments))
        all_true.extend(subject.labels.tolist())
        all_pred.extend(classes[states].tolist())
    return summarize_scores(scores, total_hours, total_pred_segments, accuracy_score(all_true, all_pred))


def evaluate_subjects(subjects: list[SubjectWindows], proba_by_subject: dict[str, np.ndarray], classes: np.ndarray, mode: str, params: dict | None = None) -> dict:
    decoded = decode_subjects(subjects, proba_by_subject, mode, params)
    return evaluate_decoded_subjects(subjects, decoded, classes, params)


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


def select_trl_params(dev_subjects: list[SubjectWindows], proba_by_subject: dict[str, np.ndarray], classes: np.ndarray) -> tuple[dict, list[dict]]:
    sequence_candidates = []
    for smooth_width in (1, 3, 5, 7, 9):
        for median_width in (1, 3, 5):
            for self_prob in (0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.97):
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
                            "top_k": 0,
                        }
                    )

    evaluated = []
    decoded_cache = {}
    for params in sequence_candidates:
        seq_key = tuple(params[key] for key in ("smooth_width", "median_width", "self_prob", "min_run"))
        decoded_cache[seq_key] = decode_subjects(dev_subjects, proba_by_subject, "trl", params)
        metrics = evaluate_decoded_subjects(dev_subjects, decoded_cache[seq_key], classes, params)
        evaluated.append({**params, **metrics, "stage": "sequence_only"})

    # Expand the best sequence decoders with HLS-HAR-style record operators,
    # scaled down to HAR70+'s short daily-activity episodes.
    sequence_rank = sorted(evaluated, key=lambda item: (item["segment_f1"], item["miou"], -item["records_per_hour"]), reverse=True)
    top_sequences = sequence_rank[:60]
    expanded_keys = set()
    for base in top_sequences:
        seq_key = tuple(base[key] for key in ("smooth_width", "median_width", "self_prob", "min_run"))
        decoded = decoded_cache[seq_key]
        for merge_gap_s in (0.0, 2.0, 4.0, 6.0, 8.0):
            for min_segment_s in (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
                for conf_threshold in (0.0, 0.15, 0.25, 0.35, 0.45, 0.55):
                    params = {
                        **{key: base[key] for key in ("smooth_width", "median_width", "self_prob", "min_run")},
                        "merge_gap_s": merge_gap_s,
                        "min_segment_s": min_segment_s,
                        "conf_threshold": conf_threshold,
                        "top_k": 0,
                    }
                    full_key = tuple(params[key] for key in ("smooth_width", "median_width", "self_prob", "min_run", "merge_gap_s", "min_segment_s", "conf_threshold", "top_k"))
                    if full_key in expanded_keys:
                        continue
                    expanded_keys.add(full_key)
                    metrics = evaluate_decoded_subjects(dev_subjects, decoded, classes, params)
                    evaluated.append({**params, **metrics, "stage": "record_ops"})

    evaluated.sort(key=lambda item: (item["segment_f1"], item["miou"], -item["records_per_hour"]), reverse=True)
    param_keys = ("smooth_width", "median_width", "self_prob", "min_run", "merge_gap_s", "min_segment_s", "conf_threshold", "top_k")
    return {key: evaluated[0][key] for key in param_keys}, evaluated


def format_table_row(name: str, metrics: dict) -> str:
    counts = f"{metrics['tp']}/{metrics['fp']}/{metrics['fn']}"
    return (
        f"{name} & {metrics['segment_f1']:.3f} & {metrics['miou']:.3f} & "
        f"{metrics['boundary_mae_s']:.2f} & {metrics['fp_per_hour']:.2f} & "
        f"{metrics['records_per_hour']:.2f} & {counts} \\\\"
    )


def write_outputs(summary: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULT_DIR / "har70_trl_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for setting, metrics in summary["test_metrics"].items():
        rows.append({"setting": setting, **metrics})
    pd.DataFrame(rows).to_csv(RESULT_DIR / "har70_trl_metrics.csv", index=False)
    pd.DataFrame(summary["dev_grid"]).to_csv(RESULT_DIR / "har70_trl_dev_grid.csv", index=False)

    baseline = summary["test_metrics"]["Window argmax + merge"]
    trl = summary["test_metrics"]["Window argmax + dev-selected TRL-style record layer"]
    table = "\n".join(
        [
            "\\begin{center}",
            "\\begin{minipage}{\\columnwidth}",
            "\\centering",
            "\\footnotesize",
            "\\refstepcounter{table}\\label{tab:har70_trl_portability}",
            "\\textsc{Table~\\thetable}\\\\[0.35ex]",
            "\\parbox{\\columnwidth}{\\centering\\scshape Lightweight HAR70+ portability check. A simple window-level linear classifier is trained on subjects 501--512, TRL constants are selected on subjects 513--515, and the table reports held-out subjects 516--518. Record-level matching excludes episodes shorter than the 4 s analysis window. This table is a temporal-interface sanity check, not a public leaderboard comparison.}\\\\[0.8ex]",
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
    (RESULT_DIR / "har70_portability_table.tex").write_text(table, encoding="utf-8")


def main() -> None:
    for path in (DATA_DIR,):
        if not path.exists():
            raise FileNotFoundError(f"Missing HAR70+ data directory: {path}")

    all_subject_ids = TRAIN_SUBJECTS + DEV_SUBJECTS + TEST_SUBJECTS
    subjects = {subject: build_subject_windows(subject) for subject in all_subject_ids}
    train = [subjects[s] for s in TRAIN_SUBJECTS]
    dev = [subjects[s] for s in DEV_SUBJECTS]
    test = [subjects[s] for s in TEST_SUBJECTS]

    clf = fit_classifier(train)
    dev_proba = predict_all(clf, dev)
    test_proba = predict_all(clf, test)
    classes = clf.classes_

    selected_params, dev_grid = select_trl_params(dev, dev_proba, classes)
    test_baseline = evaluate_subjects(test, test_proba, classes, mode="argmax")
    test_trl = evaluate_subjects(test, test_proba, classes, mode="trl", params=selected_params)

    summary = {
        "dataset": "HAR70+",
        "source": "UCI Machine Learning Repository dataset 780",
        "split": {
            "train_subjects": TRAIN_SUBJECTS,
            "dev_subjects": DEV_SUBJECTS,
            "test_subjects": TEST_SUBJECTS,
        },
        "window": {"length_s": WINDOW_S, "step_s": STEP_S, "fs_hz": FS},
        "evaluation": {
            "matching": "class-consistent one-to-one IoU > 0.5",
            "minimum_episode_s": EVAL_MIN_SEGMENT_S,
            "note": "Record-level matching excludes episodes shorter than one analysis window.",
        },
        "classifier": "StandardScaler + multinomial LogisticRegression(class_weight=balanced)",
        "labels": LABEL_NAMES,
        "selected_trl_params": selected_params,
        "dev_grid": dev_grid[:250],
        "test_metrics": {
            "Window argmax + merge": test_baseline,
            "Window argmax + dev-selected TRL-style record layer": test_trl,
        },
    }
    summary["dev_grid_csv"] = str(RESULT_DIR / "har70_trl_dev_grid.csv")
    write_outputs(summary)

    print(json.dumps(summary["selected_trl_params"], indent=2))
    print(pd.DataFrame([{"setting": key, **value} for key, value in summary["test_metrics"].items()]).to_string(index=False))
    print(f"Wrote {RESULT_DIR / 'har70_trl_summary.json'}")
    print(f"Wrote {RESULT_DIR / 'har70_portability_table.tex'}")


if __name__ == "__main__":
    main()
