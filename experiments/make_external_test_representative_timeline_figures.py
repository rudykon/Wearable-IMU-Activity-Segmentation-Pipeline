"""Representative case-timeline figure generator.

Purpose:
    Builds visual comparisons between reference segments and model predictions
    for selected users, including timeline panels and boundary-shift summaries.
Inputs:
    Reads experiment metric files, prediction rows, reference annotations, and
    selected raw signals.
Outputs:
    Saves publication-style timeline figures under `experiments/figures/`.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import experiments.run_external_test_saved_model_evaluation_suite as ex
from imu_activity_pipeline import inference as final_inf
from imu_activity_pipeline.config import EXTERNAL_TEST_GOLD_FILE
from imu_activity_pipeline.sensor_data_processing import load_gold_labels

OUT_DIR = os.path.join(SCRIPT_DIR, "results")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FUSION_UID = "HNU21033"
FAILURE_UID = "HNU22034"
HELDOUT_GOLD_PATH = EXTERNAL_TEST_GOLD_FILE
HELDOUT_VARIANT_FILES = {
    "5s8s": os.path.join(OUT_DIR, "heldout_eval_best2scale_5s8s_S0_Full_20260424.xlsx"),
    "average": os.path.join(OUT_DIR, "heldout_eval_best3scale_average_S0_20260424.xlsx"),
    "weighted_long": os.path.join(OUT_DIR, "heldout_eval_best3scale_weighted_long_S0_20260424.xlsx"),
    "lbsa": os.path.join(OUT_DIR, "heldout_eval_best3scale_local_boundary_S0_20260424.xlsx"),
}

CLASS_COLORS = {
    "跑步": "#a7c9e7",
    "羽毛球": "#b9dcb0",
    "跳绳": "#f4ccb0",
    "乒乓球": "#efc1cd",
    "飞鸟": "#cec1e8",
}

CLASS_EDGES = {
    "跑步": "#7ea8cf",
    "羽毛球": "#90bd86",
    "跳绳": "#d8a883",
    "乒乓球": "#d69aad",
    "飞鸟": "#a89acf",
}

CLASS_NAMES_EN = {
    "跑步": "Running",
    "羽毛球": "Badminton",
    "跳绳": "Rope skipping",
    "乒乓球": "Table tennis",
    "飞鸟": "Dumbbell fly",
}

CLASS_SHORT = {
    "跑步": "RUN",
    "羽毛球": "BAD",
    "跳绳": "ROPE",
    "乒乓球": "TT",
    "飞鸟": "FLY",
}


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.grid": False,
            "grid.color": "#dde3ec",
            "grid.linestyle": "-",
            "grid.linewidth": 0.45,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.frameon": False,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
        }
    )


def _style_timeline_axes(ax):
    ax.set_facecolor("white")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.55)
        ax.spines[side].set_color("#222222")
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", colors="#222222", width=0.55, length=3.0)


def _to_segments(gold_df, uid):
    sub = gold_df[gold_df["user_id"] == uid]
    segments = []
    for _, row in sub.iterrows():
        segments.append(
            {
                "class_name": row["category"],
                "start_ts": int(row["start"]),
                "end_ts": int(row["end"]),
            }
        )
    return segments


def _f1_for_user(metrics, uid):
    user_metrics = metrics["per_user"][uid]
    return {
        "f1": float(user_metrics["f1"]),
        "TP": int(user_metrics["TP"]),
        "FP": int(user_metrics["FP"]),
        "FN": int(user_metrics["FN"]),
    }


def _largest_segment_shift(reference_segments, final_segments):
    best = None
    for ref_seg, final_seg in zip(reference_segments, final_segments):
        delta_start = int(final_seg["start_ts"] - ref_seg["start_ts"])
        delta_end = int(final_seg["end_ts"] - ref_seg["end_ts"])
        shift = max(abs(delta_start), abs(delta_end))
        if shift == 0:
            continue
        candidate = {
            "class_name": ref_seg["class_name"],
            "delta_start_sec": delta_start / 1000.0,
            "delta_end_sec": delta_end / 1000.0,
            "shift_sec": shift / 1000.0,
        }
        if best is None or candidate["shift_sec"] > best["shift_sec"]:
            best = candidate
    return best


def _segment_delta(previous_segments, current_segments):
    common = min(len(previous_segments), len(current_segments))
    max_shift_ms = 0
    class_mismatch = 0
    for prev_seg, curr_seg in zip(previous_segments[:common], current_segments[:common]):
        if prev_seg["class_name"] != curr_seg["class_name"]:
            class_mismatch += 1
        max_shift_ms = max(
            max_shift_ms,
            abs(int(curr_seg["start_ts"]) - int(prev_seg["start_ts"])),
            abs(int(curr_seg["end_ts"]) - int(prev_seg["end_ts"])),
        )
    return {
        "count_delta": abs(len(previous_segments) - len(current_segments)),
        "class_mismatch": class_mismatch,
        "max_shift_sec": max_shift_ms / 1000.0,
    }


def _draw_case(ax, uid, rows, panel_tag, summary_label, summary_metrics, show_xlabel):
    all_segments = []
    for _, segments, _ in rows:
        all_segments.extend(segments)

    start_ts = min(seg["start_ts"] for seg in all_segments)
    end_ts = max(seg["end_ts"] for seg in all_segments)

    def to_minutes(ts):
        return (ts - start_ts) / 60000.0

    bar_height = 0.52
    _style_timeline_axes(ax)
    gt_segments = rows[0][1]
    boundaries = sorted({to_minutes(seg["start_ts"]) for seg in gt_segments} | {to_minutes(seg["end_ts"]) for seg in gt_segments})
    for boundary in boundaries:
        ax.axvline(boundary, color="#e6eaf1", linestyle=(0, (2, 3)), linewidth=0.7, zorder=0)

    for label, segments, y in rows:
        ax.hlines(y + bar_height / 2.0, 0, to_minutes(end_ts), colors="#edf0f5", linewidth=0.72, zorder=0)
        for seg in segments:
            class_name = seg["class_name"]
            x = to_minutes(seg["start_ts"])
            width = max(0.01, to_minutes(seg["end_ts"]) - to_minutes(seg["start_ts"]))
            color = CLASS_COLORS.get(class_name, "#7f7f7f")
            patch = mpatches.FancyBboxPatch(
                (x, y + 0.05),
                width,
                bar_height - 0.10,
                boxstyle="round,pad=0.01,rounding_size=0.08",
                linewidth=0.65,
                edgecolor=CLASS_EDGES.get(class_name, "#7f7f7f"),
                facecolor=matplotlib.colors.to_rgba(color, 0.94),
            )
            ax.add_patch(patch)
            if width >= 3.3:
                ax.text(
                    x + width / 2.0,
                    y + bar_height / 2.0,
                    CLASS_SHORT.get(class_name, ""),
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    fontweight="bold",
                    color="#314152",
                )

    ax.set_xlim(0, to_minutes(end_ts) + 0.15)
    ax.set_ylim(rows[-1][2] - 0.05, rows[0][2] + 0.64)
    ax.set_yticks([y + bar_height / 2.0 for _, _, y in rows])
    ax.set_yticklabels([label for label, _, _ in rows])
    ax.set_xlabel("Relative time (minutes)" if show_xlabel else "")
    ax.grid(False)
    ax.tick_params(axis="x", pad=3)
    ax.text(
        0.0,
        1.052,
        f"({panel_tag}) {summary_label}   ({uid})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#243044",
    )
    metric_values, metric_suffix = _parse_metric_values(summary_metrics)
    metric_transform = ax.get_yaxis_transform()
    metric_x = 1.12
    metric_fontsize = 9.2
    suffix_fontsize = 6.8
    ax.text(
        metric_x,
        rows[1][2] + bar_height / 2.0 + 0.56,
        "F1",
        transform=metric_transform,
        ha="center",
        va="center",
        fontsize=metric_fontsize,
        fontweight="semibold",
        color="#6b7280",
        clip_on=False,
    )
    for metric_idx, metric_value in enumerate(metric_values[: len(rows) - 1]):
        row_y = rows[metric_idx + 1][2] + bar_height / 2.0
        if metric_suffix and metric_idx == len(metric_values) - 1:
            ax.text(
                metric_x,
                row_y + 0.07,
                metric_value,
                transform=metric_transform,
                ha="center",
                va="center",
                fontsize=metric_fontsize,
                color="#6b7280",
                clip_on=False,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": (1.0, 1.0, 1.0, 0.92),
                    "edgecolor": "none",
                },
            )
            ax.text(
                metric_x,
                row_y - 0.14,
                metric_suffix,
                transform=metric_transform,
                ha="center",
                va="center",
                fontsize=suffix_fontsize,
                color="#6b7280",
                clip_on=False,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": (1.0, 1.0, 1.0, 0.92),
                    "edgecolor": "none",
                },
            )
            continue
        ax.text(
            metric_x,
            row_y,
            metric_value,
            transform=metric_transform,
            ha="center",
            va="center",
            fontsize=metric_fontsize,
            linespacing=0.9,
            color="#6b7280",
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": (1.0, 1.0, 1.0, 0.92),
                "edgecolor": "none",
            },
        )
    if panel_tag == "a":
        note_text = "3s branch extends\nTT record"
        note_xy = (1.003, 1.195)
    else:
        note_text = "TT remains unmatched;\nlater BAD FP"
        note_xy = (1.003, 1.055)
    ax.text(
        note_xy[0],
        note_xy[1],
        note_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color="#7a8494",
        fontstyle="italic",
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": (1.0, 1.0, 1.0, 0.95),
            "edgecolor": "none",
        },
    )


def _parse_metric_values(summary_metrics):
    f1_part, _, suffix = summary_metrics.partition(";")
    values = [
        value.strip()
        for value in f1_part.replace("F1", "", 1).strip().split(" -> ")
        if value.strip()
    ]

    suffix = suffix.strip()
    if suffix.startswith("final TP/FP/FN = "):
        suffix = "TP/FP/FN=" + suffix.replace("final TP/FP/FN = ", "")
    return values, suffix


def _save_figure(fig, out_png):
    fig.savefig(out_png)


def plot_case_combined(fusion_panel, pruning_panel, out_png):
    fig, axes = plt.subplots(2, 1, figsize=(10.1, 5.55))

    _draw_case(
        axes[0],
        fusion_panel["uid"],
        fusion_panel["rows"],
        "a",
        fusion_panel["summary_label"],
        fusion_panel["summary_metrics"],
        show_xlabel=False,
    )
    _draw_case(
        axes[1],
        pruning_panel["uid"],
        pruning_panel["rows"],
        "b",
        pruning_panel["summary_label"],
        pruning_panel["summary_metrics"],
        show_xlabel=True,
    )

    handles = [
        mpatches.Patch(color=CLASS_COLORS[class_name], label=CLASS_NAMES_EN[class_name])
        for class_name in ["跑步", "羽毛球", "跳绳", "乒乓球", "飞鸟"]
    ]
    fig.legend(
        handles=handles,
        ncol=5,
        fontsize=6.6,
        loc="upper right",
        bbox_to_anchor=(0.992, 0.952),
        columnspacing=0.72,
        handlelength=1.28,
        labelspacing=0.36,
    )
    fig.subplots_adjust(left=0.175, right=0.900, bottom=0.078, top=0.855, hspace=0.27)
    _save_figure(fig, out_png)
    plt.close(fig)


def _run_pipeline(user_data, model_groups, device, fusion_mode, top_k=3, conf_min=0.45):
    predictions = {}
    for uid, raw in user_data.items():
        predictions[uid] = final_inf.process_single_user_with_options(
            user_id=uid,
            data=raw,
            model_groups=model_groups,
            device=device,
            fusion_mode=fusion_mode,
            min_duration_sec=180,
            top_k=top_k,
            conf_min=conf_min,
            verbose=False,
        )
    return predictions


def _build_fusion_panel(uid, gold, pred_58_all, pred_ref_all, pred_weighted_all, pred_final_all, metrics_58, metrics_ref, metrics_weighted, metrics_final):
    gt = _to_segments(gold, uid)
    pred_58 = pred_58_all[uid]
    pred_ref = pred_ref_all[uid]
    pred_weighted = pred_weighted_all[uid]
    pred_final = pred_final_all[uid]

    f1_58 = _f1_for_user(metrics_58, uid)
    f1_ref = _f1_for_user(metrics_ref, uid)
    f1_weighted = _f1_for_user(metrics_weighted, uid)
    f1_final = _f1_for_user(metrics_final, uid)
    boundary_change = _largest_segment_shift(pred_ref, pred_final)

    if boundary_change is None:
        summary_metrics = (
            f"F1 {f1_58['f1']:.3f} -> {f1_ref['f1']:.3f} -> "
            f"{f1_weighted['f1']:.3f} -> {f1_final['f1']:.3f}"
        )
    else:
        summary_metrics = (
            f"F1 {f1_58['f1']:.3f} -> {f1_ref['f1']:.3f} -> "
            f"{f1_weighted['f1']:.3f} -> {f1_final['f1']:.3f}; "
            f"{CLASS_NAMES_EN[boundary_change['class_name']]} boundary shift "
            f"{boundary_change['delta_start_sec']:+.1f}/{boundary_change['delta_end_sec']:+.1f} s"
        )

    return {
        "uid": uid,
        "rows": [
            ("Ground truth", gt, 3.8),
            ("5s + 8s\nrecord list", pred_58, 2.85),
            ("3-model average\n+ TRL", pred_ref, 1.90),
            ("3-model weighted-long\n+ TRL", pred_weighted, 0.95),
            ("3-model LBSA\n+ TRL", pred_final, 0.0),
        ],
        "summary_label": "Fusion-stage case",
        "summary_metrics": summary_metrics,
        "meta": {
            "largest_shift": boundary_change,
            "metrics": {
                "5s+8s": f1_58,
                "average": f1_ref,
                "weighted_long": f1_weighted,
                "final": f1_final,
            },
        },
    }


def _normalise_heldout_label(label):
    label = str(label).strip()
    return "乒乓球" if label == "兵乓球" else label


def _segments_from_frame(frame, uid):
    segments = []
    sub = frame[frame["user_id"] == uid].copy()
    sub = sub.sort_values("start")
    for _, row in sub.iterrows():
        segments.append(
            {
                "class_name": _normalise_heldout_label(row["category"]),
                "start_ts": int(row["start"]),
                "end_ts": int(row["end"]),
            }
        )
    return segments


def _load_heldout_prediction(path):
    frame = pandas_read_excel(path)
    frame.columns = ["user_id", "category", "start", "end"]
    frame["user_id"] = frame["user_id"].astype(str)
    return frame


def pandas_read_excel(path):
    import pandas as pd

    return pd.read_excel(path)


def _load_heldout_gold():
    frame = load_gold_labels(HELDOUT_GOLD_PATH)
    frame["user_id"] = frame["user_id"].astype(str)
    return frame


def _iou(seg_a, seg_b):
    inter = max(0, min(seg_a["end_ts"], seg_b["end_ts"]) - max(seg_a["start_ts"], seg_b["start_ts"]))
    union = max(seg_a["end_ts"], seg_b["end_ts"]) - min(seg_a["start_ts"], seg_b["start_ts"])
    return inter / union if union > 0 else 0.0


def _score_segments(pred_segments, gt_segments):
    candidates = []
    for pred_idx, pred_seg in enumerate(pred_segments):
        for gt_idx, gt_seg in enumerate(gt_segments):
            if pred_seg["class_name"] != gt_seg["class_name"]:
                continue
            overlap = _iou(pred_seg, gt_seg)
            if overlap > 0.5:
                candidates.append((overlap, pred_idx, gt_idx))
    candidates.sort(reverse=True)
    matched_pred = set()
    matched_gt = set()
    for _, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
    tp = len(matched_pred)
    fp = len(pred_segments) - tp
    fn = len(gt_segments) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "TP": tp, "FP": fp, "FN": fn}


def _build_failure_panel(uid):
    gold = _load_heldout_gold()
    gt = _segments_from_frame(gold, uid)
    pred_58 = _segments_from_frame(_load_heldout_prediction(HELDOUT_VARIANT_FILES["5s8s"]), uid)
    pred_ref = _segments_from_frame(_load_heldout_prediction(HELDOUT_VARIANT_FILES["average"]), uid)
    pred_weighted = _segments_from_frame(_load_heldout_prediction(HELDOUT_VARIANT_FILES["weighted_long"]), uid)
    pred_final = _segments_from_frame(_load_heldout_prediction(HELDOUT_VARIANT_FILES["lbsa"]), uid)

    f1_58 = _score_segments(pred_58, gt)
    f1_ref = _score_segments(pred_ref, gt)
    f1_weighted = _score_segments(pred_weighted, gt)
    f1_final = _score_segments(pred_final, gt)
    summary_metrics = (
        f"F1 {f1_58['f1']:.3f} -> {f1_ref['f1']:.3f} -> "
        f"{f1_weighted['f1']:.3f} -> {f1_final['f1']:.3f}; "
        f"final TP/FP/FN = {f1_final['TP']}/{f1_final['FP']}/{f1_final['FN']}"
    )

    return {
        "uid": uid,
        "rows": [
            ("Ground truth", gt, 3.8),
            ("5s + 8s\nrecord list", pred_58, 2.85),
            ("3-model average\n+ TRL", pred_ref, 1.90),
            ("3-model weighted-long\n+ TRL", pred_weighted, 0.95),
            ("3-model LBSA\n+ TRL", pred_final, 0.0),
        ],
        "summary_label": "Failure case",
        "summary_metrics": summary_metrics,
        "meta": {
            "metrics": {
                "5s+8s": f1_58,
                "average": f1_ref,
                "weighted_long": f1_weighted,
                "final": f1_final,
            },
        },
    }


def main():
    apply_style()
    ex.set_seed(42)
    device = torch.device(final_inf.DEVICE if torch.cuda.is_available() else "cpu")
    gold = ex.load_gold()
    user_data = ex.load_split_users()
    model_groups, _ = final_inf.load_ensemble_models()

    model_groups_58 = {scale: models for scale, models in model_groups.items() if scale in {"5s", "8s"}}

    pred_58_all = _run_pipeline(user_data, model_groups_58, device, fusion_mode="average")
    pred_ref_all = _run_pipeline(user_data, model_groups, device, fusion_mode="average")
    pred_weighted_all = _run_pipeline(user_data, model_groups, device, fusion_mode="weighted_long")
    pred_final_all = _run_pipeline(user_data, model_groups, device, fusion_mode=final_inf.DEFAULT_FUSION_MODE)

    metrics_58 = ex.evaluate_segments(pred_58_all, gold)
    metrics_ref = ex.evaluate_segments(pred_ref_all, gold)
    metrics_weighted = ex.evaluate_segments(pred_weighted_all, gold)
    metrics_final = ex.evaluate_segments(pred_final_all, gold)

    fusion_panel = _build_fusion_panel(
        FUSION_UID,
        gold,
        pred_58_all,
        pred_ref_all,
        pred_weighted_all,
        pred_final_all,
        metrics_58,
        metrics_ref,
        metrics_weighted,
        metrics_final,
    )
    failure_panel = _build_failure_panel(FAILURE_UID)

    out_png = os.path.join(FIG_DIR, "fig06_representative_timeline_cases.png")
    plot_case_combined(fusion_panel, failure_panel, out_png)

    summary = {
        "reference_system": "Best-per-scale 3-model + average fusion + TRL",
        "final_system": "Best-per-scale 3-model + LBSA + TRL",
        "chosen_cases": {
            "fusion_case": {
                "user_id": fusion_panel["uid"],
                **fusion_panel["meta"],
            },
            "failure_case": {
                "user_id": failure_panel["uid"],
                **failure_panel["meta"],
            },
        },
    }
    summary_path = os.path.join(OUT_DIR, "case_study_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Saved final case figure:", out_png)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
