"""Internal-evaluation split-protocol policy-selection checks.

Purpose:
    Recomputes prediction rows, boundary metrics, duration-bin summaries,
    calibration curves, and policy-selection reports for the current data split
    protocol.
Inputs:
    Reads saved model assets, internal/external split metadata, signal files, and
    reference annotations when available.
Outputs:
    Saves policy-selection CSV/JSON tables and optional figures under
    `experiments/results/` and `experiments/figures/`.
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
from collections import Counter
from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import experiments.run_external_test_saved_model_evaluation_suite as ex
from imu_activity_pipeline.config import INTERNAL_EVAL_DATA_DIR, INTERNAL_EVAL_GOLD_FILE


OUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def subset_gold(gold_df: pd.DataFrame, users: List[str]) -> pd.DataFrame:
    return gold_df[gold_df["user_id"].isin(users)].copy().reset_index(drop=True)


def extract_records(pred_by_user: Dict[str, List[Dict]], gold_df: pd.DataFrame):
    users = sorted(gold_df["user_id"].unique())
    for uid in users:
        g_rows = gold_df[gold_df["user_id"] == uid]
        gt = [{"start": int(r["start"]), "end": int(r["end"]), "category": r["category"]} for _, r in g_rows.iterrows()]
        pred = []
        for seg in pred_by_user.get(uid, []):
            pred.append(
                {
                    "start": int(seg["start_ts"]),
                    "end": int(seg["end_ts"]),
                    "category": seg["class_name"],
                    "confidence": float(seg.get("confidence", 0.0)),
                }
            )

        matches = []
        for i, p in enumerate(pred):
            for j, g in enumerate(gt):
                if p["category"] != g["category"]:
                    continue
                iou = ex.calculate_iou(p["start"], p["end"], g["start"], g["end"])
                if iou > 0.5:
                    matches.append((iou, i, j))
        matches.sort(reverse=True, key=lambda x: x[0])

        matched_p = set()
        matched_g = set()
        accepted = []
        for iou, i, j in matches:
            if i in matched_p or j in matched_g:
                continue
            matched_p.add(i)
            matched_g.add(j)
            accepted.append((iou, pred[i], gt[j]))

        yield uid, pred, gt, accepted


def compute_boundary_metrics(
    pred_by_user: Dict[str, List[Dict]],
    gold_df: pd.DataFrame,
    user_data: Dict[str, np.ndarray],
) -> Dict:
    matched_ious = []
    start_abs = []
    start_signed = []
    end_abs = []
    end_signed = []
    dur_abs = []
    dur_signed = []
    fp_total = 0
    fn_total = 0
    tp_total = 0
    total_hours = 0.0

    for uid, pred, gt, accepted in extract_records(pred_by_user, gold_df):
        tp_total += len(accepted)
        fp_total += len(pred) - len(accepted)
        fn_total += len(gt) - len(accepted)
        raw = user_data[uid]
        duration_sec = max(0.0, float(raw[-1, 0] - raw[0, 0]) / 1000.0)
        total_hours += duration_sec / 3600.0

        for iou, p, g in accepted:
            matched_ious.append(float(iou))
            start_delta = (p["start"] - g["start"]) / 1000.0
            end_delta = (p["end"] - g["end"]) / 1000.0
            pred_dur = (p["end"] - p["start"]) / 1000.0
            gt_dur = (g["end"] - g["start"]) / 1000.0
            start_signed.append(float(start_delta))
            end_signed.append(float(end_delta))
            start_abs.append(abs(float(start_delta)))
            end_abs.append(abs(float(end_delta)))
            duration_delta = float(pred_dur - gt_dur)
            dur_signed.append(duration_delta)
            dur_abs.append(abs(duration_delta))

    def safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def safe_median(values: List[float]) -> float:
        return float(np.median(values)) if values else 0.0

    return {
        "matched_segments": len(matched_ious),
        "mean_matched_iou": safe_mean(matched_ious),
        "median_matched_iou": safe_median(matched_ious),
        "start_mae_sec": safe_mean(start_abs),
        "start_bias_sec": safe_mean(start_signed),
        "end_mae_sec": safe_mean(end_abs),
        "end_bias_sec": safe_mean(end_signed),
        "duration_mae_sec": safe_mean(dur_abs),
        "duration_bias_sec": safe_mean(dur_signed),
        "fp_per_recording_hour": float(fp_total / total_hours) if total_hours > 0 else 0.0,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "recording_hours": total_hours,
    }


def collect_prediction_rows(
    pred_by_user: Dict[str, List[Dict]],
    gold_df: pd.DataFrame,
    policy_name: str,
) -> pd.DataFrame:
    rows = []
    for uid, pred, gt, accepted in extract_records(pred_by_user, gold_df):
        matched_keys = {(p["start"], p["end"], p["category"]) for _, p, _ in accepted}
        for seg in pred:
            key = (seg["start"], seg["end"], seg["category"])
            rows.append(
                {
                    "policy": policy_name,
                    "user_id": uid,
                    "category": seg["category"],
                    "confidence": float(seg.get("confidence", 0.0)),
                    "correct": 1 if key in matched_keys else 0,
                }
            )
    return pd.DataFrame(rows)


def collect_boundary_rows(
    pred_by_user: Dict[str, List[Dict]],
    gold_df: pd.DataFrame,
    user_data: Dict[str, np.ndarray],
    policy_name: str,
) -> pd.DataFrame:
    rows = []
    for uid, pred, gt, accepted in extract_records(pred_by_user, gold_df):
        raw = user_data[uid]
        recording_sec = max(0.0, float(raw[-1, 0] - raw[0, 0]) / 1000.0)
        for iou, p, g in accepted:
            start_delta = (p["start"] - g["start"]) / 1000.0
            end_delta = (p["end"] - g["end"]) / 1000.0
            pred_dur = (p["end"] - p["start"]) / 1000.0
            gt_dur = (g["end"] - g["start"]) / 1000.0
            duration_delta = pred_dur - gt_dur
            rows.append(
                {
                    "policy": policy_name,
                    "user_id": uid,
                    "category": g["category"],
                    "recording_sec": recording_sec,
                    "gt_duration_sec": float(gt_dur),
                    "pred_duration_sec": float(pred_dur),
                    "matched_iou": float(iou),
                    "start_delta_sec": float(start_delta),
                    "end_delta_sec": float(end_delta),
                    "duration_delta_sec": float(duration_delta),
                    "start_abs_sec": abs(float(start_delta)),
                    "end_abs_sec": abs(float(end_delta)),
                    "duration_abs_sec": abs(float(duration_delta)),
                }
            )
    return pd.DataFrame(rows)


def summarize_boundary_rows(records: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(
            columns=[
                group_col,
                "matched_segments",
                "mean_matched_iou",
                "start_mae_sec",
                "start_bias_sec",
                "end_mae_sec",
                "end_bias_sec",
                "duration_mae_sec",
                "duration_bias_sec",
                "median_gt_duration_min",
            ]
        )

    rows = []
    grouped = records.groupby(group_col, sort=False)
    for key, part in grouped:
        rows.append(
            {
                group_col: key,
                "matched_segments": int(len(part)),
                "mean_matched_iou": float(part["matched_iou"].mean()),
                "start_mae_sec": float(part["start_abs_sec"].mean()),
                "start_bias_sec": float(part["start_delta_sec"].mean()),
                "end_mae_sec": float(part["end_abs_sec"].mean()),
                "end_bias_sec": float(part["end_delta_sec"].mean()),
                "duration_mae_sec": float(part["duration_abs_sec"].mean()),
                "duration_bias_sec": float(part["duration_delta_sec"].mean()),
                "median_gt_duration_min": float(part["gt_duration_sec"].median() / 60.0),
            }
        )
    return pd.DataFrame(rows)


def assign_duration_bins(records: pd.DataFrame, gold_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if records.empty:
        return records.copy(), {"q33_sec": 0.0, "q67_sec": 0.0}

    gold_durations = ((gold_df["end"] - gold_df["start"]) / 1000.0).astype(float).to_numpy()
    q33, q67 = np.quantile(gold_durations, [1.0 / 3.0, 2.0 / 3.0])

    def label_for(seconds: float) -> str:
        if seconds <= q33:
            return f"Short ($\\leq${q33 / 60.0:.1f} min)"
        if seconds <= q67:
            return f"Medium ({q33 / 60.0:.1f}--{q67 / 60.0:.1f} min)"
        return f"Long ($>{q67 / 60.0:.1f}$ min)"

    out = records.copy()
    out["duration_bin"] = out["gt_duration_sec"].apply(label_for)
    return out, {"q33_sec": float(q33), "q67_sec": float(q67)}


def binary_rank_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    sum_ranks_pos = float(ranks[pos].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_segment_calibration(records: pd.DataFrame, n_bins: int = 5) -> Tuple[Dict, pd.DataFrame]:
    if records.empty:
        empty_summary = {
            "n_pred": 0,
            "accuracy": 0.0,
            "mean_conf": 0.0,
            "calibration_gap": 0.0,
            "ece": 0.0,
            "mce": 0.0,
            "brier": 0.0,
            "nll": 0.0,
            "auroc": 0.0,
            "mean_conf_correct": 0.0,
            "mean_conf_incorrect": 0.0,
            "segments_below_conf045": 0,
            "correct_below_conf045": 0,
            "incorrect_below_conf045": 0,
        }
        return empty_summary, pd.DataFrame(columns=["bin_left", "bin_right", "count", "mean_conf", "empirical_acc"])

    y = records["correct"].to_numpy(dtype=np.float64)
    p = np.clip(records["confidence"].to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
    accuracy = float(np.mean(y))
    mean_conf = float(np.mean(p))
    brier = float(np.mean((p - y) ** 2))
    nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    auroc = binary_rank_auc(y.astype(np.int64), p)

    ece = 0.0
    mce = 0.0
    bin_rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for idx in range(n_bins):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        mask = (p >= left) & (p <= right) if idx == n_bins - 1 else (p >= left) & (p < right)
        if not np.any(mask):
            continue
        part_conf = float(np.mean(p[mask]))
        part_acc = float(np.mean(y[mask]))
        gap = abs(part_acc - part_conf)
        weight = float(np.mean(mask))
        ece += weight * gap
        mce = max(mce, gap)
        bin_rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "count": int(mask.sum()),
                "mean_conf": part_conf,
                "empirical_acc": part_acc,
                "abs_gap": gap,
            }
        )

    correct_mask = records["correct"] == 1
    incorrect_mask = records["correct"] == 0
    summary = {
        "n_pred": int(len(records)),
        "accuracy": accuracy,
        "mean_conf": mean_conf,
        "calibration_gap": float(accuracy - mean_conf),
        "ece": float(ece),
        "mce": float(mce),
        "brier": brier,
        "nll": nll,
        "auroc": auroc,
        "mean_conf_correct": float(records.loc[correct_mask, "confidence"].mean()) if correct_mask.any() else 0.0,
        "mean_conf_incorrect": float(records.loc[incorrect_mask, "confidence"].mean()) if incorrect_mask.any() else 0.0,
        "segments_below_conf045": int((records["confidence"] < 0.45).sum()),
        "correct_below_conf045": int((correct_mask & (records["confidence"] < 0.45)).sum()),
        "incorrect_below_conf045": int((incorrect_mask & (records["confidence"] < 0.45)).sum()),
    }
    return summary, pd.DataFrame(bin_rows)


def plot_segment_reliability(bin_df: pd.DataFrame) -> None:
    if not HAS_MATPLOTLIB:
        return
    policy_order = ["S5_BaselinePP", "S0_Full", "S2_NoTopKConf"]
    label_map = {
        "S5_BaselinePP": "S5 baseline",
        "S0_Full": "S0 full",
        "S2_NoTopKConf": "S2 no TopK/conf",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharex=True, sharey=True)
    for ax, policy in zip(axes, policy_order):
        sub = bin_df[bin_df["policy"] == policy].sort_values("bin_left")
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="#7a7a7a")
        if not sub.empty:
            ax.plot(sub["mean_conf"], sub["empirical_acc"], marker="o", linewidth=1.8, color="#264653")
            for _, row in sub.iterrows():
                ax.text(row["mean_conf"], row["empirical_acc"] + 0.035, f"n={int(row['count'])}", ha="center", fontsize=7)
        ax.set_title(label_map[policy], fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_xlabel("Mean confidence")
    axes[0].set_ylabel("Empirical segment accuracy")
    fig.suptitle("Segment-level reliability by post-processing policy", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "segment_reliability.png"), dpi=220)
    plt.close(fig)


def plot_boundary_distributions(records: pd.DataFrame) -> None:
    if not HAS_MATPLOTLIB:
        return
    policy_order = ["S5_BaselinePP", "S0_Full", "S2_NoTopKConf"]
    label_map = {
        "S5_BaselinePP": "S5 baseline",
        "S0_Full": "S0 full",
        "S2_NoTopKConf": "S2 no TopK/conf",
    }
    sub = records[records["policy"].isin(policy_order)].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    iou_data = [sub[sub["policy"] == policy]["matched_iou"].to_numpy(dtype=np.float64) for policy in policy_order]
    axes[0].boxplot(iou_data, tick_labels=[label_map[p] for p in policy_order], patch_artist=True)
    for patch, color in zip(axes[0].artists if hasattr(axes[0], "artists") else [], ["#b8d8d8", "#7ebdc2", "#4f6d7a"]):
        patch.set_facecolor(color)
    axes[0].set_ylabel("Matched IoU")
    axes[0].set_title("Matched IoU distribution")
    axes[0].set_ylim(0.6, 1.0)
    axes[0].grid(alpha=0.25, linewidth=0.6)

    dur_bias_data = [sub[sub["policy"] == policy]["duration_delta_sec"].to_numpy(dtype=np.float64) for policy in policy_order]
    axes[1].boxplot(dur_bias_data, tick_labels=[label_map[p] for p in policy_order], patch_artist=True)
    for patch, color in zip(axes[1].artists if hasattr(axes[1], "artists") else [], ["#f3d8c7", "#e0a458", "#c97b63"]):
        patch.set_facecolor(color)
    axes[1].axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Duration bias (s)")
    axes[1].set_title("Signed duration bias distribution")
    axes[1].grid(alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "boundary_distribution_summary.png"), dpi=220)
    plt.close(fig)


def choose_policy(candidate_metrics: Dict[str, Dict]) -> str:
    ordered = list(candidate_metrics.keys())
    best_name = ordered[0]
    best_key = (
        candidate_metrics[best_name]["overall"]["mean_user_f1"],
        candidate_metrics[best_name]["overall"]["micro_f1"],
    )
    for name in ordered[1:]:
        key = (
            candidate_metrics[name]["overall"]["mean_user_f1"],
            candidate_metrics[name]["overall"]["micro_f1"],
        )
        if key > best_key:
            best_name = name
            best_key = key
    return best_name


def paired_test_and_effect(a: List[float], b: List[float], name: str) -> Dict:
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    try:
        res = stats.wilcoxon(arr_a, arr_b, alternative="greater", zero_method="wilcox")
        p_val = float(res.pvalue)
        stat = float(res.statistic)
    except Exception:
        p_val = 1.0
        stat = 0.0

    diff = arr_a - arr_b
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    d = 0.0 if sd < 1e-12 else float(np.mean(diff) / sd)
    return {
        "comparison": name,
        "wilcoxon_stat": stat,
        "p_one_sided": p_val,
        "mean_diff": float(np.mean(diff)) if len(diff) else 0.0,
        "paired_cohens_d": d,
    }


def summarize_split_values(values: List[float], seed: int = 42) -> Dict:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci95": [0.0, 0.0],
            "q05": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q95": 0.0,
        }
    ci_low, ci_high = ex.bootstrap_ci(values, n_boot=10000, seed=seed)
    q05, q25, q50, q75, q95 = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ci95": [float(ci_low), float(ci_high)],
        "q05": float(q05),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "q95": float(q95),
    }


def run_split_separated_selection(
    gold_df: pd.DataFrame,
    policy_predictions: Dict[str, Dict[str, List[Dict]]],
    candidate_policies: List[str],
    fixed_reference_policies: List[str],
    n_repeats: int = 50,
) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
    users = sorted(gold_df["user_id"].unique())
    split_rows = []
    split_details = []
    selected_metrics = []
    fixed_metrics_store = {name: [] for name in fixed_reference_policies}

    for split_id in range(n_repeats):
        rng = np.random.default_rng(20260311 + split_id)
        shuffled = list(rng.permutation(users))
        n_tune = len(shuffled) // 2
        tune_users = sorted(shuffled[:n_tune])
        outer_eval_users = sorted(shuffled[n_tune:])

        tune_gold = subset_gold(gold_df, tune_users)
        outer_eval_gold = subset_gold(gold_df, outer_eval_users)

        tune_metrics = {}
        for name in candidate_policies:
            pred = policy_predictions[name]
            tune_metrics[name] = ex.evaluate_segments({u: pred[u] for u in tune_users}, tune_gold)
        best_name = choose_policy(tune_metrics)

        selected_outer_eval = ex.evaluate_segments(
            {u: policy_predictions[best_name][u] for u in outer_eval_users},
            outer_eval_gold,
        )
        fixed_results = {}
        for name in fixed_reference_policies:
            fixed_res = ex.evaluate_segments(
                {u: policy_predictions[name][u] for u in outer_eval_users},
                outer_eval_gold,
            )
            fixed_results[name] = fixed_res
            fixed_metrics_store[name].append(fixed_res["overall"]["mean_user_f1"])

        selected_metrics.append(selected_outer_eval["overall"]["mean_user_f1"])

        row = {
            "split_id": split_id,
            "tune_users": ",".join(tune_users),
            "outer_eval_users": ",".join(outer_eval_users),
            "selected_policy": best_name,
            "tune_mean_user_f1": tune_metrics[best_name]["overall"]["mean_user_f1"],
            "outer_eval_mean_user_f1": selected_outer_eval["overall"]["mean_user_f1"],
            "outer_eval_micro_f1": selected_outer_eval["overall"]["micro_f1"],
            "outer_eval_TP": selected_outer_eval["overall"]["TP"],
            "outer_eval_FP": selected_outer_eval["overall"]["FP"],
            "outer_eval_FN": selected_outer_eval["overall"]["FN"],
        }
        for name in fixed_reference_policies:
            row[f"fixed_{name}_outer_eval_mean_user_f1"] = fixed_results[name]["overall"]["mean_user_f1"]
            row[f"fixed_{name}_outer_eval_micro_f1"] = fixed_results[name]["overall"]["micro_f1"]
        split_rows.append(row)

        split_details.append(
            {
                "split_id": split_id,
                "tune_users": tune_users,
                "outer_eval_users": outer_eval_users,
                "selected_policy": best_name,
            }
        )

    df = pd.DataFrame(split_rows)
    chosen_counter = Counter(df["selected_policy"])
    selection_frequency = {name: int(chosen_counter.get(name, 0)) for name in candidate_policies}
    fixed_policy_outer_stats = {
        name: summarize_split_values(fixed_metrics_store[name], seed=42) for name in fixed_reference_policies
    }
    pairwise_selected_vs_fixed = {
        f"selected_vs_{name}": paired_test_and_effect(selected_metrics, fixed_metrics_store[name], f"selected vs {name}")
        for name in fixed_reference_policies
    }

    summary = {
        "repeats": n_repeats,
        "candidate_policies": candidate_policies,
        "selection_frequency": selection_frequency,
        "selected_policy_outer_mean_user_f1_mean": float(df["outer_eval_mean_user_f1"].mean()),
        "selected_policy_outer_mean_user_f1_std": float(df["outer_eval_mean_user_f1"].std(ddof=1)),
        "selected_policy_outer_micro_f1_mean": float(df["outer_eval_micro_f1"].mean()),
        "selected_policy_outer_stats": summarize_split_values(selected_metrics, seed=42),
        "fixed_policy_outer_stats": fixed_policy_outer_stats,
        "pairwise_selected_vs_fixed": pairwise_selected_vs_fixed,
    }
    return df, summary, split_details


def build_split_distribution_df(
    splits_df: pd.DataFrame,
    fixed_reference_policies: List[str],
) -> pd.DataFrame:
    rows = []
    for _, row in splits_df.iterrows():
        split_id = int(row["split_id"])
        rows.append(
            {
                "split_id": split_id,
                "setting": "SelectedPolicy",
                "outer_mean_user_f1": float(row["outer_eval_mean_user_f1"]),
            }
        )
        for name in fixed_reference_policies:
            rows.append(
                {
                    "split_id": split_id,
                    "setting": name,
                    "outer_mean_user_f1": float(row[f"fixed_{name}_outer_eval_mean_user_f1"]),
                }
            )
    return pd.DataFrame(rows)


def plot_outer_split_distribution(dist_df: pd.DataFrame) -> None:
    if not HAS_MATPLOTLIB:
        return
    if dist_df.empty:
        return
    order = [
        "SelectedPolicy",
        "S0_Full",
        "S2_NoTopKConf",
        "S6_NoTopK_KeepConf",
        "S7_KeepTopK_NoConf",
        "S5_BaselinePP",
    ]
    order = [x for x in order if x in set(dist_df["setting"].unique())]
    labels = {
        "SelectedPolicy": "Selected",
        "S0_Full": "S0",
        "S2_NoTopKConf": "S2",
        "S6_NoTopK_KeepConf": "NoTopK",
        "S7_KeepTopK_NoConf": "NoConf",
        "S5_BaselinePP": "S5",
    }
    box_data = [dist_df[dist_df["setting"] == key]["outer_mean_user_f1"].to_numpy(dtype=np.float64) for key in order]
    plt.figure(figsize=(9.5, 4.2))
    plt.boxplot(box_data, tick_labels=[labels[k] for k in order], patch_artist=True)
    plt.ylabel("Outer mean user F1")
    plt.xlabel("Split-level setting")
    plt.title("Repeated split-separated outer-F1 distributions (n=50)")
    plt.grid(axis="y", alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "outer_split_distribution.png"), dpi=220)
    plt.close()


def run_outer_split_calibration(
    prediction_records: pd.DataFrame,
    split_details: List[Dict],
    fixed_reference_policies: List[str],
    n_bins: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for detail in split_details:
        split_id = int(detail["split_id"])
        outer_users = detail["outer_eval_users"]
        selected_policy = detail["selected_policy"]
        policy_refs = [("SelectedPolicy", selected_policy)] + [(name, name) for name in fixed_reference_policies]

        for ref_name, actual_policy in policy_refs:
            part = prediction_records[
                (prediction_records["policy"] == actual_policy) & (prediction_records["user_id"].isin(outer_users))
            ].copy()
            summary, _ = compute_segment_calibration(part, n_bins=n_bins)
            rows.append(
                {
                    "split_id": split_id,
                    "policy_ref": ref_name,
                    "policy_actual": actual_policy,
                    "n_pred": int(summary["n_pred"]),
                    "accuracy": float(summary["accuracy"]),
                    "mean_conf": float(summary["mean_conf"]),
                    "calibration_gap": float(summary["calibration_gap"]),
                    "ece": float(summary["ece"]),
                    "brier": float(summary["brier"]),
                    "auroc": float(summary["auroc"]),
                    "segments_below_conf045": int(summary["segments_below_conf045"]),
                }
            )

    outer_df = pd.DataFrame(rows)
    if outer_df.empty:
        return outer_df, pd.DataFrame()

    summary_rows = []
    for ref_name, part in outer_df.groupby("policy_ref", sort=False):
        summary_rows.append(
            {
                "policy_ref": ref_name,
                "splits": int(len(part)),
                "total_pred_segments": int(part["n_pred"].sum()),
                "accuracy_mean": float(part["accuracy"].mean()),
                "accuracy_ci95_low": float(ex.bootstrap_ci(part["accuracy"].tolist(), n_boot=10000, seed=42)[0]),
                "accuracy_ci95_high": float(ex.bootstrap_ci(part["accuracy"].tolist(), n_boot=10000, seed=42)[1]),
                "mean_conf_mean": float(part["mean_conf"].mean()),
                "ece_mean": float(part["ece"].mean()),
                "ece_ci95_low": float(ex.bootstrap_ci(part["ece"].tolist(), n_boot=10000, seed=42)[0]),
                "ece_ci95_high": float(ex.bootstrap_ci(part["ece"].tolist(), n_boot=10000, seed=42)[1]),
                "brier_mean": float(part["brier"].mean()),
                "auroc_mean": float(part["auroc"].mean()),
                "segments_below_conf045_total": int(part["segments_below_conf045"].sum()),
            }
        )
    return outer_df, pd.DataFrame(summary_rows)


def run_fixed_policy_outer_report(
    gold_df: pd.DataFrame,
    user_data: Dict[str, np.ndarray],
    policy_predictions: Dict[str, Dict[str, List[Dict]]],
    policy_suite: List[Tuple[str, str, object]],
    split_details: List[Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for detail in split_details:
        split_id = int(detail["split_id"])
        outer_users = detail["outer_eval_users"]
        outer_gold = subset_gold(gold_df, outer_users)
        outer_user_data = {u: user_data[u] for u in outer_users}

        for policy_name, policy_label, _ in policy_suite:
            pred = {u: policy_predictions[policy_name][u] for u in outer_users}
            overall = ex.evaluate_segments(pred, outer_gold)["overall"]
            boundary = compute_boundary_metrics(pred, outer_gold, outer_user_data)
            rows.append(
                {
                    "split_id": split_id,
                    "policy": policy_name,
                    "label": policy_label,
                    "mean_user_f1": float(overall["mean_user_f1"]),
                    "micro_f1": float(overall["micro_f1"]),
                    "TP": int(overall["TP"]),
                    "FP": int(overall["FP"]),
                    "FN": int(overall["FN"]),
                    "matched_iou": float(boundary["mean_matched_iou"]),
                    "start_mae_sec": float(boundary["start_mae_sec"]),
                    "end_mae_sec": float(boundary["end_mae_sec"]),
                    "duration_mae_sec": float(boundary["duration_mae_sec"]),
                    "duration_bias_sec": float(boundary["duration_bias_sec"]),
                    "fp_per_recording_hour": float(boundary["fp_per_recording_hour"]),
                }
            )

    outer_df = pd.DataFrame(rows)
    if outer_df.empty:
        return outer_df, pd.DataFrame()

    summary_rows = []
    for idx, (policy_name, policy_label, _) in enumerate(policy_suite):
        part = outer_df[outer_df["policy"] == policy_name].copy()
        f1_stats = summarize_split_values(part["mean_user_f1"].tolist(), seed=42)
        micro_stats = summarize_split_values(part["micro_f1"].tolist(), seed=42)
        summary_rows.append(
            {
                "policy": policy_name,
                "label": policy_label,
                "order_id": idx,
                "splits": int(len(part)),
                "mean_user_f1_mean": float(part["mean_user_f1"].mean()),
                "mean_user_f1_ci95_low": float(f1_stats["ci95"][0]),
                "mean_user_f1_ci95_high": float(f1_stats["ci95"][1]),
                "micro_f1_mean": float(part["micro_f1"].mean()),
                "micro_f1_ci95_low": float(micro_stats["ci95"][0]),
                "micro_f1_ci95_high": float(micro_stats["ci95"][1]),
                "matched_iou_mean": float(part["matched_iou"].mean()),
                "start_mae_sec_mean": float(part["start_mae_sec"].mean()),
                "end_mae_sec_mean": float(part["end_mae_sec"].mean()),
                "duration_mae_sec_mean": float(part["duration_mae_sec"].mean()),
                "duration_bias_sec_mean": float(part["duration_bias_sec"].mean()),
                "fp_per_recording_hour_mean": float(part["fp_per_recording_hour"].mean()),
                "TP_mean": float(part["TP"].mean()),
                "FP_mean": float(part["FP"].mean()),
                "FN_mean": float(part["FN"].mean()),
            }
        )
    return outer_df, pd.DataFrame(summary_rows)


def main():
    ex.set_seed(42)
    device = torch.device(ex.DEVICE if torch.cuda.is_available() else "cpu")
    gold_df = ex.load_gold(INTERNAL_EVAL_GOLD_FILE)
    user_data = ex.load_split_users(INTERNAL_EVAL_DATA_DIR)
    groups = ex.build_model_groups(device)
    cache = ex.precompute_probs(user_data, groups["single_3s_best"]["group"], device)

    policy_suite_all = [
        ("S5_BaselinePP", ex.PP_BASELINE),
        ("S0_Full", ex.PP_FULL),
        ("S1_NoMedian", replace(ex.PP_FULL, median_filter=0)),
        ("S2_NoTopKConf", replace(ex.PP_FULL, top_k=0, conf_min=0.0)),
        ("S6_NoTopK_KeepConf", replace(ex.PP_FULL, top_k=0)),
        ("S7_KeepTopK_NoConf", replace(ex.PP_FULL, conf_min=0.0)),
        ("S3_NoBoundaryRefine", replace(ex.PP_FULL, refine_boundary=False)),
        ("S4_NoViterbi", replace(ex.PP_FULL, use_viterbi=False)),
    ]
    candidate_policies = [
        "S5_BaselinePP",
        "S0_Full",
        "S1_NoMedian",
        "S2_NoTopKConf",
        "S3_NoBoundaryRefine",
        "S4_NoViterbi",
    ]
    fixed_reference_policies = [
        "S0_Full",
        "S2_NoTopKConf",
        "S6_NoTopK_KeepConf",
        "S7_KeepTopK_NoConf",
        "S5_BaselinePP",
    ]
    pp_step_smooth180 = replace(ex.PP_BASELINE, smooth_window=7, min_duration=180)
    pp_step_strong_prior = replace(pp_step_smooth180, self_trans=0.97, cross_act=0.001)
    pp_step_median = replace(pp_step_strong_prior, median_filter=5)
    pp_step_conf = replace(pp_step_median, conf_min=0.45)
    incremental_policy_suite = [
        ("I0_Raw", "Raw posterior records", ex.PP_RAW),
        ("I1_BaseTemporal", "+ base temporal decoder", ex.PP_BASELINE),
        ("I2_WiderSmooth180", "+ wider smoothing + 180 s min", pp_step_smooth180),
        ("I3_StrongerPrior", "+ stronger temporal prior", pp_step_strong_prior),
        ("I4_Median", "+ median filter", pp_step_median),
        ("I5_ConfClip", "+ confidence clip (0.45)", pp_step_conf),
        ("I6_Full", "+ Top-K = 3 (Full)", ex.PP_FULL),
    ]

    policy_predictions = {}
    full_internal_eval_rows = []
    boundary_record_frames = []
    prediction_record_frames = []
    for name, cfg in policy_suite_all:
        pred_by_user, _ = ex.run_method(user_data, cache, cfg, 3)
        policy_predictions[name] = pred_by_user
        overall = ex.evaluate_segments(pred_by_user, gold_df)["overall"]
        boundary = compute_boundary_metrics(pred_by_user, gold_df, user_data)
        boundary_record_frames.append(collect_boundary_rows(pred_by_user, gold_df, user_data, name))
        prediction_record_frames.append(collect_prediction_rows(pred_by_user, gold_df, name))
        full_internal_eval_rows.append(
            {
                "policy": name,
                "mean_user_f1": overall["mean_user_f1"],
                "micro_f1": overall["micro_f1"],
                "TP": overall["TP"],
                "FP": overall["FP"],
                "FN": overall["FN"],
                "mean_matched_iou": boundary["mean_matched_iou"],
                "start_mae_sec": boundary["start_mae_sec"],
                "end_mae_sec": boundary["end_mae_sec"],
                "duration_mae_sec": boundary["duration_mae_sec"],
                "duration_bias_sec": boundary["duration_bias_sec"],
                "fp_per_recording_hour": boundary["fp_per_recording_hour"],
            }
        )

    incremental_predictions = {}
    for name, _, cfg in incremental_policy_suite:
        pred_by_user, _ = ex.run_method(user_data, cache, cfg, 3)
        incremental_predictions[name] = pred_by_user

    boundary_df = pd.DataFrame(full_internal_eval_rows).sort_values("mean_user_f1", ascending=False)
    boundary_df.to_csv(os.path.join(OUT_DIR, "boundary_metrics_single3s.csv"), index=False, encoding="utf-8-sig")
    disentangle_df = boundary_df[
        boundary_df["policy"].isin(["S0_Full", "S2_NoTopKConf", "S6_NoTopK_KeepConf", "S7_KeepTopK_NoConf"])
    ].copy()
    ref_row = disentangle_df[disentangle_df["policy"] == "S0_Full"].iloc[0]
    disentangle_df["delta_mean_user_f1_vs_s0"] = disentangle_df["mean_user_f1"] - float(ref_row["mean_user_f1"])
    disentangle_df["delta_micro_f1_vs_s0"] = disentangle_df["micro_f1"] - float(ref_row["micro_f1"])
    disentangle_df["delta_fp_vs_s0"] = disentangle_df["FP"] - int(ref_row["FP"])
    disentangle_df["delta_fn_vs_s0"] = disentangle_df["FN"] - int(ref_row["FN"])
    disentangle_df = disentangle_df.sort_values("mean_user_f1", ascending=False)
    disentangle_df.to_csv(
        os.path.join(OUT_DIR, "topk_conf_disentangle_single3s.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    boundary_records = pd.concat(boundary_record_frames, ignore_index=True)
    boundary_records.to_csv(os.path.join(OUT_DIR, "boundary_matches_single3s.csv"), index=False, encoding="utf-8-sig")
    plot_boundary_distributions(boundary_records)

    prediction_records = pd.concat(prediction_record_frames, ignore_index=True)
    prediction_records.to_csv(os.path.join(OUT_DIR, "segment_prediction_confidence.csv"), index=False, encoding="utf-8-sig")

    s2_records = boundary_records[boundary_records["policy"] == "S2_NoTopKConf"].copy()
    s2_records, duration_meta = assign_duration_bins(s2_records, gold_df)

    gold_support_by_class = gold_df["category"].value_counts().to_dict()
    by_class_df = summarize_boundary_rows(s2_records, "category")
    by_class_df["gt_support"] = by_class_df["category"].map(lambda x: int(gold_support_by_class.get(x, 0)))
    by_class_df["match_rate"] = by_class_df.apply(
        lambda row: float(row["matched_segments"] / row["gt_support"]) if row["gt_support"] > 0 else 0.0,
        axis=1,
    )
    by_class_df = by_class_df.sort_values("gt_support", ascending=False)
    by_class_df.to_csv(os.path.join(OUT_DIR, "boundary_by_class_single3s.csv"), index=False, encoding="utf-8-sig")

    gold_with_bins = gold_df.copy()
    gold_with_bins["gt_duration_sec"] = ((gold_with_bins["end"] - gold_with_bins["start"]) / 1000.0).astype(float)
    gold_with_bins, _ = assign_duration_bins(gold_with_bins, gold_df)
    duration_support = gold_with_bins["duration_bin"].value_counts().to_dict()

    by_duration_df = summarize_boundary_rows(s2_records, "duration_bin")
    by_duration_df["gt_support"] = by_duration_df["duration_bin"].map(lambda x: int(duration_support.get(x, 0)))
    by_duration_df["match_rate"] = by_duration_df.apply(
        lambda row: float(row["matched_segments"] / row["gt_support"]) if row["gt_support"] > 0 else 0.0,
        axis=1,
    )
    order_map = {
        label: idx for idx, label in enumerate(
            [
                f"Short ($\\leq${duration_meta['q33_sec'] / 60.0:.1f} min)",
                f"Medium ({duration_meta['q33_sec'] / 60.0:.1f}--{duration_meta['q67_sec'] / 60.0:.1f} min)",
                f"Long ($>{duration_meta['q67_sec'] / 60.0:.1f}$ min)",
            ]
        )
    }
    by_duration_df["sort_key"] = by_duration_df["duration_bin"].map(order_map)
    by_duration_df = by_duration_df.sort_values("sort_key").drop(columns=["sort_key"])
    by_duration_df.to_csv(os.path.join(OUT_DIR, "boundary_by_duration_single3s.csv"), index=False, encoding="utf-8-sig")

    calibration_rows = []
    calibration_bin_frames = []
    for policy_name in ["S5_BaselinePP", "S0_Full", "S2_NoTopKConf", "S6_NoTopK_KeepConf", "S7_KeepTopK_NoConf"]:
        summary_row, bin_df = compute_segment_calibration(
            prediction_records[prediction_records["policy"] == policy_name].copy(),
            n_bins=5,
        )
        summary_row["policy"] = policy_name
        calibration_rows.append(summary_row)
        if not bin_df.empty:
            bin_df["policy"] = policy_name
            calibration_bin_frames.append(bin_df)

    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df = calibration_df[
        [
            "policy",
            "n_pred",
            "accuracy",
            "mean_conf",
            "calibration_gap",
            "ece",
            "mce",
            "brier",
            "nll",
            "auroc",
            "mean_conf_correct",
            "mean_conf_incorrect",
            "segments_below_conf045",
            "correct_below_conf045",
            "incorrect_below_conf045",
        ]
    ]
    calibration_df.to_csv(os.path.join(OUT_DIR, "segment_calibration_summary.csv"), index=False, encoding="utf-8-sig")

    calibration_bins_df = pd.concat(calibration_bin_frames, ignore_index=True)
    calibration_bins_df.to_csv(os.path.join(OUT_DIR, "segment_calibration_bins.csv"), index=False, encoding="utf-8-sig")
    plot_segment_reliability(calibration_bins_df)

    splits_df, split_summary, split_details = run_split_separated_selection(
        gold_df=gold_df,
        policy_predictions=policy_predictions,
        candidate_policies=candidate_policies,
        fixed_reference_policies=fixed_reference_policies,
        n_repeats=50,
    )
    splits_df.to_csv(
        os.path.join(OUT_DIR, "internal_eval_split_protocol_policy_selection_splits.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    split_dist_df = build_split_distribution_df(splits_df, fixed_reference_policies=fixed_reference_policies)
    split_dist_df.to_csv(
        os.path.join(OUT_DIR, "internal_eval_split_protocol_policy_selection_distribution.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    plot_outer_split_distribution(split_dist_df)
    outer_calib_df, outer_calib_summary_df = run_outer_split_calibration(
        prediction_records=prediction_records,
        split_details=split_details,
        fixed_reference_policies=fixed_reference_policies,
        n_bins=5,
    )
    outer_calib_df.to_csv(os.path.join(OUT_DIR, "outer_split_calibration.csv"), index=False, encoding="utf-8-sig")
    outer_calib_summary_df.to_csv(
        os.path.join(OUT_DIR, "outer_split_calibration_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    incremental_outer_df, incremental_outer_summary_df = run_fixed_policy_outer_report(
        gold_df=gold_df,
        user_data=user_data,
        policy_predictions=incremental_predictions,
        policy_suite=incremental_policy_suite,
        split_details=split_details,
    )
    incremental_outer_df.to_csv(os.path.join(OUT_DIR, "incremental_outer_single3s.csv"), index=False, encoding="utf-8-sig")
    incremental_outer_summary_df.to_csv(
        os.path.join(OUT_DIR, "incremental_outer_single3s_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "fixed_checkpoint": "single_3s_best",
        "candidate_policies": candidate_policies,
        "full_internal_eval_policy_metrics": {row["policy"]: row for row in full_internal_eval_rows},
        "full_internal_eval_boundary_metrics_top": boundary_df.iloc[0].to_dict(),
        "full_internal_eval_boundary_metrics_s2": boundary_df[boundary_df["policy"] == "S2_NoTopKConf"].iloc[0].to_dict(),
        "full_internal_eval_boundary_metrics_s5": boundary_df[boundary_df["policy"] == "S5_BaselinePP"].iloc[0].to_dict(),
        "topk_conf_disentangle": {
            row["policy"]: row
            for row in disentangle_df.to_dict(orient="records")
        },
        "segment_calibration_s5": calibration_df[calibration_df["policy"] == "S5_BaselinePP"].iloc[0].to_dict(),
        "segment_calibration_s0": calibration_df[calibration_df["policy"] == "S0_Full"].iloc[0].to_dict(),
        "segment_calibration_s2": calibration_df[calibration_df["policy"] == "S2_NoTopKConf"].iloc[0].to_dict(),
        "segment_calibration_s6": calibration_df[calibration_df["policy"] == "S6_NoTopK_KeepConf"].iloc[0].to_dict(),
        "segment_calibration_s7": calibration_df[calibration_df["policy"] == "S7_KeepTopK_NoConf"].iloc[0].to_dict(),
        "outer_split_calibration_summary": {
            row["policy_ref"]: row for row in outer_calib_summary_df.to_dict(orient="records")
        },
        "incremental_outer_single3s_summary": {
            row["policy"]: row for row in incremental_outer_summary_df.to_dict(orient="records")
        },
        "boundary_duration_breakpoints_sec": duration_meta,
        "split_separated_selection": split_summary,
    }

    with open(
        os.path.join(OUT_DIR, "internal_eval_split_protocol_policy_selection_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved split-protocol policy-selection outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
