"""Internal-evaluation robustness checks for the best single 3-second baseline.

Purpose:
    Runs robustness checks, boundary tests, and effect-size summaries focused on
    the strongest single-scale baseline.
Inputs:
    Loads the selected 3-second checkpoint, normalization parameters, evaluation
    signals, and reference annotations.
Outputs:
    Writes robustness-check tables and figures under `experiments/results/` and
    `experiments/figures/`.
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
from dataclasses import replace

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
import experiments.run_internal_eval_split_protocol_policy_selection_checks as pr
from imu_activity_pipeline.config import INTERNAL_EVAL_DATA_DIR, INTERNAL_EVAL_GOLD_FILE


OUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def per_user_f1(metrics: dict):
    users = sorted(metrics["per_user"].keys())
    return [metrics["per_user"][u]["f1"] for u in users]


def test_and_effect(a, b, name: str):
    try:
        res = stats.wilcoxon(a, b, alternative="greater", zero_method="wilcox")
        p_val = float(res.pvalue)
        stat = float(res.statistic)
    except Exception:
        p_val = 1.0
        stat = 0.0

    diff = np.array(a) - np.array(b)
    sd = np.std(diff, ddof=1)
    d = 0.0 if sd < 1e-12 else float(np.mean(diff) / sd)
    return {
        "comparison": name,
        "wilcoxon_stat": stat,
        "p_one_sided": p_val,
        "mean_diff": float(np.mean(diff)),
        "paired_cohens_d": d,
    }


def save_robustness_plots(df_robust: pd.DataFrame) -> None:
    if not HAS_MATPLOTLIB:
        return
    labels = df_robust["display_label"].tolist()
    x = np.arange(len(labels))
    f1 = df_robust["mean_user_f1"].to_numpy(dtype=np.float64)
    delta = df_robust["delta_vs_clean"].to_numpy(dtype=np.float64)

    plt.figure(figsize=(11, 4.5))
    plt.plot(x, f1, marker="o", linewidth=2.0, color="#264653")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylim(max(0.6, float(f1.min()) - 0.03), min(1.0, float(f1.max()) + 0.03))
    plt.ylabel("Mean User F1")
    plt.title("Single-3s Temporal Policy Under Physical Perturbations")
    for idx, value in enumerate(f1):
        plt.text(idx, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "single3s_robustness.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(11, 4.5))
    colors = ["#2a9d8f" if v >= 0 else "#c05621" for v in delta]
    plt.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    plt.bar(x, delta, color=colors)
    plt.xticks(x, labels, rotation=25, ha="right")
    y_pad = max(0.01, float(np.max(np.abs(delta))) * 0.2)
    plt.ylim(float(delta.min()) - y_pad, float(delta.max()) + y_pad)
    plt.ylabel("Delta Mean User F1")
    plt.title("Change Relative to Clean Input")
    for idx, value in enumerate(delta):
        y = value + (0.003 if value >= 0 else -0.008)
        plt.text(idx, y, f"{value:+.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "single3s_robustness_delta.png"), dpi=220)
    plt.close()


def main():
    ex.set_seed(42)
    device = torch.device(ex.DEVICE if torch.cuda.is_available() else "cpu")
    gold_df = ex.load_gold(INTERNAL_EVAL_GOLD_FILE)
    user_data = ex.load_split_users(INTERNAL_EVAL_DATA_DIR)
    groups = ex.build_model_groups(device)
    cache = ex.precompute_probs(user_data, groups["single_3s_best"]["group"], device)

    # Best practice config from main comparison:
    # Full post-processing without top-k/confidence clipping.
    pp_best = replace(ex.PP_FULL, top_k=0, conf_min=0.0)

    # Ablation
    ablations = [
        ("S0_Full", ex.PP_FULL),
        ("S1_NoMedian", replace(ex.PP_FULL, median_filter=0)),
        ("S2_NoTopKConf", pp_best),
        ("S3_NoBoundaryRefine", replace(ex.PP_FULL, refine_boundary=False)),
        ("S4_NoViterbi", replace(ex.PP_FULL, use_viterbi=False)),
        ("S5_BaselinePP", ex.PP_BASELINE),
    ]
    rows = []
    all_metrics = {}
    for name, pp in ablations:
        pred_by_user, _ = ex.run_method(user_data, cache, pp, 3)
        metrics = ex.evaluate_segments(pred_by_user, gold_df)
        all_metrics[name] = metrics
        o = metrics["overall"]
        rows.append(
            {
                "ablation": name,
                "mean_user_f1": o["mean_user_f1"],
                "micro_f1": o["micro_f1"],
                "TP": o["TP"],
                "FP": o["FP"],
                "FN": o["FN"],
            }
        )

    df_ablation = pd.DataFrame(rows).sort_values("mean_user_f1", ascending=False)
    df_ablation.to_csv(os.path.join(OUT_DIR, "single3s_ablation.csv"), index=False, encoding="utf-8-sig")

    # Robustness on best config with more physical perturbations
    rng = np.random.default_rng(42)
    conditions = [
        ("Clean", "Clean", None),
        ("TimestampJitter_sigma8ms", "Timestamp jitter", lambda x: ex.add_timestamp_jitter(x, 8.0, rng)),
        ("BurstMissing_3pct_1s", "Burst missing", lambda x: ex.add_burst_missing(x, 0.03, 1.0, rng)),
        ("AxisSaturation_p85", "Axis saturation", lambda x: ex.add_axis_saturation(x, 0.85)),
        ("BiasDrift_0p50std", "Bias drift", lambda x: ex.add_bias_drift(x, 0.50, rng)),
        ("LowRate_2xDecim", "Low-rate distortion", lambda x: ex.add_low_rate_distortion(x, 5.0)),
    ]
    robust_rows = []
    for cname, display_label, fn in conditions:
        perturbed = ex.perturb_dataset(user_data, fn)
        perturbed_cache = ex.precompute_probs(perturbed, groups["single_3s_best"]["group"], device)
        pred_by_user, _ = ex.run_method(perturbed, perturbed_cache, pp_best, 3)
        metrics = ex.evaluate_segments(pred_by_user, gold_df)
        boundary = pr.compute_boundary_metrics(pred_by_user, gold_df, perturbed)
        o = metrics["overall"]
        robust_rows.append(
            {
                "condition": cname,
                "display_label": display_label,
                "mean_user_f1": o["mean_user_f1"],
                "ci95_low": o["ci95_low"],
                "ci95_high": o["ci95_high"],
                "micro_f1": o["micro_f1"],
                "TP": o["TP"],
                "FP": o["FP"],
                "FN": o["FN"],
                "mean_matched_iou": boundary["mean_matched_iou"],
                "start_mae_sec": boundary["start_mae_sec"],
                "end_mae_sec": boundary["end_mae_sec"],
                "duration_mae_sec": boundary["duration_mae_sec"],
                "fp_per_recording_hour": boundary["fp_per_recording_hour"],
            }
        )
    df_robust = pd.DataFrame(robust_rows)
    clean_f1 = float(df_robust.loc[df_robust["condition"] == "Clean", "mean_user_f1"].iloc[0])
    df_robust["delta_vs_clean"] = df_robust["mean_user_f1"] - clean_f1
    df_robust.to_csv(os.path.join(OUT_DIR, "single3s_robustness.csv"), index=False, encoding="utf-8-sig")
    save_robustness_plots(df_robust)

    # Significance
    arr_best = per_user_f1(all_metrics["S2_NoTopKConf"])
    arr_full = per_user_f1(all_metrics["S0_Full"])
    arr_base = per_user_f1(all_metrics["S5_BaselinePP"])
    significance = {
        "best_overall": all_metrics["S2_NoTopKConf"]["overall"],
        "full_overall": all_metrics["S0_Full"]["overall"],
        "baseline_overall": all_metrics["S5_BaselinePP"]["overall"],
        "tests": [
            test_and_effect(arr_best, arr_base, "S2_NoTopKConf vs S5_BaselinePP"),
            test_and_effect(arr_best, arr_full, "S2_NoTopKConf vs S0_Full"),
        ],
    }
    with open(os.path.join(OUT_DIR, "single3s_significance.json"), "w", encoding="utf-8") as f:
        json.dump(significance, f, indent=2, ensure_ascii=False)

    # Efficiency (best config)
    active_res = ex.run_efficiency(user_data, groups["single_3s_best"]["group"], pp_best, 3, device)
    cpu = torch.device("cpu")
    cpu_groups = ex.build_model_groups(cpu)
    cpu_res = ex.run_efficiency(user_data, cpu_groups["single_3s_best"]["group"], pp_best, 3, cpu)
    eff = {
        "active_device_single3s_best": active_res,
        "cpu_single3s_best": cpu_res,
    }
    with open(os.path.join(OUT_DIR, "single3s_efficiency.json"), "w", encoding="utf-8") as f:
        json.dump(eff, f, indent=2, ensure_ascii=False)

    print("Saved single-3s baseline robustness-check outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
