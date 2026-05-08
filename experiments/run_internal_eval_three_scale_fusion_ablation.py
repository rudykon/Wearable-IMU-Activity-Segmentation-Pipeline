"""Supplementary A/B test for cross-scale probability-fusion rules.

Purpose:
    Compares average, dynamic-boundary, local-boundary, confident-conflict, and
    weighted fusion variants for the selected 3-scale model stack.
Inputs:
    Loads saved ensemble checkpoints, normalization parameters, internal
    evaluation signals, and reference annotations.
Outputs:
    Writes comparison tables under `experiments/results/`.
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

import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import experiments.run_external_test_saved_model_evaluation_suite as ex
import experiments.run_internal_eval_split_protocol_policy_selection_checks as pr
from imu_activity_pipeline.config import INTERNAL_EVAL_DATA_DIR, INTERNAL_EVAL_GOLD_FILE
from imu_activity_pipeline.inference import load_ensemble_models, predict_multiscale_ensemble


OUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)


def precompute_probs_with_fusion(user_data, model_group, device, fusion_mode):
    cache = {}
    for uid, data in user_data.items():
        ts, probs = predict_multiscale_ensemble(data, model_group, device, fusion_mode=fusion_mode)
        cache[uid] = (ts, probs)
    return cache


def choose_recommended_fixed_policy(fixed_policy_outer_stats, preferred_order):
    def key(name: str):
        stats = fixed_policy_outer_stats[name]
        order_rank = preferred_order.index(name) if name in preferred_order else len(preferred_order)
        return (float(stats["mean"]), -float(stats["std"]), -order_rank)

    return max(fixed_policy_outer_stats.keys(), key=key)


def main():
    ex.set_seed(42)
    device = torch.device(ex.DEVICE if torch.cuda.is_available() else "cpu")
    gold_df = ex.load_gold(INTERNAL_EVAL_GOLD_FILE)
    user_data = ex.load_split_users(INTERNAL_EVAL_DATA_DIR)
    model_group, _ = load_ensemble_models()

    fusion_modes = {
        "average": "AverageScaleFusion",
        "dynamic_boundary": "DynamicBoundaryFusion",
        "local_boundary": "LocalBoundaryFusion",
        "confident_conflict": "ConfidentConflictFusion",
        "weighted_long": "WeightedLongFusion",
        "weighted_balanced": "WeightedBalancedFusion",
    }
    policy_suite = [
        ("S0_Full", ex.PP_FULL),
        ("S6_NoTopK_KeepConf", replace(ex.PP_FULL, top_k=0)),
    ]
    preferred_order = ["S0_Full", "S6_NoTopK_KeepConf"]

    all_rows = []
    split_summaries = {}
    recommended = {}

    for fusion_mode, fusion_label in fusion_modes.items():
        probs_cache = precompute_probs_with_fusion(user_data, model_group, device, fusion_mode=fusion_mode)
        policy_predictions = {}
        for policy_name, cfg in policy_suite:
            pred_by_user, elapsed = ex.run_method(user_data, probs_cache, cfg, window_sec=3)
            policy_predictions[policy_name] = pred_by_user
            overall = ex.evaluate_segments(pred_by_user, gold_df)["overall"]
            all_rows.append(
                {
                    "fusion_mode": fusion_mode,
                    "fusion_label": fusion_label,
                    "policy": policy_name,
                    "mean_user_f1": float(overall["mean_user_f1"]),
                    "ci95_low": float(overall["ci95_low"]),
                    "ci95_high": float(overall["ci95_high"]),
                    "micro_f1": float(overall["micro_f1"]),
                    "TP": int(overall["TP"]),
                    "FP": int(overall["FP"]),
                    "FN": int(overall["FN"]),
                    "elapsed_sec": float(elapsed),
                }
            )

        _, split_summary, _ = pr.run_split_separated_selection(
            gold_df=gold_df,
            policy_predictions=policy_predictions,
            candidate_policies=[name for name, _ in policy_suite],
            fixed_reference_policies=[name for name, _ in policy_suite],
            n_repeats=50,
        )
        split_summaries[fusion_mode] = split_summary
        recommended[fusion_mode] = choose_recommended_fixed_policy(
            split_summary["fixed_policy_outer_stats"],
            preferred_order=preferred_order,
        )

    df = pd.DataFrame(all_rows).sort_values(
        ["mean_user_f1", "micro_f1"], ascending=False
    )
    csv_path = os.path.join(OUT_DIR, "best3scale_dynamic_fusion_check_20260424.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "dataset": "internal 20-user evaluation set only (data/signals/internal_eval + internal_eval_annotations.csv)",
        "restriction": "Fusion comparison performed only on the internal 20-user set. Held-out evaluation labels were not used.",
        "setting": "best-per-scale 3-model + fusion A/B check",
        "results": all_rows,
        "split_separated_summaries": split_summaries,
        "recommended_policy_by_fusion_mode": recommended,
        "artifacts": {
            "csv": os.path.relpath(csv_path, BASE_DIR),
        },
    }
    json_path = os.path.join(OUT_DIR, "best3scale_dynamic_fusion_check_20260424.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
